# demo_coach/web/server.py
import json
import os
import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from demo_coach import download, replay
from demo_coach.agent import CoachAgent
from demo_coach.radar import load_calibration
from demo_coach.tools import MatchContext, build_context, dispatch

DEMOS_DIR = Path(os.environ.get("DEMO_COACH_DEMO_DIR", "demos"))
STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="demo_coach")
app.add_middleware(GZipMiddleware, minimum_size=50_000)  # replay frames compress ~10x
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
CONTEXTS: dict[str, MatchContext] = {}
AGENTS: dict[str, CoachAgent] = {}
DEMO_PATHS: dict[str, str] = {}  # demo_id -> uploaded .dem path (replay source)


@app.on_event("startup")
def _open_browser():
    """Open the UI in the default browser on server start.
    Opt out with DEMO_COACH_NO_BROWSER=1 (start.bat sets this and opens the
    browser itself, so you never get duplicate tabs)."""
    if os.environ.get("DEMO_COACH_NO_BROWSER"):
        return
    import threading
    import webbrowser
    port = os.environ.get("DEMO_COACH_PORT", "8000")
    threading.Timer(1.0, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()


class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/upload")
def upload(file: UploadFile = File(...)):
    if not file.filename.endswith(".dem"):
        raise HTTPException(400, "only .dem files are accepted")
    DEMOS_DIR.mkdir(exist_ok=True)
    dest = DEMOS_DIR / Path(file.filename).name
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        ctx = build_context(str(dest))
    except Exception as e:
        dest.unlink(missing_ok=True)
        raise HTTPException(422, f"failed to parse demo: {e}")
    CONTEXTS[ctx.parsed.demo_id] = ctx
    AGENTS.pop(ctx.parsed.demo_id, None)  # re-upload gets a fresh agent
    DEMO_PATHS[ctx.parsed.demo_id] = str(dest)
    return {"demo_id": ctx.parsed.demo_id, "map": ctx.summary["map"],
            "rounds_played": ctx.summary["rounds_played"]}


@app.get("/api/matches")
def matches():
    return [{"demo_id": c.parsed.demo_id, "map": c.summary["map"],
             "rounds_played": c.summary["rounds_played"]}
            for c in CONTEXTS.values()]


@app.post("/api/matches/{demo_id}/chat")
def chat(demo_id: str, req: ChatRequest):
    ctx = CONTEXTS.get(demo_id)
    if ctx is None:
        raise HTTPException(404, "unknown demo_id")
    if demo_id not in AGENTS:
        try:
            AGENTS[demo_id] = CoachAgent(ctx)
        except RuntimeError as e:
            raise HTTPException(503, str(e))
    try:
        reply = AGENTS[demo_id].chat(req.message, req.history)
    except Exception as e:
        raise HTTPException(502, f"LLM API error: {e}")
    return {"reply": reply}


@app.post("/api/matches/{demo_id}/chat/stream")
def chat_stream(demo_id: str, req: ChatRequest):
    """SSE chat: streams {"type":"token"|"tool"|"done"|"error"} events."""
    ctx = CONTEXTS.get(demo_id)
    if ctx is None:
        raise HTTPException(404, "unknown demo_id")
    if demo_id not in AGENTS:
        try:
            AGENTS[demo_id] = CoachAgent(ctx)
        except RuntimeError as e:
            raise HTTPException(503, str(e))

    def gen():
        try:
            for ev in AGENTS[demo_id].chat_stream(req.message, req.history):
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield ("data: " + json.dumps(
                {"type": "error", "detail": f"LLM API error: {e}"}) + "\n\n")

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/api/matches/{demo_id}/rounds/{n}/map")
def round_map(demo_id: str, n: int):
    ctx = CONTEXTS.get(demo_id)
    if ctx is None:
        raise HTTPException(404, "unknown demo_id")
    map_name = ctx.parsed.header.get("map_name") or "unknown"
    cal = load_calibration(map_name)
    if cal is None:
        raise HTTPException(404, f"no radar available for map {map_name}")
    deaths = ctx.parsed.deaths
    kills = deaths[deaths["total_rounds_played"] == n]
    return {
        "map": map_name,
        "calibration": cal,
        "image": f"/static/maps/{map_name}.png",
        "kills": [
            {
                "tick": int(k["tick"]),
                "attacker": k["attacker_name"],
                "victim": k["user_name"],
                "weapon": k["weapon"],
                "headshot": bool(k["headshot"]),
                "attacker_x": float(k["attacker_X"]),
                "attacker_y": float(k["attacker_Y"]),
                "victim_x": float(k["user_X"]),
                "victim_y": float(k["user_Y"]),
            }
            for _, k in kills.iterrows()
        ],
    }


@app.post("/api/matches/{demo_id}/tool/{name}")
def tool_call(demo_id: str, name: str, args: dict = {}):
    """Direct (no-LLM) access to tool functions for the UI panels."""
    ctx = CONTEXTS.get(demo_id)
    if ctx is None:
        raise HTTPException(404, "unknown demo_id")
    return json.loads(dispatch(ctx, name, args))


@app.get("/api/matches/{demo_id}/rounds/{n}/replay")
def round_replay(demo_id: str, n: int):
    ctx = CONTEXTS.get(demo_id)
    if ctx is None:
        raise HTTPException(404, "unknown demo_id")
    map_name = ctx.parsed.header.get("map_name") or "unknown"
    if load_calibration(map_name) is None:
        raise HTTPException(404, f"no radar available for map {map_name}")
    demo_path = DEMO_PATHS.get(demo_id)
    if demo_path is None or not Path(demo_path).exists():
        raise HTTPException(409, "demo file no longer available; re-upload to build replays")
    try:
        data = replay.get_round_replay(demo_path, demo_id, map_name, n)
    except Exception as e:
        raise HTTPException(500, f"failed to build replay: {e}")
    if data is None:
        raise HTTPException(404, f"no round {n}")
    return data


# ---- demo downloads (5E / Perfect World / Steam matchmaking) ----

class DownloadListRequest(BaseModel):
    platform: str
    creds: dict
    save: bool = True
    limit: int = 20


class DownloadFetchRequest(BaseModel):
    platform: str
    creds: dict
    match: dict
    save: bool = True


class ShareCodeRequest(BaseModel):
    share_code: str


def _register_demo(dest: Path) -> dict:
    """Parse a downloaded demo and register it like an upload."""
    try:
        ctx = build_context(str(dest))
    except Exception as e:
        dest.unlink(missing_ok=True)
        raise HTTPException(422, f"failed to parse demo: {e}")
    CONTEXTS[ctx.parsed.demo_id] = ctx
    DEMO_PATHS[ctx.parsed.demo_id] = str(dest)
    return {"demo_id": ctx.parsed.demo_id, "map": ctx.summary["map"],
            "rounds_played": ctx.summary["rounds_played"]}


@app.get("/api/download/config")
def download_config():
    return download.load_credentials()


@app.post("/api/download/list")
def download_list(req: DownloadListRequest):
    try:
        out = download.list_matches(req.platform, req.creds, req.limit)
    except KeyError as e:
        raise HTTPException(400, f"missing credential: {e}")
    except Exception as e:
        raise HTTPException(502, f"{req.platform} list failed: {e}")
    if req.save:
        saved = download.load_credentials()
        saved[req.platform] = req.creds
        download.save_credentials(saved)
    return out


@app.post("/api/download/fetch")
def download_fetch(req: DownloadFetchRequest):
    try:
        dest = download.fetch(req.platform, req.creds, req.match, DEMOS_DIR)
    except Exception as e:
        raise HTTPException(502, f"download failed: {e}")
    if req.save:
        saved = download.load_credentials()
        saved[req.platform] = req.creds
        download.save_credentials(saved)
    return _register_demo(dest)


@app.post("/api/download/sharecode")
def download_sharecode(req: ShareCodeRequest):
    """One-off Steam matchmaking demo from a share code (CSGO-XXXX-...)."""
    try:
        url = download.resolve_steam_share_code(req.share_code.strip())
        dest = download.download_demo(url, DEMOS_DIR, req.share_code.strip())
    except Exception as e:
        raise HTTPException(502, f"share code download failed: {e}")
    return _register_demo(dest)
