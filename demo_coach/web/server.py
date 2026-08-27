# demo_coach/web/server.py
import json
import os
import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from demo_coach.agent import CoachAgent
from demo_coach.radar import load_calibration
from demo_coach.tools import MatchContext, build_context, dispatch

DEMOS_DIR = Path(os.environ.get("DEMO_COACH_DEMO_DIR", "demos"))
STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="demo_coach")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
CONTEXTS: dict[str, MatchContext] = {}
AGENTS: dict[str, CoachAgent] = {}


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
    parsed = ctx.parsed
    map_name = parsed.header.get("map_name") or "unknown"
    cal = load_calibration(map_name)
    if cal is None:
        raise HTTPException(404, f"no radar available for map {map_name}")
    rounds = parsed.rounds.sort_values("tick").reset_index(drop=True)
    if n < 0 or n >= len(rounds):
        raise HTTPException(404, f"no round {n}")
    end = int(rounds.iloc[n]["tick"])
    freeze_ticks = sorted(parsed.economy["tick"].unique()) if not parsed.economy.empty else []
    start = int(freeze_ticks[n]) if n < len(freeze_ticks) else 0

    deaths = parsed.deaths
    round_kills = deaths[deaths["total_rounds_played"] == n]
    death_tick = {}
    for _, k in round_kills.iterrows():
        death_tick.setdefault(k["user_name"], int(k["tick"]))

    pos = parsed.positions
    pos = pos[(pos["tick"] >= start) & (pos["tick"] <= end)]
    players = []
    for name, g in pos.groupby("name"):
        frames = [[int(r["tick"]), round(float(r["X"]), 1), round(float(r["Y"]), 1)]
                  for _, r in g.sort_values("tick").iterrows()]
        players.append({
            "name": name,
            "team": int(g["team_num"].iloc[0]) if "team_num" in g else 0,
            "death_tick": death_tick.get(name),
            "frames": frames,
        })
    return {
        "map": map_name, "calibration": cal,
        "image": f"/static/maps/{map_name}.png",
        "start": start, "end": end,
        "players": players,
        "kills": [
            {"tick": int(k["tick"]), "attacker": k["attacker_name"],
             "victim": k["user_name"], "weapon": k["weapon"],
             "headshot": bool(k["headshot"]),
             "attacker_x": float(k["attacker_X"]), "attacker_y": float(k["attacker_Y"]),
             "victim_x": float(k["user_X"]), "victim_y": float(k["user_Y"])}
            for _, k in round_kills.iterrows()
        ],
    }
