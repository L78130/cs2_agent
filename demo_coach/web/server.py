# demo_coach/web/server.py
import os
import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from demo_coach.agent import CoachAgent
from demo_coach.tools import MatchContext, build_context

DEMOS_DIR = Path(os.environ.get("DEMO_COACH_DEMO_DIR", "demos"))
STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="demo_coach")
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
