# demo_coach/web/server.py
import json
import os
import shutil
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from demo_coach.agent import CoachAgent
from demo_coach.radar import load_calibration, short_weapon
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

    def _player_xy_at(name, tick):
        """Nearest sampled position/yaw of a player at or before a tick."""
        g = pos[(pos["name"] == name) & (pos["tick"] <= tick)]
        if g.empty:
            return None
        r = g.sort_values("tick").iloc[-1]
        return (round(float(r["X"]), 1), round(float(r["Y"]), 1),
                round(float(r["yaw"]), 1) if "yaw" in g.columns and pd.notna(r["yaw"]) else 0.0)

    fires = parsed.fires
    fire_list = []
    if not fires.empty:
        fires = fires[(fires["tick"] >= start) & (fires["tick"] <= end)]
        for _, f in fires.iterrows():
            xy = _player_xy_at(f["user_name"], int(f["tick"]))
            fire_list.append({
                "tick": int(f["tick"]),
                "name": f["user_name"],
                "x": round(float(f["user_X"]), 1) if pd.notna(f["user_X"]) else (xy[0] if xy else None),
                "y": round(float(f["user_Y"]), 1) if pd.notna(f["user_Y"]) else (xy[1] if xy else None),
                "yaw": xy[2] if xy else 0.0,
            })

    bombs = parsed.bombs
    bomb_list = []
    if not bombs.empty:
        bombs = bombs[(bombs["tick"] >= start) & (bombs["tick"] <= end)]
        for _, b in bombs.iterrows():
            xy = _player_xy_at(b["user_name"], int(b["tick"])) if pd.notna(b["user_name"]) else None
            bomb_list.append({
                "tick": int(b["tick"]),
                "event": b["event"],
                "site": b["site"] if "site" in bombs.columns and pd.notna(b["site"]) else None,
                "x": xy[0] if xy else None,
                "y": xy[1] if xy else None,
            })

    hurts = parsed.hurts
    damage_list = []
    if not hurts.empty and "user_X" in hurts.columns:
        hurts = hurts[(hurts["tick"] >= start) & (hurts["tick"] <= end)]
        for _, h in hurts.iterrows():
            if pd.isna(h.get("user_X")) or pd.isna(h.get("attacker_X")):
                continue
            damage_list.append({
                "tick": int(h["tick"]),
                "attacker": h["attacker_name"],
                "victim": h["user_name"],
                "dmg": int(h["dmg_health"]) if pd.notna(h.get("dmg_health")) else 0,
                "attacker_x": float(h["attacker_X"]),
                "attacker_y": float(h["attacker_Y"]),
                "victim_x": float(h["user_X"]),
                "victim_y": float(h["user_Y"]),
            })

    players = []
    for name, g in pos.groupby("name"):
        g = g.sort_values("tick")
        has_gear = "active_weapon_name" in g.columns
        frames = []
        gear_changes: list[list] = []
        last_inv = None
        for _, r in g.iterrows():
            weapon = short_weapon(r["active_weapon_name"]) if has_gear else ""
            yaw = round(float(r["yaw"]), 1) if "yaw" in g.columns and pd.notna(r["yaw"]) else 0.0
            frames.append([int(r["tick"]), round(float(r["X"]), 1),
                           round(float(r["Y"]), 1), weapon, yaw])
            if "inventory" in g.columns:
                raw = r["inventory"]
                items = [] if raw is None or (not hasattr(raw, "__len__") and pd.isna(raw)) \
                    else sorted(short_weapon(w) for w in raw)
                if items != last_inv:
                    gear_changes.append([int(r["tick"]), items])
                    last_inv = items
        players.append({
            "name": name,
            "team": int(g["team_num"].iloc[0]) if "team_num" in g else 0,
            "death_tick": death_tick.get(name),
            "frames": frames,
            "gear_changes": gear_changes,
        })
    return {
        "map": map_name, "calibration": cal,
        "image": f"/static/maps/{map_name}.png",
        "start": start, "end": end,
        "players": players,
        "fires": fire_list,
        "bombs": bomb_list,
        "damages": damage_list,
        "kills": [
            {"tick": int(k["tick"]), "attacker": k["attacker_name"],
             "victim": k["user_name"], "weapon": k["weapon"],
             "headshot": bool(k["headshot"]),
             "attacker_x": float(k["attacker_X"]), "attacker_y": float(k["attacker_Y"]),
             "victim_x": float(k["user_X"]), "victim_y": float(k["user_Y"])}
            for _, k in round_kills.iterrows()
        ],
    }
