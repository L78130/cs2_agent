# tests/test_render_smoke.py
"""Headless smoke test of the radar renderer (index.html <script>).

Runs the page's JS under a V8 engine (mini-racer) with stubbed DOM/canvas,
then drives render() across the static kill map and a whole replay sweep
(camera transform, frame interpolation, effects, roster, killfeed). Catches
ReferenceErrors / TypeErrors that a browser console would show but pytest
would never see (e.g. the raycastCone `sy` self-reference that killed the
replay loop).
"""
import json
import re

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from py_mini_racer import MiniRacer

from demo_coach import replay as replay_mod
from demo_coach.parsing import ParsedDemo
from demo_coach.tools import MatchContext
from demo_coach.web import server

DOM_STUB = r"""
function __makeCtx() {
  const grad = { addColorStop(t, c) {
    if (typeof c !== "string" || !c.startsWith("rgba")) throw new Error("bad color stop: " + c);
  } };
  return {
    fillStyle: "", strokeStyle: "", lineWidth: 1, globalAlpha: 1, font: "",
    globalCompositeOperation: "source-over",
    clearRect(){}, drawImage(){}, beginPath(){}, moveTo(){}, lineTo(){},
    stroke(){}, fill(){}, arc(){}, roundRect(){}, fillText(){}, save(){}, restore(){},
    closePath(){}, setLineDash(){}, setTransform(){}, translate(){}, scale(){},
    measureText(t){ return { width: String(t).length * 6 }; },
    createRadialGradient(x0,y0,r0,x1,y1,r1){
      if ([x0,y0,r0,x1,y1,r1].some(v => typeof v !== "number" || isNaN(v)))
        throw new Error("NaN gradient");
      return grad;
    },
    getImageData(x,y,w,h){ return { data: new Uint8ClampedArray(w*h*4) }; },
    putImageData(d,x,y){},
  };
}
function __makeEl(id) {
  return {
    id, innerHTML: "", textContent: "", value: "1", min: 0, max: 100, disabled: false,
    hidden: false, width: 1024, height: 1024, files: [],
    style: {}, dataset: {},
    classList: { toggle(){}, add(){}, remove(){}, contains(){ return false; } },
    appendChild(){}, addEventListener(){},
    getContext(){ return __makeCtx(); },
    getBoundingClientRect(){ return { left:0, top:0, width:1024, height:1024 }; },
    scrollTop: 0, scrollHeight: 0,
  };
}
const __els = {};
const document = {
  getElementById(id){ if (!__els[id]) __els[id] = __makeEl(id); return __els[id]; },
  createElement(tag){ return __makeEl(tag); },
  querySelectorAll(){ const a = []; a.forEach = Array.prototype.forEach.bind(a); return a; },
};
const window = { addEventListener(){} };
function Image() {
  const img = { complete: true, naturalWidth: 88, naturalHeight: 32, onload: null, onerror: null };
  Object.defineProperty(img, "src", { set(v) { if (img.onload) img.onload(); }, get(){ return ""; } });
  return img;
}
const performance = { now(){ return 1000; } };
let __rafCb = null;
function requestAnimationFrame(cb){ __rafCb = cb; }
function fetch(){ return Promise.reject(new Error("offline in harness")); }
function alert(m){ throw new Error("alert: " + m); }
"""

DRIVER = """
// static kill-map scene
mapData = JSON.parse(MAP_JSON);
radarImgReady = true;
render();
// replay scene: init and sweep the whole round, incl. seeks backward
replay = JSON.parse(REPLAY_JSON);
initReplay();
var dur = durationSec();
for (var i = 0; i <= 100; i++) seekTo(dur * i / 100);
for (var i = 100; i >= 0; i -= 7) seekTo(dur * i / 100);
cameraFit();
setPlaying(true);
"SIM_OK";
"""

_PLAYER = {
    "steamid64": "7656", "z": 10.0, "is_alive": True, "armor": 100,
    "has_helmet": True, "money": 4000, "equipment_value": 4150,
    "inventory": ["AK-47", "Glock-18"], "has_defuser": False,
    "has_c4": False, "flash_duration": 0.0,
}


def _player(name, team, x, y, yaw, hp, weapon, **kw):
    return {**_PLAYER, "name": name, "team": team, "x": x, "y": y,
            "yaw": yaw, "health": hp, "weapon": weapon, **kw}


def _replay_payload():
    cal = {"pos_x": -2953, "pos_y": 2164, "scale": 5.0,
           "content_x": 118, "content_y": 68, "content_width": 764,
           "content_height": 878}
    frames = [
        {"tick": 100, "time_sec": 0.0, "shots": [], "players": [
            _player("atk", "T", -1000.0, 1000.0, 90.0, 100, "AK-47", has_c4=True),
            _player("vic", "CT", -900.0, 900.0, -90.0, 100, "USP-S")]},
        {"tick": 164, "time_sec": 1.0, "players": [
            _player("atk", "T", -950.0, 950.0, 80.0, 100, "AK-47", has_c4=True),
            _player("vic", "CT", -890.0, 910.0, -80.0, 100, "USP-S",
                    flash_duration=2.1)],
         "shots": [{"tick": 160, "actor": "atk", "weapon": "ak47",
                    "yaw": 80.0, "pitch": -2.0, "x": -950.0, "y": 950.0}]},
        {"tick": 228, "time_sec": 2.0, "shots": [], "players": [
            _player("atk", "T", -900.0, 900.0, 70.0, 55, "AK-47"),
            _player("vic", "CT", -880.0, 920.0, -70.0, 100, "USP-S")]},
        {"tick": 300, "time_sec": 3.125, "shots": [], "players": [
            _player("atk", "T", -850.0, 850.0, 60.0, 55, "AK-47"),
            _player("vic", "CT", -870.0, 930.0, -60.0, 100, "USP-S",
                    is_alive=False, health=0)]},
    ]
    events = [
        {"type": "kill", "tick": 170, "actor": "atk", "target": "vic",
         "weapon": "ak47", "headshot": True, "assister": None,
         "actor_x": -950.0, "actor_y": 950.0, "target_x": -890.0, "target_y": 910.0},
        {"type": "plant", "tick": 140, "actor": "atk", "site": "A",
         "x": -900.0, "y": 900.0},
        {"type": "grenade", "kind": "smoke", "tick": 190, "actor": "atk",
         "x": -920.0, "y": 920.0, "z": 0.0, "throw_tick": 150,
         "trajectory": [{"tick": 150, "x": -1000.0, "y": 1000.0, "z": 30.0},
                        {"tick": 170, "x": -960.0, "y": 960.0, "z": 60.0},
                        {"tick": 190, "x": -920.0, "y": 920.0, "z": 0.0}]},
        {"type": "grenade", "kind": "he", "tick": 210, "actor": "vic",
         "x": -880.0, "y": 880.0, "z": 0.0, "throw_tick": None, "trajectory": []},
    ]
    tracks = [
        {"id": "smoke:0:190", "type": "smoke", "x": -920.0, "y": 920.0, "z": 0.0,
         "start_tick": 190, "end_tick": 190 + 18 * 64, "radius": 144.0},
        {"id": "inferno:0:220", "type": "inferno", "x": -870.0, "y": 870.0, "z": 0.0,
         "start_tick": 220, "end_tick": 220 + 7 * 64, "radius": 150.0},
    ]
    return {"round": 0, "map_name": "de_ancient", "map_transform": cal,
            "image": "/static/maps/de_ancient.png", "mask": None,
            "tick_rate": 64, "fps": 32, "start_tick": 100, "end_tick": 300,
            "frames": frames, "events": events, "effect_tracks": tracks,
            "effect_capabilities": {"inferno_cells": False, "smoke_voxels": False,
                                    "smoke_mode": "legacy_circle"}}


@pytest.fixture()
def ctx(tmp_path):
    parsed = ParsedDemo(
        demo_id="rendertest", header={"map_name": "de_ancient"},
        deaths=pd.DataFrame(
            [[150, 0, "vic", "atk", None, True, "ak47", 10.0, 20.0, 30.0, 40.0]],
            columns=["tick", "total_rounds_played", "user_name", "attacker_name",
                     "assister_name", "headshot", "weapon",
                     "user_X", "user_Y", "attacker_X", "attacker_Y"]),
        hurts=pd.DataFrame(),
        rounds=pd.DataFrame([[300, 2, 8]], columns=["tick", "winner", "reason"]),
        economy=pd.DataFrame(),
    )
    demo = tmp_path / "demo.dem"
    demo.touch()
    return MatchContext(parsed, pd.DataFrame(), pd.DataFrame(), [], {}), str(demo)


def test_radar_render_smoke(ctx, monkeypatch):
    match_ctx, demo_path = ctx
    monkeypatch.setattr(replay_mod, "get_round_replay",
                        lambda *a: _replay_payload())
    server.CONTEXTS.clear()
    server.DEMO_PATHS.clear()
    server.CONTEXTS["rendertest"] = match_ctx
    server.DEMO_PATHS["rendertest"] = demo_path
    try:
        c = TestClient(server.app)
        replay = c.get("/api/matches/rendertest/rounds/0/replay").json()
        killmap = c.get("/api/matches/rendertest/rounds/0/map").json()
    finally:
        server.CONTEXTS.clear()
        server.DEMO_PATHS.clear()

    html = (server.STATIC_DIR / "index.html").read_text(encoding="utf-8")
    js = re.search(r"<script>(.*?)</script>", html, re.S).group(1)
    js = js.replace("refreshMatches();", "")  # no fetch in the harness

    prog = (DOM_STUB + js
            + "\nvar REPLAY_JSON = " + json.dumps(json.dumps(replay)) + ";"
            + "\nvar MAP_JSON = " + json.dumps(json.dumps(killmap)) + ";\n"
            + DRIVER)
    assert MiniRacer().eval(prog) == "SIM_OK"
