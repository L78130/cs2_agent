# tests/test_render_smoke.py
"""Headless smoke test of the radar renderer (index.html <script>).

Runs the page's JS under a V8 engine (mini-racer) with stubbed DOM/canvas,
then drives render() across a whole replay and the static kill map. Catches
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
    clearRect(){}, drawImage(){}, beginPath(){}, moveTo(){}, lineTo(){},
    stroke(){}, fill(){}, arc(){}, roundRect(){}, fillText(){}, save(){}, restore(){},
    closePath(){}, setLineDash(){},
    measureText(t){ return { width: String(t).length * 6 }; },
    createRadialGradient(x0,y0,r0,x1,y1,r1){
      if ([x0,y0,r0,x1,y1,r1].some(v => typeof v !== "number" || isNaN(v)))
        throw new Error("NaN gradient");
      return grad;
    },
    getImageData(x,y,w,h){ return { data: new Uint8ClampedArray(w*h*4) }; },
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
function Image() {
  const img = { complete: true, naturalWidth: 88, naturalHeight: 32, onload: null, onerror: null };
  Object.defineProperty(img, "src", { set(v) { if (img.onload) img.onload(); }, get(){ return ""; } });
  return img;
}
const performance = { now(){ return 1000; } };
let __rafCb = null;
function requestAnimationFrame(cb){ __rafCb = cb; }
function fetch(){ throw new Error("fetch should not be called in this harness"); }
function alert(m){ throw new Error("alert: " + m); }
"""

DRIVER = """
// static kill-map scene
mapData = JSON.parse(MAP_JSON);
radarImgReady = true;
wallMask = extractWallMask(new Image());
render();
// replay scene: sweep the whole round
replayData = JSON.parse(REPLAY_JSON);
replayData.players.forEach(p => p._i = 0);
var step = (replayData.end - replayData.start) / 100 || 1;
for (var i = 0; i <= 100; i++) {
  playTick = replayData.start + step * i;
  render();
}
"SIM_OK";
"""


@pytest.fixture()
def ctx():
    parsed = ParsedDemo(
        demo_id="rendertest", header={"map_name": "de_ancient"},
        deaths=pd.DataFrame(
            [[150, 0, "vic", "atk", None, True, "ak47", 10.0, 20.0, 30.0, 40.0]],
            columns=["tick", "total_rounds_played", "user_name", "attacker_name",
                     "assister_name", "headshot", "weapon",
                     "user_X", "user_Y", "attacker_X", "attacker_Y"]),
        hurts=pd.DataFrame(
            [[160, "vic", "atk", 25, 10.0, 20.0, 30.0, 40.0]],
            columns=["tick", "user_name", "attacker_name", "dmg_health",
                     "user_X", "user_Y", "attacker_X", "attacker_Y"]),
        rounds=pd.DataFrame([[300, 2, 8]], columns=["tick", "winner", "reason"]),
        economy=pd.DataFrame(
            [[100, "atk", 2, 4000, 4000], [100, "vic", 3, 5000, 5000]],
            columns=["tick", "name", "team_num", "balance", "current_equip_value"]),
        positions=pd.DataFrame(
            [[104, "atk", 2, 1.0, 2.0, "AK-47", ["AK-47", "Glock-18"], 90.0],
             [200, "atk", 2, 3.0, 4.0, "AK-47", ["AK-47", "Glock-18"], 45.0],
             [104, "vic", 3, 5.0, 6.0, "USP-S", ["Knife", "USP-S"], -90.0]],
            columns=["tick", "name", "team_num", "X", "Y",
                     "active_weapon_name", "inventory", "yaw"]),
        fires=pd.DataFrame([[140, "atk", 3.5, 4.5, "ak47"]],
                           columns=["tick", "user_name", "user_X", "user_Y", "weapon"]),
        bombs=pd.DataFrame([[200, "bomb_planted", "atk", None]],
                           columns=["tick", "event", "user_name", "site"]),
    )
    return MatchContext(parsed, pd.DataFrame(), pd.DataFrame(), [], {})


def test_radar_render_smoke(ctx):
    server.CONTEXTS.clear()
    server.CONTEXTS["rendertest"] = ctx
    try:
        c = TestClient(server.app)
        replay = c.get("/api/matches/rendertest/rounds/0/replay").json()
        killmap = c.get("/api/matches/rendertest/rounds/0/map").json()
    finally:
        server.CONTEXTS.clear()

    html = (server.STATIC_DIR / "index.html").read_text(encoding="utf-8")
    js = re.search(r"<script>(.*?)</script>", html, re.S).group(1)
    js = js.replace("refreshMatches();", "")  # no fetch in the harness

    prog = (DOM_STUB + js
            + "\nvar REPLAY_JSON = " + json.dumps(json.dumps(replay)) + ";"
            + "\nvar MAP_JSON = " + json.dumps(json.dumps(killmap)) + ";\n"
            + DRIVER)
    assert MiniRacer().eval(prog) == "SIM_OK"
