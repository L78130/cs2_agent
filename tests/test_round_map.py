# tests/test_round_map.py
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from demo_coach.parsing import ParsedDemo
from demo_coach.tools import MatchContext
from demo_coach.web import server


@pytest.fixture()
def ctx():
    parsed = ParsedDemo(
        demo_id="maptest", header={"map_name": "de_ancient"},
        deaths=pd.DataFrame(
            [[3098, 0, "vic", "atk", None, True, "ak47",
              -431.87, 499.97, -732.03, -736.82]],
            columns=["tick", "total_rounds_played", "user_name", "attacker_name",
                     "assister_name", "headshot", "weapon",
                     "user_X", "user_Y", "attacker_X", "attacker_Y"]),
        hurts=pd.DataFrame(), rounds=pd.DataFrame(), economy=pd.DataFrame(),
    )
    return MatchContext(parsed, pd.DataFrame(), pd.DataFrame(), [], {})


def test_round_map(ctx):
    server.CONTEXTS.clear()
    server.CONTEXTS["maptest"] = ctx
    try:
        c = TestClient(server.app)
        r = c.get("/api/matches/maptest/rounds/0/map")
        assert r.status_code == 200
        j = r.json()
        assert j["map"] == "de_ancient"
        assert j["calibration"]["scale"] == 5
        assert j["image"] == "/static/maps/de_ancient.png"
        assert len(j["kills"]) == 1
        k = j["kills"][0]
        assert k["attacker"] == "atk" and k["victim"] == "vic"
        assert k["headshot"] is True
        assert abs(k["victim_x"] - (-431.87)) < 0.01
        assert abs(k["attacker_y"] - (-736.82)) < 0.01
    finally:
        server.CONTEXTS.clear()


def test_round_map_unknown_demo():
    server.CONTEXTS.clear()
    c = TestClient(server.app)
    assert c.get("/api/matches/nope/rounds/0/map").status_code == 404


def test_round_map_unknown_map(ctx):
    ctx.parsed.header["map_name"] = "de_nowhere"
    server.CONTEXTS.clear()
    server.CONTEXTS["maptest"] = ctx
    try:
        c = TestClient(server.app)
        r = c.get("/api/matches/maptest/rounds/0/map")
        assert r.status_code == 404
        assert "de_nowhere" in r.json()["detail"]
    finally:
        server.CONTEXTS.clear()
