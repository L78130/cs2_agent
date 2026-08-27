# tests/test_tool_endpoint.py
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from demo_coach.parsing import ParsedDemo
from demo_coach.tools import MatchContext
from demo_coach.web import server


@pytest.fixture()
def ctx():
    parsed = ParsedDemo(
        demo_id="tooltest", header={"map_name": "de_ancient"},
        deaths=pd.DataFrame(), hurts=pd.DataFrame(),
        rounds=pd.DataFrame(), economy=pd.DataFrame(),
    )
    return MatchContext(
        parsed,
        scoreboard=pd.DataFrame([{"name": "alice", "kills": 20}]),
        round_log=pd.DataFrame([{"round": 0, "winner_side": "T",
                                 "reason_text": "ts_win",
                                 "t_buy": "full", "ct_buy": "eco"}]),
        highlights=[], summary={},
    )


@pytest.fixture(autouse=True)
def clean_state(ctx):
    server.CONTEXTS.clear()
    server.CONTEXTS["tooltest"] = ctx
    yield
    server.CONTEXTS.clear()


def test_tool_scoreboard():
    c = TestClient(server.app)
    r = c.post("/api/matches/tooltest/tool/get_scoreboard", json={})
    assert r.status_code == 200
    assert r.json()[0]["name"] == "alice"


def test_tool_rounds_log():
    c = TestClient(server.app)
    r = c.post("/api/matches/tooltest/tool/get_rounds_log", json={})
    assert r.status_code == 200
    assert r.json()[0]["winner_side"] == "T"


def test_tool_unknown_demo():
    c = TestClient(server.app)
    assert c.post("/api/matches/nope/tool/get_scoreboard", json={}).status_code == 404


def test_tool_unknown_name_returns_error_payload():
    c = TestClient(server.app)
    r = c.post("/api/matches/tooltest/tool/nonsense", json={})
    assert r.status_code == 200
    assert "error" in r.json()
