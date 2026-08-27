# tests/test_replay.py
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from demo_coach.parsing import ParsedDemo
from demo_coach.tools import MatchContext
from demo_coach.web import server


@pytest.fixture()
def ctx():
    parsed = ParsedDemo(
        demo_id="replaytest", header={"map_name": "de_ancient"},
        deaths=pd.DataFrame(
            [[150, 0, "vic", "atk", None, False, "ak47", 10.0, 20.0, 30.0, 40.0]],
            columns=["tick", "total_rounds_played", "user_name", "attacker_name",
                     "assister_name", "headshot", "weapon",
                     "user_X", "user_Y", "attacker_X", "attacker_Y"]),
        hurts=pd.DataFrame(),
        rounds=pd.DataFrame([[300, 2, 8]], columns=["tick", "winner", "reason"]),
        economy=pd.DataFrame(
            [[100, "atk", 2, 4000, 4000], [100, "vic", 3, 5000, 5000]],
            columns=["tick", "name", "team_num", "balance", "current_equip_value"]),
        positions=pd.DataFrame(
            [[104, "atk", 2, 1.0, 2.0], [112, "atk", 2, 3.0, 4.0],
             [104, "vic", 3, 5.0, 6.0], [296, "atk", 2, 7.0, 8.0],
             [400, "atk", 2, 9.0, 9.0]],  # tick 400 is past round end -> excluded
            columns=["tick", "name", "team_num", "X", "Y"]),
    )
    return MatchContext(parsed, pd.DataFrame(), pd.DataFrame(), [], {})


@pytest.fixture(autouse=True)
def clean_state(ctx):
    server.CONTEXTS.clear()
    server.CONTEXTS["replaytest"] = ctx
    yield
    server.CONTEXTS.clear()


def test_replay_frames():
    c = TestClient(server.app)
    r = c.get("/api/matches/replaytest/rounds/0/replay")
    assert r.status_code == 200
    j = r.json()
    assert j["start"] == 100 and j["end"] == 300
    players = {p["name"]: p for p in j["players"]}
    assert players["atk"]["team"] == 2
    assert players["vic"]["team"] == 3
    # frames within [start, end] only
    assert players["atk"]["frames"] == [[104, 1.0, 2.0], [112, 3.0, 4.0], [296, 7.0, 8.0]]
    assert players["vic"]["frames"] == [[104, 5.0, 6.0]]
    # vic died at tick 150 -> death_tick exposed so frontend can fade the dot
    assert players["vic"]["death_tick"] == 150
    assert players["atk"]["death_tick"] is None
    assert len(j["kills"]) == 1


def test_replay_unknown_demo():
    c = TestClient(server.app)
    assert c.get("/api/matches/nope/rounds/0/replay").status_code == 404
