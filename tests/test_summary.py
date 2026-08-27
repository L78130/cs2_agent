# tests/test_summary.py
import json
import pandas as pd
from demo_coach import summary
from demo_coach.parsing import ParsedDemo


def test_build_summary_json_serializable():
    parsed = ParsedDemo(
        demo_id="abc123", header={"map_name": "de_mirage"},
        deaths=pd.DataFrame(), hurts=pd.DataFrame(),
        rounds=pd.DataFrame(), economy=pd.DataFrame(),
    )
    sb = pd.DataFrame([{"name": "alice", "kills": 20, "adr": 95.5}])
    rlog = pd.DataFrame([{"round": 0, "winner_side": "T",
                          "reason_text": "ts_win", "t_buy": "full", "ct_buy": "eco"}])
    hl = [{"type": "ace", "round": 0, "tick": 140, "player": "alice", "detail": "5 kills"}]
    s = summary.build_summary(parsed, sb, rlog, hl)
    assert s["demo_id"] == "abc123"
    assert s["map"] == "de_mirage"
    assert s["scoreboard"][0]["name"] == "alice"
    json.dumps(s)  # must not raise


def test_missing_map_falls_back():
    parsed = ParsedDemo(demo_id="x", header={}, deaths=pd.DataFrame(),
                        hurts=pd.DataFrame(), rounds=pd.DataFrame(),
                        economy=pd.DataFrame())
    s = summary.build_summary(parsed, pd.DataFrame(), pd.DataFrame(), [])
    assert s["map"] == "unknown"
    assert s["rounds_played"] == 0
