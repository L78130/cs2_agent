import json
import pandas as pd
from demo_coach import tools
from demo_coach.parsing import ParsedDemo


def _ctx():
    parsed = ParsedDemo(
        demo_id="abc123", header={"map_name": "de_mirage"},
        deaths=pd.DataFrame(
            [[100, 0, "alice", "bob", None, True, "ak47", 26, 0]],
            columns=["tick", "total_rounds_played", "user_name", "attacker_name",
                     "assister_name", "headshot", "weapon", "dmg_health", "dmg_armor"]),
        hurts=pd.DataFrame([["bob", 80, 0]],
                           columns=["attacker_name", "dmg_health",
                                    "total_rounds_played"]),
        rounds=pd.DataFrame([[400, 2, 8]], columns=["tick", "winner", "reason"]),
        economy=pd.DataFrame(
            [[50, "bob", 2, 4000, 4000], [50, "alice", 3, 5000, 5000]],
            columns=["tick", "name", "team_num", "balance", "current_equip_value"]),
    )
    return tools.build_context_from_parsed(parsed)


def test_dispatch_scoreboard():
    out = json.loads(tools.dispatch(_ctx(), "get_scoreboard", {}))
    assert any(r["name"] == "bob" for r in out)


def test_dispatch_round():
    out = json.loads(tools.dispatch(_ctx(), "get_round", {"round": 0}))
    assert out["winner_side"] == "T"
    assert "kills" in out  # round detail includes kill list


def test_dispatch_unknown_tool():
    out = tools.dispatch(_ctx(), "nonsense", {})
    assert "error" in out
