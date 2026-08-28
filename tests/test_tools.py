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


def test_dispatch_scoreboard_has_team():
    out = json.loads(tools.dispatch(_ctx(), "get_scoreboard", {}))
    by_name = {r["name"]: r for r in out}
    assert by_name["bob"]["team"] == 2 and by_name["alice"]["team"] == 3


def test_dispatch_scoreboard_has_utility():
    # fixture has no fires frame -> utility fields present and zero
    out = json.loads(tools.dispatch(_ctx(), "get_scoreboard", {}))
    bob = next(r for r in out if r["name"] == "bob")
    for k in ("util_dmg", "util_dmg_r", "flash_assists",
              "flashes_thrown", "nades_thrown"):
        assert bob[k] == 0


def test_dispatch_player_rounds():
    out = json.loads(tools.dispatch(_ctx(), "get_player_rounds", {"name": "bob"}))
    assert len(out) == 1          # one round in the fixture
    r = out[0]
    assert r["round"] == 0 and r["kills"] == 1 and r["headshots"] == 1
    assert r["died"] is False and r["first_kill"] is True
    assert r["dmg"] == 80         # from the hurts frame
    assert r["highlight"] is None
    out = json.loads(tools.dispatch(_ctx(), "get_player_rounds", {"name": "alice"}))
    assert out[0]["died"] is True and out[0]["kills"] == 0
