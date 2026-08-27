import pandas as pd
from demo_coach import highlights


def _ace_round():
    # round 0: alice kills 5 enemies -> ace; round 1: bob knife kill
    return pd.DataFrame([
        [100, 0, "e1", "alice", None, False, "ak47"],
        [110, 0, "e2", "alice", None, False, "ak47"],
        [120, 0, "e3", "alice", None, False, "ak47"],
        [130, 0, "e4", "alice", None, False, "ak47"],
        [140, 0, "e5", "alice", None, False, "ak47"],
        [500, 1, "e1", "bob", None, False, "knife"],
        [510, 1, "bob", "e2", None, False, "ak47"],
    ], columns=["tick", "total_rounds_played", "user_name", "attacker_name",
                "assister_name", "headshot", "weapon"])


def _rounds():
    return pd.DataFrame([[400, 2, 8], [900, 3, 9]],
                        columns=["tick", "winner", "reason"])


def _clutch_deaths():
    # round 0, 5v5: team A = {a1..a5}, team B = {b1..b5} (teams inferred
    # from kill directions). a2..a5 die first (A at 1 alive), then a1 kills
    # b1, b2, b3... leaving B at 2 alive -> a1 clutches if team A (T=2) wins.
    rows = [
        [100, 0, "a2", "b1", None, False, "ak47"],
        [101, 0, "a3", "b1", None, False, "ak47"],
        [102, 0, "a4", "b2", None, False, "ak47"],
        [103, 0, "a5", "b2", None, False, "ak47"],  # A down to a1 alone; B has 5
        [110, 0, "b1", "a1", None, False, "ak47"],
        [111, 0, "b2", "a1", None, False, "ak47"],
        [112, 0, "b3", "a1", None, False, "ak47"],
    ]
    return pd.DataFrame(rows, columns=["tick", "total_rounds_played",
                                       "user_name", "attacker_name",
                                       "assister_name", "headshot", "weapon"])


def _clutch_rounds():
    return pd.DataFrame([[400, 2, 8]], columns=["tick", "winner", "reason"])


def test_ace_and_knife():
    hl = highlights.find_highlights(_ace_round(), _rounds())
    types = {(h["type"], h["player"]) for h in hl}
    assert ("ace", "alice") in types
    assert ("knife", "bob") in types
    ace = [h for h in hl if h["type"] == "ace"][0]
    assert ace["round"] == 0


def test_clutch():
    hl = highlights.find_highlights(_clutch_deaths(), _clutch_rounds(), team_size=5)
    clutches = [h for h in hl if h["type"] == "clutch"]
    assert len(clutches) == 1
    assert clutches[0]["player"] == "a1"
