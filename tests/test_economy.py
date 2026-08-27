import pandas as pd
from demo_coach import economy


def _econ():
    # round 0: Ts average 4000 (force), CTs 5000 (full)
    # round 1: Ts 500 (eco), CTs 5200 (full)
    return pd.DataFrame([
        [100, "t1", 2, 4000, 4000], [100, "t2", 2, 4000, 4000],
        [100, "c1", 3, 5000, 5000], [100, "c2", 3, 5000, 5000],
        [500, "t1", 2, 500, 500],   [500, "t2", 2, 500, 500],
        [500, "c1", 3, 5200, 5200], [500, "c2", 3, 5200, 5200],
    ], columns=["tick", "name", "team_num", "balance", "current_equip_value"])


def _rounds():
    return pd.DataFrame([
        [400, 2, 8],   # round 0: Ts win by elimination
        [900, 3, 9],   # round 1: CTs win
    ], columns=["tick", "winner", "reason"])


def test_classify_buys():
    buys = economy.classify_buys(_econ())
    r0 = buys[buys["round"] == 0].set_index("team_num")
    assert r0.loc[2, "buy_type"] == "force"
    assert r0.loc[3, "buy_type"] == "full"
    r1 = buys[buys["round"] == 1].set_index("team_num")
    assert r1.loc[2, "buy_type"] == "eco"


def test_round_log():
    log = economy.round_log(_rounds(), economy.classify_buys(_econ()))
    assert list(log.columns) == ["round", "winner_side", "reason_text", "t_buy", "ct_buy"]
    assert log.iloc[0].winner_side == "T"
    assert log.iloc[0].reason_text == "ts_win"
    assert log.iloc[1].ct_buy == "full"
    assert log.iloc[1].t_buy == "eco"


def test_round_log_string_reasons():
    # real CS2 demos report reason/winner as strings: "ct_killed" means the
    # CTs were eliminated, so the winner is T (and vice versa)
    rounds = pd.DataFrame([
        [400, 2, "ct_killed"],   # Ts win by eliminating the CTs
        [900, 3, "t_killed"],    # CTs win by eliminating the Ts
    ], columns=["tick", "winner", "reason"])
    log = economy.round_log(rounds, economy.classify_buys(_econ()))
    assert log.iloc[0].winner_side == "T"
    assert log.iloc[0].reason_text == "ts_win"
    assert log.iloc[1].winner_side == "CT"
    assert log.iloc[1].reason_text == "cts_win"
