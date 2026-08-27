import pandas as pd

ECO_MAX = 2000
FORCE_MAX = 4500
REASONS = {1: "target_bombed", 7: "bomb_defused", 8: "ts_win",
           9: "cts_win", 12: "target_saved",
           # real CS2 demos report reason as a string, not an int code
           "bomb_exploded": "target_bombed", "bomb_defused": "bomb_defused",
           "t_killed": "ts_win", "ct_killed": "cts_win",
           "time_ran_out": "target_saved"}
SIDE = {2: "T", 3: "CT"}


def _buy_type(avg_equip: float) -> str:
    if avg_equip < ECO_MAX:
        return "eco"
    if avg_equip < FORCE_MAX:
        return "force"
    return "full"


def classify_buys(econ: pd.DataFrame) -> pd.DataFrame:
    if econ.empty:
        return pd.DataFrame(columns=["round", "team_num", "avg_equip", "buy_type"])
    ticks = sorted(econ["tick"].unique())
    round_of = {t: i for i, t in enumerate(ticks)}
    df = econ.assign(round=econ["tick"].map(round_of))
    out = (df.groupby(["round", "team_num"])["current_equip_value"]
           .mean().round(0).rename("avg_equip").reset_index())
    out["buy_type"] = out["avg_equip"].map(_buy_type)
    return out


def round_log(rounds: pd.DataFrame, buys: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pivot = buys.pivot_table(index="round", columns="team_num",
                             values="buy_type", aggfunc="first")
    for i, r in rounds.sort_values("tick").reset_index(drop=True).iterrows():
        rows.append({
            "round": i,
            "winner_side": SIDE.get(r["winner"], str(r["winner"])),
            "reason_text": REASONS.get(r["reason"], f"code_{r['reason']}"),
            "t_buy": pivot.get(2, pd.Series(dtype=object)).get(i, "unknown"),
            "ct_buy": pivot.get(3, pd.Series(dtype=object)).get(i, "unknown"),
        })
    return pd.DataFrame(rows, columns=["round", "winner_side", "reason_text",
                                       "t_buy", "ct_buy"])
