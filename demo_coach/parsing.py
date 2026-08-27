import hashlib
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from demoparser2 import DemoParser

TICK_RATE = 64  # CS2 demo tick rate; used for time-window calculations


@dataclass
class ParsedDemo:
    demo_id: str
    header: dict
    deaths: pd.DataFrame    # player_death events
    hurts: pd.DataFrame     # player_hurt events
    rounds: pd.DataFrame    # round_end events
    economy: pd.DataFrame   # balance/equip_value snapshot at each round_freeze_end


def file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _add_round_index(deaths: pd.DataFrame, rounds: pd.DataFrame) -> pd.DataFrame:
    """player_death has no round column; derive the 0-based round index from
    round_end ticks (a death belongs to the first round whose end tick is
    >= the death tick)."""
    if deaths.empty or rounds.empty:
        return deaths.assign(total_rounds_played=0)
    end_ticks = np.sort(rounds["tick"].to_numpy())
    idx = np.searchsorted(end_ticks, deaths["tick"].to_numpy(), side="left")
    idx = np.clip(idx, 0, len(end_ticks) - 1)
    return deaths.assign(total_rounds_played=idx)


def _check_round_alignment(n_rounds: int, n_freeze_ends: int) -> None:
    """Death round indices derive from round_end ticks while economy round
    indices derive from freeze_end tick order; warn if the counts disagree."""
    if n_rounds != n_freeze_ends:
        warnings.warn(
            f"round count mismatch: {n_rounds} round_end events vs "
            f"{n_freeze_ends} round_freeze_end events; death and economy "
            "round indices may be misaligned"
        )


def parse_demo(path: str) -> ParsedDemo:
    parser = DemoParser(path)
    header = dict(parser.parse_header())
    rounds = parser.parse_event("round_end")
    deaths = _add_round_index(parser.parse_event("player_death"), rounds)
    hurts = parser.parse_event("player_hurt")
    freeze_ends = parser.parse_event("round_freeze_end")
    _check_round_alignment(len(rounds), len(freeze_ends))
    if len(freeze_ends) > 0:
        economy = parser.parse_ticks(
            ["balance", "current_equip_value", "team_num"],
            ticks=freeze_ends["tick"].tolist(),
        )
    else:
        economy = pd.DataFrame(
            columns=["tick", "name", "team_num", "balance", "current_equip_value"]
        )
    return ParsedDemo(
        demo_id=file_hash(path), header=header,
        deaths=deaths, hurts=hurts, rounds=rounds, economy=economy,
    )
