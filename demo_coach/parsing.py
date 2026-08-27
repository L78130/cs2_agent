import hashlib
from dataclasses import dataclass

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


def parse_demo(path: str) -> ParsedDemo:
    parser = DemoParser(path)
    header = dict(parser.parse_header())
    deaths = parser.parse_event("player_death")
    hurts = parser.parse_event("player_hurt")
    rounds = parser.parse_event("round_end")
    freeze_ends = parser.parse_event("round_freeze_end")
    if len(freeze_ends) > 0:
        economy = parser.parse_ticks(
            ["balance", "current_equip_value"],
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
