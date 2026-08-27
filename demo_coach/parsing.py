import hashlib
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from demoparser2 import DemoParser

TICK_RATE = 64  # CS2 demo tick rate; used for time-window calculations
SAMPLE_EVERY = 8  # position sampling stride: 8 ticks = 8 samples/sec at 64 tick


@dataclass
class ParsedDemo:
    demo_id: str
    header: dict
    deaths: pd.DataFrame    # player_death events
    hurts: pd.DataFrame     # player_hurt events
    rounds: pd.DataFrame    # round_end events
    economy: pd.DataFrame   # balance/equip_value snapshot at each round_freeze_end
    positions: pd.DataFrame = field(default_factory=pd.DataFrame)  # sampled player positions (every SAMPLE_EVERY ticks)
    fires: pd.DataFrame = field(default_factory=pd.DataFrame)   # weapon_fire events (shooter X/Y + weapon)
    bombs: pd.DataFrame = field(default_factory=pd.DataFrame)   # bomb_* events with an "event" column


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
    deaths = _add_round_index(
        parser.parse_event("player_death", player=["X", "Y"]), rounds
    )
    hurts = parser.parse_event("player_hurt", player=["X", "Y"])
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
    if len(rounds) > 0:
        last_tick = int(rounds["tick"].max()) + TICK_RATE * 5
        positions = parser.parse_ticks(
            ["X", "Y", "team_num", "active_weapon_name", "inventory", "yaw"],
            ticks=list(range(0, last_tick, SAMPLE_EVERY)),
        )
    else:
        positions = pd.DataFrame(columns=["tick", "name", "team_num", "X", "Y"])
    fires = parser.parse_event("weapon_fire", player=["X", "Y"])
    bombs = _bomb_events(parser)
    return ParsedDemo(
        demo_id=file_hash(path), header=header,
        deaths=deaths, hurts=hurts, rounds=rounds, economy=economy,
        positions=positions, fires=fires, bombs=bombs,
    )


_BOMB_EVENTS = ["bomb_dropped", "bomb_pickup", "bomb_planted",
                "bomb_defused", "bomb_exploded"]


def _bomb_events(parser: DemoParser) -> pd.DataFrame:
    """All bomb_* events in one frame: tick, event, user_name, site."""
    frames = []
    for ev in _BOMB_EVENTS:
        df = parser.parse_event(ev)
        if len(df) > 0:
            keep = [c for c in ["tick", "user_name", "site"] if c in df.columns]
            frames.append(df[keep].assign(event=ev))
    if not frames:
        return pd.DataFrame(columns=["tick", "event", "user_name", "site"])
    return pd.concat(frames, ignore_index=True).sort_values("tick").reset_index(drop=True)
