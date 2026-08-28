# demo_coach/replay.py
"""2D round replay builder, modeled on the replay module of
DrEAmSs59/CS2-insight-agent (its JSON replay path).

Frames are sampled at REPLAY_FPS with a rich per-player prop set; shots are
attached to the nearest frame; kills / bomb / grenade detonations become
event tracks; smokes and infernos become "legacy circle" effect tracks
(stock demoparser2 cannot decode smoke voxel journals, so there is no
cell-level geometry — effect_capabilities.smoke_mode says so).

The heavy whole-match parse runs once per demo, lazily on the first replay
request; every round is then cached as gzip JSON under
<storage.CACHE_DIR>/replay/v<REPLAY_CACHE_VERSION>/<demo_id>/round_<n>.json.gz.
"""
import bisect
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
from demoparser2 import DemoParser

from demo_coach import storage
from demo_coach.parsing import TICK_RATE, _add_round_index, _bomb_events
from demo_coach.radar import MAPS_DIR, load_calibration, short_weapon

REPLAY_FPS = 32
STEP = max(1, TICK_RATE // REPLAY_FPS)  # sample stride in ticks (2 at 64 tick)
REPLAY_CACHE_VERSION = 1

SMOKE_DURATION_SEC = 18.0
SMOKE_RADIUS = 144.0      # world units, ~ CS2 smoke cloud radius
INFERNO_DURATION_SEC = 7.0
INFERNO_RADIUS = 150.0    # world units, rough molotov spread
ROUND_TAIL_SEC = 3.0      # tail after the last round_end
GRENADE_MAX_FLIGHT_SEC = 12.0  # longest plausible throw -> detonation window

TICK_PROPS = [
    "X", "Y", "Z", "yaw", "team_num", "is_alive", "health", "armor",
    "has_helmet", "balance", "current_equip_value", "inventory",
    "active_weapon_name", "has_defuser", "has_c4", "flash_duration",
]

_BOMB_EVENT_TYPE = {
    "bomb_planted": "plant", "bomb_defused": "defuse", "bomb_exploded": "explode",
    "bomb_dropped": "bomb_drop", "bomb_pickup": "bomb_pickup",
}
# weapon_fire names for things that are not bullets (grenades, knife, C4...)
_NON_BULLET = {
    "knife", "c4", "taser", "zeus", "smokegrenade", "hegrenade", "flashbang",
    "molotov", "incgrenade", "decoy", "breachcharge", "bumpmine", "snowball",
    "frag", "firebomb", "diversion", "tagrenade",
}
_DETONATION_KIND = {
    "smokegrenade_detonate": "smoke",
    "flashbang_detonate": "flash",
    "hegrenade_detonate": "he",
    "decoy_detonate": "decoy",
}
# fire grenades: inferno_startburn is the reliable detonation event in CS2
# demos (molotov_detonate / incgrenade_detonate often never fire); the two
# detonation events are the fallback. inferno_expire/extinguish end a track.
_FIRE_START_EVENT = "inferno_startburn"
_FIRE_FALLBACK_EVENTS = ("molotov_detonate", "incgrenade_detonate")
_FIRE_END_EVENTS = ("inferno_expire", "inferno_extinguish")


# ---------------------------------------------------------------- pure helpers

def round_windows(round_start: list[int], freeze_end: list[int],
                  round_end: list[int], demo_end_tick: int) -> list[tuple[int, int]]:
    """(start_tick, end_tick) per round: play starts at freeze end and runs
    until just before the next round starts (post-round time included); the
    last round gets a short tail after round_end."""
    windows = []
    n = len(round_end)
    for i in range(n):
        if i < len(freeze_end):
            start = freeze_end[i]
        elif i < len(round_start):
            start = round_start[i]
        else:
            start = round_end[i - 1] if i else 0
        if i + 1 < len(round_start):
            end = round_start[i + 1] - 1
        else:
            end = min(round_end[i] + int(ROUND_TAIL_SEC * TICK_RATE), demo_end_tick)
        windows.append((int(start), int(max(end, start + 1))))
    return windows


def _team_letter(team_num) -> str:
    try:
        return {2: "T", 3: "CT"}.get(int(team_num), "?")
    except (TypeError, ValueError):
        return "?"


def _num(v, nd=1):
    return round(float(v), nd) if v is not None and not pd.isna(v) else 0.0


def _int(v, default=0):
    return int(v) if v is not None and not pd.isna(v) else default


def _inventory(raw) -> list[str]:
    if raw is None or (not hasattr(raw, "__len__") and pd.isna(raw)):
        return []
    return sorted(short_weapon(w) for w in raw)


def _player_dict(r: pd.Series) -> dict:
    return {
        "name": r.get("name"),
        "steamid64": str(r["steamid"]) if "steamid" in r.index and not pd.isna(r["steamid"]) else None,
        "team": _team_letter(r.get("team_num")),
        "x": _num(r.get("X")), "y": _num(r.get("Y")), "z": _num(r.get("Z")),
        "yaw": _num(r.get("yaw")),
        "is_alive": bool(r.get("is_alive", True)),
        "health": _int(r.get("health"), 100),
        "armor": _int(r.get("armor")),
        "has_helmet": bool(r.get("has_helmet", False)),
        "money": _int(r.get("balance")),
        "equipment_value": _int(r.get("current_equip_value")),
        "inventory": _inventory(r.get("inventory")),
        "weapon": short_weapon(r.get("active_weapon_name")),
        "has_defuser": bool(r.get("has_defuser", False)),
        "has_c4": bool(r.get("has_c4", False)),
        "flash_duration": _num(r.get("flash_duration"), 2),
    }


def frames_from_ticks(ticks: pd.DataFrame, start: int, end: int) -> list[dict]:
    """One frame per sampled tick in [start, end], each with its player list."""
    if ticks.empty:
        return []
    win = ticks[(ticks["tick"] >= start) & (ticks["tick"] <= end)]
    frames = []
    for tick, g in win.groupby("tick", sort=True):
        frames.append({
            "tick": int(tick),
            "time_sec": round((int(tick) - start) / TICK_RATE, 3),
            "players": [_player_dict(r) for _, r in g.iterrows()],
            "shots": [],
        })
    return frames


def attach_shots(frames: list[dict], fires: pd.DataFrame, start: int, end: int) -> None:
    """Attach weapon_fire rows (bullets only) to the nearest frame in place."""
    if not frames or fires.empty:
        return
    ticks = [f["tick"] for f in frames]
    win = fires[(fires["tick"] >= start) & (fires["tick"] <= end)]
    for _, f in win.iterrows():
        weapon = str(f.get("weapon") or "").removeprefix("weapon_").lower()
        if any(nb in weapon for nb in _NON_BULLET):
            continue
        i = bisect.bisect_left(ticks, f["tick"])
        if i > 0 and (i == len(ticks) or f["tick"] - ticks[i - 1] <= ticks[i] - f["tick"]):
            i -= 1
        yaw = f.get("user_yaw")
        if yaw is None or pd.isna(yaw):
            yaw = f.get("yaw")
        pitch = f.get("user_pitch")
        if pitch is None or pd.isna(pitch):
            pitch = f.get("pitch")
        frames[i]["shots"].append({
            "tick": int(f["tick"]),
            "actor": f.get("user_name"),
            "weapon": weapon,
            "yaw": _num(yaw),
            "pitch": _num(pitch),
            "x": _num(f.get("user_X")) if not pd.isna(f.get("user_X")) else None,
            "y": _num(f.get("user_Y")) if not pd.isna(f.get("user_Y")) else None,
        })


def kill_events(deaths: pd.DataFrame, round_n: int) -> list[dict]:
    if deaths.empty:
        return []
    kills = deaths[deaths["total_rounds_played"] == round_n]
    out = []
    for _, k in kills.iterrows():
        out.append({
            "type": "kill",
            "tick": int(k["tick"]),
            "actor": k.get("attacker_name") if not pd.isna(k.get("attacker_name")) else None,
            "target": k.get("user_name"),
            "weapon": k.get("weapon"),
            "headshot": bool(k.get("headshot", False)),
            "assister": k.get("assister_name") if not pd.isna(k.get("assister_name")) else None,
            "actor_x": _num(k.get("attacker_X")) if not pd.isna(k.get("attacker_X")) else None,
            "actor_y": _num(k.get("attacker_Y")) if not pd.isna(k.get("attacker_Y")) else None,
            "target_x": _num(k.get("user_X")) if not pd.isna(k.get("user_X")) else None,
            "target_y": _num(k.get("user_Y")) if not pd.isna(k.get("user_Y")) else None,
        })
    return out


def _pos_lookup(ticks: pd.DataFrame) -> dict[str, tuple[list[int], list[tuple]]]:
    """name -> (sorted tick list, [(x, y), ...]) for nearest-past lookups."""
    out = {}
    if ticks.empty:
        return out
    for name, g in ticks.groupby("name"):
        g = g.sort_values("tick")
        out[name] = (g["tick"].astype(int).tolist(),
                     list(zip(g["X"].astype(float), g["Y"].astype(float))))
    return out


def _pos_at(lookup, name, tick):
    ent = lookup.get(name)
    if not ent:
        return None
    ts, xy = ent
    i = bisect.bisect_right(ts, tick) - 1
    if i < 0:
        return None
    return (round(xy[i][0], 1), round(xy[i][1], 1))


def bomb_events(bombs: pd.DataFrame, start: int, end: int, lookup) -> list[dict]:
    if bombs.empty:
        return []
    win = bombs[(bombs["tick"] >= start) & (bombs["tick"] <= end)]
    out = []
    for _, b in win.iterrows():
        xy = _pos_at(lookup, b.get("user_name"), int(b["tick"])) \
            if not pd.isna(b.get("user_name")) else None
        out.append({
            "type": _BOMB_EVENT_TYPE.get(b["event"], b["event"]),
            "tick": int(b["tick"]),
            "actor": b.get("user_name") if not pd.isna(b.get("user_name")) else None,
            "site": b.get("site") if "site" in bombs.columns and not pd.isna(b.get("site")) else None,
            "x": xy[0] if xy else None,
            "y": xy[1] if xy else None,
        })
    return out


def _kind_of_grenade(gtype: str) -> str | None:
    g = str(gtype).lower()
    if "smoke" in g:
        return "smoke"
    if "flash" in g:
        return "flash"
    if "molotov" in g:
        return "molotov"
    if "inc" in g or "firebomb" in g:
        return "incendiary"
    if "he" in g or "frag" in g:
        return "he"
    if "decoy" in g or "diversion" in g:
        return "decoy"
    return None


def _fire_detonations(detonations: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Fire-grenade detonations: inferno_startburn when the demo has it,
    else molotov/incendiary detonate events as fallback."""
    start = detonations.get(_FIRE_START_EVENT)
    if start is not None and not start.empty:
        return start
    frames = [detonations[e] for e in _FIRE_FALLBACK_EVENTS
              if detonations.get(e) is not None and not detonations[e].empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def grenade_events(detonations: dict[str, pd.DataFrame], grenades: pd.DataFrame,
                   start: int, end: int) -> list[dict]:
    """Grenade detonation events in the window, each spliced with its flight
    trajectory from parse_grenades (contiguous tail segment landing within
    2s of the detonation)."""
    sources = [(df, kind) for ev, kind in _DETONATION_KIND.items()
               if (df := detonations.get(ev)) is not None and not df.empty]
    fire = _fire_detonations(detonations)
    if not fire.empty:
        sources.append((fire, "molotov"))
    out = []
    gap = TICK_RATE * 2
    for df, kind in sources:
        win = df[(df["tick"] >= start) & (df["tick"] <= end)]
        for _, d in win.iterrows():
            det_tick = int(d["tick"])
            thrower = d.get("user_name") if not pd.isna(d.get("user_name")) else None
            traj = _trajectory(grenades, kind, thrower, det_tick, gap)
            out.append({
                "type": "grenade",
                "kind": kind,
                "tick": det_tick,
                "actor": thrower,
                "x": _num(d.get("x")) if not pd.isna(d.get("x")) else None,
                "y": _num(d.get("y")) if not pd.isna(d.get("y")) else None,
                "z": _num(d.get("z")) if not pd.isna(d.get("z")) else None,
                "throw_tick": traj[0]["tick"] if traj else None,
                "trajectory": traj,
            })
    out.sort(key=lambda e: e["tick"])
    return out


# trajectory lookup accepts these grenade class-name kinds per event kind
# (a CT incendiary and a T molotov both produce inferno_startburn events)
_KIND_MATCH = {"molotov": {"molotov", "incendiary"}}


def _trajectory(grenades: pd.DataFrame, kind: str, thrower, det_tick: int,
                gap: int) -> list[dict]:
    if grenades.empty or not {"tick", "x", "y", "z"} <= set(grenades.columns):
        return []
    g = grenades[(grenades["tick"] <= det_tick)
                 & (grenades["tick"] >= det_tick - int(GRENADE_MAX_FLIGHT_SEC * TICK_RATE))]
    # non-projectile rows (the grenade item entity) carry NaN positions
    g = g[g["x"].notna()]
    if "grenade_type" in g.columns:
        kinds = _KIND_MATCH.get(kind, {kind})
        g = g[g["grenade_type"].map(_kind_of_grenade).isin(kinds)]
    if thrower is not None and "name" in g.columns and not g.empty:
        same = g[g["name"] == thrower]
        if not same.empty:
            g = same
    if g.empty:
        return []
    g = g.sort_values("tick")
    # contiguous tail segment: cut at sampling gaps so two throws of the same
    # grenade by the same player don't merge into one trajectory
    ts = g["tick"].to_numpy()
    cut = np.flatnonzero(np.diff(ts) > gap)
    seg = g.iloc[cut[-1] + 1:] if len(cut) else g
    if det_tick - int(seg["tick"].iloc[-1]) > gap:
        return []  # last seen position is too far from the detonation
    return [{"tick": int(t), "x": round(float(x), 1), "y": round(float(y), 1),
             "z": round(float(z), 1)}
            for t, x, y, z in zip(seg["tick"], seg["x"], seg["y"], seg["z"])]


def effect_tracks(detonations: dict[str, pd.DataFrame], start: int, end: int,
                  round_n: int) -> list[dict]:
    """Legacy-circle area effects: smoke and inferno tracks with a fixed
    radius (no cell geometry from stock demoparser2)."""
    tracks = []
    smokes = detonations.get("smokegrenade_detonate")
    if smokes is not None and not smokes.empty:
        for _, d in smokes[(smokes["tick"] >= start) & (smokes["tick"] <= end)].iterrows():
            if pd.isna(d.get("x")) or pd.isna(d.get("y")):
                continue
            tick = int(d["tick"])
            tracks.append({
                "id": f"smoke:{round_n}:{tick}",
                "type": "smoke",
                "x": _num(d["x"]), "y": _num(d["y"]), "z": _num(d.get("z")),
                "start_tick": tick,
                "end_tick": tick + int(SMOKE_DURATION_SEC * TICK_RATE),
                "radius": SMOKE_RADIUS,
            })
    # infernos: end at inferno_expire/extinguish for the same entity when
    # known, else the nominal burn duration
    fire_end: dict[int, int] = {}
    for ev in _FIRE_END_EVENTS:
        df = detonations.get(ev)
        if df is None or df.empty or "entityid" not in df.columns:
            continue
        for _, d in df.iterrows():
            eid = int(d["entityid"])
            fire_end[eid] = min(fire_end.get(eid, 1 << 62), int(d["tick"]))
    fire = _fire_detonations(detonations)
    if not fire.empty:
        for _, d in fire[(fire["tick"] >= start) & (fire["tick"] <= end)].iterrows():
            if pd.isna(d.get("x")) or pd.isna(d.get("y")):
                continue
            tick = int(d["tick"])
            end_tick = tick + int(INFERNO_DURATION_SEC * TICK_RATE)
            if "entityid" in fire.columns and not pd.isna(d.get("entityid")):
                end_tick = fire_end.get(int(d["entityid"]), end_tick)
            tracks.append({
                "id": f"inferno:{round_n}:{tick}",
                "type": "inferno",
                "x": _num(d["x"]), "y": _num(d["y"]), "z": _num(d.get("z")),
                "start_tick": tick,
                "end_tick": max(end_tick, tick + 1),
                "radius": INFERNO_RADIUS,
            })
    tracks.sort(key=lambda t: t["start_tick"])
    return tracks


# ---------------------------------------------------------------- heavy build

def _try_parse_event(parser: DemoParser, name: str, **kwargs) -> pd.DataFrame:
    try:
        df = parser.parse_event(name, **kwargs)
    except Exception:
        return pd.DataFrame()
    # demoparser2 returns a plain [] (not an empty DataFrame) when the event
    # never fired in the demo
    if isinstance(df, list):
        return pd.DataFrame(df)
    return df


def replay_cache_dir(demo_id: str) -> Path:
    return storage.CACHE_DIR / "replay" / f"v{REPLAY_CACHE_VERSION}" / demo_id


def build_match_replay(demo_path: str, demo_id: str, map_name: str) -> None:
    """Parse the whole demo once and write one gzip JSON per round."""
    parser = DemoParser(demo_path)
    round_end = _try_parse_event(parser, "round_end")
    if round_end.empty:
        raise ValueError("no rounds in demo")
    round_start = _try_parse_event(parser, "round_start")
    freeze_end = _try_parse_event(parser, "round_freeze_end")

    demo_end_tick = int(round_end["tick"].max()) + int(ROUND_TAIL_SEC * TICK_RATE)
    windows = round_windows(
        round_start["tick"].astype(int).tolist() if not round_start.empty else [],
        freeze_end["tick"].astype(int).tolist() if not freeze_end.empty else [],
        round_end["tick"].astype(int).tolist(),
        demo_end_tick,
    )
    all_ticks = sorted({t for s, e in windows for t in range(s, e + 1, STEP)})
    ticks = parser.parse_ticks(TICK_PROPS, ticks=all_ticks) if all_ticks else pd.DataFrame()

    fires = _try_parse_event(parser, "weapon_fire",
                             player=["X", "Y", "yaw", "pitch"])
    deaths = _add_round_index(
        _try_parse_event(parser, "player_death", player=["X", "Y"]), round_end)
    bombs = _bomb_events(parser)
    detonations = {name: _try_parse_event(parser, name)
                   for name in [*_DETONATION_KIND, _FIRE_START_EVENT,
                                *_FIRE_FALLBACK_EVENTS, *_FIRE_END_EVENTS]}
    try:
        grenades = parser.parse_grenades()
    except Exception:
        grenades = pd.DataFrame()

    cal = load_calibration(map_name)
    mask = MAPS_DIR / f"{map_name}_mask.png"
    lookup = _pos_lookup(ticks)

    out_dir = replay_cache_dir(demo_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    for n, (start, end) in enumerate(windows):
        frames = frames_from_ticks(ticks, start, end)
        attach_shots(frames, fires, start, end)
        data = {
            "round": n,
            "map_name": map_name,
            "map_transform": cal,
            "image": f"/static/maps/{map_name}.png",
            "mask": f"/static/maps/{map_name}_mask.png" if mask.exists() else None,
            "tick_rate": TICK_RATE,
            "fps": REPLAY_FPS,
            "start_tick": start,
            "end_tick": end,
            "frames": frames,
            "events": (kill_events(deaths, n)
                       + bomb_events(bombs, start, end, lookup)
                       + grenade_events(detonations, grenades, start, end)),
            "effect_tracks": effect_tracks(detonations, start, end, n),
            "effect_capabilities": {
                "inferno_cells": False,
                "smoke_voxels": False,
                "smoke_mode": "legacy_circle",
            },
        }
        with gzip.open(out_dir / f"round_{n}.json.gz", "wt", encoding="utf-8") as f:
            json.dump(data, f, separators=(",", ":"))
    (out_dir / "meta.json").write_text(json.dumps({
        "demo_id": demo_id, "map_name": map_name,
        "version": REPLAY_CACHE_VERSION, "rounds": len(windows),
    }), encoding="utf-8")


def get_round_replay(demo_path: str, demo_id: str, map_name: str, n: int) -> dict | None:
    """Cached round replay; builds the whole-match cache on first request.
    Returns None when the demo has no round n."""
    f = replay_cache_dir(demo_id) / f"round_{n}.json.gz"
    if not f.exists():
        if (replay_cache_dir(demo_id) / "meta.json").exists():
            return None  # cache built, round genuinely absent
        build_match_replay(demo_path, demo_id, map_name)
    if not f.exists():
        return None
    with gzip.open(f, "rt", encoding="utf-8") as fh:
        return json.load(fh)
