# demo_coach/radar.py
import json
from pathlib import Path

import pandas as pd

MAPS_DIR = Path(__file__).parent / "web" / "static" / "maps"


def load_calibration(map_name: str) -> dict | None:
    """Radar calibration for a map: {pos_x, pos_y, scale} or None if unknown."""
    f = MAPS_DIR / "maps.json"
    if not f.exists():
        return None
    return json.loads(f.read_text(encoding="utf-8")).get(map_name)


def world_to_radar(x: float, y: float, cal: dict) -> tuple[float, float]:
    """World coordinates -> radar image pixels (1024x1024 source image)."""
    return (x - cal["pos_x"]) / cal["scale"], (cal["pos_y"] - y) / cal["scale"]


_KNIVES = {
    "knife", "karambit", "bayonet", "m9 bayonet", "butterfly knife",
    "flip knife", "gut knife", "falchion knife", "bowie knife",
    "huntsman knife", "shadow daggers", "paracord knife", "survival knife",
    "nomad knife", "skeleton knife", "stiletto knife", "talon knife",
    "ursus knife", "navaja knife", "classic knife", "kukri knife",
}
_WEAPON_SHORT = {
    "High Explosive Grenade": "HE",
    "Smoke Grenade": "Smoke",
    "Flashbang": "Flash",
    "Decoy Grenade": "Decoy",
    "Incendiary Grenade": "Incendiary",
    "C4 Explosive": "C4",
    "Defuse Kit": "Kit",
}


def short_weapon(name) -> str:
    """Normalize a display weapon name to its type, dropping skin variants."""
    if name is None or bool(pd.isna(name)):
        return ""
    n = str(name).strip()
    low = n.lower()
    if low in _KNIVES or "knife" in low or "bayonet" in low or "daggers" in low:
        return "Knife"
    return _WEAPON_SHORT.get(n, n)
