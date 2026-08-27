# demo_coach/radar.py
import json
from pathlib import Path

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
