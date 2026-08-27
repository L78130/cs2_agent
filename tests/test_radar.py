# tests/test_radar.py
from demo_coach import radar

CAL = {"pos_x": -2953, "pos_y": 2164, "scale": 5}  # de_ancient


def test_world_to_radar_origin():
    # world point at (pos_x, pos_y) maps to pixel (0, 0) — upper left
    assert radar.world_to_radar(-2953, 2164, CAL) == (0.0, 0.0)


def test_world_to_radar_known_point():
    # real round-1 kill on de_ancient: victim (-431.87, 499.97)
    px, py = radar.world_to_radar(-431.872314, 499.970734, CAL)
    assert abs(px - 504.2) < 0.1
    assert abs(py - 332.8) < 0.1


def test_y_axis_flips():
    # moving north (+Y in world) moves up (-py) on the radar
    _, py_low = radar.world_to_radar(0, 0, CAL)
    _, py_high = radar.world_to_radar(0, 100, CAL)
    assert py_high < py_low


def test_load_calibration():
    assert radar.load_calibration("de_ancient")["scale"] == 5
    assert radar.load_calibration("de_nonexistent") is None
