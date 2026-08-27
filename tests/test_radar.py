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


def test_short_weapon_strips_knife_skins():
    assert radar.short_weapon("Huntsman Knife") == "Knife"
    assert radar.short_weapon("Karambit") == "Knife"
    assert radar.short_weapon("knife") == "Knife"


def test_short_weapon_grenades_and_guns():
    assert radar.short_weapon("High Explosive Grenade") == "HE"
    assert radar.short_weapon("Smoke Grenade") == "Smoke"
    assert radar.short_weapon("C4 Explosive") == "C4"
    assert radar.short_weapon("AK-47") == "AK-47"  # guns pass through
    assert radar.short_weapon(None) == ""
    assert radar.short_weapon(float("nan")) == ""  # missing weapon prop


def _ui_weapon_icon_map():
    import re
    html = (radar.MAPS_DIR.parent / "index.html").read_text(encoding="utf-8")
    block = re.search(r"WEAPON_ICON = \{(.*?)\};", html, re.S).group(1)
    keys = set(re.findall(r'"([^"]+)":', block))
    slugs = re.findall(r':\s*"([a-z0-9_]+)"', block)
    return keys, slugs


def test_all_ui_weapon_icons_exist_on_disk():
    _, slugs = _ui_weapon_icon_map()
    icons = radar.MAPS_DIR.parent / "icons"
    assert len(slugs) >= 40
    for slug in slugs:
        assert (icons / f"{slug}.svg").exists(), slug


def test_short_weapon_outputs_have_ui_icons():
    keys, _ = _ui_weapon_icon_map()
    # every weapon name observed in the real test demo
    names = ["AK-47", "AWP", "C4 Explosive", "Desert Eagle", "Dual Berettas",
             "FAMAS", "Five-SeveN", "Flashbang", "Galil AR", "Glock-18",
             "High Explosive Grenade", "Huntsman Knife", "Incendiary Grenade",
             "M4A1-S", "M4A4", "MAC-10", "MP7", "MP9", "Molotov", "Negev",
             "P250", "Paracord Knife", "SSG 08", "Smoke Grenade", "Tec-9",
             "USP-S", "XM1014", "knife", "knife_t"]
    for name in names:
        assert radar.short_weapon(name) in keys, name
