import pandas as pd
from demo_coach.parsing import ParsedDemo
from demo_coach import storage


def _fake_parsed(demo_id="abc123"):
    return ParsedDemo(
        demo_id=demo_id,
        header={"map_name": "de_mirage"},
        deaths=pd.DataFrame({"user_name": ["a"], "attacker_name": ["b"]}),
        hurts=pd.DataFrame({"user_name": ["b"], "dmg_health": [26]}),
        rounds=pd.DataFrame({"tick": [1000], "winner": [3], "reason": [9]}),
        economy=pd.DataFrame({"tick": [900], "name": ["a"], "team_num": [2],
                              "balance": [800], "current_equip_value": [150]}),
    )


def test_cache_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "CACHE_DIR", tmp_path)
    parsed = _fake_parsed()
    storage.save(parsed)
    loaded = storage.load(parsed.demo_id)
    assert loaded is not None
    assert loaded.header == parsed.header
    pd.testing.assert_frame_equal(loaded.deaths, parsed.deaths)
    pd.testing.assert_frame_equal(loaded.rounds, parsed.rounds)


def test_load_missing_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "CACHE_DIR", tmp_path)
    assert storage.load("nope") is None
