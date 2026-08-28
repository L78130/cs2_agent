# tests/test_replay.py
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from demo_coach import replay
from demo_coach.parsing import ParsedDemo, TICK_RATE
from demo_coach.tools import MatchContext
from demo_coach.web import server


# ---------------------------------------------------------------- pure helpers

def _ticks_df():
    return pd.DataFrame(
        [[100, "atk", "7656", 2, 1.0, 2.0, 10.0, 90.0, True, 100, 100, True,
          4000, 4150, ["AK-47", "Glock-18"], "AK-47", False, False, 0.0],
         [104, "atk", "7656", 2, 3.0, 4.0, 10.0, 45.0, True, 80, 100, True,
          4000, 4150, ["AK-47", "Glock-18"], "AK-47", False, False, 0.0],
         [100, "vic", "7657", 3, 5.0, 6.0, 20.0, -90.0, True, 100, 0, False,
          600, 600, ["Knife", "USP-S"], "USP-S", True, False, 0.0],
         [104, "vic", "7657", 3, 5.5, 6.5, 20.0, -80.0, False, 0, 0, False,
          600, 600, ["Knife", "USP-S"], "USP-S", True, False, 1.5]],
        columns=["tick", "name", "steamid", "team_num", "X", "Y", "Z", "yaw",
                 "is_alive", "health", "armor", "has_helmet", "balance",
                 "current_equip_value", "inventory", "active_weapon_name",
                 "has_defuser", "has_c4", "flash_duration"])


def test_round_windows():
    # play starts at freeze end, runs until the next round starts; the last
    # round gets a short tail clamped to the demo end
    wins = replay.round_windows([90, 490], [100, 500], [300, 700], 900)
    assert wins == [(100, 489), (500, 892)]
    # no freeze/round_start events: fall back to 0 and round_end + tail
    wins = replay.round_windows([], [], [300], 10000)
    assert wins == [(0, 300 + int(replay.ROUND_TAIL_SEC * TICK_RATE))]


def test_frames_from_ticks():
    frames = replay.frames_from_ticks(_ticks_df(), 100, 104)
    assert [f["tick"] for f in frames] == [100, 104]
    assert frames[0]["time_sec"] == 0.0
    assert frames[1]["time_sec"] == round(4 / TICK_RATE, 3)
    atk, vic = frames[0]["players"]
    assert atk["name"] == "atk" and atk["team"] == "T" and atk["steamid64"] == "7656"
    assert atk["x"] == 1.0 and atk["yaw"] == 90.0 and atk["health"] == 100
    assert atk["inventory"] == ["AK-47", "Glock-18"] and atk["weapon"] == "AK-47"
    assert atk["money"] == 4000 and atk["equipment_value"] == 4150
    assert vic["team"] == "CT" and vic["has_defuser"] is True
    # window clipping
    assert replay.frames_from_ticks(_ticks_df(), 101, 103) == []


def test_attach_shots():
    # real demoparser2 column names for weapon_fire with player props
    fires = pd.DataFrame(
        [[103, "atk", 2.0, 3.0, 45.0, -5.0, "ak47"],
         [101, "atk", 1.0, 2.0, 90.0, 0.0, "smokegrenade"],   # not a bullet
         [999, "atk", 0.0, 0.0, 0.0, 0.0, "ak47"]],           # out of window
        columns=["tick", "user_name", "user_X", "user_Y",
                 "user_yaw", "user_pitch", "weapon"])
    frames = replay.frames_from_ticks(_ticks_df(), 100, 104)
    replay.attach_shots(frames, fires, 100, 104)
    assert len(frames[0]["shots"]) == 0
    assert len(frames[1]["shots"]) == 1   # tick 103 -> nearest frame (tie -> earlier)
    s = frames[1]["shots"][0]
    assert s["actor"] == "atk" and s["weapon"] == "ak47"
    assert s["x"] == 2.0 and s["yaw"] == 45.0 and s["pitch"] == -5.0


def test_kill_events():
    deaths = pd.DataFrame(
        [[150, 0, "vic", "atk", None, True, "ak47", 10.0, 20.0, 30.0, 40.0],
         [600, 1, "atk", "vic", None, False, "awp", 1.0, 2.0, 3.0, 4.0]],
        columns=["tick", "total_rounds_played", "user_name", "attacker_name",
                 "assister_name", "headshot", "weapon",
                 "user_X", "user_Y", "attacker_X", "attacker_Y"])
    ev = replay.kill_events(deaths, 0)
    assert ev == [{"type": "kill", "tick": 150, "actor": "atk", "target": "vic",
                   "weapon": "ak47", "headshot": True, "assister": None,
                   "actor_x": 30.0, "actor_y": 40.0,
                   "target_x": 10.0, "target_y": 20.0}]


def test_bomb_events():
    bombs = pd.DataFrame(
        [[102, "bomb_dropped", "atk", None], [103, "bomb_planted", "atk", "A"],
         [999, "bomb_exploded", None, "A"]],
        columns=["tick", "event", "user_name", "site"])
    lookup = replay._pos_lookup(_ticks_df())
    ev = replay.bomb_events(bombs, 100, 104, lookup)
    assert ev == [
        {"type": "bomb_drop", "tick": 102, "actor": "atk", "site": None, "x": 1.0, "y": 2.0},
        {"type": "plant", "tick": 103, "actor": "atk", "site": "A", "x": 1.0, "y": 2.0},
    ]


def test_grenade_events_and_trajectory():
    det = {"smokegrenade_detonate": pd.DataFrame(
        [[200, "p1", 50.0, 60.0, 0.0]],
        columns=["tick", "user_name", "x", "y", "z"])}
    grenades = pd.DataFrame(
        [[50, "smokegrenade", "p1", 1.0, 1.0, 0.0],      # old throw: tick gap cuts it
         [180, "smokegrenade", "p1", 10.0, 20.0, 30.0],
         [190, "smokegrenade", "p1", 20.0, 30.0, 40.0],
         [195, "smokegrenade", "p1", 30.0, 40.0, 50.0],
         [185, "hegrenade", "p1", 99.0, 99.0, 99.0]],    # different kind
        columns=["tick", "grenade_type", "name", "x", "y", "z"])
    ev = replay.grenade_events(det, grenades, 0, 300)
    assert len(ev) == 1
    g = ev[0]
    assert g["type"] == "grenade" and g["kind"] == "smoke" and g["tick"] == 200
    assert g["actor"] == "p1" and g["x"] == 50.0
    assert g["throw_tick"] == 180
    assert [p["tick"] for p in g["trajectory"]] == [180, 190, 195]


def test_grenade_trajectory_dropped_when_landing_too_far():
    det = {"smokegrenade_detonate": pd.DataFrame(
        [[1000, "p1", 50.0, 60.0, 0.0]], columns=["tick", "user_name", "x", "y", "z"])}
    grenades = pd.DataFrame(   # last seen >2s before the detonation
        [[800, "smokegrenade", "p1", 10.0, 20.0, 30.0]],
        columns=["tick", "grenade_type", "name", "x", "y", "z"])
    ev = replay.grenade_events(det, grenades, 0, 2000)
    assert ev[0]["trajectory"] == [] and ev[0]["throw_tick"] is None


def test_effect_tracks_legacy_circles():
    det = {
        "smokegrenade_detonate": pd.DataFrame(
            [[200, "p1", 50.0, 60.0, 0.0]], columns=["tick", "user_name", "x", "y", "z"]),
        "inferno_startburn": pd.DataFrame(
            [[11, 300, "p2", 70.0, 80.0, 0.0]],
            columns=["entityid", "tick", "user_name", "x", "y", "z"]),
        "inferno_expire": pd.DataFrame(
            [[11, 620]], columns=["entityid", "tick"]),
    }
    tracks = replay.effect_tracks(det, 0, 700, 0)
    assert [t["type"] for t in tracks] == ["smoke", "inferno"]
    smoke, inferno = tracks
    assert smoke["start_tick"] == 200
    assert smoke["end_tick"] == 200 + int(18 * TICK_RATE)
    assert smoke["radius"] == replay.SMOKE_RADIUS
    assert inferno["end_tick"] == 620   # matched inferno_expire by entityid
    # outside the window -> dropped
    assert replay.effect_tracks(det, 630, 700, 0) == []


def test_effect_tracks_molotov_fallback():
    # demos without inferno_startburn fall back to molotov/incendiary events
    det = {"molotov_detonate": pd.DataFrame(
        [[300, "p2", 70.0, 80.0, 0.0]], columns=["tick", "user_name", "x", "y", "z"])}
    tracks = replay.effect_tracks(det, 0, 700, 0)
    assert len(tracks) == 1 and tracks[0]["type"] == "inferno"
    assert tracks[0]["end_tick"] == 300 + int(7 * TICK_RATE)


def test_round_replay_cache_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(replay.storage, "CACHE_DIR", tmp_path)
    payload = {"round": 0, "frames": [], "events": []}
    built = {}

    def fake_build(demo_path, demo_id, map_name):
        d = replay.replay_cache_dir(demo_id)
        d.mkdir(parents=True, exist_ok=True)
        import gzip, json
        with gzip.open(d / "round_0.json.gz", "wt", encoding="utf-8") as f:
            json.dump(payload, f)
        (d / "meta.json").write_text("{}")
        built["called"] = True

    monkeypatch.setattr(replay, "build_match_replay", fake_build)
    assert replay.get_round_replay("/x.dem", "d1", "de_ancient", 0) == payload
    assert built.get("called")
    built.clear()
    assert replay.get_round_replay("/x.dem", "d1", "de_ancient", 0) == payload
    assert not built  # second read is a disk hit, no rebuild
    assert replay.get_round_replay("/x.dem", "d1", "de_ancient", 7) is None


# ---------------------------------------------------------------- endpoint

@pytest.fixture()
def ctx():
    parsed = ParsedDemo(
        demo_id="replaytest", header={"map_name": "de_ancient"},
        deaths=pd.DataFrame(), hurts=pd.DataFrame(),
        rounds=pd.DataFrame([[300, 2, 8]], columns=["tick", "winner", "reason"]),
        economy=pd.DataFrame(),
    )
    return MatchContext(parsed, pd.DataFrame(), pd.DataFrame(), [], {})


@pytest.fixture(autouse=True)
def clean_state(ctx, tmp_path):
    demo = tmp_path / "demo.dem"
    demo.touch()
    server.CONTEXTS.clear()
    server.DEMO_PATHS.clear()
    server.CONTEXTS["replaytest"] = ctx
    server.DEMO_PATHS["replaytest"] = str(demo)
    yield
    server.CONTEXTS.clear()
    server.DEMO_PATHS.clear()


def test_replay_endpoint(ctx, monkeypatch):
    payload = {"round": 0, "map_name": "de_ancient", "frames": [{"tick": 100}],
               "events": [], "effect_tracks": []}
    monkeypatch.setattr(replay, "get_round_replay",
                        lambda path, demo_id, map_name, n: payload if n == 0 else None)
    c = TestClient(server.app)
    r = c.get("/api/matches/replaytest/rounds/0/replay")
    assert r.status_code == 200
    assert r.json() == payload
    assert c.get("/api/matches/replaytest/rounds/9/replay").status_code == 404


def test_replay_unknown_demo():
    c = TestClient(server.app)
    assert c.get("/api/matches/nope/rounds/0/replay").status_code == 404


def test_replay_unknown_map(ctx):
    ctx.parsed.header["map_name"] = "de_nowhere"
    c = TestClient(server.app)
    assert c.get("/api/matches/replaytest/rounds/0/replay").status_code == 404


def test_replay_missing_demo_file(ctx):
    server.DEMO_PATHS["replaytest"] = "C:/nonexistent/demo.dem"
    c = TestClient(server.app)
    r = c.get("/api/matches/replaytest/rounds/0/replay")
    assert r.status_code == 409
