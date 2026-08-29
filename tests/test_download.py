# tests/test_download.py
"""Download module + endpoints, with all network/platform calls mocked."""
import bz2
import sys
from dataclasses import dataclass, field
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from demo_coach import download
from demo_coach.web import server


@dataclass
class FakeTeam:
    team_name: str = "Team A"
    score: int = 13


@dataclass
class FakeMeta:
    platform: str = "5e"
    match_id: str = "m1"
    demo_url: str = "http://x/m1.dem"
    demo_available: bool = True
    map_name: str = "de_mirage"
    map_label: str = None
    location: str = None
    match_winner: str = None
    season: int = None
    season_type: str = None
    year: int = None
    round_total: int = 24
    started_at: int = 1756000000
    ended_at: int = 1756003600
    teams: list = field(default_factory=lambda: [FakeTeam(), FakeTeam("Team B", 9)])
    players: list = field(default_factory=list)


def _fake_5e_module(metas):
    mod = ModuleType("cs_demo_downloader.core.downloader_5e")
    mod.get_all_demo_metadata = MagicMock(return_value=metas)
    mod.get_demo_url = MagicMock(return_value="http://x/m1.dem")
    return mod


def test_list_matches_5e_normalizes_metadata():
    fake = _fake_5e_module([FakeMeta()])
    with patch.dict(sys.modules, {"cs_demo_downloader.core.downloader_5e": fake}):
        out = download.list_matches("5e", {"userid": "u1"})
    assert out == [{
        "platform": "5e", "match_id": "m1", "map": "de_mirage",
        "date": out[0]["date"], "rounds": 24,
        "teams": [{"name": "Team A", "score": 13}, {"name": "Team B", "score": 9}],
        "demo_available": True, "demo_url": "http://x/m1.dem",
    }]
    assert out[0]["date"]  # formatted from started_at


def test_list_matches_unknown_platform():
    with pytest.raises(ValueError):
        download.list_matches("hltv", {})


def test_list_matches_steam_newest_first_and_ratchets_knowncode(tmp_path, monkeypatch):
    monkeypatch.setattr(download, "CREDENTIALS_FILE", tmp_path / "cfg.json")
    mod = ModuleType("cs_demo_downloader.core.downloader_steam")
    chain = {"CSGO-OLD": "CSGO-MID", "CSGO-MID": "CSGO-NEW", "CSGO-NEW": "n/a"}
    mod.get_next_share_code = MagicMock(side_effect=lambda *a: chain[a[3]])
    mod.resolve_demo_url_from_share_code = MagicMock(
        side_effect=lambda code, resolver: f"http://x/{code}.dem.bz2")
    creds = {"api_key": "k", "steamid": "s", "steamidkey": "sk",
             "knowncode": "CSGO-OLD"}
    with patch.dict(sys.modules,
                    {"cs_demo_downloader.core.downloader_steam": mod}), \
         patch.object(download, "_steam_resolver", return_value=object()):
        out = download.list_matches("steam", creds, limit=10)
    assert [m["match_id"] for m in out] == ["CSGO-NEW", "CSGO-MID"]
    assert out[0]["demo_url"] == "http://x/CSGO-NEW.dem.bz2"
    assert creds["knowncode"] == "CSGO-NEW"  # cursor ratcheted in place

    # once the cursor has caught up, the cached listing is still served
    mod.get_next_share_code = MagicMock(side_effect=lambda *a: "n/a")
    creds["knowncode"] = "CSGO-NEW"
    with patch.dict(sys.modules,
                    {"cs_demo_downloader.core.downloader_steam": mod}), \
         patch.object(download, "_steam_resolver", return_value=object()):
        out2 = download.list_matches("steam", creds, limit=10)
    assert [m["match_id"] for m in out2] == ["CSGO-NEW", "CSGO-MID"]


def test_download_demo_decompresses_bz2(tmp_path):
    payload = bz2.compress(b"fake demo bytes")
    resp = MagicMock()
    resp.content = payload
    resp.raise_for_status = lambda: None
    with patch("demo_coach.download.requests.get", return_value=resp):
        dest = download.download_demo("http://replay.valve.net/x.dem.bz2",
                                      tmp_path, "match42")
    assert dest.name == "match42.dem"
    assert dest.read_bytes() == b"fake demo bytes"


def test_credentials_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(download, "CREDENTIALS_FILE", tmp_path / "cfg.json")
    download.save_credentials({"5e": {"userid": "abc"}})
    assert download.load_credentials() == {"5e": {"userid": "abc"}}


def test_update_match_cache_fills_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(download, "CREDENTIALS_FILE", tmp_path / "cfg.json")
    download.save_credentials({"steam_match_cache": [
        {"match_id": "CSGO-A", "map": "?", "rounds": None}]})
    download.update_match_cache("steam", "CSGO-A", map="de_nuke", rounds=24)
    entry = download.load_credentials()["steam_match_cache"][0]
    assert entry["map"] == "de_nuke" and entry["rounds"] == 24
    download.update_match_cache("steam", "CSGO-MISSING", map="x")  # no-op
    assert len(download.load_credentials()["steam_match_cache"]) == 1


def test_peek_map_name_reads_bz2_header_only():
    header = (b"HL2DEMO\x00" + b"\x04\x00\x00\x00" * 2
              + b"server".ljust(260, b"\x00") + b"client".ljust(260, b"\x00")
              + b"de_mirage".ljust(260, b"\x00") + b"\x00" * 4096)
    resp = MagicMock()
    resp.iter_content = lambda n: iter([bz2.compress(header)])
    with patch("demo_coach.download.requests.get", return_value=resp):
        assert download.peek_map_name("http://x/m.dem.bz2") == "de_mirage"


def test_peek_map_name_cs2_header_uses_demoparser():
    resp = MagicMock()
    resp.iter_content = lambda n: iter([bz2.compress(b"PBDEMS2\x00" + b"\x00" * 4096)])
    fake_parser = MagicMock()
    fake_parser.parse_header.return_value = {"map_name": "de_nuke"}
    with patch("demo_coach.download.requests.get", return_value=resp), \
         patch("demoparser2.DemoParser", return_value=fake_parser):
        assert download.peek_map_name("http://x/m.dem.bz2") == "de_nuke"


def _client():
    return TestClient(server.app)


def test_download_list_endpoint_saves_creds(tmp_path, monkeypatch):
    monkeypatch.setattr(download, "CREDENTIALS_FILE", tmp_path / "cfg.json")
    fake = _fake_5e_module([FakeMeta()])
    with patch.dict(sys.modules, {"cs_demo_downloader.core.downloader_5e": fake}):
        r = _client().post("/api/download/list",
                           json={"platform": "5e", "creds": {"userid": "u1"}})
    assert r.status_code == 200
    assert r.json()[0]["match_id"] == "m1"
    assert download.load_credentials() == {"5e": {"userid": "u1"}}


def test_download_list_endpoint_bad_platform():
    r = _client().post("/api/download/list",
                       json={"platform": "nope", "creds": {}, "save": False})
    assert r.status_code == 502


def test_sharecode_endpoint_registers_demo(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "DEMOS_DIR", tmp_path)
    monkeypatch.setattr(download, "CREDENTIALS_FILE", tmp_path / "cfg.json")
    monkeypatch.setattr(download, "resolve_steam_share_code",
                        lambda code: "http://x/demo.dem.bz2")
    monkeypatch.setattr(download, "download_demo",
                        lambda url, d, mid, headers=None: tmp_path / "x.dem")
    ctx = MagicMock()
    ctx.parsed.demo_id = "demo123"
    ctx.summary = {"map": "de_mirage", "rounds_played": 24}
    monkeypatch.setattr(server, "build_context", lambda p: ctx)
    server.CONTEXTS.clear()
    r = _client().post("/api/download/sharecode",
                       json={"share_code": "CSGO-aaaa-bbbb-cccc-dddd-eeee"})
    assert r.status_code == 200
    assert r.json()["demo_id"] == "demo123"
    assert server.DEMO_PATHS["demo123"].endswith("x.dem")
    server.CONTEXTS.clear()
    server.DEMO_PATHS.clear()
