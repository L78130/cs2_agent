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
