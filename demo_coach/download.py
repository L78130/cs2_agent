# demo_coach/download.py
"""Download demos from 5EPlay / Perfect World Arena / Steam matchmaking.

Thin wrapper around the `cs-demo-downloader` package (MIT,
github.com/WangChuDi/CS-Demo-Downloader). Credentials are supplied per
request by the UI and optionally persisted in download_config.json
(gitignored — it contains access tokens).

HLTV is intentionally unsupported: hltv.org sits behind Cloudflare, so
programmatic demo scraping is unreliable; download pro demos manually.
"""
import bz2
import json
import os
import time
from pathlib import Path

import requests

# The `csgo` Steam-protobuf package (used to parse GC match lists) ships
# generated code that predates protobuf 4.x runtimes — force the pure-Python
# implementation before anything imports google.protobuf.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

CREDENTIALS_FILE = Path("download_config.json")  # gitignored: holds tokens

# credential fields per platform, in UI display order
PLATFORM_FIELDS = {
    "5e": ["userid"],
    "pwa": ["steamid", "access_token"],
    "steam": ["api_key", "steamid", "steamidkey", "knowncode"],
}


def load_credentials() -> dict:
    if CREDENTIALS_FILE.exists():
        return json.loads(CREDENTIALS_FILE.read_text(encoding="utf-8"))
    return {}


def save_credentials(creds: dict) -> None:
    CREDENTIALS_FILE.write_text(
        json.dumps(creds, ensure_ascii=False, indent=2), encoding="utf-8")


def _meta_to_dict(m) -> dict:
    teams = []
    for t in m.teams or []:
        teams.append({"name": getattr(t, "team_name", None) or getattr(t, "name", ""),
                      "score": getattr(t, "score", None)})
    date = (time.strftime("%Y-%m-%d %H:%M", time.localtime(m.started_at))
            if m.started_at else "")
    return {
        "platform": m.platform,
        "match_id": m.match_id,
        "map": m.map_name or "?",
        "date": date,
        "rounds": m.round_total,
        "teams": teams,
        "demo_available": m.demo_available if m.demo_available is not None else bool(m.demo_url),
        "demo_url": m.demo_url,  # 5e/pwa include it; may embed an access token
    }


def list_matches(platform: str, creds: dict, limit: int = 20) -> list[dict]:
    """Recent matches with demo availability for one platform."""
    if platform == "5e":
        from cs_demo_downloader.core.downloader_5e import get_all_demo_metadata
        metas = get_all_demo_metadata(creds["userid"], limit=limit)
    elif platform == "pwa":
        from cs_demo_downloader.core.downloader_pwa import get_all_demo_metadata
        metas = get_all_demo_metadata(creds["steamid"], creds["access_token"], size=limit)
    elif platform == "steam":
        from cs_demo_downloader.core.downloader_steam import get_all_demo_urls
        urls = get_all_demo_urls(creds["api_key"], creds["steamid"],
                                 creds["steamidkey"], creds["knowncode"],
                                 limit=limit, demo_url_resolver=_steam_resolver())
        return [{"platform": "steam", "match_id": code, "map": "?", "date": "",
                 "rounds": None, "teams": [], "demo_available": True,
                 "demo_url": url}
                for code, url in urls.items()]
    else:
        raise ValueError(f"unknown platform: {platform}")
    return [_meta_to_dict(m) for m in metas]


def _steam_resolver():
    """Share code -> replay URL via the Steam Game Coordinator.

    Uses akiver/boiler-writter (auto-downloaded once into cache/, git-ignored).
    Requires the Steam client to be running and logged in on this machine.
    """
    import zipfile

    from cs_demo_downloader.steam.boiler_resolver import (
        BoilerWritterResolver, download_boiler_writter)

    exe = download_boiler_writter(cache_dir="cache")
    # the library extracts only the exe, but boiler-writter also needs its
    # steam_api64.dll + steam_appid.txt siblings (they're in the zip's bin/)
    exe_dir = Path(exe).parent
    if not (exe_dir / "steam_appid.txt").exists():
        for z in exe_dir.glob("boiler-writter-*.zip"):
            with zipfile.ZipFile(z) as archive:
                for extra in ("bin/steam_api64.dll", "bin/steam_appid.txt"):
                    if extra in archive.namelist():
                        target = exe_dir / Path(extra).name
                        target.write_bytes(archive.read(extra))
    return BoilerWritterResolver(executable_path=exe).resolve_demo_url


def resolve_steam_share_code(share_code: str) -> str:
    """Share code -> demo download URL (raises if the replay expired/unavailable)."""
    from cs_demo_downloader.core.downloader_steam import resolve_demo_url_from_share_code
    url = resolve_demo_url_from_share_code(share_code, _steam_resolver())
    if not url:
        raise RuntimeError("could not resolve share code — replay may be expired, "
                           "or Steam client is not running/logged in")
    return url


def download_demo(url: str, dest_dir: Path, match_id: str,
                  headers: dict | None = None) -> Path:
    """Download one demo; Steam replays come as .dem.bz2 and are decompressed."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, headers=headers or {}, stream=True, timeout=120)
    if r.status_code == 404:
        raise RuntimeError(
            "replay file is gone from the server (HTTP 404) — Steam keeps "
            "matchmaking replays for roughly a month only; pick a recent match")
    r.raise_for_status()
    if url.split("?")[0].endswith(".bz2"):
        dest = dest_dir / f"{match_id}.dem"
        with dest.open("wb") as f:
            f.write(bz2.decompress(r.content))
    else:
        dest = dest_dir / f"{match_id}.dem"
        with dest.open("wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    return dest


def fetch(platform: str, creds: dict, match: dict, dest_dir: Path) -> Path:
    """Download a match selected from list_matches()."""
    url = match.get("demo_url")
    headers = None
    if platform == "pwa":
        from cs_demo_downloader.core.downloader_pwa import (
            build_download_headers, get_demo_url)
        headers = build_download_headers(creds["steamid"])
        if not url:
            url = get_demo_url(match["match_id"], creds["access_token"])
    elif platform == "5e":
        if not url:
            from cs_demo_downloader.core.downloader_5e import get_demo_url
            url = get_demo_url(match["match_id"])
    if not url:
        raise RuntimeError("no demo URL for this match (expired or unavailable)")
    return download_demo(url, dest_dir, match["match_id"], headers)
