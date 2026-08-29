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


def update_match_cache(platform: str, match_id: str, **fields) -> None:
    """Fill in cache fields (e.g. map/rounds) after a demo is downloaded."""
    cache_key = f"{platform}_match_cache"
    try:
        saved = load_credentials()
        for m in saved.get(cache_key, []):
            if m.get("match_id") == match_id:
                m.update(fields)
                save_credentials(saved)
                return
    except OSError:
        pass


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
        return _steam_matches(creds, limit)
    else:
        raise ValueError(f"unknown platform: {platform}")
    return [_meta_to_dict(m) for m in metas]


def _steam_resolver(meta: dict | None = None):
    """Share code -> replay URL via the Steam Game Coordinator.

    Uses akiver/boiler-writter (auto-downloaded once into cache/, git-ignored).
    Requires the Steam client to be running and logged in on this machine.

    If `meta` is given, the GC response's match time is captured per share
    code (meta[code]["matchtime"]) — the only usable metadata the CS2 GC
    match list still carries (map name is no longer included).
    """
    import zipfile

    from cs_demo_downloader.steam.boiler_resolver import (
        BoilerWritterResolver, download_boiler_writter)

    exe = None
    # avoid hitting the GitHub release API (rate-limited) when we already
    # have a cached binary from a previous run
    cached = sorted(Path("cache").glob("boiler-writter/*/boiler-writter.exe"))
    if cached:
        exe = str(cached[-1])
    if exe is None:
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
    resolver = BoilerWritterResolver(executable_path=exe)
    if meta is None:
        return resolver.resolve_demo_url

    current = {}  # calls are sequential; remember which code is being parsed

    def parser(path):
        from cs_demo_downloader.steam.boiler_resolver import (
            extract_demo_url_from_match_list)
        from csgo.protobufs import cstrike15_gcmessages_pb2 as pb2
        msg = pb2.CMsgGCCStrike15_v2_MatchList()
        with open(path, "rb") as f:
            msg.ParseFromString(f.read())
        ts = max((m.matchtime for m in msg.matches), default=0)
        meta[current["code"]] = {"matchtime": ts}
        return extract_demo_url_from_match_list(msg)

    resolver.match_list_parser = parser

    def resolve(share_code, decoded):
        current["code"] = share_code
        return resolver.resolve_demo_url(share_code, decoded)

    return resolve


# HL2DEMO (CS:GO) header: 8 magic + 4 demoprotocol + 4 networkprotocol +
# 260 servername + 260 clientname, then the 260-byte map name
_DEMO_MAP_OFFSET = 8 + 4 + 4 + 260 + 260


def peek_map_name(url: str, timeout: int = 20) -> str | None:
    """Map name from a remote demo's header, without downloading it.

    Streams just enough of the (optionally bz2-compressed) file to decode the
    demo header, then aborts the connection — under 1 MB instead of hundreds
    of MB. CS2 (PBDEMS2) headers are parsed with demoparser2 on a truncated
    temp file (its message framing changed across builds); legacy HL2DEMO
    headers are read at the fixed offset. Returns None on any failure.
    """
    try:
        r = requests.get(url, stream=True, timeout=timeout)
        r.raise_for_status()
        decomp = (bz2.BZ2Decompressor()
                  if url.split("?")[0].endswith(".bz2") else None)
        buf = b""
        for chunk in r.iter_content(64 * 1024):
            # bz2 decompressors buffer a full block (~0.5 MB in) before
            # emitting anything, so buf may stay empty for several chunks
            buf += decomp.decompress(chunk) if decomp else chunk
            if len(buf) >= 256 * 1024:
                break
        r.close()
    except Exception:
        return None
    if buf[:8] == b"PBDEMS2\x00":
        import tempfile
        from demoparser2 import DemoParser
        try:
            with tempfile.NamedTemporaryFile(suffix=".dem", delete=False) as f:
                f.write(buf)
                tmp = f.name
            try:
                return DemoParser(tmp).parse_header().get("map_name") or None
            finally:
                os.remove(tmp)
        except Exception:
            return None
    if buf[:8] == b"HL2DEMO\x00":
        raw = buf[_DEMO_MAP_OFFSET:_DEMO_MAP_OFFSET + 260].split(b"\x00")[0]
        return raw.decode("ascii", "replace") or None
    return None


def _steam_matches(creds: dict, limit: int) -> list[dict]:
    """Walk Steam share-code history forward from knowncode, newest first.

    The Web API only steps one match per request from a known code, and each
    match needs a GC resolve for its replay URL, so listing is slow when
    knowncode is old. Two mitigations: transient API timeouts are retried
    instead of ending the walk, and creds["knowncode"] is ratcheted to the
    newest code seen (in place — callers persisting creds keep the cursor).
    """
    from cs_demo_downloader.core.downloader_steam import (
        get_next_share_code, resolve_demo_url_from_share_code)

    meta = {}
    resolver = _steam_resolver(meta)
    out = []
    code = creds["knowncode"]
    seen = {code}
    for _ in range(limit):
        nxt = None
        for _attempt in range(3):
            nxt = get_next_share_code(creds["api_key"], creds["steamid"],
                                      creds["steamidkey"], code)
            if nxt:
                break
            time.sleep(2)
        if not nxt or nxt == "n/a" or nxt in seen:
            break
        seen.add(nxt)
        code = nxt
        url = resolve_demo_url_from_share_code(nxt, resolver)
        if url:
            ts = meta.get(nxt, {}).get("matchtime", 0)
            date = (time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))
                    if ts else "")
            out.append({"platform": "steam", "match_id": nxt, "map": "?",
                        "date": date, "rounds": None, "teams": [],
                        "demo_available": True, "demo_url": url})
    if code != creds.get("knowncode"):
        creds["knowncode"] = code
    out.reverse()
    # The ratchet means future walks only see NEW matches; keep a small cache
    # so "list recent matches" still shows recent history after the cursor
    # has caught up. CS2's GC listings no longer carry map names, so peek the
    # demo header (a few KB) for entries we haven't enriched yet.
    try:
        saved = load_credentials()
        fresh = {m["match_id"] for m in out}
        merged = (out + [m for m in saved.get("steam_match_cache", [])
                         if m.get("match_id") not in fresh])[:30]
        changed = merged != saved.get("steam_match_cache")
        for m in merged[:10]:
            if m.get("map") in (None, "", "?") and m.get("demo_url"):
                name = peek_map_name(m["demo_url"])
                if name:
                    m["map"] = name
                    changed = True
        if merged and changed:
            saved["steam_match_cache"] = merged
            save_credentials(saved)
        if merged:
            out = merged
    except OSError:
        pass
    return out


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
