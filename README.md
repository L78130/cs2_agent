# demo_coach

A CS2 demo-analysis agent. It parses a `.dem` file with
[demoparser2](https://github.com/LaihoE/demoparser2), computes a scoreboard,
per-round economy log, and highlight moments, then lets you chat about the
match with a Chinese-speaking LLM coach (Moonshot/Kimi API,
OpenAI-compatible) that can call tools over the parsed data.

## Setup

Requires Python 3.10+ (developed on 3.13). Dependencies are already installed in the project
virtualenv — always invoke Python through it:

```bash
./.venv/Scripts/python.exe --version
```

Set your Moonshot API key (required for the chat features, not for parsing):

```bash
# bash / Git Bash
export MOONSHOT_API_KEY=sk-...
```

```powershell
# PowerShell
$env:MOONSHOT_API_KEY="sk-..."
```

Both kimi.com coding-plan keys (`sk-kimi-...`) and Moonshot platform keys
(`sk-...`) work. The defaults target the kimi.com coding endpoint
(`https://api.kimi.com/coding/v1`, model `k3-256k`). For a Moonshot platform
key, override both:

```powershell
$env:KIMI_BASE_URL="https://api.moonshot.cn/v1"
$env:KIMI_MODEL="kimi-k2-0711-preview"
```

Optional environment variables:

- `KIMI_MODEL` — chat model (default `k3-256k`; the coding endpoint also
  offers `k3`, `kimi-for-coding`, `kimi-for-coding-highspeed`)
- `KIMI_BASE_URL` — API base URL (default `https://api.kimi.com/coding/v1`)
- `DEMO_COACH_CACHE_DIR` — where parsed demos are cached as parquet
  (default `cache/`, keyed by file hash)
- `DEMO_COACH_DEMO_DIR` — where the web UI stores uploaded demos
  (default `demos/`)

## CLI

```bash
export MOONSHOT_API_KEY=sk-...
./.venv/Scripts/python.exe -m demo_coach.agent path/to/demo.dem
```

The first parse of a demo can take a few minutes; results are cached, so
subsequent runs load instantly. Type questions about the match, `quit` to
exit.

## Web UI

Double-click **`start.bat`** (or pin it to the taskbar): it starts the server
if it isn't running and opens the page — one click, no terminal. Starting the
server manually with the command below also auto-opens the browser tab
(opt out with `DEMO_COACH_NO_BROWSER=1`).

```bash
./.venv/Scripts/python.exe -m uvicorn demo_coach.web.server:app
```

Open http://127.0.0.1:8000, upload a `.dem` file, and chat about it.
Features:

- **2D round replay** — animated top-down playback of any round (player
  positions, shots, kills, grenades, bomb) with game callout names; the first
  build of a demo parses all ticks once, afterwards each round loads from a
  per-round cache. Current round info (winner / end reason / buys) is shown
  under the replay window.
- **Player cards** — scoreboard rendered as one column per team; click a card
  for "Analyze performance" (LLM review of that player) or "Highlight rounds"
  (marks that player's multi-kill/clutch rounds on the round selector).
- **Streaming chat** — the coach's answer streams token by token (SSE);
  token usage per message and cumulative per demo is shown in the top-right
  corner of the header (locally estimated when the API omits usage, marked 约).

### Downloading demos from platforms

The sidebar's **⬇ Download demos** panel pulls your matches straight from
5EPlay, Perfect World Arena, or Steam matchmaking (via the MIT-licensed
[cs-demo-downloader](https://github.com/WangChuDi/CS-Demo-Downloader)).
Credentials you enter are stored only in the local, git-ignored
`download_config.json`:

- **5EPlay** — just your userid: the tail of your profile URL
  (`https://www.5eplay.com/player/<userid>`). No login needed.
- **Perfect World (PWA)** — your SteamID64 plus an `access_token`: log in at
  `partner.wmpvp.com`, then copy the token from the cookies (phone login) or
  from the `#/login?...&token=...` URL fragment (Steam login). Tokens expire;
  re-copy if downloads stop working.
- **Steam matchmaking** — either paste a single share code
  (`CSGO-XXXX-...`) for a one-off download, or fill in Steam Web API key +
  SteamID64 + match sharing auth key + one known share code to list recent
  official matches. The first listing walks your history match-by-match from
  the known code, so it can take minutes; afterwards a saved cursor makes it
  fast (requires the Steam client running and logged in). Expired replays
  (Steam keeps them ~1 month) can no longer be fetched.

HLTV is not supported: hltv.org sits behind Cloudflare, so programmatic
scraping breaks constantly — download pro-match demos manually instead.

## Tests

```bash
# unit tests (synthetic fixtures, no demo needed)
./.venv/Scripts/python.exe -m pytest -v

# integration test against a real demo (skipped unless TEST_DEMO is set)
TEST_DEMO="path/to/demo.dem" ./.venv/Scripts/python.exe -m pytest tests/test_integration.py -v
```

## Project layout

- `demo_coach/parsing.py` — demoparser2 wrapper; derives per-death round index
- `demo_coach/storage.py` — parquet cache of parsed demos
- `demo_coach/replay.py` — 2D round replay builder (32 Hz frames with rich
  player props, shots, kills/bomb/grenade events, legacy-circle smoke/inferno
  tracks), lazily parsed once per demo and cached as per-round gzip JSON;
  modeled on the replay module of DrEAmSs59/CS2-insight-agent
- `demo_coach/stats.py` — scoreboard: K/D/A, ADR, HS%, KAST, first kills,
  utility usage (utility damage/round, flash assists, flashes & grenades
  thrown from weapon_fire events)
- `demo_coach/economy.py` — buy classification (eco/force/full) and round log
- `demo_coach/highlights.py` — aces, 4k/3k, clutches, knife kills
- `demo_coach/summary.py` — JSON-serializable match summary
- `demo_coach/tools.py` — match context + tool definitions for the agent
- `demo_coach/agent.py` — tool-calling chat loop (streaming + grounding
  verification pass) and CLI entry point; Chinese system prompt with
  WHAT/WHY/DO structure, capability limits, callout names, and utility-usage
  evaluation (benchmarks ~5-10 util dmg/round, cf. Leetify/scope.gg)
- `demo_coach/web/server.py` — FastAPI backend (upload, tools, SSE chat stream)
- `demo_coach/web/static/index.html` — single-file vanilla-JS frontend
  (replay canvas, player cards, chat, token-usage readout)
