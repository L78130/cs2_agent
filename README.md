# demo_coach

A CS2 demo-analysis agent. It parses a `.dem` file with
[demoparser2](https://github.com/LaihoE/demoparser2), computes a scoreboard,
per-round economy log, and highlight moments, then lets you chat about the
match with an LLM coach (Moonshot/Kimi API, OpenAI-compatible) that can call
tools over the parsed data.

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

```bash
./.venv/Scripts/python.exe -m uvicorn demo_coach.web.server:app
```

Open http://127.0.0.1:8000, upload a `.dem` file, and chat about it.

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
- `demo_coach/stats.py` — scoreboard: K/D/A, ADR, HS%, KAST, first kills
- `demo_coach/economy.py` — buy classification (eco/force/full) and round log
- `demo_coach/highlights.py` — aces, 4k/3k, clutches, knife kills
- `demo_coach/summary.py` — JSON-serializable match summary
- `demo_coach/tools.py` — match context + tool definitions for the agent
- `demo_coach/agent.py` — tool-calling chat loop and CLI entry point
- `demo_coach/web/server.py` — FastAPI web UI (upload + chat)
