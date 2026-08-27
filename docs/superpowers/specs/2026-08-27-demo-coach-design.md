# demo_coach — CS2 Demo Analysis Agent (Phase 1 Core) — Design Spec

Date: 2026-08-27
Status: Approved (design), pending spec review

## Purpose

A lightweight local agent that analyzes the user's own CS2 demo files (`.dem`) for:

1. **Personal coaching** — ADR/KAST, deaths, economy decisions, positioning-relevant stats, with LLM-generated feedback.
2. **Highlight detection** — rule-based detection of aces, 4k/3k rounds, 1vX clutches, knife kills.

Objective match stats (scoreboard, round-by-round, weapon stats) support both purposes.

## Architecture

Hybrid: a **deterministic stats engine** computes all numbers in Python; the **Kimi LLM** (Moonshot API) only interprets results and answers chat questions, with tool calling to query details on demand. The LLM never does arithmetic on raw tick data.

Rationale: LLMs hallucinate arithmetic over raw events; rules detect highlights reliably and cheaply; LLM adds value in coaching commentary and interactive Q&A.

## Components

Python package `demo_coach` in the repo root (`D:\gaming\demo_agent`), running in the existing `.venv` (Python 3.13, demoparser2 0.42.0 already installed).

```
demo_coach/
├── parsing.py      # demoparser2 wrappers: .dem → normalized DataFrames (events, ticks)
├── stats.py        # per-player: K/D/A, ADR, KAST, HS%, first kills; utility dmg, flash stats (deferred to a later phase)
├── economy.py      # per-round: buy type (eco/force/full), equipment value, round outcome+reason
├── highlights.py   # rule-based: 3k/4k/ace, 1vX clutches, knife kills
├── summary.py      # compact match-summary JSON (a few KB) seeded into Kimi's context
├── agent.py        # Kimi client (OpenAI-compatible), chat loop with tool calling
├── tools.py        # tools Kimi can call: get_round(n), get_player(name), get_highlights(...)
├── storage.py      # parse-once cache keyed by demo file hash (parquet + json)
└── web/
    ├── server.py   # FastAPI: upload .dem, list matches, POST /chat
    └── static/     # single simple page: upload, stats panels, chat box
```

## Data Flow

1. User uploads/drops a `.dem` file.
2. Parser runs once; results cached on disk keyed by the demo file's SHA-256 hash.
3. Stats/economy/highlights engines compute derived data deterministically.
4. A compact match-summary JSON goes into Kimi's system context.
5. User chats. When Kimi needs detail it calls a tool; the backend answers from cached data without re-parsing.

## LLM Integration

- Kimi API is OpenAI-compatible: base URL `https://api.moonshot.cn/v1`, official `openai` Python SDK.
- API key via `MOONSHOT_API_KEY` environment variable; never hardcoded or committed.
- Model name configurable (default: a Kimi K2-class model).
- Tool calling via standard `tools` / `tool_calls` schema.

## Error Handling

- Corrupt or outdated demo version → clear error at upload time.
- Missing game props in a demo → partial stats plus a warnings list; never a crash.
- Kimi API failure → one retry with backoff, then a plain error message in the chat UI.

## Testing

- Unit tests for stats/economy/highlights on small synthetic DataFrames (no demo file needed).
- One integration test requiring a real `.dem` supplied by the user.
- `pytest` as the test runner.

## Dependencies to Add

`fastapi`, `uvicorn`, `openai`, `python-multipart`, `pytest` — all installed into the existing venv.

## Implementation Order

1. `parsing.py` + `storage.py` (parse + cache)
2. `stats.py`
3. `economy.py`
4. `highlights.py`
5. `summary.py`
6. `agent.py` + `tools.py` (CLI chat first — cheap to test)
7. `web/server.py` + `web/static/` (FastAPI + single-page UI)

## Out of Scope (Later Phases)

- Phase 2: Steam match share-code demo download (share-code decode is documented and feasible).
- Phase 3: 5E / Perfect World platform demo downloads — requires a feasibility spike first; no public APIs, demos are behind login and platform-controlled. Manual `.dem` export from those platforms remains the fallback.
- Weapon stats, utility damage and flash stats beyond what scoreboard provides — deferred.

## Reference Projects

- [LaihoE/demoparser](https://github.com/LaihoE/demoparser) — parser (demoparser2)
- [pnxenopoulos/awpy](https://github.com/pnxenopoulos/awpy) — higher-level CS2 analytics
- [Twoos123/cs2-meta-engine](https://github.com/Twoos123/cs2-meta-engine) — demoparser2 + LLM analysis platform
- [DrEAmSs59/CS2-insight-agent](https://github.com/DrEAmSs59/CS2-insight-agent) — demo library + optional LLM commentary
- [Starfie1d1272/cs2-demo-analysis-kit](https://github.com/Starfie1d1272/cs2-demo-analysis-kit) — query-first analysis toolkit
