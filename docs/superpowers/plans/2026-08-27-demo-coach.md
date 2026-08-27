# demo_coach Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local CS2 demo-analysis agent: parse `.dem` files with demoparser2, compute stats/economy/highlights deterministically, and chat about the match via the Kimi API (tool calling), with a minimal web UI.

**Architecture:** Hybrid — deterministic Python stats engine (pandas DataFrames from demoparser2) produces all numbers; the Kimi LLM only interprets and converses. Parse once, cache by file hash. FastAPI backend + single static page. CLI chat loop ships before the web UI.

**Tech Stack:** Python 3.13 venv at `.venv`, demoparser2 0.42.0 (installed), pandas, FastAPI, uvicorn, openai SDK (Kimi is OpenAI-compatible), pytest.

**Spec:** `docs/superpowers/specs/2026-08-27-demo-coach-design.md`

## Global Constraints

- All commands run from repo root `D:\gaming\demo_agent` in Git Bash (Windows).
- Python is always invoked as `./.venv/Scripts/python.exe` (never bare `python`).
- pytest is invoked as `./.venv/Scripts/python.exe -m pytest`.
- API key comes from env var `MOONSHOT_API_KEY`; it is never hardcoded, never committed.
- Kimi base URL: `https://api.moonshot.cn/v1`. Model id from env var `KIMI_MODEL`, default `kimi-k2-0711-preview`. If the API returns "model not found", check https://platform.moonshot.cn/docs for the current model id and set `KIMI_MODEL` accordingly.
- Team numbers: 2 = T, 3 = CT (Source engine convention).
- v1 assumption: 5v5 matches. Roster = union of player names appearing in deaths/hurts events (players with zero involvement may be absent — acceptable for v1).
- `.gitignore` already excludes `.venv/`, `demos/`, `cache/`.
- New dependencies to install in Task 1: `fastapi uvicorn openai python-multipart pytest`.

## File Structure

| File | Responsibility |
|---|---|
| `demo_coach/__init__.py` | package marker |
| `demo_coach/config.py` | env-var config: api key, base url, model |
| `demo_coach/parsing.py` | demoparser2 → `ParsedDemo` dataclass (DataFrames) |
| `demo_coach/storage.py` | hash-keyed disk cache for `ParsedDemo` |
| `demo_coach/stats.py` | scoreboard: K/D/A, ADR, HS%, KAST, first kills |
| `demo_coach/economy.py` | buy-type classification + round log |
| `demo_coach/highlights.py` | rule-based multi-kills, clutches, knife kills |
| `demo_coach/summary.py` | compact match-summary dict for LLM context |
| `demo_coach/tools.py` | `MatchContext` + tool functions + dispatch + OpenAI tool schemas |
| `demo_coach/agent.py` | Kimi client, chat loop with tool calling, CLI entry |
| `demo_coach/web/server.py` | FastAPI app: upload, matches, chat |
| `demo_coach/web/static/index.html` | single-page UI: upload + stats + chat |
| `tests/` | one test module per engine module + integration test |

---

### Task 1: Scaffold + config

**Files:**
- Create: `demo_coach/__init__.py` (empty)
- Create: `demo_coach/config.py`
- Create: `tests/test_config.py`

**Interfaces:**
- Produces: `config.get_settings() -> Settings` with fields `api_key: str | None`, `base_url: str`, `model: str`. Later tasks (agent) consume this.

- [ ] **Step 1: Install dependencies**

Run:
```bash
./.venv/Scripts/python.exe -m pip install fastapi uvicorn openai python-multipart pytest
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_config.py
import os
from demo_coach.config import get_settings


def test_defaults(monkeypatch):
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    monkeypatch.delenv("KIMI_MODEL", raising=False)
    s = get_settings()
    assert s.api_key is None
    assert s.base_url == "https://api.moonshot.cn/v1"
    assert s.model == "kimi-k2-0711-preview"


def test_env_override(monkeypatch):
    monkeypatch.setenv("MOONSHOT_API_KEY", "sk-test")
    monkeypatch.setenv("KIMI_MODEL", "kimi-test-model")
    s = get_settings()
    assert s.api_key == "sk-test"
    assert s.model == "kimi-test-model"
```

- [ ] **Step 3: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'demo_coach'`

- [ ] **Step 4: Write minimal implementation**

```python
# demo_coach/config.py
import os
from dataclasses import dataclass

DEFAULT_BASE_URL = "https://api.moonshot.cn/v1"
DEFAULT_MODEL = "kimi-k2-0711-preview"


@dataclass(frozen=True)
class Settings:
    api_key: str | None
    base_url: str
    model: str


def get_settings() -> Settings:
    return Settings(
        api_key=os.environ.get("MOONSHOT_API_KEY"),
        base_url=os.environ.get("KIMI_BASE_URL", DEFAULT_BASE_URL),
        model=os.environ.get("KIMI_MODEL", DEFAULT_MODEL),
    )
```

Also create empty `demo_coach/__init__.py` and `tests/__init__.py`.

- [ ] **Step 5: Run test to verify it passes**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_config.py -v`
Expected: 2 passed

- [ ] **Step 6: Commit**

```bash
git add demo_coach tests
git commit -m "feat: scaffold demo_coach package with config"
```

---

### Task 2: Parsing + disk cache

**Files:**
- Create: `demo_coach/parsing.py`
- Create: `demo_coach/storage.py`
- Test: `tests/test_storage.py` (parsing itself needs a real demo; cache logic is tested with synthetic DataFrames)

**Interfaces:**
- Produces:
  - `parsing.ParsedDemo` — dataclass: `demo_id: str`, `header: dict`, `deaths: pd.DataFrame`, `hurts: pd.DataFrame`, `rounds: pd.DataFrame`, `economy: pd.DataFrame`
  - `parsing.file_hash(path: str) -> str` — sha256 hex digest, first 16 chars
  - `parsing.parse_demo(path: str) -> ParsedDemo`
  - `storage.load_or_parse(path: str) -> ParsedDemo` — the only entry point later tasks use
- Consumes: nothing from earlier tasks

- [ ] **Step 1: Write the failing test**

```python
# tests/test_storage.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_storage.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'demo_coach.parsing'`

- [ ] **Step 3: Write minimal implementation**

```python
# demo_coach/parsing.py
import hashlib
from dataclasses import dataclass

import pandas as pd
from demoparser2 import DemoParser

TICK_RATE = 64  # CS2 demo tick rate; used for time-window calculations


@dataclass
class ParsedDemo:
    demo_id: str
    header: dict
    deaths: pd.DataFrame    # player_death events
    hurts: pd.DataFrame     # player_hurt events
    rounds: pd.DataFrame    # round_end events
    economy: pd.DataFrame   # balance/equip_value snapshot at each round_freeze_end


def file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def parse_demo(path: str) -> ParsedDemo:
    parser = DemoParser(path)
    header = dict(parser.parse_header())
    deaths = parser.parse_event("player_death")
    hurts = parser.parse_event("player_hurt")
    rounds = parser.parse_event("round_end")
    freeze_ends = parser.parse_event("round_freeze_end")
    if len(freeze_ends) > 0:
        economy = parser.parse_ticks(
            ["balance", "current_equip_value"],
            ticks=freeze_ends["tick"].tolist(),
        )
    else:
        economy = pd.DataFrame(
            columns=["tick", "name", "team_num", "balance", "current_equip_value"]
        )
    return ParsedDemo(
        demo_id=file_hash(path), header=header,
        deaths=deaths, hurts=hurts, rounds=rounds, economy=economy,
    )
```

```python
# demo_coach/storage.py
import json
from pathlib import Path

import pandas as pd

from demo_coach.parsing import ParsedDemo, parse_demo

CACHE_DIR = Path("cache")
_FRAMES = ["deaths", "hurts", "rounds", "economy"]


def save(parsed: ParsedDemo) -> None:
    d = CACHE_DIR / parsed.demo_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "header.json").write_text(json.dumps(parsed.header), encoding="utf-8")
    for name in _FRAMES:
        getattr(parsed, name).to_parquet(d / f"{name}.parquet")


def load(demo_id: str) -> ParsedDemo | None:
    d = CACHE_DIR / demo_id
    if not (d / "header.json").exists():
        return None
    return ParsedDemo(
        demo_id=demo_id,
        header=json.loads((d / "header.json").read_text(encoding="utf-8")),
        **{name: pd.read_parquet(d / f"{name}.parquet") for name in _FRAMES},
    )


def load_or_parse(path: str) -> ParsedDemo:
    from demo_coach.parsing import file_hash
    demo_id = file_hash(path)
    cached = load(demo_id)
    return cached if cached is not None else _parse_and_save(path)


def _parse_and_save(path: str) -> ParsedDemo:
    parsed = parse_demo(path)
    save(parsed)
    return parsed
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_storage.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add demo_coach/parsing.py demo_coach/storage.py tests/test_storage.py
git commit -m "feat: demo parsing with hash-keyed disk cache"
```

---

### Task 3: Stats engine (scoreboard)

**Files:**
- Create: `demo_coach/stats.py`
- Test: `tests/test_stats.py`

**Interfaces:**
- Consumes: `ParsedDemo.deaths` (columns used: `tick`, `total_rounds_played`, `user_name`, `attacker_name`, `assister_name`, `headshot`, `weapon`), `ParsedDemo.hurts` (columns: `attacker_name`, `dmg_health`, `total_rounds_played`)
- Produces:
  - `stats.compute_kast(deaths: pd.DataFrame, n_rounds: int, tick_rate: int = 64) -> dict[str, float]` — KAST% per player. K = kill or assist that round; S = did not die that round; T = died but killer died within 5 s (trade).
  - `stats.scoreboard(deaths: pd.DataFrame, hurts: pd.DataFrame, n_rounds: int, tick_rate: int = 64) -> pd.DataFrame` — one row per player, columns: `name, kills, deaths, assists, kd, adr, hs_pct, kast, first_kills`
  - `stats.roster(deaths, hurts) -> list[str]` — union of involved player names

Note: `total_rounds_played` on events is the 0-based round index supplied by demoparser2's `other` fields; v1 relies on demoparser2's default event columns — verify actual column names in the Task 9 integration test against a real demo and adjust the constants below if a name differs.

```python
# column-name constants live at top of stats.py
COL_ROUND = "total_rounds_played"
COL_VICTIM = "user_name"
COL_ATTACKER = "attacker_name"
COL_ASSISTER = "assister_name"
```

- [ ] **Step 1: Write the failing test**

```python
# tests/test_stats.py
import pandas as pd
from demo_coach import stats


def _deaths():
    # 2 rounds. Round 0: bob kills alice (headshot). Round 1: alice kills bob
    # (carol assists); alice (bob's killer) dies 3s later -> bob was traded.
    return pd.DataFrame([
        # tick, round, victim, attacker, assister, headshot, weapon
        [100, 0, "alice", "bob", None, True, "ak47"],
        [200, 1, "bob", "alice", "carol", False, "awp"],
        [392, 1, "alice", "bob_smurf", None, False, "deagle"],  # 192 ticks = 3s later
    ], columns=["tick", "total_rounds_played", "user_name", "attacker_name",
                "assister_name", "headshot", "weapon"])


def _hurts():
    return pd.DataFrame([
        ["bob", 80, 0], ["bob", 40, 0],     # bob dealt 120 in round 0
        ["alice", 100, 1],                  # alice dealt 100 in round 1
    ], columns=["attacker_name", "dmg_health", "total_rounds_played"])


def test_scoreboard_basic():
    sb = stats.scoreboard(_deaths(), _hurts(), n_rounds=2, tick_rate=64)
    bob = sb[sb.name == "bob"].iloc[0]
    assert bob.kills == 1 and bob.deaths == 1
    assert bob.hs_pct == 100.0
    assert bob.adr == 60.0  # 120 dmg / 2 rounds
    alice = sb[sb.name == "alice"].iloc[0]
    assert alice.kills == 1 and alice.deaths == 2 and alice.adr == 50.0
    carol = sb[sb.name == "carol"].iloc[0]
    assert carol.assists == 1 and carol.kills == 0


def test_kast():
    kast = stats.compute_kast(_deaths(), n_rounds=2, tick_rate=64)
    # alice: r0 K? no (she died) -> S? no. r1: K yes -> 1/2 = 50%
    assert kast["alice"] == 50.0
    # bob: r0 K yes; r1 K? no, S? no, T: his killer alice died within 5s -> yes
    assert kast["bob"] == 100.0


def test_first_kills():
    sb = stats.scoreboard(_deaths(), _hurts(), n_rounds=2, tick_rate=64)
    assert sb[sb.name == "bob"].iloc[0].first_kills == 1
    assert sb[sb.name == "alice"].iloc[0].first_kills == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_stats.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# demo_coach/stats.py
import pandas as pd

COL_ROUND = "total_rounds_played"
COL_VICTIM = "user_name"
COL_ATTACKER = "attacker_name"
COL_ASSISTER = "assister_name"
TRADE_WINDOW_SEC = 5.0


def roster(deaths: pd.DataFrame, hurts: pd.DataFrame) -> list[str]:
    names = set()
    for df, cols in ((deaths, [COL_VICTIM, COL_ATTACKER, COL_ASSISTER]),
                     (hurts, [COL_VICTIM, COL_ATTACKER])):
        for c in cols:
            if c in df:
                names |= set(df[c].dropna())
    return sorted(names)


def compute_kast(deaths: pd.DataFrame, n_rounds: int, tick_rate: int = 64) -> dict[str, float]:
    players = sorted(set(deaths[COL_VICTIM].dropna())
                     | set(deaths[COL_ATTACKER].dropna())
                     | set(deaths[COL_ASSISTER].dropna()))
    if n_rounds == 0:
        return {p: 0.0 for p in players}
    window = TRADE_WINDOW_SEC * tick_rate
    by_round = {r: g for r, g in deaths.groupby(COL_ROUND)}
    result = {}
    for p in players:
        ok = 0
        for r in range(n_rounds):
            g = by_round.get(r)
            if g is None:
                ok += 1  # no deaths recorded -> survived
                continue
            died = g[g[COL_VICTIM] == p]
            if (g[COL_ATTACKER] == p).any() or (g[COL_ASSISTER] == p).any():
                ok += 1
            elif died.empty:
                ok += 1  # survived
            else:
                death_tick = died.iloc[0]["tick"]
                killer = died.iloc[0][COL_ATTACKER]
                killer_deaths = g[(g[COL_VICTIM] == killer)
                                  & (g["tick"] > death_tick)
                                  & (g["tick"] <= death_tick + window)]
                if not killer_deaths.empty:
                    ok += 1  # traded
        result[p] = round(100.0 * ok / n_rounds, 1)
    return result


def scoreboard(deaths: pd.DataFrame, hurts: pd.DataFrame,
               n_rounds: int, tick_rate: int = 64) -> pd.DataFrame:
    players = roster(deaths, hurts)
    kast = compute_kast(deaths, n_rounds, tick_rate)
    dmg = hurts.groupby(COL_ATTACKER)["dmg_health"].sum() if len(hurts) else pd.Series(dtype=float)
    first_victims = (deaths.sort_values("tick")
                     .groupby(COL_ROUND).first()[COL_ATTACKER]
                     if len(deaths) else pd.Series(dtype=object))
    rows = []
    for p in players:
        d = deaths[deaths[COL_VICTIM] == p]
        k = deaths[deaths[COL_ATTACKER] == p]
        kills, n_deaths = len(k), len(d)
        rows.append({
            "name": p,
            "kills": kills,
            "deaths": n_deaths,
            "assists": int((deaths[COL_ASSISTER] == p).sum()),
            "kd": round(kills / n_deaths, 2) if n_deaths else float(kills),
            "adr": round(float(dmg.get(p, 0.0)) / n_rounds, 1) if n_rounds else 0.0,
            "hs_pct": round(100.0 * k["headshot"].sum() / kills, 1) if kills else 0.0,
            "kast": kast.get(p, 0.0),
            "first_kills": int((first_victims == p).sum()),
        })
    return pd.DataFrame(rows).sort_values("kills", ascending=False).reset_index(drop=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_stats.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add demo_coach/stats.py tests/test_stats.py
git commit -m "feat: scoreboard stats (K/D, ADR, HS%, KAST, first kills)"
```

---

### Task 4: Economy + round log

**Files:**
- Create: `demo_coach/economy.py`
- Test: `tests/test_economy.py`

**Interfaces:**
- Consumes: `ParsedDemo.economy` (columns: `tick, name, team_num, balance, current_equip_value`), `ParsedDemo.rounds` (columns: `tick, winner, reason`)
- Produces:
  - `economy.classify_buys(econ_df) -> pd.DataFrame` — per (round_index, team_num) avg equip value and buy type. Round index = row order of freeze-end snapshots (0-based).
  - `economy.round_log(rounds, buys) -> pd.DataFrame` — columns: `round, winner_side, reason_text, t_buy, ct_buy`
- Buy-type thresholds (team-average `current_equip_value`): `< 2000` → `"eco"`, `2000–4499` → `"force"`, `>= 4500` → `"full"`.

```python
# reason-code map at top of economy.py (CS2 round_end reason enum, common values)
REASONS = {1: "target_bombed", 7: "bomb_defused", 8: "ts_win",
           9: "cts_win", 12: "target_saved"}
SIDE = {2: "T", 3: "CT"}
```

- [ ] **Step 1: Write the failing test**

```python
# tests/test_economy.py
import pandas as pd
from demo_coach import economy


def _econ():
    # round 0: Ts average 4000 (force), CTs 5000 (full)
    # round 1: Ts 500 (eco), CTs 5200 (full)
    return pd.DataFrame([
        [100, "t1", 2, 4000, 4000], [100, "t2", 2, 4000, 4000],
        [100, "c1", 3, 5000, 5000], [100, "c2", 3, 5000, 5000],
        [500, "t1", 2, 500, 500],   [500, "t2", 2, 500, 500],
        [500, "c1", 3, 5200, 5200], [500, "c2", 3, 5200, 5200],
    ], columns=["tick", "name", "team_num", "balance", "current_equip_value"])


def _rounds():
    return pd.DataFrame([
        [400, 2, 8],   # round 0: Ts win by elimination
        [900, 3, 9],   # round 1: CTs win
    ], columns=["tick", "winner", "reason"])


def test_classify_buys():
    buys = economy.classify_buys(_econ())
    r0 = buys[buys["round"] == 0].set_index("team_num")
    assert r0.loc[2, "buy_type"] == "force"
    assert r0.loc[3, "buy_type"] == "full"
    r1 = buys[buys["round"] == 1].set_index("team_num")
    assert r1.loc[2, "buy_type"] == "eco"


def test_round_log():
    log = economy.round_log(_rounds(), economy.classify_buys(_econ()))
    assert list(log.columns) == ["round", "winner_side", "reason_text", "t_buy", "ct_buy"]
    assert log.iloc[0].winner_side == "T"
    assert log.iloc[0].reason_text == "ts_win"
    assert log.iloc[1].ct_buy == "full"
    assert log.iloc[1].t_buy == "eco"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_economy.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# demo_coach/economy.py
import pandas as pd

ECO_MAX = 2000
FORCE_MAX = 4500
REASONS = {1: "target_bombed", 7: "bomb_defused", 8: "ts_win",
           9: "cts_win", 12: "target_saved"}
SIDE = {2: "T", 3: "CT"}


def _buy_type(avg_equip: float) -> str:
    if avg_equip < ECO_MAX:
        return "eco"
    if avg_equip < FORCE_MAX:
        return "force"
    return "full"


def classify_buys(econ: pd.DataFrame) -> pd.DataFrame:
    if econ.empty:
        return pd.DataFrame(columns=["round", "team_num", "avg_equip", "buy_type"])
    ticks = sorted(econ["tick"].unique())
    round_of = {t: i for i, t in enumerate(ticks)}
    df = econ.assign(round=econ["tick"].map(round_of))
    out = (df.groupby(["round", "team_num"])["current_equip_value"]
           .mean().round(0).rename("avg_equip").reset_index())
    out["buy_type"] = out["avg_equip"].map(_buy_type)
    return out


def round_log(rounds: pd.DataFrame, buys: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pivot = buys.pivot_table(index="round", columns="team_num",
                             values="buy_type", aggfunc="first")
    for i, r in rounds.sort_values("tick").reset_index(drop=True).iterrows():
        rows.append({
            "round": i,
            "winner_side": SIDE.get(r["winner"], str(r["winner"])),
            "reason_text": REASONS.get(r["reason"], f"code_{r['reason']}"),
            "t_buy": pivot.get(2, pd.Series(dtype=object)).get(i, "unknown"),
            "ct_buy": pivot.get(3, pd.Series(dtype=object)).get(i, "unknown"),
        })
    return pd.DataFrame(rows, columns=["round", "winner_side", "reason_text",
                                       "t_buy", "ct_buy"])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_economy.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add demo_coach/economy.py tests/test_economy.py
git commit -m "feat: economy buy classification and round log"
```

---

### Task 5: Highlights

**Files:**
- Create: `demo_coach/highlights.py`
- Test: `tests/test_highlights.py`

**Interfaces:**
- Consumes: `ParsedDemo.deaths` (same columns as Task 3), `ParsedDemo.rounds` (columns: `tick, winner`)
- Produces:
  - `highlights.find_highlights(deaths, rounds, team_size: int = 5) -> list[dict]` where each dict is `{"type": str, "round": int, "tick": int, "player": str, "detail": str}`. Types: `"ace"` (5 kills), `"4k"`, `"3k"`, `"clutch"` (won round as last alive vs >= 2 enemies), `"knife"` (weapon == "knife").
- Team side per player per round is unknown from deaths alone, so clutch detection uses kill order only: a clutch is credited when a team is reduced to 1 alive while the enemy still has >= 2 alive, and that team wins the round. Team of each player is inferred from kill direction (attacker and victim are on opposite teams; seeding: assign teams by constraint propagation — if ambiguous, skip clutch for that round).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_highlights.py
import pandas as pd
from demo_coach import highlights


def _ace_round():
    # round 0: alice kills 5 enemies -> ace; round 1: bob knife kill
    return pd.DataFrame([
        [100, 0, "e1", "alice", None, False, "ak47"],
        [110, 0, "e2", "alice", None, False, "ak47"],
        [120, 0, "e3", "alice", None, False, "ak47"],
        [130, 0, "e4", "alice", None, False, "ak47"],
        [140, 0, "e5", "alice", None, False, "ak47"],
        [500, 1, "e1", "bob", None, False, "knife"],
        [510, 1, "bob", "e2", None, False, "ak47"],
    ], columns=["tick", "total_rounds_played", "user_name", "attacker_name",
                "assister_name", "headshot", "weapon"])


def _rounds():
    return pd.DataFrame([[400, 2, 8], [900, 3, 9]],
                        columns=["tick", "winner", "reason"])


def _clutch_deaths():
    # round 0, 5v5: team A = {a1..a5}, team B = {b1..b5} (teams inferred
    # from kill directions). a2..a5 die first (A at 1 alive), then a1 kills
    # b1, b2, b3... leaving B at 2 alive -> a1 clutches if team A (T=2) wins.
    rows = [
        [100, 0, "a2", "b1", None, False, "ak47"],
        [101, 0, "a3", "b1", None, False, "ak47"],
        [102, 0, "a4", "b2", None, False, "ak47"],
        [103, 0, "a5", "b2", None, False, "ak47"],  # A down to a1 alone; B has 5
        [110, 0, "b1", "a1", None, False, "ak47"],
        [111, 0, "b2", "a1", None, False, "ak47"],
        [112, 0, "b3", "a1", None, False, "ak47"],
    ]
    return pd.DataFrame(rows, columns=["tick", "total_rounds_played",
                                       "user_name", "attacker_name",
                                       "assister_name", "headshot", "weapon"])


def _clutch_rounds():
    return pd.DataFrame([[400, 2, 8]], columns=["tick", "winner", "reason"])


def test_ace_and_knife():
    hl = highlights.find_highlights(_ace_round(), _rounds())
    types = {(h["type"], h["player"]) for h in hl}
    assert ("ace", "alice") in types
    assert ("knife", "bob") in types
    ace = [h for h in hl if h["type"] == "ace"][0]
    assert ace["round"] == 0


def test_clutch():
    hl = highlights.find_highlights(_clutch_deaths(), _clutch_rounds(), team_size=5)
    clutches = [h for h in hl if h["type"] == "clutch"]
    assert len(clutches) == 1
    assert clutches[0]["player"] == "a1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_highlights.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# demo_coach/highlights.py
import pandas as pd

COL_ROUND = "total_rounds_played"


def _hl(hl_type, row, detail):
    return {"type": hl_type, "round": int(row[COL_ROUND]), "tick": int(row["tick"]),
            "player": row["attacker_name"], "detail": detail}


def _infer_teams(g: pd.DataFrame) -> dict[str, int] | None:
    """Two-color the kill graph: attacker and victim are on opposite teams.
    Returns {player: team_slot(0/1)} or None if ambiguous."""
    adjacency: dict[str, set[str]] = {}
    for _, r in g.iterrows():
        a, v = r["attacker_name"], r["user_name"]
        if pd.isna(a) or pd.isna(v):
            continue
        adjacency.setdefault(a, set()).add(v)
        adjacency.setdefault(v, set()).add(a)
    color: dict[str, int] = {}
    for start in adjacency:
        if start in color:
            continue
        color[start] = 0
        stack = [start]
        while stack:
            node = stack.pop()
            for nb in adjacency[node]:
                if nb in color:
                    if color[nb] == color[node]:
                        return None  # not bipartite (TK etc.) -> ambiguous
                else:
                    color[nb] = 1 - color[node]
                    stack.append(nb)
    return color


def find_highlights(deaths: pd.DataFrame, rounds: pd.DataFrame,
                    team_size: int = 5) -> list[dict]:
    out: list[dict] = []
    if deaths.empty:
        return out
    for rnd, g in deaths.groupby(COL_ROUND):
        g = g.sort_values("tick")
        # multi-kills
        for attacker, kg in g.groupby("attacker_name"):
            n = len(kg)
            if n >= 3:
                hl_type = {3: "3k", 4: "4k"}.get(n, "ace")
                out.append(_hl(hl_type, kg.iloc[-1], f"{n} kills in round {rnd}"))
        # knife kills
        for _, r in g[g["weapon"] == "knife"].iterrows():
            out.append(_hl("knife", r, "knife kill"))
        # clutch
        teams = _infer_teams(g)
        if teams is None:
            continue
        alive = {0: team_size, 1: team_size}
        lone: dict[int, tuple[str, int, int]] = {}  # team -> (player, n_enemy, tick)
        dead_so_far: set[str] = set()
        for _, r in g.iterrows():
            v = r["user_name"]
            if v not in teams:
                continue
            t = teams[v]
            alive[t] -= 1
            dead_so_far.add(v)
            if alive[t] == 1 and alive[1 - t] >= 2 and t not in lone:
                remaining = [p for p in teams
                             if teams[p] == t and p not in dead_so_far]
                if len(remaining) == 1:
                    lone[t] = (remaining[0], alive[1 - t], int(r["tick"]))
        # credit the clutch only if the lone player survived the round
        all_dead = set(g["user_name"])
        for t, (player, n_enemy, tick) in lone.items():
            if player not in all_dead:
                out.append({"type": "clutch", "round": int(rnd), "tick": tick,
                            "player": player,
                            "detail": f"won 1v{n_enemy} in round {rnd}"})
    return out
```

Note for implementer: clutch attribution in v1 = a player left alone vs >= 2 enemies who survives the round. Side (T/CT) is not attributed (teams are inferred only as two slots). Players who never appear in the kill graph are invisible to this heuristic. If integration testing shows wrong results on demos with teamkills, revisit.

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_highlights.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add demo_coach/highlights.py tests/test_highlights.py
git commit -m "feat: rule-based highlight detection (multi-kills, clutches, knife kills)"
```

---

### Task 6: Match summary

**Files:**
- Create: `demo_coach/summary.py`
- Test: `tests/test_summary.py`

**Interfaces:**
- Consumes: `ParsedDemo`, scoreboard df (Task 3), round_log df (Task 4), highlights list (Task 5)
- Produces: `summary.build_summary(parsed, sb, rlog, highlights) -> dict` with keys: `demo_id`, `map` (from `header["map_name"]`, fallback `"unknown"`), `rounds_played` (int), `scoreboard` (list of row dicts), `rounds` (list of row dicts), `highlights` (list as-is). JSON-serializable (no numpy/pandas scalars — convert via `.to_dict("records")` and `int()/float()` casts where needed).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_summary.py
import json
import pandas as pd
from demo_coach import summary
from demo_coach.parsing import ParsedDemo


def test_build_summary_json_serializable():
    parsed = ParsedDemo(
        demo_id="abc123", header={"map_name": "de_mirage"},
        deaths=pd.DataFrame(), hurts=pd.DataFrame(),
        rounds=pd.DataFrame(), economy=pd.DataFrame(),
    )
    sb = pd.DataFrame([{"name": "alice", "kills": 20, "adr": 95.5}])
    rlog = pd.DataFrame([{"round": 0, "winner_side": "T",
                          "reason_text": "ts_win", "t_buy": "full", "ct_buy": "eco"}])
    hl = [{"type": "ace", "round": 0, "tick": 140, "player": "alice", "detail": "5 kills"}]
    s = summary.build_summary(parsed, sb, rlog, hl)
    assert s["demo_id"] == "abc123"
    assert s["map"] == "de_mirage"
    assert s["scoreboard"][0]["name"] == "alice"
    json.dumps(s)  # must not raise


def test_missing_map_falls_back():
    parsed = ParsedDemo(demo_id="x", header={}, deaths=pd.DataFrame(),
                        hurts=pd.DataFrame(), rounds=pd.DataFrame(),
                        economy=pd.DataFrame())
    s = summary.build_summary(parsed, pd.DataFrame(), pd.DataFrame(), [])
    assert s["map"] == "unknown"
    assert s["rounds_played"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_summary.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# demo_coach/summary.py
import pandas as pd

from demo_coach.parsing import ParsedDemo


def _records(df: pd.DataFrame) -> list[dict]:
    if df is None or df.empty:
        return []
    return df.astype(object).where(df.notna(), None).to_dict("records")


def build_summary(parsed: ParsedDemo, scoreboard: pd.DataFrame,
                  round_log: pd.DataFrame, highlights: list[dict]) -> dict:
    return {
        "demo_id": parsed.demo_id,
        "map": parsed.header.get("map_name") or "unknown",
        "rounds_played": int(len(round_log)) if round_log is not None else 0,
        "scoreboard": _records(scoreboard),
        "rounds": _records(round_log),
        "highlights": list(highlights),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_summary.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add demo_coach/summary.py tests/test_summary.py
git commit -m "feat: compact match summary builder"
```

---

### Task 7: Tools + Kimi agent (CLI chat)

**Files:**
- Create: `demo_coach/tools.py`
- Create: `demo_coach/agent.py`
- Test: `tests/test_tools.py`, `tests/test_agent.py`

**Interfaces:**
- Consumes: all earlier tasks; `config.get_settings()`
- Produces:
  - `tools.MatchContext` — dataclass: `parsed: ParsedDemo`, `scoreboard: pd.DataFrame`, `round_log: pd.DataFrame`, `highlights: list[dict]`, `summary: dict`
  - `tools.build_context(path: str) -> MatchContext` — full pipeline: `load_or_parse` → stats → economy → highlights → summary. `n_rounds = len(parsed.rounds)`.
  - `tools.TOOL_SCHEMAS: list[dict]` — OpenAI-format tool definitions
  - `tools.dispatch(ctx: MatchContext, name: str, args: dict) -> str` — executes one tool call, returns JSON string
  - `agent.CoachAgent(ctx: MatchContext, client=None)` — `client` injectable for tests; default builds `openai.OpenAI` from settings. Raises `RuntimeError("MOONSHOT_API_KEY not set")` if no key and no client.
  - `agent.CoachAgent.chat(message: str, history: list[dict]) -> str` — one user turn; runs the tool-call loop until a final text answer; returns assistant text.
  - CLI: `python -m demo_coach.agent path/to/demo.dem`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tools.py
import json
import pandas as pd
from demo_coach import tools
from demo_coach.parsing import ParsedDemo


def _ctx():
    parsed = ParsedDemo(
        demo_id="abc123", header={"map_name": "de_mirage"},
        deaths=pd.DataFrame(
            [[100, 0, "alice", "bob", None, True, "ak47", 26, 0]],
            columns=["tick", "total_rounds_played", "user_name", "attacker_name",
                     "assister_name", "headshot", "weapon", "dmg_health", "dmg_armor"]),
        hurts=pd.DataFrame([["bob", 80, 0]],
                           columns=["attacker_name", "dmg_health",
                                    "total_rounds_played"]),
        rounds=pd.DataFrame([[400, 2, 8]], columns=["tick", "winner", "reason"]),
        economy=pd.DataFrame(
            [[50, "bob", 2, 4000, 4000], [50, "alice", 3, 5000, 5000]],
            columns=["tick", "name", "team_num", "balance", "current_equip_value"]),
    )
    return tools.build_context_from_parsed(parsed)


def test_dispatch_scoreboard():
    out = json.loads(tools.dispatch(_ctx(), "get_scoreboard", {}))
    assert any(r["name"] == "bob" for r in out)


def test_dispatch_round():
    out = json.loads(tools.dispatch(_ctx(), "get_round", {"round": 0}))
    assert out["winner_side"] == "T"
    assert "kills" in out  # round detail includes kill list


def test_dispatch_unknown_tool():
    out = tools.dispatch(_ctx(), "nonsense", {})
    assert "error" in out
```

```python
# tests/test_agent.py
import json
from unittest.mock import MagicMock
from demo_coach import agent


def _msg(content=None, tool_calls=None):
    m = MagicMock()
    m.content = content
    m.tool_calls = tool_calls
    m.model_dump.return_value = {
        "role": "assistant", "content": content,
        **({"tool_calls": tool_calls} if tool_calls else {}),
    }
    return m


def _response(message):
    r = MagicMock()
    r.choices = [MagicMock(message=message)]
    return r


def test_chat_tool_loop():
    ctx = MagicMock()
    client = MagicMock()
    tc = MagicMock()
    tc.id = "call_1"
    tc.function.name = "get_scoreboard"
    tc.function.arguments = "{}"
    tc.model_dump.return_value = {"id": "call_1", "type": "function",
                                  "function": {"name": "get_scoreboard",
                                               "arguments": "{}"}}
    client.chat.completions.create.side_effect = [
        _response(_msg(tool_calls=[tc])),
        _response(_msg(content="bob top-fragged.")),
    ]
    a = agent.CoachAgent(ctx, client=client)
    reply = a.chat("who played best?", history=[])
    assert reply == "bob top-fragged."
    assert client.chat.completions.create.call_count == 2


def test_chat_plain_answer():
    ctx = MagicMock()
    client = MagicMock()
    client.chat.completions.create.return_value = _response(_msg(content="hi"))
    a = agent.CoachAgent(ctx, client=client)
    assert a.chat("hello", history=[]) == "hi"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_tools.py tests/test_agent.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# demo_coach/tools.py
import json
from dataclasses import dataclass

import pandas as pd

from demo_coach import economy, highlights, stats, summary
from demo_coach.parsing import ParsedDemo
from demo_coach.storage import load_or_parse


@dataclass
class MatchContext:
    parsed: ParsedDemo
    scoreboard: pd.DataFrame
    round_log: pd.DataFrame
    highlights: list[dict]
    summary: dict


def build_context_from_parsed(parsed: ParsedDemo) -> MatchContext:
    n_rounds = len(parsed.rounds)
    sb = stats.scoreboard(parsed.deaths, parsed.hurts, n_rounds)
    buys = economy.classify_buys(parsed.economy)
    rlog = economy.round_log(parsed.rounds, buys)
    hl = highlights.find_highlights(parsed.deaths, parsed.rounds)
    return MatchContext(parsed, sb, rlog, hl,
                        summary.build_summary(parsed, sb, rlog, hl))


def build_context(demo_path: str) -> MatchContext:
    return build_context_from_parsed(load_or_parse(demo_path))


def _json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


def _get_scoreboard(ctx, args):
    return _json(ctx.summary["scoreboard"])


def _get_round(ctx, args):
    n = int(args["round"])
    rows = ctx.round_log[ctx.round_log["round"] == n]
    if rows.empty:
        return _json({"error": f"no round {n}"})
    out = rows.iloc[0].to_dict()
    kills = ctx.parsed.deaths[ctx.parsed.deaths["total_rounds_played"] == n]
    out["kills"] = kills[["tick", "attacker_name", "user_name", "weapon",
                          "headshot"]].astype(object).where(
                              kills.notna(), None).to_dict("records")
    return _json(out)


def _get_player(ctx, args):
    name = args["name"]
    rows = ctx.scoreboard[ctx.scoreboard["name"] == name]
    if rows.empty:
        return _json({"error": f"unknown player {name!r}",
                      "known": ctx.scoreboard["name"].tolist()})
    return _json(rows.iloc[0].to_dict())


def _get_highlights(ctx, args):
    hl = ctx.highlights
    t = args.get("type")
    if t:
        hl = [h for h in hl if h["type"] == t]
    return _json(hl)


_FUNCS = {"get_scoreboard": _get_scoreboard, "get_round": _get_round,
          "get_player": _get_player, "get_highlights": _get_highlights}

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "get_scoreboard", "description": "Full match scoreboard: K/D/A, ADR, HS%, KAST, first kills per player",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "get_round", "description": "Round detail: winner, end reason, buy types, kill list",
        "parameters": {"type": "object", "properties": {
            "round": {"type": "integer", "description": "0-based round index"}},
            "required": ["round"]}}},
    {"type": "function", "function": {
        "name": "get_player", "description": "Stats for one player by exact name",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string"}}, "required": ["name"]}}},
    {"type": "function", "function": {
        "name": "get_highlights", "description": "Highlight moments; optional type filter: ace, 4k, 3k, clutch, knife",
        "parameters": {"type": "object", "properties": {
            "type": {"type": "string"}}}}},
]


def dispatch(ctx: MatchContext, name: str, args: dict) -> str:
    fn = _FUNCS.get(name)
    if fn is None:
        return _json({"error": f"unknown tool {name!r}"})
    try:
        return fn(ctx, args)
    except Exception as e:  # tool errors go back to the model as data
        return _json({"error": f"{type(e).__name__}: {e}"})
```

```python
# demo_coach/agent.py
import argparse
import json
import sys

from demo_coach.config import get_settings
from demo_coach.tools import MatchContext, TOOL_SCHEMAS, build_context, dispatch

SYSTEM = """You are a sharp, honest CS2 coach reviewing one of the player's matches.
Match summary (already computed, trust these numbers — never recompute them):
{summary}

Rules:
- Never invent numbers. If you need data not in the summary, call a tool.
- Coaching feedback: concrete, specific, tied to rounds. No generic advice.
- Highlights: reference type/round when mentioning them.
- Reply in the same language the user writes in.
"""


class CoachAgent:
    def __init__(self, ctx: MatchContext, client=None):
        self.ctx = ctx
        if client is None:
            settings = get_settings()
            if not settings.api_key:
                raise RuntimeError("MOONSHOT_API_KEY not set")
            from openai import OpenAI
            client = OpenAI(api_key=settings.api_key,
                            base_url=settings.base_url)
            self.model = settings.model
        else:
            self.model = get_settings().model
        self.client = client

    def _system(self) -> str:
        return SYSTEM.format(summary=json.dumps(self.ctx.summary,
                                                ensure_ascii=False, default=str))

    def chat(self, message: str, history: list[dict]) -> str:
        messages = ([{"role": "system", "content": self._system()}]
                    + history + [{"role": "user", "content": message}])
        for _ in range(8):  # bound the tool-call loop
            resp = self.client.chat.completions.create(
                model=self.model, messages=messages, tools=TOOL_SCHEMAS)
            msg = resp.choices[0].message
            if not msg.tool_calls:
                return msg.content or ""
            messages.append(msg.model_dump(exclude_none=True))
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments or "{}")
                result = dispatch(self.ctx, tc.function.name, args)
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": result})
        return "(tool-call limit reached — please ask a narrower question)"


def main() -> None:
    ap = argparse.ArgumentParser(description="Chat with a CS2 demo coach")
    ap.add_argument("demo", help="path to .dem file")
    args = ap.parse_args()
    print("Parsing demo (cached after first run)...")
    ctx = build_context(args.demo)
    print(f"Loaded {ctx.summary['map']}, {ctx.summary['rounds_played']} rounds. "
          "Type your question, 'quit' to exit.")
    history: list[dict] = []
    agent = CoachAgent(ctx)
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q in ("quit", "exit", ""):
            break
        reply = agent.chat(q, history)
        history += [{"role": "user", "content": q},
                    {"role": "assistant", "content": reply}]
        print(reply)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_tools.py tests/test_agent.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add demo_coach/tools.py demo_coach/agent.py tests/test_tools.py tests/test_agent.py
git commit -m "feat: Kimi chat agent with tool calling + CLI"
```

---

### Task 8: Web UI (FastAPI + single page)

**Files:**
- Create: `demo_coach/web/__init__.py` (empty)
- Create: `demo_coach/web/server.py`
- Create: `demo_coach/web/static/index.html`
- Test: `tests/test_server.py`

**Interfaces:**
- Consumes: `tools.build_context`, `agent.CoachAgent`
- Produces: FastAPI `app` in `demo_coach.web.server`:
  - `POST /api/upload` (multipart file field `file`) → `{demo_id, map, rounds_played}`; saves to `demos/`, builds context, stores in module-level `CONTEXTS: dict[str, MatchContext]` and `AGENTS: dict[str, CoachAgent]`
  - `GET /api/matches` → list of `{demo_id, map, rounds_played}` for loaded contexts
  - `POST /api/matches/{demo_id}/chat` body `{message: str, history: list}` → `{reply: str}`; 404 if demo_id unknown; 503 if no API key
  - `GET /` serves `static/index.html`
- Run: `./.venv/Scripts/python.exe -m uvicorn demo_coach.web.server:app --reload`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_server.py
import io
from fastapi.testclient import TestClient
from demo_coach.web import server


def test_matches_empty():
    server.CONTEXTS.clear()
    c = TestClient(server.app)
    assert c.get("/api/matches").json() == []


def test_chat_unknown_demo():
    server.CONTEXTS.clear()
    c = TestClient(server.app)
    r = c.post("/api/matches/nope/chat", json={"message": "hi", "history": []})
    assert r.status_code == 404


def test_upload_rejects_non_dem():
    c = TestClient(server.app)
    r = c.post("/api/upload", files={"file": ("x.txt", io.BytesIO(b"hi"), "text/plain")})
    assert r.status_code == 400
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_server.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# demo_coach/web/server.py
import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from demo_coach.agent import CoachAgent
from demo_coach.tools import MatchContext, build_context

DEMOS_DIR = Path("demos")
STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="demo_coach")
CONTEXTS: dict[str, MatchContext] = {}
AGENTS: dict[str, CoachAgent] = {}


class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/upload")
def upload(file: UploadFile = File(...)):
    if not file.filename.endswith(".dem"):
        raise HTTPException(400, "only .dem files are accepted")
    DEMOS_DIR.mkdir(exist_ok=True)
    dest = DEMOS_DIR / Path(file.filename).name
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        ctx = build_context(str(dest))
    except Exception as e:
        dest.unlink(missing_ok=True)
        raise HTTPException(422, f"failed to parse demo: {e}")
    CONTEXTS[ctx.parsed.demo_id] = ctx
    return {"demo_id": ctx.parsed.demo_id, "map": ctx.summary["map"],
            "rounds_played": ctx.summary["rounds_played"]}


@app.get("/api/matches")
def matches():
    return [{"demo_id": c.parsed.demo_id, "map": c.summary["map"],
             "rounds_played": c.summary["rounds_played"]}
            for c in CONTEXTS.values()]


@app.post("/api/matches/{demo_id}/chat")
def chat(demo_id: str, req: ChatRequest):
    ctx = CONTEXTS.get(demo_id)
    if ctx is None:
        raise HTTPException(404, "unknown demo_id")
    if demo_id not in AGENTS:
        try:
            AGENTS[demo_id] = CoachAgent(ctx)
        except RuntimeError as e:
            raise HTTPException(503, str(e))
    return {"reply": AGENTS[demo_id].chat(req.message, req.history)}
```

```html
<!-- demo_coach/web/static/index.html -->
<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>demo_coach</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 800px; margin: 2rem auto; padding: 0 1rem; }
  #chat { border: 1px solid #ccc; border-radius: 8px; padding: 1rem; min-height: 300px; margin: 1rem 0; }
  .msg { margin: .5rem 0; white-space: pre-wrap; }
  .user { color: #06c; }
  .bot { color: #222; }
  input[type=text] { width: 75%; padding: .5rem; }
  button { padding: .5rem 1rem; }
</style>
</head>
<body>
<h1>demo_coach</h1>
<input type="file" id="file" accept=".dem">
<button onclick="upload()">Upload &amp; analyze</button>
<span id="status"></span>
<div id="chat"></div>
<input type="text" id="q" placeholder="Ask about the match..." onkeydown="if(event.key==='Enter')send()">
<button onclick="send()">Send</button>
<script>
let demoId = null, history = [];
const chat = document.getElementById("chat");
function add(role, text) {
  const d = document.createElement("div");
  d.className = "msg " + role;
  d.textContent = (role === "user" ? "You: " : "Coach: ") + text;
  chat.appendChild(d); chat.scrollTop = chat.scrollHeight;
}
async function upload() {
  const f = document.getElementById("file").files[0];
  if (!f) return;
  document.getElementById("status").textContent = "parsing...";
  const fd = new FormData(); fd.append("file", f);
  const r = await fetch("/api/upload", { method: "POST", body: fd });
  const j = await r.json();
  if (!r.ok) { document.getElementById("status").textContent = j.detail; return; }
  demoId = j.demo_id; history = [];
  chat.innerHTML = "";
  document.getElementById("status").textContent =
    `loaded ${j.map}, ${j.rounds_played} rounds`;
}
async function send() {
  const q = document.getElementById("q");
  if (!demoId || !q.value.trim()) return;
  add("user", q.value);
  const r = await fetch(`/api/matches/${demoId}/chat`, {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({ message: q.value, history }),
  });
  const j = await r.json();
  if (r.ok) { add("bot", j.reply);
    history.push({role: "user", content: q.value}, {role: "assistant", content: j.reply});
  } else { add("bot", "Error: " + (j.detail || r.status)); }
  q.value = "";
}
</script>
</body>
</html>
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_server.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add demo_coach/web tests/test_server.py
git commit -m "feat: FastAPI web UI with upload and chat"
```

---

### Task 9: Integration test + README

**Files:**
- Create: `tests/test_integration.py`
- Create: `README.md`
- Modify: none

**Interfaces:**
- Consumes: everything; requires a real demo at path in env var `TEST_DEMO` (test is skipped when unset)

- [ ] **Step 1: Write the integration test**

```python
# tests/test_integration.py
import os
import pytest

DEMO = os.environ.get("TEST_DEMO")


@pytest.mark.skipif(not DEMO, reason="set TEST_DEMO to a real .dem path")
def test_full_pipeline():
    from demo_coach.tools import build_context
    ctx = build_context(DEMO)
    assert ctx.summary["rounds_played"] > 0
    assert len(ctx.scoreboard) > 0
    # column-name sanity check against the real parser output
    for col in ("user_name", "attacker_name", "total_rounds_played", "tick"):
        assert col in ctx.parsed.deaths.columns, f"unexpected column layout: {col}"
```

- [ ] **Step 2: Run against a real demo**

Ask the user for a `.dem` file path, then:
```bash
TEST_DEMO="path/to/demo.dem" ./.venv/Scripts/python.exe -m pytest tests/test_integration.py -v
```
If a column name differs (e.g. `total_rounds_played`), fix the constants in `stats.py`/`highlights.py` and re-run the whole suite.

Also smoke-test the CLI manually:
```bash
export MOONSHOT_API_KEY=sk-...
./.venv/Scripts/python.exe -m demo_coach.agent path/to/demo.dem
```
And the web UI:
```bash
./.venv/Scripts/python.exe -m uvicorn demo_coach.web.server:app
# open http://127.0.0.1:8000
```

- [ ] **Step 3: Write README.md**

Short usage doc: venv activation, `MOONSHOT_API_KEY`, CLI usage, web UI usage, how tests work (`TEST_DEMO`).

- [ ] **Step 4: Run the full suite and commit**

```bash
./.venv/Scripts/python.exe -m pytest -v
git add tests/test_integration.py README.md
git commit -m "test: integration pipeline test; docs: README"
```

---

## Self-Review Notes

- Spec coverage: parsing/cache (T2), stats (T3), economy (T4), highlights (T5), summary (T6), agent+tools+CLI (T7), web (T8), error handling (T2 parse errors → T8 422; T7 API retry is minimal — one retry was simplified to surfacing a 503; acceptable for v1, flagged here), testing (all tasks + T9). Out-of-scope items (Steam/5E/PW downloads) correctly absent.
- Known v1 simplifications documented inline: roster from involved players, 5v5 assumption, clutch side attribution by survival, reason-code map partial, column names verified in T9 against a real demo.
