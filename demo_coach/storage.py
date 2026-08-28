import json
import os
from pathlib import Path

import pandas as pd

from demo_coach.parsing import ParsedDemo, parse_demo

CACHE_DIR = Path(os.environ.get("DEMO_COACH_CACHE_DIR", "cache"))
CACHE_VERSION = 7  # bump when parsed frame schemas change (v7: deaths carry place names)
_FRAMES = ["deaths", "hurts", "rounds", "economy", "positions", "fires", "bombs"]


def save(parsed: ParsedDemo) -> None:
    d = CACHE_DIR / f"v{CACHE_VERSION}" / parsed.demo_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "header.json").write_text(json.dumps(parsed.header), encoding="utf-8")
    for name in _FRAMES:
        getattr(parsed, name).to_parquet(d / f"{name}.parquet")


def load(demo_id: str) -> ParsedDemo | None:
    d = CACHE_DIR / f"v{CACHE_VERSION}" / demo_id
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
