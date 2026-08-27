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
