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
    df = ctx.scoreboard
    if df is None or df.empty:
        return _json([])
    return _json(df.astype(object).where(df.notna(), None).to_dict("records"))


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


def _get_rounds_log(ctx, args):
    df = ctx.round_log
    if df is None or df.empty:
        return _json([])
    return _json(df.astype(object).where(df.notna(), None).to_dict("records"))


_FUNCS = {"get_scoreboard": _get_scoreboard, "get_round": _get_round,
          "get_player": _get_player, "get_highlights": _get_highlights,
          "get_rounds_log": _get_rounds_log}

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
    {"type": "function", "function": {
        "name": "get_rounds_log", "description": "Whole match round log: winner, end reason, buy types for every round",
        "parameters": {"type": "object", "properties": {}}}},
]


def dispatch(ctx: MatchContext, name: str, args: dict) -> str:
    fn = _FUNCS.get(name)
    if fn is None:
        return _json({"error": f"unknown tool {name!r}"})
    try:
        return fn(ctx, args)
    except Exception as e:  # tool errors go back to the model as data
        return _json({"error": f"{type(e).__name__}: {e}"})
