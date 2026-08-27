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
