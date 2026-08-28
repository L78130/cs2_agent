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


# weapons whose damage counts as utility damage (internal weapon_fire names)
UTIL_DMG_WEAPONS = {"hegrenade", "molotov", "inferno", "incgrenade"}
# thrown-utility weapons (grenades pulled the pin on)
NADE_WEAPONS = {"flashbang", "smokegrenade", "hegrenade", "molotov",
                "incgrenade", "decoy"}


def utility_stats(deaths: pd.DataFrame, hurts: pd.DataFrame,
                  fires: pd.DataFrame | None, n_rounds: int) -> dict[str, dict]:
    """Per-player utility usage: damage dealt by HE/molotov, flash assists,
    flashes and total grenades thrown. Tolerates missing columns/frames."""
    players = set(roster(deaths, hurts))
    if fires is not None and len(fires) and "user_name" in fires:
        players |= set(fires["user_name"].dropna())
    out = {p: {"util_dmg": 0, "util_dmg_r": 0.0, "flash_assists": 0,
               "flashes_thrown": 0, "nades_thrown": 0} for p in sorted(players)}

    if len(hurts) and "weapon" in hurts:
        uh = hurts[hurts["weapon"].isin(UTIL_DMG_WEAPONS)]
        dmg = uh.groupby(COL_ATTACKER)["dmg_health"].sum()
        for p, v in dmg.items():
            if p in out:
                out[p]["util_dmg"] = int(v)
                out[p]["util_dmg_r"] = round(float(v) / n_rounds, 1) if n_rounds else 0.0

    if len(deaths) and "assistedflash" in deaths:
        fa = deaths[deaths["assistedflash"].fillna(False).astype(bool)]
        for p in players:
            out[p]["flash_assists"] = int((fa[COL_ASSISTER] == p).sum()) \
                if COL_ASSISTER in fa else 0

    if fires is not None and len(fires) and "weapon" in fires:
        fw = fires["weapon"].str.replace(r"^weapon_", "", regex=True)
        for p in players:
            w = fw[fires["user_name"] == p]
            out[p]["flashes_thrown"] = int((w == "flashbang").sum())
            out[p]["nades_thrown"] = int(w.isin(NADE_WEAPONS).sum())
    return out


def scoreboard(deaths: pd.DataFrame, hurts: pd.DataFrame,
               n_rounds: int, tick_rate: int = 64,
               fires: pd.DataFrame | None = None) -> pd.DataFrame:
    players = roster(deaths, hurts)
    kast = compute_kast(deaths, n_rounds, tick_rate)
    util = utility_stats(deaths, hurts, fires, n_rounds)
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
            **util.get(p, {"util_dmg": 0, "util_dmg_r": 0.0,
                           "flash_assists": 0, "flashes_thrown": 0,
                           "nades_thrown": 0}),
        })
    return pd.DataFrame(rows).sort_values("kills", ascending=False).reset_index(drop=True)
