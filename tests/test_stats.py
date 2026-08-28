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


def test_utility_stats():
    deaths = _deaths()
    deaths["assistedflash"] = [False, True, False]  # carol's assist was a flash
    hurts = _hurts()
    hurts["weapon"] = ["ak47", "hegrenade", "awp"]  # bob: 40 of his 120 was HE
    fires = pd.DataFrame(
        [["bob", "flashbang"], ["bob", "hegrenade"], ["bob", "smokegrenade"],
         ["alice", "ak47"]],
        columns=["user_name", "weapon"])
    u = stats.utility_stats(deaths, hurts, fires, n_rounds=2)
    assert u["bob"]["util_dmg"] == 40 and u["bob"]["util_dmg_r"] == 20.0
    assert u["bob"]["flashes_thrown"] == 1 and u["bob"]["nades_thrown"] == 3
    assert u["carol"]["flash_assists"] == 1
    assert u["alice"]["util_dmg"] == 0 and u["alice"]["nades_thrown"] == 0
    # scoreboard merges the same fields
    sb = stats.scoreboard(deaths, hurts, n_rounds=2, tick_rate=64, fires=fires)
    bob = sb[sb.name == "bob"].iloc[0]
    assert bob.util_dmg_r == 20.0 and bob.nades_thrown == 3


def test_utility_stats_missing_columns():
    # hurts without a weapon column, no fires -> all zeros, no crash
    u = stats.utility_stats(_deaths(), _hurts(), None, n_rounds=2)
    assert u["bob"]["util_dmg"] == 0 and u["bob"]["flash_assists"] == 0
    assert u["bob"]["flashes_thrown"] == 0 and u["bob"]["nades_thrown"] == 0
