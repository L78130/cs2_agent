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
