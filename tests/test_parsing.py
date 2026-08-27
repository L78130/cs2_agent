# tests/test_parsing.py
import warnings

import pytest

from demo_coach.parsing import _check_round_alignment


def test_round_alignment_warns_on_mismatch():
    with pytest.warns(UserWarning, match="round count mismatch"):
        _check_round_alignment(24, 23)


def test_round_alignment_silent_on_match():
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning becomes an error
        _check_round_alignment(24, 24)
