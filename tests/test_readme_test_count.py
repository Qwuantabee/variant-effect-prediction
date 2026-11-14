"""The README's test counts must equal the tests that actually exist.

Two numbers in this README are hand-typed: the total in "Layout" and the
`pytest -m "not slow"` count in "Setup". Both went stale during the figure and
review passes, and a reader uses them to judge whether the suite is
substantial and how much of it runs without a model download.

Counting `def test_` is not enough because parametrised tests expand, so this
asks pytest. Two subprocess collections, read-only.
"""
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text()


def _collect(args) -> int:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-q", "--collect-only", *args],
        cwd=ROOT, capture_output=True, text=True, timeout=900)
    out = proc.stdout + proc.stderr
    m = re.search(r"(\d+)(?:/(\d+))? tests? collected", out)
    if not m:
        pytest.fail(f"could not read a collected count from pytest:\n{out[-2000:]}")
    # With -m deselection, pytest prints "251/274 tests collected".
    return int(m.group(1))


def test_readme_total_test_count_matches_the_collected_count():
    """`tests/   N tests` in the Layout section must be the real N."""
    m = re.search(r"^tests/\s+(\d+)\s+tests", README, re.M)
    assert m, "the Layout section no longer states a test count in the form " \
              "'tests/   N tests'; update this test if the wording moved"
    stated = int(m.group(1))
    actual = _collect([])
    assert stated == actual, (
        f"README says {stated} tests, pytest collects {actual}. Update the "
        f"README count; this guard exists so it cannot drift unnoticed again.")


def test_readme_not_slow_count_matches_the_collected_count():
    """`pytest -m "not slow"   # NNN tests, no model needed` must be real.

    This is the number a reader on a machine without a GPU relies on, so it is
    the one that matters most when it is wrong.
    """
    m = re.search(r'pytest -m "not slow"\s*#\s*(\d+) tests', README)
    assert m, 'the Setup section no longer states the not-slow count in the ' \
              'form \'pytest -m "not slow"   # NNN tests\'; update this test ' \
              'if the wording moved'
    stated = int(m.group(1))
    actual = _collect(["-m", "not slow"])
    assert stated == actual, (
        f"README says {stated} not-slow tests, pytest collects {actual} with "
        f"that marker. Update the README count.")
