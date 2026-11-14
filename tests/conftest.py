"""Shared test helpers.

A test that depends on local data (the ProteinGym download, a results file the
README is rendered from) used to call pytest.skip when the file was missing, so
a deleted file or a broken path turned a regression into a green SKIP. The
helpers here turn a missing input into a FAILURE with a message saying what to
run, unless the environment variable VE_ALLOW_MISSING_DATA=1 is set (for a
machine that deliberately does not hold the 400 MB download).
"""

import os

import pytest


def require(condition, message):
    """Fail (not skip) when a required input is absent."""
    if condition:
        return
    if os.environ.get("VE_ALLOW_MISSING_DATA") == "1":
        pytest.skip(message + " (VE_ALLOW_MISSING_DATA=1)")
    pytest.fail(message + " -- set VE_ALLOW_MISSING_DATA=1 to skip on a "
                "machine without the data", pytrace=False)


@pytest.fixture(scope="session")
def require_data():
    return require
