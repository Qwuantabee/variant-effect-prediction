"""RESULTS.md carries the bulky generated tables the README used to inline.

The reason these tables are generated is that a number typed by hand can drift
from the file it came from. Moving them to a second page must not weaken that:
each block on the page has to equal what its renderer produces from the committed
results, byte for byte, exactly like the replicated-epistasis block in the README.
"""

import importlib.util
import re
from pathlib import Path

import pytest

from conftest import require

ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / "RESULTS.md"
PAGE_NAMES = ("topk_ceiling", "latent_truth", "sim_controls", "epistasis",
              "family_viral_gap", "scaling_paired", "scaling_subgroups")


def _report_tables():
    spec = importlib.util.spec_from_file_location(
        "report_tables", ROOT / "scripts" / "report_tables.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _blocks(text):
    return {m.group(1): m.group(2) for m in re.finditer(
        r"<!-- BEGIN GENERATED: (\w+) -->\n(.*?)<!-- END GENERATED: \1 -->", text,
        re.S)}


def test_page_holds_exactly_the_declared_tables():
    require(PAGE.exists(), "RESULTS.md missing: run scripts/report_tables.py --write")
    mod = _report_tables()
    assert tuple(mod.PAGE_TABLES) == PAGE_NAMES
    assert set(_blocks(PAGE.read_text())) == set(PAGE_NAMES)


@pytest.mark.parametrize("name", PAGE_NAMES)
def test_page_block_equals_its_renderer(name):
    require(PAGE.exists(), "RESULTS.md missing: run scripts/report_tables.py --write")
    block = _blocks(PAGE.read_text())[name]
    fresh = _report_tables().render(name)
    assert block.strip() == fresh.strip(), (
        f"RESULTS.md block {name} differs from the renderer; "
        "run python scripts/report_tables.py --write")
