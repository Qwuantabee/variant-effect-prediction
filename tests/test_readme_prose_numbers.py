"""Hand-typed numbers in the README prose must equal what the committed CSVs give.

The generated tables are already guarded by `report_tables.py`; this covers the
numbers a person typed into sentences and into the baselines table. Each check
recomputes the value from a committed artefact under `results/` and asserts the
README contains it, rounded the way the README prints it. A figure that drifts,
or a sentence edited without its number, fails here.
"""
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text().replace("−", "-")
RES = ROOT / "results"


def _has(text: str) -> bool:
    return text in README


def _minus(x: float, nd: int) -> str:
    return f"{x:.{nd}f}"


@pytest.fixture(scope="module")
def nonviral_means():
    b = pd.read_csv(RES / "benchmark_esm2_t33_650M_UR50D.csv")
    nv = b[b.taxon != "Virus"]
    assert nv.assay.nunique() == 8
    return nv.groupby("method").spearman.mean(), b


@pytest.mark.parametrize("method", ["esm2", "blosum62", "hydropathy", "volume",
                                    "pro_gly", "charge", "random"])
def test_baseline_table_values(nonviral_means, method):
    means, _ = nonviral_means
    assert _has(_minus(means[method], 3)), (method, means[method])


def test_nine_assay_mean_over_all_nine(nonviral_means):
    _, b = nonviral_means
    assert _has(_minus(b[b.method == "esm2"].spearman.mean(), 3))


@pytest.fixture(scope="module")
def per_assay():
    return pd.read_csv(RES / "full" / "esm2_t33_650M_UR50D_per_assay.csv")


def test_top10_recall_means_over_217(per_assay):
    esm = per_assay[per_assay.method == "esm2"]
    rnd = per_assay[per_assay.method == "random"]
    blo = per_assay[per_assay.method == "blosum62"]
    assert esm.assay.nunique() == 217
    assert _has(_minus(esm.top10_recall.mean(), 3))
    assert _has(_minus(rnd.top10_recall.mean(), 3)) or _has(_minus(rnd.top10_recall.mean(), 2))
    assert _has(_minus(blo.top10_recall.mean(), 3))
    n_low = int((esm.top10_recall <= 0.2 + 1e-9).sum())
    assert _has(f"{n_low} of the 217") or _has(f"on {n_low} of them"), n_low


@pytest.fixture(scope="module")
def ceiling():
    return pd.read_csv(RES / "topk" / "cross_assay_ceiling.csv")


def test_replicate_medians(ceiling):
    assert len(ceiling) == 37
    for col, nd in [("top10_recall_mean", 3), ("spearman_whole", 3),
                    ("jaccard_top_decile", 3), ("noise_sd_z", 3)]:
        assert _has(_minus(ceiling[col].median(), nd)), (col, ceiling[col].median())


def test_top_decile_sd_is_assay_a_only(ceiling):
    """0.294 is assay A's median top-decile sd; assay B's is 0.235."""
    a, b = ceiling.top_decile_sd_a.median(), ceiling.top_decile_sd_b.median()
    assert _has(_minus(a, 3)), a
    assert round(a, 3) != round(b, 3)
    assert "assay A" in README or "first assay" in README, \
        "the README must say 0.294 is one assay's spread, not both"


def test_family_gap_numbers():
    g = pd.read_csv(RES / "families" / "per_method_viral_gap.csv")
    row = g[g.method == "ESM2 (650M)"].iloc[0]
    assert _has(_minus(row.nonviral_mean, 4)), row.nonviral_mean
    assert _has(_minus(-row.gap, 3)), row.gap
    assert _has(f"{round(100 * -row.gap / row.nonviral_mean)}%")
