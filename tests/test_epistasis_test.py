"""Tests for the within-assay singles-vs-multi epistasis script.

The script had no tests. The review (C2, C7, C8, V6) found a variant-level
bootstrap on position-clustered data, a column named different_quantity_flag
that held a composition string, a 'variance share of the epistatic term' that
exceeded 1, and a position-matched control that matched nothing for four of
five assays. Each is pinned here on a small synthetic assay.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import run_epistasis_test as et  # noqa: E402

AA = "ACDEFGHIKLMNPQRSTVWY"


# ----------------------------------------------------------------------
# additive_residual: the non-additive residual is not the epistatic term
# ----------------------------------------------------------------------

def _frame(singles, doubles):
    rows = [(m, v, 1) for m, v in singles] + [(m, v, 2) for m, v in doubles]
    return pd.DataFrame(rows, columns=["mutant", "DMS_score", "n_sub"])


def _pairs(n_pos):
    return [(p, q) for p in range(1, n_pos + 1) for q in range(p + 1, n_pos + 1)]


def test_residual_share_is_nonzero_with_no_epistasis_when_singles_are_noisy():
    """Planted: the TRUE doubles are exactly additive (zero epistasis) but the
    singles are measured with noise. The 'epistasis' variance share is still
    well above zero, so the share cannot be read as an epistasis share."""
    rng = np.random.default_rng(0)
    n_pos = 40
    true = rng.normal(0, 1, n_pos)
    singles = [(f"A{p}V", true[p - 1] + rng.normal(0, 0.8))
               for p in range(1, n_pos + 1)]
    doubles = [(f"A{p}V:A{q}V", true[p - 1] + true[q - 1] + rng.normal(0, 0.05))
               for p, q in _pairs(n_pos)]
    out = et.additive_residual(_frame(singles, doubles), seed=0)
    assert out["nonadditive_residual_var_share"] > 0.1
    assert "epistasis_var_share" not in out
    assert "epistasis_sd_z" not in out


def test_residual_can_exceed_the_total_and_says_so():
    """HIS7-style: no signal in the singles, noisy measurements. The residual
    variance exceeds the doubles' total variance (share > 1), which a real
    variance component cannot do, and the flag fires."""
    rng = np.random.default_rng(1)
    n_pos = 40
    singles = [(f"A{p}V", rng.normal(0, 1.0)) for p in range(1, n_pos + 1)]
    doubles = [(f"A{p}V:A{q}V", rng.normal(0, 0.3)) for p, q in _pairs(n_pos)]
    out = et.additive_residual(_frame(singles, doubles), seed=0)
    assert out["nonadditive_residual_var_share"] > 1.0
    assert out["residual_exceeds_total"] is True
    assert out["nonadditive_residual_sd_z"] > out["double_total_sd_z"]


def test_residual_flag_is_false_for_a_clean_additive_assay():
    rng = np.random.default_rng(2)
    n_pos = 40
    true = rng.normal(0, 1, n_pos)
    singles = [(f"A{p}V", true[p - 1]) for p in range(1, n_pos + 1)]
    doubles = [(f"A{p}V:A{q}V", true[p - 1] + true[q - 1])
               for p, q in _pairs(n_pos)]
    out = et.additive_residual(_frame(singles, doubles), seed=0)
    assert out["residual_exceeds_total"] is False
    assert out["nonadditive_residual_var_share"] < 0.05
    assert out["additive_spearman"] > 0.99


# ----------------------------------------------------------------------
# gap_ci: the position bootstrap
# ----------------------------------------------------------------------

def test_gap_ci_by_position_is_wider_than_by_variant_on_clustered_data():
    rng = np.random.default_rng(3)
    n_pos, per = 30, 40
    muts_m = [f"A{p}V:G{p + 1}L" for p in range(1, n_pos + 1)
              for _ in range(per)]
    eff = rng.normal(size=n_pos)
    t_m = np.repeat(eff, per)
    p_m = np.repeat(0.5 * eff + rng.normal(size=n_pos), per)
    muts_s = [f"A{p}V" for p in range(1, 41)]
    t_s = rng.normal(size=40)
    p_s = t_s + rng.normal(size=40)
    lo_v, hi_v = et.gap_ci(p_s, t_s, p_m, t_m, n_boot=120)
    lo_c, hi_c = et.gap_ci(p_s, t_s, p_m, t_m, muts_s, muts_m, n_boot=120)
    assert (hi_c - lo_c) > 1.5 * (hi_v - lo_v)


def test_stratum_metrics_interval_is_by_position_when_mutants_are_given():
    rng = np.random.default_rng(4)
    n_pos, per = 25, 40
    muts = [f"A{p}V:G{p + 1}L" for p in range(1, n_pos + 1) for _ in range(per)]
    eff = rng.normal(size=n_pos)
    truth = np.repeat(eff, per)
    pred = np.repeat(0.5 * eff + rng.normal(size=n_pos), per)
    et.N_BOOT = 150
    try:
        v = et.stratum_metrics(pred, truth)
        c = et.stratum_metrics(pred, truth, mutants=muts)
    finally:
        et.N_BOOT = 1000
    assert (c["spearman_hi"] - c["spearman_lo"]) > 1.5 * (
        v["spearman_hi"] - v["spearman_lo"])


# ----------------------------------------------------------------------
# run_assay end to end on a synthetic assay with a fake scorer
# ----------------------------------------------------------------------

class _FakeScorer:
    def __init__(self, n_pos, seed=0):
        self.rng = np.random.default_rng(seed)
        self.calls = 0

    def masked_marginal_logprobs(self, seq, block, batch_size=8):
        self.calls += 1
        return {p: self.rng.normal(size=20) for p in block}


def _build_assay(tmp_path, assay_id, n_pos=60, seed=0):
    rng = np.random.default_rng(seed)
    seq = "".join(rng.choice(list(AA), n_pos))
    rows = []
    for p in range(1, n_pos + 1):
        wt = seq[p - 1]
        for mut in rng.choice([a for a in AA if a != wt], 5, replace=False):
            rows.append((f"{wt}{p}{mut}", rng.normal()))
    for _ in range(600):
        p, q = sorted(rng.choice(np.arange(1, n_pos + 1), 2, replace=False))
        m1 = rng.choice([a for a in AA if a != seq[p - 1]])
        m2 = rng.choice([a for a in AA if a != seq[q - 1]])
        rows.append((f"{seq[p - 1]}{p}{m1}:{seq[q - 1]}{q}{m2}", rng.normal()))
    pd.DataFrame(rows, columns=["mutant", "DMS_score"]).to_csv(
        tmp_path / f"{assay_id}.csv", index=False)
    ref = pd.DataFrame([{
        "DMS_id": assay_id, "UniProt_ID": "SYN", "taxon": "Human",
        "source_organism": "x", "seq_len": n_pos, "target_seq": seq,
        "DMS_number_single_mutants": n_pos * 5,
        "coarse_selection_type": "Binding", "selection_assay": "syn"}])
    return ref


@pytest.fixture
def synthetic_run(tmp_path, monkeypatch):
    def go(assay_id):
        ref = _build_assay(tmp_path, assay_id)
        monkeypatch.setattr(et.data, "ASSAY_DIR", tmp_path)
        monkeypatch.setattr(et, "OUT", tmp_path)
        monkeypatch.setattr(et, "CACHE", tmp_path / "_cache")
        monkeypatch.setattr(et, "N_BOOT", 80)   # gap_bootstrap needs >= 50
        return et.run_assay(_FakeScorer(60), assay_id, ref, cap=500, seed=0)
    return go


def test_different_quantity_flag_is_a_real_boolean_true_for_a_listed_assay(
        synthetic_run):
    """C8. The column used to hold '1sub:300, 2sub:200'. For an assay that is
    a member of a DIFFERENT_QUANTITY pair it must be True; for any other, False;
    and the composition string lives under its own name."""
    listed = synthetic_run("RASK_HUMAN_Weng_2022_abundance")
    assert listed["different_quantity_flag"].dtype == bool
    assert listed["different_quantity_flag"].all()
    # the composition is per stratum: the single row says 1sub, the others 2sub
    assert listed["n_sub_composition"].iloc[0].startswith("1sub:")
    assert listed["n_sub_composition"].str.contains("sub:").all()


def test_different_quantity_flag_is_false_for_an_unlisted_assay(synthetic_run):
    other = synthetic_run("SYN_ASSAY_NOT_LISTED")
    assert other["different_quantity_flag"].dtype == bool
    assert not other["different_quantity_flag"].any()
    assert (other["reference_coarse_selection_type"] == "Binding").all()


def test_run_assay_reports_position_clustered_gaps_and_the_controls(
        synthetic_run, tmp_path):
    df = synthetic_run("SYN_ASSAY_A")
    r = df.iloc[0]
    assert r["gap_resampling_unit"].startswith("position")
    assert r["n_position_clusters"] == 60
    for col in ("gap_lo", "gap_hi", "range_matched_gap_lo",
                "range_matched_gap_hi", "gap_top_decile_lo",
                "gap_single_minus_nsub2", "pos_matched_binds",
                "n_single_positions", "n_multi_positions"):
        assert col in df.columns, col
    assert r["gap_lo"] <= r["gap_single_minus_multi"] <= r["gap_hi"]
    # the scored per-variant frame is saved so intervals can be recomputed
    assert (tmp_path / "scored_SYN_ASSAY_A.csv.gz").exists()


def test_position_matched_control_says_when_it_binds(synthetic_run):
    """V6. Here every single sits on a position some multi-mutant touches, so
    the 'control' is the unrestricted singles stratum again and the output must
    say it does not bind."""
    df = synthetic_run("SYN_ASSAY_B")
    r = df.iloc[0]
    assert bool(r["pos_matched_binds"]) == bool(r["pos_matched_n"] < 300)
    if r["n_multi_positions"] >= r["n_single_positions"]:
        assert not r["pos_matched_binds"]
        assert r["pos_matched_n"] == 300



def test_load_strata_keeps_the_first_duplicate_and_reports_the_count(
        tmp_path, monkeypatch):
    """C12: a mutant listed twice used to enter the strata twice, silently."""
    monkeypatch.setattr(et.data, "ASSAY_DIR", tmp_path)
    rows = [(f"A{p}V", float(p)) for p in range(1, 61)]
    rows.append(("A1V", 99.0))                     # duplicate of the first
    rows += [(f"A{p}V:G{p + 1}L", float(p)) for p in range(1, 60)]
    pd.DataFrame(rows, columns=["mutant", "DMS_score"]).to_csv(
        tmp_path / "DUP.csv", index=False)
    strata, full = et.load_strata("DUP", cap=500)
    assert strata["single"]["mutant"].is_unique
    first = strata["single"].loc[strata["single"]["mutant"] == "A1V",
                                 "DMS_score"].iloc[0]
    assert first == 1.0
    assert all(f.attrs["duplicates_dropped"] == 1 for f in strata.values())
    assert full["mutant"].is_unique
