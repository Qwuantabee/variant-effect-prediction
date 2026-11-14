"""The simulated controls behind the self-inclusion and decile-reliability
rewrites (review C13, C14)."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import sim_pooling_controls as sim  # noqa: E402


def test_self_inclusion_alone_lifts_recall_when_the_assays_share_no_signal():
    """Two independent noise columns: A vs B recall sits at chance (about 0.1),
    but A vs pool(A,B) is high because A is half of its own target. So 'an assay
    scored against the pool gains X' proves nothing about attenuation."""
    out = sim.selfinclusion(noise_sd=0.6, n=800, reps=40, seed=1)
    none = out["no_shared_signal"]
    assert none["a_to_b"] < 0.25
    assert none["a_to_pooled"] > 0.6
    assert none["self_inclusion_gain_a_to_pooled_minus_a_to_b"] > 0.4
    # a predictor that knows the latent value gains nothing from A: there is
    # no latent signal in this regime
    assert none["latent_to_pooled"] < 0.25


def test_self_inclusion_is_smaller_but_present_when_the_assays_do_share_signal():
    out = sim.selfinclusion(noise_sd=0.6, n=800, reps=40, seed=2)
    shared = out["shared_latent"]
    assert shared["a_to_pooled"] > shared["a_to_b"]
    assert shared["latent_to_pooled"] > shared["latent_to_single_a"]


def test_latent_sd_makes_the_assay_total_variance_one():
    for noise in (0.3, 0.6, 0.9):
        assert sim._latent_sd(noise) ** 2 + noise ** 2 == pytest.approx(1.0)


def test_decile_formula_goes_negative_while_the_latent_is_still_rankable():
    """The README claim was: a subset whose reliability is near zero cannot be
    ranked by anything. Selecting the top decile on the noisy assay itself
    drives the formula below zero although the latent value inside the subset
    has real variance and the two assays still correlate there."""
    out = sim.decile_reliability(noise_sd=0.6, n=1500, reps=40, seed=3)
    assert out["formula_reliability"] < 0
    assert out["share_of_reps_with_negative_formula"] > 0.9
    assert out["var_latent_in_subset"] > 0.1
    assert out["corr_a_b_in_subset"] > 0.1
    assert out["corr_latent_a_in_subset"] > out["corr_a_b_in_subset"]


def test_decile_formula_is_not_misleading_when_selecting_independently():
    """Control: the same formula applied to a subset chosen on a variable
    independent of the noise (here a random decile) is close to the true
    reliability, so the failure is specific to selecting on the assay itself."""
    from variant_effect import assay_stats
    rng = np.random.default_rng(4)
    n = 6000
    latent = rng.normal(size=n) * sim._latent_sd(0.6)
    a = latent + rng.normal(0, 0.6, n)
    b = latent + rng.normal(0, 0.6, n)
    za = assay_stats.zscore(a)
    sigma = assay_stats.noise_sd_z(a, b)
    random_mask = rng.random(n) < 0.5
    rel = assay_stats.reliability_z(za[random_mask], sigma)
    # true reliability of a z-scored assay with noise sd 0.6 is 1 - 0.6**2
    assert abs(rel - (1 - 0.6 ** 2)) < 0.1
