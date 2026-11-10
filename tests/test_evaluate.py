import numpy as np
import pytest

from variant_effect import evaluate


def test_spearman_perfect_and_inverted():
    truth = np.array([1.0, 2, 3, 4, 5])
    assert evaluate.spearman(truth, truth) == pytest.approx(1.0)
    assert evaluate.spearman(-truth, truth) == pytest.approx(-1.0)


def test_spearman_is_rank_based():
    """Monotone transform of the predictions must not change it. This is
    why Spearman and not Pearson -- DMS readouts are on arbitrary nonlinear
    scales."""
    rng = np.random.default_rng(0)
    truth = rng.normal(size=200)
    pred = truth + rng.normal(scale=0.5, size=200)
    assert evaluate.spearman(np.exp(pred), truth) == pytest.approx(
        evaluate.spearman(pred, truth))


def test_spearman_handles_nan():
    pred = np.array([1.0, np.nan, 3, 4])
    truth = np.array([1.0, 2, 3, 4])
    assert evaluate.spearman(pred, truth) == pytest.approx(1.0)


def test_spearman_ci_brackets_estimate():
    rng = np.random.default_rng(1)
    truth = rng.normal(size=300)
    pred = truth + rng.normal(scale=1.0, size=300)
    point = evaluate.spearman(pred, truth)
    lo, hi = evaluate.spearman_ci(pred, truth, n_boot=400, seed=1)
    assert lo < point < hi


def test_ci_is_wider_for_small_assays():
    rng = np.random.default_rng(2)
    big_truth = rng.normal(size=500)
    big_pred = big_truth + rng.normal(scale=1.0, size=500)
    small_truth, small_pred = big_truth[:40], big_pred[:40]

    big_lo, big_hi = evaluate.spearman_ci(big_pred, big_truth, n_boot=400, seed=2)
    sm_lo, sm_hi = evaluate.spearman_ci(small_pred, small_truth, n_boot=400, seed=2)
    assert (sm_hi - sm_lo) > (big_hi - big_lo)


def test_top_k_recall_perfect_predictor():
    truth = np.arange(100, dtype=float)
    assert evaluate.top_k_recall(truth, truth, k=10, top_frac=0.1) == 1.0


def test_top_k_recall_inverted_predictor():
    truth = np.arange(100, dtype=float)
    assert evaluate.top_k_recall(-truth, truth, k=10, top_frac=0.1) == 0.0


def test_ndcg_bounds():
    rng = np.random.default_rng(3)
    truth = rng.normal(size=100)
    assert evaluate.ndcg(truth, truth) == pytest.approx(1.0)
    noise = rng.normal(size=100)
    assert 0 <= evaluate.ndcg(noise, truth) <= 1


def test_evaluate_returns_all_keys():
    rng = np.random.default_rng(4)
    truth = rng.normal(size=120)
    pred = truth + rng.normal(scale=0.8, size=120)
    out = evaluate.evaluate(pred, truth, truth_bin=(truth > 0).astype(int))
    for key in ("spearman", "spearman_lo", "spearman_hi", "ndcg",
                "top10_recall", "n", "auc"):
        assert key in out
    assert out["n"] == 120
