"""Scoring the scorers.

Spearman is the headline number because the experimental scale is arbitrary
and usually nonlinear -- a DMS assay measures growth or binding or
fluorescence, not free energy, so only the ranking is meaningful.

The other metrics are here because Spearman over a whole assay hides the
thing you would actually use a predictor for. If you are picking 10 variants
to make, you care whether the top 10 are good, not whether the model got the
ordering right among the 400 dead ones.
"""

import numpy as np
from scipy import stats


def spearman(pred, truth):
    ok = np.isfinite(pred) & np.isfinite(truth)
    if ok.sum() < 3:
        return np.nan
    return float(stats.spearmanr(pred[ok], truth[ok]).statistic)


def spearman_ci(pred, truth, n_boot=1000, seed=0):
    """Bootstrap CI on Spearman.

    Worth reporting because assays vary from 50 to 10000 variants and a
    correlation from the small ones is a lot shakier than the headline
    number suggests.
    """
    rng = np.random.default_rng(seed)
    ok = np.isfinite(pred) & np.isfinite(truth)
    p, t = pred[ok], truth[ok]
    n = len(p)
    if n < 10:
        return (np.nan, np.nan)
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(t[idx])) < 3:
            continue
        draws.append(stats.spearmanr(p[idx], t[idx]).statistic)
    return (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))


def top_k_recall(pred, truth, k=10, top_frac=0.1):
    """Of the k variants the model ranks highest, how many are truly in the
    top `top_frac` of the assay?

    This is the design-campaign question: you can afford to build k variants,
    how many are worth building.
    """
    ok = np.isfinite(pred) & np.isfinite(truth)
    p, t = pred[ok], truth[ok]
    if len(p) < k:
        return np.nan
    threshold = np.quantile(t, 1 - top_frac)
    top_idx = np.argsort(-p)[:k]
    return float(np.mean(t[top_idx] >= threshold))


def auc_binary(pred, truth_bin):
    """ROC AUC against a binary label, when the assay provides one."""
    from sklearn.metrics import roc_auc_score
    ok = np.isfinite(pred) & np.isfinite(truth_bin)
    if len(np.unique(truth_bin[ok])) < 2:
        return np.nan
    return float(roc_auc_score(truth_bin[ok], pred[ok]))


def ndcg(pred, truth, k=None):
    """Normalised discounted cumulative gain, with the experimental score
    shifted to be non-negative and used as the gain."""
    ok = np.isfinite(pred) & np.isfinite(truth)
    p, t = pred[ok], truth[ok]
    if len(p) == 0:
        return np.nan
    gain = t - t.min()
    k = k or len(p)

    def dcg(order):
        g = gain[order][:k]
        return float(np.sum(g / np.log2(np.arange(2, len(g) + 2))))

    actual = dcg(np.argsort(-p))
    ideal = dcg(np.argsort(-t))
    return actual / ideal if ideal > 0 else np.nan


def evaluate(pred, truth, truth_bin=None, k=10):
    """All of the above in one dict."""
    lo, hi = spearman_ci(pred, truth)
    out = {
        "spearman": spearman(pred, truth),
        "spearman_lo": lo,
        "spearman_hi": hi,
        "ndcg": ndcg(pred, truth),
        f"top{k}_recall": top_k_recall(pred, truth, k=k),
        "n": int(np.isfinite(pred).sum()),
    }
    if truth_bin is not None:
        out["auc"] = auc_binary(pred, np.asarray(truth_bin, dtype=float))
    return out
