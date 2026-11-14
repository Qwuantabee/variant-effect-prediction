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


def _rank_rows(a):
    """Average ranks along axis 1, ties sharing the mean rank.

    Fully vectorised, and not by preference: scipy's rankdata takes only 1-D
    input, and both alternatives are too slow here. A Python loop over 1000
    bootstrap replicates costs ~15 s on a 50,000-variant assay; a per-row
    np.unique costs about 2 s. This version ranks every replicate in one pass
    of bincount over a flattened group array.

    Ties get average ranks. ProteinGym assays are full of repeated fitness
    values, and ordinal ranks would break ties in whatever order they happened
    to appear, which changes rho by more than the effect being measured.
    """
    a = np.atleast_2d(np.asarray(a, dtype=float))
    nr, nc = a.shape
    order = np.argsort(a, axis=1, kind="stable")
    s = np.take_along_axis(a, order, axis=1)

    # group id per element, restarting at 0 for each row
    g = np.zeros((nr, nc), dtype=np.int64)
    if nc > 1:
        g[:, 1:] = np.cumsum(s[:, 1:] != s[:, :-1], axis=1)

    # offset each row's group ids so one bincount covers the whole matrix
    per_row = g[:, -1] + 1
    offsets = np.zeros(nr, dtype=np.int64)
    if nr > 1:
        offsets[1:] = np.cumsum(per_row)[:-1]
    flat = g + offsets[:, None]
    n_groups = int(flat.max()) + 1 if flat.size else 0

    counts = np.bincount(flat.ravel(), minlength=n_groups).astype(float)
    ends = np.cumsum(counts)
    starts = ends - counts
    avg = (starts + 1.0 + ends) / 2.0          # mean of 1-based ranks in group
    ranks_sorted = avg[flat]

    ranks = np.empty((nr, nc), dtype=float)
    np.put_along_axis(ranks, order, ranks_sorted, axis=1)
    return ranks


def spearman_ci(pred, truth, n_boot=1000, seed=0, max_resample=20000,
                groups=None):
    """Bootstrap CI on Spearman.

    RESAMPLING UNIT: with groups=None the unit is the VARIANT (iid draw of
    rows). That is only valid when variants are independent. DMS variants are
    not: they cluster by position, and a variant-level interval on clustered
    data is too narrow (RASK_HUMAN doubles: about 3x). Whenever the position of
    each variant is known pass groups, one entry per input row -- a label, or
    an iterable of labels for a multi-position variant (see
    resampling.ClusterIndex). Then the unit is the POSITION: positions are
    drawn with replacement and each variant is weighted by the product of the
    multiplicities of the positions it touches.

    Variant-level mode (groups=None) caps the draw at max_resample variants per
    replicate: without a cap the largest ProteinGym assay (HIS7_YEAST, 496,137
    variants) spends minutes inside 1000 scipy calls and a (n_boot, n) float64
    matrix would need 4 GB. For n above max_resample the cap draws fewer
    variants than exist, so the interval is WIDER than the full-n variant-level
    one by about sqrt(n / max_resample) (it is not "within 1% of full-n"
    beyond n = max_resample; every n in this repo's reported intervals is at
    most 21,442, where the factor is about 1.04). Cluster mode does not cap: it
    realises the full resampled set.
    """
    rng = np.random.default_rng(seed)
    ok = np.isfinite(pred) & np.isfinite(truth)
    p = np.asarray(pred[ok], dtype=float)
    t = np.asarray(truth[ok], dtype=float)
    n = len(p)
    if n < 10:
        return (np.nan, np.nan)

    if groups is not None:
        from .resampling import ClusterIndex, expand
        grp = [g for g, keep in zip(groups, ok) if keep]
        index = ClusterIndex(grp)
        reps = []
        for _ in range(n_boot):
            idx = expand(index.draw(rng))
            if len(idx) < 10:
                continue
            r = stats.spearmanr(p[idx], t[idx]).statistic
            if np.isfinite(r):
                reps.append(r)
        if len(reps) < 10:
            return (np.nan, np.nan)
        return (float(np.percentile(reps, 2.5)),
                float(np.percentile(reps, 97.5)))

    draw = int(min(n, max_resample))
    idx = rng.integers(0, n, (n_boot, draw))
    rp = _rank_rows(p[idx])
    rt = _rank_rows(t[idx])
    rp -= rp.mean(axis=1, keepdims=True)
    rt -= rt.mean(axis=1, keepdims=True)
    num = (rp * rt).sum(axis=1)
    den = np.sqrt((rp ** 2).sum(axis=1) * (rt ** 2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = np.where(den > 0, num / den, np.nan)
    rho = rho[np.isfinite(rho)]
    if len(rho) < 10:
        return (np.nan, np.nan)
    return (float(np.percentile(rho, 2.5)), float(np.percentile(rho, 97.5)))


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


def evaluate(pred, truth, truth_bin=None, k=10, groups=None):
    """All of the above in one dict. groups is passed to spearman_ci (None =
    variant-level interval, which understates the uncertainty on clustered
    data; see spearman_ci)."""
    lo, hi = spearman_ci(pred, truth, groups=groups)
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
