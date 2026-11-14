"""What a DMS assay can and cannot resolve, measured from the assays themselves.

The top-10 recall section of the README asks why the model's ten highest picks
are no better than random on five of nine assays while its Spearman over the
same data is 0.6. Burial is ruled out. Two candidates are left, and both are
properties of the *assay* rather than of the model:

  - the assay noise floor: a DMS experiment measures a noisy proxy, and no
    predictor can rank better than the measurement can resolve;
  - the heterogeneity of the top decile: "the top 10%" is a threshold applied
    to assays whose scales, ceilings and dynamic ranges are not the same thing.

Both are measurable with data already on disk, because ProteinGym contains
several proteins assayed more than once. Two assays of the same protein scored
on the same variants are two independent measurements of the same underlying
biology (or, where the assay condition differs, two different truths). Either
way they bound what any predictor could achieve.

The estimator used here

    sigma_noise = sd(a - b) / sqrt(2)

assumes the two measurements carry independent errors of equal variance, which
is what replicate assays of the same protein are for. The reliability of a
measurement within a subset of variants is then

    rel = 1 - sigma_noise^2 / var(observed within subset)

and rel is the ceiling on any *linear* correlation between two independent
measurements of that subset -- the attenuation limit. A subset with rel near
zero cannot be ranked by anything, no matter how good the predictor.
"""

import numpy as np
from scipy import stats


def top_decile_mask(truth, frac=0.1):
    """Boolean mask of the variants in the top `frac` of the assay.

    Uses the same `quantile(1 - frac)` threshold as ``evaluate.top_k_recall``,
    so the subset here is the subset that metric scores against. The
    threshold is inclusive, which on an assay with heavy ties can put more
    than `frac` of the variants inside.
    """
    truth = np.asarray(truth, dtype=float)
    ok = np.isfinite(truth)
    mask = np.zeros(len(truth), dtype=bool)
    if ok.sum() < 20:
        return mask
    thr = np.quantile(truth[ok], 1 - frac)
    mask[ok] = truth[ok] >= thr
    return mask


def noise_sd(a, b):
    """Per-measurement noise sd, in the assay's own units.

    Only meaningful when the two measurements are already on a common scale.
    Two different DMS assays never are -- they use different reporters, so a
    raw ``sd(a - b)`` is dominated by the difference in units and dynamic
    range rather than by error. Use :func:`noise_sd_z` for cross-assay pairs.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 10 or len(a) != len(b):
        return np.nan
    d = a - b
    if not np.all(np.isfinite(d)):
        return np.nan
    return float(d.std(ddof=1) / np.sqrt(2.0))


def zscore(values):
    """Standardise, so two assays' differences are differences in units of
    their own spread rather than in their own units."""
    v = np.asarray(values, dtype=float)
    ok = np.isfinite(v)
    if ok.sum() < 10:
        return np.full(v.shape, np.nan)
    mu = v[ok].mean()
    sd = v[ok].std(ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return np.full(v.shape, np.nan)
    return (v - mu) / sd


def noise_sd_z(a, b):
    """Per-measurement noise sd in pooled standard-deviation units.

    Each measurement is standardised over the shared variants before the
    difference is taken, so a scale factor or offset between the two assays --
    which carries no information about error -- cancels. What is left is the
    disagreement in units of the assays' own spread.

    The estimator assumes the two errors are independent and equal in
    variance, which is the usual assumption when two assays of the same
    protein are read as replicates. It is a point estimate from one pair, and
    is reported as one.
    """
    za = zscore(a)
    zb = zscore(b)
    ok = np.isfinite(za) & np.isfinite(zb)
    if ok.sum() < 10:
        return np.nan
    d = za[ok] - zb[ok]
    return float(d.std(ddof=1) / np.sqrt(2.0))


def reliability_z(values, sigma_z):
    """Reliability of a subset, in standardised units.

    ``1 - sigma^2 / var`` where both are in units of the whole assay's spread.
    On the whole shared set ``var`` is 1, so this reduces to ``1 - sigma^2``.
    Inside a subset that spans less of the range than the whole assay ``var``
    is below 1 and the reliability is correspondingly lower -- that is the
    range-restriction attenuation the top-k question turns on.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 5 or not np.isfinite(sigma_z):
        return np.nan
    var = float(v.var(ddof=1))
    if var <= 0:
        return np.nan
    return float(1.0 - (sigma_z ** 2) / var)


def reliability(values, sigma):
    """Fraction of a subset's observed variance that is signal rather than
    the measurement noise estimated by :func:`noise_sd`.

    Capped at 1 (a point estimate from a finite sample can come out negative
    or above one) but never floored at 0: a negative reliability is the
    informative statement that the noise is larger than the subset's whole
    spread, so it is reported as measured.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 5 or not np.isfinite(sigma):
        return np.nan
    var = float(v.var(ddof=1))
    if var <= 0:
        return np.nan
    return float(min(1.0, 1.0 - (sigma ** 2) / var))


def paired_ranks_agreement(a, b):
    """Spearman between two measurements restricted to a shared subset."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 20 or np.unique(a).size < 3 or np.unique(b).size < 3:
        return np.nan
    return float(stats.spearmanr(a, b).statistic)


def top_decile_overlap(a, b, frac=0.1):
    """Jaccard overlap of two assays' top deciles, over shared variants.

    Two different questions from the rank correlation: not "do they order the
    top decile the same way" but "is it the same set of variants".
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 50:
        return np.nan
    ta = top_decile_mask(a, frac)
    tb = top_decile_mask(b, frac)
    union = int((ta | tb).sum())
    if union == 0:
        return np.nan
    return float((ta & tb).sum() / union)


def decile_shape(truth, frac=0.1):
    """How spread out the top decile is, in the assay's own units.

    ``spread_ratio`` is sd(top decile) / sd(whole assay). A decile with a
    large ratio is heterogeneous -- the top 10% is a wide band of genuinely
    different variants. A small ratio means the top decile is a plateau: the
    variants are all near the same score, so their ordering is decided by
    differences far below the assay's overall dynamic range.
    """
    truth = np.asarray(truth, dtype=float)
    ok = np.isfinite(truth)
    truth = truth[ok]
    if len(truth) < 50:
        return {}
    mask = top_decile_mask(truth, frac)
    top = truth[mask]
    if len(top) < 5:
        return {}
    sd_all = float(truth.std(ddof=1))
    sd_top = float(top.std(ddof=1))
    rng = float(truth.max() - truth.min())
    return {
        "n": int(len(truth)),
        "n_top": int(len(top)),
        "sd_all": sd_all,
        "sd_top": sd_top,
        "spread_ratio": sd_top / sd_all if sd_all > 0 else np.nan,
        # share of the whole assay's range that the top decile spans
        "range_share": (float(top.max() - top.min()) / rng) if rng > 0 else np.nan,
        # distinct measured values per variant: 1.0 means no ties at all
        "distinct_frac": float(np.unique(truth).size / len(truth)),
        "at_max_frac": float(np.mean(truth >= truth.max() - 1e-12)),
        "n_positions_proxy_ties": int(len(np.unique(top))),
    }
