"""A better target than one assay: the average of two replicate measurements.

The top-k section of the README compares the model's top-10 recall to the
recall one assay achieves against another, and calls that second number a
ceiling. It is not one. Two assays of the same protein are two measurements of
the same latent quantity, each carrying measurement error, and the agreement
between two noisy measurements is attenuated by both errors. A predictor that
tracked the latent quantity closely would beat their mutual agreement while
contradicting nothing.

Averaging the two measurements reduces that error. For a pair of assays on the
variants they share, each standardised to zero mean and unit variance over the
shared set,

    pooled = (z_A + z_B) / 2

If the two measurement errors are independent, of equal variance, and
independent of the latent value -- the assumptions the repo's existing
``sd(a - b)/sqrt(2)`` noise estimator already makes -- then ``pooled`` has half
the measurement variance of either assay and its correlation with the latent
truth is higher. Pooling is therefore closer to the truth than either assay,
though not equal to it:

  * half the independent noise is still there, so a metric computed against
    ``pooled`` is a *lower* bound on what the same predictor would score
    against the latent quantity. Measured recall understates headroom.
  * error the two assays *share* is untouched by averaging. A common variant
    library, a common reporter, a common growth readout: all survive pooling.
    Where the pair measures deliberately different quantities, their average
    is a pseudo-quantity no experiment measured, so a metric against it can
    be optimistic rather than conservative.

Both caveats are properties of the data, not of this code, and the caller
reports the pairs for which each applies.
"""

import numpy as np
from scipy import stats


def zscore(values):
    """Standardise over the finite entries. All-NaN input stays NaN."""
    v = np.asarray(values, dtype=float)
    ok = np.isfinite(v)
    out = np.full(v.shape, np.nan)
    if ok.sum() < 10:
        return out
    sd = v[ok].std(ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return out
    out[ok] = (v[ok] - v[ok].mean()) / sd
    return out


def pool_z(a, b):
    """Mean of two z-scored measurements, the lower-noise truth estimate.

    Each is standardised over its own finite entries first, so the pooling is
    scale-free: an assay reported in fluorescence units and one in growth
    rate contribute equally. Variants missing from one measurement come out
    NaN rather than silently using the other alone, because a half-pooled
    value is a different quantity from a full one and mixing them into one
    target would make the target inconsistent across variants.
    """
    za = zscore(a)
    zb = zscore(b)
    ok = np.isfinite(za) & np.isfinite(zb)
    out = np.full(za.shape, np.nan)
    out[ok] = (za[ok] + zb[ok]) / 2.0
    return out


def split_half_reliability(a, b):
    """Correlation between the two replicate measurements, z-scored.

    This is the split-half reliability of the pair: the share of the ordering
    one assay reproduces in the other. It equals the correlation of the pooled
    estimate with the latent value only under the equal-error, independent-
    error assumptions above -- but as a first-order diagnostic it is exact,
    because Spearman is invariant to the affine z-scoring.

    Spearman rather than Pearson because the assays are only ordinal
    comparably across reporters; and because the repo's headline metric is a
    rank metric.
    """
    za = zscore(a)
    zb = zscore(b)
    ok = np.isfinite(za) & np.isfinite(zb)
    if ok.sum() < 20:
        return np.nan
    if np.unique(za[ok]).size < 3 or np.unique(zb[ok]).size < 3:
        return np.nan
    return float(stats.spearmanr(za[ok], zb[ok]).statistic)


def pooled_noise_sd(a, b):
    """Per-measurement noise sd in z units -- the same estimator the ceiling
    table uses, restated here so the pooled target's remaining noise is
    explicit rather than implied: pooling leaves ``noise_sd / sqrt(2)``."""
    za = zscore(a)
    zb = zscore(b)
    ok = np.isfinite(za) & np.isfinite(zb)
    if ok.sum() < 10:
        return np.nan
    return float((za[ok] - zb[ok]).std(ddof=1) / np.sqrt(2.0))


def reliability_ceiling(split_half):
    """Spearman-Brown: the reliability of the *pooled* measure implied by the
    reliability of one measurement.

    ``2r / (1 + r)`` is the standard correction for doubling the number of
    measurements, and it states quantitatively what pooling buys: a pair at
    r = 0.5 gives a pooled target at 0.667, not at 1.0.
    """
    r = float(split_half)
    if not np.isfinite(r) or r <= -1.0:
        return np.nan
    return float(2.0 * r / (1.0 + r))
