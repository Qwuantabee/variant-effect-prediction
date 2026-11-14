"""Controls for comparing a model's accuracy across two variant strata.

Singles and multi-mutants differ in more than the number of substitutions: the
multi-mutant library carries a long deleterious tail the singles library does
not, so its measured fitness spreads over a wider range, and a rank correlation
over a wider spread is mechanically easier. range_matched_rho is the control
that holds the spread fixed.
"""

import numpy as np

from . import evaluate


def range_window(truth_a, truth_b, frac=0.90):
    """The fitness window both strata share: the intersection of each
    stratum's central frac of measured values. None if it is empty."""
    a = np.asarray(truth_a, float)
    b = np.asarray(truth_b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 50 or len(b) < 50:
        return None
    tail = (1.0 - frac) / 2.0
    a_lo, a_hi = np.quantile(a, [tail, 1 - tail])
    b_lo, b_hi = np.quantile(b, [tail, 1 - tail])
    lo, hi = max(a_lo, b_lo), min(a_hi, b_hi)
    return (float(lo), float(hi)) if hi > lo else None


def range_matched_rho(pred, truth, other_pred, other_truth, frac=0.90):
    """Spearman on the fitness window the two strata share.

    A stratum of multi-mutants is not a random sample of the protein's fitness
    distribution: it carries a long deleterious tail that a single-substitution
    library does not. A rank correlation computed over a wider spread is
    mechanically easier, so an apparent multi-mutant advantage (or, when the
    singles are the wider stratum, an apparent singles advantage) can be an
    artefact of that spread. Trimming both strata to the intersection of their
    central frac of measured fitness removes the tails and asks the same
    question on comparable ground.

    Returns (rho, other_rho, n, other_n); NaN rhos when a stratum keeps fewer
    than 50 variants. Point estimate only; the interval is gap_bootstrap over
    range_matched_gap_fn, resampled by position.
    """
    pred, truth = np.asarray(pred, float), np.asarray(truth, float)
    other_pred = np.asarray(other_pred, float)
    other_truth = np.asarray(other_truth, float)
    ok = np.isfinite(pred) & np.isfinite(truth)
    ok2 = np.isfinite(other_pred) & np.isfinite(other_truth)
    if ok.sum() < 50 or ok2.sum() < 50:
        return (np.nan, np.nan, 0, 0)
    win = range_window(truth[ok], other_truth[ok2], frac)
    if win is None:
        return (np.nan, np.nan, 0, 0)
    lo, hi = win
    ka = ok & (truth >= lo) & (truth <= hi)
    kb = ok2 & (other_truth >= lo) & (other_truth <= hi)
    if ka.sum() < 50 or kb.sum() < 50:
        return (np.nan, np.nan, int(ka.sum()), int(kb.sum()))
    return (evaluate.spearman(pred[ka], truth[ka]),
            evaluate.spearman(other_pred[kb], other_truth[kb]),
            int(ka.sum()), int(kb.sum()))


def range_matched_gap(single, double, frac=0.90):
    """rho(single) - rho(double) on the shared window; NaN if undefined.

    single and double are (pred, truth) pairs. This is the scalar the position
    bootstrap resamples (it needs the window recomputed per replicate, so it
    takes both strata at once rather than being a per-stratum metric).
    """
    rs, rd, _, _ = range_matched_rho(single[0], single[1],
                                     double[0], double[1], frac=frac)
    return rs - rd
