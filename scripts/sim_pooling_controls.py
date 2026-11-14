"""Simulated controls for two README claims that the data alone cannot support.

Both claims were flagged in review as 'vacuous or false under a simple
generative model', and a simulation is the proper way to find out how vacuous.

C13 -- self-inclusion in 'assay vs pooled target'
-------------------------------------------------
run_latent_truth_test reports recall of assay A against pooled = (zA + zB)/2.
A is half of that target, so A's own measurement noise counts as signal. How
much of the gain over A-vs-B recall is that self-inclusion alone? Simulated
here with (i) two assays that share NO signal and (ii) two assays measuring one
latent value at the repo's median noise level, scoring A->B, A->pool, B->pool,
and a predictor that knows the latent value exactly.

C14 -- decile 'reliability' selected on the same noisy assay
------------------------------------------------------------
assay_stats.reliability_z = 1 - sigma^2 / var(subset) is applied to the top
decile chosen BY the measurement it is then computed on. Selecting on a noisy
y inflates the noise in the selected subset's y-values (regression to the mean)
and shrinks the subset's variance, so the formula can be negative while the
latent value inside the subset is still partly rankable. Simulated here: the
formula's value, the true latent variance inside the subset, and the
within-subset correlation of the two assays and of assay with latent.

Everything is seeded; outputs go to results/sim_controls/summary.json.
Run:  python scripts/sim_pooling_controls.py
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from variant_effect import assay_stats, evaluate, pooled_truth  # noqa: E402

OUT = ROOT / "results" / "sim_controls"
N = 2000
REPS = 400
K = 10
TOP = 0.1
SEED = 0


def _noise_level():
    """The repo's own median per-assay noise sd (z units) from the real pairs."""
    path = ROOT / "results" / "latent_truth" / "summary.json"
    if path.exists():
        return float(json.loads(path.read_text())["assays"]["median_noise_sd_z"])
    return 0.6


def _latent_sd(noise_sd):
    """Latent sd that makes each assay's TOTAL sd 1, so that noise_sd is in the
    same z-score units as assay_stats.noise_sd_z (a z-scored assay has unit
    variance: signal variance 1 - noise_sd**2)."""
    return float(np.sqrt(max(1.0 - noise_sd ** 2, 1e-9)))


def selfinclusion(noise_sd, n=N, reps=REPS, seed=SEED):
    """Mean top-10 recall over reps for each (predictor, target) combination."""
    rng = np.random.default_rng(seed)
    out = {}
    for label, shared in (("no_shared_signal", False), ("shared_latent", True)):
        acc = {k: [] for k in ("a_to_b", "a_to_pooled", "b_to_pooled",
                               "latent_to_single_a", "latent_to_pooled")}
        for _ in range(reps):
            if shared:
                latent = rng.normal(size=n) * _latent_sd(noise_sd)
                a = latent + rng.normal(0, noise_sd, n)
                b = latent + rng.normal(0, noise_sd, n)
            else:
                latent = rng.normal(size=n)
                a = rng.normal(size=n)
                b = rng.normal(size=n)
            za, zb = pooled_truth.zscore(a), pooled_truth.zscore(b)
            pool = pooled_truth.pool_z(a, b)
            acc["a_to_b"].append(evaluate.top_k_recall(za, b, k=K, top_frac=TOP))
            acc["a_to_pooled"].append(
                evaluate.top_k_recall(za, pool, k=K, top_frac=TOP))
            acc["b_to_pooled"].append(
                evaluate.top_k_recall(zb, pool, k=K, top_frac=TOP))
            acc["latent_to_single_a"].append(
                evaluate.top_k_recall(latent, a, k=K, top_frac=TOP))
            acc["latent_to_pooled"].append(
                evaluate.top_k_recall(latent, pool, k=K, top_frac=TOP))
        out[label] = {k: float(np.mean(v)) for k, v in acc.items()}
        out[label]["self_inclusion_gain_a_to_pooled_minus_a_to_b"] = (
            out[label]["a_to_pooled"] - out[label]["a_to_b"])
    out["noise_sd_z"] = noise_sd
    out["n_variants"] = n
    out["reps"] = reps
    return out


def decile_reliability(noise_sd, n=N, reps=REPS, seed=SEED):
    """The reliability_z formula applied to the top decile of assay A."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(reps):
        latent = rng.normal(size=n) * _latent_sd(noise_sd)
        a = latent + rng.normal(0, noise_sd, n)
        b = latent + rng.normal(0, noise_sd, n)
        za, zb = assay_stats.zscore(a), assay_stats.zscore(b)
        sigma_hat = assay_stats.noise_sd_z(a, b)
        mask = assay_stats.top_decile_mask(a, TOP)
        sub = za[mask]
        rows.append({
            "sigma_hat": sigma_hat,
            "formula_reliability": assay_stats.reliability_z(sub, sigma_hat),
            "var_observed_in_subset": float(sub.var(ddof=1)),
            "var_latent_in_subset": float(
                (latent[mask] / a.std(ddof=1)).var(ddof=1)),
            "corr_a_b_in_subset": float(np.corrcoef(za[mask], zb[mask])[0, 1]),
            "corr_latent_a_in_subset": float(
                np.corrcoef(latent[mask], za[mask])[0, 1]),
        })
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]} | {
        "noise_sd_z": noise_sd, "n_variants": n, "reps": reps,
        "share_of_reps_with_negative_formula": float(np.mean(
            [r["formula_reliability"] < 0 for r in rows]))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    noise = _noise_level()
    summary = {"seed": SEED, "top_frac": TOP, "k": K,
               "selfinclusion": selfinclusion(noise),
               "decile_reliability": decile_reliability(noise)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
