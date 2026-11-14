"""Is the top-k failure epistasis? The one candidate that can be tested.

README ("the ceiling is the assay") leaves three candidates for the model's
mediocre top-10 recall: measurement noise, the heterogeneity of "top 10%", and
epistasis. The first two were tested and the noise won; epistasis was left as
untested on the grounds that the replicate subset is almost all
single-substitution assays. That framing is about the *replicate pairs*, not
about the data: the full local ProteinGym copy holds 217 assays, 69 of them
with multi-mutants, 65 with at least 100 singles and 100 multi-mutants, and
1.77 million multi-mutant rows. The model can be tested on them.

The test
--------

`ESMScorer.score_variants` sums independent per-site log-odds:

    score(m1:m2) = dlogp(m1) + dlogp(m2)

That is a *model of independence*. If intra-protein epistasis is real and the
top of a DMS assay is where it bites, this predictor must rank genuine double
and triple mutants worse than it ranks single mutants on the same protein, in
the same units, against the same measured readout. Nothing about the model is
tuned here -- one masked-marginal table per assay is computed once and both
strata are scored from it, which is exactly the independent-site model.

Per assay and per stratum (single / all multi / doubles / triples) it reports:

  * Spearman against DMS_score within the stratum, with a bootstrap CI
    resampled by POSITION (variants cluster by position, so a variant-level
    interval is too narrow; every interval and gap here uses the position
    bootstrap of variant_effect.resampling)
  * top-decile recall: the share of the measured top decile that appears in
    the model's top decile, so strata of 1,000 and 500,000 are comparable
  * top-10 recall, the repo's headline convention, for continuity
  * n, and the stratum's mutation-count composition

Confounds that are reported rather than hidden:

  * summed log-odds carry a variance penalty in the number of substitutions
    (`sd` grows like sqrt(k) if the sites were independent), so the multi
    strata are reported split by k as well as pooled;
  * an assay's multi-mutants may not be the same quantity as its singles (a
    combinatorial library scored for binding, a singles library scored for
    abundance). The reference table cannot say that per library, so the output
    carries (a) a boolean different_quantity_flag that is True only if the
    assay belongs to a pair on the repo's DIFFERENT_QUANTITY list
    (variant_effect.quantity), and (b) the reference table's
    coarse_selection_type, so a reader can check the claim. The mutation-count
    composition of each stratum is in n_sub_composition (it used to be written
    into the flag column by mistake);
  * the non-additive residual (additive_residual) is NOT the epistatic term: it
    contains the measurement noise of the double and of both singles, and has
    no replicate-based way to subtract it (see that function);
  * a sample of the largest strata is scored (cap, default 2000 per class)
    because 1.77M variants is not a CPU budget; the sample is seeded and
    stratified by mutation count, and the position table it needs is far
    smaller than the row count.

Cost is one masked forward pass per *mutated position*, not per variant, so a
500k-row assay costs the same as a 2k-row one over the same positions. The
table is cached to disk per assay and filled in chunks, so a long run can be
resumed rather than restarted.

Run:
    python scripts/run_epistasis_test.py                 # default assay set
    python scripts/run_epistasis_test.py --assay SPG1_STRSG_Olson_2014
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from variant_effect import data, evaluate, quantity, resampling  # noqa: E402
from variant_effect.strata import range_matched_rho  # noqa: E402,F401
from variant_effect.mutations import parse_variant  # noqa: E402

OUT = ROOT / "results" / "epistasis"
CACHE = OUT / "_cache"

# Largest multi-mutant assays that are also cheap enough on CPU: the top five
# by multi-mutant row count. CAPSD_AAV2S_Sinai_2021 (41,796 multi, 735 aa) is
# sixth and is left out on cost -- its masked table needs a pass per position
# over a 735-residue protein, which does not fit the 10-minute command budget.
# It is listed in the manifest so the exclusion is visible rather than silent.
ASSAYS = [
    "SPG1_STRSG_Olson_2014",        # 448 aa,  1,045 single / 535,917 multi
    "HIS7_YEAST_Pokusaeva_2019",    # 220 aa,    168 single / 495,969 multi
    "PHOT_CHLRE_Chen_2023",         # 118 aa,  2,122 single / 165,407 multi
    "GRB2_HUMAN_Faure_2021",        # 217 aa,  1,034 single /  62,332 multi
    "GFP_AEQVI_Sarkisyan_2016",     # 238 aa,  1,084 single /  50,630 multi
]
EXCLUDED = ["CAPSD_AAV2S_Sinai_2021"]

AA20 = "ACDEFGHIKLMNPQRSTVWY"
ORDER = {a: i for i, a in enumerate(AA20)}
TOP_FRAC = 0.1
N_BOOT = 1000


# ----------------------------------------------------------------------
# strata
# ----------------------------------------------------------------------

def n_substitutions(mutant):
    return str(mutant).count(":") + 1


def load_strata(assay_id, cap=2000, seed=0, single_cap=None):
    """Split one assay into singles and multi-mutants, one frame per stratum.

    The multi sample is drawn *within* each mutation-count class, because a
    plain random sample of HIS7's 496k rows is 60% heptamutants: doubles and
    triples would be nearly absent from a 2,000-row sample even though they
    are the strata the independent-site claim is actually about.
    """
    df = pd.read_csv(data.ASSAY_DIR / f"{assay_id}.csv",
                     usecols=["mutant", "DMS_score"])
    df = df[np.isfinite(df["DMS_score"])].reset_index(drop=True)
    # A mutant listed twice keeps its FIRST row; the count is recorded on every
    # returned frame (attrs["duplicates_dropped"]) instead of being silent.
    n_dup = int(df.duplicated("mutant").sum())
    df = df.drop_duplicates("mutant", keep="first").reset_index(drop=True)
    df["n_sub"] = df["mutant"].map(n_substitutions)

    singles = df[df["n_sub"] == 1]
    if single_cap:
        singles = singles.sample(min(len(singles), single_cap), random_state=seed)

    multis = df[df["n_sub"] > 1]
    rng = np.random.default_rng(seed)
    keep = []
    for k, grp in multis.groupby("n_sub"):
        idx = grp.index.to_numpy()
        if len(idx) > cap:
            idx = rng.choice(idx, cap, replace=False)
        keep.append(pd.Series(idx))
    if keep:
        multis = df.loc[pd.concat(keep).to_numpy()]

    strata = {"single": singles.reset_index(drop=True),
              "multi": multis.reset_index(drop=True)}
    for k in (2, 3):
        s = multis[multis["n_sub"] == k]
        if len(s) >= 50:
            strata[f"nsub{k}"] = s.reset_index(drop=True)
    # high-order stratum: this is where an independent-site sum must fail
    # hardest if intra-protein epistasis is real, because the score is a sum
    # of that many independent log-odds and the true fitness is not.
    deep = multis[multis["n_sub"] >= 5]
    if len(deep) >= 50:
        strata["nsub5plus"] = deep.reset_index(drop=True)
    for frame in strata.values():
        frame.attrs["duplicates_dropped"] = n_dup
    return strata, df


def stratum_positions(strata):
    """Every 0-indexed position touched by any variant in any stratum."""
    pos = set()
    for df in strata.values():
        for m in df["mutant"]:
            for s in parse_variant(m):
                pos.add(s.idx)
    return np.array(sorted(pos), dtype=np.int64)


# ----------------------------------------------------------------------
# masked-marginal table, cached and resumable
# ----------------------------------------------------------------------

def masked_table(scorer, sequence, positions, cache_path, chunk=64):
    """(len(positions), 20) log-probabilities, filled chunk by chunk on disk.

    A full pass over a 448-residue protein is about 1.4 s per position on CPU,
    so one assay can take ten minutes and a collective run longer than any
    single command should. Storing the table as it fills means an interrupted
    run resumes at the next position instead of starting over.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if cache_path.exists():
        with np.load(cache_path) as z:
            done = {int(p): z["table"][i] for i, p in enumerate(z["positions"])}

    todo = [int(p) for p in positions if int(p) not in done]
    for start in range(0, len(todo), chunk):
        block = todo[start:start + chunk]
        tab = scorer.masked_marginal_logprobs(sequence, block, batch_size=8)
        for p in block:
            done[p] = tab[p]
        arr = np.stack([done[int(p)] for p in positions if int(p) in done])
        saved = np.array([int(p) for p in positions if int(p) in done])
        np.savez(cache_path, positions=saved, table=arr)
        print(f"    positions {min(start + chunk, len(todo))}/{len(todo)}",
              flush=True)

    return {int(p): done[int(p)] for p in positions}


def score_from_table(table, variants):
    """Independent-site score: sum of per-site log-odds. No model call."""
    out = np.zeros(len(variants), dtype=np.float64)
    for i, m in enumerate(variants):
        total = 0.0
        for s in parse_variant(m):
            row = table[s.idx]
            total += row[ORDER[s.mut]] - row[ORDER[s.wt]]
        out[i] = total
    return out


# ----------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------

def top_decile_fraction(pred, truth, frac=TOP_FRAC):
    """Overlap of the model's top decile with the measured top decile,
    as a fraction of the decile -- so an n=900 stratum and an n=200,000
    stratum are on the same scale. 1.0 is perfect, 0.1 is chance."""
    ok = np.isfinite(pred) & np.isfinite(truth)
    p, t = pred[ok], truth[ok]
    m = int(max(1, round(len(p) * frac)))
    if m < 5:
        return np.nan
    cut = np.quantile(t, 1 - frac)
    picked = np.argsort(-p)[:m]
    return float(np.mean(t[picked] >= cut))


def additive_residual(full, seed=0, cap=20000):
    """How much of a double mutant's measured fitness the additive model misses.

    A data-only measurement, no model involved. For every double mutant whose
    two constituent singles were also assayed, the additive prediction is the
    sum of the two singles' z-scored fitnesses. The residual (measured minus
    additive) is the NON-ADDITIVE RESIDUAL.

    It is NOT the epistatic term. The residual is

        epistasis + e_double - e_single1 - e_single2

    where each e is the measurement noise of that variant. The variance of the
    residual is therefore an UPPER bound on the epistasis variance, and it can
    exceed the total variance of the doubles when the singles are noisy: HIS7's
    168 singles (singles truth sd 0.19, model rho 0.028) give a residual sd
    larger than the doubles' total sd, a share above 1, which no variance
    share of a real component can be. ProteinGym carries no per-variant error
    for these assays, so the noise cannot be deconvolved; the quantity is
    reported under its honest name and residual_exceeds_total flags the cases
    where noise visibly dominates.

    It matters because the two questions are different: does non-additivity
    exist in the data (this function, at best as an upper bound), and does it
    degrade the model's ranking (the strata comparison). A protein can have
    large non-additive residuals and still be ranked fine if they are not
    concentrated in the top decile.
    """
    singles = full[full["n_sub"] == 1]
    sd = float(np.nanstd(singles["DMS_score"], ddof=1))
    mu = float(singles["DMS_score"].mean())
    if not np.isfinite(sd) or sd <= 0:
        return None
    z = {(s.wt, s.pos, s.mut): (v - mu) / sd
         for m, v in zip(singles["mutant"], singles["DMS_score"])
         for s in parse_variant(m)}

    doubles = full[full["n_sub"] == 2]
    if len(doubles) > cap:
        doubles = doubles.sample(cap, random_state=seed)
    add, meas = [], []
    for m, v in zip(doubles["mutant"], doubles["DMS_score"]):
        subs = parse_variant(m)
        try:
            parts = [z[(s.wt, s.pos, s.mut)] for s in subs]
        except KeyError:
            continue
        add.append(sum(parts))
        meas.append((v - mu) / sd)
    if len(meas) < 50:
        return None
    add = np.asarray(add)
    meas = np.asarray(meas)
    resid = meas - add
    var_meas = float(np.nanvar(meas, ddof=1))
    share = (float(np.nanvar(resid, ddof=1)) / var_meas
             if var_meas > 0 else np.nan)
    return {
        "n_doubles_with_singles": int(len(meas)),
        "additive_spearman": evaluate.spearman(add, meas),
        "additive_r2": (float(np.corrcoef(add, meas)[0, 1] ** 2)
                        if np.nanstd(add) > 0 else np.nan),
        "nonadditive_residual_sd_z": float(np.nanstd(resid, ddof=1)),
        "double_total_sd_z": float(np.sqrt(var_meas)),
        "nonadditive_residual_var_share": share,
        "residual_exceeds_total": bool(np.isfinite(share) and share > 1.0),
    }


def stratum_metrics(pred, truth, mutants=None):
    """Metrics for one stratum. RESAMPLING UNIT of the Spearman interval: the
    POSITION when mutants (the stratum's mutant strings, aligned to pred) are
    given; the VARIANT otherwise, which is too narrow for clustered data and
    kept only for callers that have no mutant strings."""
    groups = (None if mutants is None else
              [resampling.substitution_positions(m) for m in mutants])
    lo, hi = evaluate.spearman_ci(pred, truth, n_boot=N_BOOT, groups=groups)
    ok = np.isfinite(pred) & np.isfinite(truth)
    p, t = pred[ok], truth[ok]
    # calibration: slope of truth on the independent-site score. If the summed
    # log-odds inflate with the number of substitutions, the slope shrinks.
    slope = np.nan
    r2 = np.nan
    if len(p) > 10 and np.nanstd(p) > 0:
        b, a = np.polyfit(p, t, 1)
        resid = t - (a + b * p)
        slope = float(b)
        ss = np.sum((t - t.mean()) ** 2)
        r2 = float(1 - np.sum(resid ** 2) / ss) if ss > 0 else np.nan
    return {
        "n": int(ok.sum()),
        "spearman": evaluate.spearman(pred, truth),
        "spearman_lo": lo,
        "spearman_hi": hi,
        "top_decile_recall": top_decile_fraction(pred, truth),
        "top10_recall": evaluate.top_k_recall(pred, truth, k=10),
        "score_sd": float(np.nanstd(pred, ddof=1)),
        "truth_sd": float(np.nanstd(truth, ddof=1)),
        "truth_p05": float(np.quantile(t, 0.05)),
        "truth_p95": float(np.quantile(t, 0.95)),
        "calibration_slope": slope,
        "calibration_r2": r2,
    }


def gap_ci(pred_s, truth_s, pred_m, truth_m, mutants_s=None, mutants_m=None,
           n_boot=None, seed=0, metric=None):
    """CI on (metric_single - metric_multi), default Spearman.

    RESAMPLING UNIT: the POSITION when mutants_s and mutants_m are given (one
    draw shared by both strata, variants weighted by the product of their
    positions' multiplicities; resampling.gap_bootstrap). Without them the unit
    is the VARIANT, each stratum resampled independently, which understates the
    uncertainty on clustered data (RASK doubles: about 3x) and is kept only for
    synthetic tests. Returns (lo, hi).
    """
    if min(len(pred_s), len(pred_m)) < 10:
        return (np.nan, np.nan)
    fn = metric or evaluate.spearman
    _, lo, hi, _ = resampling.gap_bootstrap(
        fn, (pred_s, truth_s), (pred_m, truth_m), mutants_s, mutants_m,
        n_boot=N_BOOT if n_boot is None else n_boot, seed=seed, min_n=10)
    return (lo, hi)


def range_matched_gap_ci(pred_s, truth_s, pred_m, truth_m, mutants_s, mutants_m,
                         n_boot=None, seed=0):
    """CI on the range-matched gap, window recomputed in every replicate,
    resampled by position. Returns (lo, hi)."""
    from variant_effect.strata import range_matched_gap
    _, lo, hi, _ = resampling.gap_bootstrap(
        range_matched_gap, (pred_s, truth_s), (pred_m, truth_m),
        mutants_s, mutants_m, n_boot=N_BOOT if n_boot is None else n_boot,
        seed=seed, min_n=50, joint=True)
    return (lo, hi)


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def run_assay(scorer, assay_id, reference, cap, seed):
    meta = data.assay_metadata(assay_id, reference)
    sequence = reference.loc[reference["DMS_id"] == assay_id,
                             "target_seq"].iloc[0]

    strata, full = load_strata(assay_id, cap=cap, seed=seed)
    positions = stratum_positions(strata)
    print(f"  {assay_id}: {len(positions)} positions, "
          f"single={len(strata['single'])} multi={len(strata['multi'])}",
          flush=True)

    table = masked_table(scorer, sequence, positions,
                         CACHE / f"{assay_id}.npz")

    rows = []
    per_stratum = {}
    curve = []
    epi = additive_residual(full, seed=seed)
    if epi:
        print(f"    additive model on measured singles: "
              f"rho={epi['additive_spearman']:+.3f} "
              f"non-additive residual var share="
              f"{epi['nonadditive_residual_var_share']:.3f} "
              f"(n={epi['n_doubles_with_singles']})", flush=True)
    different_quantity_flag = quantity.in_different_quantity_pair(assay_id)
    scored = []
    for name, df in strata.items():
        pred = score_from_table(table, df["mutant"].tolist())
        truth = df["DMS_score"].to_numpy(dtype=float)
        per_stratum[name] = (pred, truth)
        if name in ("single", "multi"):
            scored.append(pd.DataFrame({
                "mutant": df["mutant"].to_numpy(), "n_sub": df["n_sub"].to_numpy(),
                "stratum": name, "DMS_score": truth, "model": pred}))
        m = stratum_metrics(pred, truth, mutants=df["mutant"].tolist())
        m.update({
            "assay": assay_id,
            "stratum": name,
            "taxon": meta["taxon"],
            "selection": meta["selection"],
            "seq_len": meta["seq_len"],
            **(epi or {}),
            "n_sub_composition": ", ".join(
                f"{int(k)}sub:{int(v)}" for k, v in
                df["n_sub"].value_counts().sort_index().items()),
            "different_quantity_flag": bool(different_quantity_flag),
            "reference_coarse_selection_type": str(
                reference.loc[reference["DMS_id"] == assay_id,
                              "coarse_selection_type"].iloc[0]),
            "score_truth_rho": evaluate.spearman(pred, truth),
            "assay_total_rows": int(len(full)),
            "assay_single_rows": int((full["n_sub"] == 1).sum()),
            "assay_multi_rows": int((full["n_sub"] > 1).sum()),
        })
        rows.append(m)

    # rho as a function of the number of substitutions: the mechanism test.
    # Independence predicts a monotone decline in k; nothing else does.
    for k, grp in (strata["multi"].groupby("n_sub")
                   if "multi" in strata else []):
        if len(grp) < 100:
            continue
        p = score_from_table(table, grp["mutant"].tolist())
        t = grp["DMS_score"].to_numpy(dtype=float)
        curve.append({
            "assay": assay_id, "n_sub": int(k), "n": int(len(grp)),
            "spearman": evaluate.spearman(p, t),
            "top_decile_recall": top_decile_fraction(p, t),
            "score_sd": float(np.nanstd(p, ddof=1)),
            "truth_sd": float(np.nanstd(t, ddof=1)),
        })
    if curve:
        pd.DataFrame(curve).to_csv(OUT / f"rho_vs_nsub_{assay_id}.csv",
                                   index=False)

    # position-matched control: the multi-mutants of these combinatorial
    # libraries sit on a handful of positions, while the single library may
    # span hundreds. A single-stratum restricted to the same positions is the
    # only comparison that holds the sequence neighbourhood fixed.
    multi_pos = set()
    for m in strata["multi"]["mutant"]:
        for s in parse_variant(m):
            multi_pos.add(s.wt + str(s.pos) + "_" + str(s.idx))
    idxs = {int(p.split("_")[1]) for p in multi_pos}
    singles_df = strata["single"]
    pmask = singles_df["mutant"].map(
        lambda m: all(s.idx in idxs for s in parse_variant(m)))
    pos_matched = singles_df[pmask]
    pm_metrics = None
    if len(pos_matched) >= 50:
        pm_pred = score_from_table(table, pos_matched["mutant"].tolist())
        pm_truth = pos_matched["DMS_score"].to_numpy(dtype=float)
        pm_metrics = stratum_metrics(
            pm_pred, pm_truth, mutants=pos_matched["mutant"].tolist())
        print(f"    position-matched singles: n={pm_metrics['n']} "
              f"rho={pm_metrics['spearman']:+.3f} "
              f"rec={pm_metrics['top_decile_recall']:.3f}", flush=True)

    # the headline gap: singles vs all multi-mutants
    s_pred, s_truth = per_stratum["single"]
    m_pred, m_truth = per_stratum["multi"]
    s_mut = strata["single"]["mutant"].tolist()
    m_mut = strata["multi"]["mutant"].tolist()
    n_pos_clusters = resampling.ClusterIndex.from_mutants(
        s_mut + m_mut).n_clusters
    lo, hi = gap_ci(s_pred, s_truth, m_pred, m_truth, s_mut, m_mut, seed=seed)
    lo_rec, hi_rec = gap_ci(s_pred, s_truth, m_pred, m_truth, s_mut, m_mut,
                            seed=seed, metric=top_decile_fraction)
    rm_s, rm_m, n_s, n_m = range_matched_rho(s_pred, s_truth, m_pred, m_truth)
    lo_rm, hi_rm = range_matched_gap_ci(s_pred, s_truth, m_pred, m_truth,
                                        s_mut, m_mut, seed=seed)
    lo_pm = np.nan
    hi_pm = np.nan
    if pm_metrics is not None:
        lo_pm, hi_pm = gap_ci(pm_pred, pm_truth, m_pred, m_truth,
                              pos_matched["mutant"].tolist(), m_mut, seed=seed)
    # doubles-only comparison: singles vs the nsub2 stratum alone. This is the
    # like-for-like contrast; the pooled multi stratum mixes mutation counts
    # (HIS7 and GFP carry up to 13 substitutions) with the singles' positions.
    lo_d2 = hi_d2 = gap_d2 = np.nan
    if "nsub2" in per_stratum:
        d_pred, d_truth = per_stratum["nsub2"]
        gap_d2 = (evaluate.spearman(s_pred, s_truth)
                  - evaluate.spearman(d_pred, d_truth))
        lo_d2, hi_d2 = gap_ci(s_pred, s_truth, d_pred, d_truth, s_mut,
                              strata["nsub2"]["mutant"].tolist(), seed=seed)
    for r in rows:
        r["gap_single_minus_multi"] = (rows[0]["spearman"] - rows[1]["spearman"])
        r["gap_lo"] = lo
        r["gap_hi"] = hi
        r["gap_resampling_unit"] = "position (joint draw, both strata)"
        r["n_position_clusters"] = int(n_pos_clusters)
        r["gap_top_decile"] = (rows[0]["top_decile_recall"]
                               - rows[1]["top_decile_recall"])
        r["gap_top_decile_lo"] = lo_rec
        r["gap_top_decile_hi"] = hi_rec
        r["gap_single_minus_nsub2"] = gap_d2
        r["gap_single_minus_nsub2_lo"] = lo_d2
        r["gap_single_minus_nsub2_hi"] = hi_d2
        r["range_matched_gap_lo"] = lo_rm
        r["range_matched_gap_hi"] = hi_rm
        r["range_matched_rho_single"] = rm_s
        r["range_matched_rho_multi"] = rm_m
        r["range_matched_gap"] = rm_s - rm_m
        r["range_matched_n_single"] = n_s
        r["range_matched_n_multi"] = n_m
        r["pos_matched_n"] = pm_metrics["n"] if pm_metrics else 0
        # the control restricts the singles to positions the multi-mutants
        # touch; it only does anything if that drops some singles. When every
        # single sits on a multi-mutant position the "control" is the
        # unrestricted singles stratum again.
        r["pos_matched_binds"] = bool(len(pos_matched)
                                      < len(singles_df))
        r["n_single_positions"] = int(len(
            {i for m in s_mut for i in
             (x.idx for x in parse_variant(m))}))
        r["n_multi_positions"] = int(len(idxs))
        r["pos_matched_rho"] = (pm_metrics["spearman"] if pm_metrics else np.nan)
        r["pos_matched_top_decile"] = (
            pm_metrics["top_decile_recall"] if pm_metrics else np.nan)
        r["pos_matched_gap_lo"] = lo_pm
        r["pos_matched_gap_hi"] = hi_pm
    if scored:
        pd.concat(scored, ignore_index=True).to_csv(
            OUT / f"scored_{assay_id}.csv.gz", index=False)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assay", action="append", default=None)
    ap.add_argument("--cap", type=int, default=2000,
                    help="max variants scored per mutation-count class")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--refresh", action="store_true",
                    help="recompute assays whose per_stratum CSV exists")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    assays = args.assay or ASSAYS
    reference = data.reference_table()

    frames = []
    for assay_id in assays:
        out_csv = OUT / f"per_stratum_{assay_id}.csv"
        if out_csv.exists() and not args.refresh:
            print(f"{assay_id}: cached", flush=True)
            frames.append(pd.read_csv(out_csv))
            continue
        t0 = time.time()
        from variant_effect.scoring import ESMScorer
        scorer = ESMScorer(device="cpu")
        df = run_assay(scorer, assay_id, reference, args.cap, args.seed)
        df["seconds"] = round(time.time() - t0, 1)
        df.to_csv(out_csv, index=False)
        frames.append(df)
        print(df[["assay", "stratum", "n", "spearman", "top_decile_recall",
                  "top10_recall"]].to_string(index=False), flush=True)

    per = pd.concat(frames, ignore_index=True)
    per.to_csv(OUT / "per_stratum_all.csv", index=False)

    # pooled across assays: equal weight per assay, single vs multi
    pooled = []
    for stratum in ("single", "multi", "nsub2", "nsub3"):
        sub = per[per["stratum"] == stratum]
        if sub.empty:
            continue
        pooled.append({
            "stratum": stratum,
            "n_assays": int(len(sub)),
            "mean_spearman": float(sub["spearman"].mean()),
            "median_spearman": float(sub["spearman"].median()),
            "sd_spearman": float(sub["spearman"].std(ddof=1)),
            "mean_top_decile_recall": float(sub["top_decile_recall"].mean()),
            "median_top_decile_recall": float(sub["top_decile_recall"].median()),
            "mean_top10_recall": float(sub["top10_recall"].mean()),
            "total_n": int(sub["n"].sum()),
        })
    pooled_df = pd.DataFrame(pooled)
    pooled_df.to_csv(OUT / "pooled.csv", index=False)

    # paired gap across assays
    piv = per.pivot_table(index="assay", columns="stratum",
                          values=["spearman", "top_decile_recall"])
    gaps = pd.DataFrame({
        "assay": piv.index,
        "rho_single": piv[("spearman", "single")].to_numpy(),
        "rho_multi": piv[("spearman", "multi")].to_numpy(),
        "rec_single": piv[("top_decile_recall", "single")].to_numpy(),
        "rec_multi": piv[("top_decile_recall", "multi")].to_numpy(),
    })
    gaps["gap_rho"] = gaps["rho_single"] - gaps["rho_multi"]
    gaps["gap_rec"] = gaps["rec_single"] - gaps["rec_multi"]
    ctrl = (per.drop_duplicates("assay")[[
        "assay", "gap_lo", "gap_hi", "n_position_clusters",
        "gap_top_decile_lo", "gap_top_decile_hi",
        "gap_single_minus_nsub2", "gap_single_minus_nsub2_lo",
        "gap_single_minus_nsub2_hi",
        "range_matched_gap", "range_matched_gap_lo", "range_matched_gap_hi",
        "range_matched_rho_single", "range_matched_rho_multi",
        "pos_matched_rho", "pos_matched_top_decile", "pos_matched_n",
        "pos_matched_binds", "n_single_positions", "n_multi_positions",
        "different_quantity_flag", "reference_coarse_selection_type",
        "nonadditive_residual_var_share", "residual_exceeds_total",
    ]].set_index("assay"))
    gaps = gaps.set_index("assay").join(ctrl).reset_index()
    gaps["pos_matched_gap_rho"] = gaps["pos_matched_rho"] - gaps["rho_multi"]
    gaps.to_csv(OUT / "paired_gaps.csv", index=False)

    # The 5+ substitution stratum exists only for assays whose multi-mutants
    # go that deep (not all five). Compare it with the singles and doubles of
    # THE SAME assays, not with five-assay means.
    deep = per[per["stratum"] == "nsub5plus"]["assay"].tolist()
    if deep:
        piv2 = per.pivot_table(index="assay", columns="stratum",
                               values="spearman")
        like = piv2.loc[deep, [c for c in ("single", "nsub2", "multi",
                                           "nsub5plus") if c in piv2.columns]]
        like.loc["mean over these assays"] = like.mean()
        like.insert(0, "n_assays", len(deep))
        like.to_csv(OUT / "nsub5plus_like_for_like.csv")
        print("\n5+ substitution stratum, like-for-like (assays that have it: "
              f"{len(deep)} of {per['assay'].nunique()})")
        print(like.to_string())

    manifest = {
        "assays": list(assays),
        "excluded_on_cost": EXCLUDED,
        "cap_per_mutation_class": args.cap,
        "seed": args.seed,
        "top_frac": TOP_FRAC,
        "model": "facebook/esm2_t33_650M_UR50D",
        "scheme": "masked marginals, independent-site sum",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print("\npooled")
    print(pooled_df.to_string(index=False))
    print("\npaired gaps")
    print(gaps.to_string(index=False))
    if len(gaps) > 1:
        t, p = stats.ttest_1samp(gaps["gap_rho"].dropna(), 0.0)
        print(f"\nmean gap_rho = {gaps['gap_rho'].mean():+.4f} "
              f"(paired t={t:+.2f}, p={p:.3g}, n={len(gaps)} assays)")
        t, p = stats.ttest_1samp(gaps["gap_rec"].dropna(), 0.0)
        print(f"mean gap_top_decile = {gaps['gap_rec'].mean():+.4f} "
              f"(paired t={t:+.2f}, p={p:.3g})")
        c = gaps[["range_matched_gap", "pos_matched_gap_rho"]].dropna()
        if len(c):
            print(f"range-matched mean gap = {c['range_matched_gap'].mean():+.4f}; "
                  f"position-matched mean gap = "
                  f"{c['pos_matched_gap_rho'].mean():+.4f} (n={len(c)})")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
