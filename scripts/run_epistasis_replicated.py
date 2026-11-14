"""Does the independent-site model degrade on double mutants, against a pooled
two-assay target?

The earlier test (scripts/run_epistasis_test.py) compared singles with
multi-mutants inside a single assay. That comparison has a confound it cannot
remove: in most of those assays the singles library and the combinatorial
library are different measurement sets, so a rank difference between the two
strata can be a difference in the target, not in the model.

This script uses a pair of assays of the same protein as the target, scores
singles and doubles of the SAME shared variant list with the same model, and
compares the two strata.

What the pair is -- read this before the numbers
------------------------------------------------

Pooling two assays and calling their agreement a reliability presumes the two
assays are PARALLEL measurements of one underlying score. Neither pair here
meets that, and the script now says so in every artefact (see
variant_effect.quantity.pair_status):

  * RASK_HUMAN: Weng 2022 abundance vs binding-DARPin_K55. The repo's own
    hand-curated DIFFERENT_QUANTITY list flags this pair (reference table:
    coarse_selection_type Expression vs Binding). Abundance and binding are
    different quantities; the model's ranks track binding far better than
    abundance. The pair is NOT a replicate pair, the Spearman-Brown
    "pooled reliability" is not computed for it, and the pooled target is a
    pseudo-quantity.
  * SPG1_STRSG: Olson 2014 (lnW) vs Wu 2016 (Fitness). Same biology, but the
    two scales are related by a nonlinear monotone map (rank agreement is very
    high; Pearson of the z-scores is lower), so a z-score pool averages unlike
    scales. Spearman-Brown is withheld here too.

Design
------

1. The two assays are joined on the CANONICAL mutant string: substitutions
   sorted by position (resampling.canonical_mutant). The raw string drops every
   double that one lab wrote as D266W:V265S and the other as V265S:D266W (709
   SPG1 doubles at the time of the review); the join report in the manifest
   records exact vs canonical counts so the difference is visible.
2. Four targets are always reported side by side: assay A alone, assay B alone,
   the z-score pool, and the rank-normalised pool (pooled_truth.pool_rank,
   valid under monotone nonlinear scale differences). The rank pool is the
   designated primary target because it needs the fewest assumptions; none of
   the four is chosen after looking at results, and a conclusion is only as
   strong as it is across all four.
3. The same ESM-2 independent-site scorer scores every shared variant (the sum
   of per-site log-odds; one masked-marginal table per mutated position).
4. Spearman and top-decile recall are computed per stratum. The gap
   (singles - doubles) comes with three intervals-worth of care:
     * RESAMPLING UNIT = POSITION. Variants cluster by position (RASK doubles:
       20,124 variants on 185 positions), so the bootstrap draws positions and
       weights each variant by the product of its positions' multiplicities,
       with ONE draw shared by both strata (resampling.gap_bootstrap). A
       variant-level bootstrap is kept only as an explicit fallback for tests.
     * A RANGE-MATCHED gap is reported beside the raw gap: both strata are
       trimmed to the fitness window they share, because the doubles' truth has
       a wider spread than the singles' and a rank correlation over a wider
       spread is mechanically larger.
     * The gap is reported against each assay alone, so a pooled number cannot
       hide that the two assays disagree about it.

The prediction under the epistasis hypothesis is a positive gap (doubles
predicted worse). A gap whose interval spans zero is "not distinguishable",
never "no effect".

Run:
    python scripts/run_epistasis_replicated.py
    python scripts/run_epistasis_replicated.py --pair rask
    python scripts/run_epistasis_replicated.py --pooled-only   # skip the model
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from variant_effect import data, evaluate, quantity, resampling  # noqa: E402
from variant_effect import pooled_truth as pt  # noqa: E402
from variant_effect import strata  # noqa: E402
from variant_effect.mutations import parse_variant  # noqa: E402

OUT = ROOT / "results" / "epistasis_replicated"
CACHE = OUT / "_cache"

AA20 = "ACDEFGHIKLMNPQRSTVWY"
ORDER = {a: i for i, a in enumerate(AA20)}
TOP_FRAC = 0.1
N_BOOT = 1000
RESAMPLING_UNIT = ("position: positions drawn with replacement, variants "
                   "weighted by the product of their positions' "
                   "multiplicities, one draw shared by both strata")

# The target the verdict and the README headline read. Chosen before the run
# because it needs the fewest assumptions (no affine-scale assumption, unlike
# pooled_z); every other target is reported next to it.
PRIMARY_TARGET = "pooled_rank"
TARGETS = ("pooled_rank", "pooled_z", "assay_a", "assay_b")

# Two assays of the same protein with a shared variant library. Verified counts
# are re-derived at run time and written to the manifest rather than trusted.
# role says why the pair is here, not that it is a replicate pair: neither is
# a clean parallel pair (see quantity.pair_status, recorded in the manifest).
PAIRS = {
    "rask": {
        "protein": "RASK_HUMAN",
        "assay_a": "RASK_HUMAN_Weng_2022_abundance",
        "assay_b": "RASK_HUMAN_Weng_2022_binding-DARPin_K55",
        "role": "larger pair; DIFFERENT quantity (abundance vs binding)",
    },
    "spg1": {
        "protein": "SPG1_STRSG",
        "assay_a": "SPG1_STRSG_Olson_2014",
        "assay_b": "SPG1_STRSG_Wu_2016",
        "role": "smaller pair; same biology, different scale (lnW vs Fitness)",
    },
}

# F7YBW8_MESOW_Aakre_2015 / _Ding_2023 were checked and share zero variants on
# the exact string, so there is no pooled target for that protein.
SKIPPED = {
    "F7YBW8_MESOW": "zero shared variants between the two assays on disk",
}


# ----------------------------------------------------------------------
# loading and pooling
# ----------------------------------------------------------------------

def n_substitutions(mutant):
    return str(mutant).count(":") + 1


def _canonical_or_raw(mutant):
    """Canonical spelling; an unparseable notation keeps its raw string and is
    counted by the caller instead of crashing a 500k-row load."""
    try:
        return resampling.canonical_mutant(mutant), False
    except ValueError:
        return str(mutant), True


def _load_assay(assay):
    """One assay's finite singles and doubles, keyed by canonical mutant.

    Returns (frame, stats). Duplicates (after canonicalisation) keep the FIRST
    row and are counted in stats rather than dropped silently; so are
    unparseable notations (kept under their raw string).
    """
    df = pd.read_csv(data.ASSAY_DIR / f"{assay}.csv",
                     usecols=["mutant", "DMS_score"])
    n_rows = len(df)
    df = df[np.isfinite(df["DMS_score"])]
    df = df[df["mutant"].map(n_substitutions).isin((1, 2))].copy()
    df["raw_mutant"] = df["mutant"]
    keyed = [_canonical_or_raw(m) for m in df["mutant"]]
    df["mutant"] = [k for k, _ in keyed]
    n_unparsed = int(sum(bad for _, bad in keyed))
    n_dup = int(df.duplicated("mutant").sum())
    df = df.drop_duplicates("mutant", keep="first")
    return df, {"rows_read": int(n_rows), "singles_doubles_finite": int(len(df)
                + n_dup), "duplicates_dropped_first_kept": n_dup,
                "unparseable_notations": n_unparsed}


def join_report(assay_a, assay_b):
    """Exact-string vs canonical-order intersection of two assays, and the
    per-assay load statistics. The exact count is what a raw join would keep;
    the difference is the number of variants a raw join silently drops."""
    (a, sa), (b, sb) = _load_assay(assay_a), _load_assay(assay_b)
    exact = set(a["raw_mutant"]) & set(b["raw_mutant"])
    canon = set(a["mutant"]) & set(b["mutant"])
    return {
        "exact_string_intersection": int(len(exact)),
        "canonical_intersection": int(len(canon)),
        "dropped_by_exact_string_join": int(len(canon) - len(exact)),
        "exact_is_subset_of_canonical": bool(
            {resampling.canonical_mutant(m) for m in exact} <= canon),
        assay_a: sa,
        assay_b: sb,
    }


def load_pair(assay_a, assay_b):
    """Intersection of two assays on the CANONICAL mutant string.

    The mutant column of the result is the canonical spelling (substitutions
    sorted by position). Joining on the raw string is NOT equivalent: Olson
    2014 writes D266W:V265S where Wu 2016 writes V265S:D266W (Olson is
    position-ordered for 92% of its SPG1 rows, Wu for all of them), so an
    exact-string join silently drops those doubles. join_report() returns both
    counts; tests assert the canonical set is a superset of the exact set and
    that on a reversed-order fixture the two differ.

    Only singles and doubles are kept. A variant duplicated within one assay
    (after canonicalisation) keeps its first row; the count is in
    join_report(). The exact one-to-one validation is on the canonical key.
    """
    a, _ = _load_assay(assay_a)
    b, _ = _load_assay(assay_b)
    a = a[["mutant", "DMS_score"]].rename(
        columns={"DMS_score": f"score_{assay_a}"})
    b = b[["mutant", "DMS_score"]].rename(
        columns={"DMS_score": f"score_{assay_b}"})
    merged = a.merge(b, on="mutant", how="inner", validate="one_to_one")
    merged["n_sub"] = merged["mutant"].map(n_substitutions)
    return merged[merged["n_sub"].isin((1, 2))].reset_index(drop=True)


def pooled_frame(merged, assay_a, assay_b):
    """Attach every target: the z-score pool, the rank-normalised pool and the
    per-assay z / rank columns.

    All are computed over the COMBINED singles+doubles set, so one variant has
    one target value whichever stratum it is read in. The price is that each
    assay's weight in the pool differs by stratum (the two assays' spreads
    differ between singles and doubles); reliability_report records the
    stratum-wise correlation of the pool with each assay so that is visible.
    """
    out = merged.copy()
    sa = out[f"score_{assay_a}"].to_numpy()
    sb = out[f"score_{assay_b}"].to_numpy()
    out["pooled"] = pt.pool_z(sa, sb)
    out["pooled_rank"] = pt.pool_rank(sa, sb)
    out["z_a"] = pt.zscore(sa)
    out["z_b"] = pt.zscore(sb)
    return out


def reliability_report(df, assay_a, assay_b, parallel=True):
    """Inter-assay agreement of the pair, overall and per stratum.

    split_half_spearman is the plain Spearman between the two assays. It is a
    reliability, and pooled_reliability (Spearman-Brown) is only computed, when
    parallel=True; for a pair that quantity.pair_status says is not parallel
    pooled_reliability is NaN, because the formula presumes one true score
    measured twice. per_assay_noise_sd_z is sd(zA - zB)/sqrt(2); for a
    non-parallel pair it includes the real difference between the two
    quantities and the scale nonlinearity, not just noise.

    Also recorded per stratum, because the pool is a different mixture in each:
    sd of each assay's z-score, Pearson of the two z-scores (against the
    Spearman: a large gap means a nonlinear scale difference), and the
    correlation of the z-pool with each assay.
    """
    rows = []

    def one(label, sub):
        sa = sub[f"score_{assay_a}"].to_numpy()
        sb = sub[f"score_{assay_b}"].to_numpy()
        za, zb = sub["z_a"].to_numpy(), sub["z_b"].to_numpy()
        pooled = sub["pooled"].to_numpy()
        r = pt.split_half_reliability(sa, sb)
        ok = np.isfinite(za) & np.isfinite(zb)
        rows.append({
            "stratum": label,
            "n": int(len(sub)),
            "parallel_pair": bool(parallel),
            "split_half_spearman": r,
            "pooled_reliability": pt.reliability_ceiling(r, parallel=parallel),
            "per_assay_noise_sd_z": pt.pooled_noise_sd(sa, sb),
            "pearson_z_a_z_b": (float(np.corrcoef(za[ok], zb[ok])[0, 1])
                                if ok.sum() > 2 else np.nan),
            "sd_z_a": float(np.nanstd(za, ddof=1)),
            "sd_z_b": float(np.nanstd(zb, ddof=1)),
            "corr_pooled_z_with_a": (float(np.corrcoef(pooled[ok], za[ok])[0, 1])
                                     if ok.sum() > 2 else np.nan),
            "corr_pooled_z_with_b": (float(np.corrcoef(pooled[ok], zb[ok])[0, 1])
                                     if ok.sum() > 2 else np.nan),
        })

    one("all", df)
    one("singles", df[df.n_sub == 1])
    one("doubles", df[df.n_sub == 2])
    return rows


# ----------------------------------------------------------------------
# model scoring
# ----------------------------------------------------------------------

def masked_table(scorer, sequence, positions, cache_path, chunk=64):
    """(position -> 20 log-probabilities), cached to disk and resumable.

    scorer may be None when every position is already cached; the model is then
    never loaded. If a position is missing and scorer is None the ESM-2 scorer
    is constructed on CPU.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if cache_path.exists():
        with np.load(cache_path) as z:
            done = {int(p): z["table"][i] for i, p in enumerate(z["positions"])}

    todo = [int(p) for p in positions if int(p) not in done]
    if todo and scorer is None:
        from variant_effect.scoring import ESMScorer
        scorer = ESMScorer(device="cpu")
    for start in range(0, len(todo), chunk):
        block = todo[start:start + chunk]
        tab = scorer.masked_marginal_logprobs(sequence, block, batch_size=8)
        for p in block:
            done[p] = tab[p]
        saved = np.array([int(p) for p in positions if int(p) in done])
        arr = np.stack([done[int(p)] for p in saved]) if len(saved) else \
            np.zeros((0, 20), dtype=np.float32)
        np.savez(cache_path, positions=saved, table=arr)
        print(f"    positions {min(start + chunk, len(todo))}/{len(todo)}",
              flush=True)
    return {int(p): done[int(p)] for p in positions}


def score_from_table(table, variants):
    """Independent-site score: the sum of per-site log-odds, nothing else."""
    out = np.zeros(len(variants), dtype=np.float64)
    for i, m in enumerate(variants):
        total = 0.0
        for s in parse_variant(m):
            row = table[s.idx]
            total += row[ORDER[s.mut]] - row[ORDER[s.wt]]
        out[i] = total
    return out


def score_pair(pair, df, sequence, cache_dir):
    """Score every shared variant; one table per mutated position."""
    positions = sorted({s.idx for m in df["mutant"] for s in parse_variant(m)})
    cache_path = cache_dir / f"{pair['protein']}.npz"
    print(f"  {pair['protein']}: {len(positions)} distinct positions",
          flush=True)

    t0 = time.time()
    table = masked_table(None, sequence, positions, cache_path)
    scores = score_from_table(table, df["mutant"].tolist())
    print(f"  scored {len(df)} variants in {time.time() - t0:.0f}s", flush=True)
    return scores, len(positions)


# ----------------------------------------------------------------------
# the comparison
# ----------------------------------------------------------------------

def _rho(pred, truth):
    return evaluate.spearman(np.asarray(pred, float), np.asarray(truth, float))


def _top_decile(pred, truth):
    """Share of the measured top decile that the model also puts in its top
    decile. Chosen over top-10 recall because the two strata differ in size by
    a factor of seven (20,124 vs 2,948 on RASK) and a fixed k is not
    comparable across them. Note: for a stratum of n = 76 the decile is k = 8
    variants, so a recall of 0.25 is 2 of 8; the interval says how much that
    is worth."""
    pred = np.asarray(pred, float)
    truth = np.asarray(truth, float)
    ok = np.isfinite(pred) & np.isfinite(truth)
    p, t = pred[ok], truth[ok]
    if len(p) < 50:
        return np.nan
    k = max(1, int(round(TOP_FRAC * len(t))))
    thr = np.quantile(t, 1 - TOP_FRAC)
    chosen = np.argsort(-p)[:k]
    return float(np.mean(t[chosen] >= thr))


def bootstrap_gap(single, double, n_boot=N_BOOT, seed=0, metric="spearman",
                  mutants_s=None, mutants_d=None):
    """CI on metric(singles) - metric(doubles).

    RESAMPLING UNIT: pass mutants_s and mutants_d (the mutant strings of each
    stratum) and the unit is the POSITION, drawn jointly for both strata
    (resampling.gap_bootstrap). Omit them and the unit is the VARIANT, each
    stratum resampled independently with iid draws -- valid only for synthetic
    data or data with one row per position, and about three times too narrow on
    RASK doubles. Every analysis in this script passes the mutants.

    In the variant-level mode the two strata are resampled independently
    rather than paired: a paired bootstrap over the union would estimate
    something else, the difference in the model's accuracy on the same variants.
    """
    fn = _rho if metric == "spearman" else _top_decile
    point, lo, hi, _ = resampling.gap_bootstrap(
        fn, single, double, mutants_s, mutants_d,
        n_boot=n_boot, seed=seed, min_n=30)
    return point, lo, hi


def _memberships(mutants):
    return [resampling.substitution_positions(m) for m in mutants]


def target_arrays(df, assay_a, assay_b):
    """The four targets as arrays aligned to df."""
    return {
        "pooled_rank": df["pooled_rank"].to_numpy(float),
        "pooled_z": df["pooled"].to_numpy(float),
        "assay_a": df[f"score_{assay_a}"].to_numpy(float),
        "assay_b": df[f"score_{assay_b}"].to_numpy(float),
    }


def stratified(pair, df, scores, table_positions, n_boot=N_BOOT, seed=0):
    """Per-stratum metrics and the singles-minus-doubles gap, per target.

    Every interval is resampled by position (see bootstrap_gap and
    evaluate.spearman_ci(groups=)). The range-matched gap trims both strata to
    the fitness window they share, per target, and its interval recomputes that
    window in every replicate.
    """
    df = df.copy()
    df["model"] = scores
    s = df[df.n_sub == 1]
    d = df[df.n_sub == 2]
    ms, md = s["mutant"].tolist(), d["mutant"].tolist()
    groups_s, groups_d = _memberships(ms), _memberships(md)
    targets = target_arrays(df, pair["assay_a"], pair["assay_b"])
    sel_s, sel_d = (df.n_sub == 1).to_numpy(), (df.n_sub == 2).to_numpy()
    n_clusters = resampling.ClusterIndex(groups_s + groups_d).n_clusters

    rows, gap_rows = [], []
    for tname in TARGETS:
        t_all = targets[tname]
        for label, sel, groups, sub in (("singles", sel_s, groups_s, s),
                                        ("doubles", sel_d, groups_d, d)):
            pred = sub["model"].to_numpy()
            truth = t_all[sel]
            lo, hi = evaluate.spearman_ci(pred, truth, n_boot=n_boot,
                                          seed=seed, groups=groups)
            rows.append({
                "protein": pair["protein"],
                "pair_role": pair["role"],
                "target": tname,
                "stratum": label,
                "n": int(len(sub)),
                "spearman": _rho(pred, truth),
                "spearman_lo": lo,
                "spearman_hi": hi,
                "top_decile_recall": _top_decile(pred, truth),
                "top10_recall": evaluate.top_k_recall(pred, truth, k=10),
                "model_score_sd": (float(np.nanstd(pred)) if len(pred)
                                   else np.nan),
                "resampling_unit": "position",
            })
        single = (s["model"].to_numpy(), t_all[sel_s])
        double = (d["model"].to_numpy(), t_all[sel_d])
        g_rho = resampling.gap_bootstrap(_rho, single, double, ms, md,
                                         n_boot=n_boot, seed=seed)
        g_rec = resampling.gap_bootstrap(_top_decile, single, double, ms, md,
                                         n_boot=n_boot, seed=seed)
        rm = strata.range_matched_rho(single[0], single[1],
                                      double[0], double[1])
        g_rm = resampling.gap_bootstrap(strata.range_matched_gap, single,
                                        double, ms, md, n_boot=n_boot,
                                        seed=seed, joint=True)
        gap_rows.append({
            "protein": pair["protein"],
            "pair_role": pair["role"],
            "target": tname,
            "is_primary_target": tname == PRIMARY_TARGET,
            "n_singles": int(len(s)),
            "n_doubles": int(len(d)),
            "positions_scored": int(table_positions),
            "n_position_clusters": int(n_clusters),
            "resampling_unit": "position (joint draw, both strata)",
            "gap_spearman": g_rho[0],
            "gap_spearman_lo": g_rho[1],
            "gap_spearman_hi": g_rho[2],
            "gap_spearman_sd": g_rho[3],
            "gap_recall": g_rec[0],
            "gap_recall_lo": g_rec[1],
            "gap_recall_hi": g_rec[2],
            "range_matched_rho_singles": rm[0],
            "range_matched_rho_doubles": rm[1],
            "range_matched_n_singles": rm[2],
            "range_matched_n_doubles": rm[3],
            "range_matched_gap": g_rm[0],
            "range_matched_gap_lo": g_rm[1],
            "range_matched_gap_hi": g_rm[2],
            "truth_sd_singles": float(np.nanstd(t_all[sel_s], ddof=1)),
            "truth_sd_doubles": float(np.nanstd(t_all[sel_d], ddof=1)),
        })
    return rows, gap_rows


# A position bootstrap needs enough positions to resample. With a handful the
# replicates are dominated by which positions were drawn and the percentile
# interval is not an interval for anything (SPG1: 4 positions).
MIN_CLUSTERS = 20


def _state(point, lo, hi, n_d, n_s, n_clusters=None):
    if n_clusters is not None and n_clusters < MIN_CLUSTERS:
        return "insufficient_clusters"
    if not np.isfinite(point) or n_d < 100 or n_s < 50:
        return "insufficient"
    if not np.isfinite(lo):
        return "insufficient"
    if lo > 0:
        return "degrades"
    if hi < 0:
        return "better_on_doubles"
    return "not_distinguishable"


def verdict_text(per_stratum, gaps, reliability):
    """Plain-language reading of the numbers, without overclaiming.

    The three states the data can be in are stated explicitly, and the wording
    of each is fixed here rather than improvised in the README: a positive gap
    whose interval excludes zero is degradation; an interval spanning zero is
    "not distinguishable"; a negative gap with an interval excluding zero means
    the model does *better* on doubles, which contradicts the prediction. Fewer
    than MIN_CLUSTERS positions makes any position-bootstrap interval
    uninterpretable, and the state is then "insufficient_clusters" whatever the
    interval says.

    Rows with a target field are read per target: out[protein] is the
    PRIMARY_TARGET (or the only row when there is no target field) and every
    other target is keyed "protein|target". A row with range_matched_gap_* adds
    range_matched_state, and the README reads the controlled state, not the
    raw one, when they differ. A row with pair_parallel False adds a loud
    pair_note.
    """
    out = {}
    for gap in gaps:
        protein = gap["protein"]
        target = gap.get("target")
        key = protein if target in (None, PRIMARY_TARGET) else \
            f"{protein}|{target}"
        lo, hi = gap["gap_spearman_lo"], gap["gap_spearman_hi"]
        point = gap["gap_spearman"]
        n_d = gap["n_doubles"]
        n_s = gap["n_singles"]
        rel = next((r for r in reliability
                    if r["stratum"] == "doubles"
                    and r.get("protein") == protein), None)
        relv = rel["pooled_reliability"] if rel else float("nan")

        ncl = gap.get("n_position_clusters")
        state = _state(point, lo, hi, n_d, n_s, ncl)

        if np.isfinite(relv) and relv < 0.4:
            target_note = ("pool too noisy (doubles reliability "
                           f"{relv:.2f}) -- a null here is uninformative")
        elif np.isfinite(relv):
            target_note = f"doubles reliability {relv:.2f}"
        elif rel is not None and rel.get("parallel_pair") is False:
            target_note = ("reliability not defined: the two assays are not "
                           "parallel measures of one quantity")
        else:
            target_note = "reliability unavailable"

        entry = {
            "state": state,
            "gap_spearman": point,
            "gap_ci": [lo, hi],
            "n_singles": n_s,
            "n_doubles": n_d,
            "target_note": target_note,
            "n_position_clusters": ncl,
        }
        if target is not None:
            entry["target"] = target
        if "range_matched_gap" in gap:
            rm = gap["range_matched_gap"]
            entry["range_matched_gap"] = rm
            entry["range_matched_ci"] = [gap["range_matched_gap_lo"],
                                         gap["range_matched_gap_hi"]]
            entry["range_matched_state"] = _state(
                rm, gap["range_matched_gap_lo"], gap["range_matched_gap_hi"],
                n_d, n_s, ncl)
        if gap.get("pair_parallel") is False:
            entry["pair_note"] = ("NOT A REPLICATE PAIR: "
                                  + str(gap.get("pair_reasons", "")))
        out[key] = entry
    return out


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def run_pair(key, pooled_only=False, cap=None, seed=0, n_boot=N_BOOT,
             reference=None):
    pair = PAIRS[key]
    status = quantity.pair_status(pair["assay_a"], pair["assay_b"], reference)
    print(f"\n== {pair['protein']} ({pair['role']}): "
          f"{pair['assay_a']} vs {pair['assay_b']}", flush=True)
    if not status["parallel"]:
        print("  NOT A PARALLEL PAIR: " + "; ".join(status["reasons"]),
              flush=True)

    join = join_report(pair["assay_a"], pair["assay_b"])
    print(f"  join: exact string {join['exact_string_intersection']}, "
          f"canonical {join['canonical_intersection']} "
          f"(dropped by exact join: {join['dropped_by_exact_string_join']})",
          flush=True)
    merged = load_pair(pair["assay_a"], pair["assay_b"])
    print(f"  shared variants: {len(merged)} "
          f"(singles {(merged.n_sub == 1).sum()}, "
          f"doubles {(merged.n_sub == 2).sum()})", flush=True)
    if cap and len(merged) > cap:
        # Stratified cap: a plain sample of RASK's shared set is 87% doubles, so
        # a cap without stratification would leave almost no singles.
        rng = np.random.default_rng(seed)
        keep = []
        for _, grp in merged.groupby("n_sub"):
            idx = grp.index.to_numpy()
            if len(idx) > cap:
                idx = rng.choice(idx, cap, replace=False)
            keep.append(pd.Series(idx))
        merged = merged.loc[pd.concat(keep).to_numpy()].reset_index(drop=True)
        print(f"  capped to {len(merged)} "
              f"(singles {(merged.n_sub == 1).sum()}, "
              f"doubles {(merged.n_sub == 2).sum()})", flush=True)

    df = pooled_frame(merged, pair["assay_a"], pair["assay_b"])
    rel = reliability_report(df, pair["assay_a"], pair["assay_b"],
                             parallel=status["parallel"])
    for r in rel:
        print(f"  agreement {r['stratum']:>8} n={r['n']:>6} "
              f"spearman(A,B)={r['split_half_spearman']:+.3f} "
              f"pearson(zA,zB)={r['pearson_z_a_z_b']:+.3f} "
              f"pooled-reliability={r['pooled_reliability']:.3f}", flush=True)

    base = {"pair": pair, "df": df, "reliability": rel, "status": status,
            "join": join}
    if pooled_only:
        return {**base, "per_stratum": [], "gaps": [], "positions": 0}

    _, sequence = data.load_assay(pair["assay_a"], singles_only=False)
    scores, n_pos = score_pair(pair, df, sequence, CACHE)
    df["model"] = scores
    rows, gaps = stratified(pair, df, scores, n_pos, n_boot=n_boot, seed=seed)
    for g in gaps:
        g["pair_parallel"] = status["parallel"]
        g["pair_different_quantity"] = status["different_quantity"]
        g["pair_different_scale"] = status["different_scale"]
        g["pair_reasons"] = "; ".join(status["reasons"])
    for r in rows:
        if r["target"] == PRIMARY_TARGET:
            print(f"  {r['stratum']:>8} n={r['n']:>6} rho={r['spearman']:+.3f} "
                  f"[{r['spearman_lo']:+.3f}, {r['spearman_hi']:+.3f}] "
                  f"topdec={r['top_decile_recall']:.3f}", flush=True)
    for g in gaps:
        print(f"  gap vs {g['target']:>11}: rho {g['gap_spearman']:+.3f} "
              f"[{g['gap_spearman_lo']:+.3f}, {g['gap_spearman_hi']:+.3f}]  "
              f"range-matched {g['range_matched_gap']:+.3f} "
              f"[{g['range_matched_gap_lo']:+.3f}, "
              f"{g['range_matched_gap_hi']:+.3f}]", flush=True)
    return {**base, "per_stratum": rows, "gaps": gaps, "positions": n_pos}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="all", choices=["all", *PAIRS])
    ap.add_argument("--pooled-only", action="store_true",
                    help="reliability only; no model forward passes")
    ap.add_argument("--cap", type=int, default=None,
                    help="max variants per stratum (stratified cap)")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args()

    keys = list(PAIRS) if args.pair == "all" else [args.pair]
    OUT.mkdir(parents=True, exist_ok=True)
    reference = data.reference_table()

    per_stratum, gaps, rel_all = [], [], []
    manifest = {"pairs": {}, "skipped": SKIPPED, "top_frac": TOP_FRAC,
                "n_boot": args.n_boot, "pooled_only": args.pooled_only,
                "resampling_unit": RESAMPLING_UNIT,
                "primary_target": PRIMARY_TARGET, "targets": list(TARGETS)}

    for key in keys:
        r = run_pair(key, pooled_only=args.pooled_only, cap=args.cap,
                     n_boot=args.n_boot, reference=reference)
        p = r["pair"]
        manifest["pairs"][key] = {
            "protein": p["protein"],
            "assay_a": p["assay_a"],
            "assay_b": p["assay_b"],
            "role": p["role"],
            "pair_status": r["status"],
            "join": r["join"],
            "shared": int(len(r["df"])),
            "shared_singles": int((r["df"].n_sub == 1).sum()),
            "shared_doubles": int((r["df"].n_sub == 2).sum()),
            "positions_scored": r["positions"],
            "reliability": r["reliability"],
        }
        rel_all.extend({"protein": p["protein"], **x} for x in r["reliability"])
        per_stratum.extend(r["per_stratum"])
        gaps.extend(r["gaps"])
        if not args.pooled_only:
            cols = ["mutant", "n_sub", "pooled", "pooled_rank", "model",
                    f"score_{p['assay_a']}", f"score_{p['assay_b']}"]
            r["df"][cols].to_csv(OUT / f"scored_{p['protein']}.csv",
                                 index=False)

    pd.DataFrame(rel_all).to_csv(OUT / "reliability.csv", index=False)
    if per_stratum:
        pd.DataFrame(per_stratum).to_csv(OUT / "per_stratum.csv", index=False)
    if gaps:
        pd.DataFrame(gaps).to_csv(OUT / "gaps.csv", index=False)
        manifest["verdict"] = verdict_text(per_stratum, gaps, rel_all)
    with open(OUT / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
