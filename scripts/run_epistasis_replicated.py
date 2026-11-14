"""Does the independent-site model degrade on *replicated* double mutants?

The earlier test (``scripts/run_epistasis_test.py``) compared singles with
multi-mutants inside a single assay. That comparison has a confound it cannot
remove: in most of those assays the singles library and the combinatorial
library are different measurement sets -- sometimes different reporters,
always a different variant distribution -- so a rank difference between the
two strata can be a difference in the *target*, not in the model.

This script removes that confound by using a pair of assays of the same
protein as a pooled target.

The design
----------

1. Two assays of the same protein are intersected on the exact variant string.
   Every shared variant is therefore measured twice, by two independent
   experiments of the same quantity.
2. Each assay is z-scored over the shared set and the two are averaged
   (``pooled_truth.pool_z``). That average carries about half of each assay's
   independent measurement noise, so it is a lower-noise estimate of the
   latent fitness than either assay alone.
3. The same ESM-2 independent-site scorer scores *every* shared variant.
   ``parse_variant`` already splits ``A2M:K5R`` on ``:`` and ``score_variants``
   already sums per-site log-odds -- that sum *is* the model of independence,
   unmodified. One masked-marginal table per mutated position serves both
   strata, so singles and doubles are ranked by exactly the same model, in the
   same units, against exactly the same target.
4. Spearman and top-decile recall are computed separately on the singles and
   on the doubles. The gap (singles - doubles) is bootstrapped.

The prediction under the epistasis hypothesis is a *positive* gap: the scorer
should do worse on doubles, because that is where summing independent terms is
most wrong. Singles are scored by the same model and are the control.

Why this is a different question from the earlier run
-----------------------------------------------------

The earlier run found multi-mutants ranking *better* than singles (pooled
+0.481 vs +0.346 Spearman). It compared two different variant sets within one
assay. Here both strata are subsets of one shared variant list, measured by
two assays, against one target. If doubles still rank as well as singles here,
the earlier "better, not worse" cannot be explained by a target mismatch, and
the independent-site explanation loses support. If doubles now fall behind,
the earlier result is the one that was confounded.

Two proteins are run and reported separately, never pooled: RASK_HUMAN (188 aa,
23,072 shared of which 20,124 doubles and 2,948 singles; Weng 2022 abundance vs
binding-DARPin_K55) and SPG1_STRSG (448 aa, 1,458 shared of which 1,382 doubles
and 76 singles; Olson 2014 vs Wu 2016). The second is a replication with only
76 singles, so its single stratum is fragile and is reported with its n.

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

from variant_effect import data, evaluate  # noqa: E402
from variant_effect import pooled_truth as pt  # noqa: E402
from variant_effect.mutations import parse_variant  # noqa: E402

OUT = ROOT / "results" / "epistasis_replicated"
CACHE = OUT / "_cache"

AA20 = "ACDEFGHIKLMNPQRSTVWY"
ORDER = {a: i for i, a in enumerate(AA20)}
TOP_FRAC = 0.1
N_BOOT = 1000

# Two assays of the same protein, with a shared variant library. Verified counts
# are re-derived at run time and written to the manifest rather than trusted.
PAIRS = {
    "rask": {
        "protein": "RASK_HUMAN",
        "assay_a": "RASK_HUMAN_Weng_2022_abundance",
        "assay_b": "RASK_HUMAN_Weng_2022_binding-DARPin_K55",
        "role": "primary",
    },
    "spg1": {
        "protein": "SPG1_STRSG",
        "assay_a": "SPG1_STRSG_Olson_2014",
        "assay_b": "SPG1_STRSG_Wu_2016",
        "role": "replication",
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


def load_pair(assay_a, assay_b):
    """Intersection of two assays on the exact mutant string.

    Matching on the exact string is deliberate. Normalising by sorting the
    sub-mutations inside a double (``A2M:K5R`` vs ``K5R:A2M``) would inflate
    the intersection if the two libraries happened to order them differently,
    and this project's two pairs do not need it -- the shared sets are
    identical either way (verified: sorted-key and exact-string intersections
    are both 23,072 for RASK and 1,458 for SPG1). Using the strict key means a
    joined row is one variant, not two notations of one variant.
    """
    frames = []
    for assay in (assay_a, assay_b):
        df = pd.read_csv(data.ASSAY_DIR / f"{assay}.csv",
                         usecols=["mutant", "DMS_score"])
        df = df[np.isfinite(df["DMS_score"])].drop_duplicates("mutant")
        frames.append(df.rename(columns={"DMS_score": f"score_{assay}"}))
    a, b = frames
    merged = a.merge(b, on="mutant", how="inner", validate="one_to_one")
    merged["n_sub"] = merged["mutant"].map(n_substitutions)
    return merged[merged["n_sub"].isin((1, 2))].reset_index(drop=True)


def pooled_frame(merged, assay_a, assay_b):
    """Attach the z-scored pooled target and the per-stratum reliability."""
    out = merged.copy()
    sa = out[f"score_{assay_a}"].to_numpy()
    sb = out[f"score_{assay_b}"].to_numpy()
    out["pooled"] = pt.pool_z(sa, sb)
    out["z_a"] = pt.zscore(sa)
    out["z_b"] = pt.zscore(sb)
    return out


def reliability_report(df, assay_a, assay_b):
    """Split-half reliability of the pair, overall and per stratum.

    A low value here means the pooled target is mostly noise and any model-vs-
    target correlation measured against it is attenuated towards zero, which
    would make a null result uninformative. It is reported first for that
    reason.
    """
    rows = []

    def one(label, sub):
        sa = sub[f"score_{assay_a}"].to_numpy()
        sb = sub[f"score_{assay_b}"].to_numpy()
        r = pt.split_half_reliability(sa, sb)
        rows.append({
            "stratum": label,
            "n": int(len(sub)),
            "split_half_spearman": r,
            "pooled_reliability": pt.reliability_ceiling(r),
            "per_assay_noise_sd_z": pt.pooled_noise_sd(sa, sb),
        })

    one("all", df)
    one("singles", df[df.n_sub == 1])
    one("doubles", df[df.n_sub == 2])
    return rows


# ----------------------------------------------------------------------
# model scoring
# ----------------------------------------------------------------------

def masked_table(scorer, sequence, positions, cache_path, chunk=64):
    """(position -> 20 log-probabilities), cached to disk and resumable."""
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
    from variant_effect.scoring import ESMScorer

    positions = sorted({s.idx for m in df["mutant"] for s in parse_variant(m)})
    cache_path = cache_dir / f"{pair['protein']}.npz"
    print(f"  {pair['protein']}: {len(positions)} distinct positions",
          flush=True)

    t0 = time.time()
    scorer = ESMScorer(device="cpu")
    table = masked_table(scorer, sequence, positions, cache_path)
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
    comparable across them."""
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


def bootstrap_gap(single, double, n_boot=N_BOOT, seed=0, metric="spearman"):
    """CI on metric(singles) - metric(doubles), resampling each stratum.

    The two strata are disjoint variant sets, so they are resampled
    independently rather than paired. A paired bootstrap over the union would
    estimate something else -- the difference in the model's accuracy on the
    same variants, which is not what a single-vs-double comparison is.
    """
    fn = _rho if metric == "spearman" else _top_decile
    rng = np.random.default_rng(seed)
    ps, ts = single
    pd_, td = double
    ps, ts = np.asarray(ps, float), np.asarray(ts, float)
    pd_, td = np.asarray(pd_, float), np.asarray(td, float)
    ns, nd = len(ps), len(pd_)
    if ns < 30 or nd < 30:
        return np.nan, np.nan, np.nan

    gaps = np.empty(n_boot)
    for i in range(n_boot):
        si = rng.integers(0, ns, ns)
        di = rng.integers(0, nd, nd)
        a = fn(ps[si], ts[si])
        b = fn(pd_[di], td[di])
        gaps[i] = a - b
    gaps = gaps[np.isfinite(gaps)]
    if len(gaps) < 50:
        return np.nan, np.nan, np.nan
    point = fn(ps, ts) - fn(pd_, td)
    return (float(point), float(np.percentile(gaps, 2.5)),
            float(np.percentile(gaps, 97.5)))


def stratified(pair, df, scores, table_positions):
    """Per-stratum metrics and the singles-minus-doubles gap."""
    df = df.copy()
    df["model"] = scores

    rows = []
    for label, k in (("singles", 1), ("doubles", 2)):
        sub = df[df.n_sub == k]
        pred = sub["model"].to_numpy()
        truth = sub["pooled"].to_numpy()
        rho = _rho(pred, truth)
        lo, hi = evaluate.spearman_ci(pred, truth, n_boot=1000)
        rows.append({
            "protein": pair["protein"],
            "pair_role": pair["role"],
            "stratum": label,
            "n": int(len(sub)),
            "spearman": rho,
            "spearman_lo": lo,
            "spearman_hi": hi,
            "top_decile_recall": _top_decile(pred, truth),
            "top10_recall": evaluate.top_k_recall(pred, truth, k=10),
            "model_score_sd": float(np.nanstd(pred)) if len(pred) else np.nan,
        })

    s = df[df.n_sub == 1]
    d = df[df.n_sub == 2]
    g_rho = bootstrap_gap((s["model"], s["pooled"]),
                          (d["model"], d["pooled"]), metric="spearman")
    g_rec = bootstrap_gap((s["model"], s["pooled"]),
                          (d["model"], d["pooled"]), metric="recall")

    gap_rows = [{
        "protein": pair["protein"],
        "pair_role": pair["role"],
        "n_singles": int(len(s)),
        "n_doubles": int(len(d)),
        "positions_scored": int(table_positions),
        "gap_spearman": g_rho[0],
        "gap_spearman_lo": g_rho[1],
        "gap_spearman_hi": g_rho[2],
        "gap_recall": g_rec[0],
        "gap_recall_lo": g_rec[1],
        "gap_recall_hi": g_rec[2],
    }]
    return rows, gap_rows


def verdict_text(per_stratum, gaps, reliability):
    """Plain-language reading of the numbers, without overclaiming.

    The three states the data can be in are stated explicitly, and the wording
    of each is fixed here rather than improvised in the README: a positive gap
    whose interval excludes zero is degradation; an interval spanning zero is
    "not distinguishable"; a negative gap with an interval excluding zero means
    the model does *better* on doubles, which contradicts the prediction.
    """
    out = {}
    for gap in gaps:
        protein = gap["protein"]
        lo, hi = gap["gap_spearman_lo"], gap["gap_spearman_hi"]
        point = gap["gap_spearman"]
        n_d = gap["n_doubles"]
        n_s = gap["n_singles"]
        rel = next((r for r in reliability
                    if r["stratum"] == "doubles"
                    and r.get("protein") == protein), None)
        relv = rel["pooled_reliability"] if rel else float("nan")

        if not np.isfinite(point) or n_d < 100 or n_s < 50:
            state = "insufficient"
        elif not np.isfinite(lo):
            state = "insufficient"
        elif lo > 0:
            state = "degrades"
        elif hi < 0:
            state = "better_on_doubles"
        else:
            state = "not_distinguishable"

        if np.isfinite(relv) and relv < 0.4:
            target_note = ("pool too noisy (doubles reliability "
                           f"{relv:.2f}) -- a null here is uninformative")
        else:
            target_note = f"doubles reliability {relv:.2f}" if np.isfinite(relv) \
                else "reliability unavailable"

        out[protein] = {
            "state": state,
            "gap_spearman": point,
            "gap_ci": [lo, hi],
            "n_singles": n_s,
            "n_doubles": n_d,
            "target_note": target_note,
        }
    return out


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def run_pair(key, pooled_only=False, cap=None, seed=0):
    pair = PAIRS[key]
    print(f"\n== {pair['protein']} ({pair['role']}): "
          f"{pair['assay_a']} vs {pair['assay_b']}", flush=True)

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
    rel = reliability_report(df, pair["assay_a"], pair["assay_b"])
    for r in rel:
        print(f"  reliability {r['stratum']:>8} n={r['n']:>6} "
              f"split-half={r['split_half_spearman']:+.3f} "
              f"pooled={r['pooled_reliability']:.3f}", flush=True)

    if pooled_only:
        return {"pair": pair, "df": df, "reliability": rel,
                "per_stratum": [], "gaps": [], "positions": 0}

    _, sequence = data.load_assay(pair["assay_a"], singles_only=False)
    scores, n_pos = score_pair(pair, df, sequence, CACHE)
    df["model"] = scores
    rows, gaps = stratified(pair, df, scores, n_pos)
    for r in rows:
        print(f"  {r['stratum']:>8} n={r['n']:>6} rho={r['spearman']:+.3f} "
              f"[{r['spearman_lo']:+.3f}, {r['spearman_hi']:+.3f}] "
              f"topdec={r['top_decile_recall']:.3f}", flush=True)
    g = gaps[0]
    print(f"  gap singles-doubles rho {g['gap_spearman']:+.3f} "
          f"[{g['gap_spearman_lo']:+.3f}, {g['gap_spearman_hi']:+.3f}]",
          flush=True)
    return {"pair": pair, "df": df, "reliability": rel,
            "per_stratum": rows, "gaps": gaps, "positions": n_pos}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="all", choices=["all", *PAIRS])
    ap.add_argument("--pooled-only", action="store_true",
                    help="reliability only; no model forward passes")
    ap.add_argument("--cap", type=int, default=None,
                    help="max variants per stratum (stratified cap)")
    args = ap.parse_args()

    keys = list(PAIRS) if args.pair == "all" else [args.pair]
    OUT.mkdir(parents=True, exist_ok=True)

    results, per_stratum, gaps, rel_all = [], [], [], []
    manifest = {"pairs": {}, "skipped": SKIPPED, "top_frac": TOP_FRAC,
                "n_boot": N_BOOT, "pooled_only": args.pooled_only}

    for key in keys:
        r = run_pair(key, pooled_only=args.pooled_only, cap=args.cap)
        p = r["pair"]
        manifest["pairs"][key] = {
            "protein": p["protein"],
            "assay_a": p["assay_a"],
            "assay_b": p["assay_b"],
            "role": p["role"],
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
            cols = ["mutant", "n_sub", "pooled", "model"]
            r["df"][cols].to_csv(OUT / f"scored_{p['protein']}.csv",
                                 index=False)

    pd.DataFrame(rel_all).to_csv(OUT / "reliability.csv", index=False)
    if per_stratum:
        pd.DataFrame(per_stratum).to_csv(OUT / "per_stratum.csv", index=False)
    if gaps:
        pd.DataFrame(gaps).to_csv(OUT / "gaps.csv", index=False)
        manifest["verdict"] = verdict_text(per_stratum, gaps, rel_all)
    with open(OUT / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
