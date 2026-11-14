"""Figures from finished runs.

Two groups, selected with --only:

  benchmark  the four figures from scripts/run_benchmark.py (needs
             results/benchmark_*.csv, the ProteinGym download and, for the
             landscape, an ESM-2 checkpoint)
  findings   the figures for the pooled-target, resampling-unit and
             family-test findings. These read ONLY files committed under
             results/ -- no model, no download, no re-run of the latent-truth or
             epistasis analyses -- and finish in seconds.

    python scripts/make_figures.py                  # both, as before
    python scripts/make_figures.py --only findings  # no model needed

Every number a findings figure draws is read from a results/ file. Where a
figure also computes something (a per-protein mean, a count of pairs below a
line) it asserts that the value agrees with the stored summary, so a figure
cannot drift from the README's tables without failing loudly.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from variant_effect import data, evaluate  # noqa: E402

FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
RESULTS = ROOT / "results"

MODEL = "facebook/esm2_t33_650M_UR50D"
TAG = "esm2_t33_650M_UR50D"


def load_results():
    path = RESULTS / f"benchmark_{TAG}.csv"
    if not path.exists():
        raise SystemExit(f"no results at {path}; run scripts/run_benchmark.py first")
    return pd.read_csv(path)


def fig_method_comparison(df):
    """Every method on every assay, so the spread is visible rather than
    hidden behind a mean."""
    order = (df.groupby("method")["spearman"].mean()
               .sort_values(ascending=False).index.tolist())

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, method in enumerate(order):
        vals = df.loc[df["method"] == method, "spearman"].to_numpy()
        ax.scatter(np.full(len(vals), i) + np.random.uniform(-.12, .12, len(vals)),
                   vals, s=26, alpha=0.75, edgecolor="black", linewidth=0.4,
                   color="C0" if method == "esm2" else "C7")
        ax.hlines(vals.mean(), i - 0.3, i + 0.3, color="C3", lw=2, zorder=5)

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=30, ha="right")
    ax.axhline(0, ls="--", lw=1, color="grey")
    ax.set_ylabel("Spearman vs experimental fitness")
    ax.set_title("Zero-shot variant effect prediction, one point per assay",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "method_comparison.png", dpi=150)
    plt.close(fig)


def fig_vs_official(df):
    """Our per-assay Spearman against ProteinGym's published value for the
    same model. Points should sit on the diagonal."""
    official = data.official_scores()
    ours = df[df["method"] == "esm2"].set_index("assay")["spearman"]

    xs, ys, labels = [], [], []
    for assay, val in ours.items():
        if assay in official.index and "ESM2 (650M)" in official.columns:
            xs.append(float(official.loc[assay, "ESM2 (650M)"]))
            ys.append(val)
            labels.append(assay.split("_")[0])

    fig, ax = plt.subplots(figsize=(5, 5))
    lim = [min(xs + ys) - 0.05, max(xs + ys) + 0.05]
    ax.plot(lim, lim, ls="--", lw=1, color="grey")
    ax.scatter(xs, ys, s=50, edgecolor="black", linewidth=0.5, zorder=3)
    for x, y, lab in zip(xs, ys, labels):
        ax.annotate(lab, (x, y), xytext=(4, 3), textcoords="offset points",
                    fontsize=7)

    resid = np.array(ys) - np.array(xs)
    ax.set(xlabel="ProteinGym published Spearman",
           ylabel="this implementation",
           xlim=lim, ylim=lim,
           title=f"ESM-2 650M reproduction\nmean diff {resid.mean():+.4f}, "
                 f"max |diff| {np.abs(resid).max():.4f}")
    ax.title.set_fontsize(10)
    fig.tight_layout()
    fig.savefig(FIG / "reproduction_check.png", dpi=150)
    plt.close(fig)


def fig_landscape(assay_id="IF1_ECOLI_Kelsic_2016"):
    """Predicted mutational landscape next to the measured one.

    This is the picture that makes the point: the model has never seen a
    fitness measurement, and the dark stripes still line up with the real
    ones. Those stripes are the positions that cannot take a substitution,
    which in a small protein is usually the hydrophobic core plus whatever
    does the chemistry.
    """
    from variant_effect.scoring import ESMScorer

    ref = data.reference_table()
    df, sequence = data.load_assay(assay_id, reference=ref)
    aa_order = "ACDEFGHIKLMNPQRSTVWY"

    scorer = ESMScorer(MODEL)
    table = scorer.masked_marginal_logprobs(sequence)
    order = {a: i for i, a in enumerate(aa_order)}
    pred = np.full((20, len(sequence)), np.nan)
    for pos in range(len(sequence)):
        wt_lp = table[pos][order[sequence[pos]]]
        for j, aa in enumerate(aa_order):
            pred[j, pos] = table[pos][order[aa]] - wt_lp

    obs = np.full((20, len(sequence)), np.nan)
    from variant_effect.mutations import parse_variant
    for mutant, score in zip(df["mutant"], df["DMS_score"]):
        (s,) = parse_variant(mutant)
        if s.mut in order:
            obs[order[s.mut], s.idx] = score

    fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    for ax, mat, title, cmap in [
            (axes[0], pred, "ESM-2 650M predicted effect (log-odds vs wildtype)",
             "RdBu"),
            (axes[1], obs, "Measured fitness (Kelsic 2016)", "RdBu")]:
        v = np.nanpercentile(np.abs(mat), 98)
        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=-v, vmax=v,
                       interpolation="nearest")
        ax.set_yticks(range(20))
        ax.set_yticklabels(aa_order, fontsize=6)
        ax.set_title(title, fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.015, pad=0.01)

    axes[1].set_xlabel("residue position")
    rho = evaluate.spearman(pred[~np.isnan(obs)], obs[~np.isnan(obs)])
    fig.suptitle(f"{assay_id}   Spearman = {rho:.3f}", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "landscape.png", dpi=150)
    plt.close(fig)
    print(f"landscape Spearman {rho:.4f}")


def fig_by_taxon(df):
    """Performance split by taxon, because the viral assay is not a bad
    draw -- it is a known structural weakness."""
    esm = df[df["method"] == "esm2"]
    if "taxon" not in esm.columns:
        return
    fig, ax = plt.subplots(figsize=(5.5, 4))
    groups = esm.groupby("taxon")["spearman"]
    names = list(groups.groups)
    ax.bar(names, [groups.get_group(n).mean() for n in names],
           color=["C3" if n == "Virus" else "C0" for n in names],
           edgecolor="black", linewidth=0.5)
    for i, n in enumerate(names):
        vals = groups.get_group(n)
        ax.scatter(np.full(len(vals), i), vals, color="black", s=14, zorder=3)
    ax.set_ylabel("Spearman")
    ax.set_title("ESM-2 650M by taxon", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "by_taxon.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Findings figures: committed results/ files only
# ---------------------------------------------------------------------------

LT = RESULTS / "latent_truth"
REP = RESULTS / "epistasis_replicated"
EPI = RESULTS / "epistasis"
FAM = RESULTS / "families"
# Paired recall differences are multiples of 0.05 computed by float subtraction.
# The repo rounds them to nine decimals before it ranks or counts ties
# (scripts/run_latent_truth_test.py), and so does every count below.
TIE_DECIMALS = 9
TOL = 1e-9


def _need(path):
    if not Path(path).exists():
        raise SystemExit(f"missing {path}: the findings figures read committed "
                         "results/ files only")
    return Path(path)


def _agree(checks, figure, source):
    """Stop if a figure's own arithmetic disagrees with the stored summary."""
    bad = [name for ok, name in checks if not ok]
    if bad:
        raise SystemExit(f"figure {figure} disagrees with {source} on: "
                         + ", ".join(bad))


def _signed(x, nd=3):
    return f"{x:+.{nd}f}".replace("-", "\u2212")


def _p_text(p):
    """A p-value for a label: three decimals, or one significant digit."""
    if p < 1e-3:
        return f"{p:.0e}".replace("e-0", "e\u2212")
    return f"{p:.3f}"


def _source_note(fig, text, y=0.008):
    fig.text(0.012, y, text, fontsize=7, color="0.35", ha="left", va="bottom")


def _save(fig, name):
    out = FIG / name
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")


# ---- a) the pooled target raises the model's rank correlation --------------

def rank_gain_data():
    """Per-pair and per-protein Spearman gain from pooling two assays.

    The gain is Spearman(model, pooled target) - Spearman(model, single assay)
    on the same shared variants, one value per replicate pair; the per-protein
    value is the mean over that protein's pairs, exactly as the test of record
    aggregates it (run_latent_truth_test.protein_level). Checked against
    summary.json before anything is drawn.
    """
    per = pd.read_csv(_need(LT / "per_pair.csv"))
    summ = json.loads(_need(LT / "summary.json").read_text())
    raw = per["spearman_model_to_pooled"] - per["spearman_model_to_single"]
    per = per.assign(gain=raw.round(TIE_DECIMALS))
    prot = raw.groupby(per["uniprot"]).mean().round(TIE_DECIMALS)
    same = per[~per["different_quantity"].astype(bool)]
    a, sq = summ["assays"], summ["same_quantity"]
    w, pw = (a["wilcoxon_spearman_pooled_minus_single"],
             a["protein_level_spearman_gain"])
    sw = sq["wilcoxon_spearman_pooled_minus_single"]
    _agree([
        (len(per) == a["n_pairs"], "number of pairs"),
        (int((per["gain"] > 0).sum()) == w["n_better"], "pairs that gain"),
        (abs(per["gain"].median() - w["median_diff"]) < TOL, "median pair gain"),
        (len(prot) == pw["n_units"], "number of proteins"),
        (int((prot > 0).sum()) == pw["n_better"], "proteins that gain"),
        (abs(prot.median() - pw["median_diff"]) < TOL, "median protein gain"),
        (len(same) == sq["n_pairs"], "same-quantity pair count"),
        (abs(same["gain"].median() - sw["median_diff"]) < TOL,
         "same-quantity median gain"),
    ], "a (rank gain)", "results/latent_truth/summary.json")
    return per, prot, summ


def draw_rank_dumbbell(ax, per, label_size=6.5, compact=False):
    """One row per replicate pair, ordered by gain; blue -> red is the move
    from a single assay to the pooled target."""
    d = per.sort_values("gain", kind="mergesort").reset_index(drop=True)
    y = np.arange(len(d))
    lo = d["spearman_model_to_single"].to_numpy()
    hi = d["spearman_model_to_pooled"].to_numpy()
    same = ~d["different_quantity"].to_numpy(dtype=bool)
    ax.hlines(y[~same], lo[~same], hi[~same], color="0.72", lw=1.4, zorder=1)
    ax.hlines(y[same], lo[same], hi[same], color="black", lw=2.8, zorder=2)
    ax.scatter(lo, y, s=24, color="C0", edgecolor="black", linewidth=0.4,
               zorder=3)
    ax.scatter(hi, y, s=24, color="C3", edgecolor="black", linewidth=0.4,
               zorder=4)
    ax.set_yticks(y)
    if compact:     # protein names are unreadable at README scale; see figure a
        ax.set_yticklabels([])
        ax.set_ylabel(f"{len(d)} replicate pairs, ordered by gain", fontsize=9)
    else:
        ax.set_yticklabels(d["uniprot"], fontsize=label_size)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", color="0.93", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    # a scale bar as long as the median gain, so the median is a length
    med = per["gain"].median()
    x0 = float(min(lo.min(), hi.min())) - 0.01
    ax.annotate("", xy=(x0 + med, -2.1), xytext=(x0, -2.1),
                arrowprops=dict(arrowstyle="<->", lw=1.3, color="black",
                                shrinkA=0, shrinkB=0))
    ax.text(x0, -1.35, f"median gain {_signed(med)}", fontsize=8, va="bottom")
    ax.set_ylim(-3.0, len(d) - 0.4)
    ax.set_xlim(x0 - 0.02, float(max(lo.max(), hi.max())) + 0.03)
    handles = [
        Line2D([], [], marker="o", ls="", color="C0", mec="black", mew=0.4,
               label="model vs one assay" if compact else
               "model vs one assay (mean of the pair's two)"),
        Line2D([], [], marker="o", ls="", color="C3", mec="black", mew=0.4,
               label="model vs pooled target"),
        Line2D([], [], color="black", lw=2.8,
               label=f"same-quantity pair ({int(same.sum())})" if compact else
               f"same-quantity pair (n = {int(same.sum())})"),
        Line2D([], [], color="0.72", lw=1.4,
               label=f"different-quantity pair ({int((~same).sum())})" if compact else
               f"pair flagged as different quantity (n = {int((~same).sum())})"),
    ]
    ax.legend(handles=handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.19 if compact else -0.115), ncol=2,
              fontsize=6.5 if compact else 7.5, frameon=False)
    ax.set_xlabel("Spearman ρ, model vs target" if compact else
                  "Spearman ρ, ESM-2 650M score vs target, on the pair's shared variants")


def _stack_dots(ax, values, same=None, binw=0.01, size=30):
    """Dot histogram: one dot per unit, stacked within bins of width binw.
    Positions on the x axis are the exact values; only the stacking is binned."""
    v = np.asarray(values, float)
    order = np.argsort(v)
    height, xs, ys, fl = {}, [], [], []
    for i in order:
        k = int(np.floor(v[i] / binw + 1e-9))
        h = height.get(k, 0)
        height[k] = h + 1
        xs.append(v[i])
        ys.append(h)
        fl.append(False if same is None else bool(same[i]))
    xs, ys, fl = np.array(xs), np.array(ys), np.array(fl)
    ax.scatter(xs[~fl], ys[~fl], marker="o", s=size, color="0.78",
               edgecolor="black", linewidth=0.4, zorder=3)
    if fl.any():
        ax.scatter(xs[fl], ys[fl], marker="D", s=size * 0.8, color="black",
                   edgecolor="black", linewidth=0.4, zorder=4)
    ax.set_ylim(-0.9, max(ys) + 4.4)
    ax.set_yticks([])
    for s in ("left", "right", "top"):
        ax.spines[s].set_visible(False)


def fig_rank_gain():
    """a) Pooling two assays raises the model's Spearman in every pair."""
    per, prot, summ = rank_gain_data()
    a = summ["assays"]
    fig = plt.figure(figsize=(11, 5.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1], height_ratios=[1, 1],
                          left=0.112, right=0.985, top=0.835, bottom=0.215,
                          wspace=0.17, hspace=0.62)
    axl = fig.add_subplot(gs[:, 0])
    draw_rank_dumbbell(axl, per)
    axl.set_title("One row per replicate pair: the line runs from the single-assay\n"
                  "Spearman (blue) to the pooled-target Spearman (red)", fontsize=9)
    axl.set_ylabel("replicate pair (protein), ordered by gain", fontsize=8)

    lo_g = float(min(per["gain"].min(), prot.min()))
    hi_g = float(max(per["gain"].max(), prot.max()))
    xl = (min(-0.012, lo_g - 0.01), hi_g + 0.012)
    same = ~per["different_quantity"].to_numpy(dtype=bool)
    axt = fig.add_subplot(gs[0, 1])
    _stack_dots(axt, per["gain"], same)
    axb = fig.add_subplot(gs[1, 1], sharex=axt)
    _stack_dots(axb, prot.to_numpy())
    med_p = per["gain"].median()
    med_s = per.loc[~per["different_quantity"], "gain"].median()
    med_q = prot.median()
    for ax in (axt, axb):
        ax.axvline(0, color="grey", lw=1.2, zorder=1)
        ax.set_xlim(*xl)
    axt.axvline(med_p, color="C3", ls="--", lw=1.5, zorder=2)
    axt.axvline(med_s, color="black", ls=":", lw=1.5, zorder=2)
    axb.axvline(med_q, color="C3", ls="--", lw=1.5, zorder=2)
    n_up = int((per["gain"] > 0).sum())
    n_pairs = len(per)
    axt.set_title(f"Gain per pair: {n_up} of {n_pairs} above zero\n"
                  f"\u25c6 same-quantity pair ({int(same.sum())})   "
                  f"\u25cf different-quantity pair ({int((~same).sum())})", fontsize=9)
    axb.set_title(f"Gain per protein (mean of its pairs): "
                  f"{int((prot > 0).sum())} of {len(prot)} above zero", fontsize=9)
    ytop = axt.get_ylim()[1]
    axt.text(med_p, ytop * 0.99, f" all {n_pairs}: median {_signed(med_p)}",
             color="C3", fontsize=7.5, va="top", ha="left")
    axt.text(med_s, ytop * 0.99,
             f"same-quantity {int(same.sum())}:\nmedian {_signed(med_s)} ",
             color="black", fontsize=7.5, va="top", ha="right")
    axb.text(med_q, axb.get_ylim()[1] * 0.99, f" median {_signed(med_q)}",
             color="C3", fontsize=8, va="top", ha="left")
    axb.set_xlabel("gain in Spearman \u03c1 (pooled target \u2212 one assay)")
    plt.setp(axt.get_xticklabels(), visible=False)
    fig.suptitle("Pooling two assays raises the model's rank correlation in "
                 f"{n_up} of {n_pairs} pairs and {int((prot > 0).sum())} of "
                 f"{len(prot)} proteins", fontsize=11, x=0.5, y=0.975)
    _source_note(fig, f"source: results/latent_truth/per_pair.csv, summary.json  "
                      f"|  {n_pairs} replicate pairs, {len(prot)} proteins, "
                      f"{a['n_variants']:,} shared variant measurements  |  "
                      "protein = unit of the test of record")
    _save(fig, "pooled_target_rank_gain.png")


# ---- b) recall gain is marginal where the rank gain is not -----------------

P_METHODS = (("p", "Wilcoxon, tie-corrected", "o"),
             ("p_exact", "exact Wilcoxon", "s"),
             ("p_signflip", "sign-flip permutation", "^"))


def pvalue_data():
    """Every p-value of the pooled-vs-single comparison, from summary.json.

    Returns rows ordered primary subset first. Each row carries the three
    tests (P_METHODS) for top-10 recall and for Spearman. The protein-level
    rows are the primary tests; pair-level rows are secondary because the pairs
    share proteins and assays.
    """
    summ = json.loads(_need(LT / "summary.json").read_text())
    rows = []
    for key, name in (("assays", "all pairs"),
                      ("same_quantity", "same-quantity pairs only"),
                      ("reliable_only", "reliable pairs only (split-half > 0.3)")):
        blk = summ[key]
        for level, rk, sk, tag in (
                ("protein", "protein_level_recall_gain",
                 "protein_level_spearman_gain", "primary"),
                ("pair", "wilcoxon_model_pooled_minus_single",
                 "wilcoxon_spearman_pooled_minus_single", "secondary")):
            r, s = blk[rk], blk[sk]
            rows.append({
                "subset": name, "level": level, "tag": tag, "n": r["n_units"],
                "recall": [r[m] for m, _, _ in P_METHODS],
                "spearman": [s[m] for m, _, _ in P_METHODS],
                "primary_subset": key == "assays"})
    return rows


def pvalue_panel_rows(rows, which, color):
    """Rows for draw_pvalue_rows: one metric (recall or spearman) of pvalue_data.
    A little extra space opens before each new subset."""
    out = []
    for i, r in enumerate(rows):
        new_subset = i > 0 and r["subset"] != rows[i - 1]["subset"]
        tag = " (primary)" if r["primary_subset"] and r["tag"] == "primary" else ""
        out.append({"label": f"{r['subset']}\n{r['level']} level, n = {r['n']}{tag}",
                    "ps": r[which], "color": color,
                    "annotate": r["primary_subset"], "gap": 0.35 if new_subset else 0.0})
    return out


def draw_pvalue_rows(ax, rows, log=False, label_size=7, text_size=7):
    """One row per entry of rows (dicts: label, ps, color, annotate, gap). The
    three tests of a row sit on three sub-lines so markers never hide each
    other. The p = 0.05 line is drawn at full weight."""
    offs = (0.27, 0.0, -0.27)
    ys, y = [], 0.0
    for i, r in enumerate(rows):
        y -= 1.0 + (r.get("gap", 0.0) if i else 0.0)
        ys.append(y)
        for p, off, (_, _, mk) in zip(r["ps"], offs, P_METHODS):
            ax.scatter([p], [y + off], marker=mk, s=46, color=r["color"],
                       edgecolor="black", linewidth=0.5, zorder=3)
            if r.get("annotate"):
                ax.annotate(_p_text(p), (p, y + off), xytext=(6, 0),
                            textcoords="offset points", va="center",
                            fontsize=text_size)
    ax.axvline(0.05, color="black", ls="--", lw=1.3, zorder=2)
    if log:
        ax.set_xscale("log")
    ax.set_yticks(ys)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=label_size)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(min(ys) - 0.7, max(ys) + 0.8)
    ax.grid(axis="x", color="0.93", lw=0.5)
    ax.set_axisbelow(True)


def fig_recall_vs_rank_p():
    """b) The recall gain is marginal where the rank gain is not."""
    rows = pvalue_data()
    rec = [p for r in rows for p in r["recall"]]
    spe = [p for r in rows for p in r["spearman"]]
    n_tests = len(rec)
    cross = sum(p < 0.05 for p in rec)
    fig, (axr, axs) = plt.subplots(1, 2, figsize=(11, 5.4), sharey=True,
                                   gridspec_kw=dict(wspace=0.06, left=0.2,
                                                    right=0.985, top=0.8,
                                                    bottom=0.17))
    draw_pvalue_rows(axr, pvalue_panel_rows(rows, "recall", "C3"))
    axr.set_xlim(0, 0.34)
    draw_pvalue_rows(axs, pvalue_panel_rows(rows, "spearman", "C0"), log=True)
    axs.set_xlim(1e-10, 1.0)
    axr.set_title("Top-10 recall (the ten picks)\n"
                  + (f"{cross} of {n_tests} tests below p = 0.05; smallest p = "
                     f"{min(rec):.3f}" if cross else
                     f"no test reaches p < 0.05; smallest p = {min(rec):.3f}"),
                  fontsize=9.5)
    axs.set_title("Spearman \u03c1 (the whole ordering)\n"
                  f"all {len(spe)} tests below p = 0.05; largest p = "
                  f"{max(spe):.4f}", fontsize=9.5)
    axr.set_xlabel("p-value (linear axis)")
    axs.set_xlabel("p-value (log axis)")
    ymid = float(np.mean(axr.get_ylim()))
    axr.text(0.05, ymid, "p = 0.05 ", fontsize=8, rotation=90, va="center",
             ha="right")
    axs.text(0.05, ymid, " p = 0.05", fontsize=8, rotation=90, va="center",
             ha="left")
    handles = [Line2D([], [], marker=mk, ls="", color="0.8", mec="black",
                      mew=0.5, ms=6, label=name) for _, name, mk in P_METHODS]
    axs.legend(handles=handles, title="test (null: no gain)", loc="lower left",
               fontsize=7.5, title_fontsize=7.5, framealpha=0.95)
    fig.suptitle("Pooling the target improves the model's ordering decisively "
                 "and its top-of-list recall only marginally", fontsize=11,
                 x=0.5, y=0.975)
    _source_note(fig, "source: results/latent_truth/summary.json  |  paired gain "
                      "(pooled target minus single assay) on the same variants of the "
                      "same pairs  |  unit: protein (primary) or pair (secondary)\n"
                      "values are printed for the all-pairs rows; the other rows show "
                      "the same tests on subsets  |  null hypothesis: no gain",
                 y=0.012)
    _save(fig, "recall_vs_rank_significance.png")


# ---- c) the model is still below truth-vs-truth -----------------------------

def ceiling_data():
    """Per-pair model recall against the pair's truth-vs-truth recall."""
    per = pd.read_csv(_need(LT / "per_pair.csv"))
    summ = json.loads(_need(LT / "summary.json").read_text())
    x = per["recall_truth_vs_truth"].round(TIE_DECIMALS)
    out = {"n": len(per), "mean_truth": float(x.mean())}
    for col, key in (("recall_model_to_single", "single"),
                     ("recall_model_to_pooled", "pooled")):
        y = per[col].round(TIE_DECIMALS)
        d = y - x
        out[key] = {"x": x.to_numpy(), "y": y.to_numpy(),
                    "below": int((d < 0).sum()), "on": int((d == 0).sum()),
                    "above": int((d > 0).sum()), "mean_model": float(y.mean())}
    a = summ["assays"]
    _agree([
        (out["n"] == a["n_pairs"], "number of pairs"),
        (abs(out["mean_truth"] - a["mean_recall_truth_vs_truth"]) < 1e-6,
         "mean truth-vs-truth recall"),
        (abs(float(np.median(out["single"]["y"]))
             - a["median_recall_model_to_single"]) < TOL,
         "median model recall vs single assay"),
        (abs(float(np.median(out["pooled"]["y"]))
             - a["median_recall_model_to_pooled"]) < TOL,
         "median model recall vs pooled target"),
    ], "c (model vs truth-vs-truth)", "results/latent_truth/summary.json")
    return out


def draw_ceiling(ax, d, ylabel, title_fs=9.5):
    pts = (pd.DataFrame({"x": d["x"], "y": d["y"]})
           .groupby(["x", "y"]).size().reset_index(name="n"))
    side = np.sign(np.round(pts["y"] - pts["x"], TIE_DECIMALS)).astype(int)
    colors = np.where(side < 0, "C3", np.where(side > 0, "C0", "0.6"))
    ax.plot([0, 1], [0, 1], ls="--", lw=1.2, color="grey", zorder=1)
    ax.scatter(pts["x"], pts["y"], s=34 * pts["n"], c=colors, edgecolor="black",
               linewidth=0.5, alpha=0.9, zorder=3)
    for _, r in pts[pts["n"] > 1].iterrows():
        ax.text(r["x"], r["y"], str(int(r["n"])), ha="center", va="center",
                fontsize=6.5, color="white", fontweight="bold", zorder=4)
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-0.04, 1.04)
    ax.set_aspect("equal")
    ax.grid(color="0.93", lw=0.5)
    ax.set_axisbelow(True)
    ax.set_xlabel("truth-vs-truth top-10 recall of the pair\n(assay A scored by B, and B by A: mean)")
    ax.set_ylabel(ylabel)
    ax.text(0.97, 0.97, f"below the line: {d['below']}\non the line: {d['on']}\n"
                        f"above the line: {d['above']}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.6", lw=0.6))


def fig_model_vs_ceiling():
    """c) The model's top-10 recall against the cross-assay agreement."""
    d = ceiling_data()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5.6),
                                 gridspec_kw=dict(left=0.075, right=0.985,
                                                  top=0.84, bottom=0.18,
                                                  wspace=0.12))
    draw_ceiling(a1, d["single"], "model top-10 recall vs one assay\n(mean over the pair's two assays)")
    draw_ceiling(a2, d["pooled"], "model top-10 recall vs the pooled target")
    for ax, key, what in ((a1, "single", "vs one assay"),
                          (a2, "pooled", "vs the pooled target")):
        ax.set_title(f"model {what}: mean {d[key]['mean_model']:.3f}, "
                     f"truth-vs-truth mean {d['mean_truth']:.3f}", fontsize=9)
    handles = [
        Line2D([], [], marker="o", ls="", color="C3", mec="black", mew=0.5,
               label="below the line (model lower)"),
        Line2D([], [], marker="o", ls="", color="0.6", mec="black", mew=0.5,
               label="on the line"),
        Line2D([], [], marker="o", ls="", color="C0", mec="black", mew=0.5,
               label="above the line (model higher)"),
        Line2D([], [], ls="--", color="grey", label="y = x (equal recall)"),
    ]
    a1.legend(handles=handles, loc="upper left", fontsize=7, framealpha=0.95)
    s = d["single"]
    fig.suptitle(f"The model's top-10 recall is below truth-vs-truth recall on "
                 f"{s['below']} of {d['n']} pairs (above on {s['above']}, level "
                 f"on {s['on']})", fontsize=11, x=0.5, y=0.975)
    _source_note(fig, f"source: results/latent_truth/per_pair.csv  |  {d['n']} "
                      "replicate pairs, k = 10 picks (recall takes multiples of "
                      "0.05: marker area = number of pairs at that point, count "
                      "printed where > 1)  |  counts only, no test is drawn here")
    _save(fig, "model_vs_truth_vs_truth.png")


# ---- d) position clustering decides the RASK sign ---------------------------

# The variant-level interval of the first RASK run, kept verbatim: results/
# epistasis_replicated/gaps.csv as committed in 081e21b, before the intervals
# were re-drawn by position (fa74eb4). The point estimate is identical to the
# current z-pool row; only the resampling unit differs.
SUPERSEDED_GAPS = REP / "gaps_variant_level_081e21b.csv"


def resampling_unit_data():
    """RASK singles-minus-doubles gap, z-pool target, by resampling unit.

    Variant level: the superseded first-run artefact. Position level: the
    current gaps.csv. Same data, same statistic, same seed -- the figure stops
    if the two point estimates differ, because then they would not be the same
    gap.
    """
    cur = pd.read_csv(_need(REP / "gaps.csv"))
    old = pd.read_csv(_need(SUPERSEDED_GAPS))
    c = cur[(cur["protein"] == "RASK_HUMAN") & (cur["target"] == "pooled_z")]
    o = old[old["protein"] == "RASK_HUMAN"]
    _agree([
        (len(c) == 1 and len(o) == 1, "one RASK z-pool row in each file"),
        (abs(float(c["gap_spearman"].iloc[0] - o["gap_spearman"].iloc[0])) < 1e-6,
         "Spearman gap point estimate"),
        (abs(float(c["gap_recall"].iloc[0] - o["gap_recall"].iloc[0])) < 1e-6,
         "top-decile recall gap point estimate"),
        (int(c["n_doubles"].iloc[0]) == int(o["n_doubles"].iloc[0]),
         "number of doubles"),
    ], "d (resampling unit)", "results/epistasis_replicated/gaps.csv")
    c, o = c.iloc[0], o.iloc[0]
    return {
        "n_singles": int(c["n_singles"]), "n_doubles": int(c["n_doubles"]),
        "positions": int(c["n_position_clusters"]),
        "spearman": {"point": float(c["gap_spearman"]),
                     "variant": (float(o["gap_spearman_lo"]), float(o["gap_spearman_hi"])),
                     "position": (float(c["gap_spearman_lo"]), float(c["gap_spearman_hi"]))},
        "recall": {"point": float(c["gap_recall"]),
                   "variant": (float(o["gap_recall_lo"]), float(o["gap_recall_hi"])),
                   "position": (float(c["gap_recall_lo"]), float(c["gap_recall_hi"]))},
    }


def draw_unit_forest(ax, d, metric, xlabel, title_fs=9.5, note_fs=8):
    """Two intervals for one gap: variant-level (top) and position-level."""
    m = d[metric]
    rows = (("variant-level bootstrap\n(doubles treated as independent)",
             m["variant"], "C0", 1),
            (f"position-clustered bootstrap\n({d['positions']} positions drawn)",
             m["position"], "C3", 0))
    ax.axvline(0, color="grey", ls="--", lw=1.3, zorder=1)
    for label, (lo, hi), color, y in rows:
        ax.hlines(y, lo, hi, color=color, lw=5, zorder=2, capstyle="butt")
        ax.scatter([m["point"]], [y], s=70, color="white", edgecolor="black",
                   linewidth=1.0, zorder=4)
        excl = (lo > 0) or (hi < 0)
        ax.text(hi + 0.012, y, f"[{_signed(lo)}, {_signed(hi)}]\n"
                f"{'excludes' if excl else 'includes'} 0", fontsize=note_fs,
                va="center", ha="left", color="black")
    ax.set_yticks([r[3] for r in rows])
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.7, 1.6)
    lo_all = min(r[1][0] for r in rows)
    hi_all = max(r[1][1] for r in rows)
    ax.set_xlim(lo_all - 0.03, hi_all + 0.17)
    ax.set_xlabel(xlabel)
    # the two markers sit at the same x: join them so that is visible
    ax.vlines(m["point"], 0, 1, color="black", ls=":", lw=1.0, zorder=3)
    ax.text(m["point"] + 0.004, 0.5, f"same point estimate\n{_signed(m['point'])}",
            ha="left", va="center", fontsize=7.5)
    ax.grid(axis="x", color="0.93", lw=0.5)
    ax.set_axisbelow(True)


def fig_resampling_unit():
    """d) Resampling variants instead of positions flips the verdict on RASK."""
    d = resampling_unit_data()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True,
                                 gridspec_kw=dict(left=0.205, right=0.985,
                                                  top=0.8, bottom=0.27,
                                                  wspace=0.06))
    draw_unit_forest(a1, d, "spearman",
                     "gap = Spearman(singles) \u2212 Spearman(doubles)\n"
                     "\u2190 doubles rank better        singles rank better \u2192")
    draw_unit_forest(a2, d, "recall",
                     "gap = top-decile recall(singles) \u2212 recall(doubles)\n"
                     "\u2190 doubles rank better        singles rank better \u2192")
    a1.set_title("Spearman \u03c1: the interval crosses zero once positions are the unit",
                 fontsize=9.5)
    a2.set_title("Top-decile recall: excludes zero either way (barely, by position)",
                 fontsize=9.5)
    fig.suptitle("Resampling positions instead of variants moves RASK's rank-correlation "
                 "gap from 'excludes zero' to 'includes zero'", fontsize=11, x=0.5,
                 y=0.975)
    _source_note(fig, "source: results/epistasis_replicated/gaps.csv (position level, "
                      "z-pool target, 1,000 draws) and its superseded first run "
                      "gaps_variant_level_081e21b.csv (variant level)\n"
                      f"RASK_HUMAN, {d['n_singles']:,} shared singles and "
                      f"{d['n_doubles']:,} shared doubles on {d['positions']} positions  |  "
                      "SPG1 is not drawn: its first run used a different join, so its two "
                      "intervals are not for the same gap", y=0.012)
    _save(fig, "rask_resampling_unit.png")


# ---- e) range matching removes most of the gap ------------------------------

def _short_assay(name):
    parts = name.split("_")
    return f"{parts[0]} ({parts[-2]} {parts[-1]})"


def range_match_data():
    """Raw and range-matched singles-minus-multi gaps with position CIs."""
    five = pd.read_csv(_need(EPI / "paired_gaps.csv"))
    rep = pd.read_csv(_need(REP / "gaps.csv"))
    man = json.loads(_need(REP / "manifest.json").read_text())
    names = {}
    for key, p in man["pairs"].items():
        for slot in ("a", "b"):
            parts = p[f"assay_{slot}"].split("_")[2:]
            names[(p["protein"], f"assay_{slot}")] = " ".join(parts)
    # short labels for the four targets (the full assay ids are in manifest.json)
    short = {"Weng 2022 abundance": "abundance", "Weng 2022 binding-DARPin K55": "DARPin binding"}
    names = {k: short.get(v, v) for k, v in names.items()}
    left = [(_short_assay(r.assay), r.gap_rho, r.gap_lo, r.gap_hi,
             r.range_matched_gap, r.range_matched_gap_lo,
             r.range_matched_gap_hi) for r in five.itertuples()]
    label = {"pooled_rank": "rank pool (primary)", "pooled_z": "z-pool"}
    right = []
    for r in rep.itertuples():
        t = label.get(r.target) or f"{names[(r.protein, r.target)]} alone"
        right.append((f"{r.protein.split('_')[0]}: {t}", r.gap_spearman,
                      r.gap_spearman_lo, r.gap_spearman_hi, r.range_matched_gap,
                      r.range_matched_gap_lo, r.range_matched_gap_hi,
                      r.protein))
    mean_raw = float(five["gap_rho"].mean())
    mean_rm = float(five["range_matched_gap"].mean())
    z = rep[(rep["protein"] == "RASK_HUMAN") & (rep["target"] == "pooled_z")].iloc[0]
    return {"left": left, "right": right, "mean_raw": mean_raw,
            "mean_rm": mean_rm, "rask_raw": float(z["gap_spearman"]),
            "rask_rm": float(z["range_matched_gap"]),
            "spg1_positions": int(rep[rep["protein"] == "SPG1_STRSG"]["n_position_clusters"].iloc[0])}


def draw_gap_dumbbell(ax, rows, xlabel, hatch_proteins=()):
    """Raw gap (blue) and range-matched gap (red), each with its position CI."""
    n = len(rows)
    ax.axvline(0, color="grey", ls="--", lw=1.3, zorder=1)
    for i, r in enumerate(rows):
        y = n - 1 - i
        raw, rlo, rhi, rm, mlo, mhi = r[1:7]
        faint = len(r) > 7 and r[7] in hatch_proteins
        a = 0.45 if faint else 1.0
        ax.annotate("", xy=(rm, y - 0.17), xytext=(raw, y + 0.17),
                    arrowprops=dict(arrowstyle="-|>", color="0.45", lw=1.1,
                                    shrinkA=4, shrinkB=4, alpha=a), zorder=2)
        ax.hlines(y + 0.17, rlo, rhi, color="C0", lw=2.2, alpha=a, zorder=2)
        ax.hlines(y - 0.17, mlo, mhi, color="C3", lw=2.2, alpha=a, zorder=2)
        ax.scatter([raw], [y + 0.17], s=34, color="C0", edgecolor="black",
                   linewidth=0.5, alpha=a, zorder=4)
        ax.scatter([rm], [y - 0.17], s=34, color="C3", edgecolor="black",
                   linewidth=0.5, alpha=a, zorder=4)
        ax.text(rhi + 0.012, y + 0.17, _signed(raw), fontsize=6.5, va="center",
                color="C0", alpha=1.0)
        ax.text(mhi + 0.012, y - 0.17, _signed(rm), fontsize=6.5, va="center",
                color="C3", alpha=1.0)
    ax.set_yticks(range(n))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=7.5)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlim(min(r[2] for r in rows) - 0.05, max(r[3] for r in rows) + 0.17)
    ax.grid(axis="x", color="0.93", lw=0.5)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, fontsize=8.5)


def fig_range_matching():
    """e) Matching the fitness range shrinks most of the gap."""
    d = range_match_data()
    fig, (a1, a2) = plt.subplots(
        1, 2, figsize=(11, 5.0), gridspec_kw=dict(left=0.125, right=0.985, top=0.8,
                                                  bottom=0.3, wspace=0.62,
                                                  width_ratios=[1, 1.15]))
    draw_gap_dumbbell(a1, d["left"], "gap = Spearman(singles) \u2212 Spearman(multi)")
    draw_gap_dumbbell(a2, d["right"],
                      "gap = Spearman(singles) \u2212 Spearman(doubles)",
                      hatch_proteins=("SPG1_STRSG",))
    a1.set_title(f"Five single assays: mean gap {_signed(d['mean_raw'])} "
                 f"\u2192 {_signed(d['mean_rm'])}", fontsize=9.5)
    a2.set_title(f"Two replicated pairs: RASK z-pool {_signed(d['rask_raw'])} "
                 f"\u2192 {_signed(d['rask_rm'])}", fontsize=9.5)
    handles = [
        Line2D([], [], marker="o", ls="-", color="C0", mec="black", mew=0.5,
               label="raw gap, singles vs all multi-mutants (or doubles)"),
        Line2D([], [], marker="o", ls="-", color="C3", mec="black", mew=0.5,
               label="range-matched gap (both strata trimmed to the fitness window they share)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=7.5,
               frameon=False, bbox_to_anchor=(0.5, 0.095))
    fig.suptitle("Matching the fitness range shrinks the gap on average (and on RASK "
                 "and HIS7) but widens it on GRB2 and GFP", fontsize=11, x=0.5,
                 y=0.975)
    _source_note(fig, "source: results/epistasis/paired_gaps.csv (left), "
                      "results/epistasis_replicated/gaps.csv (right)  |  bars = 95% "
                      "position-bootstrap intervals (resampling unit: position)\n"
                      "gap > 0 is the direction the independent-site hypothesis predicts "
                      "(model worse on multi-mutants)  |  faint SPG1 rows: "
                      f"{d['spg1_positions']} positions, not interpretable", y=0.012)
    _save(fig, "range_matching_gap.png")


# ---- f) family-level vs checkpoint-level test --------------------------------

def _load_family_script():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "run_family_analysis", ROOT / "scripts" / "run_family_analysis.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def family_data():
    """Per-checkpoint viral gaps grouped by family, and both p-values.

    Checkpoint gaps come from per_method_viral_gap.csv, the family assignment
    from scripts/run_family_analysis.py (the same dictionaries the test of
    record uses). The recomputed counts, medians, U and p must reproduce
    family_separation.csv before anything is drawn.
    """
    from scipy import stats
    mod = _load_family_script()
    pm = pd.read_csv(_need(FAM / "per_method_viral_gap.csv")).set_index("method")["gap"]
    fam = pd.read_csv(_need(FAM / "family_viral_gap.csv"))
    sep = pd.read_csv(_need(FAM / "family_separation.csv")).iloc[0]
    rows = []
    for family, methods in mod.FAMILIES.items():
        for m in methods:
            if m in pm.index:
                rows.append({"family": family, "method": m, "gap": float(pm[m]),
                             "reads_alignment": family not in mod.NO_ALIGNMENT})
    ck = pd.DataFrame(rows)
    fam = fam.assign(reads_alignment=fam["reads_alignment_of_the_family"].astype(bool))
    no_c = ck.loc[~ck["reads_alignment"], "gap"]
    yes_c = ck.loc[ck["reads_alignment"], "gap"]
    u_c, p_c = stats.mannwhitneyu(no_c, yes_c, alternative="less")
    no_f = fam.loc[~fam["reads_alignment"], "gap"]
    yes_f = fam.loc[fam["reads_alignment"], "gap"]
    u_f, p_f = stats.mannwhitneyu(no_f, yes_f, alternative="less", method="exact")
    _agree([
        (len(no_c) == sep["n_no_alignment"] and len(yes_c) == sep["n_alignment"],
         "checkpoint counts"),
        (abs(u_c - sep["mannwhitney_u"]) < 1e-9, "checkpoint-level U"),
        (abs(p_c - sep["p_no_alignment_worse"]) < 1e-9, "checkpoint-level p"),
        (len(no_f) == sep["n_families_no_alignment"]
         and len(yes_f) == sep["n_families_alignment"], "family counts"),
        (abs(u_f - sep["mannwhitney_u_family_level"]) < 1e-9, "family-level U"),
        (abs(p_f - sep["p_family_level_exact"]) < 1e-9, "family-level p"),
        (abs(no_f.median() - sep["median_family_gap_no_alignment"]) < 1e-9,
         "family median, no alignment"),
        (abs(yes_f.median() - sep["median_family_gap_alignment"]) < 1e-9,
         "family median, alignment"),
    ], "f (family test)", "results/families/family_separation.csv")
    return {"ck": ck, "fam": fam, "sep": sep}


def fig_family_test_unit():
    """f) The same data tested with two different units."""
    d = family_data()
    ck, fam, sep = d["ck"], d["fam"], d["sep"]
    order = fam.sort_values("gap", ascending=False).reset_index(drop=True)
    fig = plt.figure(figsize=(11, 5.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1], left=0.265, right=0.985,
                          top=0.8, bottom=0.29, wspace=0.12)
    ax = fig.add_subplot(gs[0, 0])
    n = len(order)
    ax.axvline(0, color="grey", ls="--", lw=1.2, zorder=1)
    for i, r in order.iterrows():
        y = n - 1 - i
        color = "C0" if r["reads_alignment"] else "C3"
        g = ck.loc[ck["family"] == r["family"], "gap"].to_numpy()
        ax.scatter(g, np.full(len(g), y), s=22, color=color, alpha=0.55,
                   edgecolor="black", linewidth=0.35, zorder=3)
        ax.scatter([r["gap"]], [y], s=85, marker="D", color=color,
                   edgecolor="black", linewidth=0.9, zorder=4)
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"{f}  (n = {int(m)})" for f, m in
                        zip(order["family"], order["methods"])][::-1], fontsize=7.5)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", color="0.94", lw=0.5)
    ax.set_axisbelow(True)
    ax.set_xlabel("viral gap = mean Spearman on viral assays \u2212 on non-viral assays")
    ax.set_title("14 families, 42 checkpoints (small dot = one checkpoint, "
                 "diamond = family mean)", fontsize=9)
    ax.legend(handles=[
        Line2D([], [], marker="D", ls="", color="C3", mec="black", ms=7,
               label=f"does not read an alignment ({int(sep['n_families_no_alignment'])} families, "
                     f"{int(sep['n_no_alignment'])} checkpoints)"),
        Line2D([], [], marker="D", ls="", color="C0", mec="black", ms=7,
               label=f"reads an alignment ({int(sep['n_families_alignment'])} families, "
                     f"{int(sep['n_alignment'])} checkpoints)")],
        loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1, fontsize=7.5,
        frameon=False)

    axp = fig.add_subplot(gs[0, 1])
    tests = (("unit = checkpoint, "
              f"{int(sep['n_no_alignment'])} vs {int(sep['n_alignment'])}\n"
              "(counts one family many times)", float(sep["p_no_alignment_worse"]), "C7"),
             ("unit = family, "
              f"{int(sep['n_families_no_alignment'])} vs {int(sep['n_families_alignment'])}\n"
              "(the unit that varies: test of record)", float(sep["p_family_level_exact"]), "C3"))
    for i, (label, p, color) in enumerate(tests):
        y = 1 - i
        axp.scatter([p], [y], s=110, color=color, edgecolor="black", linewidth=0.8,
                    zorder=4, marker="o")
        axp.text(p, y + 0.2, f"p = {p:.5f}" if p < 0.001 else f"p = {p:.4f}",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")
        axp.text(0.5, y - 0.2, label, transform=axp.get_yaxis_transform(),
                 ha="center", va="top", fontsize=7.5)
    axp.axvline(0.05, color="black", ls="--", lw=1.2, zorder=2)
    axp.set_xscale("log")
    axp.set_xlim(1e-4, 1)
    axp.set_ylim(-0.75, 1.75)
    axp.set_yticks([])
    axp.grid(axis="x", color="0.93", lw=0.5)
    axp.set_axisbelow(True)
    axp.text(0.05, 1.7, " p = 0.05", fontsize=8, va="top", ha="left")
    axp.set_xlabel("p-value (log axis)")
    axp.set_title("The same data, two units, two p-values", fontsize=9)
    fig.suptitle(f"Tested per family the viral-gap separation is p = "
                 f"{sep['p_family_level_exact']:.4f}, not the "
                 f"p = {sep['p_no_alignment_worse']:.5f} that counting checkpoints gave",
                 fontsize=11, x=0.5, y=0.975)
    _source_note(fig, "source: results/families/per_method_viral_gap.csv, "
                      "family_viral_gap.csv, family_separation.csv  |  ProteinGym "
                      "published per-assay Spearman, 217 assays\n"
                      "median family gap "
                      f"{_signed(sep['median_family_gap_no_alignment'])} (no alignment) vs "
                      f"{_signed(sep['median_family_gap_alignment'])} (alignment), U = "
                      f"{sep['mannwhitney_u_family_level']:.0f} (families); median checkpoint gap "
                      f"{_signed(sep['median_gap_no_alignment'])} vs "
                      f"{_signed(sep['median_gap_alignment'])}, U = "
                      f"{sep['mannwhitney_u']:.0f} (checkpoints)", y=0.012)
    _save(fig, "family_test_unit.png")


# ---- g) the HIS7 residual exceeds the total variance --------------------------

def residual_data():
    """Per assay: sd of the non-additive residual vs sd of the doubles."""
    ps = pd.read_csv(_need(EPI / "per_stratum_all.csv"))
    a = ps.drop_duplicates("assay").set_index("assay")
    cols = ["nonadditive_residual_sd_z", "double_total_sd_z",
            "nonadditive_residual_var_share", "residual_exceeds_total",
            "n_doubles_with_singles"]
    a = a[cols].copy()
    a["share_check"] = (a["nonadditive_residual_sd_z"] / a["double_total_sd_z"]) ** 2
    _agree([
        (len(a) == 5, "five assays"),
        (bool((abs(a["share_check"] - a["nonadditive_residual_var_share"]) < 1e-6).all()),
         "variance share = (residual sd / total sd)^2"),
        (bool(((a["nonadditive_residual_var_share"] > 1)
               == a["residual_exceeds_total"].astype(bool)).all()),
         "residual_exceeds_total flag"),
    ], "g (residual vs total variance)", "results/epistasis/per_stratum_all.csv")
    singles = ps[ps["stratum"] == "single"].set_index("assay")["n"]
    a["n_singles"] = singles.reindex(a.index)
    return a.sort_values("nonadditive_residual_var_share", ascending=False)


def fig_his7_residual():
    """g) On HIS7 the 'epistatic' residual is larger than the doubles' spread."""
    a = residual_data()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 5.0),
                                 gridspec_kw=dict(left=0.21, right=0.985, top=0.8,
                                                  bottom=0.27, wspace=0.08,
                                                  width_ratios=[1.2, 1]),
                                 sharey=True)
    n = len(a)
    ys = np.arange(n)[::-1]
    h = 0.34
    for y, (assay, r) in zip(ys, a.iterrows()):
        flag = bool(r["residual_exceeds_total"])
        a1.barh(y + h / 2, r["double_total_sd_z"], height=h, color="0.78",
                edgecolor="black", linewidth=0.5, zorder=3)
        a1.barh(y - h / 2, r["nonadditive_residual_sd_z"], height=h,
                color="C3" if flag else "C0", edgecolor="black", linewidth=0.5,
                zorder=3)
        a1.text(r["double_total_sd_z"] + 0.02, y + h / 2,
                f"{r['double_total_sd_z']:.2f}", va="center", fontsize=7.5)
        a1.text(r["nonadditive_residual_sd_z"] + 0.02, y - h / 2,
                f"{r['nonadditive_residual_sd_z']:.2f}", va="center", fontsize=7.5,
                fontweight="bold" if flag else "normal")
        a2.barh(y, r["nonadditive_residual_var_share"], height=0.5,
                color="C3" if flag else "C0", edgecolor="black", linewidth=0.5,
                zorder=3)
        a2.text(r["nonadditive_residual_var_share"] + 0.02, y,
                f"{r['nonadditive_residual_var_share']:.3f}"
                + ("  > 1" if flag else ""), va="center", fontsize=8,
                fontweight="bold" if flag else "normal")
    a1.set_yticks(ys)
    a1.set_yticklabels([f"{_short_assay(i)}\n{int(r['n_doubles_with_singles']):,} doubles "
                        f"with both singles" for i, r in a.iterrows()], fontsize=7.5)
    a1.tick_params(axis="y", length=0)
    a2.axvline(1.0, color="black", ls="--", lw=1.4, zorder=2)
    a2.text(1.03, 1.2, "a share above 1 is impossible\nfor a real variance component:\n"
            "the residual is mostly noise", fontsize=7.5, va="center", ha="left")
    a2.set_xlim(0, 1.75)
    a1.set_xlim(0, float(a["double_total_sd_z"].max()) * 1.2)
    a2.set_ylim(-0.6, n - 0.1)
    for ax in (a1, a2):
        ax.grid(axis="x", color="0.93", lw=0.5)
        ax.set_axisbelow(True)
    a1.set_xlabel("standard deviation, z-scored fitness units")
    a2.set_xlabel("residual variance / variance of the doubles")
    a1.legend(handles=[
        Line2D([], [], marker="s", ls="", color="0.78", mec="black", ms=8,
               label="sd of the measured doubles (total)"),
        Line2D([], [], marker="s", ls="", color="C0", mec="black", ms=8,
               label="sd of the non-additive residual"),
        Line2D([], [], marker="s", ls="", color="C3", mec="black", ms=8,
               label="residual larger than the total")],
        loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, fontsize=7.5,
        frameon=False)
    a1.set_title("The residual is an upper bound on epistasis: it also holds noise",
                 fontsize=9)
    a2.set_title("Share of the doubles' variance left as residual", fontsize=9)
    his = a.loc[[i for i in a.index if i.startswith("HIS7")][0]]
    fig.suptitle(f"On HIS7 the non-additive 'residual' (sd {his['nonadditive_residual_sd_z']:.2f}) "
                 f"is larger than the doubles' whole spread (sd {his['double_total_sd_z']:.2f}), "
                 "so it is mostly noise, not epistasis", fontsize=11, x=0.5, y=0.975)
    _source_note(fig, "source: results/epistasis/per_stratum_all.csv (nonadditive_residual_sd_z, "
                      "double_total_sd_z, nonadditive_residual_var_share)  |  doubles whose two "
                      "singles were also assayed\n"
                      "additive prediction = sum of the two singles' z-scores  |  "
                      f"HIS7 has {int(his['n_singles'])} singles; ProteinGym carries no per-variant "
                      "error, so measurement noise cannot be subtracted from the residual", y=0.012)
    _save(fig, "his7_residual_vs_variance.png")


# ---- at a glance: the four panels that carry the conclusion -------------------

def fig_at_a_glance():
    """(a)-(d) in one image, for the top of the README. Same drawing code and
    same data checks as the four standalone figures; only the layout is compact."""
    per, prot, summ = rank_gain_data()
    prow = pvalue_data()
    ceil = ceiling_data()
    unit = resampling_unit_data()
    fig = plt.figure(figsize=(11.5, 10.4))
    gs = fig.add_gridspec(2, 2, left=0.085, right=0.985, top=0.9, bottom=0.115,
                          wspace=0.42, hspace=0.62)
    n_pairs, n_prot = len(per), len(prot)
    n_up = int((per["gain"] > 0).sum())

    ax = fig.add_subplot(gs[0, 0])
    draw_rank_dumbbell(ax, per, label_size=5.8, compact=True)
    ax.set_title(f"(a) Pooling two assays raises the model's Spearman\n"
                 f"in {n_up} of {n_pairs} pairs (median gain "
                 f"{_signed(per['gain'].median())})", fontsize=10)

    ax = fig.add_subplot(gs[0, 1])
    prim = [r for r in prow if r["primary_subset"]]
    rows = []
    for which, color, name in (("recall", "C3", "top-10 recall"),
                               ("spearman", "C0", "Spearman \u03c1")):
        for k, r in enumerate(prim):
            rows.append({"label": f"{name}\n{r['level']} level, n = {r['n']}",
                         "ps": r[which], "color": color, "annotate": True,
                         "gap": 0.5 if (which == "spearman" and k == 0) else 0.0})
    draw_pvalue_rows(ax, rows, log=True, label_size=7.5, text_size=6.5)
    ax.set_xlim(1e-10, 1.0)
    ax.set_xlabel("p-value of the gain (log axis)")
    ticks = ax.get_yticks()      # row positions, top to bottom
    n_rec = len(prim)
    ygap = float((ticks[n_rec - 1] + ticks[n_rec]) / 2)   # the gap between the metrics
    ax.text(0.05, ygap, "p = 0.05 ", fontsize=8, rotation=90, va="center", ha="right")
    ax.legend(handles=[Line2D([], [], marker=mk, ls="", color="0.8", mec="black",
                              mew=0.5, ms=6, label=name) for _, name, mk in P_METHODS],
              loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=3,
              fontsize=6.5, frameon=False, handletextpad=0.2, columnspacing=0.9)
    rec = [p for r in prow for p in r["recall"]]
    ax.set_title("(b) The gain in top-10 recall is not significant by any\n"
                 f"test (smallest p = {min(rec):.3f}); the Spearman gain is by all",
                 fontsize=10)

    ax = fig.add_subplot(gs[1, 0])
    draw_ceiling(ax, ceil["single"], "model top-10 recall\nvs one assay")
    s = ceil["single"]
    ax.set_title(f"(c) The model's top-10 recall is below truth-vs-truth\n"
                 f"on {s['below']} of {ceil['n']} pairs (above on {s['above']}, level on {s['on']})",
                 fontsize=10)

    ax = fig.add_subplot(gs[1, 1])
    draw_unit_forest(ax, unit, "spearman",
                     "gap = Spearman(singles) \u2212 Spearman(doubles)\n"
                     "\u2190 doubles rank better     singles rank better \u2192")
    ax.set_title("(d) RASK: the same gap, resampled by variant vs by position;\n"
                 "the interval only crosses zero when positions are the unit", fontsize=10)

    fig.suptitle("At a glance: pooling two assays orders the variants better in every pair, "
                 "but the top-of-list gain is marginal", fontsize=12, x=0.5, y=0.975)
    _source_note(fig, "a, b, c: results/latent_truth/per_pair.csv + summary.json "
                      f"({n_pairs} replicate pairs, {n_prot} proteins)\n"
                      "d: results/epistasis_replicated/gaps.csv + its superseded variant-level run "
                      f"(RASK_HUMAN, {unit['n_doubles']:,} doubles on {unit['positions']} positions)  |  "
                      "each panel is one of the standalone figures, drawn by the same code", y=0.01)
    _save(fig, "at_a_glance.png")


FINDINGS = {
    "rank_gain": fig_rank_gain,
    "recall_vs_rank_p": fig_recall_vs_rank_p,
    "model_vs_ceiling": fig_model_vs_ceiling,
    "resampling_unit": fig_resampling_unit,
    "range_matching": fig_range_matching,
    "family_test_unit": fig_family_test_unit,
    "his7_residual": fig_his7_residual,
    "at_a_glance": fig_at_a_glance,
}


def main():
    ap = argparse.ArgumentParser(description="Figures from finished runs.")
    ap.add_argument("--only", choices=["all", "benchmark", "findings"],
                    default="all")
    ap.add_argument("--figure", choices=sorted(FINDINGS), default=None,
                    help="draw one findings figure by name")
    args = ap.parse_args()
    if args.figure:
        FINDINGS[args.figure]()
        return
    if args.only in ("all", "benchmark"):
        df = load_results()
        fig_method_comparison(df)
        fig_vs_official(df)
        fig_by_taxon(df)
        fig_landscape()
        print(f"figures written to {FIG}")
    if args.only in ("all", "findings"):
        for fn in FINDINGS.values():
            fn()


if __name__ == "__main__":
    main()
