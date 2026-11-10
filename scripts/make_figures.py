"""Figures from a finished benchmark run.

Run after scripts/run_benchmark.py:
    python scripts/make_figures.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


if __name__ == "__main__":
    df = load_results()
    fig_method_comparison(df)
    fig_vs_official(df)
    fig_by_taxon(df)
    fig_landscape()
    print(f"figures written to {FIG}")
