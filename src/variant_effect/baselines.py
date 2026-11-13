"""Cheap baselines, so the language model has something to beat.

A zero-shot Spearman of 0.4 sounds impressive until you find out that a
substitution matrix lookup with no model at all gets 0.3 on the same assay.
These are here to make that comparison automatic rather than optional.

All scorers follow the same convention as the ESM scores: higher means
predicted to be more tolerated.
"""

import numpy as np

from .mutations import parse_variant

# Kyte-Doolittle hydropathy
HYDROPATHY = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5,
    "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9,
    "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9,
    "Y": -1.3, "V": 4.2,
}

# side chain volume, cubic angstroms (Zamyatnin 1972)
VOLUME = {
    "A": 88.6, "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5, "Q": 143.8,
    "E": 138.4, "G": 60.1, "H": 153.2, "I": 166.7, "L": 166.7, "K": 168.6,
    "M": 162.9, "F": 189.9, "P": 112.7, "S": 89.0, "T": 116.1, "W": 227.8,
    "Y": 193.6, "V": 140.0,
}

CHARGE = {aa: 0 for aa in "ACDEFGHIKLMNPQRSTVWY"}
CHARGE.update({"D": -1, "E": -1, "K": 1, "R": 1, "H": 0.1})


def _blosum62():
    """BLOSUM62 as a dict of (a, b) -> score, via Biopython."""
    from Bio.Align import substitution_matrices
    m = substitution_matrices.load("BLOSUM62")
    return {(a, b): float(m[a, b]) for a in m.alphabet for b in m.alphabet}


_B62 = None


def blosum62_score(sequence, variants):
    """Sum of BLOSUM62 substitution scores.

    Position-blind: it knows nothing about where in the protein the residue
    sits, only how often that swap is seen in aligned homologous blocks.
    The gap between this and a language model is a reasonable measure of how
    much the model is getting out of context.
    """
    global _B62
    if _B62 is None:
        _B62 = _blosum62()

    wts, _, muts = _parse_table(variants)
    out = np.zeros(len(variants), dtype=np.float32)
    for i, (ws, ms) in enumerate(zip(wts, muts)):
        total = 0.0
        for w, m in zip(ws, ms):
            total += _B62.get((w, m), _B62.get((m, w), 0.0))
        out[i] = total
    return out


def _parse_table(variants):
    """Vectorised parse of variant notation into (wt, pos, mut) arrays.

    The baselines are called on every assay in the benchmark, and ProteinGym's
    multi-mutant assays carry up to 3.4 million substitutions in one file
    (HIS7_YEAST has 496,137 variants averaging seven substitutions each). A
    Python loop over ``parse_variant`` costs minutes per assay there, which is
    what stalled the full sweep; a single regex pass over the joined strings
    costs milliseconds.

    Returns three lists-of-lists, one entry per variant, so the per-variant
    sums stay exact for multi-mutants.
    """
    import re

    pattern = re.compile(r"([A-Z])(\d+)([A-Z*])")
    wts, poss, muts = [], [], []
    for v in variants:
        found = pattern.findall(str(v))
        wts.append([m[0] for m in found])
        poss.append([int(m[1]) for m in found])
        muts.append([m[2] for m in found])
    return wts, poss, muts


def _property_delta(variants, table, signed=False):
    wts, _, muts = _parse_table(variants)
    out = np.zeros(len(variants), dtype=np.float32)
    for i, (ws, ms) in enumerate(zip(wts, muts)):
        total = 0.0
        for w, m in zip(ws, ms):
            d = table.get(m, 0.0) - table.get(w, 0.0)
            total += d if signed else -abs(d)
        out[i] = total
    return out


def hydropathy_change(sequence, variants):
    """Negative absolute change in Kyte-Doolittle hydropathy.

    Big hydrophobicity swaps are destabilising more often than not, so
    smaller change scores higher.
    """
    return _property_delta(variants, HYDROPATHY)


def volume_change(sequence, variants):
    """Negative absolute change in side chain volume."""
    return _property_delta(variants, VOLUME)


def charge_change(sequence, variants):
    """Negative absolute change in formal charge at pH 7."""
    return _property_delta(variants, CHARGE)


def proline_glycine_penalty(sequence, variants):
    """Crude structural heuristic.

    Introducing proline breaks backbone hydrogen bonding and restricts phi;
    losing glycine removes backbone flexibility that is often needed in
    tight turns. Both are classic destabilising moves, and a surprising
    fraction of what any predictor gets right on a thermostability assay is
    just this.
    """
    wts, _, muts = _parse_table(variants)
    out = np.zeros(len(variants), dtype=np.float32)
    for i, (ws, ms) in enumerate(zip(wts, muts)):
        penalty = 0.0
        for w, m in zip(ws, ms):
            if m == "P":
                penalty -= 1.0
            if w == "G":
                penalty -= 1.0
            if w == "P":
                penalty -= 0.5
            if w == "C":           # likely disulfide
                penalty -= 0.5
        out[i] = penalty
    return out


def random_baseline(sequence, variants, seed=0):
    """Pure noise. The floor -- anything not clearly above this is nothing."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(len(variants)).astype(np.float32)


BASELINES = {
    "blosum62": blosum62_score,
    "hydropathy": hydropathy_change,
    "volume": volume_change,
    "charge": charge_change,
    "pro_gly": proline_glycine_penalty,
    "random": random_baseline,
}
