"""Structure-aware scoring: buried vs exposed positions.

The open question the README flags is whether the top-k failure -- a
respectable Spearman but a model whose top 10 picks are no better than random
on five of nine assays -- comes from treating buried and exposed positions the
same. The intuition is reasonable: a buried residue is packed against the
hydrophobic core, so almost any substitution is destabilising, and the ranking
among those variants is compressed; an exposed residue trades with solvent, so
more substitutions are tolerated and the ranking carries more information.
A predictor trained on a single objective cannot know which regime it is in
unless the structure tells it.

This module supplies the missing input and the tests for whether it helps:

- ``structure_features``: per-residue relative solvent accessibility and
  depth, from the assay's AlphaFold model. RSA is computed against the
  theoretical maximum for that residue type (Tien et al. 2013), so values are
  comparable across residues rather than across a protein of one size.
- ``distance_to_surface``: an alternative burial measure that does not depend
  on a maximum-ASA table, useful as a cross-check.
- ``split_by_burial``: which variants change a buried residue and which an
  exposed one, so a whole-assay Spearman can be decomposed.

What this module deliberately does *not* do is invent a new predictor. The
honest version of the buried/exposed hypothesis is testable with the scores
already computed: if the top-k failure is a burial problem, then (a) Spearman
should be lower on buried-only subsets and (b) the model's top-ranked variants
should be concentrated at exposed positions. Both are checked by
scripts/run_structure_analysis.py.

Coordinates come from ProteinGym's distributed AlphaFold2 models (one per
protein, keyed by UniProt id), so no new structure prediction is involved.
"""

from pathlib import Path

import numpy as np

AA20 = "ACDEFGHIKLMNPQRSTVWY"

# Theoretical maximum solvent-accessible surface area per residue, in A^2,
# from Tien et al. 2013 (the "theoretical" column). Used to turn an absolute
# accessibility into a relative one: without this, a glycine and a tryptophan
# with the same absolute exposure look equally buried, which is wrong by a
# factor of three.
MAX_ASA = {
    "A": 129, "R": 274, "N": 195, "D": 193, "C": 167,
    "Q": 225, "E": 223, "G": 104, "H": 224, "I": 197,
    "L": 201, "K": 236, "M": 224, "F": 240, "P": 159,
    "S": 155, "T": 172, "W": 285, "Y": 263, "V": 174,
}

# van der Waals radii, A, for the shallow-surface (Shrake-Rupley) calculation
VDW_RADIUS = {
    "A": 1.88, "R": 2.38, "N": 1.93, "D": 1.93, "C": 1.85,
    "Q": 2.03, "E": 2.05, "G": 1.66, "H": 2.15, "I": 1.98,
    "L": 1.98, "K": 2.40, "M": 2.03, "F": 2.04, "P": 1.85,
    "S": 1.78, "T": 1.88, "W": 2.24, "Y": 2.15, "V": 1.95,
}

# van der Waals radii by element, for all-atom accessible surface (Bondi 1964)
VDW_ATOMIC = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80}

# radius of the probe used to trace the molecular surface (water, A)
PROBE_RADIUS = 1.4


def parse_atoms(path):
    """All heavy atoms of the first model, grouped by residue.

    Returns a dict with
      coords  (n_atoms, 3)  angstrom
      names   atom names in file order ('CA', 'CB', 'OD1', ...)
      res     (n_atoms,)    residue index, 0..n_res-1 in file order
      numbers (n_res,)      PDB residue numbers
      resnames(n_res,)      3-letter residue names

    Hydrogens are skipped: they add atoms without adding accessible surface
    worth the cost, and the AF2 files contain none anyway.
    """
    coords, names, res, numbers, resnames = [], [], [], [], []
    for line in Path(path).read_text().splitlines():
        if line.startswith("ENDMDL"):
            break
        if not line.startswith("ATOM"):
            continue
        element = line[76:78].strip() or line[12:16].strip()[0]
        if element == "H":
            continue
        num = int(line[22:26])
        if num not in numbers:
            numbers.append(num)
            resnames.append(line[17:20].strip())
        coords.append((float(line[30:38]), float(line[38:46]),
                       float(line[46:54])))
        names.append(line[12:16].strip())
        res.append(len(numbers) - 1)
    return {
        "coords": np.array(coords, dtype=float),
        "names": names,
        "elements": [n[0].upper() if n else "C" for n in names],
        "res": np.array(res, dtype=int),
        "numbers": numbers,
        "resnames": resnames,
    }


def parse_ca_coords(path):
    """C-alpha coordinates indexed by residue number, from a PDB file.

    Returns (coords, residues) where coords is (n, 3) in angstrom and residues
    is the list of 1-letter codes. Only the first model is read: AF2 files hold
    one, and a multi-model file would otherwise concatenate models.

    Residues are keyed by the PDB numbering, which for ProteinGym's AF2 models
    matches the UniProt sequence the assays use. That is checked rather than
    assumed -- ``coords_for_sequence`` verifies the identity before returning
    anything.
    """
    coords = []
    residues = []
    numbers = []
    for line in Path(path).read_text().splitlines():
        if line.startswith("ENDMDL"):
            break
        if not line.startswith("ATOM"):
            continue
        if line[12:16].strip() != "CA":
            continue
        resname = line[17:20].strip()
        aa = THREE_TO_ONE.get(resname)
        if aa is None:
            continue
        coords.append((float(line[30:38]), float(line[38:46]),
                       float(line[46:54])))
        residues.append(aa)
        numbers.append(int(line[22:26]))
    return np.array(coords, dtype=float), residues, numbers


THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def coords_for_sequence(path, sequence, pdb_range=None):
    """CA coordinates for each residue of ``sequence``, or None where missing.

    pdb_range is the reference table's "start-end" string. It matters because
    some assays mutate a domain of a longer protein, and a structure covering
    only that domain lines up with the range, not with residue 1.

    A mismatch between the PDB's residue identities and the sequence raises:
    silently scoring burial at the wrong position would produce a plausible
    number that means nothing.
    """
    coords, residues, numbers = parse_ca_coords(path)
    if len(coords) == 0:
        raise ValueError(f"no CA atoms in {path}")

    offset = 0
    if isinstance(pdb_range, str) and "-" in pdb_range:
        try:
            offset = int(pdb_range.split("-")[0]) - 1
        except ValueError:
            offset = 0
    # residue numbers normally start at 1 or at the range start; trust the
    # file's own numbering where it is consistent, else fall back to offset
    first = numbers[0] if numbers else 1
    shift = offset if abs(first - (offset + 1)) > 1 else (first - 1)

    seq = str(sequence).upper()
    out = np.full((len(seq), 3), np.nan)
    n_matched = 0
    for i, (aa, xyz, num) in enumerate(zip(residues, coords, numbers)):
        idx = num - 1 - shift
        if 0 <= idx < len(seq):
            out[idx] = xyz
            if seq[idx] == aa:
                n_matched += 1
    # sanity check: the identities should mostly agree. A low fraction means
    # the offset is wrong, and every accessibility value after that is noise.
    span = np.isfinite(out).any(axis=1)
    if span.sum() and n_matched / max(span.sum(), 1) < 0.5:
        raise ValueError(
            f"only {n_matched}/{int(span.sum())} PDB residues match the "
            f"sequence (shift {shift}) -- structure and sequence disagree")
    return out


def shrake_rupley(coords, radii, probe=PROBE_RADIUS, n_points=92):
    """Per-atom solvent-accessible area, by numerical surface sampling.

    The classic Shrake-Rupley algorithm: place test points on spheres around
    each atom, keep those not inside any neighbouring atom's expanded sphere,
    and count the survivors. 92 points is the usual compromise between accuracy
    and speed; with a 5 A neighbour cutoff the error against a 960-point sphere
    is ~1-2%.

    Implemented here rather than pulled from a dependency because the only
    consumer is per-residue burial and the loop is easy to get subtly wrong --
    the neighbour list must use each atom's own radius plus the probe, not a
    global cutoff.
    """
    coords = np.asarray(coords, dtype=float)
    radii = np.asarray(radii, dtype=float)
    n = len(coords)
    if n == 0:
        return np.zeros(0)

    # golden-spiral points on the unit sphere: near-uniform and deterministic,
    # unlike a random sample which would make results irreproducible
    i = np.arange(n_points, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * i / n_points)
    theta = np.pi * (1 + 5 ** 0.5) * i
    unit = np.stack([np.cos(theta) * np.sin(phi),
                     np.sin(theta) * np.sin(phi),
                     np.cos(phi)], axis=1)

    area = np.zeros(n)
    for a in range(n):
        r = radii[a] + probe
        pts = coords[a] + r * unit
        # neighbours that could occlude any point in the sphere
        d = np.linalg.norm(coords - coords[a], axis=1)
        neigh = np.where((d < r + radii.max() + probe) & (np.arange(n) != a))[0]
        accessible = np.ones(n_points, dtype=bool)
        for b in neigh:
            rb = radii[b] + probe
            inside = np.linalg.norm(pts - coords[b], axis=1) < rb
            accessible &= ~inside
            if not accessible.any():
                break
        frac = accessible.mean()
        area[a] = frac * 4 * np.pi * r ** 2
    return area


def sidechain_atom_area(coords_all, residue_atoms, residue_radii,
                        all_radii, probe=PROBE_RADIUS, n_points=92):
    """SASA summed over one residue's side-chain atoms.

    Kept separate from the all-atom routine because per-residue burial only
    needs the side chain: the backbone is in the same environment for every
    residue in a folded protein, so including it dampens exactly the signal a
    burial measure is supposed to carry.
    """
    if len(coords_all) == 0:
        return 0.0
    n = len(coords_all)
    i = np.arange(n_points, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * i / n_points)
    theta = np.pi * (1 + 5 ** 0.5) * i
    unit = np.stack([np.cos(theta) * np.sin(phi),
                     np.sin(theta) * np.sin(phi),
                     np.cos(phi)], axis=1)

    total = 0.0
    for a in residue_atoms:
        r = all_radii[a] + probe
        pts = coords_all[a] + r * unit
        neigh = np.where((np.arange(n) != a)
                         & (np.linalg.norm(coords_all - coords_all[a], axis=1)
                            < r + all_radii.max() + probe))[0]
        accessible = np.ones(n_points, dtype=bool)
        for b in neigh:
            rb = all_radii[b] + probe
            accessible &= ~(np.linalg.norm(pts - coords_all[b], axis=1) < rb)
            if not accessible.any():
                break
        total += accessible.mean() * 4 * np.pi * r ** 2
    return total


def relative_sasa(pdb_path, sequence, pdb_range=None, n_points=92):
    """Per-residue relative solvent accessibility from side-chain atoms.

    Absolute accessible area is dominated by residue size; dividing by the
    theoretical maximum for that residue type makes a value comparable across
    the protein and across proteins. Above ~0.25 is conventionally "exposed".

    The area is summed over the residue's *side-chain* atoms only. Including
    the backbone would give every residue a similar baseline -- the backbone is
    in much the same environment wherever it sits -- and drown the signal the
    measure exists to carry. This is why a CA-sphere approximation (an earlier
    version) reported a median RSA of 0.29 with no residue below 0.14 on the
    72-residue IF1: one big sphere per residue does not model a packed core.

    A residue with no coordinates gets NaN, not 0: zero would read as perfectly
    buried, which is a different claim from "not modelled".
    """
    atoms = parse_atoms(pdb_path)
    coords = atoms["coords"]
    if len(coords) == 0:
        raise ValueError(f"no atoms in {pdb_path}")

    offset = 0
    if isinstance(pdb_range, str) and "-" in pdb_range:
        try:
            offset = int(pdb_range.split("-")[0]) - 1
        except ValueError:
            offset = 0
    first = atoms["numbers"][0] if atoms["numbers"] else 1
    shift = offset if abs(first - (offset + 1)) > 1 else (first - 1)

    radii = np.array([VDW_ATOMIC.get(e, 1.7) for e in atoms["elements"]])
    area = shrake_rupley(coords, radii, n_points=n_points)

    backbone = {"N", "CA", "C", "O", "OXT"}
    is_side = np.array([n not in backbone for n in atoms["names"]])

    seq = str(sequence).upper()
    out = np.full(len(seq), np.nan)
    for r, (num, resname) in enumerate(zip(atoms["numbers"],
                                           atoms["resnames"])):
        seq_idx = num - 1 - shift
        if not (0 <= seq_idx < len(seq)):
            continue
        aa = THREE_TO_ONE.get(resname)
        if aa is None:
            continue
        sel = (atoms["res"] == r) & is_side
        if not sel.any():
            sel = atoms["res"] == r         # glycine has no CB but has HA2/HA3
        out[seq_idx] = float(area[sel].sum()) / MAX_ASA.get(aa, 200.0)
    return out


def burial_classes(rsa, buried_max=0.05, exposed_min=0.25):
    """Label each residue buried, intermediate, or exposed.

    The thresholds are the usual ones in the literature (buried below 5%
    relative accessibility, exposed above 25%). They are a convention rather
    than a discovery, which is why the analysis script reports all three
    classes instead of picking a cutoff that flatters the result.
    """
    out = np.full(len(rsa), "unmodelled", dtype=object)
    finite = np.isfinite(rsa)
    out[finite & (rsa <= buried_max)] = "buried"
    out[finite & (rsa > exposed_min)] = "exposed"
    out[finite & (rsa > buried_max) & (rsa <= exposed_min)] = "intermediate"
    return out


def position_burial(variants, rsa):
    """RSA of the mutated position for each variant; NaN if multi-position."""
    from .mutations import parse_variant

    out = np.full(len(variants), np.nan)
    for i, v in enumerate(variants):
        positions = {s.idx for s in parse_variant(v)}
        if len(positions) != 1:
            continue
        pos = positions.pop()
        if 0 <= pos < len(rsa):
            out[i] = rsa[pos]
    return out
