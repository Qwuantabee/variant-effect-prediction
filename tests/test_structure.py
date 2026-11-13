"""Tests for the structure-aware scoring.

The results are only worth reporting if the geometry underneath them is right,
and burial is easy to get wrong in ways that still produce plausible numbers.
Three failure modes are covered here, each of which passes silently otherwise:

1. **Wrong residue alignment.** If the PDB numbering is read against the wrong
   offset, every accessibility value belongs to a different residue. Handling
   this by comparing the PDB's residue identities to the sequence catches it.
2. **No immersed core.** A measure that never reports anything buried cannot
   test a buried-versus-exposed hypothesis. An earlier CA-sphere version of
   this module gave IF1 a minimum relative accessibility of 0.14 -- no core at
   all -- which is why the real calculation is all-atom.
3. **Gaps read as zero.** A residue without coordinates must be NaN. Returning
   0.0 would enter the "buried" class and contaminate exactly the group the
   analysis is about.

The geometry tests build a small synthetic structure, so they run without any
download. The ProteinGym-structure tests are marked slow.
"""

import numpy as np
import pytest

from variant_effect import structure


# a tiny PDB with 3 alanines in a line, 4 A apart, plus one far away
SYNTHETIC = """
ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N
ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00 20.00           C
ATOM      3  C   ALA A   1       2.000   0.000   0.000  1.00 20.00           C
ATOM      4  O   ALA A   1       3.000   0.000   0.000  1.00 20.00           O
ATOM      5  CB  ALA A   1       1.000   1.500   0.000  1.00 20.00           C
ATOM      6  N   ALA A   2       4.000   0.000   0.000  1.00 20.00           N
ATOM      7  CA  ALA A   2       5.000   0.000   0.000  1.00 20.00           C
ATOM      8  C   ALA A   2       6.000   0.000   0.000  1.00 20.00           C
ATOM      9  O   ALA A   2       7.000   0.000   0.000  1.00 20.00           O
ATOM     10  CB  ALA A   2       5.000   1.500   0.000  1.00 20.00           C
ATOM     11  N   ALA A   3       8.000   0.000   0.000  1.00 20.00           N
ATOM     12  CA  ALA A   3       9.000   0.000   0.000  1.00 20.00           C
ATOM     13  C   ALA A   3      10.000   0.000   0.000  1.00 20.00           C
ATOM     14  O   ALA A   3      11.000   0.000   0.000  1.00 20.00           O
ATOM     15  CB  ALA A   3       9.000   1.500   0.000  1.00 20.00           C
END
"""


@pytest.fixture
def synthetic_pdb(tmp_path):
    p = tmp_path / "tiny.pdb"
    p.write_text(SYNTHETIC)
    return p


def test_parse_atoms_skips_hydrogens_and_groups_residues(synthetic_pdb):
    atoms = structure.parse_atoms(synthetic_pdb)
    assert len(atoms["coords"]) == 15
    assert len(atoms["numbers"]) == 3
    assert atoms["resnames"] == ["ALA", "ALA", "ALA"]
    assert atoms["res"].tolist() == [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]


def test_parse_ca_coords(synthetic_pdb):
    coords, residues, numbers = structure.parse_ca_coords(synthetic_pdb)
    assert coords.shape == (3, 3)
    assert residues == ["A", "A", "A"]
    assert numbers == [1, 2, 3]
    assert coords[1][0] == pytest.approx(5.0)


def test_shrake_rupley_isolated_atom_matches_sphere_area():
    """One atom with no neighbours has its full sphere accessible, so the
    result must be 4*pi*(r+probe)^2. This pins the formula and the probe."""
    area = structure.shrake_rupley(np.array([[0.0, 0.0, 0.0]]), [1.7])
    expected = 4 * np.pi * (1.7 + structure.PROBE_RADIUS) ** 2
    assert area[0] == pytest.approx(expected, rel=0.02)


def test_shrake_rupley_buried_atom_is_occluded():
    """An atom surrounded by neighbours loses nearly all its area. If this
    ever returns the full sphere, the neighbour loop is broken and every
    burial value downstream is meaningless."""
    centre = np.zeros((1, 3))
    shell = []
    for dx in (-3.0, 0.0, 3.0):
        for dy in (-3.0, 0.0, 3.0):
            for dz in (-3.0, 0.0, 3.0):
                if (dx, dy, dz) != (0, 0, 0):
                    shell.append((dx, dy, dz))
    coords = np.vstack([centre, np.array(shell)])
    radii = [1.7] * len(coords)
    area = structure.shrake_rupley(coords, radii)
    free = 4 * np.pi * (1.7 + structure.PROBE_RADIUS) ** 2
    assert area[0] < 0.05 * free


def test_relative_sasa_flags_a_buried_residue(tmp_path):
    """A residue packed inside a shell of neighbours must come out buried.

    The shell atoms are given their own residue numbers. Sharing one number
    would sum 26 atoms into a single residue and divide by a one-residue
    maximum, producing an impossible value above 1 -- which is how an earlier
    version of this fixture failed, and worth keeping in mind as a check: a
    relative accessibility over 1.0 means the per-residue grouping is wrong.
    """
    lines = []
    serial = 1
    lines.append(f"ATOM  {serial:5d}  CA  ALA A   1    "
                 f"{0.0:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 20.00           C")
    serial += 1
    lines.append(f"ATOM  {serial:5d}  CB  ALA A   1    "
                 f"{1.0:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 20.00           C")
    serial += 1

    resnum = 2
    for dx in (-2.5, 0.0, 2.5):
        for dy in (-2.5, 0.0, 2.5):
            for dz in (-2.5, 0.0, 2.5):
                if (dx, dy, dz) == (0, 0, 0):
                    continue
                lines.append(f"ATOM  {serial:5d}  CB  ALA A{resnum:4d}    "
                             f"{dx:8.3f}{dy:8.3f}{dz:8.3f}  1.00 20.00           C")
                serial += 1
                resnum += 1
    p = tmp_path / "core.pdb"
    p.write_text("\n".join(lines) + "\nEND\n")

    seq = "A" * (resnum - 1)
    rsa = structure.relative_sasa(p, seq, f"1-{resnum - 1}")
    assert rsa[0] < 0.1, f"central residue should be buried, got {rsa[0]:.3f}"
    assert np.nanmax(rsa) > rsa[0]


def test_relative_sasa_returns_nan_without_coordinates(synthetic_pdb):
    """A residue the structure does not cover must be NaN, never 0.

    Zero would sort into the "buried" class and contaminate the buried group
    with residues that simply were not modelled -- which is the one thing that
    would make a buried-versus-exposed comparison meaningless.
    """
    rsa = structure.relative_sasa(synthetic_pdb, "AAAXXX", "1-3")
    assert np.isfinite(rsa[:3]).all()
    assert np.isnan(rsa[3:]).all()


def test_coords_for_sequence_rejects_a_mismatched_structure(synthetic_pdb):
    """If the PDB and the sequence disagree, raise rather than return the
    accessibility of some other residue."""
    with pytest.raises(ValueError, match="disagree|shift"):
        structure.coords_for_sequence(synthetic_pdb, "WWWWWWWWWW", "1-3")


def test_burial_classes_use_the_stated_cutoffs():
    rsa = np.array([0.0, 0.03, 0.05, 0.06, 0.2, 0.25, 0.26, 0.9, np.nan])
    cls = structure.burial_classes(rsa, buried_max=0.05, exposed_min=0.25)
    assert list(cls[:3]) == ["buried"] * 3
    assert cls[3] == "intermediate"
    assert cls[6] == "exposed"
    assert cls[8] == "unmodelled"


def test_position_burial_maps_variants_and_rejects_multi_mutants():
    rsa = np.array([0.9, 0.01, 0.5])
    out = structure.position_burial(["A1W", "A2W", "A3W", "A1W:A2W"], rsa)
    assert out[0] == pytest.approx(0.9)
    assert out[1] == pytest.approx(0.01)
    assert out[2] == pytest.approx(0.5)
    assert np.isnan(out[3]), "a two-position variant has no single burial value"


# ---------------------------------------------------------------------------
# the real structures
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_real_structure_has_a_core():
    """The check that a burial measure is actually measuring burial.

    On E. coli IF1 (72 residues, AlphaFold model) at least a handful of
    residues must come out buried and the spread must be wide. A version that
    reports every residue above 0.14 is not measuring a core -- and an earlier
    implementation of this module, using one sphere per residue, did exactly
    that in a way that looked reasonable.
    """
    from variant_effect import data

    ref = data.reference_table()
    aid = "IF1_ECOLI_Kelsic_2016"
    row = ref[ref["DMS_id"] == aid].iloc[0]
    rsa = structure.relative_sasa(data.structure_path(aid, reference=ref),
                                  row["target_seq"], row["pdb_range"])
    finite = rsa[np.isfinite(rsa)]
    assert len(finite) == len(row["target_seq"])
    assert np.nanmin(rsa) < 0.02, f"no buried residue: min {np.nanmin(rsa):.3f}"
    assert np.nanmax(rsa) > 0.4, f"no exposed residue: max {np.nanmax(rsa):.3f}"
    assert finite.std() > 0.1
    cls = structure.burial_classes(rsa)
    assert (cls == "buried").sum() >= 3
    assert (cls == "exposed").sum() >= 10
