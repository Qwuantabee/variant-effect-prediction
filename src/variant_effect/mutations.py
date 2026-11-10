"""Parsing and applying mutation notation.

ProteinGym (and most DMS papers) write substitutions as WT-position-MUT with
a 1-indexed position, e.g. "A2M" means alanine at residue 2 becomes
methionine. Multiple substitutions in one variant are joined by a colon:
"A2M:K5R".

The wildtype letter in the notation is redundant with the sequence, which
makes it a useful consistency check -- if they disagree, something is off by
one somewhere, and that is a mistake worth catching loudly rather than
silently scoring the wrong residue.
"""

import re
from dataclasses import dataclass

MUT_RE = re.compile(r"^([A-Z])(\d+)([A-Z*])$")

AA20 = "ACDEFGHIKLMNPQRSTVWY"


@dataclass(frozen=True)
class Substitution:
    wt: str
    pos: int        # 1-indexed, as written in the notation
    mut: str

    @property
    def idx(self):
        """0-indexed position into the sequence string."""
        return self.pos - 1

    def __str__(self):
        return f"{self.wt}{self.pos}{self.mut}"


def parse_variant(notation):
    """'A2M:K5R' -> [Substitution(A,2,M), Substitution(K,5,R)]"""
    subs = []
    for part in str(notation).split(":"):
        part = part.strip()
        m = MUT_RE.match(part)
        if not m:
            raise ValueError(f"could not parse mutation {part!r}")
        wt, pos, mut = m.group(1), int(m.group(2)), m.group(3)
        if pos < 1:
            raise ValueError(f"position must be 1-indexed, got {pos} in {part!r}")
        subs.append(Substitution(wt, pos, mut))
    return subs


def check_against_sequence(subs, sequence):
    """Raise if the wildtype letters in the notation disagree with the sequence.

    This catches the classic off-by-one, and also the case where the assay's
    numbering is relative to a construct that includes a tag or starts at a
    different residue than the UniProt sequence.
    """
    for s in subs:
        if s.idx >= len(sequence):
            raise ValueError(
                f"{s} is past the end of a {len(sequence)}-residue sequence")
        actual = sequence[s.idx]
        if actual != s.wt:
            raise ValueError(
                f"{s} expects {s.wt} at position {s.pos} but sequence has "
                f"{actual} -- numbering mismatch")


def apply_variant(sequence, subs):
    """Return the mutated sequence. Does not validate; call check first."""
    seq = list(sequence)
    for s in subs:
        seq[s.idx] = s.mut
    return "".join(seq)


def mutated_positions(variants):
    """Every 0-indexed position touched by any variant in the list.

    Used to decide which positions need a masked forward pass. For a typical
    single-substitution DMS this is every position in the assayed region, but
    for sparser assays it can be far fewer, and masking only what is needed
    is the difference between a 30 second run and a 10 minute one.
    """
    positions = set()
    for v in variants:
        for s in parse_variant(v):
            positions.add(s.idx)
    return sorted(positions)


def is_single_substitution(notation):
    return ":" not in str(notation)
