"""Alignment-based variant scoring, from a family-specific MSA.

The README's explanation for the viral failure is that ESM-2 learns constraint
from UniRef-wide patterns of conservation, and viral proteins are under immune
selection as well, so positions vary across strains for reasons unrelated to
whether the protein folds. The model reads that variation as tolerance.

That explanation makes a testable prediction: a method that builds an
alignment of the specific viral family and reads conservation off it should do
*fine* on the same assays. If instead every method struggles -- including the
family-specific ones -- then the assays are just noisy and the immune-selection
story is unnecessary.

This module implements the simplest such method, the site-independent model of
Hopf et al. 2017: per-position amino acid frequencies from the MSA, scored as a
log-ratio against the wildtype residue. It is deliberately weak -- it has no
coupling terms, so it cannot see that two positions covary -- because the point
is not to win but to establish that family-specific signal exists at all.

Two details that matter and are easy to get wrong:

- **Sequence weighting.** MSAs are dominated by near-duplicates, and an
  unweighted frequency profile is effectively an average over one clade. Rows
  are clustered at a percent-identity threshold and weighted by inverse cluster
  size, which is the standard Neff-style correction.
- **Column mapping.** The .a2m files hold lowercase columns for insertions
  relative to the query and uppercase columns for match states. Only match
  states correspond to residues of the target sequence, and the offset between
  match column 1 and target residue 1 is given per assay by MSA_start.
"""

from pathlib import Path

import numpy as np

AA20 = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {a: i for i, a in enumerate(AA20)}


class MSA:
    """One alignment: the query row plus de-duplicated homolog rows.

    sequences[i] is a string the same length as the match-state columns, using
    '-' for gaps, with insertion columns already removed. Lowercase insertion
    residues are dropped because a variant at a target residue cannot be scored
    by a column that has no counterpart in the target.

    ``column_for_residue`` is the mapping that actually matters. An .a2m query
    row is the protein sequence *gapped* to its homologs, so MSA column c does
    not correspond to target residue (msa_start + c - 1) unless the query has
    no gaps at all. On 20 of the 31 viral assays the query is gapped, and
    reading column = position - start puts every downstream mutation on the
    wrong column -- which looks like a weak method rather than a bug. The map
    is built once from the query row: each non-gap query column is the next
    target residue in order.
    """

    def __init__(self, query, sequences, msa_start, target_seq):
        self.query = query
        self.sequences = sequences
        self.msa_start = int(msa_start)      # target residue number of column 1
        self.target_seq = target_seq

        # column index for each residue of the target, in MSA coordinates
        col = 0
        self.column_for_residue = {}
        residue = self.msa_start
        for c, ch in enumerate(query):
            if ch == "-":
                continue
            self.column_for_residue[residue] = c
            residue += 1
            col = c
        self.n_residues_mapped = residue - self.msa_start

    def __len__(self):
        return len(self.sequences)

    def column_of(self, pos):
        """MSA column for a 1-indexed target residue, or None if unmapped."""
        return self.column_for_residue.get(int(pos))

    def __repr__(self):
        return (f"MSA({len(self.sequences)} rows, {len(self.query)} columns, "
                f"target {self.msa_start}-{self.msa_start + self.n_residues_mapped - 1})")


def parse_a2m(text):
    """Split an .a2m into (query, other rows), dropping insertion columns.

    In a2m, lowercase letters are insertions relative to the query. They are
    kept in place while parsing so that rows stay aligned to each other, then
    removed everywhere, leaving one character per match state.
    """
    header = None
    query = None
    others = []
    chunks = []

    def flush():
        if not chunks:
            return None
        row = "".join(chunks)
        chunks.clear()
        return row

    for raw in text.splitlines():
        line = raw.rstrip("\r\n")
        if not line:
            continue
        if line.startswith(">"):
            row = flush()
            if row is not None:
                if query is None:
                    query = row
                else:
                    others.append(row)
            header = line
            continue
        chunks.append(line.strip())

    row = flush()
    if row is not None:
        if query is None:
            query = row
        else:
            others.append(row)

    if query is None:
        raise ValueError("no sequences in alignment")

    # a match state is any column where the query carries an uppercase residue
    keep = [i for i, c in enumerate(query) if c.isupper() or c == "-"]
    query_match = "".join(query[i] for i in keep)
    others_match = ["".join(r[i] for i in keep) if len(r) > max(keep, default=0)
                    else "" for r in others]
    # any row of a different length is a malformed record; drop rather than
    # silently misalign it
    n = len(query_match)
    others_match = [r for r in others_match if len(r) == n]
    return query_match, others_match


def load_msa(path, msa_start, target_seq, max_seqs=None):
    """Read an .a2m file into an MSA, optionally truncating for speed."""
    text = Path(path).read_text(errors="replace")
    query, others = parse_a2m(text)
    if max_seqs is not None:
        others = others[:max_seqs]
    return MSA(query, others, msa_start, target_seq)


def sequence_weights(sequences, identity_threshold=0.8, max_rows=2000):
    """Inverse-cluster-size weights, the standard Neff correction.

    Rows are clustered greedily by percent identity to a cluster seed; each row
    gets weight 1/(cluster size), so a clade of 500 near-identical sequences
    contributes about as much as one divergent sequence.

    max_rows caps the cost, which is quadratic in the number of rows. When the
    alignment is larger than the cap, rows are taken as a fixed-seed random
    sample rather than the first N, because the first N are sorted by
    significance and would systematically under-represent diversity.
    """
    n = len(sequences)
    if n == 0:
        return np.zeros(0)
    if n > max_rows:
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(n, max_rows, replace=False))
        sequences = [sequences[i] for i in idx]
        n = max_rows

    # encode as a 2-D uint8 matrix so identity is a vectorised comparison
    width = max(len(s) for s in sequences)
    mat = np.full((n, width), ord("-"), dtype=np.uint8)
    for i, s in enumerate(sequences):
        arr = np.frombuffer(s.encode(), dtype=np.uint8)
        mat[i, :len(arr)] = arr
    nongap = mat != ord("-")

    assigned = np.zeros(n, dtype=bool)
    cluster_of = np.full(n, -1, dtype=np.int64)
    n_clusters = 0
    for seed in range(n):
        if assigned[seed]:
            continue
        both = nongap & nongap[seed]
        both_n = both.sum(axis=1)
        same_n = ((mat == mat[seed]) & both).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            ident = np.where(both_n > 0, same_n / both_n, 0.0)
        members = (~assigned) & (ident >= identity_threshold)
        members[seed] = True
        cluster_of[members] = n_clusters
        assigned |= members
        n_clusters += 1

    counts = np.bincount(cluster_of, minlength=n_clusters).astype(float)
    return (1.0 / counts)[cluster_of]


def positional_frequencies(msa, weights=None, pseudocount=0.01):
    """(n_positions, 20) weighted amino acid frequencies.

    A pseudocount is added so that an amino acid never observed at a position
    does not give an infinite log-ratio. It is deliberately small: with
    thousands of homologs the observed frequencies dominate, and at very
    shallow positions the pseudocount keeps the score finite instead of
    producing a spurious 6-nat penalty.

    The multinomial pseudocount is spread over all 20 residues, so its total
    weight is 20 * pseudocount.
    """
    seqs = msa.sequences
    n_rows = len(seqs)
    n_cols = len(msa.query)
    if n_rows == 0:
        raise ValueError("alignment has no homolog rows")

    if weights is None:
        weights = np.ones(n_rows)
    weights = np.asarray(weights, dtype=float)

    counts = np.zeros((n_cols, 20), dtype=float)
    for row, w in zip(seqs, weights):
        if not np.isfinite(w) or w <= 0:
            continue
        arr = np.frombuffer(row.encode(), dtype=np.uint8)
        if len(arr) != n_cols:
            continue
        for j, aa in enumerate(AA20):
            counts[:, j] += w * (arr == ord(aa))

    counts += pseudocount
    totals = counts.sum(axis=1, keepdims=True)
    return counts / totals


def site_independent_scores(msa, variants, weights=None, pseudocount=0.01,
                            background=None):
    """Score each variant by the log-odds of the mutant against the wildtype.

        score = sum over substitutions of
                    [log f(mut|position) - log bg(mut)]
                  - [log f(wt|position)  - log bg(wt)]

    Position-specific, family-specific, and no coupling: this is the weakest
    thing that could be called "an alignment-based method", which makes it the
    right control. If it beats ESM-2 on viral assays while losing to it
    elsewhere, the family-specific-conservation explanation has support.

    The background term is not optional. A raw ratio of the two conditional
    frequencies punishes every conservative swap of a rare residue as if the
    position were intolerant, because amino acids differ by an order of
    magnitude in abundance; without it this method scored 0.16 on the viral
    set against 0.37 for the published version. Dividing through by the
    family's own composition leaves the part that is about *this position*.

    background: optional 20-vector of composition frequencies. Defaults to the
    mean over positions, which is the alignment's own composition and therefore
    already family-specific -- using a generic UniProt composition would put
    back the family-agnostic assumption the method is meant to avoid.
    """
    freqs = positional_frequencies(msa, weights=weights, pseudocount=pseudocount)
    if background is None:
        bg = freqs.mean(axis=0)
    else:
        bg = np.asarray(background, dtype=float)
    bg = bg / bg.sum()
    log_bg = np.log(bg)

    from .mutations import parse_variant

    out = np.zeros(len(variants), dtype=np.float32)
    for i, v in enumerate(variants):
        total = 0.0
        for s in parse_variant(v):
            col = msa.column_of(s.pos)           # target residue -> MSA column
            if col is None:
                out[i] = np.nan
                break
            w_i = AA_INDEX[s.wt]
            m_i = AA_INDEX[s.mut]
            if freqs[col, m_i] <= 0 or freqs[col, w_i] <= 0:
                out[i] = np.nan
                break
            total += ((np.log(freqs[col, m_i]) - log_bg[m_i])
                      - (np.log(freqs[col, w_i]) - log_bg[w_i]))
        else:
            out[i] = total
    return out


def column_entropy(msa, weights=None, pseudocount=0.01):
    """Per-column Shannon entropy in nats, for the conservation analysis.

    Low entropy means the position tolerates little variation across the
    family, which is the alignment's own statement about how constrained the
    position is -- independent of any model.
    """
    freqs = positional_frequencies(msa, weights=weights, pseudocount=pseudocount)
    return -(freqs * np.log(freqs)).sum(axis=1)
