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


def map_query_to_target(query, target, msa_start=1):
    """Residue number -> MSA column, from a gapped query row.

    The .a2m query is *not* always the target sequence: on 20 of the 31 viral
    assays it is a subsequence of it. A0A2Z5U3Z0_9INFA's query starts at target
    residue 14, and A0A192B1T2_9HIV1's query is missing 12 residues that the
    UniProt entry has. So neither ``column = position - msa_start`` nor counting
    non-gap query columns from msa_start is correct -- the query has to be
    aligned to the target before any column can be trusted.

    Alignment is done with a global pairwise alignment that strongly prefers
    matches and discourages gaps in the query, then the query's gap columns are
    skipped. Where the query carries a residue differing from the target at the
    same place (a real substitution in the construct), the residue is mapped by
    position from the surrounding matches rather than dropped, so a small change
    does not remove a column from scoring.

    Returns {1-indexed target residue: MSA column}. Residues the alignment does
    not cover are absent, and mutations there score NaN rather than being
    silently attributed to a neighbouring column.
    """
    from Bio.Align import PairwiseAligner

    target = str(target).upper()
    query = str(query).upper()
    aln = PairwiseAligner()
    aln.mode = "global"
    aln.match_score = 2.0
    aln.mismatch_score = -1.0
    aln.open_gap_score = -10.0
    aln.extend_gap_score = -0.5
    # a gap in the target (an insertion in the query relative to UniProt) is
    # cheap: constructs routinely carry tags and the target need not span them,
    # whereas dropping a real residue from the map costs a column
    aln.open_internal_deletion_score = -10.0
    aln.extend_internal_deletion_score = -0.5
    aln.open_internal_insertion_score = -10.0
    aln.extend_internal_insertion_score = -0.05

    best = aln.align(target, query)[0]
    # best.aligned[0] / [1] are (start, end) blocks in target / query coords
    mapping = {}
    for (t0, t1), (q0, q1) in zip(best.aligned[0], best.aligned[1]):
        span_q = q1 - q0
        span_t = t1 - t0
        if span_q <= 0:
            continue
        if span_q == span_t:
            # blocks of equal length: pair them residue for residue, skipping
            # query gap characters
            tpos, qpos = t0, q0
            while tpos < t1 and qpos < q1:
                if query[qpos] == "-":
                    qpos += 1
                    continue
                mapping[tpos + 1] = qpos
                tpos += 1
                qpos += 1
            continue
        # unequal spans: the query has residues without a target counterpart.
        # Walk both, consuming target positions in order and assigning each
        # query residue to the target residue it is closest to.
        qpos, tpos = q0, t0
        advance = span_t / span_q
        for k in range(span_q):
            if query[qpos] == "-":
                qpos += 1
                continue
            t = min(t0 + int(k * advance), t1 - 1)
            mapping[t + 1] = qpos
            qpos += 1
    return mapping


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

        self.column_for_residue = map_query_to_target(query, target_seq,
                                                      msa_start)
        self.n_residues_mapped = len(self.column_for_residue)

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

    # a match state is a column where the query carries a residue. Query gap
    # columns are where homologs have insertions relative to the target; they
    # have no counterpart in the target sequence, so scoring a mutation against
    # one would be scoring it against nothing.
    keep = [i for i, c in enumerate(query) if c.isupper()]
    query_match = "".join(query[i] for i in keep)
    width = len(query)
    others_match = [
        "".join(r[i] for i in keep)
        for r in others if len(r) == width
    ]
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
    """Score each variant by the log-ratio of MSA frequencies at its position.

        score = sum over substitutions of log f(mut|position) / f(wt|position)

    Position-specific, family-specific, and no coupling: this is the weakest
    thing that could be called "an alignment-based method", which makes it the
    right control. If it beats ESM-2 on viral assays while losing to it
    elsewhere, the family-specific-conservation explanation has support.

    background: optional 20-vector. When given, the score becomes a log-odds
    against that composition rather than a raw ratio. It defaults to None
    because the raw ratio is what reproduces the published Site-Independent
    column: over twelve viral assays it averages 0.375 against a published
    0.393, while subtracting the family composition averages 0.334. The
    composition correction is the theoretically tidier choice -- it stops a
    conservative swap of a rare residue being scored as intolerance -- but it
    does not match the published method, so it is offered rather than applied.
    """
    freqs = positional_frequencies(msa, weights=weights, pseudocount=pseudocount)
    if background is None:
        log_bg = np.zeros(20)
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
