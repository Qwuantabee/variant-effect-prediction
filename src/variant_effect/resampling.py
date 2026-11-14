"""Resampling by position, and one canonical spelling for a mutant.

Why this module exists
----------------------
Variants in a deep mutational scan are not independent draws. RASK_HUMAN's
20,124 shared doubles sit on 1,390 position pairs over 185 positions (median
114 doubles touch any one position, maximum 2,150): two doubles that share a
position share that position's biology, the model's per-position log-odds
table, and the position's masked-marginal error. A bootstrap that resamples
*variants* treats all 20,124 as independent and so reports an interval that is
far too narrow; the unit of independent information is closer to the position.

``ClusterIndex`` implements the resampling unit this repo uses wherever a
confidence interval or a gap is quoted:

  * Draw the positions (clusters) with replacement. Every position gets a
    multiplicity c_p -- how many times it was drawn.
  * A variant carries the product of the multiplicities of the distinct
    positions it touches: w_v = prod over p in v of c_p. A single substitution
    at a position drawn twice appears twice; a double whose two positions were
    drawn 2 and 3 times appears 6 times; a double with an un-drawn position
    does not appear.

For single-position variants this is the ordinary cluster bootstrap. For
variants touching several positions it is the vertex (pigeonhole) bootstrap
for crossed random effects: the expected weight of every variant is 1, and
two variants that share a position have positively correlated weights, which
is exactly the dependence a variant-level bootstrap throws away.

Caveats stated here so each caller does not have to repeat them:

  * The cluster is the *position*, not the 3-D neighbourhood or the domain.
    Neighbouring positions are still treated as independent, so the interval is
    narrower than a domain-level one would be.
  * High-order multi-mutants (HIS7 and GFP carry up to 13 substitutions) have
    product weights with very high variance, and most are absent from any one
    replicate. The interval honestly widens; it is not an artefact to tune.

``canonical_mutant`` is the second job. ProteinGym spells a multi-mutant as
the colon-joined substitutions in whatever order the source lab wrote them, so
D266W:V265S (Olson 2014) and V265S:D266W (Wu 2016, always position-ordered) are
one variant in two notations. Joining two assays on the raw string silently
drops every such variant.
"""

import numpy as np

from .mutations import parse_variant


def substitution_positions(mutant):
    """Sorted distinct 1-indexed positions a mutant touches.

    'D266W:V265S' -> (265, 266). Raises ValueError on a notation the parser
    rejects, rather than silently producing an empty cluster.
    """
    return tuple(sorted({s.pos for s in parse_variant(mutant)}))


def canonical_mutant(mutant):
    """One spelling per variant: substitutions sorted by position (then text).

    'D266W:V265S' and 'V265S:D266W' both become 'V265S:D266W'. A single
    substitution is returned unchanged. Two different variants never collide:
    the canonical string keeps every substitution.
    """
    subs = parse_variant(mutant)
    ordered = sorted(subs, key=lambda s: (s.pos, str(s)))
    return ":".join(str(s) for s in ordered)


def canonical_series(mutants):
    """canonical_mutant over an iterable, as a list."""
    return [canonical_mutant(m) for m in mutants]


class ClusterIndex:
    """Which positions each variant touches, prepared once for many draws.

    Resampling unit: the position. Pass memberships as one iterable of cluster
    labels per variant (for example substitution_positions(m)); a bare label
    is treated as a one-element membership.
    """

    def __init__(self, memberships):
        labels = {}
        rows = []
        for m in memberships:
            if np.isscalar(m) or isinstance(m, str):
                m = (m,)
            distinct = sorted(set(m), key=lambda x: (str(type(x)), x))
            ids = []
            for lab in distinct:
                if lab not in labels:
                    labels[lab] = len(labels)
                ids.append(labels[lab])
            rows.append(ids)
        self.n_variants = len(rows)
        self.n_clusters = len(labels)
        kmax = max((len(r) for r in rows), default=1)
        pad = self.n_clusters            # sentinel slot whose multiplicity is 1
        table = np.full((self.n_variants, max(kmax, 1)), pad, dtype=np.int64)
        for i, r in enumerate(rows):
            table[i, :len(r)] = r
        self._table = table

    @classmethod
    def from_mutants(cls, mutants):
        """Cluster = position; one membership per mutant string."""
        return cls([substitution_positions(m) for m in mutants])

    def draw(self, rng):
        """Integer weight per variant for one cluster-bootstrap replicate."""
        g = self.n_clusters
        if g == 0:
            return np.zeros(0, dtype=np.int64)
        counts = np.bincount(rng.integers(0, g, g), minlength=g)
        ext = np.append(counts, 1).astype(np.int64)
        return ext[self._table].prod(axis=1)


def expand(weights):
    """Row indices that realise integer weights: index i repeated w_i times."""
    return np.repeat(np.arange(len(weights)), np.asarray(weights, dtype=np.int64))


def gap_bootstrap(fn, single, double, mutants_s=None, mutants_d=None,
                  n_boot=1000, seed=0, min_n=30, joint=False):
    """Percentile CI on fn(single) - fn(double) for two variant strata.

    single and double are (pred, truth) pairs; fn(pred, truth) -> float is a
    metric such as Spearman or top-decile recall. With joint=True fn takes both
    strata, fn((pred_s, truth_s), (pred_d, truth_d)) -> the gap itself, for
    controls that need both strata at once (a range-matched window). Returns
    (point, lo, hi, sd) with NaN where a stratum has fewer than min_n rows or
    fewer than 50 replicates are finite.

    RESAMPLING UNIT, stated because it decides whether an interval excludes 0:

      * mutants_s and mutants_d given (the mutant strings of each stratum): the
        POSITION. One draw of positions is shared by BOTH strata -- they sit on
        the same positions, so the singles' and the doubles' resampling noise
        is correlated through the positions and a joint draw is the honest
        one. Variants are weighted by the product of their positions'
        multiplicities (see ClusterIndex). Use this for every real analysis.
      * mutants_s / mutants_d omitted: the VARIANT, each stratum resampled
        independently with iid row draws. Valid only when the rows really are
        independent (synthetic tests, or data with one row per position).
    """
    ps, ts = np.asarray(single[0], float), np.asarray(single[1], float)
    pd_, td = np.asarray(double[0], float), np.asarray(double[1], float)
    ns, nd = len(ps), len(pd_)
    nan4 = (np.nan, np.nan, np.nan, np.nan)
    if ns < min_n or nd < min_n:
        return nan4
    rng = np.random.default_rng(seed)
    if (mutants_s is None) != (mutants_d is None):
        raise ValueError("pass mutants for both strata or for neither")
    index = None
    if mutants_s is not None:
        index = ClusterIndex.from_mutants(list(mutants_s) + list(mutants_d))
    gaps = []
    for _ in range(n_boot):
        if index is not None:
            w = index.draw(rng)
            si, di = expand(w[:ns]), expand(w[ns:])
            if len(si) < min_n or len(di) < min_n:
                continue
        else:
            si = rng.integers(0, ns, ns)
            di = rng.integers(0, nd, nd)
        if joint:
            g = fn((ps[si], ts[si]), (pd_[di], td[di]))
        else:
            g = fn(ps[si], ts[si]) - fn(pd_[di], td[di])
        if np.isfinite(g):
            gaps.append(g)
    if len(gaps) < 50:
        return nan4
    gaps = np.asarray(gaps)
    point = (fn((ps, ts), (pd_, td)) if joint
             else fn(ps, ts) - fn(pd_, td))
    return (float(point), float(np.percentile(gaps, 2.5)),
            float(np.percentile(gaps, 97.5)), float(gaps.std(ddof=1)))
