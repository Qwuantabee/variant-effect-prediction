"""Scoring tests.

The ones that need the model are marked slow and download an 8M-parameter
checkpoint (about 30 MB). Run them with:

    pytest -m slow

test_reproduces_official_proteingym_score is the one that actually matters.
Everything else can pass while the implementation is quietly off by one; that
test cannot, because it checks the output against a published number computed
by someone else's code.
"""

import numpy as np
import pytest

from variant_effect import baselines, evaluate, mutations

SEQ = ("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGCEKAVQVKVKALPDAQ"
       "FEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQ")


@pytest.fixture(scope="module")
def scorer():
    pytest.importorskip("transformers")
    from variant_effect.scoring import ESMScorer
    return ESMScorer("facebook/esm2_t6_8M_UR50D")


@pytest.mark.slow
def test_logprob_table_is_normalised(scorer):
    """Rows are log-softmax over the full vocabulary, so the 20 standard
    amino acids sum to less than 1 in probability but not by much -- if they
    sum to far less, we are reading the wrong axis."""
    table = scorer.wt_marginal_logprobs(SEQ[:40])
    assert table.shape == (40, 20)
    probs = np.exp(table).sum(axis=1)
    assert np.all(probs > 0.5) and np.all(probs <= 1.0 + 1e-4)


@pytest.mark.slow
def test_synonymous_substitution_scores_zero(scorer):
    """X->X is log p(X) - log p(X) = 0 by construction. Catches a swapped
    index between the mutant and wildtype lookup."""
    seq = SEQ[:40]
    variants = [f"{seq[i]}{i + 1}{seq[i]}" for i in (3, 10, 20)]
    scores = scorer.score_variants(seq, variants, scheme="masked")
    assert np.allclose(scores, 0.0, atol=1e-5)


@pytest.mark.slow
def test_masked_and_wt_schemes_agree_in_direction(scorer):
    """They disagree on magnitude, which is the point, but should broadly
    rank the same variants badly."""
    seq = SEQ[:60]
    variants = [f"{seq[i]}{i + 1}W" for i in range(0, 50, 5)]
    masked = scorer.score_variants(seq, variants, scheme="masked")
    wt = scorer.score_variants(seq, variants, scheme="wt")
    assert evaluate.spearman(masked, wt) > 0.5


@pytest.mark.slow
def test_scoring_catches_numbering_mismatch(scorer):
    """A wrong wildtype letter must raise rather than score the wrong
    residue. This is the failure mode that silently costs 10 points of
    Spearman."""
    with pytest.raises(ValueError, match="numbering mismatch"):
        scorer.score_variants(SEQ[:40], ["A1A"], scheme="wt")


@pytest.mark.slow
def test_masked_only_scores_requested_positions(scorer):
    """Positions nobody asked about stay NaN, which is what makes restricting
    the loop to mutated positions safe."""
    table = scorer.masked_marginal_logprobs(SEQ[:40], positions=[5, 10])
    assert np.isfinite(table[5]).all()
    assert np.isfinite(table[10]).all()
    assert np.isnan(table[0]).all()


@pytest.mark.slow
def test_reproduces_official_proteingym_score(scorer):
    """End to end against a published number.

    ProteinGym reports Spearman 0.193 for ESM-2 8M on IF1_ECOLI_Kelsic_2016.
    Our masked-marginals implementation must land on the same value. A
    tolerance of 0.01 allows for their rounding; anything larger means a bug
    in indexing, tokenisation or the BOS offset.
    """
    from variant_effect import data

    ref = data.reference_table()
    df, sequence = data.load_assay("IF1_ECOLI_Kelsic_2016", reference=ref)
    scores = scorer.score_variants(sequence, df["mutant"].tolist(),
                                   scheme="masked")
    ours = evaluate.spearman(scores, df["DMS_score"].to_numpy())

    official = float(data.official_scores().loc["IF1_ECOLI_Kelsic_2016",
                                                "ESM2 (8M)"])
    assert abs(ours - official) < 0.01, (
        f"ours {ours:.4f} vs official {official:.4f}")


def test_long_sequence_rejected_without_windowing():
    """Guard rail: the non-windowed path must refuse rather than silently
    truncate a protein and score the wrong residues."""
    from variant_effect.scoring import MAX_RESIDUES

    class Dummy:
        score_variants = None

    long_seq = "A" * (MAX_RESIDUES + 1)
    # exercise the length check directly without loading a model
    from variant_effect import scoring
    with pytest.raises(ValueError, match="over the"):
        scoring.ESMScorer.score_variants(
            Dummy(), long_seq, ["A1G"], scheme="wt")


def test_baselines_and_esm_use_same_convention():
    """Both must return higher-is-more-tolerated, or the comparison table is
    nonsense. Checked on BLOSUM62, where the answer is unambiguous."""
    variants = ["K2R", "K2W"]
    scores = baselines.blosum62_score(SEQ, variants)
    assert scores[0] > scores[1]


def test_mutated_positions_drives_the_masking_loop():
    """Cost control: a 300-residue protein with 50 assayed positions should
    only need 50 forward passes, not 300."""
    variants = ["M1A", "K2A", "K2R", "T3A"]
    assert mutations.mutated_positions(variants) == [0, 1, 2]
