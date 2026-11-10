import numpy as np

from variant_effect import baselines

SEQ = "MKTAYIAKQRGPC"


def test_blosum_prefers_conservative_substitutions():
    """K->R is a swap between two basic residues and scores positively in
    BLOSUM62; K->W is not and does not."""
    scores = baselines.blosum62_score(SEQ, ["K2R", "K2W"])
    assert scores[0] > scores[1]
    assert scores[0] > 0


def test_blosum_identity_is_highest():
    scores = baselines.blosum62_score(SEQ, ["K2K", "K2R", "K2D"])
    assert scores[0] > scores[1] > scores[2]


def test_hydropathy_penalises_large_swaps():
    # I6 (4.5) -> V (4.2) is a small change; I6 -> D (-3.5) is a big one
    small, large = baselines.hydropathy_change(SEQ, ["I6V", "I6D"])
    assert small > large


def test_volume_penalises_large_swaps():
    small, large = baselines.volume_change(SEQ, ["A4G", "A4W"])
    assert small > large


def test_charge_change_symmetric():
    # K->E flips +1 to -1, K->R stays +1
    flip, same = baselines.charge_change(SEQ, ["K2E", "K2R"])
    assert same > flip


def test_proline_glycine_penalty():
    scores = dict(zip(
        ["A4P", "G11A", "A4S"],
        baselines.proline_glycine_penalty(SEQ, ["A4P", "G11A", "A4S"])))
    assert scores["A4P"] < scores["A4S"]      # introducing proline is bad
    assert scores["G11A"] < scores["A4S"]     # losing glycine is bad
    assert scores["A4S"] == 0.0               # neutral swap, no penalty


def test_multi_mutant_sums():
    single = baselines.blosum62_score(SEQ, ["K2R"])[0]
    double = baselines.blosum62_score(SEQ, ["K2R:Y5F"])[0]
    other = baselines.blosum62_score(SEQ, ["Y5F"])[0]
    assert double == single + other


def test_random_baseline_is_reproducible():
    a = baselines.random_baseline(SEQ, ["K2R"] * 5, seed=1)
    b = baselines.random_baseline(SEQ, ["K2R"] * 5, seed=1)
    assert np.array_equal(a, b)


def test_all_baselines_return_right_shape():
    variants = ["K2R", "Y5F", "A4P"]
    for name, fn in baselines.BASELINES.items():
        out = fn(SEQ, variants)
        assert out.shape == (3,), name
        assert np.all(np.isfinite(out)), name
