"""Which assay pairs measure the same quantity -- one place, consulted by code.

Pooling two assays (pooled_truth) and calling their rank agreement a
reliability are only meaningful if the two assays are parallel measurements of
one underlying score. Three different reasons make a pair non-parallel, and
this module records each separately so that no caller can use a pair without
asking:

  * DIFFERENT_QUANTITY -- hand-curated: the two assays measure different
    biology (expression against binding or activity, gain against loss of
    function, different drug conditions). Their disagreement is partly signal.
  * DIFFERENT_SCALE -- the two assays measure the same biology on scales that
    are not related by an affine map (SPG1: Olson 2014 reports lnW, Wu 2016
    reports Fitness; the two rank almost identically but a z-score removes only
    an affine difference, so pooling z-scores averages unlike scales).
  * the reference table's coarse_selection_type differs -- a mechanical check
    that the hand list can be tested against (reference_disagreements).

pair_status() returns all three, never a single yes/no, because the right
reaction differs: a different-quantity pair should not be called a replicate
at all; a different-scale pair is a replicate whose z-pool is approximate and
which should be pooled on ranks.
"""

DIFFERENT_QUANTITY = {
    frozenset(("CAR11_HUMAN_Meitlis_2020_gof", "CAR11_HUMAN_Meitlis_2020_lof")),
    frozenset(("RPC1_LAMBD_Li_2019_high-expression",
               "RPC1_LAMBD_Li_2019_low-expression")),
    frozenset(("KCNE1_HUMAN_Muhammad_2023_expression",
               "KCNE1_HUMAN_Muhammad_2023_function")),
    frozenset(("OXDA_RHOTO_Vanella_2023_activity",
               "OXDA_RHOTO_Vanella_2023_expression")),
    frozenset(("HXK4_HUMAN_Gersing_2022_activity",
               "HXK4_HUMAN_Gersing_2023_abundance")),
    frozenset(("VKOR1_HUMAN_Chiasson_2020_abundance",
               "VKOR1_HUMAN_Chiasson_2020_activity")),
    frozenset(("KCNJ2_MOUSE_Coyote-Maestas_2022_function",
               "KCNJ2_MOUSE_Coyote-Maestas_2022_surface")),
    frozenset(("PTEN_HUMAN_Matreyek_2021", "PTEN_HUMAN_Mighell_2018")),
    frozenset(("Q53Z42_HUMAN_McShan_2019_binding-TAPBPR",
               "Q53Z42_HUMAN_McShan_2019_expression")),
    frozenset(("RASK_HUMAN_Weng_2022_abundance",
               "RASK_HUMAN_Weng_2022_binding-DARPin_K55")),
    frozenset(("S22A1_HUMAN_Yee_2023_abundance",
               "S22A1_HUMAN_Yee_2023_activity")),
    frozenset(("CP2C9_HUMAN_Amorosi_2021_abundance",
               "CP2C9_HUMAN_Amorosi_2021_activity")),
    frozenset(("SPIKE_SARS2_Starr_2020_binding",
               "SPIKE_SARS2_Starr_2020_expression")),
    frozenset(("P53_HUMAN_Giacomelli_2018_Null_Etoposide",
               "P53_HUMAN_Giacomelli_2018_Null_Nutlin")),
    frozenset(("P53_HUMAN_Giacomelli_2018_Null_Etoposide",
               "P53_HUMAN_Giacomelli_2018_WT_Nutlin")),
    frozenset(("P53_HUMAN_Giacomelli_2018_Null_Nutlin",
               "P53_HUMAN_Giacomelli_2018_WT_Nutlin")),
}

# Same biology, different reported scale. The reference table's
# raw_DMS_phenotype_name column records lnW for Olson 2014 and Fitness for Wu
# 2016 (both coarse type Binding, both selection_assay "Binding (IgG)").
DIFFERENT_SCALE = {
    frozenset(("SPG1_STRSG_Olson_2014", "SPG1_STRSG_Wu_2016")):
        "lnW (Olson 2014) vs Fitness (Wu 2016)",
}


def pair_status(assay_a, assay_b, reference=None):
    """Everything known about whether two assays are parallel measures.

    Returns a dict:
      different_quantity            hand-curated list says different biology
      different_scale               same biology, non-affinely-related scales
      reference_coarse_differs      reference coarse_selection_type differs
                                    (None when no reference table was given)
      parallel                      True only if none of the above holds
      reasons                       human-readable list, empty when parallel
    """
    key = frozenset((assay_a, assay_b))
    dq = key in DIFFERENT_QUANTITY
    ds = key in DIFFERENT_SCALE
    coarse = None
    reasons = []
    if dq:
        reasons.append("hand-curated DIFFERENT_QUANTITY list")
    if ds:
        reasons.append("different scale: " + DIFFERENT_SCALE[key])
    if reference is not None:
        r = reference.set_index("DMS_id")["coarse_selection_type"]
        if assay_a in r.index and assay_b in r.index:
            coarse = bool(r.loc[assay_a] != r.loc[assay_b])
            if coarse:
                reasons.append(
                    f"reference coarse_selection_type differs: "
                    f"{r.loc[assay_a]} vs {r.loc[assay_b]}")
    return {
        "different_quantity": dq,
        "different_scale": ds,
        "reference_coarse_differs": coarse,
        "parallel": not (dq or ds or bool(coarse)),
        "reasons": reasons,
    }


def in_different_quantity_pair(assay_id):
    """True if the assay is a member of any hand-listed different-quantity pair."""
    return any(assay_id in pair for pair in DIFFERENT_QUANTITY)


def reference_disagreements(reference):
    """Same-protein pairs the hand list and the reference table disagree on.

    Returns (listed_but_same_coarse, unlisted_but_coarse_differs), each a
    sorted list of (assay_a, assay_b). The first is expected to be non-empty
    (RPC1 high/low expression, the P53 conditions and CAR11 gof/lof are listed
    for reasons the coarse label cannot express); the second is a to-do list of
    pairs a reader might expect to be flagged and are not.
    """
    import itertools
    r = reference.set_index("DMS_id")["coarse_selection_type"]
    listed_same, unlisted_diff = [], []
    for _, grp in reference.groupby("UniProt_ID"):
        for a, b in itertools.combinations(sorted(grp["DMS_id"]), 2):
            listed = frozenset((a, b)) in DIFFERENT_QUANTITY
            differs = r.loc[a] != r.loc[b]
            if listed and not differs:
                listed_same.append((a, b))
            if differs and not listed:
                unlisted_diff.append((a, b))
    return sorted(listed_same), sorted(unlisted_diff)
