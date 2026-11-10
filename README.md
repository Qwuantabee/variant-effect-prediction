# Zero-shot variant effect prediction with ESM-2

Scoring missense variants with a protein language model that has never seen a
fitness measurement, benchmarked against experimental deep mutational scanning
data from ProteinGym, and checked against the official published numbers.

The question is whether a model trained only to fill in masked residues across
UniRef has learned enough about protein constraint to predict which mutations
break a protein. The answer is a qualified yes, and the qualifications are the
interesting part.

## Does the implementation actually work

This is the first thing to establish, because a variant effect pipeline that is
off by one still produces plausible-looking correlations. ProteinGym publishes
per-assay Spearman values for ~97 predictors, so an independent implementation
can be checked against the reference rather than asserted to be correct.

| assay | this implementation | ProteinGym published | difference |
|---|---|---|---|
| IF1_ECOLI_Kelsic_2016 | 0.598 | 0.599 | −0.001 |
| CCDB_ECOLI_Tripathi_2016 | 0.511 | 0.511 | 0.000 |
| SUMO1_HUMAN_Weile_2017 | 0.509 | 0.509 | 0.000 |
| RL40A_YEAST_Roscoe_2013 | 0.598 | 0.599 | −0.001 |
| DYR_ECOLI_Thompson_2019 | 0.480 | 0.480 | 0.000 |
| NUD15_HUMAN_Suiter_2020 | 0.526 | 0.526 | 0.000 |
| TPMT_HUMAN_Matreyek_2018 | 0.539 | 0.539 | 0.000 |
| BLAT_ECOLX_Stiffler_2015 | 0.732 | 0.731 | +0.001 |
| R1AB_SARS2_Flynn_2022 | 0.105 | 0.105 | 0.000 |

Mean difference −0.0001, maximum absolute difference 0.0006. The residual is
rounding in the published table.

![reproduction check](figures/reproduction_check.png)

Getting here required fixing one real bug. Benchmark rows were built as
`{"assay": dms_id, **metadata}`, and the metadata dictionary had its own key
called `assay` holding the assay description. The merge silently replaced the
DMS id with a string like `"Yeast growth"`, so every lookup against the
official table missed and the comparison printed an empty table rather than
failing. `test_metadata_does_not_collide_with_result_columns` now covers it.

## What the model is worth against cheap alternatives

A correlation of 0.5 means nothing on its own. The comparison that matters is
against methods that need no model at all.

Mean Spearman over the 8 non-viral assays:

| method | mean Spearman | what it knows |
|---|---|---|
| **ESM-2 650M** | **0.562** | sequence context, learned from UniRef |
| BLOSUM62 | 0.242 | how often this swap appears in aligned homologs |
| hydropathy change | 0.152 | Kyte-Doolittle scale only |
| side-chain volume change | 0.123 | size only |
| proline/glycine heuristic | 0.108 | three structural rules of thumb |
| charge change | −0.003 | formal charge at pH 7 |
| random | −0.008 | nothing |

![method comparison](figures/method_comparison.png)

So context is worth roughly a factor of two over a position-blind substitution
matrix. That gap is the part of the signal that comes from knowing *where in
this particular protein* the residue sits, rather than what the swap is in
general.

Worth noting that BLOSUM62 at 0.242 is not nothing. Any report of a new method
at 0.3 should be read against that number.

## The caveat that matters for actually using it

Spearman is computed over the whole assay, and most of an assay is variants
that are obviously fine or obviously dead. If the intended use is choosing a
handful of variants to build, the relevant question is how many of the model's
top picks are genuinely good.

Top-10 recall — of the 10 variants the model ranks highest, the fraction truly
in the top 10% of the assay — across the nine assays: 0.0, 0.1, 0.1, 0.2, 0.2,
0.2, 0.2, 0.6, 0.9. Random guessing scores 0.1.

So on five of nine assays the model's top 10 picks are no better than random,
despite a respectable Spearman on the same data. The model is good at telling
the dead variants from the live ones and much weaker at ranking within the
live ones, which is exactly the regime a protein engineering campaign cares
about. Reporting only the Spearman would hide this completely.

## Where it fails, and why that is predictable

![by taxon](figures/by_taxon.png)

The SARS-CoV-2 assay scores 0.105 against 0.56 for everything else, and
BLOSUM62 beats ESM-2 on it (0.207 vs 0.105). This is not a bad draw. ESM-2's
published average on viral proteins across the full benchmark is 0.261 against
0.457 on human proteins.

The reason is in the training data. The model learns constraint from patterns
of conservation across UniRef, and that signal assumes residues are conserved
because they are structurally or functionally required. Viral proteins are
under immune selection as well, so positions vary across strains for reasons
that have nothing to do with whether the protein folds. The model reads that
variation as tolerance.

Practical consequence: for viral targets, use a method that builds an alignment
of the specific viral family rather than relying on a general protein model.

## The mutational landscape

![landscape](figures/landscape.png)

Predicted effect of all 20 amino acids at every position of E. coli IF1, above
the measured fitness from the same assay. The model has seen no experimental
data. The vertical dark stripes are positions that tolerate nothing, and they
line up with the measured ones at Spearman 0.598.

## Masked marginals vs the cheap alternative

Scoring uses masked marginals, the scheme from the ESM paper: mask the mutated
position, take the log-softmax over the vocabulary, and score the variant as

```
log p(mutant | context) − log p(wildtype | context)
```

The alternative is a single forward pass over the unmutated sequence
("wt-marginals"), which is one pass instead of one per position. On
IF1_ECOLI with ESM-2 8M, wt-marginals actually scored slightly *higher*
(0.217 vs 0.193). That inversion does not hold at 650M, where masking is
clearly better, but it is a reminder that the expensive scheme is not
automatically the right one at every model size.

Masking only positions that appear in the assay, rather than every position in
the protein, gives identical results and is the difference between seconds and
minutes. All 25,501 variants across nine assays scored in 22 seconds on an
RTX 5070 Ti.

## Usage

```python
from variant_effect import ESMScorer, data, evaluate

df, sequence = data.load_assay("BLAT_ECOLX_Stiffler_2015")
scorer = ESMScorer("facebook/esm2_t33_650M_UR50D")
scores = scorer.score_variants(sequence, df["mutant"].tolist())
print(evaluate.spearman(scores, df["DMS_score"].to_numpy()))
```

```
python scripts/run_benchmark.py --include-hard
python scripts/make_figures.py
```

## Layout

```
src/variant_effect/
    mutations.py   parsing "A2M:K5R", validating against the sequence
    scoring.py     ESM-2 masked-marginals and wt-marginals, windowing for
                   proteins over 1022 residues
    baselines.py   BLOSUM62, hydropathy, volume, charge, Pro/Gly, random
    evaluate.py    Spearman with bootstrap CI, NDCG, top-k recall, AUC
    data.py        ProteinGym download and loading
scripts/
    run_benchmark.py   the benchmark and the comparison against official numbers
    make_figures.py
tests/             36 tests; the ones needing a GPU and a model download are
                   marked slow
```

## Setup

```
pip install -e ".[dev]"
pytest -m "not slow"        # 30 tests, no model needed
pytest -m slow              # downloads an 8M checkpoint, ~30 MB
```

Two environment notes, both of which cost time to work out:

- `HF_HUB_OFFLINE=1` after the first download. Without it, `from_pretrained`
  makes a hub request that can hang for tens of minutes on some networks even
  when the model is fully cached. Loading the 650M model went from hanging
  indefinitely to 10 seconds.
- `pip_system_certs` breaks `huggingface_hub` with a `RecursionError` deep in
  `ssl.py`. If downloads fail that way, uninstall it from the environment.

## Still to do

- Run the full 217-assay benchmark rather than a 9-assay subset, and aggregate
  the way ProteinGym does (by UniProt id, then by functional category)
- Compare against an alignment-based method on the viral assay, to test the
  immune-selection explanation directly
- Try ESM-2 3B, which scores *worse* than 650M on the official benchmark
  (0.406 vs 0.414) — worth confirming and understanding
- Structure-aware scoring, since the top-k failure is plausibly about buried
  vs exposed positions

## Note on how this was built

Written with Claude (Anthropic) as a coding assistant. I directed the work:
chose the benchmark and the validation strategy, decided what the baselines
needed to be, and interpreted the results. The findings worth defending —
exact reproduction of the published numbers, the gap between Spearman and
top-k recall, and the viral failure mode — came from insisting on checks that
could fail rather than on output that looked reasonable.
