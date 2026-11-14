# Result tables

Every table on this page is written by `python scripts/report_tables.py --write` from the
files under `results/`; nothing here is typed by hand, and a test checks each one
against its renderer. The [README](README.md) discusses them and names the table
each section refers to.

## Replicate-assay agreement and decile shape

`results/topk/` from `scripts/run_topk_analysis.py`. The 37 replicate pairs, what their
agreement is, and whether the shape of the top decile predicts the model's recall.

<!-- BEGIN GENERATED: topk_ceiling -->
| measurement | value |
|---|---|
| replicate assay pairs | 37 |
| proteins covered | 22 |
| median top-10 recall, truth against truth | 0.350 |
| median Spearman over the whole shared set | 0.610 |
| median Jaccard overlap of the two top deciles | 0.161 |
| median per-measurement error (sd units) | 0.592 |
| median top-decile internal spread (sd units) | 0.294 |
| pairs whose decile spread is smaller than one measurement's error, on both sides | 29 of 37 |
| pairs whose top-decile 'reliability' formula is at or below zero (decile selected on the same assay: see the simulated control) | 32 of 37 |
| the model's mean top-10 recall, 217 assays | 0.323 |

Does the shape of the top decile predict the model's recall?

| decile-shape predictor | assays | spearman | p |
|---|---|---|---|
| spread_ratio | 217 | -0.077 | +0.260 |
| range_share | 217 | -0.084 | +0.219 |
| distinct_frac | 217 | +0.083 | +0.225 |
| at_max_frac | 217 | +0.122 | +0.072 |
<!-- END GENERATED: topk_ceiling -->

## Pooled two-assay target

`results/latent_truth/` from `scripts/run_latent_truth_test.py`. Every number behind figures
a, b and c, including the subsets (same-quantity pairs, reliable pairs).

<!-- BEGIN GENERATED: latent_truth -->
| pooled-truth measurement | value |
|---|---|
| replicate pairs with >= 1000 shared variants | 32 |
| proteins covered | 20 |
| shared variant measurements | 143550 |
| median truth-vs-truth top-10 recall (the old 0.350 definition) | 0.350 |
| mean truth-vs-truth top-10 recall, same pairs | 0.408 |
| median top-10 recall, assay against the pooled target | 1.000 |
| median top-10 recall, model against one assay | 0.250 |
| median top-10 recall, model against the pooled target | 0.250 |
| PROTEIN level (primary): proteins | 20 |
| protein-level median paired gain, top-10 recall | +0.033 |
| protein-level proteins better / worse / tied | 12 / 4 / 4 |
| protein-level p, top-10 recall (Wilcoxon / exact / sign-flip) | 0.1023 / 0.1167 / 0.1671 |
| pair level (secondary): median paired gain, top-10 recall | +0.025 |
| pair-level pairs better / worse / tied | 16 / 9 / 7 of 32 |
| pair-level p, top-10 recall (Wilcoxon / exact / sign-flip) | 0.0655 / 0.0710 / 0.1017 |
| median Spearman, model against one assay | 0.478 |
| median Spearman, model against the pooled target | 0.522 |
| protein-level median paired gain, Spearman | +0.051 |
| protein-level proteins better / worse | 20 / 0 |
| protein-level p, Spearman (Wilcoxon / exact / sign-flip) | < 0.0001 / < 0.0001 / < 0.0001 |
| pair-level median paired gain, Spearman | +0.045 |
| pair-level pairs better / worse | 32 / 0 |
| pair-level p, Spearman (Wilcoxon / exact / sign-flip) | < 0.0001 / < 0.0001 / < 0.0001 |
| median inter-assay rank agreement of a pair (a reliability only for same-quantity pairs) | 0.611 |
| pairs at or below 0.3 agreement, pooling uninformative | 4 of 32 |
| same-quantity pairs only: n | 13 |
| same-quantity median paired gain, top-10 recall | +0.100 |
| same-quantity pair-level p, top-10 recall (Wilcoxon / exact / sign-flip) | 0.1246 / 0.1514 / 0.1671 |
| same-quantity proteins | 8 |
| same-quantity protein-level p, top-10 recall (Wilcoxon / exact / sign-flip) | 0.2041 / 0.2969 / 0.2349 |
| same-quantity protein-level p, Spearman (Wilcoxon / exact / sign-flip) | 0.0117 / 0.0078 / 0.0078 |
| reliable pairs only (split-half > 0.3): n | 28 |
| reliable-pair median paired gain, top-10 recall | +0.025 |
| reliable-pair protein-level p, top-10 recall (Wilcoxon / exact / sign-flip) | 0.0769 / 0.0923 / 0.1487 |
| model mean top-10 recall, all 217 assays | 0.323 |
| model mean top-10 recall, the 46 assays that have a replicate | 0.326 |

Self-inclusion control, same pairs: an assay scored against a pooled target that contains it gains +0.475 over truth-against-truth (Wilcoxon p < 0.0001). That gain is self-inclusion (the assay is half of its own target), not a measure of how much attenuation 0.350 contained: see the simulated control below.
<!-- END GENERATED: latent_truth -->

## Simulated controls

`results/sim_controls/` from `scripts/sim_pooling_controls.py`. What self-inclusion does to a
pooled target, and what the decile-reliability formula does on simulated data.

<!-- BEGIN GENERATED: sim_controls -->
Top-10 recall, mean over 400 simulations of 2000 variants:

| simulation | A vs B | A vs pool(A,B) | B vs pool(A,B) | latent vs A | latent vs pool |
|---|---|---|---|---|---|
| two assays with NO shared signal | 0.109 | 0.855 | 0.845 | 0.096 | 0.097 |
| two assays of one latent value, noise sd 0.595 (the real median) | 0.764 | 0.998 | 0.997 | 0.945 | 0.991 |

Decile reliability, formula against truth, top decile of A selected on A itself:

| reps | sigma_hat | variance of A in its top decile | reliability formula | reps with negative formula | true latent variance in that decile | corr(A, B) in that decile | corr(latent, A) in that decile |
|---|---|---|---|---|---|---|---|
| 400 | 0.595 | 0.168 | -1.140 | 100% | 0.298 | +0.323 | +0.480 |
<!-- END GENERATED: sim_controls -->

## Epistasis, five assays

`results/epistasis/` from `scripts/run_epistasis_test.py`. Singles against multi-mutants,
per assay and pooled, with the range-matched gap and the non-additive residual
share.

<!-- BEGIN GENERATED: epistasis -->
| assay | stratum | n | spearman | top-decile recall |
|---|---|---|---|---|
| GFP_AEQVI_Sarkisyan_2016 | multi | 14,481 | +0.123 | 0.069 |
| GFP_AEQVI_Sarkisyan_2016 | single | 1,084 | +0.052 | 0.139 |
| GRB2_HUMAN_Faure_2021 | multi | 2,000 | +0.651 | 0.430 |
| GRB2_HUMAN_Faure_2021 | single | 1,034 | +0.726 | 0.320 |
| HIS7_YEAST_Pokusaeva_2019 | multi | 21,442 | +0.538 | 0.193 |
| HIS7_YEAST_Pokusaeva_2019 | single | 168 | +0.028 | 0.176 |
| PHOT_CHLRE_Chen_2023 | multi | 20,354 | +0.788 | 0.532 |
| PHOT_CHLRE_Chen_2023 | single | 2,122 | +0.649 | 0.217 |
| SPG1_STRSG_Olson_2014 | multi | 2,000 | +0.307 | 0.245 |
| SPG1_STRSG_Olson_2014 | single | 1,045 | +0.276 | 0.212 |

| stratum | assays | variants scored | mean spearman | mean top-decile recall |
|---|---|---|---|---|
| single mutants | 5 | 5,453 | +0.346 | 0.213 |
| all multi-mutants | 5 | 60,277 | +0.481 | 0.294 |
| doubles only | 5 | 7,651 | +0.362 | 0.256 |
| triples only | 3 | 4,978 | +0.321 | 0.196 |
| 5+ substitutions | 3 | 41,648 | +0.443 | 0.279 |

| assay | spearman, single | spearman, multi | gap (single - multi) | range-matched gap | non-additive residual share (includes noise) |
|---|---|---|---|---|---|
| GFP_AEQVI_Sarkisyan_2016 | +0.052 | +0.123 | -0.071 | -0.102 | 0.202 |
| GRB2_HUMAN_Faure_2021 | +0.726 | +0.651 | +0.075 | +0.124 | 0.548 |
| HIS7_YEAST_Pokusaeva_2019 | +0.028 | +0.538 | -0.510 | -0.171 | 1.036 |
| PHOT_CHLRE_Chen_2023 | +0.649 | +0.788 | -0.139 | -0.117 | 0.894 |
| SPG1_STRSG_Olson_2014 | +0.276 | +0.307 | -0.031 | -0.002 | 0.249 |
<!-- END GENERATED: epistasis -->

## Viral gap by method family

`results/families/` from `scripts/run_family_analysis.py`. The 14 families behind figure f.

<!-- BEGIN GENERATED: family_viral_gap -->
| family | checkpoints | reads an alignment | virus | non-viral | gap |
|---|---|---|---|---|---|
| ESM-C (seq LM, diverse training set) | 2 | no | +0.238 | +0.466 | -0.228 |
| ESM-2 (seq LM, UniRef) | 6 | no | +0.211 | +0.397 | -0.185 |
| ESM-1 (seq LM, UniRef) | 3 | no | +0.272 | +0.439 | -0.167 |
| ProtSSN (structure-aware) | 3 | no | +0.364 | +0.481 | -0.118 |
| Progen2 (seq LM, autoregressive) | 5 | no | +0.335 | +0.393 | -0.058 |
| VESPA (MSA + score priors) | 3 | yes | +0.424 | +0.461 | -0.037 |
| EVE / EVmutation (MSA only) | 3 | yes | +0.420 | +0.448 | -0.029 |
| Tranception, retrieval (MSA) | 3 | yes | +0.417 | +0.444 | -0.027 |
| TranceptEVE (MSA + retrieval) | 3 | yes | +0.451 | +0.475 | -0.025 |
| GEMME (MSA only) | 1 | yes | +0.466 | +0.476 | -0.009 |
| Protein-RAG (retrieval-augmented) | 2 | yes | +0.518 | +0.521 | -0.003 |
| Site-Independent (MSA only) | 1 | yes | +0.369 | +0.369 | +0.000 |
| Tranception, no retrieval (MSA) | 3 | yes | +0.353 | +0.346 | +0.006 |
| RITA (seq LM, autoregressive) | 4 | no | +0.376 | +0.355 | +0.021 |
<!-- END GENERATED: family_viral_gap -->
