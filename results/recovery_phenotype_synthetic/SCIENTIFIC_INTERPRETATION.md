# Scientific interpretation: recovery-phenotype synthetic study

## Status

The frozen 2,000-run study passed the prespecified mechanistic gate in all four
graph families. Every graph--scenario cell contains 100 independent seeds.

## Primary findings

Across graph families, the graph-aligned recovery phenotype achieved AUC
0.935--0.956 in the isolated aligned condition and AUC 0.952--0.973 under
fourfold heterogeneous node-noise scales. Exact recovery of the injected graph
mode was 1.0. Mean fitted-graph alignment advantage over the strongest identity,
random, or permuted-basis control was 0.498--0.573.

The graph-free recovery score detected slowing in every regime (overall AUC
0.976--0.988). This is intentional: recovery slowing alone is not evidence for
a graph mechanism. The null-calibrated joint phenotype rejected a slowing
global common mode (overall AUC 0.500), a localized non-graph instability (AUC
0.500), and substantially attenuated non-graph slowing (AUC 0.577). The maximum
graph-specific falsification AUC within any graph family was 0.589, below the
frozen 0.65 ceiling.

## Supported claim

Under an identifiable isolated recovery mode, the estimator distinguishes
graph-aligned critical slowing from graph-independent multichannel slowing,
common-mode persistence, localized non-graph instability, and heterogeneous
noise. This supports progression to the locked CHB-MIT bridge.

## Unsupported claims

These simulations do not establish seizure prediction, universal seizure
criticality, anatomical seizure-onset localization, or clinical utility. Those
claims require the complete locked real-data analysis.

## Audit note

The first execution exposed two protocol-inconsistent implementation defects:
the identity basis was treated as one degenerate graph eigenspace, and negative
absolute graph advantage could receive a positive baseline-relative score.
Both were corrected to match the already written requirement of positive
absolute alignment excess, regression tests were added, all 27 project tests
passed, and the entire 2,000-run study was rerun from the beginning. Only the
corrected outputs are scientifically valid.
