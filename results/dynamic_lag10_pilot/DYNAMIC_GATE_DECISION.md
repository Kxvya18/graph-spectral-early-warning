# Version-2 graph-dynamics gate decision

**Decision: failed. Do not add machine learning to this estimator or use it for
the confirmatory EEG study.**

This decision covers the covariance-lag/generalized-eigenvector estimator in
`results/dynamic_pilot` and the prespecified 10-step-lag revision in this
directory. Both pilots used 20 independent seeds for four graph families and
two transition models (160 model--graph--seed runs), with three fixed horizons.
Neither was a reduced smoke experiment.

## What version 2 tested

- A ridge-regularized generalized eigenproblem for the slow dynamical direction.
- A graph-constrained modal autoregressive estimator.
- A graph-alignment concentration statistic.
- Diagonal graph-dynamic fit and cross-covariance diagonality.
- Identity, random, permuted, and perturbed graph controls.
- Exact-mode, top-three, near-critical-band, spectral-error, node-localization,
  and subspace-localization endpoints.
- Identical causal windowing and post-processing for every basis.

## Predictive result

At the nearest horizon (160 steps), `GraphDyn_Level` exceeded NodeMax by 0.0107
AUC (95% paired bootstrap interval 0.0039 to 0.0185) in the graph-commuting OU
model and by 0.0144 (0.0024 to 0.0269) in the nonlinear model. Differences at
the two earlier horizons were not reliably positive.

This modest predictive gain is not graph-specific. In the linear model, all
paired differences between the true graph and identity/random/permuted/
perturbed bases included zero. In the nonlinear model at horizon 160, the true
graph was worse than identity, random, and perturbed controls. The direct
graph-fit and diagonality endpoints were small and inconsistent; the perturbed
graph was sometimes equivalent or superior.

## Localization result

At horizon 160, exact interval mode recovery was 8.75% in both models. Recovery
of any member of the prespecified near-critical band was 15.0% in the linear
model and 22.5% in the nonlinear model. Top-three band recovery was 32.5% and
38.75%, respectively. Node and subspace localization correlations remained
close to zero. Performance deteriorated at earlier horizons.

## Identifiability diagnosis

The simulator's exact critical eigenmode is not adequately separated at the
available sampling scale. At horizons 160, 360, and 560, the mean gap between
the largest and second-largest one-step persistence values is approximately
0.0017, 0.0011, and 0.0006. Each realization contains roughly two to six modes
within 0.05 continuous-time drift units of the leading mode. These differences
are substantially smaller than finite-window estimation uncertainty for a
160-sample, 48-dimensional window.

This explains why exact-mode localization is unstable, but it does not excuse
the graph-specificity failure: the correct graph also fails the matched
wrong-basis comparisons and direct graph-fit endpoints.

## Consequence for machine learning

A classifier or GNN may learn to distinguish late-transition windows because
the generic multichannel signal is strong. It would not demonstrate that graph
topology causes or uniquely resolves the warning. Adding ML now would therefore
change the subject to empirical transition classification while obscuring the
failed mechanistic hypothesis.

ML remains conditional on a new estimator passing a population-to-finite-sample
identifiability study. It must then enter as a nested-cross-validated incremental
model after graph-free features, not as a replacement for the graph controls.

## Required next study

1. Verify the estimator against the population covariance and lag-covariance,
   where sampling error is absent.
2. Map recovery as a function of eigengap, window length, dimension, lag, and
   signal-to-noise ratio.
3. Include both a deliberately separated single-mode condition and a
   near-degenerate critical-subspace condition.
4. Use exact-mode recovery only in the separated condition; use projector or
   principal-angle recovery for the near-degenerate subspace.
5. Accumulate graph likelihood over a long stationary training segment instead
   of expecting a short warning window to identify the graph.
6. Freeze a new estimator before repeating the 100-seed confirmatory gate.

Until those steps pass, the defensible conclusion remains a multichannel
variance/persistence warning effect without demonstrated graph specificity or
localization.
