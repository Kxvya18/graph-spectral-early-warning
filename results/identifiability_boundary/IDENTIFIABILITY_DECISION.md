# Mechanistic identifiability decision

**Conclusion:** the graph-dynamical mechanism is mathematically identifiable,
but the original transition experiment asks a short window to resolve a mode
gap that is two orders of magnitude too small. The estimator passes when the
eigengap and training information are made commensurate with finite-sample
uncertainty.

## Experiments completed

- Primary grid: 9,600 experiments covering four graph families, four critical
  persistence gaps, three training lengths, two observation-noise levels, and
  100 independent graph/noise seeds per condition.
- Dimension grid: 2,400 experiments covering 12, 24, and 48 nodes at gap 0.05,
  640 training observations, two noise levels, four graph families, and 100
  seeds per condition.
- Boundary grid: 3,200 experiments covering gaps 0.10 and 0.15, training lengths
  2,560 and 5,120, two noise levels, four graph families, and 100 seeds per
  condition.
- Every fitted graph model was evaluated on an independent 640-observation test
  segment. Identity, random, permuted, and perturbed graphs were competitors.

## Population result

The true graph basis diagonalizes the population transition operator to
numerical precision: the maximum true-basis operator residual was
5.1e-30 and the minimum cross-covariance diagonality was effectively 1. The
expected held-out prediction error of the true diagonal graph model was lower
than the best wrong graph in every population condition. The mechanistic model
is therefore identifiable with unlimited data.

## Finite-sample result

The frozen pass criterion required all of the following lower 95% confidence
bounds to exceed 0.80: correct graph selection, exact mode recovery, and squared
cosine recovery of the critical vector. The held-out MSE advantage over the
best wrong graph also had to have a lower confidence bound above zero.

Only 3 of 96 initial 48-node conditions passed. All used gap 0.15, 1,280 clean
training observations, and ring, small-world, or modular graphs. Scale-free
graphs required more data.

The complete all-graph pass regimes were:

1. Persistence gap 0.15, 2,560 training observations, no observation noise.
2. Persistence gap 0.15, 5,120 observations, with either zero or 50% observation noise.
3. Persistence gap 0.10, 5,120 observations, no observation noise.

At gap 0.15 and 5,120 clean observations, graph-selection lower confidence
bounds were 0.90--0.97 across graph families, exact-mode recovery was 100%,
critical-vector squared-cosine lower bounds were 0.935--0.959, and every
held-out MSE advantage was positive. With 50% observation noise, the same gap
and length still passed all graph families; vector-recovery lower bounds were
0.888--0.943.

At gap 0.05 and 640 samples, no tested dimension passed the complete gate. Reducing
the problem to 12 nodes improved mode and vector recovery but did not make graph
selection reliable against the best perturbed/wrong graph.

## Why the transition pilots failed

The original transition benchmark used 160-sample warning windows. Its mean
one-step persistence gap was only approximately 0.0017 at the nearest horizon,
0.0011 at the middle horizon, and 0.0006 at the earliest horizon. The
identifiability study required gaps of roughly 0.10--0.15 and 2,560--5,120
training observations for reliable 48-node recovery. The original problem was
therefore attempting exact localization with approximately 60--250 times less
mode separation and 16--32 times fewer observations than the demonstrated pass
regime.

Standardizing a maximum modal score still detects the global variance/
persistence change, explaining the high AUC. But it cannot reliably attribute
that change to one graph eigenmode, and wrong orthogonal bases inherit the same
global transition signal.

## How the mechanistic gate can be passed honestly

The next transition benchmark must separate graph learning from online warning:

1. Estimate and select the graph-dynamical basis using a long stationary
   training segment of at least 5,120 observations, with held-out likelihood
   against identity/random/permuted/perturbed graphs.
2. Freeze that basis before the parameter ramp begins.
3. Include an isolated-mode condition with a one-step persistence gap of at
   least 0.10. Exact mode and node localization are valid endpoints only here.
4. Retain the original near-degenerate spectral-band condition, but define the
   estimand as a critical subspace. Evaluate projector distance or principal
   angles, not exact eigenvector index.
5. Accumulate warning evidence across longer causal histories or a sequential
   state-space filter; do not expect a two-minute/high-dimensional window to
   re-identify the graph.
6. Rerun a new pilot and require the correct frozen graph to beat every wrong
   basis as well as NodeMax before launching 100-seed confirmatory experiments.

These changes do not manufacture a positive result. They align the claim with
what is statistically identifiable: exact localization for an isolated mode,
and subspace localization for near-degenerate modes.

## Machine-learning decision

Machine learning is still premature for the near-degenerate 160-sample regime.
After the redesigned mechanistic benchmark passes, an elastic-net model may be
added to test incremental prediction from frozen graph-dynamic features. A GNN
should remain secondary and must retain identical wrong-graph and graph-free
architecture controls.
