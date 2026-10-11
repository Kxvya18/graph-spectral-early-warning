# Version-3 two-regime pilot decision

## Decision

Promote the isolated-mode regime to 100-seed confirmation. Do not promote the
near-degenerate spectral-band regime.

The pilot contained 320 independent runs: 20 seeds for each combination of two
models, two spectral regimes, and four graph families. Every run used 5,120
stationary graph-fitting observations, an independent 640-observation graph
validation segment, a frozen graph basis, and matched identity/random/permuted/
perturbed controls.

## Isolated mode

The isolated regime produced perfect exact-mode and node-pattern recovery in
both models. The raw graph-dynamic warning score exceeded NodeMax and every
wrong/perturbed graph in the pooled paired analysis at all three horizons.
Held-out graph selection was 90--100% by model and graph family, although the
20-seed linear scale-free lower interval was 0.75 and required confirmation.

## Near-degenerate band

The near-degenerate regime did not pass. Held-out graph-selection lower bounds
were only 0.40--0.60. At the nearest horizon, subspace-overlap lower bounds were
0.57 for the linear model and 0.63 for the nonlinear model; performance fell at
earlier horizons. The true graph did not consistently outperform the mildly
perturbed graph.

This negative result is retained as a boundary condition: short causal windows
cannot uniquely identify or localize a tightly clustered critical graph
subspace under the present estimator.

## Consequence

Only the isolated regime was advanced to `configs/v3_isolated_primary.json`.
The near-degenerate result must remain visible in the journal manuscript and
prevents generalizing exact-mode localization to arbitrary graph spectra.
