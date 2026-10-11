# Synthetic graph-specificity gate decision

**Decision:** Do not launch or publish the 100-seed-per-graph production study
under the current GSEWS estimator.

## What was corrected

- The drift is a graph-commuting spectral operator rather than an arbitrarily
  planted modal schedule disconnected from the theory.
- The nonlinear experiment has a verified graph pattern-forming bifurcation.
- Node noise is equal and isotropic.
- Every window ends before its eligibility cutoff.
- Smoothing and trend estimation are trailing-only.
- Baseline calibration is separated from the parameter ramp.
- Comparator post-processing is matched.
- Identity, random, permuted, and perturbed graph bases are explicit controls.
- Three prespecified prediction horizons are evaluated from identical runs.
- Results use 80 runs per model (20 seeds in each of four graph families).

## Gate results

The best level-based modal statistic separated baseline and pre-transition
periods and exceeded NodeMax at the two nearer horizons. For the linear
graph-commuting OU model, the paired AUC advantage over NodeMax was 0.0132
(95% bootstrap interval 0.0065 to 0.0210) at horizon 160 and 0.0143 (0.0074 to
0.0236) at horizon 360. For the nonlinear graph pattern normal form, the
corresponding differences were 0.0242 (0.0150 to 0.0344) and 0.0161 (0.0097 to
0.0245).

However, the true graph basis did not outperform the identity, random, or
permuted bases in a consistent or material way. Most paired differences were
near zero with intervals spanning zero. This means the observed discrimination
cannot be attributed specifically to the supplied graph spectrum.

Localization also failed the prespecified scientific standard. Depending on
model and horizon, exact interval-level mode recovery was 0% to 11.25%, top-3
recovery was 6.25% to 23.75%, mean normalized spectral error was approximately
0.16 to 0.25, and localization correlations were close to zero.

## Defensible conclusion

The corrected experiments support a multichannel variance/persistence warning
effect in the simulated transition. They do not support the stronger claims
that the current GSEWS score is graph-specific, that it reliably identifies the
least stable graph mode, or that it localizes the critical node pattern.

## Research consequence

The 800-run production configuration remains in the repository but should not
be executed or cited as confirmatory evidence until the estimator is redesigned
and passes this gate. A future redesign should estimate the modal recovery
operator directly, for example through regularized state-space/OU likelihood,
joint covariance-lag generalized eigenanalysis, or a graph-constrained dynamic
mode estimator. It must then beat the same wrong-basis controls without changing
the locked gate criteria.

## Files supporting this decision

- `run_metrics.csv`: run-level scores for both models and all horizons.
- `summary.csv`: descriptive summaries.
- `paired_auc_bootstrap.csv`: paired comparisons for the trend statistic.
- `paired_level_auc_bootstrap.csv`: paired comparisons for the level statistic.
- `run_truth.csv`: seeds and true critical spectral modes.

