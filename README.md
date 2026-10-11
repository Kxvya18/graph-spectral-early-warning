# Graph-Spectral Early Warning Signals for Physiological Transitions

This project is independent from the CCLD chapter.

Goal: build theory-driven experiments for graph-spectral early warning signals
(GSEWS) in coupled physiological networks, then extend the pipeline to real
EEG/ECG/ICU datasets.

The synthetic experiments are designed so the true destabilizing graph mode is
known. This lets us test the central claims:

1. Graph eigenmodes exhibit critical slowing down before transition.
2. Modal variance and autocorrelation rise before node-averaged baselines.
3. The dominant warning eigenvector localizes the unstable network region.
4. Coupling topology changes which warning mode becomes unstable.

## Research status after validity audit

The original results under `results/linear_theory_gsews`,
`results/nonlinear_physiology_gsews`, and `results/chbmit_subset_*` are retained
as legacy pilot results. They must not be used as journal evidence because the
legacy pipeline used unequal modal noise, centered (noncausal) smoothing,
center-based window eligibility, and a theory/simulation mismatch.

The corrected pipeline is implemented in:

- `src/rigorous_simulate.py`: graph-commuting OU and nonlinear graph-pattern
  models with a verified spectral transition and isotropic node noise.
- `src/causal_gsews.py`: causal windowing, trailing trend/level scores, matched
  comparators, and efficient wrong-graph controls.
- `src/rigorous_evaluate.py`: strict pre-transition endpoints and paired
  bootstrap comparisons.
- `src/run_rigorous_synthetic.py`: locked multi-graph benchmark.
- `src/download_chbmit_locked.py`: resumable 24-subject CHB-MIT cohort download.
- `src/run_chbmit_locked.py`: training-only patient-specific evaluation with a
  locked seizure prediction horizon and occurrence period.
- `src/run_chbmit_v3_ml.py`: version-3 EEG evaluation with independent graph
  validation, matched wrong-graph controls, isolation diagnostics, and nested
  grouped graph-free versus graph-augmented elastic-net models.
- `src/ml_incremental.py`: record-grouped inner model selection and
  record-held-out probability calibration.

The 160-run corrected synthetic gate found strong transition discrimination
but no consistent advantage of the true graph basis over identity, random, or
permuted bases, and poor critical-mode recovery. The 800-run production study
is therefore halted under the current estimator. See
`results/rigorous_pilot/SYNTHETIC_GATE_DECISION.md`.

A separate covariance-lag/generalized-eigenvector version 2 and a fixed
10-step-lag version 2.1 were subsequently implemented and evaluated in two more
full 160-run pilots. They also failed graph-specificity and localization. The
longer-lag version modestly beat NodeMax at the nearest horizon but did not beat
wrong graph bases. See
`results/dynamic_lag10_pilot/DYNAMIC_GATE_DECISION.md`. Machine learning is
therefore deferred until an identifiability study establishes recoverable graph
dynamics; adding a classifier now would not repair the mechanistic claim.

The subsequent 15,200-run population/finite-sample identifiability study found
that the theory is identifiable but the original 160-sample warning regime is
not. Universal recovery across all four graph families required a persistence
gap of 0.15 with 2,560 clean observations, or 5,120 observations under 50%
measurement noise. See
`results/identifiability_boundary/IDENTIFIABILITY_DECISION.md`. This establishes
the design requirements for a version-3 transition benchmark: learn and freeze
the graph on a long stationary segment, test exact localization only for an
isolated mode, and use subspace recovery for near-degenerate modes.

Version 3 is now complete. The two-regime pilot rejected the near-degenerate
subspace condition and advanced only the isolated-mode condition. The
confirmatory isolated study used 100 seeds for every model--graph combination
(800 runs). Both models passed held-out graph selection, NodeMax comparisons,
identity/random/permuted controls, and exact localization. The nonlinear model
also passed the strict perturbed-graph gate. The linear model was equivalent to
the 20%-weight-perturbed ring graph at the nearest horizon, so it passes the
core mechanism gate but not the frozen strict all-controls gate. See
`results/v3_isolated_primary/V3_DECISION.md`. This justifies proceeding to the
locked EEG bridge; it does not itself establish seizure prediction.

## Reproduce the corrected synthetic gate

```bash
cd "/Users/boolu/Desktop/simulations/ALL PROJECTS/graph_spectral_ews_physiology"
MPLCONFIGDIR=.matplotlib_cache .venv/bin/python -m unittest discover -s tests -v
MPLCONFIGDIR=.matplotlib_cache .venv/bin/python src/run_rigorous_synthetic.py \
  --config configs/rigorous_pilot.json
```

## Acquire and evaluate the locked CHB-MIT version-3 cohort

The frozen manifest contains all 141 seizure-containing records, six
temporally spread nonseizure records per subject (144 records), and 57 extra
immediate-predecessor records needed for causal preictal context: 342 unique
EDFs in total. Six controls are required so every outer fold can use four
graph-fit records, one independent graph-validation record, and one held test
record. Plan for roughly 15 GB of downloads. The downloader resumes an
interrupted `.part` file.

```bash
cd "/Users/boolu/Desktop/simulations/ALL PROJECTS/graph_spectral_ews_physiology"
./run_chbmit_v3_full.sh
```

The current workspace contains all subject summaries and three CHB01 seizure
EDFs in the version-3 data directory. Feature extraction has been verified on
those real EDFs (22 usable channels and 1,800 causal two-second epochs per
record). The full 342-record analysis will not start until the locked manifest
is complete. Its primary test is `GraphDyn_Raw` versus `NodeMax`; its secondary
ML test is graph-augmented versus graph-free elastic net with every split kept
at record level.

The scientific decision rules and journal route are frozen in
`manuscripts/journal/JOURNAL_PROTOCOL_AND_DECISION.md`.

## Graph-aligned recovery-rate phenotype

The versioned recovery-phenotype extension separates graph-free slowing from
graph-specific spatial alignment. It estimates a regularized lagged recovery
operator, maps the leading reversible eigenvector back into state space, and
requires its graph-modal concentration to exceed matched identity, random, and
permuted-basis controls. Common-mode and non-graph slowing are explicit
falsification conditions rather than positive examples.

- `src/recovery_rate.py`: causal recovery estimator, state-direction mapping,
  graph alignment, and training-only phenotype calibration.
- `src/recovery_simulate.py`: verified aligned and falsification regimes.
- `src/run_recovery_synthetic.py`: frozen 100-seed-per-condition study.
- `src/run_chbmit_recovery.py`: locked seizure-held-out CHB-MIT evaluation.
- `manuscripts/journal/RECOVERY_PHENOTYPE_PROTOCOL.md`: claims and decision
  rules frozen before full CHB-MIT evaluation.

Run the full synthetic study with:

```bash
MPLCONFIGDIR=.matplotlib_cache .venv/bin/python \
  src/run_recovery_synthetic.py \
  --config configs/recovery_phenotype_synthetic.json
```

After the frozen EDF manifest is available, run the real-data workflow with:

```bash
./run_recovery_phenotype_full.sh
```

## Environment

Use `requirements-journal.txt` for the exact package versions used during the
corrected audit. A public journal release still requires the author's choice of
software license, final author metadata, a tagged repository release, and a DOI
archive.
