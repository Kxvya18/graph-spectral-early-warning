# Claim Status for GSEWS Chapter

This file separates claims supported by theory-driven simulations from claims supported by real clinical data.

## Satisfied Claims

- Linearized graph-spectral early warning: satisfied. GSEWS AUC `0.727` with exact modal localization in controlled modal OU experiments.
- Closed-form modal mechanism: experimentally supported under the linearized SDE setting where the true destabilizing eigenmode is known.
- Nonlinear physiology-inspired warning robustness: partly-to-strongly supported. GSEWS AUC `0.737` in reduced excitable-network simulations.
- Actual multichannel EEG processing: real EDF-based CHB-MIT subset validation. Best current CHB-MIT setting gives GSEWS AUC `0.671` versus best baseline `0.618`.

## Partially Satisfied Claims

- Physiological transitions such as seizure/arrhythmia/sepsis: seizure EEG validation is preliminarily supported; arrhythmia and sepsis remain future datasets.
- Spatial localization in physiological recordings: exact localization is validated only in linearized simulations. Real EEG localization is currently interpretable modal mapping, not ground-truth anatomical localization.
- Network-aware extension beyond aggregation: strengthened with NodeMax, GraphSmooth, and EdgeCorr baselines in addition to NodeAvg, PCA, and Energy.

## Not Yet Fully Satisfied

- Multi-subject clinical validation: not yet. Current real EEG subset is from one CHB-MIT subject and three seizure EDFs.
- Cardiac arrhythmia and septic-collapse validation: not yet.
- Closed-loop clinical intervention: not experimentally demonstrated; should remain a future application.

## Honest Chapter Wording Right Now

The chapter can claim that GSEWS is theoretically derived and validated in controlled coupled stochastic network models, and that preliminary real CHB-MIT EEG experiments show seizure-data feasibility. It should not yet claim full clinical seizure prediction, arrhythmia prediction, sepsis prediction, or ground-truth localization in patients.