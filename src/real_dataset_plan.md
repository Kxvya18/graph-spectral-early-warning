# Real Dataset Integration Plan

The chapter should first present the controlled synthetic result, because the
true destabilizing eigenmode is known. Real datasets can then be used as
external plausibility and translational validation.

## Candidate Datasets

1. EEG seizure onset
   - CHB-MIT scalp EEG.
   - Use channels as graph nodes.
   - Estimate graph from coherence, phase-locking value, or correlation in
     preictal windows.
   - Endpoint: preictal versus interictal discrimination and warning lead time.

2. ECG arrhythmia
   - MIT-BIH Arrhythmia Database.
   - For single-lead records, graph construction is weak.
   - Better option: multi-lead ECG datasets where leads define anatomical views.
   - Endpoint: warning before arrhythmia-labeled segments.

3. ICU/sepsis physiology
   - PhysioNet Challenge or MIMIC-derived multivariate time series.
   - Variables become nodes; graph can be estimated from lagged correlation or
     clinical organ-system grouping.
   - Endpoint: deterioration/sepsis onset classification.

## Real-Data Claims Should Be Conservative

Synthetic experiments can prove modal mechanism and localization accuracy.
Real datasets should claim early warning performance and interpretable modal
patterns, not exact ground-truth eigenmode localization.

## Real-Data Pipeline

1. Preprocess channels, remove artifacts, and resample.
2. Estimate an adjacency matrix from a baseline window.
3. Construct the graph Laplacian and eigenbasis.
4. Project sliding windows into graph modes.
5. Compute GSEWS score and compare with node-averaged EWS, PCA-EWS, and raw
   signal energy.
6. Report AUC, warning lead time, false alarm rate, and qualitative
   localization maps.

