# CHB-MIT version-3 execution status

Status on 4 August 2026: protocol and code are frozen; full cohort execution is
pending the resumable EDF download.

## Completed

- Frozen manifest: 342 EDFs (141 seizure, 144 nonseizure, 57 context-only).
- Causal feature extraction verified on three real CHB01 seizure EDFs.
- Each verified record produced 1,800 two-second epochs across 22 usable
  channels.
- Frozen training graph, independent graph validation, identity/random/
  permuted/20%-weight-perturbed controls, modal-isolation diagnostics, and
  record-grouped elastic-net comparisons are implemented.
- All 22 unit and end-to-end tests pass.

## Not yet scientific results

The three available EDFs are insufficient for a fold and are not an analysis
cohort. No EEG performance or mechanism claim may be made from the partial
inventory.

## Required terminal run

```bash
cd "/Users/boolu/Desktop/simulations/ALL PROJECTS/graph_spectral_ews_physiology"
./run_chbmit_v3_full.sh
```

The downloader resumes `.part` files. Do not modify the configuration or
manifest after examining any held-out result.
