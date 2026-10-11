# CHB-MIT recovery-phenotype execution status

## Completed

- Frozen recovery-phenotype protocol and estimator.
- Locked 24-subject manifest retained without cohort changes.
- Participant/seizure-held-out graph fitting, scaling, calibration, thresholds,
  and matched wrong-graph controls implemented.
- Real EDF preprocessing verified on every currently available CHB01 file:
  14 records, 22 usable channels, generally 1,800 causal two-second epochs.
- Twenty-seven deterministic project tests pass.

## Not yet a clinical result

Only 14 of 342 locked EDF records are currently available. The current files
are an acquisition/readiness subset and must not be summarized as seizure
prediction evidence.

## Storage blocker

On 10 August 2026 the internal data volume had approximately 20 GiB free. The
remaining locked download requires roughly 15 GiB before feature caches and
analysis outputs. The full workflow was therefore not started, because it
would leave unsafe operating-system headroom.

Free at least 25--30 GiB more or place the graph-spectral project/data directory
on an external SSD, then run:

```bash
cd "/Users/boolu/Desktop/simulations/ALL_PROJECTS/graph_spectral_ews_physiology"
caffeinate -dimsu ./run_recovery_phenotype_full.sh
```

The downloader resumes the existing partial file and the real analysis starts
only after the complete locked manifest is present.
