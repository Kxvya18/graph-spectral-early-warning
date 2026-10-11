from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from graphs import eigensystem, laplacian
from recovery_rate import (
    RecoveryTrajectory,
    alignment_excess,
    apply_phenotype_calibration,
    compute_recovery_trajectory,
    fit_phenotype_calibration,
    realign_trajectory,
)
from rigorous_simulate import perturb_adjacency
from run_chbmit_locked import (
    RecordFeatures,
    apply_scaler,
    build_context_lookup,
    count_alarm_episodes,
    estimate_training_graph,
    event_eligibility_frame,
    fit_calibration,
    fit_robust_scaler,
    graph_bases,
    inventory_frame,
    preictal_overlaps_prior_seizure,
    prepare_features,
    score_record,
    select_channels,
    subject_cluster_bootstrap,
    summarize_overall,
    window_raw,
)


CORE_CONTROLS = ["IdentityBasis", "RandomGraph", "PermutedGraph"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/chbmit_recovery_phenotype.json")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--subjects", nargs="*")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if args.subjects:
        cfg["subjects"] = args.subjects
    source_cfg = json.loads(Path(cfg["manifest_source_config"]).read_text())
    source_hash = hashlib.sha256(
        json.dumps(source_cfg, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    root = Path(cfg["data_dir"])
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / "locked_manifest.json").read_text())
    if {row.get("config_sha256") for row in manifest} != {source_hash}:
        raise RuntimeError("locked manifest no longer matches its source protocol")
    manifest = [row for row in manifest if row["subject"] in cfg["subjects"]]
    available = [row for row in manifest if Path(row["local_path"]).exists()]
    missing = [row for row in manifest if not Path(row["local_path"]).exists()]
    (output / "missing_manifest.json").write_text(json.dumps(missing, indent=2))
    if not available:
        raise RuntimeError("no locked CHB-MIT EDF records are available")
    cache = Path(cfg["feature_cache_dir"])
    cache.mkdir(parents=True, exist_ok=True)
    records = prepare_features(available, root, cache, cfg)
    inventory = inventory_frame(records, manifest)
    inventory.to_csv(output / "record_inventory.csv", index=False)
    context_lookup = build_context_lookup(manifest)
    event_eligibility_frame(records, context_lookup, cfg).to_csv(
        output / "event_eligibility.csv", index=False
    )
    if args.prepare_only:
        print(inventory.to_string(index=False))
        return
    if missing:
        raise RuntimeError(
            f"locked cohort is incomplete ({len(missing)} EDFs missing); resume download"
        )

    by_key = {(record.subject, record.record): record for record in records}
    event_rows: list[dict] = []
    fold_rows: list[dict] = []
    for subject in cfg["subjects"]:
        subject_records = [record for record in records if record.subject == subject]
        seizures = [
            record for record in subject_records
            if record.kind == "seizure" and record.seizures
        ]
        nonseizures = [record for record in subject_records if record.kind == "nonseizure"]
        if len(seizures) < int(cfg["minimum_seizure_records_for_patient_specific_cv"]):
            continue
        if len(nonseizures) < int(cfg["nonseizure_records_per_subject"]):
            continue
        for fold, held_seizure in enumerate(seizures):
            held_non = nonseizures[fold % len(nonseizures)]
            outer_non = [record for record in nonseizures if record.record != held_non.record]
            graph_validation = outer_non[fold % len(outer_non)]
            graph_training = [
                record for record in outer_non
                if record.record != graph_validation.record
            ]
            context_name = context_lookup.get((subject, held_seizure.record))
            context = by_key.get((subject, context_name)) if context_name else None
            fold_records = [held_seizure, held_non, graph_validation, *graph_training]
            if context is not None:
                fold_records.append(context)
            common = sorted(set.intersection(*(set(record.channels) for record in fold_records)))
            if len(common) < int(cfg["minimum_common_channels"]):
                fold_rows.append({
                    "subject": subject, "fold": fold,
                    "held_seizure": held_seizure.record,
                    "status": "insufficient_common_channels", "n_common": len(common),
                })
                continue
            try:
                rows = run_fold(
                    subject, fold, held_seizure, held_non, context,
                    graph_training, graph_validation, common, cfg,
                )
            except Exception as exc:
                fold_rows.append({
                    "subject": subject, "fold": fold,
                    "held_seizure": held_seizure.record,
                    "status": f"failed: {exc}",
                })
                continue
            event_rows.extend(rows)
            fold_rows.append({
                "subject": subject, "fold": fold,
                "held_seizure": held_seizure.record,
                "held_nonseizure": held_non.record,
                "graph_validation": graph_validation.record,
                "status": "ok", "n_common": len(common),
                "n_events": len(held_seizure.seizures),
            })
            print(f"complete subject={subject} fold={fold}", flush=True)

    events = pd.DataFrame(event_rows)
    folds = pd.DataFrame(fold_rows)
    folds.to_csv(output / "fold_inventory.csv", index=False)
    if events.empty:
        raise RuntimeError("no eligible recovery-phenotype folds completed")
    events.to_csv(output / "event_metrics.csv", index=False)
    subjects = summarize_recovery_subjects(events)
    subjects.to_csv(output / "subject_metrics.csv", index=False)
    overall = summarize_overall(subjects)
    overall.to_csv(output / "overall_metrics.csv", index=False)
    comparisons = subject_cluster_bootstrap(subjects, cfg)
    comparisons.to_csv(output / "subject_clustered_comparisons.csv", index=False)
    write_report(output, cfg, inventory, folds, overall, comparisons, events)
    print(overall.to_string(index=False))


def run_fold(subject: str, fold: int, held_seizure: RecordFeatures,
             held_non: RecordFeatures, held_context: RecordFeatures | None,
             graph_training: list[RecordFeatures], graph_validation: RecordFeatures,
             channels: list[str], cfg: dict) -> list[dict]:
    train_arrays = [select_channels(record, channels) for record in graph_training]
    if sum(len(array) for array in train_arrays) < int(cfg["graph_fit_minimum_epochs"]):
        raise RuntimeError("insufficient graph-fitting epochs")
    median, scale = fit_robust_scaler(train_arrays)
    train_scaled = [apply_scaler(array, median, scale) for array in train_arrays]
    validation_scaled = apply_scaler(select_channels(graph_validation, channels), median, scale)
    if len(validation_scaled) < int(cfg["graph_validation_minimum_epochs"]):
        raise RuntimeError("insufficient graph-validation epochs")
    graph = estimate_training_graph(train_scaled, int(cfg["graph_top_k"]))
    true_values, true_vectors = eigensystem(laplacian(graph))
    rng = np.random.default_rng(int(cfg["seed"]) + fold)
    bases = {
        ("TrainingGraph" if name == "TrueGraph" else name):
        (basis, adjacency, exclude)
        for name, basis, adjacency, exclude in graph_bases(graph, true_vectors, rng)
    }
    perturbed = perturb_adjacency(
        graph, float(cfg["graph_perturbation_strength"]),
        np.random.default_rng(int(cfg["seed"]) + 100_000 + fold),
    )
    perturbed_values, perturbed_vectors = eigensystem(laplacian(perturbed))
    bases["PerturbedGraph"] = (perturbed_vectors, perturbed, True)

    validation_bundle = trajectory_bundle(
        validation_scaled, true_vectors, true_values, bases, cfg
    )
    true_validation = validation_bundle["TrainingGraph"]
    wrong_validation = [validation_bundle[name] for name in CORE_CONTROLS]
    calibration = fit_phenotype_calibration(
        [true_validation], [[record] for record in wrong_validation]
    )
    validation_scores = apply_phenotype_calibration(
        true_validation, wrong_validation, calibration
    )
    thresholds = {
        method: float(np.quantile(values, float(cfg["alarm_training_quantile"])))
        for method, values in validation_scores.items()
    }

    legacy_raw = [window_raw(array, true_vectors, graph, cfg) for array in train_scaled]
    legacy_calibration = fit_calibration(legacy_raw, cfg, exclude_constant=True)

    seizure_array = select_channels(held_seizure, channels)
    context_seconds = 0.0
    all_seizures: list[tuple[float, float]] = []
    if held_context is not None:
        context_array = select_channels(held_context, channels)
        context_seconds = len(context_array) * float(cfg["epoch_seconds"])
        seizure_array = np.vstack([context_array, seizure_array])
        all_seizures.extend(held_context.seizures)
    all_seizures.extend(
        (start + context_seconds, stop + context_seconds)
        for start, stop in held_seizure.seizures
    )
    seizure_scaled = apply_scaler(seizure_array, median, scale)
    non_scaled = apply_scaler(select_channels(held_non, channels), median, scale)
    test_bundle = trajectory_bundle(seizure_scaled, true_vectors, true_values, bases, cfg)
    non_bundle = trajectory_bundle(non_scaled, true_vectors, true_values, bases, cfg)
    test_scores = apply_phenotype_calibration(
        test_bundle["TrainingGraph"],
        [test_bundle[name] for name in CORE_CONTROLS], calibration,
    )
    non_scores = apply_phenotype_calibration(
        non_bundle["TrainingGraph"],
        [non_bundle[name] for name in CORE_CONTROLS], calibration,
    )
    legacy_test = score_record(
        window_raw(seizure_scaled, true_vectors, graph, cfg), legacy_calibration, cfg
    )
    legacy_non = score_record(
        window_raw(non_scaled, true_vectors, graph, cfg), legacy_calibration, cfg
    )
    for method in ["NodeMax", "GSEWS_Level", "PCA", "Energy"]:
        test_scores[method] = legacy_test[method]["score"]
        non_scores[method] = legacy_non[method]["score"]
        thresholds[method] = legacy_calibration.thresholds[method]

    primary_ends = test_bundle["TrainingGraph"].ends
    non_ends = non_bundle["TrainingGraph"].ends
    rows: list[dict] = []
    target_count = len(held_seizure.seizures)
    for event_index, (onset, _) in enumerate(all_seizures[-target_count:]):
        if preictal_overlaps_prior_seizure(onset, all_seizures, cfg):
            continue
        start = onset - float(cfg["seizure_prediction_horizon_seconds"]) - float(
            cfg["seizure_occurrence_period_seconds"]
        )
        stop = onset - float(cfg["seizure_prediction_horizon_seconds"])
        positive = (
            primary_ends * float(cfg["epoch_seconds"]) > start
        ) & (
            primary_ends * float(cfg["epoch_seconds"]) <= stop
        )
        isolated_fraction = float(np.mean(
            test_bundle["TrainingGraph"].isolated[positive]
        )) if np.any(positive) else 0.0
        excess = alignment_excess(
            test_bundle["TrainingGraph"],
            [test_bundle[name] for name in CORE_CONTROLS],
        )
        perturbed_advantage = (
            test_bundle["TrainingGraph"].concentration
            - test_bundle["PerturbedGraph"].concentration
        )
        phenotype_positive = float(
            np.any(test_scores["RecoveryPhenotype"][positive] >= thresholds["RecoveryPhenotype"])
            and isolated_fraction >= float(cfg["minimum_isolated_preictal_fraction"])
            and float(np.median(excess[positive])) > 0.0
        ) if np.any(positive) else 0.0
        for method, positive_values_all in test_scores.items():
            if method in {"NodeMax", "GSEWS_Level", "PCA", "Energy"}:
                legacy_ends = legacy_test[method]["ends"]
                legacy_positive = (
                    legacy_ends * float(cfg["epoch_seconds"]) > start
                ) & (
                    legacy_ends * float(cfg["epoch_seconds"]) <= stop
                )
                positive_values = positive_values_all[legacy_positive]
                negative_values = non_scores[method]
                alarm_ends = legacy_non[method]["ends"]
                n_positive = int(legacy_positive.sum())
            else:
                positive_values = positive_values_all[positive]
                negative_values = non_scores[method]
                alarm_ends = non_ends
                n_positive = int(positive.sum())
            if n_positive < 3 or len(negative_values) < 3:
                continue
            labels = np.r_[np.ones(len(positive_values)), np.zeros(len(negative_values))]
            combined = np.r_[positive_values, negative_values]
            threshold = thresholds[method]
            alarm_times = alarm_ends[negative_values >= threshold] * float(cfg["epoch_seconds"])
            duration_hours = len(held_non.features) * float(cfg["epoch_seconds"]) / 3600.0
            rows.append({
                "subject": subject, "fold": fold, "event_index": event_index,
                "seizure_record": held_seizure.record,
                "nonseizure_record": held_non.record,
                "method": method,
                "auc": float(roc_auc_score(labels, combined)),
                "event_predicted": float(np.any(positive_values >= threshold)),
                "false_predictions_per_hour": count_alarm_episodes(
                    alarm_times, float(cfg["alarm_refractory_seconds"])
                ) / max(duration_hours, 1e-12),
                "time_in_warning": float(np.mean(negative_values >= threshold)),
                "n_positive_windows": len(positive_values),
                "n_negative_windows": len(negative_values),
                "threshold": threshold, "n_channels": len(channels),
                "isolated_preictal_fraction": isolated_fraction,
                "median_graph_alignment_excess": float(np.median(excess[positive])),
                "median_perturbed_alignment_advantage": float(
                    np.median(perturbed_advantage[positive])
                ),
                "phenotype_positive": phenotype_positive,
            })
    return rows


def trajectory_bundle(features: np.ndarray, true_vectors: np.ndarray,
                      true_values: np.ndarray, bases: dict, cfg: dict
                      ) -> dict[str, RecoveryTrajectory]:
    true = compute_recovery_trajectory(
        features, true_vectors, true_values,
        int(cfg["window_epochs"]), int(cfg["stride_epochs"]),
        int(cfg["lag_steps"]), float(cfg["ridge_fraction"]),
        int(cfg["trailing_smooth_windows"]),
        float(cfg["isolated_mode_minimum_rho_gap"]),
        float(cfg["low_frequency_fraction"]), True,
    )
    output = {"TrainingGraph": true}
    for name, (basis, adjacency, exclude_constant) in bases.items():
        if name == "TrainingGraph":
            continue
        values = None
        if name != "IdentityBasis":
            values, _ = eigensystem(laplacian(adjacency))
        output[name] = realign_trajectory(
            true, basis, values, float(cfg["low_frequency_fraction"]),
            exclude_constant, int(cfg["trailing_smooth_windows"]),
        )
    return output


def summarize_recovery_subjects(events: pd.DataFrame) -> pd.DataFrame:
    return events.groupby(["subject", "method"], as_index=False).agg(
        n_events=("event_index", "size"), auc=("auc", "mean"),
        sensitivity=("event_predicted", "mean"),
        false_predictions_per_hour=("false_predictions_per_hour", "mean"),
        time_in_warning=("time_in_warning", "mean"),
        phenotype_prevalence=("phenotype_positive", "mean"),
        isolated_preictal_fraction=("isolated_preictal_fraction", "mean"),
        graph_alignment_excess=("median_graph_alignment_excess", "mean"),
    )


def write_report(output: Path, cfg: dict, inventory: pd.DataFrame,
                 folds: pd.DataFrame, overall: pd.DataFrame,
                 comparisons: pd.DataFrame, events: pd.DataFrame) -> None:
    phenotype = events[events.method == "RecoveryPhenotype"]
    lines = [
        "# Locked CHB-MIT graph-aligned recovery phenotype", "",
        "## Protocol", "", "```json", json.dumps(cfg, indent=2), "```", "",
        "## Inventory", "",
        inventory.groupby(["kind", "available"]).size().reset_index(name="records").to_csv(index=False).strip(), "",
        "## Fold status", "", folds.status.value_counts().to_csv().strip(), "",
        "## Overall metrics", "", overall.to_csv(index=False).strip(), "",
        "## Subject-clustered comparisons", "", comparisons.to_csv(index=False).strip(), "",
        "## Phenotype diagnostics", "",
        phenotype[["phenotype_positive", "isolated_preictal_fraction", "median_graph_alignment_excess", "median_perturbed_alignment_advantage"]].describe().to_csv().strip(), "",
    ]
    (output / "REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
