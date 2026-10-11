from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib_cache").resolve()))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from causal_gsews import ac_transform, trailing_mean
from ml_incremental import fit_grouped_elastic_net, grouped_out_of_fold_probabilities
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
    prepare_features,
    preictal_overlaps_prior_seizure,
    score_record,
    select_channels,
    subject_cluster_bootstrap,
    summarize_overall,
    summarize_subjects,
    window_raw,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/chbmit_v3_ml_protocol.json")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--subjects", nargs="*")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    config_hash = hashlib.sha256(
        json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if args.subjects:
        cfg["subjects"] = args.subjects
    root = Path(cfg["data_dir"])
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "feature_cache"
    cache.mkdir(exist_ok=True)
    manifest = json.loads((root / "locked_manifest.json").read_text())
    hashes = {row.get("config_sha256") for row in manifest}
    if hashes != {config_hash}:
        raise RuntimeError("configuration does not match the frozen v3 EEG manifest")
    manifest = [row for row in manifest if row["subject"] in cfg["subjects"]]
    available = [row for row in manifest if Path(row["local_path"]).exists()]
    missing = [row for row in manifest if not Path(row["local_path"]).exists()]
    (output / "missing_manifest.json").write_text(json.dumps(missing, indent=2))
    if not available:
        raise RuntimeError("no frozen CHB-MIT v3 EDF records are available")
    records = prepare_features(available, root, cache, cfg)
    inventory = inventory_frame(records, manifest)
    inventory.to_csv(output / "record_inventory.csv", index=False)
    if args.prepare_only:
        print(inventory.to_string(index=False))
        return
    if missing:
        raise RuntimeError(
            f"locked cohort is incomplete ({len(missing)} EDFs missing); resume the downloader"
        )

    context_lookup = build_context_lookup(manifest)
    event_eligibility_frame(records, context_lookup, cfg).to_csv(
        output / "event_eligibility.csv", index=False
    )
    by_key = {(record.subject, record.record): record for record in records}
    event_rows, fold_rows, graph_rows, ml_rows = [], [], [], []
    for subject in cfg["subjects"]:
        subject_records = [record for record in records if record.subject == subject]
        seizures = [record for record in subject_records if record.kind == "seizure" and record.seizures]
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
                record for record in outer_non if record.record != graph_validation.record
            ]
            training_seizures = [
                record for record in seizures if record.record != held_seizure.record
            ]
            fold_records = [held_seizure, held_non, graph_validation, *graph_training, *training_seizures]
            for seizure_record in [held_seizure, *training_seizures]:
                context_name = context_lookup.get((subject, seizure_record.record))
                context = by_key.get((subject, context_name)) if context_name else None
                if context is not None:
                    fold_records.append(context)
            common = sorted(set.intersection(*(set(record.channels) for record in fold_records)))
            if len(common) < int(cfg["minimum_common_channels"]):
                fold_rows.append(
                    {"subject": subject, "fold": fold, "status": "insufficient_common_channels",
                     "n_common": len(common)}
                )
                continue
            try:
                result = run_fold_v3(
                    subject, fold, held_seizure, held_non, training_seizures,
                    graph_training, graph_validation, common, context_lookup,
                    by_key, cfg,
                )
            except Exception as exc:
                fold_rows.append(
                    {"subject": subject, "fold": fold, "held_seizure": held_seizure.record,
                     "status": f"failed: {exc}"}
                )
                continue
            event_rows.extend(result["events"])
            graph_rows.extend(result["graphs"])
            ml_rows.extend(result["ml"])
            fold_rows.append(
                {"subject": subject, "fold": fold, "held_seizure": held_seizure.record,
                 "held_nonseizure": held_non.record, "status": "ok",
                 "n_common": len(common), "n_events": len(held_seizure.seizures)}
            )

    events = pd.DataFrame(event_rows)
    folds = pd.DataFrame(fold_rows)
    graphs = pd.DataFrame(graph_rows)
    ml = pd.DataFrame(ml_rows)
    folds.to_csv(output / "fold_inventory.csv", index=False)
    graphs.to_csv(output / "graph_validation.csv", index=False)
    ml.to_csv(output / "ml_hyperparameters.csv", index=False)
    if events.empty:
        raise RuntimeError("no complete v3 EEG folds produced event metrics")
    events.to_csv(output / "event_metrics.csv", index=False)
    subjects = summarize_subjects(events)
    subjects.to_csv(output / "subject_metrics.csv", index=False)
    overall = summarize_overall(subjects)
    overall.to_csv(output / "overall_metrics.csv", index=False)
    comparisons = subject_cluster_bootstrap(subjects, cfg)
    comparisons.insert(0, "comparison_family", "mechanistic_primary")
    if {"ML_GraphIncremental", "ML_NoGraph"}.issubset(set(subjects.method)):
        ml_cfg = dict(cfg)
        ml_cfg["primary_method"] = "ML_GraphIncremental"
        ml_comparisons = subject_cluster_bootstrap(subjects, ml_cfg)
        ml_comparisons = ml_comparisons[
            ml_comparisons.comparator == "ML_NoGraph"
        ].copy()
        ml_comparisons.insert(0, "comparison_family", "incremental_ml")
        comparisons = pd.concat([comparisons, ml_comparisons], ignore_index=True)
    comparisons.to_csv(output / "subject_clustered_comparisons.csv", index=False)
    write_report(output, cfg, inventory, folds, graphs, overall, comparisons)
    print(overall.to_string(index=False))


def run_fold_v3(subject: str, fold: int, held_seizure: RecordFeatures,
                held_non: RecordFeatures, training_seizures: list[RecordFeatures],
                graph_training: list[RecordFeatures], graph_validation: RecordFeatures,
                channels: list[str], context_lookup: dict[tuple[str, str], str],
                by_key: dict[tuple[str, str], RecordFeatures], cfg: dict) -> dict:
    graph_train_arrays = [select_channels(record, channels) for record in graph_training]
    if sum(len(array) for array in graph_train_arrays) < int(cfg["graph_fit_minimum_epochs"]):
        raise RuntimeError("insufficient graph-fitting epochs")
    if len(graph_validation.features) < int(cfg["graph_validation_minimum_epochs"]):
        raise RuntimeError("insufficient graph-validation epochs")
    median, scale = fit_robust_scaler(graph_train_arrays)
    graph_train_scaled = [apply_scaler(array, median, scale) for array in graph_train_arrays]
    validation_scaled = apply_scaler(select_channels(graph_validation, channels), median, scale)
    graph = estimate_training_graph(graph_train_scaled, int(cfg["graph_top_k"]))
    from graphs import eigensystem, laplacian
    _, training_u = eigensystem(laplacian(graph))
    rng = np.random.default_rng(int(cfg["seed"]) + fold)
    bases = [
        ("TrainingGraph" if name == "TrueGraph" else name, basis, adjacency, exclude)
        for name, basis, adjacency, exclude in graph_bases(graph, training_u, rng)
    ]
    from rigorous_simulate import perturb_adjacency
    perturbed = perturb_adjacency(
        graph, float(cfg["graph_perturbation_strength"]), rng
    )
    _, perturbed_u = eigensystem(laplacian(perturbed))
    bases.append(("PerturbedGraph", perturbed_u, perturbed, True))

    graph_fits = []
    for name, basis, _, exclude_constant in bases:
        fitted = fit_multi_record_dynamics(
            graph_train_scaled, validation_scaled, basis,
            float(cfg["graph_fit_ridge_fraction"]), exclude_constant,
        )
        graph_fits.append((name, fitted))
    training_mse = next(
        row[1]["validation_mse"] for row in graph_fits if row[0] == "TrainingGraph"
    )
    best_control = min(
        row[1]["validation_mse"] for row in graph_fits if row[0] != "TrainingGraph"
    )
    graph_rows = [
        {"subject": subject, "fold": fold, "basis": name,
         "validation_mse": fitted["validation_mse"],
         "estimated_max_rho": fitted["estimated_max_rho"],
         "graph_selected": float(training_mse < best_control),
         "training_graph_mse_advantage_over_best_control": best_control - training_mse}
        for name, fitted in graph_fits
    ]

    training_raw = {
        name: [window_raw(array, basis, adjacency, cfg) for array in graph_train_scaled]
        for name, basis, adjacency, _ in bases
    }
    calibrations = {
        name: fit_calibration(
            raw_records, cfg,
            exclude_constant=next(flag for basis_name, _, _, flag in bases if basis_name == name),
        )
        for name, raw_records in training_raw.items()
    }
    training_scores = {
        name: [score_record_v3(raw, calibrations[name], cfg) for raw in raw_records]
        for name, raw_records in training_raw.items()
    }
    thresholds = {
        name: fit_score_thresholds(score_records, cfg)
        for name, score_records in training_scores.items()
    }

    test_array, context_seconds, test_seizures = seizure_sequence(
        held_seizure, channels, context_lookup, by_key, cfg
    )
    test_scaled = apply_scaler(test_array, median, scale)
    held_non_scaled = apply_scaler(select_channels(held_non, channels), median, scale)
    test_scores, non_scores = {}, {}
    for name, basis, adjacency, _ in bases:
        seizure_scored = score_record_v3(
            window_raw(test_scaled, basis, adjacency, cfg), calibrations[name], cfg
        )
        non_scored = score_record_v3(
            window_raw(held_non_scaled, basis, adjacency, cfg), calibrations[name], cfg
        )
        if name == "TrainingGraph":
            test_scores.update(seizure_scored)
            non_scores.update(non_scored)
        else:
            for method in ["GraphDyn_Raw", "GSEWS_Level"]:
                test_scores[f"{name}:{method}"] = seizure_scored[method]
                non_scores[f"{name}:{method}"] = non_scored[method]

    events = evaluate_score_events(
        subject, fold, held_seizure, held_non, test_seizures,
        test_scores, non_scores, thresholds, cfg, channels,
    )
    ml_events, ml_rows = fit_and_evaluate_ml(
        subject, fold, held_seizure, held_non, training_seizures,
        graph_training + [graph_validation], channels, context_lookup, by_key,
        median, scale, training_u, graph, calibrations["TrainingGraph"], cfg,
        test_scores, non_scores, test_seizures,
    )
    events.extend(ml_events)
    return {"events": events, "graphs": graph_rows, "ml": ml_rows}


def fit_multi_record_dynamics(train_arrays: list[np.ndarray], validation: np.ndarray,
                              basis: np.ndarray, ridge_fraction: float,
                              exclude_constant: bool) -> dict:
    numerator = np.zeros(basis.shape[1])
    denominator = np.zeros(basis.shape[1])
    count = 0
    for array in train_arrays:
        modal = array @ basis
        finite = np.all(np.isfinite(modal[:-1]), axis=1) & np.all(
            np.isfinite(modal[1:]), axis=1
        )
        numerator += np.sum(modal[:-1][finite] * modal[1:][finite], axis=0)
        denominator += np.sum(modal[:-1][finite] ** 2, axis=0)
        count += int(finite.sum())
    if count == 0:
        raise RuntimeError("no finite adjacent graph-training epochs")
    ridge = ridge_fraction * max(float(denominator.mean() / max(count, 1)), 1e-12) * count
    rho = numerator / (denominator + ridge)
    modal_validation = validation @ basis
    finite = np.all(np.isfinite(modal_validation[:-1]), axis=1) & np.all(
        np.isfinite(modal_validation[1:]), axis=1
    )
    if not np.any(finite):
        raise RuntimeError("no finite adjacent graph-validation epochs")
    residual = (
        modal_validation[1:][finite]
        - modal_validation[:-1][finite] * rho[None, :]
    )
    modes = np.arange(len(rho))
    if exclude_constant and len(modes) > 1:
        modes = modes[1:]
    return {
        "validation_mse": float(np.mean(residual**2)),
        "estimated_max_rho": float(np.max(rho[modes])),
    }


def score_record_v3(raw: dict, calibration, cfg: dict) -> dict:
    scores = score_record(raw, calibration, cfg)
    modes = np.arange(raw["modal_ac"].shape[1])
    if calibration.exclude_constant and len(modes) > 1:
        modes = modes[1:]
    rho = np.clip(raw["modal_ac"][:, modes], -0.999, 0.999)
    persistence = ac_transform(rho)
    ordered = np.sort(rho, axis=1)
    modal_gap = ordered[:, -1] - ordered[:, -2] if rho.shape[1] > 1 else np.zeros(len(rho))
    span = int(cfg["trailing_smooth_windows"])
    scores["GraphDyn_Raw"] = {
        "score": trailing_mean(np.max(persistence, axis=1), span),
        "ends": raw["ends"],
    }
    scores["ModalGap"] = {
        "score": trailing_mean(modal_gap, span), "ends": raw["ends"]
    }
    return scores


def fit_score_thresholds(records: list[dict], cfg: dict) -> dict[str, float]:
    methods = records[0].keys()
    return {
        method: float(
            np.quantile(
                np.concatenate([record[method]["score"] for record in records]),
                float(cfg["alarm_training_quantile"]),
            )
        )
        for method in methods
    }


def seizure_sequence(record: RecordFeatures, channels: list[str],
                     context_lookup: dict[tuple[str, str], str],
                     by_key: dict[tuple[str, str], RecordFeatures], cfg: dict
                     ) -> tuple[np.ndarray, float, list[tuple[float, float]]]:
    array = select_channels(record, channels)
    context_name = context_lookup.get((record.subject, record.record))
    context = by_key.get((record.subject, context_name)) if context_name else None
    context_seconds = 0.0
    context_seizures: list[tuple[float, float]] = []
    if context is not None:
        context_array = select_channels(context, channels)
        context_seconds = len(context_array) * float(cfg["epoch_seconds"])
        array = np.vstack([context_array, array])
        context_seizures = list(context.seizures)
    shifted = [
        (start + context_seconds, stop + context_seconds)
        for start, stop in record.seizures
    ]
    return array, context_seconds, context_seizures + shifted


def evaluate_score_events(subject: str, fold: int, held_seizure: RecordFeatures,
                          held_non: RecordFeatures, all_seizures: list[tuple[float, float]],
                          test_scores: dict, non_scores: dict, thresholds: dict,
                          cfg: dict, channels: list[str]) -> list[dict]:
    rows = []
    target_count = len(held_seizure.seizures)
    targets = all_seizures[-target_count:] if target_count else []
    for event_index, (onset, _) in enumerate(targets):
        if preictal_overlaps_prior_seizure(onset, all_seizures, cfg):
            continue
        for method, values in test_scores.items():
            basis, base_method = (
                method.split(":", 1)
                if ":" in method else ("TrainingGraph", method)
            )
            threshold = thresholds[basis][base_method]
            ends = values["ends"] * float(cfg["epoch_seconds"])
            start = onset - float(cfg["seizure_prediction_horizon_seconds"]) - float(cfg["seizure_occurrence_period_seconds"])
            stop = onset - float(cfg["seizure_prediction_horizon_seconds"])
            positive = (ends > start) & (ends <= stop)
            negative = non_scores[method]["score"]
            if positive.sum() < 3 or len(negative) < 3:
                continue
            combined = np.r_[values["score"][positive], negative]
            labels = np.r_[np.ones(positive.sum()), np.zeros(len(negative))]
            alarm_times = non_scores[method]["ends"][negative >= threshold] * float(cfg["epoch_seconds"])
            duration = len(held_non.features) * float(cfg["epoch_seconds"]) / 3600.0
            rows.append(
                {"subject": subject, "fold": fold, "event_index": event_index,
                 "seizure_record": held_seizure.record, "nonseizure_record": held_non.record,
                 "method": method, "auc": float(roc_auc_score(labels, combined)),
                 "event_predicted": float(np.any(values["score"][positive] >= threshold)),
                 "false_predictions_per_hour": count_alarm_episodes(
                     alarm_times, float(cfg["seizure_occurrence_period_seconds"])
                 ) / max(duration, 1e-12),
                 "time_in_warning": float(np.mean(negative >= threshold)),
                 "n_positive_windows": int(positive.sum()),
                 "n_negative_windows": int(len(negative)), "threshold": threshold,
                 "n_channels": len(channels),
                 "isolated_window_fraction": float(
                     np.mean(test_scores["ModalGap"]["score"][positive]
                             >= float(cfg["isolated_mode_minimum_rho_gap"]))
                 ) if method == "GraphDyn_Raw" else np.nan}
            )
    return rows


def feature_matrix(scores: dict, names: list[str]) -> np.ndarray:
    return np.column_stack([scores[name]["score"] for name in names])


def fit_and_evaluate_ml(subject: str, fold: int, held_seizure: RecordFeatures,
                        held_non: RecordFeatures, training_seizures: list[RecordFeatures],
                        training_non: list[RecordFeatures], channels: list[str],
                        context_lookup: dict[tuple[str, str], str],
                        by_key: dict[tuple[str, str], RecordFeatures], median: np.ndarray,
                        scale: np.ndarray, basis: np.ndarray, graph: np.ndarray,
                        calibration, cfg: dict, test_scores: dict, non_scores: dict,
                        test_seizures: list[tuple[float, float]]) -> tuple[list[dict], list[dict]]:
    score_cache = {}
    for record in training_seizures:
        array, _, seizures = seizure_sequence(record, channels, context_lookup, by_key, cfg)
        score_cache[("seizure", record.record)] = (
            score_record_v3(window_raw(apply_scaler(array, median, scale), basis, graph, cfg), calibration, cfg),
            seizures,
        )
    for record in training_non:
        array = apply_scaler(select_channels(record, channels), median, scale)
        score_cache[("non", record.record)] = (
            score_record_v3(window_raw(array, basis, graph, cfg), calibration, cfg), []
        )
    ml_events, ml_rows = [], []
    target_count = len(held_seizure.seizures)
    held_targets = test_seizures[-target_count:] if target_count else []
    for model_name, feature_names in cfg["ml_models"].items():
        x_parts, y_parts, group_parts = [], [], []
        for record in training_seizures:
            scores, seizures = score_cache[("seizure", record.record)]
            ends = scores[feature_names[0]]["ends"] * float(cfg["epoch_seconds"])
            for onset, _ in seizures[-len(record.seizures):]:
                if preictal_overlaps_prior_seizure(onset, seizures, cfg):
                    continue
                start = onset - float(cfg["seizure_prediction_horizon_seconds"]) - float(cfg["seizure_occurrence_period_seconds"])
                stop = onset - float(cfg["seizure_prediction_horizon_seconds"])
                mask = (ends > start) & (ends <= stop)
                if mask.sum() >= 3:
                    x_parts.append(feature_matrix(scores, feature_names)[mask])
                    y_parts.append(np.ones(mask.sum(), dtype=int))
                    group_parts.append(np.full(mask.sum(), f"seizure:{record.record}"))
        for record in training_non:
            scores, _ = score_cache[("non", record.record)]
            values = feature_matrix(scores, feature_names)
            x_parts.append(values); y_parts.append(np.zeros(len(values), dtype=int))
            group_parts.append(np.full(len(values), f"non:{record.record}"))
        if not x_parts:
            continue
        x_train = np.vstack(x_parts); y_train = np.concatenate(y_parts); groups = np.concatenate(group_parts)
        model = fit_grouped_elastic_net(
            x_train, y_train, groups, cfg["ml_hyperparameters"],
            int(cfg["seed"]) + fold,
        )
        oof_probability = grouped_out_of_fold_probabilities(
            x_train, y_train, groups, model.C, model.l1_ratio,
            int(cfg["seed"]) + 200_000 + fold,
        )
        negative_training = (y_train == 0) & np.isfinite(oof_probability)
        if negative_training.sum() < 3:
            raise RuntimeError("insufficient record-held-out negative probabilities")
        threshold = float(np.quantile(
            oof_probability[negative_training],
            float(cfg["alarm_training_quantile"]),
        ))
        held_positive_matrix = feature_matrix(test_scores, feature_names)
        held_negative_matrix = feature_matrix(non_scores, feature_names)
        positive_probability = model.predict_proba(held_positive_matrix)
        negative_probability = model.predict_proba(held_negative_matrix)
        ends = test_scores[feature_names[0]]["ends"] * float(cfg["epoch_seconds"])
        for event_index, (onset, _) in enumerate(held_targets):
            if preictal_overlaps_prior_seizure(onset, test_seizures, cfg):
                continue
            start = onset - float(cfg["seizure_prediction_horizon_seconds"]) - float(cfg["seizure_occurrence_period_seconds"])
            stop = onset - float(cfg["seizure_prediction_horizon_seconds"])
            positive = (ends > start) & (ends <= stop)
            if positive.sum() < 3:
                continue
            combined = np.r_[positive_probability[positive], negative_probability]
            labels = np.r_[np.ones(positive.sum()), np.zeros(len(negative_probability))]
            alarm_times = non_scores[feature_names[0]]["ends"][negative_probability >= threshold] * float(cfg["epoch_seconds"])
            duration = len(held_non.features) * float(cfg["epoch_seconds"]) / 3600.0
            ml_events.append(
                {"subject": subject, "fold": fold, "event_index": event_index,
                 "seizure_record": held_seizure.record, "nonseizure_record": held_non.record,
                 "method": model_name, "auc": float(roc_auc_score(labels, combined)),
                 "event_predicted": float(np.any(positive_probability[positive] >= threshold)),
                 "false_predictions_per_hour": count_alarm_episodes(
                     alarm_times, float(cfg["seizure_occurrence_period_seconds"])
                 ) / max(duration, 1e-12),
                 "time_in_warning": float(np.mean(negative_probability >= threshold)),
                 "n_positive_windows": int(positive.sum()),
                 "n_negative_windows": int(len(negative_probability)),
                 "threshold": threshold, "n_channels": len(channels),
                 "isolated_window_fraction": np.nan}
            )
        ml_rows.append(
            {"subject": subject, "fold": fold, "model": model_name,
             "C": model.C, "l1_ratio": model.l1_ratio,
             "inner_auc": model.inner_auc, "n_training_windows": len(y_train),
             "n_positive_groups": len(set(groups[y_train == 1])),
             "n_negative_groups": len(set(groups[y_train == 0]))}
        )
    return ml_events, ml_rows


def write_report(output: Path, cfg: dict, inventory: pd.DataFrame, folds: pd.DataFrame,
                 graphs: pd.DataFrame, overall: pd.DataFrame,
                 comparisons: pd.DataFrame) -> None:
    lines = [
        "# Locked CHB-MIT version-3 graph/ML evaluation", "",
        "## Protocol", "", "```json", json.dumps(cfg, indent=2), "```", "",
        "## Inventory", "", inventory.groupby(["kind", "available"]).size().reset_index(name="records").to_csv(index=False).strip(), "",
        "## Fold status", "", folds.status.value_counts().to_csv().strip(), "",
        "## Graph validation", "", graphs.groupby("basis").agg(graph_selection=("graph_selected", "mean"), validation_mse=("validation_mse", "mean")).to_csv().strip(), "",
        "## Overall metrics", "", overall.to_csv(index=False).strip(), "",
        "## Subject-clustered comparisons", "", comparisons.to_csv(index=False).strip(), "",
    ]
    (output / "REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
