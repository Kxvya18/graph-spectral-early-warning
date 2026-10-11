from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib_cache").resolve()))

import numpy as np
import pandas as pd
import pyedflib
from scipy.signal import butter, iirnotch, lfilter, sosfilt
from sklearn.metrics import roc_auc_score

from causal_gsews import ac_transform, causal_slope_matrix, detrend_window, edge_correlation, graph_smoothness, lag1, trailing_mean
from graphs import eigensystem, laplacian
from rigorous_evaluate import paired_bootstrap


@dataclass
class RecordFeatures:
    subject: str
    record: str
    kind: str
    channels: list[str]
    features: np.ndarray
    epoch_seconds: float
    seizures: list[tuple[float, float]]


@dataclass
class Calibration:
    mean: dict[str, np.ndarray]
    std: dict[str, np.ndarray]
    thresholds: dict[str, float]
    exclude_constant: bool


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/chbmit_locked_protocol.json")
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
    manifest_hashes = {row.get("config_sha256") for row in manifest}
    if manifest_hashes != {config_hash}:
        raise RuntimeError(
            "configuration does not match the frozen manifest; regenerate only "
            "under a new protocol version if a scientific change is intended"
        )
    manifest = [row for row in manifest if row["subject"] in cfg["subjects"]]
    context_lookup = build_context_lookup(manifest)
    available = [row for row in manifest if Path(row["local_path"]).exists()]
    missing = [row for row in manifest if not Path(row["local_path"]).exists()]
    (output / "missing_manifest.json").write_text(json.dumps(missing, indent=2))
    if not available:
        raise RuntimeError("no locked CHB-MIT EDF records are available")

    records = prepare_features(available, root, cache, cfg)
    inventory = inventory_frame(records, manifest)
    inventory.to_csv(output / "record_inventory.csv", index=False)
    event_eligibility_frame(records, context_lookup, cfg).to_csv(
        output / "event_eligibility.csv", index=False
    )
    if args.prepare_only:
        print(inventory.to_string(index=False))
        return

    event_rows: list[dict] = []
    fold_rows: list[dict] = []
    for subject in cfg["subjects"]:
        subject_records = [r for r in records if r.subject == subject]
        seizure_records = [r for r in subject_records if r.kind == "seizure" and r.seizures]
        nonseizure_records = [r for r in subject_records if r.kind == "nonseizure"]
        if len(seizure_records) < int(cfg["minimum_seizure_records_for_patient_specific_cv"]):
            continue
        if len(nonseizure_records) < 2:
            continue
        for fold, held_seizure in enumerate(seizure_records):
            held_non = nonseizure_records[fold % len(nonseizure_records)]
            training_non = [r for r in nonseizure_records if r.record != held_non.record]
            context_record_name = context_lookup.get((subject, held_seizure.record))
            held_context = next(
                (r for r in subject_records if r.record == context_record_name), None
            )
            fold_records = [held_seizure, held_non, *training_non]
            if held_context is not None:
                fold_records.append(held_context)
            common = sorted(set.intersection(*(set(r.channels) for r in fold_records)))
            if len(common) < int(cfg["minimum_common_channels"]):
                fold_rows.append(
                    {
                        "subject": subject, "fold": fold,
                        "held_seizure": held_seizure.record,
                        "held_nonseizure": held_non.record,
                        "context_record": context_record_name or "",
                        "status": "insufficient_common_channels", "n_common": len(common),
                    }
                )
                continue
            try:
                result = run_fold(
                    subject, fold, held_seizure, held_context, held_non,
                    training_non, common, cfg
                )
            except Exception as exc:
                fold_rows.append(
                    {"subject": subject, "fold": fold, "held_seizure": held_seizure.record,
                     "held_nonseizure": held_non.record,
                     "context_record": context_record_name or "",
                     "status": f"failed: {exc}"}
                )
                continue
            event_rows.extend(result)
            fold_rows.append(
                {"subject": subject, "fold": fold, "held_seizure": held_seizure.record,
                 "held_nonseizure": held_non.record,
                 "context_record": context_record_name or "", "status": "ok",
                 "n_common": len(common),
                 "n_training_nonseizure": len(training_non), "n_events": len(held_seizure.seizures)}
            )

    events = pd.DataFrame(event_rows)
    folds = pd.DataFrame(fold_rows)
    folds.to_csv(output / "fold_inventory.csv", index=False)
    if events.empty:
        raise RuntimeError("no eligible patient-specific folds completed")
    events.to_csv(output / "event_metrics.csv", index=False)
    subject_summary = summarize_subjects(events)
    subject_summary.to_csv(output / "subject_metrics.csv", index=False)
    overall = summarize_overall(subject_summary)
    overall.to_csv(output / "overall_metrics.csv", index=False)
    clustered = subject_cluster_bootstrap(subject_summary, cfg)
    clustered.to_csv(output / "subject_clustered_comparisons.csv", index=False)
    write_report(output, cfg, inventory, folds, overall, clustered)
    print(overall.to_string(index=False))
    print(f"\nWrote locked CHB-MIT results to {output}")


def prepare_features(rows: list[dict], root: Path, cache: Path, cfg: dict) -> list[RecordFeatures]:
    summaries = {
        subject: parse_summary(root / subject / f"{subject}-summary.txt")
        for subject in sorted({row["subject"] for row in rows})
    }
    prepared = []
    for index, row in enumerate(rows, start=1):
        cache_path = cache / f"{row['subject']}__{row['record']}.npz"
        if cache_path.exists():
            data = np.load(cache_path, allow_pickle=True)
            channels = [str(x) for x in data["channels"]]
            features = data["features"]
        else:
            channels, features = extract_record_features(Path(row["local_path"]), cfg)
            np.savez_compressed(cache_path, channels=np.asarray(channels), features=features)
        prepared.append(
            RecordFeatures(
                subject=row["subject"], record=row["record"], kind=row["kind"],
                channels=channels, features=features,
                epoch_seconds=float(cfg["epoch_seconds"]),
                seizures=summaries[row["subject"]].get(row["record"], []),
            )
        )
        if index % 10 == 0:
            print(f"prepared {index}/{len(rows)} records")
    return prepared


def extract_record_features(path: Path, cfg: dict) -> tuple[list[str], np.ndarray]:
    reader = pyedflib.EdfReader(str(path))
    try:
        labels = reader.getSignalLabels()
        selected: list[tuple[int, str]] = []
        seen = set()
        for index, raw in enumerate(labels):
            label = canonical_channel(raw)
            if not label or label in seen:
                continue
            fs = float(reader.getSampleFrequency(index))
            if fs <= 0:
                continue
            selected.append((index, label))
            seen.add(label)
        if len(selected) < int(cfg["minimum_common_channels"]):
            raise RuntimeError(f"only {len(selected)} usable channels")
        fs = float(reader.getSampleFrequency(selected[0][0]))
        selected = [(i, label) for i, label in selected if abs(reader.getSampleFrequency(i) - fs) < 1e-6]
        signals = np.vstack([reader.readSignal(i) for i, _ in selected]).T
        channels = [label for _, label in selected]
    finally:
        reader.close()
    filtered = causal_filter(signals, fs, cfg)
    epoch = int(round(fs * float(cfg["epoch_seconds"])))
    n_epochs = len(filtered) // epoch
    shaped = filtered[: n_epochs * epoch].reshape(n_epochs, epoch, filtered.shape[1])
    line_length = np.mean(np.abs(np.diff(shaped, axis=1)), axis=1)
    features = np.log1p(line_length)
    # Discard the causal filter transient without consulting future data.
    transient_epochs = max(1, int(round(10.0 / float(cfg["epoch_seconds"]))))
    features[:transient_epochs] = np.nan
    return channels, features


def causal_filter(signals: np.ndarray, fs: float, cfg: dict) -> np.ndarray:
    low, high = [float(x) for x in cfg["bandpass_hz"]]
    nyquist = fs / 2.0
    sos = butter(4, [low / nyquist, min(high / nyquist, 0.99)], btype="bandpass", output="sos")
    filtered = sosfilt(sos, signals, axis=0)
    notch = float(cfg["notch_hz"])
    if 0 < notch < nyquist:
        b, a = iirnotch(notch, 30.0, fs=fs)
        filtered = lfilter(b, a, filtered, axis=0)
    return filtered


def run_fold(subject: str, fold: int, held_seizure: RecordFeatures,
             held_context: RecordFeatures | None, held_non: RecordFeatures,
             training_non: list[RecordFeatures], channels: list[str], cfg: dict) -> list[dict]:
    train_arrays = [select_channels(record, channels) for record in training_non]
    median, scale = fit_robust_scaler(train_arrays)
    train_scaled = [apply_scaler(a, median, scale) for a in train_arrays]
    graph = estimate_training_graph(train_scaled, int(cfg["graph_top_k"]))
    evals, true_u = eigensystem(laplacian(graph))
    rng = np.random.default_rng(int(cfg["seed"]) + fold)
    bases = list(graph_bases(graph, true_u, rng))

    training_raw = {
        name: [window_raw(a, u, adjacency, cfg) for a in train_scaled]
        for name, u, adjacency, _ in bases
    }
    calibrations = {
        name: fit_calibration(
            raw_records, cfg,
            exclude_constant=next(flag for basis, _, _, flag in bases if basis == name),
        )
        for name, raw_records in training_raw.items()
    }
    held_seizure_features = select_channels(held_seizure, channels)
    context_seconds = 0.0
    if held_context is not None:
        context_features = select_channels(held_context, channels)
        context_seconds = len(context_features) * float(cfg["epoch_seconds"])
        held_seizure_features = np.vstack([context_features, held_seizure_features])
    held_seizure_scaled = apply_scaler(held_seizure_features, median, scale)
    held_non_scaled = apply_scaler(select_channels(held_non, channels), median, scale)
    test_scores = {}
    non_scores = {}
    for name, u, adjacency, _ in bases:
        test_raw = window_raw(held_seizure_scaled, u, adjacency, cfg)
        non_raw = window_raw(held_non_scaled, u, adjacency, cfg)
        test_scores.update(prefix_scores(score_record(test_raw, calibrations[name], cfg), name))
        non_scores.update(prefix_scores(score_record(non_raw, calibrations[name], cfg), name))

    primary_cal = calibrations["TrueGraph"]
    rows = []
    shifted_context_seizures = []
    if held_context is not None:
        shifted_context_seizures = list(held_context.seizures)
    shifted_record_seizures = [
        (start + context_seconds, stop + context_seconds)
        for start, stop in held_seizure.seizures
    ]
    all_test_seizures = shifted_context_seizures + shifted_record_seizures
    for event_index, (record_onset, _) in enumerate(held_seizure.seizures):
        onset = record_onset + context_seconds
        if preictal_overlaps_prior_seizure(onset, all_test_seizures, cfg):
            continue
        for method, values in test_scores.items():
            base_method = method.split(":", 1)[-1]
            threshold = primary_cal.thresholds.get(base_method, np.nan)
            if method.startswith("IdentityBasis:"):
                threshold = calibrations["IdentityBasis"].thresholds.get(base_method, np.nan)
            elif method.startswith("RandomGraph:"):
                threshold = calibrations["RandomGraph"].thresholds.get(base_method, np.nan)
            elif method.startswith("PermutedGraph:"):
                threshold = calibrations["PermutedGraph"].thresholds.get(base_method, np.nan)
            ends_sec = values["ends"] * float(cfg["epoch_seconds"])
            sop_start = onset - float(cfg["seizure_prediction_horizon_seconds"]) - float(cfg["seizure_occurrence_period_seconds"])
            sop_end = onset - float(cfg["seizure_prediction_horizon_seconds"])
            positive = (ends_sec > sop_start) & (ends_sec <= sop_end)
            negative_values = non_scores[method]["score"]
            if positive.sum() < 3 or len(negative_values) < 3:
                continue
            labels = np.r_[np.ones(positive.sum(), dtype=int), np.zeros(len(negative_values), dtype=int)]
            combined = np.r_[values["score"][positive], negative_values]
            auc = float(roc_auc_score(labels, combined))
            predicted = bool(np.any(values["score"][positive] >= threshold))
            non_duration_hours = len(held_non.features) * float(cfg["epoch_seconds"]) / 3600.0
            alarm_times = non_scores[method]["ends"][negative_values >= threshold] * float(cfg["epoch_seconds"])
            false_alarms = count_alarm_episodes(
                alarm_times, float(cfg.get("alarm_refractory_seconds", cfg["seizure_occurrence_period_seconds"]))
            )
            rows.append(
                {
                    "subject": subject, "fold": fold, "event_index": event_index,
                    "seizure_record": held_seizure.record, "nonseizure_record": held_non.record,
                    "context_record": held_context.record if held_context is not None else "",
                    "record_onset_seconds": record_onset,
                    "method": method, "auc": auc, "event_predicted": float(predicted),
                    "false_predictions_per_hour": false_alarms / max(non_duration_hours, 1e-12),
                    "time_in_warning": float(np.mean(negative_values >= threshold)),
                    "n_positive_windows": int(positive.sum()),
                    "n_negative_windows": int(len(negative_values)),
                    "threshold": float(threshold), "n_channels": len(channels),
                }
            )
    return rows


def window_raw(features: np.ndarray, u: np.ndarray, adjacency: np.ndarray, cfg: dict) -> dict:
    window = int(cfg["window_epochs"])
    stride = int(cfg["stride_epochs"])
    starts = np.arange(0, len(features) - window + 1, stride, dtype=int)
    ends = starts + window
    modal_var, modal_ac, node_var, node_ac = [], [], [], []
    pca_var, pca_ac, energy, smooth, edgecorr = [], [], [], [], []
    edges = np.argwhere(np.triu(adjacency, 1) > 0)
    keep_starts, keep_ends = [], []
    for start, end in zip(starts, ends):
        block = features[start:end]
        if not np.all(np.isfinite(block)):
            continue
        wx = detrend_window(block)
        wy = wx @ u
        modal_var.append(np.var(wy, axis=0, ddof=1)); modal_ac.append([lag1(wy[:, k]) for k in range(wy.shape[1])])
        node_var.append(np.var(wx, axis=0, ddof=1)); node_ac.append([lag1(wx[:, k]) for k in range(wx.shape[1])])
        _, _, vh = np.linalg.svd(wx, full_matrices=False); pc = wx @ vh[0]
        pca_var.append(np.var(pc, ddof=1)); pca_ac.append(lag1(pc)); energy.append(np.mean(wx**2))
        smooth.append(graph_smoothness(wx, adjacency)); edgecorr.append(edge_correlation(wx, edges))
        keep_starts.append(start); keep_ends.append(end)
    return {
        "starts": np.asarray(keep_starts), "ends": np.asarray(keep_ends),
        "modal_var": np.asarray(modal_var), "modal_ac": np.asarray(modal_ac),
        "node_var": np.asarray(node_var), "node_ac": np.asarray(node_ac),
        "pca_var": np.asarray(pca_var), "pca_ac": np.asarray(pca_ac),
        "energy": np.asarray(energy), "smooth": np.asarray(smooth), "edgecorr": np.asarray(edgecorr),
    }


def raw_components(raw: dict, cfg: dict) -> dict[str, np.ndarray]:
    span = int(cfg["trend_span_windows"])
    modal_level = np.log1p(raw["modal_var"]) + ac_transform(raw["modal_ac"])
    modal_trend = causal_slope_matrix(np.log1p(raw["modal_var"]), span) + causal_slope_matrix(ac_transform(raw["modal_ac"]), span)
    node_level = np.log1p(raw["node_var"]) + ac_transform(raw["node_ac"])
    node_trend = causal_slope_matrix(np.log1p(raw["node_var"]), span) + causal_slope_matrix(ac_transform(raw["node_ac"]), span)
    pca_level = np.log1p(raw["pca_var"]) + ac_transform(raw["pca_ac"])
    return {
        "modal_level": modal_level, "modal_trend": modal_trend,
        "node_level": node_level, "node_trend": node_trend,
        "pca_level": pca_level[:, None], "energy": np.log1p(raw["energy"])[:, None],
        "smooth": raw["smooth"][:, None], "edgecorr": raw["edgecorr"][:, None],
    }


def fit_calibration(raw_records: list[dict], cfg: dict, exclude_constant: bool) -> Calibration:
    components_by_record = [raw_components(raw, cfg) for raw in raw_records]
    names = components_by_record[0].keys()
    mean, std = {}, {}
    for name in names:
        values = np.concatenate([record[name] for record in components_by_record], axis=0)
        mean[name] = values.mean(axis=0, keepdims=True)
        sd = values.std(axis=0, ddof=1, keepdims=True)
        std[name] = np.where(sd < 1e-8, 1.0, sd)
    training_scores = []
    for raw in raw_records:
        training_scores.append(score_with_parameters(raw, mean, std, cfg, exclude_constant))
    thresholds = {}
    for method in training_scores[0]:
        values = np.concatenate([record[method]["score"] for record in training_scores])
        thresholds[method] = float(np.quantile(values, float(cfg["alarm_training_quantile"])))
    return Calibration(mean=mean, std=std, thresholds=thresholds, exclude_constant=exclude_constant)


def score_record(raw: dict, calibration: Calibration, cfg: dict) -> dict:
    return score_with_parameters(
        raw, calibration.mean, calibration.std, cfg, calibration.exclude_constant
    )


def score_with_parameters(raw: dict, mean: dict, std: dict, cfg: dict,
                          exclude_constant: bool) -> dict:
    c = raw_components(raw, cfg)
    z = {name: (values - mean[name]) / std[name] for name, values in c.items()}
    span = int(cfg["trailing_smooth_windows"])
    first_mode = 1 if exclude_constant else 0
    scores = {
        "GSEWS_Level": np.max(z["modal_level"][:, first_mode:], axis=1),
        "GSEWS": np.max(z["modal_trend"][:, first_mode:], axis=1),
        "NodeAvg": np.mean(z["node_trend"], axis=1),
        "NodeMax": np.max(z["node_level"], axis=1),
        "PCA": z["pca_level"][:, 0], "Energy": z["energy"][:, 0],
        "GraphSmooth": z["smooth"][:, 0], "EdgeCorr": z["edgecorr"][:, 0],
    }
    return {name: {"score": trailing_mean(values, span), "ends": raw["ends"]} for name, values in scores.items()}


def graph_bases(graph: np.ndarray, true_u: np.ndarray, rng: np.random.Generator):
    n = len(graph)
    yield "TrueGraph", true_u, graph, True
    identity = np.eye(n)
    yield "IdentityBasis", identity, np.zeros_like(graph), False
    density = int(np.triu(graph > 0, 1).sum())
    pairs = np.asarray([(i, j) for i in range(n) for j in range(i + 1, n)])
    chosen = rng.choice(len(pairs), density, replace=False)
    random_graph = np.zeros_like(graph)
    for i, j in pairs[chosen]: random_graph[i, j] = random_graph[j, i] = 1.0
    _, random_u = eigensystem(laplacian(random_graph))
    yield "RandomGraph", random_u, random_graph, True
    permutation = rng.permutation(n)
    permuted = graph[np.ix_(permutation, permutation)]
    _, permuted_u = eigensystem(laplacian(permuted))
    yield "PermutedGraph", permuted_u, permuted, True


def prefix_scores(scores: dict, basis: str) -> dict:
    if basis == "TrueGraph": return scores
    return {f"{basis}:{method}": value for method, value in scores.items() if method.startswith("GSEWS")}


def fit_robust_scaler(arrays: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    values = np.concatenate(arrays, axis=0)
    values = values[np.all(np.isfinite(values), axis=1)]
    median = np.median(values, axis=0, keepdims=True)
    mad = np.median(np.abs(values - median), axis=0, keepdims=True)
    return median, np.where(mad < 1e-8, 1.0, 1.4826 * mad)


def apply_scaler(values: np.ndarray, median: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return np.clip((values - median) / scale, -10.0, 10.0)


def estimate_training_graph(arrays: list[np.ndarray], top_k: int) -> np.ndarray:
    values = np.concatenate(arrays, axis=0)
    values = values[np.all(np.isfinite(values), axis=1)]
    corr = np.nan_to_num(np.abs(np.corrcoef(values.T)), nan=0.0)
    np.fill_diagonal(corr, 0.0)
    graph = np.zeros_like(corr)
    for i in range(len(corr)):
        for j in np.argsort(corr[i])[-top_k:]:
            graph[i, j] = graph[j, i] = corr[i, j]
    return graph


def select_channels(record: RecordFeatures, channels: list[str]) -> np.ndarray:
    index = {channel: i for i, channel in enumerate(record.channels)}
    return record.features[:, [index[channel] for channel in channels]]


def canonical_channel(label: str) -> str:
    text = label.strip().upper().replace("EEG ", "")
    text = re.sub(r"-(REF|LE)$", "", text)
    if text in {"", "-", "ECG", "VNS", "LOC", "ROC"} or text.startswith("ECG"):
        return ""
    return text


def parse_summary(path: Path) -> dict[str, list[tuple[float, float]]]:
    lines = path.read_text(errors="ignore").splitlines()
    result: dict[str, list[tuple[float, float]]] = {}
    current = None; starts: list[float] = []
    for line in lines:
        match = re.search(r"File Name:\s*(\S+)\.edf", line)
        if match:
            current = match.group(1); result.setdefault(current, []); starts = []; continue
        if current is None: continue
        start = re.search(r"Seizure(?: \d+)? Start Time:\s*(\d+)\s*seconds", line)
        end = re.search(r"Seizure(?: \d+)? End Time:\s*(\d+)\s*seconds", line)
        if start: starts.append(float(start.group(1)))
        if end and starts: result[current].append((starts.pop(0), float(end.group(1))))
    return result


def count_alarm_episodes(times: np.ndarray, refractory_seconds: float) -> int:
    if len(times) == 0: return 0
    count = 1; last = float(times[0])
    for value in times[1:]:
        if float(value) - last >= refractory_seconds:
            count += 1; last = float(value)
    return count


def summarize_subjects(events: pd.DataFrame) -> pd.DataFrame:
    return events.groupby(["subject", "method"], as_index=False).agg(
        n_events=("event_index", "size"), auc=("auc", "mean"),
        sensitivity=("event_predicted", "mean"),
        false_predictions_per_hour=("false_predictions_per_hour", "mean"),
        time_in_warning=("time_in_warning", "mean"),
    )


def summarize_overall(subjects: pd.DataFrame) -> pd.DataFrame:
    return subjects.groupby("method", as_index=False).agg(
        n_subjects=("subject", "nunique"), auc_mean=("auc", "mean"), auc_std=("auc", "std"),
        sensitivity_mean=("sensitivity", "mean"),
        false_predictions_per_hour_mean=("false_predictions_per_hour", "mean"),
        time_in_warning_mean=("time_in_warning", "mean"),
    ).sort_values("auc_mean", ascending=False)


def subject_cluster_bootstrap(subjects: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    primary = str(cfg["primary_method"])
    comparators = [m for m in subjects.method.unique() if m != primary]
    rng = np.random.default_rng(int(cfg["seed"]))
    rows = []
    metrics = ["auc", "sensitivity", "false_predictions_per_hour", "time_in_warning"]
    for metric in metrics:
        metric_row_indices = []
        for comparator in comparators:
            wide = subjects[subjects.method.isin([primary, comparator])].pivot(
                index="subject", columns="method", values=metric
            ).dropna()
            if len(wide) < 2:
                continue
            difference = (wide[primary] - wide[comparator]).to_numpy()
            draws = rng.choice(difference, size=(5000, len(difference)), replace=True).mean(axis=1)
            favorable_sign = 1.0 if metric in {"auc", "sensitivity"} else -1.0
            rows.append(
                {
                    "metric": metric, "reference": primary, "comparator": comparator,
                    "n_subjects": len(difference), "difference_mean": difference.mean(),
                    "ci95_low": np.quantile(draws, .025),
                    "ci95_high": np.quantile(draws, .975),
                    "probability_gt_zero": np.mean(draws > 0),
                    "favorable_direction": "higher" if favorable_sign > 0 else "lower",
                    "probability_favorable": np.mean(favorable_sign * draws > 0),
                    "sign_flip_p_two_sided": paired_sign_flip_pvalue(difference, rng),
                }
            )
            metric_row_indices.append(len(rows) - 1)
        if metric_row_indices:
            adjusted = holm_adjust(
                np.asarray([rows[index]["sign_flip_p_two_sided"] for index in metric_row_indices])
            )
            for index, value in zip(metric_row_indices, adjusted):
                rows[index]["holm_adjusted_p"] = value
    return pd.DataFrame(rows).sort_values(["metric", "difference_mean"], ascending=[True, False])


def paired_sign_flip_pvalue(difference: np.ndarray, rng: np.random.Generator) -> float:
    observed = abs(float(np.mean(difference)))
    n = len(difference)
    if n <= 16:
        integers = np.arange(2**n, dtype=np.uint64)[:, None]
        bits = (integers >> np.arange(n, dtype=np.uint64)) & 1
        signs = np.where(bits, 1.0, -1.0)
    else:
        signs = rng.choice([-1.0, 1.0], size=(100_000, n))
    null = np.abs(np.mean(signs * difference[None, :], axis=1))
    return float((np.sum(null >= observed) + 1) / (len(null) + 1))


def holm_adjust(pvalues: np.ndarray) -> np.ndarray:
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (len(pvalues) - rank) * float(pvalues[index]))
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def inventory_frame(records: list[RecordFeatures], manifest: list[dict]) -> pd.DataFrame:
    prepared = {(r.subject, r.record): r for r in records}
    rows = []
    for item in manifest:
        record = prepared.get((item["subject"], item["record"]))
        rows.append({"subject": item["subject"], "record": item["record"], "kind": item["kind"],
                     "available": record is not None, "n_channels": len(record.channels) if record else np.nan,
                     "n_epochs": len(record.features) if record else np.nan,
                     "n_seizures": len(record.seizures) if record else np.nan})
    return pd.DataFrame(rows)


def build_context_lookup(manifest: list[dict]) -> dict[tuple[str, str], str]:
    """Map each target seizure record to its locked predecessor record."""
    lookup: dict[tuple[str, str], str] = {}
    for item in manifest:
        predecessor = item["record"]
        for target in item.get("context_for", []):
            target_record = Path(target).stem
            key = (item["subject"], target_record)
            if key in lookup and lookup[key] != predecessor:
                raise RuntimeError(f"multiple context records declared for {key}")
            lookup[key] = predecessor
    return lookup


def preictal_overlaps_prior_seizure(target_onset: float,
                                    all_seizures: list[tuple[float, float]],
                                    cfg: dict) -> bool:
    preictal_start = target_onset - (
        float(cfg["seizure_prediction_horizon_seconds"])
        + float(cfg["seizure_occurrence_period_seconds"])
    )
    return any(
        other_onset < target_onset and other_offset > preictal_start
        for other_onset, other_offset in all_seizures
    )


def event_eligibility_frame(records: list[RecordFeatures],
                            context_lookup: dict[tuple[str, str], str],
                            cfg: dict) -> pd.DataFrame:
    required_history = (
        float(cfg["seizure_prediction_horizon_seconds"])
        + float(cfg["seizure_occurrence_period_seconds"])
        + float(cfg["window_epochs"]) * float(cfg["epoch_seconds"])
    )
    by_key = {(record.subject, record.record): record for record in records}
    rows = []
    for record in records:
        for event_index, (onset, offset) in enumerate(record.seizures):
            context_name = context_lookup.get((record.subject, record.record))
            context = by_key.get((record.subject, context_name)) if context_name else None
            context_seconds = (
                len(context.features) * float(cfg["epoch_seconds"])
                if context is not None else 0.0
            )
            available_history = onset + context_seconds
            shifted_onset = onset + context_seconds
            all_seizures = []
            if context is not None:
                all_seizures.extend(context.seizures)
            all_seizures.extend(
                (start + context_seconds, stop + context_seconds)
                for start, stop in record.seizures
            )
            overlap = preictal_overlaps_prior_seizure(shifted_onset, all_seizures, cfg)
            eligible = available_history >= required_history and not overlap
            if eligible:
                exclusion_reason = ""
            elif context_name and context is None:
                exclusion_reason = "locked_context_record_not_downloaded"
            elif overlap:
                exclusion_reason = "preceding_seizure_overlaps_preictal_interval"
            else:
                exclusion_reason = "insufficient_preictal_history"
            rows.append(
                {
                    "subject": record.subject,
                    "record": record.record,
                    "event_index": event_index,
                    "onset_seconds": onset,
                    "offset_seconds": offset,
                    "context_record": context_name or "",
                    "context_available": context is not None,
                    "available_history_seconds": available_history,
                    "required_history_seconds": required_history,
                    "prior_seizure_overlap": overlap,
                    "eligible": eligible,
                    "exclusion_reason": exclusion_reason,
                }
            )
    return pd.DataFrame(rows)


def write_report(output: Path, cfg: dict, inventory: pd.DataFrame, folds: pd.DataFrame,
                 overall: pd.DataFrame, clustered: pd.DataFrame) -> None:
    lines = ["# Locked multi-subject CHB-MIT evaluation", "", "## Protocol", "", "```json",
             json.dumps(cfg, indent=2), "```", "", "## Inventory", "",
             inventory.groupby(["kind", "available"]).size().reset_index(name="records").to_csv(index=False).strip(),
             "", "## Completed folds", "", folds.status.value_counts().to_csv().strip(), "",
             "## Subject-level summary", "", overall.to_csv(index=False).strip(), "",
             "## Subject-clustered paired comparisons", "", clustered.to_csv(index=False).strip(), ""]
    (output / "REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
