from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib_cache").resolve()))

import numpy as np
import pandas as pd
import pyedflib

from evaluate import MetricRow, rows_to_frame
from graphs import eigensystem, laplacian
from gsews import compute_gsews
from make_figures import set_style


EEG_IGNORE = {"-", "ECG", "VNS"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/chbmit_subset.json")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    data_dir = Path(cfg["data_dir"])
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)

    rows = []
    record_summaries = []
    representative = None

    for subject in cfg["subjects"]:
        subject_dir = data_dir / subject
        summary = parse_summary(subject_dir / f"{subject}-summary.txt")
        for edf_path in sorted(subject_dir.glob("*.edf")):
            record = edf_path.stem
            seizures = summary.get(record, [])
            if not seizures:
                continue
            try:
                result = analyze_record(edf_path, seizures, cfg)
            except Exception as exc:
                record_summaries.append({"subject": subject, "record": record, "status": f"failed: {exc}"})
                continue
            rows.extend(result["rows"])
            record_summaries.append(
                {
                    "subject": subject,
                    "record": record,
                    "status": "ok",
                    "n_channels": result["n_channels"],
                    "n_epochs": result["n_epochs"],
                    "n_seizures": len(seizures),
                }
            )
            if representative is None:
                representative = result

    if not rows:
        raise RuntimeError("no analyzable seizure records found; download data first")

    metrics = rows_to_frame(rows)
    metrics.to_csv(out / "chbmit_metrics.csv", index=False)
    pd.DataFrame(record_summaries).to_csv(out / "record_summary.csv", index=False)
    summary = summarize(metrics)
    summary.to_csv(out / "summary_by_method.csv", index=False)
    write_report(out, summary, metrics)
    if representative is not None:
        plot_real_representative(out / "figures", representative)
    plot_real_summary(out / "figures", summary)
    print(summary.to_string(index=False))
    print(f"\nWrote CHB-MIT real EEG results to {out}")


def analyze_record(edf_path: Path, seizures: list[tuple[int, int]], cfg: dict) -> dict:
    eeg_labels, signals, fs = load_eeg(edf_path, int(cfg["min_channels"]))
    features = epoch_features(signals, fs, float(cfg["epoch_seconds"]), cfg.get("feature_set", "line_length"))
    centers_seconds = (np.arange(len(features)) + 0.5) * float(cfg["epoch_seconds"])
    adjacency = estimate_graph(features, int(cfg["graph_top_k"]))
    l = laplacian(adjacency, normalized=False)
    _, evecs = eigensystem(l)
    ews = compute_gsews(features, evecs, int(cfg["window_epochs"]), int(cfg["stride_epochs"]), adjacency=adjacency)

    rows = []
    for idx, (onset, offset) in enumerate(seizures):
        centers = ews.centers
        centers_sec = centers_seconds[centers]
        transition_epoch = int(onset / float(cfg["epoch_seconds"]))
        preictal = (centers_sec >= onset - float(cfg["preictal_seconds"])) & (centers_sec < onset)
        baseline = centers_sec < max(onset - float(cfg["preictal_seconds"]), float(cfg["early_baseline_seconds"]))
        usable = preictal | baseline
        if np.sum(preictal) < 3 or np.sum(baseline) < 3:
            continue
        pseudo_loc = np.abs(evecs[:, int(most_common(ews.dominant_mode[preictal]))])
        pseudo_loc = pseudo_loc / (pseudo_loc.max() + 1e-12)
        clinical_labels = preictal[usable].astype(int)
        rows.extend(
            evaluate_real_scores(
                record=edf_path.stem,
                trial=idx,
                centers=centers[usable],
                labels=clinical_labels,
                onset_epoch=transition_epoch,
                global_score=ews.global_score[usable],
                dominant_mode=ews.dominant_mode[usable],
                true_mode=int(most_common(ews.dominant_mode[preictal])),
                baselines={
                    "NodeAvg": ews.node_avg_score[usable],
                    "NodeMax": ews.node_max_score[usable],
                    "PCA": ews.pca_score[usable],
                    "Energy": ews.energy_score[usable],
                    "GraphSmooth": ews.graph_smoothness_score[usable],
                    "EdgeCorr": ews.edge_corr_score[usable],
                },
            )
        )

    return {
        "rows": rows,
        "edf_path": edf_path,
        "labels": eeg_labels,
        "features": features,
        "adjacency": adjacency,
        "evecs": evecs,
        "ews": ews,
        "seizures": seizures,
        "centers_seconds": centers_seconds,
        "n_channels": len(eeg_labels),
        "n_epochs": len(features),
    }


def load_eeg(edf_path: Path, min_channels: int) -> tuple[list[str], np.ndarray, float]:
    reader = pyedflib.EdfReader(str(edf_path))
    try:
        labels = reader.getSignalLabels()
        selected = []
        clean_labels = []
        for i, label in enumerate(labels):
            simple = label.strip().split("-")[0] if label.strip() != "-" else "-"
            if label.strip() in EEG_IGNORE or simple in EEG_IGNORE:
                continue
            fs = reader.getSampleFrequency(i)
            if fs <= 0:
                continue
            selected.append(i)
            clean_labels.append(label.strip())
        if len(selected) < min_channels:
            raise RuntimeError(f"only {len(selected)} EEG channels")
        fs0 = reader.getSampleFrequency(selected[0])
        keep = [i for i in selected if abs(reader.getSampleFrequency(i) - fs0) < 1e-6]
        keep_labels = [labels[i].strip() for i in keep]
        signals = np.vstack([reader.readSignal(i) for i in keep]).T
        signals = robust_standardize(signals)
        return keep_labels, signals, float(fs0)
    finally:
        reader.close()


def epoch_features(signals: np.ndarray, fs: float, epoch_seconds: float, feature_set: str) -> np.ndarray:
    epoch = int(round(fs * epoch_seconds))
    n_epochs = len(signals) // epoch
    trimmed = signals[: n_epochs * epoch]
    x = trimmed.reshape(n_epochs, epoch, signals.shape[1])
    if feature_set == "line_length":
        feat = np.mean(np.abs(np.diff(x, axis=1)), axis=1)
    elif feature_set == "rms":
        feat = np.sqrt(np.mean(x**2, axis=1))
    elif feature_set == "mixed":
        rms = np.sqrt(np.mean(x**2, axis=1))
        diff_rms = np.sqrt(np.mean(np.diff(x, axis=1) ** 2, axis=1))
        feat = rms + 0.5 * diff_rms
    else:
        raise ValueError(f"unknown feature_set: {feature_set}")
    feat = np.log1p(feat)
    return robust_standardize(feat)


def estimate_graph(features: np.ndarray, top_k: int) -> np.ndarray:
    n = features.shape[1]
    baseline = features[: max(20, len(features) // 3)]
    corr = np.corrcoef(baseline.T)
    corr = np.nan_to_num(np.abs(corr), nan=0.0)
    np.fill_diagonal(corr, 0.0)
    a = np.zeros((n, n), dtype=float)
    for i in range(n):
        nbrs = np.argsort(corr[i])[-top_k:]
        for j in nbrs:
            if corr[i, j] > 0:
                a[i, j] = a[j, i] = corr[i, j]
    return a


def parse_summary(path: Path) -> dict[str, list[tuple[int, int]]]:
    text = path.read_text(errors="ignore").splitlines()
    current = None
    out: dict[str, list[tuple[int, int]]] = {}
    starts: list[int] = []
    for line in text:
        m = re.search(r"File Name:\s*(\S+)\.edf", line)
        if m:
            current = m.group(1)
            out.setdefault(current, [])
            starts = []
            continue
        if current is None:
            continue
        s = re.search(r"Seizure(?: \d+)? Start Time:\s*(\d+)\s*seconds", line)
        e = re.search(r"Seizure(?: \d+)? End Time:\s*(\d+)\s*seconds", line)
        if s:
            starts.append(int(s.group(1)))
        if e and starts:
            out[current].append((starts.pop(0), int(e.group(1))))
    return out


def robust_standardize(x: np.ndarray) -> np.ndarray:
    med = np.median(x, axis=0, keepdims=True)
    mad = np.median(np.abs(x - med), axis=0, keepdims=True) + 1e-8
    return np.clip((x - med) / (1.4826 * mad), -8.0, 8.0)


def most_common(v: np.ndarray) -> int:
    vals, counts = np.unique(v, return_counts=True)
    return int(vals[np.argmax(counts)])


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("method", as_index=False)
        .agg(
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            lead_mean=("lead_time_steps", "mean"),
            false_alarm_mean=("false_alarm_rate", "mean"),
        )
        .sort_values("method")
    )


def evaluate_real_scores(
    record: str,
    trial: int,
    centers: np.ndarray,
    labels: np.ndarray,
    onset_epoch: int,
    global_score: np.ndarray,
    dominant_mode: np.ndarray,
    true_mode: int,
    baselines: dict[str, np.ndarray],
) -> list[MetricRow]:
    rows = [
        real_row("GSEWS", record, trial, centers, labels, onset_epoch, global_score, dominant_mode, true_mode)
    ]
    for name, score in baselines.items():
        rows.append(real_row(name, record, trial, centers, labels, onset_epoch, score, None, true_mode))
    return rows


def real_row(
    method: str,
    record: str,
    trial: int,
    centers: np.ndarray,
    labels: np.ndarray,
    onset_epoch: int,
    score: np.ndarray,
    dominant_mode: np.ndarray | None,
    true_mode: int,
) -> MetricRow:
    from sklearn.metrics import roc_auc_score

    auc = float(roc_auc_score(labels, score)) if len(np.unique(labels)) == 2 else float("nan")
    baseline_score = score[labels == 0]
    threshold = baseline_score.mean() + 3.0 * baseline_score.std()
    alarms = centers[score >= threshold]
    pre_alarms = alarms[alarms < onset_epoch]
    lead = float(onset_epoch - pre_alarms[0]) if len(pre_alarms) else 0.0
    false_alarm = float(np.mean((score >= threshold) & (labels == 0))) if np.any(labels == 0) else 0.0
    if dominant_mode is None:
        mode_acc = float("nan")
    else:
        mode_acc = float(np.mean(dominant_mode[labels == 1] == true_mode)) if np.any(labels == 1) else float("nan")
    return MetricRow(
        graph="CHB-MIT",
        model=record,
        trial=trial,
        true_mode=true_mode,
        method=method,
        auc=auc,
        lead_time_steps=lead,
        false_alarm_rate=false_alarm,
        dominant_mode_accuracy=mode_acc,
        localization_corr=float("nan"),
    )


def write_report(out: Path, summary: pd.DataFrame, metrics: pd.DataFrame) -> None:
    lines = [
        "# CHB-MIT Real EEG GSEWS Report",
        "",
        "This is real multichannel EEG validation on downloaded CHB-MIT EDF recordings.",
        "Labels are preictal windows before annotated seizure onset versus early baseline windows from the same record.",
        "Localization is reported as an interpretable dominant modal pattern, not as ground-truth pathological-region accuracy.",
        "",
        "## Method Summary",
        "",
        summary.to_csv(index=False).strip(),
        "",
        "## Records",
        "",
        metrics.groupby(["model", "method"]).size().reset_index(name="events").to_csv(index=False).strip(),
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(lines))


def plot_real_summary(out: Path, summary: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    set_style()
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 2.8), constrained_layout=True)
    axes[0].bar(summary["method"], summary["auc_mean"], color="#222222")
    axes[0].set_ylim(0.35, 1.02)
    axes[0].set_ylabel("preictal AUC")
    axes[0].tick_params(axis="x", rotation=35)
    axes[1].bar(summary["method"], summary["lead_mean"], color="#1f77b4")
    axes[1].set_ylabel("lead time (epochs)")
    axes[1].tick_params(axis="x", rotation=35)
    axes[2].bar(summary["method"], summary["false_alarm_mean"], color="#d95f02")
    axes[2].set_ylabel("false alarm rate")
    axes[2].tick_params(axis="x", rotation=35)
    fig.savefig(out / "fig01_chbmit_real_eeg_summary.pdf", bbox_inches="tight")
    fig.savefig(out / "fig01_chbmit_real_eeg_summary.png", bbox_inches="tight")
    plt.close(fig)


def plot_real_representative(out: Path, result: dict) -> None:
    import matplotlib.pyplot as plt

    set_style()
    ews = result["ews"]
    centers_sec = result["centers_seconds"][ews.centers]
    onset = result["seizures"][0][0]
    fig, axes = plt.subplots(3, 1, figsize=(7.4, 5.2), constrained_layout=True)
    axes[0].plot(np.arange(len(result["features"])) * 2.0, result["features"][:, : min(8, result["features"].shape[1])], lw=0.6, alpha=0.7)
    axes[0].axvline(onset, color="crimson", lw=1.2, ls="--")
    axes[0].set_ylabel("EEG features")
    axes[0].set_title(result["edf_path"].stem)
    axes[1].plot(centers_sec, ews.global_score, color="#222222", lw=1.5, label="GSEWS")
    axes[1].plot(centers_sec, ews.node_avg_score, color="#d95f02", lw=1.0, label="NodeAvg")
    axes[1].plot(centers_sec, ews.edge_corr_score, color="#a6761d", lw=1.0, label="EdgeCorr")
    axes[1].axvline(onset, color="crimson", lw=1.2, ls="--")
    axes[1].set_ylabel("warning score")
    axes[1].legend(frameon=False, ncol=3)
    dom = int(most_common(ews.dominant_mode[centers_sec < onset]))
    loc = np.abs(result["evecs"][:, dom])
    axes[2].stem(np.arange(len(loc)), loc / (loc.max() + 1e-12), linefmt="#1b9e77", markerfmt="o", basefmt=" ")
    axes[2].set_ylabel("|dominant mode|")
    axes[2].set_xlabel("EEG channel index")
    fig.savefig(out / "fig02_chbmit_representative_warning.pdf", bbox_inches="tight")
    fig.savefig(out / "fig02_chbmit_representative_warning.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
