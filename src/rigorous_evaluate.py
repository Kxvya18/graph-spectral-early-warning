from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def evaluate_causal_run(
    run_id: str,
    graph: str,
    model: str,
    seed: int,
    scores,
    transition_step: int,
    baseline_end: int,
    pretransition_start: int,
    pretransition_end: int,
    critical_mode: int,
    true_eigenvectors: np.ndarray,
    true_eigenvalues: np.ndarray,
    prediction_horizon: int,
    method_prefix: str = "",
    critical_modes: np.ndarray | None = None,
) -> list[dict]:
    negative = scores.ends <= baseline_end
    if pretransition_end > transition_step:
        raise ValueError("pretransition_end cannot exceed the transition")
    positive = (scores.ends > pretransition_start) & (scores.ends <= pretransition_end)
    usable = negative | positive
    labels = positive[usable].astype(int)
    rows = []
    for method, values in scores.scores.items():
        if len(np.unique(labels)) < 2:
            auc = float("nan")
        else:
            auc = float(roc_auc_score(labels, values[usable]))
        baseline_values = values[negative]
        threshold = float(baseline_values.mean() + 3.0 * baseline_values.std(ddof=1))
        false_alarm = float(np.mean(baseline_values >= threshold))
        pre_alarm_ends = scores.ends[positive & (values >= threshold)]
        lead = float(transition_step - pre_alarm_ends[0]) if len(pre_alarm_ends) else 0.0
        mode_acc = float("nan")
        loc_corr = float("nan")
        interval_mode_correct = float("nan")
        interval_mode_top3 = float("nan")
        spectral_error = float("nan")
        band_mode_accuracy = float("nan")
        interval_band_recovery = float("nan")
        interval_top3_band_recovery = float("nan")
        subspace_localization_corr = float("nan")
        interval_subspace_overlap = float("nan")
        localization = getattr(scores, "localization", {})
        localization_data = localization.get(method)
        if method == "GSEWS" or localization_data is not None:
            if localization_data is None:
                selected_all = scores.dominant_mode
                evidence_all = scores.modal_variance[:, scores.monitored_modes]
                evidence_modes = scores.monitored_modes
                vectors_all = None
            else:
                selected_all = localization_data["dominant_mode"]
                evidence_all = localization_data["mode_evidence"]
                evidence_modes = localization_data["monitored_modes"]
                vectors_all = localization_data.get("dominant_vector")
            selected = selected_all[positive]
            if len(selected):
                critical_band = np.asarray(
                    [critical_mode] if critical_modes is None else critical_modes,
                    dtype=int,
                )
                mode_acc = float(np.mean(selected == critical_mode))
                band_mode_accuracy = float(np.mean(np.isin(selected, critical_band)))
                vals, counts = np.unique(selected, return_counts=True)
                voted = int(vals[np.argmax(counts)])
                truth = np.abs(true_eigenvectors[:, critical_mode])
                if vectors_all is None:
                    predicted = np.abs(true_eigenvectors[:, voted])
                else:
                    vectors = vectors_all[positive]
                    projector = np.einsum("ti,tj->ij", vectors, vectors) / len(vectors)
                    _, vecs = np.linalg.eigh(projector)
                    predicted = np.abs(vecs[:, -1])
                loc_corr = safe_corr(predicted, truth)
                band_truth = np.sqrt(
                    np.sum(true_eigenvectors[:, critical_band] ** 2, axis=1)
                )
                subspace_localization_corr = safe_corr(predicted, band_truth)
                interval_evidence = evidence_all[positive].mean(axis=0)
                order = evidence_modes[np.argsort(interval_evidence)[::-1]]
                interval_mode_correct = float(order[0] == critical_mode)
                interval_mode_top3 = float(critical_mode in order[:3])
                interval_band_recovery = float(order[0] in critical_band)
                interval_top3_band_recovery = float(
                    np.any(np.isin(order[:3], critical_band))
                )
                subspace_size = len(critical_band)
                estimated_subspace = true_eigenvectors[:, order[:subspace_size]]
                true_subspace = true_eigenvectors[:, critical_band]
                interval_subspace_overlap = float(
                    np.sum((true_subspace.T @ estimated_subspace) ** 2)
                    / max(subspace_size, 1)
                )
                spectral_error = float(
                    abs(true_eigenvalues[order[0]] - true_eigenvalues[critical_mode])
                    / max(true_eigenvalues[-1], 1e-12)
                )
        rows.append(
            {
                "run_id": run_id,
                "graph": graph,
                "model": model,
                "seed": seed,
                "prediction_horizon": int(prediction_horizon),
                "method": f"{method_prefix}{method}",
                "auc": auc,
                "lead_steps": lead,
                "false_alarm_rate": false_alarm,
                "mode_accuracy": mode_acc,
                "localization_corr": loc_corr,
                "interval_mode_correct": interval_mode_correct,
                "interval_mode_top3": interval_mode_top3,
                "normalized_spectral_error": spectral_error,
                "band_mode_accuracy": band_mode_accuracy,
                "interval_band_recovery": interval_band_recovery,
                "interval_top3_band_recovery": interval_top3_band_recovery,
                "subspace_localization_corr": subspace_localization_corr,
                "interval_subspace_overlap": interval_subspace_overlap,
                "n_negative_windows": int(negative.sum()),
                "n_positive_windows": int(positive.sum()),
            }
        )
    return rows


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby(["model", "prediction_horizon", "method"], as_index=False)
        .agg(
            n_runs=("run_id", "nunique"),
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            lead_mean=("lead_steps", "mean"),
            false_alarm_mean=("false_alarm_rate", "mean"),
            mode_accuracy_mean=("mode_accuracy", "mean"),
            localization_corr_mean=("localization_corr", "mean"),
            interval_mode_recovery=("interval_mode_correct", "mean"),
            interval_top3_recovery=("interval_mode_top3", "mean"),
            normalized_spectral_error_mean=("normalized_spectral_error", "mean"),
            band_mode_accuracy_mean=("band_mode_accuracy", "mean"),
            interval_band_recovery=("interval_band_recovery", "mean"),
            interval_top3_band_recovery=("interval_top3_band_recovery", "mean"),
            subspace_localization_corr_mean=("subspace_localization_corr", "mean"),
            interval_subspace_overlap_mean=("interval_subspace_overlap", "mean"),
        )
        .sort_values(["model", "prediction_horizon", "auc_mean"], ascending=[True, False, False])
    )


def paired_bootstrap(
    metrics: pd.DataFrame,
    reference: str = "GSEWS",
    n_boot: int = 4000,
    seed: int = 20260804,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for (model, horizon), frame in metrics.groupby(["model", "prediction_horizon"]):
        wide = frame.pivot_table(index="run_id", columns="method", values="auc", aggfunc="first")
        if reference not in wide:
            continue
        for method in wide.columns:
            if method == reference:
                continue
            pair = wide[[reference, method]].dropna()
            if len(pair) < 2:
                continue
            differences = (pair[reference] - pair[method]).to_numpy()
            draws = rng.choice(differences, size=(n_boot, len(differences)), replace=True).mean(axis=1)
            rows.append(
                {
                    "model": model,
                    "prediction_horizon": int(horizon),
                    "reference": reference,
                    "comparator": method,
                    "n_pairs": len(differences),
                    "auc_difference_mean": float(differences.mean()),
                    "ci95_low": float(np.quantile(draws, 0.025)),
                    "ci95_high": float(np.quantile(draws, 0.975)),
                    "probability_difference_gt_zero": float(np.mean(draws > 0)),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["model", "prediction_horizon", "auc_difference_mean"],
        ascending=[True, True, False],
    )


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])
