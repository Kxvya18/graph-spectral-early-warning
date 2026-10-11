from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score


@dataclass
class MetricRow:
    graph: str
    model: str
    trial: int
    true_mode: int
    method: str
    auc: float
    lead_time_steps: float
    false_alarm_rate: float
    dominant_mode_accuracy: float
    localization_corr: float


def evaluate_scores(
    graph: str,
    model: str,
    trial: int,
    centers: np.ndarray,
    transition_step: int,
    true_mode: int,
    true_localization: np.ndarray,
    eigenvectors: np.ndarray,
    global_score: np.ndarray,
    dominant_mode: np.ndarray,
    baselines: dict[str, np.ndarray],
) -> list[MetricRow]:
    rows = []
    rows.append(
        make_row(
            graph,
            model,
            trial,
            true_mode,
            "GSEWS",
            centers,
            transition_step,
            global_score,
            dominant_mode,
            true_localization,
            eigenvectors,
        )
    )
    for name, score in baselines.items():
        rows.append(
            make_row(
                graph,
                model,
                trial,
                true_mode,
                name,
                centers,
                transition_step,
                score,
                None,
                true_localization,
                eigenvectors,
            )
        )
    return rows


def make_row(
    graph: str,
    model: str,
    trial: int,
    true_mode: int,
    method: str,
    centers: np.ndarray,
    transition_step: int,
    score: np.ndarray,
    dominant_mode: np.ndarray | None,
    true_localization: np.ndarray,
    eigenvectors: np.ndarray,
) -> MetricRow:
    labels = ((centers >= int(0.72 * transition_step)) & (centers < transition_step)).astype(int)
    if len(np.unique(labels)) < 2:
        auc = float("nan")
    else:
        auc = float(roc_auc_score(labels, score))
    threshold = score[centers < int(0.55 * transition_step)].mean() + 3.0 * score[centers < int(0.55 * transition_step)].std()
    alarms = centers[score >= threshold]
    valid_alarms = alarms[alarms < transition_step]
    if len(valid_alarms) == 0:
        lead = 0.0
    else:
        lead = float(transition_step - valid_alarms[0])
    early_false = alarms[alarms < int(0.55 * transition_step)]
    far = float(len(early_false) / max(np.sum(centers < int(0.55 * transition_step)), 1))

    if dominant_mode is None:
        mode_acc = float("nan")
        loc_corr = float("nan")
    else:
        pre = (centers >= int(0.72 * transition_step)) & (centers < transition_step)
        if np.sum(pre) == 0:
            mode_acc = 0.0
        else:
            mode_acc = float(np.mean(dominant_mode[pre] == true_mode))
        chosen = int(mode_vote(dominant_mode[pre])) if np.sum(pre) else int(dominant_mode[-1])
        pred_loc = np.abs(eigenvectors[:, chosen])
        pred_loc = pred_loc / (pred_loc.max() + 1e-12)
        loc_corr = corr(pred_loc, true_localization)

    return MetricRow(
        graph=graph,
        model=model,
        trial=trial,
        true_mode=true_mode,
        method=method,
        auc=auc,
        lead_time_steps=lead,
        false_alarm_rate=far,
        dominant_mode_accuracy=mode_acc,
        localization_corr=loc_corr,
    )


def mode_vote(v: np.ndarray) -> int:
    if len(v) == 0:
        return 0
    values, counts = np.unique(v, return_counts=True)
    return int(values[np.argmax(counts)])


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def rows_to_frame(rows):
    import pandas as pd

    return pd.DataFrame([r.__dict__ for r in rows])

