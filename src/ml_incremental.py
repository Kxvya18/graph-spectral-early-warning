from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


@dataclass
class GroupedElasticNet:
    scaler: StandardScaler
    classifier: LogisticRegression
    C: float
    l1_ratio: float
    inner_auc: float

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.classifier.predict_proba(self.scaler.transform(x))[:, 1]


def fit_grouped_elastic_net(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    candidates: list[dict],
    seed: int,
) -> GroupedElasticNet:
    """Select elastic-net parameters using grouped inner validation.

    Each validation fold contains one positive-bearing group and one negative
    group. All samples from a record therefore remain on one side of a split.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)
    if x.ndim != 2 or len(x) != len(y) or len(y) != len(groups):
        raise ValueError("x, y, and groups have incompatible shapes")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("both classes are required")
    positive_groups = sorted(set(groups[y == 1]))
    negative_groups = sorted(set(groups[y == 0]))
    if len(positive_groups) < 2 or len(negative_groups) < 2:
        raise ValueError("at least two positive and two negative record groups are required")

    candidate_scores = []
    for candidate_index, candidate in enumerate(candidates):
        fold_scores = []
        for fold, positive_group in enumerate(positive_groups):
            negative_group = negative_groups[fold % len(negative_groups)]
            validation = (groups == positive_group) | (groups == negative_group)
            training = ~validation
            if len(np.unique(y[training])) < 2 or len(np.unique(y[validation])) < 2:
                continue
            scaler, classifier = fit_model(
                x[training], y[training], float(candidate["C"]),
                float(candidate["l1_ratio"]), seed + candidate_index * 1000 + fold,
            )
            probability = classifier.predict_proba(scaler.transform(x[validation]))[:, 1]
            fold_scores.append(float(roc_auc_score(y[validation], probability)))
        score = float(np.mean(fold_scores)) if fold_scores else float("-inf")
        candidate_scores.append(score)
    best = int(np.argmax(candidate_scores))
    if not np.isfinite(candidate_scores[best]):
        raise RuntimeError("no valid grouped inner folds")
    chosen = candidates[best]
    scaler, classifier = fit_model(
        x, y, float(chosen["C"]), float(chosen["l1_ratio"]), seed + 90_000,
    )
    return GroupedElasticNet(
        scaler=scaler, classifier=classifier,
        C=float(chosen["C"]), l1_ratio=float(chosen["l1_ratio"]),
        inner_auc=candidate_scores[best],
    )


def fit_model(x: np.ndarray, y: np.ndarray, C: float, l1_ratio: float,
              seed: int) -> tuple[StandardScaler, LogisticRegression]:
    scaler = StandardScaler().fit(x)
    classifier = LogisticRegression(
        penalty="elasticnet", solver="saga", C=C, l1_ratio=l1_ratio,
        class_weight="balanced", max_iter=5000, random_state=seed,
    )
    classifier.fit(scaler.transform(x), y)
    return scaler, classifier


def grouped_out_of_fold_probabilities(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    C: float,
    l1_ratio: float,
    seed: int,
) -> np.ndarray:
    """Return record-held-out probabilities for leakage-safe calibration."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)
    probability = np.full(len(y), np.nan, dtype=float)
    for fold, group in enumerate(sorted(set(groups))):
        validation = groups == group
        training = ~validation
        if len(np.unique(y[training])) < 2:
            continue
        scaler, classifier = fit_model(
            x[training], y[training], C, l1_ratio, seed + fold,
        )
        probability[validation] = classifier.predict_proba(
            scaler.transform(x[validation])
        )[:, 1]
    return probability
