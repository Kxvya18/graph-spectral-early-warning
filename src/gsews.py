from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EWSResult:
    centers: np.ndarray
    modal_score: np.ndarray
    global_score: np.ndarray
    dominant_mode: np.ndarray
    modal_variance: np.ndarray
    modal_ac1: np.ndarray
    node_avg_score: np.ndarray
    node_max_score: np.ndarray
    pca_score: np.ndarray
    energy_score: np.ndarray
    graph_smoothness_score: np.ndarray
    edge_corr_score: np.ndarray


def compute_gsews(
    x: np.ndarray,
    eigenvectors: np.ndarray,
    window: int,
    stride: int,
    adjacency: np.ndarray | None = None,
) -> EWSResult:
    y = detrend(x) @ eigenvectors
    centers = []
    modal_var = []
    modal_ac = []
    node_scores = []
    node_max_scores = []
    pca_scores = []
    energy_scores = []
    smoothness_scores = []
    edge_corr_scores = []
    if adjacency is None:
        adjacency = np.ones((x.shape[1], x.shape[1]), dtype=float) - np.eye(x.shape[1])
    edges = np.argwhere(np.triu(adjacency, k=1) > 0)

    for start in range(0, len(x) - window + 1, stride):
        end = start + window
        centers.append(start + window // 2)
        wy = detrend(y[start:end])
        wx = detrend(x[start:end])
        v = np.var(wy, axis=0)
        ac = np.asarray([lag1_autocorr(wy[:, k]) for k in range(wy.shape[1])])
        modal_var.append(v)
        modal_ac.append(ac)

        node_var = np.var(wx, axis=0)
        node_ac = np.asarray([lag1_autocorr(wx[:, k]) for k in range(wx.shape[1])])
        node_ews = np.log1p(node_var) - np.log(np.clip(1.0 - node_ac, 1e-4, None))
        node_scores.append(np.mean(node_ews))
        node_max_scores.append(np.max(node_ews))

        svals = np.linalg.svd(wx - wx.mean(axis=0), full_matrices=False, compute_uv=False)
        pca_scores.append(np.log1p(svals[0] ** 2 / max(window - 1, 1)))
        energy_scores.append(np.log1p(np.mean(wx**2)))
        smoothness_scores.append(graph_smoothness(wx, adjacency))
        edge_corr_scores.append(edge_correlation(wx, edges))

    modal_var = np.asarray(modal_var)
    modal_ac = np.asarray(modal_ac)
    raw_score = zscore_time(np.log1p(modal_var)) + zscore_time(-np.log(np.clip(1.0 - modal_ac, 1e-4, None)))
    score = smooth_time(raw_score, radius=2)
    trend_score = cumulative_trend_score(score)
    localization_score = smooth_time(modal_var, radius=2)
    dominant = np.argmax(localization_score, axis=1)
    global_score = np.max(trend_score, axis=1)
    return EWSResult(
        centers=np.asarray(centers, dtype=int),
        modal_score=score,
        global_score=global_score,
        dominant_mode=dominant,
        modal_variance=modal_var,
        modal_ac1=modal_ac,
        node_avg_score=zscore_1d(np.asarray(node_scores)),
        node_max_score=zscore_1d(np.asarray(node_max_scores)),
        pca_score=zscore_1d(np.asarray(pca_scores)),
        energy_score=zscore_1d(np.asarray(energy_scores)),
        graph_smoothness_score=zscore_1d(np.asarray(smoothness_scores)),
        edge_corr_score=zscore_1d(np.asarray(edge_corr_scores)),
    )


def detrend(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    t = np.arange(len(x), dtype=float)
    t = (t - t.mean()) / (t.std() + 1e-12)
    out = np.empty_like(x)
    design = np.column_stack([np.ones_like(t), t])
    pinv = np.linalg.pinv(design)
    for k in range(x.shape[1]):
        beta = pinv @ x[:, k]
        out[:, k] = x[:, k] - design @ beta
    return out


def lag1_autocorr(v: np.ndarray) -> float:
    v = np.asarray(v, dtype=float)
    v = v - v.mean()
    if len(v) < 3 or np.std(v) < 1e-12:
        return 0.0
    ac = np.corrcoef(v[:-1], v[1:])[0, 1]
    return float(np.nan_to_num(ac, nan=0.0, posinf=0.0, neginf=0.0))


def zscore_time(a: np.ndarray) -> np.ndarray:
    base = max(3, min(len(a) // 3, len(a) - 1))
    mu = a[:base].mean(axis=0, keepdims=True)
    sd = a[:base].std(axis=0, keepdims=True) + 1e-8
    return (a - mu) / sd


def zscore_1d(v: np.ndarray) -> np.ndarray:
    base = max(3, min(len(v) // 3, len(v) - 1))
    return (v - v[:base].mean()) / (v[:base].std() + 1e-8)


def smooth_time(a: np.ndarray, radius: int = 2) -> np.ndarray:
    if radius <= 0 or len(a) < 3:
        return a
    out = np.empty_like(a)
    for i in range(len(a)):
        lo = max(0, i - radius)
        hi = min(len(a), i + radius + 1)
        out[i] = a[lo:hi].mean(axis=0)
    return out


def cumulative_trend_score(score: np.ndarray) -> np.ndarray:
    base = max(3, min(len(score) // 3, len(score) - 1))
    baseline = score[:base].mean(axis=0, keepdims=True)
    trend = score - baseline
    return np.maximum(trend, 0.0)


def graph_smoothness(wx: np.ndarray, adjacency: np.ndarray) -> float:
    degree = adjacency.sum(axis=1)
    lap = np.diag(degree) - adjacency
    vals = np.einsum("ti,ij,tj->t", wx, lap, wx)
    denom = np.mean(wx**2) + 1e-8
    return float(np.log1p(np.mean(vals) / denom))


def edge_correlation(wx: np.ndarray, edges: np.ndarray) -> float:
    if len(edges) == 0:
        return 0.0
    corr_vals = []
    for i, j in edges:
        a = wx[:, i]
        b = wx[:, j]
        if np.std(a) < 1e-10 or np.std(b) < 1e-10:
            continue
        corr_vals.append(abs(np.corrcoef(a, b)[0, 1]))
    if not corr_vals:
        return 0.0
    return float(np.mean(corr_vals))
