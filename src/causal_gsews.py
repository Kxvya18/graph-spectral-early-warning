from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CausalScores:
    starts: np.ndarray
    ends: np.ndarray
    centers: np.ndarray
    scores: dict[str, np.ndarray]
    modal_variance: np.ndarray
    modal_ac1: np.ndarray
    dominant_mode: np.ndarray
    monitored_modes: np.ndarray


def compute_causal_scores(
    x: np.ndarray,
    eigenvectors: np.ndarray,
    adjacency: np.ndarray,
    window: int,
    stride: int,
    baseline_end: int,
    trailing_smooth: int = 3,
    trend_span: int = 6,
    exclude_constant: bool = True,
) -> CausalScores:
    """Compute GSEWS and comparators using information available by each end."""
    x = np.asarray(x, dtype=float)
    u = np.asarray(eigenvectors, dtype=float)
    starts = np.arange(0, len(x) - window + 1, stride, dtype=int)
    ends = starts + window
    centers = starts + window // 2
    if len(starts) < 4:
        raise ValueError("too few windows")
    baseline_mask = ends <= int(baseline_end)
    if baseline_mask.sum() < 3:
        raise ValueError("at least three complete baseline windows are required")

    modes = np.arange(u.shape[1], dtype=int)
    if exclude_constant and len(modes) > 1:
        modes = modes[1:]

    modal_var, modal_ac = [], []
    node_var, node_ac = [], []
    pca_var, pca_ac = [], []
    energy, smoothness, edge_corr = [], [], []
    edges = np.argwhere(np.triu(adjacency, 1) > 0)

    for start, end in zip(starts, ends):
        wx = detrend_window(x[start:end])
        wy = wx @ u
        modal_var.append(np.var(wy, axis=0, ddof=1))
        modal_ac.append([lag1(wy[:, k]) for k in range(wy.shape[1])])
        node_var.append(np.var(wx, axis=0, ddof=1))
        node_ac.append([lag1(wx[:, k]) for k in range(wx.shape[1])])
        _, _, vh = np.linalg.svd(wx, full_matrices=False)
        pc = wx @ vh[0]
        pca_var.append(float(np.var(pc, ddof=1)))
        pca_ac.append(lag1(pc))
        energy.append(float(np.mean(wx**2)))
        smoothness.append(graph_smoothness(wx, adjacency))
        edge_corr.append(edge_correlation(wx, edges))

    modal_var = np.asarray(modal_var, dtype=float)
    modal_ac = np.asarray(modal_ac, dtype=float)
    node_var = np.asarray(node_var, dtype=float)
    node_ac = np.asarray(node_ac, dtype=float)

    modal_level_var = baseline_z(np.log1p(modal_var), baseline_mask)
    modal_level_ac = baseline_z(ac_transform(modal_ac), baseline_mask)
    mv = baseline_z(causal_slope_matrix(np.log1p(modal_var), trend_span), baseline_mask)
    ma = baseline_z(causal_slope_matrix(ac_transform(modal_ac), trend_span), baseline_mask)
    nv = baseline_z(causal_slope_matrix(np.log1p(node_var), trend_span), baseline_mask)
    na = baseline_z(causal_slope_matrix(ac_transform(node_ac), trend_span), baseline_mask)
    pv = baseline_z(
        causal_slope_matrix(np.log1p(np.asarray(pca_var))[:, None], trend_span), baseline_mask
    )[:, 0]
    pa = baseline_z(
        causal_slope_matrix(ac_transform(np.asarray(pca_ac))[:, None], trend_span), baseline_mask
    )[:, 0]

    raw = {
        "GSEWS": np.max(mv[:, modes] + ma[:, modes], axis=1),
        "GSEWS_Var": np.max(mv[:, modes], axis=1),
        "GSEWS_AC": np.max(ma[:, modes], axis=1),
        "GSEWS_Level": np.max(modal_level_var[:, modes] + modal_level_ac[:, modes], axis=1),
        "NodeAvg": np.mean(nv + na, axis=1),
        "NodeMax": np.max(nv + na, axis=1),
        "PCA": pv + pa,
        "Energy": baseline_z(
            causal_slope_matrix(np.log1p(np.asarray(energy))[:, None], trend_span), baseline_mask
        )[:, 0],
        "GraphSmooth": baseline_z(
            causal_slope_matrix(np.asarray(smoothness)[:, None], trend_span), baseline_mask
        )[:, 0],
        "EdgeCorr": baseline_z(
            causal_slope_matrix(np.asarray(edge_corr)[:, None], trend_span), baseline_mask
        )[:, 0],
    }
    scores = {name: trailing_mean(values, trailing_smooth) for name, values in raw.items()}
    # Under equal isotropic noise, the modal variance law identifies the least
    # stable mode by absolute variance. This estimator is kept separate from
    # the global trend score and is valid for the primary controlled model.
    dominant = modes[
        np.argmax(trailing_matrix(modal_var[:, modes], max(trend_span, trailing_smooth)), axis=1)
    ]
    return CausalScores(
        starts=starts,
        ends=ends,
        centers=centers,
        scores=scores,
        modal_variance=modal_var,
        modal_ac1=modal_ac,
        dominant_mode=dominant,
        monitored_modes=modes,
    )


def compute_causal_modal_scores(
    x: np.ndarray,
    eigenvectors: np.ndarray,
    window: int,
    stride: int,
    baseline_end: int,
    trailing_smooth: int = 3,
    trend_span: int = 6,
    exclude_constant: bool = True,
) -> CausalScores:
    """Efficient GSEWS-only path for wrong-graph negative controls."""
    x = np.asarray(x, dtype=float)
    u = np.asarray(eigenvectors, dtype=float)
    starts = np.arange(0, len(x) - window + 1, stride, dtype=int)
    ends = starts + window
    centers = starts + window // 2
    baseline_mask = ends <= int(baseline_end)
    if baseline_mask.sum() < 3:
        raise ValueError("at least three complete baseline windows are required")
    modes = np.arange(u.shape[1], dtype=int)
    if exclude_constant and len(modes) > 1:
        modes = modes[1:]
    modal_var, modal_ac = [], []
    for start, end in zip(starts, ends):
        wy = detrend_window(x[start:end]) @ u
        modal_var.append(np.var(wy, axis=0, ddof=1))
        modal_ac.append([lag1(wy[:, k]) for k in range(wy.shape[1])])
    modal_var = np.asarray(modal_var, dtype=float)
    modal_ac = np.asarray(modal_ac, dtype=float)
    mv = baseline_z(causal_slope_matrix(np.log1p(modal_var), trend_span), baseline_mask)
    ma = baseline_z(causal_slope_matrix(ac_transform(modal_ac), trend_span), baseline_mask)
    level_var = baseline_z(np.log1p(modal_var), baseline_mask)
    level_ac = baseline_z(ac_transform(modal_ac), baseline_mask)
    raw = {
        "GSEWS": np.max(mv[:, modes] + ma[:, modes], axis=1),
        "GSEWS_Level": np.max(level_var[:, modes] + level_ac[:, modes], axis=1),
        "GSEWS_Var": np.max(mv[:, modes], axis=1),
        "GSEWS_AC": np.max(ma[:, modes], axis=1),
    }
    dominant = modes[
        np.argmax(trailing_matrix(modal_var[:, modes], max(trend_span, trailing_smooth)), axis=1)
    ]
    return CausalScores(
        starts=starts,
        ends=ends,
        centers=centers,
        scores={name: trailing_mean(values, trailing_smooth) for name, values in raw.items()},
        modal_variance=modal_var,
        modal_ac1=modal_ac,
        dominant_mode=dominant,
        monitored_modes=modes,
    )


def baseline_z(values: np.ndarray, baseline_mask: np.ndarray) -> np.ndarray:
    a = np.asarray(values, dtype=float)
    base = a[baseline_mask]
    mu = base.mean(axis=0, keepdims=True)
    sd = base.std(axis=0, ddof=1, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (a - mu) / sd


def ac_transform(ac: np.ndarray) -> np.ndarray:
    return -np.log(np.clip(1.0 - np.asarray(ac, dtype=float), 1e-4, None))


def trailing_mean(values: np.ndarray, span: int) -> np.ndarray:
    return trailing_matrix(np.asarray(values, dtype=float)[:, None], span)[:, 0]


def trailing_matrix(values: np.ndarray, span: int) -> np.ndarray:
    a = np.asarray(values, dtype=float)
    if span <= 1:
        return a.copy()
    out = np.empty_like(a)
    for i in range(len(a)):
        lo = max(0, i - span + 1)
        out[i] = a[lo : i + 1].mean(axis=0)
    return out


def causal_slope_matrix(values: np.ndarray, span: int) -> np.ndarray:
    """Trailing ordinary-least-squares slopes with no future samples."""
    a = np.asarray(values, dtype=float)
    out = np.zeros_like(a)
    for i in range(len(a)):
        lo = max(0, i - max(span, 3) + 1)
        block = a[lo : i + 1]
        if len(block) < 3:
            continue
        t = np.arange(len(block), dtype=float)
        t -= t.mean()
        denom = float(np.sum(t**2))
        out[i] = (t[:, None] * (block - block.mean(axis=0))).sum(axis=0) / denom
    return out


def detrend_window(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    t = np.linspace(-1.0, 1.0, len(a))
    design = np.column_stack([np.ones(len(a)), t])
    return a - design @ (np.linalg.pinv(design) @ a)


def lag1(v: np.ndarray) -> float:
    a = np.asarray(v, dtype=float)
    a = a - a.mean()
    if len(a) < 3 or np.std(a) < 1e-12:
        return 0.0
    return float(np.nan_to_num(np.corrcoef(a[:-1], a[1:])[0, 1]))


def graph_smoothness(wx: np.ndarray, adjacency: np.ndarray) -> float:
    lap = np.diag(adjacency.sum(axis=1)) - adjacency
    numerator = np.mean(np.einsum("ti,ij,tj->t", wx, lap, wx))
    denominator = np.mean(wx**2) + 1e-12
    return float(np.log1p(max(numerator / denominator, 0.0)))


def edge_correlation(wx: np.ndarray, edges: np.ndarray) -> float:
    values = []
    for i, j in edges:
        if np.std(wx[:, i]) > 1e-12 and np.std(wx[:, j]) > 1e-12:
            values.append(abs(np.corrcoef(wx[:, i], wx[:, j])[0, 1]))
    return float(np.mean(values)) if values else 0.0
