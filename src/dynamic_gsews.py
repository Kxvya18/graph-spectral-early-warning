from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh

from causal_gsews import (
    ac_transform,
    baseline_z,
    causal_slope_matrix,
    detrend_window,
    trailing_mean,
    trailing_matrix,
)


@dataclass
class DynamicScores:
    starts: np.ndarray
    ends: np.ndarray
    centers: np.ndarray
    scores: dict[str, np.ndarray]
    modal_variance: np.ndarray
    modal_ac1: np.ndarray
    dominant_mode: np.ndarray
    monitored_modes: np.ndarray
    localization: dict[str, dict[str, np.ndarray]]
    generalized_persistence: np.ndarray
    generalized_vector: np.ndarray
    graph_concentration: np.ndarray
    graph_fit: np.ndarray
    graph_diagonality: np.ndarray


def compute_dynamic_scores(
    x: np.ndarray,
    graph_eigenvectors: np.ndarray,
    window: int,
    stride: int,
    baseline_end: int,
    ridge_fraction: float = 0.05,
    lag_steps: int = 1,
    trailing_smooth: int = 3,
    trend_span: int = 6,
    exclude_constant: bool = True,
    include_generalized: bool = True,
) -> DynamicScores:
    """Causal covariance-lag and graph-constrained dynamic warning scores.

    The generalized eigenproblem estimates the slowest direction without using
    a graph. The graph-constrained estimator restricts candidate directions to
    individual graph eigenvectors. Wrong-graph controls therefore undergo the
    identical computation but impose the wrong dynamical coordinate system.
    """
    x = np.asarray(x, dtype=float)
    u = np.asarray(graph_eigenvectors, dtype=float)
    if x.ndim != 2 or u.shape != (x.shape[1], x.shape[1]):
        raise ValueError("x and graph_eigenvectors have incompatible shapes")
    if not np.allclose(u.T @ u, np.eye(len(u)), atol=1e-6):
        raise ValueError("graph_eigenvectors must be orthonormal")
    if lag_steps < 1 or lag_steps >= window - 2:
        raise ValueError("lag_steps must be positive and smaller than the window")
    starts = np.arange(0, len(x) - window + 1, stride, dtype=int)
    ends = starts + window
    centers = starts + window // 2
    baseline_mask = ends <= int(baseline_end)
    if baseline_mask.sum() < 3:
        raise ValueError("at least three complete baseline windows are required")
    modes = np.arange(u.shape[1], dtype=int)
    if exclude_constant and len(modes) > 1:
        modes = modes[1:]

    modal_rho = []
    slow_rho = []
    slow_vectors = []
    concentrations = []
    projections = []
    fit_scores = []
    diagonalities = []
    for start, end in zip(starts, ends):
        block = detrend_window(x[start:end])
        x0, x1 = block[:-lag_steps], block[lag_steps:]
        scale = max(float(np.trace(x0.T @ x0) / (len(x0) * x.shape[1])), 1e-10)
        ridge = max(float(ridge_fraction) * scale, 1e-10)

        c0 = (x0.T @ x0) / len(x0)
        cross = (x0.T @ x1) / len(x0)
        c1 = 0.5 * (cross + cross.T)
        if include_generalized:
            values, vectors = eigh(c1, c0 + ridge * np.eye(x.shape[1]), check_finite=False)
            index = int(np.argmax(values))
            vector = vectors[:, index]
            vector /= max(float(np.linalg.norm(vector)), 1e-12)
            rho = float(np.clip(values[index], -0.999, 0.999))
            weights = np.square(u.T @ vector)
            weights /= max(float(weights.sum()), 1e-12)
            concentrations.append(float(np.max(weights[modes])))
            projections.append(weights)
            slow_rho.append(rho)
            slow_vectors.append(vector)

        y0, y1 = x0 @ u, x1 @ u
        modal_cross = (y0.T @ y1) / len(y0)
        diagonal_energy = float(np.sum(np.diag(modal_cross) ** 2))
        total_cross_energy = float(np.sum(modal_cross ** 2))
        diagonalities.append(diagonal_energy / max(total_cross_energy, 1e-12))
        denominator = np.sum(y0 * y0, axis=0) + ridge * len(y0)
        rho_modes = np.sum(y0 * y1, axis=0) / denominator
        rho_modes = np.clip(rho_modes, -0.999, 0.999)
        modal_rho.append(rho_modes)
        residual = y1 - y0 * rho_modes[None, :]
        residual_ratio = np.sum(residual * residual) / max(np.sum(y1 * y1), 1e-12)
        fit_scores.append(float(-np.log(max(residual_ratio, 1e-12))))

    modal_rho = np.asarray(modal_rho)
    slow_rho = np.asarray(slow_rho)
    slow_vectors = np.asarray(slow_vectors)
    concentrations = np.asarray(concentrations)
    projections = np.asarray(projections)
    fit_scores = np.asarray(fit_scores)
    diagonalities = np.asarray(diagonalities)

    modal_persistence = ac_transform(modal_rho)
    modal_level_z = baseline_z(modal_persistence, baseline_mask)
    modal_trend_z = baseline_z(
        causal_slope_matrix(modal_persistence, trend_span), baseline_mask
    )
    fit_level_z = baseline_z(fit_scores[:, None], baseline_mask)[:, 0]
    diagonal_level_z = baseline_z(diagonalities[:, None], baseline_mask)[:, 0]

    raw_scores = {
        "GraphDyn_Raw": np.max(modal_persistence[:, modes], axis=1),
        "GraphDyn_Level": np.max(modal_level_z[:, modes], axis=1),
        "GraphDyn_Trend": np.max(modal_trend_z[:, modes], axis=1),
        "GraphFit_Level": fit_level_z,
        "GraphDiagonal_Level": diagonal_level_z,
    }
    if include_generalized:
        slow_persistence = ac_transform(slow_rho)
        slow_level_z = baseline_z(slow_persistence[:, None], baseline_mask)[:, 0]
        aligned_raw = slow_persistence * concentrations
        aligned_level_z = baseline_z(aligned_raw[:, None], baseline_mask)[:, 0]
        raw_scores.update(
            {
                "SlowMode_Level": slow_level_z,
                "GraphAligned_Level": aligned_level_z,
            }
        )
    scores = {
        name: trailing_mean(values, trailing_smooth)
        for name, values in raw_scores.items()
    }

    modal_evidence = trailing_matrix(modal_persistence, trailing_smooth)
    dominant_modal = modes[np.argmax(modal_evidence[:, modes], axis=1)]
    graph_localization = {
        "dominant_mode": dominant_modal,
        "mode_evidence": modal_evidence[:, modes],
        "monitored_modes": modes,
    }
    localization = {
        "GraphDyn_Raw": graph_localization,
        "GraphDyn_Level": graph_localization,
        "GraphDyn_Trend": graph_localization,
    }
    if include_generalized:
        projection_evidence = trailing_matrix(
            projections * slow_persistence[:, None], trailing_smooth
        )
        dominant_generalized = modes[np.argmax(projection_evidence[:, modes], axis=1)]
        smoothed_vectors = sign_stable_trailing_vectors(slow_vectors, trailing_smooth)
        generalized_localization = {
            "dominant_mode": dominant_generalized,
            "mode_evidence": projection_evidence[:, modes],
            "monitored_modes": modes,
            "dominant_vector": smoothed_vectors,
        }
        localization["SlowMode_Level"] = generalized_localization
        localization["GraphAligned_Level"] = generalized_localization
    else:
        slow_rho = np.full(len(starts), np.nan)
        smoothed_vectors = np.full((len(starts), x.shape[1]), np.nan)
        concentrations = np.full(len(starts), np.nan)
    return DynamicScores(
        starts=starts,
        ends=ends,
        centers=centers,
        scores=scores,
        modal_variance=modal_persistence,
        modal_ac1=modal_rho,
        dominant_mode=dominant_modal,
        monitored_modes=modes,
        localization=localization,
        generalized_persistence=slow_rho,
        generalized_vector=smoothed_vectors,
        graph_concentration=concentrations,
        graph_fit=fit_scores,
        graph_diagonality=diagonalities,
    )


def sign_stable_trailing_vectors(vectors: np.ndarray, span: int) -> np.ndarray:
    """Average directions causally after resolving eigenvector sign ambiguity."""
    values = np.asarray(vectors, dtype=float)
    out = np.empty_like(values)
    for index in range(len(values)):
        lo = max(0, index - max(span, 1) + 1)
        block = values[lo : index + 1].copy()
        reference = block[-1]
        signs = np.sign(block @ reference)
        signs[signs == 0] = 1.0
        vector = np.mean(block * signs[:, None], axis=0)
        out[index] = vector / max(float(np.linalg.norm(vector)), 1e-12)
    return out
