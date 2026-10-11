from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.linalg import eigh

from causal_gsews import ac_transform, detrend_window, trailing_mean


@dataclass(frozen=True)
class RecoveryEstimate:
    rho: float
    rho_gap: float
    direction: np.ndarray
    raw_spectral_radius: float
    ridge: float


@dataclass(frozen=True)
class RecoveryTrajectory:
    starts: np.ndarray
    ends: np.ndarray
    rho: np.ndarray
    rho_gap: np.ndarray
    persistence: np.ndarray
    direction: np.ndarray
    concentration: np.ndarray
    low_frequency_mass: np.ndarray
    constant_mass: np.ndarray
    graph_smoothness: np.ndarray
    dominant_graph_mode: np.ndarray
    isolated: np.ndarray


@dataclass(frozen=True)
class PhenotypeCalibration:
    persistence_center: float
    persistence_scale: float
    alignment_center: float
    alignment_scale: float


def estimate_recovery(
    block: np.ndarray,
    lag_steps: int = 1,
    ridge_fraction: float = 0.05,
) -> RecoveryEstimate:
    """Estimate a reversible lagged recovery mode from one past-only window.

    For column-state dynamics x[t+lag] = R x[t] + noise, the population
    lag covariance satisfies C_lag = R C_0.  The raw diagnostic therefore uses
    C_lag (C_0 + ridge I)^-1.  The primary, real-valued recovery estimate uses
    the symmetric whitened lag operator.  This is the regularized reversible
    estimator and avoids interpreting complex finite-sample eigenvectors as a
    physiological spatial pattern.

    The returned ``direction`` is mapped back from whitened coordinates with
    C_0^(1/2).  It is a state-space pattern, rather than the dual generalized
    eigenvector, and is therefore the appropriate object to compare with graph
    Laplacian eigenvectors.
    """
    values = np.asarray(block, dtype=float)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("block must be a two-dimensional multichannel array")
    if lag_steps < 1 or lag_steps >= len(values) - 2:
        raise ValueError("lag_steps must be positive and smaller than the window")
    if ridge_fraction <= 0:
        raise ValueError("ridge_fraction must be positive")
    if not np.all(np.isfinite(values)):
        raise ValueError("block contains non-finite values")

    centered = detrend_window(values)
    past = centered[:-lag_steps]
    future = centered[lag_steps:]
    count = len(past)
    c0 = (past.T @ past) / count
    c_lag = (future.T @ past) / count
    scale = max(float(np.trace(c0) / c0.shape[0]), 1e-12)
    ridge = max(float(ridge_fraction) * scale, 1e-12)
    regularized = c0 + ridge * np.eye(c0.shape[0])

    raw_operator = np.linalg.solve(regularized.T, c_lag.T).T
    raw_values = np.linalg.eigvals(raw_operator)
    raw_radius = float(np.max(np.abs(raw_values)))

    covariance_values, covariance_vectors = eigh(regularized, check_finite=False)
    covariance_values = np.maximum(covariance_values, 1e-12)
    inverse_root = (
        covariance_vectors * (1.0 / np.sqrt(covariance_values))[None, :]
    ) @ covariance_vectors.T
    root = (
        covariance_vectors * np.sqrt(covariance_values)[None, :]
    ) @ covariance_vectors.T
    symmetric_lag = 0.5 * (c_lag + c_lag.T)
    whitened = inverse_root @ symmetric_lag @ inverse_root
    recovery_values, recovery_vectors = eigh(whitened, check_finite=False)
    order = np.argsort(recovery_values)
    leading = int(order[-1])
    second = int(order[-2])
    rho = float(np.clip(recovery_values[leading], -0.999, 0.999))
    rho_gap = float(max(recovery_values[leading] - recovery_values[second], 0.0))
    direction = root @ recovery_vectors[:, leading]
    direction /= max(float(np.linalg.norm(direction)), 1e-12)
    return RecoveryEstimate(
        rho=rho,
        rho_gap=rho_gap,
        direction=direction,
        raw_spectral_radius=raw_radius,
        ridge=ridge,
    )


def graph_alignment(
    direction: np.ndarray,
    graph_eigenvectors: np.ndarray,
    graph_eigenvalues: np.ndarray | None = None,
    low_frequency_fraction: float = 0.25,
    exclude_constant: bool = True,
) -> dict[str, float | int]:
    vector = np.asarray(direction, dtype=float)
    basis = np.asarray(graph_eigenvectors, dtype=float)
    if basis.shape != (len(vector), len(vector)):
        raise ValueError("graph eigenvectors have incompatible shape")
    if not np.allclose(basis.T @ basis, np.eye(len(vector)), atol=1e-6):
        raise ValueError("graph eigenvectors must be orthonormal")
    weights = np.square(basis.T @ vector)
    weights /= max(float(weights.sum()), 1e-12)
    first = 1 if exclude_constant and len(vector) > 1 else 0
    monitored = np.arange(first, len(vector))
    low_count = max(1, int(np.ceil(len(monitored) * low_frequency_fraction)))
    dominant = int(monitored[np.argmax(weights[monitored])])
    if graph_eigenvalues is None:
        concentration = float(np.max(weights[monitored]))
    else:
        eigenvalues_for_groups = np.asarray(graph_eigenvalues, dtype=float)
        scale = max(float(np.max(np.abs(eigenvalues_for_groups))), 1.0)
        tolerance = 1e-6 * scale
        group_masses = []
        group_start = first
        for index in range(first + 1, len(vector) + 1):
            boundary = (
                index == len(vector)
                or abs(eigenvalues_for_groups[index] - eigenvalues_for_groups[index - 1])
                > tolerance
            )
            if boundary:
                group_masses.append(float(np.sum(weights[group_start:index])))
                group_start = index
        concentration = max(group_masses)
    if graph_eigenvalues is None:
        smoothness = np.nan
    else:
        eigenvalues = np.asarray(graph_eigenvalues, dtype=float)
        if eigenvalues.shape != (len(vector),):
            raise ValueError("graph eigenvalues have incompatible shape")
        maximum = max(float(eigenvalues[-1]), 1e-12)
        smoothness = float(np.clip(1.0 - np.dot(weights, eigenvalues) / maximum, 0.0, 1.0))
    return {
        "concentration": concentration,
        "low_frequency_mass": float(np.sum(weights[monitored[:low_count]])),
        "constant_mass": float(weights[0]) if first else 0.0,
        "graph_smoothness": smoothness,
        "dominant_graph_mode": dominant,
    }


def compute_recovery_trajectory(
    features: np.ndarray,
    graph_eigenvectors: np.ndarray,
    graph_eigenvalues: np.ndarray | None,
    window: int,
    stride: int,
    lag_steps: int = 1,
    ridge_fraction: float = 0.05,
    trailing_smooth: int = 1,
    isolated_gap: float = 0.10,
    low_frequency_fraction: float = 0.25,
    exclude_constant: bool = True,
) -> RecoveryTrajectory:
    """Compute strictly causal recovery and graph-alignment trajectories."""
    x = np.asarray(features, dtype=float)
    starts = np.arange(0, len(x) - window + 1, stride, dtype=int)
    estimates: list[RecoveryEstimate] = []
    alignments: list[dict[str, float | int]] = []
    kept_starts: list[int] = []
    kept_ends: list[int] = []
    for start in starts:
        end = int(start + window)
        block = x[start:end]
        if not np.all(np.isfinite(block)):
            continue
        estimate = estimate_recovery(block, lag_steps, ridge_fraction)
        alignment = graph_alignment(
            estimate.direction,
            graph_eigenvectors,
            graph_eigenvalues,
            low_frequency_fraction,
            exclude_constant,
        )
        estimates.append(estimate)
        alignments.append(alignment)
        kept_starts.append(int(start))
        kept_ends.append(end)
    if not estimates:
        raise ValueError("no complete finite windows were available")
    directions = sign_stabilize(np.asarray([item.direction for item in estimates]))
    rho = np.asarray([item.rho for item in estimates])
    gaps = np.asarray([item.rho_gap for item in estimates])
    persistence = trailing_mean(ac_transform(rho), trailing_smooth)
    return RecoveryTrajectory(
        starts=np.asarray(kept_starts),
        ends=np.asarray(kept_ends),
        rho=rho,
        rho_gap=gaps,
        persistence=persistence,
        direction=directions,
        concentration=trailing_mean(
            np.asarray([float(item["concentration"]) for item in alignments]),
            trailing_smooth,
        ),
        low_frequency_mass=trailing_mean(
            np.asarray([float(item["low_frequency_mass"]) for item in alignments]),
            trailing_smooth,
        ),
        constant_mass=trailing_mean(
            np.asarray([float(item["constant_mass"]) for item in alignments]),
            trailing_smooth,
        ),
        graph_smoothness=trailing_mean(
            np.asarray([float(item["graph_smoothness"]) for item in alignments]),
            trailing_smooth,
        ),
        dominant_graph_mode=np.asarray(
            [int(item["dominant_graph_mode"]) for item in alignments]
        ),
        isolated=gaps >= float(isolated_gap),
    )


def realign_trajectory(
    trajectory: RecoveryTrajectory,
    graph_eigenvectors: np.ndarray,
    graph_eigenvalues: np.ndarray | None,
    low_frequency_fraction: float = 0.25,
    exclude_constant: bool = True,
    trailing_smooth: int = 1,
) -> RecoveryTrajectory:
    alignments = [
        graph_alignment(
            direction,
            graph_eigenvectors,
            graph_eigenvalues,
            low_frequency_fraction,
            exclude_constant,
        )
        for direction in trajectory.direction
    ]
    return replace(
        trajectory,
        concentration=trailing_mean(
            np.asarray([float(item["concentration"]) for item in alignments]),
            trailing_smooth,
        ),
        low_frequency_mass=trailing_mean(
            np.asarray([float(item["low_frequency_mass"]) for item in alignments]),
            trailing_smooth,
        ),
        constant_mass=trailing_mean(
            np.asarray([float(item["constant_mass"]) for item in alignments]),
            trailing_smooth,
        ),
        graph_smoothness=trailing_mean(
            np.asarray([float(item["graph_smoothness"]) for item in alignments]),
            trailing_smooth,
        ),
        dominant_graph_mode=np.asarray(
            [int(item["dominant_graph_mode"]) for item in alignments]
        ),
    )


def phenotype_scores(
    true_graph: RecoveryTrajectory,
    wrong_graphs: list[RecoveryTrajectory],
    baseline_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """Create a slowing score and a graph-null-calibrated phenotype score.

    Identity, random and permuted bases are the core nulls.  The positive
    alignment excess is zero unless the fitted graph concentrates the slow
    state direction more strongly than every supplied wrong basis.
    """
    if not wrong_graphs:
        raise ValueError("at least one wrong-graph trajectory is required")
    for control in wrong_graphs:
        if not np.array_equal(control.ends, true_graph.ends):
            raise ValueError("true and wrong graph trajectories are not aligned")
    mask = np.asarray(baseline_mask, dtype=bool)
    if mask.shape != true_graph.rho.shape or mask.sum() < 3:
        raise ValueError("at least three baseline windows are required")
    calibration = fit_phenotype_calibration(
        [true_graph], [[control] for control in wrong_graphs], [mask]
    )
    return apply_phenotype_calibration(true_graph, wrong_graphs, calibration)


def fit_phenotype_calibration(
    true_records: list[RecoveryTrajectory],
    wrong_records: list[list[RecoveryTrajectory]],
    masks: list[np.ndarray] | None = None,
) -> PhenotypeCalibration:
    if not true_records or not wrong_records:
        raise ValueError("calibration requires true and wrong graph records")
    if len(wrong_records) < 1 or any(
        len(records) != len(true_records) for records in wrong_records
    ):
        raise ValueError("wrong graph calibration records are misaligned")
    if masks is None:
        masks = [np.ones(len(record.rho), dtype=bool) for record in true_records]
    persistence_parts = []
    alignment_parts = []
    for index, (record, mask) in enumerate(zip(true_records, masks)):
        selected = np.asarray(mask, dtype=bool)
        if selected.shape != record.rho.shape:
            raise ValueError("calibration mask has incompatible shape")
        controls = [records[index] for records in wrong_records]
        for control in controls:
            if not np.array_equal(control.ends, record.ends):
                raise ValueError("calibration trajectories are not aligned")
        persistence_parts.append(record.persistence[selected])
        alignment_parts.append(alignment_excess(record, controls)[selected])
    persistence = np.concatenate(persistence_parts)
    alignment = np.concatenate(alignment_parts)
    if len(persistence) < 3:
        raise ValueError("at least three calibration windows are required")
    return PhenotypeCalibration(
        persistence_center=float(np.median(persistence)),
        persistence_scale=robust_scale(persistence),
        alignment_center=float(np.median(alignment)),
        alignment_scale=robust_scale(alignment),
    )


def apply_phenotype_calibration(
    true_graph: RecoveryTrajectory,
    wrong_graphs: list[RecoveryTrajectory],
    calibration: PhenotypeCalibration,
) -> dict[str, np.ndarray]:
    persistence_z = (
        true_graph.persistence - calibration.persistence_center
    ) / calibration.persistence_scale
    excess = alignment_excess(true_graph, wrong_graphs)
    alignment_z = (excess - calibration.alignment_center) / calibration.alignment_scale
    isolated = true_graph.isolated.astype(float)
    phenotype = (
        np.maximum(persistence_z, 0.0)
        * np.maximum(alignment_z, 0.0)
        * (excess > 0.0)
        * isolated
    )
    return {
        "RecoveryOnly": persistence_z,
        "GraphAlignmentExcess": alignment_z,
        "RecoveryPhenotype": phenotype,
        "TrueGraphJoint": persistence_z * true_graph.concentration,
    }


def alignment_excess(
    true_graph: RecoveryTrajectory,
    wrong_graphs: list[RecoveryTrajectory],
) -> np.ndarray:
    if not wrong_graphs:
        raise ValueError("at least one wrong-graph trajectory is required")
    for control in wrong_graphs:
        if not np.array_equal(control.ends, true_graph.ends):
            raise ValueError("true and wrong graph trajectories are not aligned")
    worst_control = np.max(
        np.vstack([control.concentration for control in wrong_graphs]), axis=0
    )
    return true_graph.concentration - worst_control


def robust_scale(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    return max(1.4826 * mad, float(np.std(array, ddof=1)), 1e-8)


def robust_z(values: np.ndarray, baseline_mask: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    baseline = array[np.asarray(baseline_mask, dtype=bool)]
    return (array - np.median(baseline)) / robust_scale(baseline)


def sign_stabilize(vectors: np.ndarray) -> np.ndarray:
    values = np.asarray(vectors, dtype=float).copy()
    for index in range(1, len(values)):
        if float(values[index] @ values[index - 1]) < 0:
            values[index] *= -1.0
    return values
