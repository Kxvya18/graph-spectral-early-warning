from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RecoverySimulation:
    x: np.ndarray
    rho: np.ndarray
    critical_direction: np.ndarray
    critical_graph_mode: int | None
    transition_step: int
    ramp_start: int
    scenario: str


def simulate_recovery_scenario(
    graph_eigenvectors: np.ndarray,
    graph_eigenvalues: np.ndarray,
    n_steps: int,
    ramp_start: int,
    transition_step: int,
    scenario: str,
    rng: np.random.Generator,
    noise_sigma: float = 0.20,
    critical_baseline_rho: float = 0.72,
    heterogeneous_noise_ratio: float = 4.0,
    burn_in: int = 500,
) -> RecoverySimulation:
    """Generate verified transition and falsification regimes.

    ``aligned`` and ``heterogeneous_noise`` commute with the supplied graph.
    The remaining regimes contain real recovery slowing but deliberately do
    not support the graph-mechanistic claim.
    """
    u = np.asarray(graph_eigenvectors, dtype=float)
    eigenvalues = np.asarray(graph_eigenvalues, dtype=float)
    n = len(eigenvalues)
    monitored = np.arange(1, n)
    critical_mode = int(monitored[np.argmin(np.abs(
        eigenvalues[monitored] / max(float(eigenvalues[-1]), 1e-12) - 0.35
    ))])

    if scenario in {"aligned", "heterogeneous_noise"}:
        dynamics_basis = u
        critical_index = critical_mode
        critical_direction = u[:, critical_mode]
        reported_mode: int | None = critical_mode
    elif scenario == "common_mode_artifact":
        dynamics_basis = u
        critical_index = 0
        critical_direction = u[:, 0]
        reported_mode = None
    elif scenario == "non_graph_slowing":
        direction = rng.normal(size=n)
        direction -= direction.mean()
        direction /= np.linalg.norm(direction)
        dynamics_basis = orthonormal_basis_with_first(direction, rng)
        critical_index = 0
        critical_direction = dynamics_basis[:, 0]
        reported_mode = None
    elif scenario == "localized_instability":
        direction = np.zeros(n)
        direction[int(rng.integers(0, n))] = 1.0
        direction -= direction.mean()
        direction /= np.linalg.norm(direction)
        dynamics_basis = orthonormal_basis_with_first(direction, rng)
        critical_index = 0
        critical_direction = dynamics_basis[:, 0]
        reported_mode = None
    else:
        raise ValueError(f"unknown scenario: {scenario}")

    base = np.linspace(0.25, 0.45, n)
    base[critical_index] = critical_baseline_rho
    schedule = np.tile(base, (n_steps, 1))
    for index in range(ramp_start, n_steps):
        fraction = np.clip(
            (index - ramp_start) / max(transition_step - ramp_start, 1), 0.0, 1.0
        )
        schedule[index, critical_index] = (
            critical_baseline_rho + fraction * (1.0 - critical_baseline_rho)
        )
    if not np.isclose(schedule[transition_step, critical_index], 1.0):
        raise RuntimeError("critical persistence does not reach one at transition")
    if np.max(schedule[transition_step - 1]) >= 1.0:
        raise RuntimeError("a mode crosses before the declared transition")

    noise_scale = np.full(n, noise_sigma)
    if scenario == "heterogeneous_noise":
        factors = np.geomspace(1.0, heterogeneous_noise_ratio, n)
        noise_scale = noise_sigma * factors[rng.permutation(n)] / np.sqrt(np.mean(factors**2))
    state = np.zeros(n)
    for _ in range(burn_in):
        modal = state @ dynamics_basis
        state = (modal * base) @ dynamics_basis.T + rng.normal(scale=noise_scale, size=n)
    x = np.empty((n_steps, n), dtype=float)
    x[0] = state
    for index in range(1, n_steps):
        modal = x[index - 1] @ dynamics_basis
        x[index] = (
            (modal * schedule[index - 1]) @ dynamics_basis.T
            + rng.normal(scale=noise_scale, size=n)
        )
    return RecoverySimulation(
        x=x,
        rho=schedule,
        critical_direction=critical_direction,
        critical_graph_mode=reported_mode,
        transition_step=transition_step,
        ramp_start=ramp_start,
        scenario=scenario,
    )


def orthonormal_basis_with_first(
    direction: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    vector = np.asarray(direction, dtype=float)
    vector /= max(float(np.linalg.norm(vector)), 1e-12)
    candidates = np.column_stack([vector, rng.normal(size=(len(vector), len(vector) - 1))])
    q, _ = np.linalg.qr(candidates)
    if float(q[:, 0] @ vector) < 0:
        q[:, 0] *= -1.0
    return q
