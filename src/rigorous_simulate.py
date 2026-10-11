from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RigorousSimulation:
    x: np.ndarray
    alpha: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    critical_mode: int
    transition_step: int
    model: str
    spectral_center: float


def spectral_drift_schedule(
    eigenvalues: np.ndarray,
    n_steps: int,
    transition_step: int,
    ramp_start: int = 0,
    spectral_center: float = 0.35,
    base_damping: float = 0.80,
    bandwidth: float = 0.05,
) -> tuple[np.ndarray, int]:
    """Return a stable-to-unstable graph-filter drift schedule.

    The node-space drift is B(t)=g_t(L), with
    g_t(lambda) = b(lambda) + mu(t) d h(lambda), where h is a narrow fixed
    spectral band. The band, rather than an eigenvector index, is prespecified.
    The drive d is calibrated so the first nonconstant mode reaches zero at
    transition_step while modes outside the band remain damped.
    """
    vals = np.asarray(eigenvalues, dtype=float)
    scale = max(float(vals[-1]), 1e-12)
    z = vals / scale
    base = -base_damping - 0.15 * z
    profile = np.exp(-0.5 * ((z - spectral_center) / max(bandwidth, 1e-6)) ** 2)
    monitored = np.arange(1, len(vals))
    crossing_drive = -base[monitored] / np.maximum(profile[monitored], 1e-12)
    critical_mode = int(monitored[np.argmin(crossing_drive)])
    drive = float(-base[critical_mode] / profile[critical_mode])
    alpha = np.empty((n_steps, len(vals)), dtype=float)
    if not 0 <= ramp_start < transition_step:
        raise ValueError("ramp_start must precede transition_step")
    for t in range(n_steps):
        mu = max(t - ramp_start, 0) / max(transition_step - ramp_start, 1)
        alpha[t] = base + drive * mu * profile
    return alpha, critical_mode


def simulate_graph_commuting_ou(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    n_steps: int,
    dt: float,
    transition_step: int,
    noise_sigma: float,
    rng: np.random.Generator,
    ramp_start: int = 0,
    spectral_center: float = 0.35,
    heterogeneity: float = 0.0,
) -> RigorousSimulation:
    """Simulate dx=B(t)x dt + sigma dW in node space with isotropic noise."""
    alpha, critical_mode = spectral_drift_schedule(
        eigenvalues, n_steps, transition_step, ramp_start=ramp_start,
        spectral_center=spectral_center
    )
    n = len(eigenvalues)
    x = np.zeros((n_steps, n), dtype=float)
    if heterogeneity > 0:
        h = rng.normal(size=n)
        h -= h.mean()
        h /= np.linalg.norm(h) + 1e-12
        perturbation = heterogeneity * np.diag(h)
    else:
        perturbation = np.zeros((n, n), dtype=float)
    for t in range(1, n_steps):
        drift_matrix = (eigenvectors * alpha[t - 1]) @ eigenvectors.T + perturbation
        x[t] = (
            x[t - 1]
            + dt * (drift_matrix @ x[t - 1])
            + noise_sigma * np.sqrt(dt) * rng.normal(size=n)
        )
    return RigorousSimulation(
        x=x,
        alpha=alpha,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        critical_mode=critical_mode,
        transition_step=transition_step,
        model="graph_commuting_ou",
        spectral_center=spectral_center,
    )


def simulate_graph_pattern_normal_form(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    n_steps: int,
    dt: float,
    transition_step: int,
    noise_sigma: float,
    rng: np.random.Generator,
    ramp_start: int = 0,
    spectral_center: float = 0.35,
    cubic_strength: float = 1.0,
    heterogeneity: float = 0.0,
) -> RigorousSimulation:
    """Simulate a nonlinear graph pattern-forming normal form.

    The linearization is the same graph-commuting operator used by the OU
    theorem test. A node-wise saturating cubic term bounds the post-bifurcation
    state. The theoretical transition is verified by max_k alpha_k=0.
    """
    alpha, critical_mode = spectral_drift_schedule(
        eigenvalues, n_steps, transition_step, ramp_start=ramp_start,
        spectral_center=spectral_center
    )
    n = len(eigenvalues)
    x = np.zeros((n_steps, n), dtype=float)
    if heterogeneity > 0:
        h = rng.normal(size=n)
        h -= h.mean()
        h /= np.linalg.norm(h) + 1e-12
        perturbation = heterogeneity * np.diag(h)
    else:
        perturbation = np.zeros((n, n), dtype=float)
    for t in range(1, n_steps):
        linear = (eigenvectors * alpha[t - 1]) @ eigenvectors.T + perturbation
        state = x[t - 1]
        drift = linear @ state - cubic_strength * state**3
        x[t] = state + dt * drift + noise_sigma * np.sqrt(dt) * rng.normal(size=n)
    return RigorousSimulation(
        x=x,
        alpha=alpha,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        critical_mode=critical_mode,
        transition_step=transition_step,
        model="graph_pattern_normal_form",
        spectral_center=spectral_center,
    )


def perturb_adjacency(
    adjacency: np.ndarray,
    strength: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Perturb existing positive weights without using future signal data."""
    a = np.asarray(adjacency, dtype=float).copy()
    if strength <= 0:
        return a
    mask = np.triu(a > 0, 1)
    noise = np.zeros_like(a)
    draws = np.exp(strength * rng.normal(size=int(mask.sum())))
    noise[mask] = draws
    noise = noise + noise.T
    out = a * np.where(noise > 0, noise, 1.0)
    np.fill_diagonal(out, 0.0)
    return out
