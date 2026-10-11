from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SimulationResult:
    x: np.ndarray
    times: np.ndarray
    transition_step: int
    true_mode: int
    true_localization: np.ndarray
    alpha: np.ndarray
    laplacian_eigenvalues: np.ndarray
    laplacian_eigenvectors: np.ndarray


def simulate_modal_ou(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    n_steps: int,
    dt: float,
    transition_step: int,
    true_mode: int,
    noise_sigma: float,
    rng: np.random.Generator,
) -> SimulationResult:
    n = len(eigenvalues)
    y = np.zeros((n_steps, n), dtype=float)
    alpha = modal_stability_schedule(eigenvalues, n_steps, transition_step, true_mode)

    mode_noise = np.full(n, 0.30 * noise_sigma, dtype=float)
    mode_noise[true_mode] = noise_sigma

    for t in range(1, n_steps):
        drift = alpha[t - 1] * y[t - 1]
        noise = mode_noise * np.sqrt(dt) * rng.normal(size=n)
        y[t] = y[t - 1] + drift * dt + noise

    x = y @ eigenvectors.T
    loc = np.abs(eigenvectors[:, true_mode])
    loc = loc / (loc.max() + 1e-12)
    return SimulationResult(
        x=x,
        times=np.arange(n_steps) * dt,
        transition_step=transition_step,
        true_mode=true_mode,
        true_localization=loc,
        alpha=alpha,
        laplacian_eigenvalues=eigenvalues,
        laplacian_eigenvectors=eigenvectors,
    )


def simulate_fhn_reduced(
    laplacian: np.ndarray,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    n_steps: int,
    dt: float,
    transition_step: int,
    true_mode: int,
    noise_sigma: float,
    rng: np.random.Generator,
) -> SimulationResult:
    n = laplacian.shape[0]
    x = np.zeros((n_steps, n), dtype=float)
    recovery = np.zeros((n_steps, n), dtype=float)
    mode_pattern = eigenvectors[:, true_mode]
    mode_pattern = mode_pattern / (np.max(np.abs(mode_pattern)) + 1e-12)
    bias = -0.30 + 0.52 * np.linspace(0, 1, n_steps)
    local_drive = 0.18 * mode_pattern
    coupling = 0.18

    for t in range(1, n_steps):
        ramp = smooth_ramp(t, transition_step)
        a_t = bias[t] + ramp * local_drive
        v = x[t - 1]
        w = recovery[t - 1]
        dv = v - (v**3) / 3.0 - w + a_t - coupling * (laplacian @ v)
        dw = 0.08 * (v + 0.7 - 0.8 * w)
        x[t] = v + dt * dv + noise_sigma * np.sqrt(dt) * rng.normal(size=n)
        recovery[t] = w + dt * dw + 0.35 * noise_sigma * np.sqrt(dt) * rng.normal(size=n)

    alpha = np.full((n_steps, n), np.nan, dtype=float)
    loc = np.abs(eigenvectors[:, true_mode])
    loc = loc / (loc.max() + 1e-12)
    return SimulationResult(
        x=x,
        times=np.arange(n_steps) * dt,
        transition_step=transition_step,
        true_mode=true_mode,
        true_localization=loc,
        alpha=alpha,
        laplacian_eigenvalues=eigenvalues,
        laplacian_eigenvectors=eigenvectors,
    )


def modal_stability_schedule(
    eigenvalues: np.ndarray,
    n_steps: int,
    transition_step: int,
    true_mode: int,
) -> np.ndarray:
    n = len(eigenvalues)
    alpha = np.zeros((n_steps, n), dtype=float)
    base = -2.20 - 0.18 * eigenvalues
    for t in range(n_steps):
        r = min(t / max(transition_step, 1), 1.25)
        alpha[t] = base
        alpha[t, true_mode] = -0.95 + 0.945 * r
        if t > transition_step:
            alpha[t, true_mode] = 0.03 + 0.10 * (t - transition_step) / max(n_steps - transition_step, 1)
    return alpha


def smooth_ramp(t: int, transition_step: int) -> float:
    z = (t - 0.70 * transition_step) / (0.12 * transition_step)
    return float(1.0 / (1.0 + np.exp(-z)))


def lag1_autocorr(v: np.ndarray) -> float:
    v = np.asarray(v, dtype=float)
    v = v - v.mean()
    if len(v) < 3 or np.std(v) < 1e-12:
        return 0.0
    return float(np.corrcoef(v[:-1], v[1:])[0, 1])
