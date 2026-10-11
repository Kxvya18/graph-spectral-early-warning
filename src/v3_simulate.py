from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class V3Simulation:
    x: np.ndarray
    rho: np.ndarray
    critical_mode: int
    transition_step: int
    ramp_start: int
    regime: str
    model: str


def discrete_spectral_schedule(
    eigenvalues: np.ndarray,
    n_steps: int,
    ramp_start: int,
    transition_step: int,
    regime: str,
    spectral_center: float = 0.35,
    band_width: float = 0.05,
    isolated_baseline_gap: float = 0.15,
) -> tuple[np.ndarray, int]:
    values = np.asarray(eigenvalues, dtype=float)
    z = values / max(float(values[-1]), 1e-12)
    base = 0.55 + 0.14 * (1.0 - z)
    base[0] = 0.50
    monitored = np.arange(1, len(values))
    if regime == "isolated_mode":
        critical = int(monitored[np.argmin(np.abs(z[monitored] - spectral_center))])
        other_max = float(np.max(np.delete(base, critical)))
        base[critical] = min(other_max + isolated_baseline_gap, 0.90)
        profile = np.zeros_like(z)
        profile[critical] = 1.0
        drive = 1.0 - base[critical]
    elif regime == "near_degenerate_band":
        profile = np.exp(-0.5 * ((z - spectral_center) / max(band_width, 1e-8)) ** 2)
        profile[0] = 0.0
        crossing_drive = (1.0 - base[monitored]) / np.maximum(profile[monitored], 1e-12)
        critical = int(monitored[np.argmin(crossing_drive)])
        drive = float((1.0 - base[critical]) / profile[critical])
    else:
        raise ValueError(f"unknown regime: {regime}")
    schedule = np.empty((n_steps, len(values)), dtype=float)
    for index in range(n_steps):
        fraction = np.clip(
            (index - ramp_start) / max(transition_step - ramp_start, 1), 0.0, 1.2
        )
        schedule[index] = base + fraction * drive * profile
    if not np.isclose(schedule[transition_step, critical], 1.0, atol=1e-12):
        raise RuntimeError("critical persistence does not reach one at transition")
    if np.max(schedule[transition_step - 1, monitored]) >= 1.0:
        raise RuntimeError("a monitored mode crosses before the declared transition")
    return schedule, critical


def simulate_v3(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    n_steps: int,
    ramp_start: int,
    transition_step: int,
    regime: str,
    model: str,
    noise_sigma: float,
    rng: np.random.Generator,
    spectral_center: float = 0.35,
    band_width: float = 0.05,
    isolated_baseline_gap: float = 0.15,
    nonlinear_cubic: float = 0.01,
) -> V3Simulation:
    rho, critical = discrete_spectral_schedule(
        eigenvalues, n_steps, ramp_start, transition_step, regime,
        spectral_center, band_width, isolated_baseline_gap,
    )
    n = len(eigenvalues)
    x = np.zeros((n_steps, n), dtype=float)
    for index in range(1, n_steps):
        state = x[index - 1]
        linear = ((state @ eigenvectors) * rho[index - 1]) @ eigenvectors.T
        if model == "linear_var":
            drifted = linear
        elif model == "nonlinear_graph_map":
            drifted = linear - nonlinear_cubic * state**3
        else:
            raise ValueError(f"unknown model: {model}")
        x[index] = drifted + noise_sigma * rng.normal(size=n)
    return V3Simulation(
        x=x, rho=rho, critical_mode=critical,
        transition_step=transition_step, ramp_start=ramp_start,
        regime=regime, model=model,
    )
