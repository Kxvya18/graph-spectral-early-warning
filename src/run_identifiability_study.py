from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import eigh

from graphs import adjacency_matrix, eigensystem, laplacian
from run_rigorous_synthetic import graph_controls, weighted_adjacency


def modal_dynamics(eigenvalues: np.ndarray, critical_gap: float,
                   spectral_center: float = 0.35) -> tuple[np.ndarray, int]:
    """Construct a graph-commuting, uniquely ordered stationary VAR spectrum."""
    values = np.asarray(eigenvalues, dtype=float)
    z = values / max(float(values[-1]), 1e-12)
    critical = int(np.argmin(np.abs(z[1:] - spectral_center)) + 1)
    rho = 0.55 + 0.14 * (1.0 - z)
    rho[0] = 0.50
    competitors = np.delete(rho, critical)
    rho[critical] = min(float(np.max(competitors) + critical_gap), 0.97)
    return rho, critical


def population_metrics(eigenvectors: np.ndarray, rho: np.ndarray,
                       bases: list[tuple[str, np.ndarray]]) -> list[dict]:
    """Exact graph identifiability metrics with no finite-sample error."""
    u = eigenvectors
    operator = (u * rho) @ u.T
    modal_variance = 1.0 / np.maximum(1.0 - rho**2, 1e-12)
    covariance = (u * modal_variance) @ u.T
    lag_covariance = operator @ covariance
    oracle_noise = len(rho)
    rows = []
    for name, basis in bases:
        transformed_operator = basis.T @ operator @ basis
        diagonal_operator = (basis * np.diag(transformed_operator)) @ basis.T
        difference = operator - diagonal_operator
        misspecification = float(np.trace(difference @ covariance @ difference.T))
        cross = basis.T @ lag_covariance @ basis
        diagonal_energy = float(np.sum(np.diag(cross) ** 2))
        total_energy = float(np.sum(cross**2))
        rows.append(
            {
                "basis": name,
                "operator_residual_ratio": float(
                    np.sum(difference**2) / max(np.sum(operator**2), 1e-12)
                ),
                "expected_test_mse": oracle_noise + misspecification,
                "crosscov_diagonality": diagonal_energy / max(total_energy, 1e-12),
            }
        )
    return rows


def simulate_stationary_modal(rho: np.ndarray, burnin: int, train_length: int,
                              test_length: int, observation_noise_ratio: float,
                              eigenvectors: np.ndarray,
                              rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    total = burnin + train_length + test_length
    y = np.zeros((total, len(rho)), dtype=float)
    innovations = rng.normal(size=y.shape)
    for index in range(1, total):
        y[index] = rho * y[index - 1] + innovations[index]
    x = y @ eigenvectors.T
    if observation_noise_ratio > 0:
        channel_scale = np.std(x[burnin : burnin + train_length], axis=0, ddof=1)
        x += rng.normal(size=x.shape) * (observation_noise_ratio * channel_scale)
    train = x[burnin : burnin + train_length]
    test = x[burnin + train_length :]
    return train, test


def fit_diagonal_dynamics(train: np.ndarray, test: np.ndarray, basis: np.ndarray,
                          ridge_fraction: float, exclude_constant: bool) -> dict:
    mean = train.mean(axis=0, keepdims=True)
    train_modal = (train - mean) @ basis
    test_modal = (test - mean) @ basis
    y0, y1 = train_modal[:-1], train_modal[1:]
    scale = max(float(np.mean(y0**2)), 1e-12)
    ridge = ridge_fraction * scale * len(y0)
    rho_hat = np.sum(y0 * y1, axis=0) / (np.sum(y0**2, axis=0) + ridge)
    t0, t1 = test_modal[:-1], test_modal[1:]
    test_mse = float(np.mean((t1 - t0 * rho_hat[None, :]) ** 2))
    modes = np.arange(len(rho_hat))
    if exclude_constant and len(modes) > 1:
        modes = modes[1:]
    dominant = int(modes[np.argmax(rho_hat[modes])])
    return {"rho_hat": rho_hat, "dominant_mode": dominant, "test_mse": test_mse}


def generalized_slow_direction(train: np.ndarray, ridge_fraction: float) -> tuple[float, np.ndarray]:
    centered = train - train.mean(axis=0, keepdims=True)
    x0, x1 = centered[:-1], centered[1:]
    covariance = (x0.T @ x0) / len(x0)
    cross = (x0.T @ x1) / len(x0)
    lag_covariance = 0.5 * (cross + cross.T)
    scale = max(float(np.trace(covariance) / train.shape[1]), 1e-12)
    values, vectors = eigh(
        lag_covariance,
        covariance + ridge_fraction * scale * np.eye(train.shape[1]),
        check_finite=False,
    )
    vector = vectors[:, -1]
    vector /= max(float(np.linalg.norm(vector)), 1e-12)
    return float(values[-1]), vector


def run_condition(graph_name: str, n_nodes: int, critical_gap: float,
                  train_length: int, observation_noise_ratio: float,
                  replicate: int, cfg: dict) -> tuple[list[dict], list[dict]]:
    graph_index = cfg["graphs"].index(graph_name)
    graph_seed = int(cfg["seed"]) + 1_000_003 * graph_index + replicate
    graph_rng = np.random.default_rng(graph_seed)
    adjacency = weighted_adjacency(adjacency_matrix(graph_name, n_nodes, graph_rng), graph_rng)
    eigenvalues, eigenvectors = eigensystem(laplacian(adjacency))
    rho, critical = modal_dynamics(eigenvalues, critical_gap, float(cfg["spectral_center"]))
    control_rng = np.random.default_rng(graph_seed + 400_000_037)
    bases = [("TrueGraph", eigenvectors, True)]
    bases.extend(
        (name, basis, exclude_constant)
        for name, basis, _, exclude_constant in graph_controls(
            adjacency, control_rng, float(cfg["graph_perturbation"])
        )
    )
    population = population_metrics(
        eigenvectors, rho, [(name, basis) for name, basis, _ in bases]
    )
    data_seed = (
        graph_seed + int(round(critical_gap * 1_000_000))
        + int(round(observation_noise_ratio * 10_000))
    )
    train, test = simulate_stationary_modal(
        rho, int(cfg["burnin"]), train_length, int(cfg["test_length"]),
        observation_noise_ratio, eigenvectors, np.random.default_rng(data_seed),
    )
    finite = []
    for name, basis, exclude_constant in bases:
        fitted = fit_diagonal_dynamics(
            train, test, basis, float(cfg["ridge_fraction"]), exclude_constant
        )
        finite.append(
            {
                "basis": name, "test_mse": fitted["test_mse"],
                "estimated_max_rho": float(np.max(fitted["rho_hat"])),
                "selected_mode": fitted["dominant_mode"],
            }
        )
    _, direction = generalized_slow_direction(train, float(cfg["ridge_fraction"]))
    true_vector = eigenvectors[:, critical]
    vector_cosine_squared = float((direction @ true_vector) ** 2)
    true_row = next(row for row in finite if row["basis"] == "TrueGraph")
    wrong_mse = min(row["test_mse"] for row in finite if row["basis"] != "TrueGraph")
    graph_selected = float(true_row["test_mse"] < wrong_mse)
    exact_recovery = float(true_row["selected_mode"] == critical)
    common = {
        "graph": graph_name, "n_nodes": n_nodes, "critical_gap": critical_gap,
        "train_length": train_length,
        "observation_noise_ratio": observation_noise_ratio,
        "replicate": replicate, "graph_seed": graph_seed,
        "critical_mode": critical, "true_critical_rho": float(rho[critical]),
        "second_rho": float(np.sort(np.delete(rho, critical))[-1]),
        "graph_selected": graph_selected, "exact_mode_recovery": exact_recovery,
        "generalized_vector_cosine_squared": vector_cosine_squared,
        "heldout_mse_advantage_over_best_wrong": wrong_mse - true_row["test_mse"],
    }
    finite_rows = [{**common, **row} for row in finite]
    population_rows = [{**common, **row} for row in population]
    return finite_rows, population_rows


def bootstrap_summary(runs: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    true = runs[runs.basis == "TrueGraph"].copy()
    group_columns = [
        "graph", "n_nodes", "critical_gap", "train_length", "observation_noise_ratio"
    ]
    metrics = [
        "graph_selected", "exact_mode_recovery", "generalized_vector_cosine_squared",
        "heldout_mse_advantage_over_best_wrong",
    ]
    rng = np.random.default_rng(int(cfg["seed"]) + 91)
    rows = []
    for keys, frame in true.groupby(group_columns):
        row = dict(zip(group_columns, keys))
        row["n_seeds"] = len(frame)
        for metric in metrics:
            values = frame[metric].to_numpy()
            draws = rng.choice(
                values,
                size=(int(cfg["bootstrap_replicates"]), len(values)), replace=True,
            ).mean(axis=1)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_ci95_low"] = float(np.quantile(draws, 0.025))
            row[f"{metric}_ci95_high"] = float(np.quantile(draws, 0.975))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/identifiability_primary.json")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    finite_rows: list[dict] = []
    population_rows: list[dict] = []
    conditions = []
    for graph in cfg["graphs"]:
        for n_nodes in cfg["dimensions"]:
            for gap in cfg["critical_gaps"]:
                for length in cfg["train_lengths"]:
                    for noise in cfg["observation_noise_ratios"]:
                        conditions.append((graph, n_nodes, gap, length, noise))
    total = len(conditions) * int(cfg["n_seeds_per_condition"])
    completed = 0
    for graph, n_nodes, gap, length, noise in conditions:
        for replicate in range(int(cfg["n_seeds_per_condition"])):
            finite, population = run_condition(
                graph, int(n_nodes), float(gap), int(length), float(noise),
                replicate, cfg,
            )
            finite_rows.extend(finite)
            population_rows.extend(population)
            completed += 1
            if completed % 250 == 0:
                print(f"completed {completed}/{total}", flush=True)
    finite = pd.DataFrame(finite_rows)
    population = pd.DataFrame(population_rows)
    summary = bootstrap_summary(finite, cfg)
    finite.to_csv(output / "finite_sample_runs.csv", index=False)
    population.to_csv(output / "population_oracle.csv", index=False)
    summary.to_csv(output / "identifiability_summary.csv", index=False)
    write_report(output, cfg, summary)
    print(f"Wrote identifiability study to {output}")


def write_report(output: Path, cfg: dict, summary: pd.DataFrame) -> None:
    gate = cfg["mechanistic_gate"]
    passed = summary[
        (summary.graph_selected_ci95_low >= float(gate["minimum_graph_selection_ci_low"]))
        & (summary.exact_mode_recovery_ci95_low >= float(gate["minimum_mode_recovery_ci_low"]))
        & (summary.generalized_vector_cosine_squared_ci95_low >= float(gate["minimum_vector_cosine_ci_low"]))
        & (summary.heldout_mse_advantage_over_best_wrong_ci95_low > 0)
    ]
    lines = [
        "# Graph-dynamics identifiability study", "",
        "## Frozen configuration", "", "```json", json.dumps(cfg, indent=2), "```", "",
        f"Conditions passing every mechanistic criterion: {len(passed)}/{len(summary)}", "",
        "## Passing conditions", "",
        (passed.to_csv(index=False).strip() if len(passed) else "None."), "",
        "## Complete summary", "", summary.to_csv(index=False).strip(), "",
    ]
    (output / "REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
