from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib_cache").resolve()))

import numpy as np
import pandas as pd

from causal_gsews import compute_causal_scores
from dynamic_gsews import compute_dynamic_scores
from graphs import adjacency_matrix, eigensystem, laplacian
from rigorous_evaluate import evaluate_causal_run, paired_bootstrap, summarize_metrics
from run_identifiability_study import fit_diagonal_dynamics
from run_rigorous_synthetic import graph_controls, weighted_adjacency
from v3_simulate import simulate_v3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v3_pilot.json")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    master = np.random.default_rng(int(cfg["seed"]))
    metric_rows: list[dict] = []
    graph_rows: list[dict] = []
    truth_rows: list[dict] = []

    graph_train = int(cfg["graph_train_length"])
    graph_validation = int(cfg["graph_validation_length"])
    score_baseline = int(cfg["score_baseline_length"])
    ramp_length = int(cfg["ramp_length"])
    ramp_start = graph_train + graph_validation + score_baseline
    transition = ramp_start + ramp_length
    n_steps = transition + 1
    score_start = graph_train + graph_validation
    local_transition = transition - score_start
    local_baseline_end = score_baseline - int(cfg["baseline_guard"])

    total_runs = (
        len(cfg["models"]) * len(cfg["regimes"]) * len(cfg["graphs"])
        * int(cfg["n_seeds_per_graph"])
    )
    completed = 0
    for model in cfg["models"]:
        for regime in cfg["regimes"]:
            for graph_name in cfg["graphs"]:
                for replicate in range(int(cfg["n_seeds_per_graph"])):
                    seed = int(master.integers(0, 2**31 - 1))
                    rng = np.random.default_rng(seed)
                    adjacency = weighted_adjacency(
                        adjacency_matrix(graph_name, int(cfg["n_nodes"]), rng), rng
                    )
                    eigenvalues, eigenvectors = eigensystem(laplacian(adjacency))
                    simulation = simulate_v3(
                        eigenvalues, eigenvectors, n_steps, ramp_start, transition,
                        regime, model, float(cfg["noise_sigma"]), rng,
                        float(cfg["spectral_center"]), float(cfg["band_width"]),
                        float(cfg["isolated_baseline_gap"]),
                        float(cfg["nonlinear_cubic"]),
                    )
                    run_id = f"{model}:{regime}:{graph_name}:{replicate}:{seed}"
                    bases = [("TrueGraph", eigenvectors, True)]
                    bases.extend(
                        (name, basis, exclude_constant)
                        for name, basis, _, exclude_constant in graph_controls(
                            adjacency, rng, float(cfg["graph_perturbation"])
                        )
                    )
                    train = simulation.x[:graph_train]
                    validation = simulation.x[graph_train : graph_train + graph_validation]
                    fitted_bases = []
                    for name, basis, exclude_constant in bases:
                        fitted = fit_diagonal_dynamics(
                            train, validation, basis,
                            float(cfg["graph_fit_ridge_fraction"]), exclude_constant,
                        )
                        fitted_bases.append((name, basis, exclude_constant, fitted))
                    true_mse = next(item[3]["test_mse"] for item in fitted_bases if item[0] == "TrueGraph")
                    best_wrong_mse = min(item[3]["test_mse"] for item in fitted_bases if item[0] != "TrueGraph")
                    for name, _, _, fitted in fitted_bases:
                        graph_rows.append(
                            {
                                "run_id": run_id, "model": model, "regime": regime,
                                "graph": graph_name, "seed": seed, "basis": name,
                                "validation_mse": fitted["test_mse"],
                                "graph_selected": float(true_mse < best_wrong_mse),
                                "true_mse_advantage_over_best_wrong": best_wrong_mse - true_mse,
                            }
                        )

                    scoring_x = simulation.x[score_start:]
                    legacy = compute_causal_scores(
                        scoring_x, eigenvectors, adjacency,
                        int(cfg["window"]), int(cfg["stride"]), local_baseline_end,
                        int(cfg["trailing_smooth"]), int(cfg["trend_span"]), True,
                    )
                    dynamic_by_basis = []
                    for name, basis, exclude_constant, _ in fitted_bases:
                        scores = compute_dynamic_scores(
                            scoring_x, basis, int(cfg["window"]), int(cfg["stride"]),
                            local_baseline_end, float(cfg["score_ridge_fraction"]),
                            int(cfg["lag_steps"]), int(cfg["trailing_smooth"]),
                            int(cfg["trend_span"]), exclude_constant,
                            include_generalized=(
                                name == "TrueGraph"
                                and bool(cfg.get("include_generalized", True))
                            ),
                        )
                        dynamic_by_basis.append((name, scores))

                    for horizon in cfg["prediction_horizons"]:
                        local_pre_end = local_transition - int(horizon)
                        local_pre_start = max(
                            local_baseline_end,
                            local_pre_end - int(cfg["prediction_window"]),
                        )
                        global_pre_end = transition - int(horizon)
                        rho_at_horizon = simulation.rho[global_pre_end]
                        if regime == "isolated_mode":
                            critical_band = np.asarray([simulation.critical_mode])
                        else:
                            monitored = rho_at_horizon[1:]
                            tolerance = float(cfg["critical_band_rho_tolerance"])
                            critical_band = np.flatnonzero(
                                rho_at_horizon >= np.max(monitored) - tolerance
                            )
                            critical_band = critical_band[critical_band > 0]
                        metric_rows.extend(
                            evaluate_causal_run(
                                run_id, graph_name, f"{model}:{regime}", seed, legacy,
                                local_transition, local_baseline_end,
                                local_pre_start, local_pre_end,
                                simulation.critical_mode, eigenvectors, eigenvalues,
                                int(horizon), method_prefix="Legacy:",
                                critical_modes=critical_band,
                            )
                        )
                        for basis_name, scores in dynamic_by_basis:
                            prefix = "" if basis_name == "TrueGraph" else f"{basis_name}:"
                            metric_rows.extend(
                                evaluate_causal_run(
                                    run_id, graph_name, f"{model}:{regime}", seed, scores,
                                    local_transition, local_baseline_end,
                                    local_pre_start, local_pre_end,
                                    simulation.critical_mode, eigenvectors, eigenvalues,
                                    int(horizon), method_prefix=prefix,
                                    critical_modes=critical_band,
                                )
                            )
                        truth_rows.append(
                            {
                                "run_id": run_id, "model": model, "regime": regime,
                                "graph": graph_name, "seed": seed,
                                "prediction_horizon": int(horizon),
                                "critical_mode": simulation.critical_mode,
                                "critical_band_size": len(critical_band),
                                "critical_rho": float(rho_at_horizon[simulation.critical_mode]),
                                "second_rho": float(
                                    np.sort(np.delete(rho_at_horizon[1:], simulation.critical_mode - 1))[-1]
                                ),
                            }
                        )
                    completed += 1
                    if completed % 10 == 0:
                        print(f"completed {completed}/{total_runs}", flush=True)

    metrics = pd.DataFrame(metric_rows)
    graph_selection = pd.DataFrame(graph_rows)
    truth = pd.DataFrame(truth_rows)
    summary = summarize_metrics(metrics)
    comparisons = pd.concat(
        [
            paired_bootstrap(
                metrics, reference=reference,
                n_boot=int(cfg["bootstrap_replicates"]),
                seed=int(cfg["seed"]) + index,
            ).assign(estimator=reference)
            for index, reference in enumerate(["GraphDyn_Raw", "GraphDyn_Level"])
        ], ignore_index=True,
    )
    localization = localization_bootstrap(metrics, cfg)
    graph_summary = graph_selection_bootstrap(graph_selection, cfg)
    metrics.to_csv(output / "run_metrics.csv", index=False)
    graph_selection.to_csv(output / "graph_selection_runs.csv", index=False)
    truth.to_csv(output / "run_truth.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    comparisons.to_csv(output / "paired_auc_bootstrap.csv", index=False)
    localization.to_csv(output / "localization_bootstrap.csv", index=False)
    graph_summary.to_csv(output / "graph_selection_bootstrap.csv", index=False)
    write_report(output, cfg, summary, comparisons, localization, graph_summary)
    print(f"Wrote version-3 pilot to {output}")


def bootstrap_interval(values: np.ndarray, rng: np.random.Generator,
                       n_boot: int) -> tuple[float, float, float]:
    draws = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return float(values.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def graph_selection_bootstrap(frame: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    true = frame[frame.basis == "TrueGraph"]
    rng = np.random.default_rng(int(cfg["seed"]) + 501)
    rows = []
    for keys, group in true.groupby(["model", "regime", "graph"]):
        row = dict(zip(["model", "regime", "graph"], keys))
        for metric in ["graph_selected", "true_mse_advantage_over_best_wrong"]:
            mean, low, high = bootstrap_interval(
                group[metric].to_numpy(), rng, int(cfg["bootstrap_replicates"])
            )
            row[f"{metric}_mean"] = mean
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
        row["n_runs"] = len(group)
        rows.append(row)
    return pd.DataFrame(rows)


def localization_bootstrap(metrics: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    frame = metrics[metrics.method.isin(["GraphDyn_Raw", "GraphDyn_Level"])]
    rng = np.random.default_rng(int(cfg["seed"]) + 601)
    rows = []
    fields = [
        "interval_mode_correct", "localization_corr",
        "interval_subspace_overlap", "interval_band_recovery",
    ]
    for keys, group in frame.groupby(["model", "prediction_horizon", "method"]):
        model_regime, horizon, method = keys
        model, regime = model_regime.split(":", 1)
        row = {"model": model, "regime": regime,
               "prediction_horizon": int(horizon), "method": method,
               "n_runs": group.run_id.nunique()}
        for field in fields:
            values = group[field].dropna().to_numpy()
            mean, low, high = bootstrap_interval(
                values, rng, int(cfg["bootstrap_replicates"])
            )
            row[f"{field}_mean"] = mean
            row[f"{field}_ci95_low"] = low
            row[f"{field}_ci95_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(output: Path, cfg: dict, summary: pd.DataFrame,
                 comparisons: pd.DataFrame, localization: pd.DataFrame,
                 graph_summary: pd.DataFrame) -> None:
    lines = [
        "# Version-3 frozen-graph transition pilot", "",
        "## Configuration", "", "```json", json.dumps(cfg, indent=2), "```", "",
        "## Graph selection", "", graph_summary.to_csv(index=False).strip(), "",
        "## Transition summary", "", summary.to_csv(index=False).strip(), "",
        "## Paired AUC comparisons", "", comparisons.to_csv(index=False).strip(), "",
        "## Localization", "", localization.to_csv(index=False).strip(), "",
    ]
    (output / "REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
