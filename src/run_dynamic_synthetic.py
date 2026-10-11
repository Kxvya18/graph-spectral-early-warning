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
from run_rigorous_synthetic import graph_controls, simulate, weighted_adjacency


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dynamic_pilot.json")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    rng_master = np.random.default_rng(int(cfg["seed"]))
    metric_rows: list[dict] = []
    truth_rows: list[dict] = []
    specificity_rows: list[dict] = []

    for model in cfg["models"]:
        for graph_name in cfg["graphs"]:
            for replicate in range(int(cfg["n_seeds_per_graph"])):
                seed = int(rng_master.integers(0, 2**31 - 1))
                rng = np.random.default_rng(seed)
                adjacency = weighted_adjacency(
                    adjacency_matrix(graph_name, int(cfg["n_nodes"]), rng), rng
                )
                eigenvalues, eigenvectors = eigensystem(laplacian(adjacency))
                simulation = simulate(
                    model, eigenvalues, eigenvectors, cfg, rng,
                    heterogeneity=float(cfg.get("heterogeneity", 0.0)),
                )
                run_id = f"{model}:{graph_name}:{replicate}:{seed}"
                truth_rows.append(
                    {
                        "run_id": run_id, "model": model, "graph": graph_name,
                        "seed": seed, "critical_mode": simulation.critical_mode,
                        "critical_eigenvalue": float(eigenvalues[simulation.critical_mode]),
                    }
                )
                baseline_end = int(cfg["baseline_end"])
                legacy = compute_causal_scores(
                    simulation.x, eigenvectors, adjacency,
                    int(cfg["window"]), int(cfg["stride"]), baseline_end,
                    int(cfg["trailing_smooth"]), int(cfg["trend_span"]),
                    exclude_constant=True,
                )
                true_dynamic = compute_dynamic_scores(
                    simulation.x, eigenvectors,
                    int(cfg["window"]), int(cfg["stride"]), baseline_end,
                    float(cfg["ridge_fraction"]), int(cfg.get("lag_steps", 1)),
                    int(cfg["trailing_smooth"]),
                    int(cfg["trend_span"]), exclude_constant=True,
                )
                bases = [("TrueGraph", eigenvectors, true_dynamic)]
                for control_name, control_u, _, exclude_constant in graph_controls(
                    adjacency, rng, float(cfg.get("graph_perturbation", 0.20))
                ):
                    control = compute_dynamic_scores(
                        simulation.x, control_u,
                        int(cfg["window"]), int(cfg["stride"]), baseline_end,
                        float(cfg["ridge_fraction"]), int(cfg.get("lag_steps", 1)),
                        int(cfg["trailing_smooth"]),
                        int(cfg["trend_span"]), exclude_constant=exclude_constant,
                    )
                    bases.append((control_name, control_u, control))

                for horizon in cfg["prediction_horizons"]:
                    pre_end = int(cfg["transition_step"]) - int(horizon)
                    pre_start = max(baseline_end, pre_end - int(cfg["prediction_window"]))
                    alpha_at_horizon = simulation.alpha[pre_end]
                    monitored_alpha = alpha_at_horizon[1:]
                    tolerance = float(cfg.get("critical_band_alpha_tolerance", 0.05))
                    critical_band = np.flatnonzero(
                        alpha_at_horizon >= np.max(monitored_alpha) - tolerance
                    )
                    critical_band = critical_band[critical_band > 0]
                    metric_rows.extend(
                        evaluate_causal_run(
                            run_id, graph_name, model, seed, legacy,
                            int(cfg["transition_step"]), baseline_end, pre_start, pre_end,
                            simulation.critical_mode, eigenvectors, eigenvalues, int(horizon),
                            method_prefix="Legacy:",
                            critical_modes=critical_band,
                        )
                    )
                    for basis_name, _, scores in bases:
                        prefix = "" if basis_name == "TrueGraph" else f"{basis_name}:"
                        metric_rows.extend(
                            evaluate_causal_run(
                                run_id, graph_name, model, seed, scores,
                                int(cfg["transition_step"]), baseline_end,
                                pre_start, pre_end, simulation.critical_mode,
                                eigenvectors, eigenvalues, int(horizon),
                                method_prefix=prefix,
                                critical_modes=critical_band,
                            )
                        )
                        baseline = scores.ends <= baseline_end
                        pretransition = (scores.ends > pre_start) & (scores.ends <= pre_end)
                        specificity_rows.append(
                            {
                                "run_id": run_id, "model": model, "graph": graph_name,
                                "seed": seed, "prediction_horizon": int(horizon),
                                "basis": basis_name,
                                "baseline_graph_fit": float(np.mean(scores.graph_fit[baseline])),
                                "pretransition_graph_fit": float(np.mean(scores.graph_fit[pretransition])),
                                "baseline_concentration": float(np.mean(scores.graph_concentration[baseline])),
                                "pretransition_concentration": float(np.mean(scores.graph_concentration[pretransition])),
                                "baseline_diagonality": float(np.mean(scores.graph_diagonality[baseline])),
                                "pretransition_diagonality": float(np.mean(scores.graph_diagonality[pretransition])),
                            }
                        )

    metrics = pd.DataFrame(metric_rows)
    truth = pd.DataFrame(truth_rows)
    specificity = pd.DataFrame(specificity_rows)
    summary = summarize_metrics(metrics)
    references = ["GraphDyn_Raw", "GraphDyn_Level", "SlowMode_Level", "GraphAligned_Level"]
    comparisons = pd.concat(
        [
            paired_bootstrap(
                metrics, reference=reference,
                n_boot=int(cfg.get("bootstrap_replicates", 4000)),
                seed=int(cfg["seed"]) + index,
            ).assign(estimator=reference)
            for index, reference in enumerate(references)
        ],
        ignore_index=True,
    )
    specificity_comparisons = compare_graph_specificity(
        specificity, int(cfg.get("bootstrap_replicates", 4000)), int(cfg["seed"])
    )
    metrics.to_csv(output / "run_metrics.csv", index=False)
    truth.to_csv(output / "run_truth.csv", index=False)
    specificity.to_csv(output / "graph_specificity.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    comparisons.to_csv(output / "paired_auc_bootstrap.csv", index=False)
    specificity_comparisons.to_csv(output / "paired_graph_specificity.csv", index=False)
    write_report(output, cfg, summary, comparisons, specificity_comparisons)
    print(summary.to_string(index=False))
    print(f"\nWrote dynamic-estimator pilot to {output}")


def compare_graph_specificity(frame: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    endpoints = [
        "baseline_graph_fit", "pretransition_graph_fit",
        "baseline_concentration", "pretransition_concentration",
        "baseline_diagonality", "pretransition_diagonality",
    ]
    for (model, horizon), group in frame.groupby(["model", "prediction_horizon"]):
        for endpoint in endpoints:
            wide = group.pivot(index="run_id", columns="basis", values=endpoint)
            if "TrueGraph" not in wide:
                continue
            for comparator in [c for c in wide.columns if c != "TrueGraph"]:
                pair = wide[["TrueGraph", comparator]].dropna()
                difference = (pair["TrueGraph"] - pair[comparator]).to_numpy()
                draws = rng.choice(difference, size=(n_boot, len(difference)), replace=True).mean(axis=1)
                rows.append(
                    {
                        "model": model, "prediction_horizon": int(horizon),
                        "endpoint": endpoint, "comparator": comparator,
                        "n_pairs": len(difference), "difference_mean": difference.mean(),
                        "ci95_low": np.quantile(draws, 0.025),
                        "ci95_high": np.quantile(draws, 0.975),
                    }
                )
    return pd.DataFrame(rows)


def write_report(output: Path, cfg: dict, summary: pd.DataFrame,
                 comparisons: pd.DataFrame, specificity: pd.DataFrame) -> None:
    lines = [
        "# Version-2 covariance-lag graph-dynamics pilot", "",
        "This is a full prespecified pilot, not a reduced smoke experiment.",
        "The legacy estimator is retained as a matched benchmark.", "",
        "## Configuration", "", "```json", json.dumps(cfg, indent=2), "```", "",
        "## Summary", "", summary.to_csv(index=False).strip(), "",
        "## Paired AUC comparisons", "", comparisons.to_csv(index=False).strip(), "",
        "## Direct graph-specificity endpoints", "",
        specificity.to_csv(index=False).strip(), "",
    ]
    (output / "REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
