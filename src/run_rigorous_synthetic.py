from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib_cache").resolve()))

import numpy as np
import pandas as pd

from causal_gsews import compute_causal_modal_scores, compute_causal_scores
from graphs import adjacency_matrix, eigensystem, laplacian
from rigorous_evaluate import evaluate_causal_run, paired_bootstrap, summarize_metrics
from rigorous_simulate import (
    perturb_adjacency,
    simulate_graph_commuting_ou,
    simulate_graph_pattern_normal_form,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rigorous_smoke.json")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    master = np.random.default_rng(int(cfg["seed"]))
    rows: list[dict] = []
    truth_rows: list[dict] = []
    for model in cfg["models"]:
        for graph_name in cfg["graphs"]:
            for replicate in range(int(cfg["n_seeds_per_graph"])):
                seed = int(master.integers(0, 2**31 - 1))
                rng = np.random.default_rng(seed)
                adjacency = weighted_adjacency(
                    adjacency_matrix(graph_name, int(cfg["n_nodes"]), rng), rng
                )
                evals, evecs = eigensystem(laplacian(adjacency))
                simulation = simulate(
                    model, evals, evecs, cfg, rng,
                    heterogeneity=float(cfg.get("heterogeneity", 0.0)),
                )
                run_id = f"{model}:{graph_name}:{replicate}:{seed}"
                assert abs(simulation.alpha[int(cfg["transition_step"]), simulation.critical_mode]) < 1e-10
                assert np.max(simulation.alpha[int(cfg["transition_step"]) - 1, 1:]) < 0
                truth_rows.append(
                    {
                        "run_id": run_id,
                        "graph": graph_name,
                        "model": model,
                        "seed": seed,
                        "critical_mode": simulation.critical_mode,
                        "critical_eigenvalue": float(evals[simulation.critical_mode]),
                        "spectral_center": float(cfg["spectral_center"]),
                    }
                )

                baseline_end = int(cfg["baseline_end"])
                primary = compute_causal_scores(
                    simulation.x, evecs, adjacency,
                    int(cfg["window"]), int(cfg["stride"]), baseline_end,
                    int(cfg["trailing_smooth"]), int(cfg["trend_span"]),
                    exclude_constant=True,
                )
                horizons = cfg.get("prediction_horizons", [cfg.get("prediction_horizon", 0)])
                for horizon in horizons:
                    pre_end = int(cfg["transition_step"]) - int(horizon)
                    pre_start = max(baseline_end, pre_end - int(cfg["prediction_window"]))
                    horizon_rows = evaluate_causal_run(
                        run_id, graph_name, model, seed, primary,
                        int(cfg["transition_step"]), baseline_end,
                        pre_start, pre_end,
                        simulation.critical_mode, evecs, evals, int(horizon),
                    )
                    rows.extend(horizon_rows)

                for control_name, control_u, control_a, exclude_constant in graph_controls(
                    adjacency, rng, float(cfg.get("graph_perturbation", 0.20))
                ):
                    control_scores = compute_causal_modal_scores(
                        simulation.x, control_u,
                        int(cfg["window"]), int(cfg["stride"]), baseline_end,
                        int(cfg["trailing_smooth"]), int(cfg["trend_span"]),
                        exclude_constant=exclude_constant,
                    )
                    for horizon in horizons:
                        pre_end = int(cfg["transition_step"]) - int(horizon)
                        pre_start = max(baseline_end, pre_end - int(cfg["prediction_window"]))
                        control_rows = evaluate_causal_run(
                            run_id, graph_name, model, seed, control_scores,
                            int(cfg["transition_step"]), baseline_end,
                            pre_start, pre_end,
                            simulation.critical_mode, evecs, evals, int(horizon),
                            method_prefix=f"{control_name}:",
                        )
                        rows.extend(control_rows)

    metrics = pd.DataFrame(rows)
    truth = pd.DataFrame(truth_rows)
    summary = summarize_metrics(metrics)
    paired = paired_bootstrap(metrics, n_boot=int(cfg.get("bootstrap_replicates", 4000)))
    paired_level = paired_bootstrap(
        metrics, reference="GSEWS_Level",
        n_boot=int(cfg.get("bootstrap_replicates", 4000)),
    )
    metrics.to_csv(out / "run_metrics.csv", index=False)
    truth.to_csv(out / "run_truth.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    paired.to_csv(out / "paired_auc_bootstrap.csv", index=False)
    paired_level.to_csv(out / "paired_level_auc_bootstrap.csv", index=False)
    write_report(out, cfg, summary, paired)
    print(summary.to_string(index=False))
    print(f"\nWrote rigorous results to {out}")


def simulate(model: str, evals: np.ndarray, evecs: np.ndarray, cfg: dict, rng, heterogeneity: float):
    common = dict(
        eigenvalues=evals,
        eigenvectors=evecs,
        n_steps=int(cfg["n_steps"]),
        dt=float(cfg["dt"]),
        transition_step=int(cfg["transition_step"]),
        ramp_start=int(cfg["ramp_start"]),
        noise_sigma=float(cfg["noise_sigma"]),
        rng=rng,
        spectral_center=float(cfg["spectral_center"]),
        heterogeneity=heterogeneity,
    )
    if model == "graph_commuting_ou":
        return simulate_graph_commuting_ou(**common)
    if model == "graph_pattern_normal_form":
        return simulate_graph_pattern_normal_form(
            **common, cubic_strength=float(cfg.get("cubic_strength", 1.0))
        )
    raise ValueError(f"unknown model: {model}")


def weighted_adjacency(adjacency: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Break exact spectral degeneracies with prespecified mild edge weights."""
    mask = np.triu(adjacency > 0, 1)
    weights = np.zeros_like(adjacency, dtype=float)
    weights[mask] = rng.uniform(0.9, 1.1, size=int(mask.sum()))
    weights = weights + weights.T
    return np.where(adjacency > 0, weights, 0.0)


def graph_controls(adjacency: np.ndarray, rng: np.random.Generator, perturbation: float):
    n = len(adjacency)
    identity_u = np.eye(n)
    identity_a = np.zeros_like(adjacency)
    yield "IdentityBasis", identity_u, identity_a, False

    density = int(np.triu(adjacency > 0, 1).sum())
    candidates = np.array([(i, j) for i in range(n) for j in range(i + 1, n)], dtype=int)
    chosen = rng.choice(len(candidates), size=density, replace=False)
    random_a = np.zeros_like(adjacency)
    for i, j in candidates[chosen]:
        random_a[i, j] = random_a[j, i] = 1.0
    _, random_u = eigensystem(laplacian(random_a))
    yield "RandomGraph", random_u, random_a, True

    permutation = rng.permutation(n)
    permuted_a = adjacency[np.ix_(permutation, permutation)]
    _, permuted_u = eigensystem(laplacian(permuted_a))
    yield "PermutedGraph", permuted_u, permuted_a, True

    perturbed_a = perturb_adjacency(adjacency, perturbation, rng)
    _, perturbed_u = eigensystem(laplacian(perturbed_a))
    yield "PerturbedGraph", perturbed_u, perturbed_a, True


def write_report(out: Path, cfg: dict, summary: pd.DataFrame, paired: pd.DataFrame) -> None:
    lines = [
        "# Rigorous causal GSEWS synthetic benchmark",
        "",
        "This benchmark uses equal isotropic node noise, a graph-commuting drift",
        "operator, trailing-only processing, and window-end eligibility.",
        "",
        "## Locked configuration",
        "",
        "```json",
        json.dumps(cfg, indent=2),
        "```",
        "",
        "## Summary",
        "",
        summary.to_csv(index=False).strip(),
        "",
        "## Paired AUC differences (GSEWS minus comparator)",
        "",
        paired.to_csv(index=False).strip(),
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
