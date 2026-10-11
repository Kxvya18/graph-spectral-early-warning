from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from graphs import adjacency_matrix, eigensystem, laplacian
from recovery_rate import (
    compute_recovery_trajectory,
    phenotype_scores,
    realign_trajectory,
)
from recovery_simulate import simulate_recovery_scenario
from rigorous_simulate import perturb_adjacency
from run_chbmit_locked import graph_bases
from run_rigorous_synthetic import weighted_adjacency


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/recovery_phenotype_synthetic.json")
    args = parser.parse_args()
    config_path = Path(args.config)
    cfg = json.loads(config_path.read_text())
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)

    master = np.random.default_rng(int(cfg["seed"]))
    rows: list[dict] = []
    baseline_length = int(cfg["baseline_length"])
    transition = baseline_length + int(cfg["ramp_length"])
    total = (
        len(cfg["graphs"]) * len(cfg["scenarios"])
        * int(cfg["n_seeds_per_condition"])
    )
    completed = 0
    for graph_name in cfg["graphs"]:
        for scenario in cfg["scenarios"]:
            for replicate in range(int(cfg["n_seeds_per_condition"])):
                seed = int(master.integers(0, 2**31 - 1))
                rng = np.random.default_rng(seed)
                adjacency = weighted_adjacency(
                    adjacency_matrix(graph_name, int(cfg["n_nodes"]), rng), rng
                )
                eigenvalues, eigenvectors = eigensystem(laplacian(adjacency))
                simulation = simulate_recovery_scenario(
                    eigenvectors,
                    eigenvalues,
                    transition + 1,
                    baseline_length,
                    transition,
                    scenario,
                    rng,
                    float(cfg["noise_sigma"]),
                    float(cfg["critical_baseline_rho"]),
                    float(cfg["heterogeneous_noise_ratio"]),
                )
                true = compute_recovery_trajectory(
                    simulation.x,
                    eigenvectors,
                    eigenvalues,
                    int(cfg["window"]),
                    int(cfg["stride"]),
                    int(cfg["lag_steps"]),
                    float(cfg["ridge_fraction"]),
                    int(cfg["trailing_smooth"]),
                    float(cfg["isolated_gap"]),
                    float(cfg["low_frequency_fraction"]),
                    True,
                )
                control_trajectories: dict[str, object] = {}
                for name, basis, control_adjacency, exclude_constant in graph_bases(
                    adjacency, eigenvectors, np.random.default_rng(seed + 10_000)
                ):
                    if name == "TrueGraph":
                        continue
                    control_values = None
                    if name != "IdentityBasis":
                        control_values, _ = eigensystem(laplacian(control_adjacency))
                    control_trajectories[name] = realign_trajectory(
                        true,
                        basis,
                        control_values,
                        float(cfg["low_frequency_fraction"]),
                        exclude_constant,
                        int(cfg["trailing_smooth"]),
                    )
                perturbed = perturb_adjacency(
                    adjacency, float(cfg["graph_perturbation"]), np.random.default_rng(seed + 20_000)
                )
                perturbed_values, perturbed_vectors = eigensystem(laplacian(perturbed))
                control_trajectories["PerturbedGraph"] = realign_trajectory(
                    true,
                    perturbed_vectors,
                    perturbed_values,
                    float(cfg["low_frequency_fraction"]),
                    True,
                    int(cfg["trailing_smooth"]),
                )

                baseline = true.ends <= baseline_length - int(cfg["baseline_guard"])
                positive_end = transition - int(cfg["prediction_horizon"])
                positive_start = positive_end - int(cfg["prediction_window"])
                positive = (true.ends > positive_start) & (true.ends <= positive_end)
                if baseline.sum() < 3 or positive.sum() < 3:
                    raise RuntimeError("synthetic masks contain too few windows")
                core_names = ["IdentityBasis", "RandomGraph", "PermutedGraph"]
                scores = phenotype_scores(
                    true, [control_trajectories[name] for name in core_names], baseline
                )
                for name, control in control_trajectories.items():
                    scores[f"{name}:Joint"] = scores["RecoveryOnly"] * control.concentration
                scores["TrueGraph:Joint"] = scores["RecoveryOnly"] * true.concentration
                labels = np.r_[np.zeros(baseline.sum(), dtype=int), np.ones(positive.sum(), dtype=int)]
                for method, values in scores.items():
                    selected = np.r_[values[baseline], values[positive]]
                    rows.append(
                        base_row(
                            graph_name, scenario, replicate, seed, method,
                            safe_auc(labels, selected), simulation, true, positive,
                            control_trajectories,
                        )
                    )
                completed += 1
                if completed % 25 == 0 or completed == total:
                    print(f"completed {completed}/{total}", flush=True)

    metrics = pd.DataFrame(rows)
    summary = summarize(metrics, cfg)
    gates = evaluate_gates(metrics, cfg)
    metrics.to_csv(output / "run_metrics.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    gates.to_csv(output / "mechanistic_gate.csv", index=False)
    write_report(output, cfg, summary, gates)
    print(gates.to_string(index=False))


def base_row(graph: str, scenario: str, replicate: int, seed: int, method: str,
             auc: float, simulation, true, positive: np.ndarray,
             controls: dict[str, object]) -> dict:
    wrong_max = np.max(
        np.vstack([
            controls[name].concentration
            for name in ["IdentityBasis", "RandomGraph", "PermutedGraph"]
        ]),
        axis=0,
    )
    if simulation.critical_graph_mode is None:
        mode_recovery = np.nan
    else:
        eligible = positive & true.isolated
        mode_recovery = float(np.mean(
            true.dominant_graph_mode[eligible] == simulation.critical_graph_mode
        )) if np.any(eligible) else 0.0
    direction_corr = float(np.mean(np.abs(
        true.direction[positive] @ simulation.critical_direction
    )))
    return {
        "run_id": f"{graph}:{scenario}:{replicate}:{seed}",
        "graph": graph,
        "scenario": scenario,
        "replicate": replicate,
        "seed": seed,
        "method": method,
        "auc": auc,
        "true_alignment_advantage": float(np.mean(
            true.concentration[positive] - wrong_max[positive]
        )),
        "perturbed_alignment_advantage": float(np.mean(
            true.concentration[positive]
            - controls["PerturbedGraph"].concentration[positive]
        )),
        "mode_recovery": mode_recovery,
        "direction_correlation": direction_corr,
        "isolated_fraction": float(np.mean(true.isolated[positive])),
        "constant_mass": float(np.mean(true.constant_mass[positive])),
    }


def safe_auc(labels: np.ndarray, values: np.ndarray) -> float:
    if np.std(values) < 1e-12:
        return 0.5
    return float(roc_auc_score(labels, values))


def summarize(metrics: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rng = np.random.default_rng(int(cfg["seed"]) + 900)
    rows = []
    fields = [
        "auc", "true_alignment_advantage", "perturbed_alignment_advantage",
        "mode_recovery", "direction_correlation", "isolated_fraction", "constant_mass",
    ]
    for keys, frame in metrics.groupby(["scenario", "graph", "method"]):
        row = dict(zip(["scenario", "graph", "method"], keys))
        for field in fields:
            values = frame[field].dropna().to_numpy()
            if not len(values):
                row[f"{field}_mean"] = np.nan
                row[f"{field}_ci95_low"] = np.nan
                row[f"{field}_ci95_high"] = np.nan
                continue
            draws = rng.choice(
                values,
                size=(int(cfg["bootstrap_replicates"]), len(values)),
                replace=True,
            ).mean(axis=1)
            row[f"{field}_mean"] = float(values.mean())
            row[f"{field}_ci95_low"] = float(np.quantile(draws, 0.025))
            row[f"{field}_ci95_high"] = float(np.quantile(draws, 0.975))
        row["n_runs"] = frame.run_id.nunique()
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_gates(metrics: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    gate = cfg["mechanistic_gate"]
    primary = metrics[metrics.method == "RecoveryPhenotype"]
    rows = []
    for graph, frame in primary.groupby("graph"):
        aligned = frame[frame.scenario == "aligned"]
        heterogeneous = frame[frame.scenario == "heterogeneous_noise"]
        falsification = frame[frame.scenario.isin([
            "non_graph_slowing", "common_mode_artifact", "localized_instability"
        ])]
        row = {
            "graph": graph,
            "aligned_phenotype_auc": float(aligned.auc.mean()),
            "heterogeneous_phenotype_auc": float(heterogeneous.auc.mean()),
            "aligned_mode_recovery": float(aligned.mode_recovery.mean()),
            "aligned_alignment_advantage": float(aligned.true_alignment_advantage.mean()),
            "maximum_falsification_auc": float(
                falsification.groupby("scenario").auc.mean().max()
            ),
        }
        row["pass"] = bool(
            row["aligned_phenotype_auc"] >= float(gate["minimum_aligned_phenotype_auc"])
            and row["heterogeneous_phenotype_auc"] >= float(gate["minimum_heterogeneous_phenotype_auc"])
            and row["aligned_mode_recovery"] >= float(gate["minimum_mode_recovery"])
            and row["aligned_alignment_advantage"] >= float(gate["minimum_true_alignment_advantage"])
            and row["maximum_falsification_auc"] <= float(gate["maximum_falsification_phenotype_auc"])
        )
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(output: Path, cfg: dict, summary: pd.DataFrame,
                 gates: pd.DataFrame) -> None:
    decision = bool(len(gates) and gates["pass"].all())
    lines = [
        "# Graph-aligned recovery-rate synthetic study", "",
        f"**Frozen mechanistic gate: {'PASS' if decision else 'FAIL'}.**", "",
        "A failure is retained as a scientific result and does not authorize",
        "post-hoc changes to the frozen primary gate.", "",
        "## Configuration", "", "```json", json.dumps(cfg, indent=2), "```", "",
        "## Mechanistic gate", "", gates.to_csv(index=False).strip(), "",
        "## Summary", "", summary.to_csv(index=False).strip(), "",
    ]
    (output / "REPORT.md").write_text("\n".join(lines))
    (output / "RECOVERY_GATE_DECISION.md").write_text("\n".join(lines[:12]))


if __name__ == "__main__":
    main()
