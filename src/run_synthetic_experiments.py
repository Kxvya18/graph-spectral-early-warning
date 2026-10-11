from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib_cache").resolve()))

import numpy as np

from evaluate import evaluate_scores, rows_to_frame
from graphs import adjacency_matrix, eigensystem, laplacian
from gsews import compute_gsews
from make_figures import plot_representative, plot_summary
from simulate import simulate_fhn_reduced, simulate_modal_ou


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synthetic_gsews.json")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)

    rng_master = np.random.default_rng(int(cfg["seed"]))
    rows = []
    representative_done = False

    for graph_name in cfg["graphs"]:
        for model_name in cfg["models"]:
            for trial in range(int(cfg["n_trials"])):
                rng = np.random.default_rng(int(rng_master.integers(0, 2**31 - 1)))
                a = adjacency_matrix(graph_name, int(cfg["n_nodes"]), rng)
                l = laplacian(a, normalized=False)
                evals, evecs = eigensystem(l)
                true_mode = choose_true_mode(graph_name, evals, rng)

                if model_name == "modal_ou":
                    sim = simulate_modal_ou(
                        evals,
                        evecs,
                        int(cfg["n_steps"]),
                        float(cfg["dt"]),
                        int(cfg["transition_step"]),
                        true_mode,
                        float(cfg["noise_sigma"]),
                        rng,
                    )
                elif model_name == "fhn_reduced":
                    sim = simulate_fhn_reduced(
                        l,
                        evals,
                        evecs,
                        int(cfg["n_steps"]),
                        float(cfg["dt"]),
                        int(cfg["transition_step"]),
                        true_mode,
                        float(cfg["noise_sigma"]),
                        rng,
                    )
                else:
                    raise ValueError(f"unknown model: {model_name}")

                ews = compute_gsews(sim.x, evecs, int(cfg["window"]), int(cfg["stride"]), adjacency=a)
                rows.extend(
                    evaluate_scores(
                        graph_name,
                        model_name,
                        trial,
                        ews.centers,
                        sim.transition_step,
                        sim.true_mode,
                        sim.true_localization,
                        evecs,
                        ews.global_score,
                        ews.dominant_mode,
                        {
                            "NodeAvg": ews.node_avg_score,
                            "NodeMax": ews.node_max_score,
                            "PCA": ews.pca_score,
                            "Energy": ews.energy_score,
                            "GraphSmooth": ews.graph_smoothness_score,
                            "EdgeCorr": ews.edge_corr_score,
                        },
                    )
                )

                if not args.no_figures and not representative_done and graph_name == "modular" and model_name == "modal_ou":
                    plot_representative(out / "figures", graph_name, model_name, sim, ews)
                    representative_done = True

    df = rows_to_frame(rows)
    metrics_csv = out / "synthetic_metrics.csv"
    df.to_csv(metrics_csv, index=False)
    summary = summarize(df)
    summary.to_csv(out / "summary_by_method.csv", index=False)
    graph_summary = summarize(df[df["method"] == "GSEWS"], by=["graph", "model"])
    graph_summary.to_csv(out / "summary_by_graph_model.csv", index=False)
    if not args.no_figures:
        plot_summary(metrics_csv, out / "figures")
    write_report(out, summary, graph_summary)
    print(summary.to_string(index=False))
    print(f"\nWrote results to {out}")


def choose_true_mode(graph_name: str, evals: np.ndarray, rng: np.random.Generator) -> int:
    n = len(evals)
    if graph_name == "modular":
        return int(min(4, n - 1))
    if graph_name == "ring":
        return int(rng.choice(np.arange(2, min(8, n))))
    if graph_name == "small_world":
        return int(rng.choice(np.arange(3, min(12, n))))
    if graph_name == "scale_free":
        return int(rng.choice(np.arange(2, min(10, n))))
    return int(rng.choice(np.arange(1, n)))


def summarize(df, by=None):
    if by is None:
        by = ["method"]
    return (
        df.groupby(by, as_index=False)
        .agg(
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            lead_mean=("lead_time_steps", "mean"),
            false_alarm_mean=("false_alarm_rate", "mean"),
            mode_acc_mean=("dominant_mode_accuracy", "mean"),
            loc_corr_mean=("localization_corr", "mean"),
        )
        .sort_values(by)
    )


def write_report(out: Path, summary, graph_summary) -> None:
    lines = [
        "# Synthetic GSEWS Experiment Report",
        "",
        "The synthetic experiments use known destabilizing graph eigenmodes.",
        "This validates the central theory claim before moving to real datasets.",
        "",
        "## Method Summary",
        "",
        frame_as_text(summary),
        "",
        "## GSEWS by Graph and Model",
        "",
        frame_as_text(graph_summary),
        "",
        "## Recommended Chapter Figures",
        "",
        "- `fig01_representative_modal_warning.pdf`: modal variance and global warning before transition.",
        "- `fig02_benchmark_summary.pdf`: GSEWS versus node-average, PCA, and signal-energy baselines.",
        "- `fig03_topology_comparison.pdf`: topology-dependent warning behavior.",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(lines))


def frame_as_text(df) -> str:
    return df.to_csv(index=False).strip()


if __name__ == "__main__":
    main()
