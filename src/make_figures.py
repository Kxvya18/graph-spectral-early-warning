from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
            "savefig.dpi": 320,
        }
    )


def plot_representative(
    out: Path,
    graph_name: str,
    model_name: str,
    sim,
    ews,
) -> None:
    set_style()
    out.mkdir(parents=True, exist_ok=True)
    t = sim.times
    centers_t = sim.times[ews.centers]
    true_mode = sim.true_mode
    fig, axes = plt.subplots(4, 1, figsize=(7.2, 7.4), constrained_layout=True)

    axes[0].plot(t, sim.x[:, : min(8, sim.x.shape[1])], lw=0.65, alpha=0.75)
    axes[0].axvline(t[sim.transition_step], color="crimson", lw=1.2, ls="--")
    axes[0].set_ylabel("node signals")
    axes[0].set_title(f"{graph_name} / {model_name}: multichannel state")

    axes[1].plot(centers_t, ews.modal_variance[:, true_mode], color="#1f77b4", lw=1.6, label=f"true mode {true_mode}")
    top_other = np.argsort(ews.modal_variance[-1])[-4:]
    for k in top_other:
        if k != true_mode:
            axes[1].plot(centers_t, ews.modal_variance[:, k], color="0.68", lw=0.7, alpha=0.8)
    axes[1].axvline(t[sim.transition_step], color="crimson", lw=1.2, ls="--")
    axes[1].set_ylabel("modal variance")
    axes[1].legend(frameon=False, loc="upper left")

    axes[2].plot(centers_t, ews.global_score, color="#222222", lw=1.6, label="GSEWS")
    axes[2].plot(centers_t, ews.node_avg_score, color="#d95f02", lw=1.0, label="node-average")
    axes[2].plot(centers_t, ews.pca_score, color="#7570b3", lw=1.0, label="PCA")
    axes[2].axvline(t[sim.transition_step], color="crimson", lw=1.2, ls="--")
    axes[2].set_ylabel("warning score")
    axes[2].legend(frameon=False, ncol=3, loc="upper left")

    axes[3].stem(np.arange(len(sim.true_localization)), sim.true_localization, linefmt="#1b9e77", markerfmt="o", basefmt=" ")
    axes[3].set_ylabel("|u*|")
    axes[3].set_xlabel("node")
    axes[3].set_title("dominant eigenvector localization")

    fig.savefig(out / "fig01_representative_modal_warning.pdf", bbox_inches="tight")
    fig.savefig(out / "fig01_representative_modal_warning.png", bbox_inches="tight")
    plt.close(fig)


def plot_summary(metrics_csv: Path, out: Path) -> None:
    set_style()
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(metrics_csv)
    summary = df.groupby("method", as_index=False).agg(
        auc=("auc", "mean"),
        lead=("lead_time_steps", "mean"),
        false_alarm=("false_alarm_rate", "mean"),
        mode_acc=("dominant_mode_accuracy", "mean"),
        loc_corr=("localization_corr", "mean"),
    )
    order = ["GSEWS", "NodeAvg", "NodeMax", "PCA", "Energy", "GraphSmooth", "EdgeCorr"]
    summary["method"] = pd.Categorical(summary["method"], categories=order, ordered=True)
    summary = summary.sort_values("method")

    fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.7), constrained_layout=True)
    colors = ["#222222", "#d95f02", "#e7298a", "#7570b3", "#66a61e", "#1b9e77", "#a6761d"]
    axes[0].bar(summary["method"].astype(str), summary["auc"], color=colors)
    axes[0].set_ylabel("pre-transition AUC")
    axes[0].set_ylim(0.45, 1.02)
    axes[0].tick_params(axis="x", rotation=25)

    axes[1].bar(summary["method"].astype(str), summary["lead"], color=colors)
    axes[1].set_ylabel("lead time (steps)")
    axes[1].tick_params(axis="x", rotation=25)

    loc = summary[summary["method"].astype(str) == "GSEWS"]
    axes[2].bar(["mode accuracy", "localization corr."], [float(loc["mode_acc"].iloc[0]), float(loc["loc_corr"].iloc[0])], color=["#1f77b4", "#1b9e77"])
    axes[2].set_ylim(0, 1.02)
    axes[2].set_ylabel("GSEWS localization")

    fig.savefig(out / "fig02_benchmark_summary.pdf", bbox_inches="tight")
    fig.savefig(out / "fig02_benchmark_summary.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.3, 3.0), constrained_layout=True)
    graph_summary = df[df["method"] == "GSEWS"].groupby("graph", as_index=False).agg(
        auc=("auc", "mean"),
        lead=("lead_time_steps", "mean"),
        loc=("localization_corr", "mean"),
    )
    x = np.arange(len(graph_summary))
    width = 0.26
    ax.bar(x - width, graph_summary["auc"], width, label="AUC", color="#222222")
    ax.bar(x, graph_summary["lead"] / max(graph_summary["lead"].max(), 1), width, label="lead normalized", color="#1f77b4")
    ax.bar(x + width, graph_summary["loc"], width, label="localization corr.", color="#1b9e77")
    ax.set_xticks(x)
    ax.set_xticklabels(graph_summary["graph"], rotation=20)
    ax.set_ylim(0, 1.05)
    ax.set_title("Topology reshapes graph-spectral warning behavior")
    ax.legend(frameon=False, ncol=3)
    fig.savefig(out / "fig03_topology_comparison.pdf", bbox_inches="tight")
    fig.savefig(out / "fig03_topology_comparison.png", bbox_inches="tight")
    plt.close(fig)
