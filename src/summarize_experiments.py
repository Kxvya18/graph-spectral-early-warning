from __future__ import annotations

from pathlib import Path

import pandas as pd


RESULT_DIRS = [
    ("Linearized modal OU theorem test", Path("results/linear_theory_gsews")),
    ("Nonlinear reduced excitable physiology test", Path("results/nonlinear_physiology_gsews")),
    ("Smoke mixed test", Path("results/smoke_synthetic_gsews")),
]


def main() -> None:
    lines = [
        "# GSEWS Experiment Evidence Summary",
        "",
        "This file summarizes the experiment evidence currently available for the separate GSEWS book chapter.",
        "",
        "## Claim Status",
        "",
        "- Linearized graph-spectral theorem: supported by controlled modal OU experiments with known critical eigenmode.",
        "- Global early warning: supported in both linearized and nonlinear reduced simulations.",
        "- Exact eigenmode localization: supported in the linearized theorem setting; not claimed as exact in nonlinear reduced FHN simulations.",
        "- Actual clinical dataset validation: pending dataset download/preprocessing.",
        "",
    ]
    for title, path in RESULT_DIRS:
        summary_path = path / "summary_by_method.csv"
        graph_path = path / "summary_by_graph_model.csv"
        if not summary_path.exists():
            continue
        summary = pd.read_csv(summary_path)
        lines.extend(["## " + title, "", "### Method Summary", "", summary.to_csv(index=False).strip(), ""])
        if graph_path.exists():
            graph_summary = pd.read_csv(graph_path)
            lines.extend(["### By Graph/Model", "", graph_summary.to_csv(index=False).strip(), ""])
    lines.extend(
        [
            "## Figure Inventory",
            "",
            "- `results/linear_theory_gsews/figures/fig01_representative_modal_warning.pdf`",
            "- `results/linear_theory_gsews/figures/fig02_benchmark_summary.pdf`",
            "- `results/linear_theory_gsews/figures/fig03_topology_comparison.pdf`",
            "- `results/nonlinear_physiology_gsews/figures/fig02_benchmark_summary.pdf`",
            "- `results/nonlinear_physiology_gsews/figures/fig03_topology_comparison.pdf`",
            "",
            "## Next Clinical Dataset Step",
            "",
            "Use CHB-MIT scalp EEG first because seizure onset is a canonical coupled-network transition and channels naturally define graph nodes. The clinical claim should be early-warning discrimination and interpretable modal patterns, not exact ground-truth eigenmode recovery.",
            "",
        ]
    )
    out = Path("results/GSEWS_EXPERIMENT_EVIDENCE_SUMMARY.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    print(out)


if __name__ == "__main__":
    main()

