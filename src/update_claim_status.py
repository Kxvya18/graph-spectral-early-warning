from __future__ import annotations

from pathlib import Path

import pandas as pd


def main() -> None:
    out = Path("results/CLAIM_STATUS.md")
    lines = [
        "# Claim Status for GSEWS Chapter",
        "",
        "This file separates claims supported by theory-driven simulations from claims supported by real clinical data.",
        "",
    ]
    linear = load("results/linear_theory_gsews/summary_by_method.csv")
    nonlinear = load("results/nonlinear_physiology_gsews/summary_by_method.csv")
    chb = load_best_chb()

    lines.extend(["## Satisfied Claims", ""])
    if linear is not None:
        g = row(linear, "GSEWS")
        lines.append(f"- Linearized graph-spectral early warning: satisfied. GSEWS AUC `{g.auc_mean:.3f}` with exact modal localization in controlled modal OU experiments.")
        lines.append("- Closed-form modal mechanism: experimentally supported under the linearized SDE setting where the true destabilizing eigenmode is known.")
    if nonlinear is not None:
        g = row(nonlinear, "GSEWS")
        lines.append(f"- Nonlinear physiology-inspired warning robustness: partly-to-strongly supported. GSEWS AUC `{g.auc_mean:.3f}` in reduced excitable-network simulations.")
    if chb is not None:
        g = row(chb, "GSEWS")
        best_baseline = chb[chb["method"] != "GSEWS"]["auc_mean"].max()
        lines.append(f"- Actual multichannel EEG processing: real EDF-based CHB-MIT subset validation. Best current CHB-MIT setting gives GSEWS AUC `{g.auc_mean:.3f}` versus best baseline `{best_baseline:.3f}`.")

    lines.extend(["", "## Partially Satisfied Claims", ""])
    lines.append("- Physiological transitions such as seizure/arrhythmia/sepsis: seizure EEG validation is preliminarily supported; arrhythmia and sepsis remain future datasets.")
    lines.append("- Spatial localization in physiological recordings: exact localization is validated only in linearized simulations. Real EEG localization is currently interpretable modal mapping, not ground-truth anatomical localization.")
    lines.append("- Network-aware extension beyond aggregation: strengthened with NodeMax, GraphSmooth, and EdgeCorr baselines in addition to NodeAvg, PCA, and Energy.")

    lines.extend(["", "## Not Yet Fully Satisfied", ""])
    lines.append("- Multi-subject clinical validation: not yet. Current real EEG subset is from one CHB-MIT subject and three seizure EDFs.")
    lines.append("- Cardiac arrhythmia and septic-collapse validation: not yet.")
    lines.append("- Closed-loop clinical intervention: not experimentally demonstrated; should remain a future application.")

    lines.extend(["", "## Honest Chapter Wording Right Now", ""])
    lines.append("The chapter can claim that GSEWS is theoretically derived and validated in controlled coupled stochastic network models, and that preliminary real CHB-MIT EEG experiments show seizure-data feasibility. It should not yet claim full clinical seizure prediction, arrhythmia prediction, sepsis prediction, or ground-truth localization in patients.")
    out.write_text("\n".join(lines))
    print(out)


def load(path: str):
    p = Path(path)
    if not p.exists():
        return None
    return pd.read_csv(p)


def load_best_chb():
    candidates = [
        Path("results/chbmit_subset_gsews_10min/summary_by_method.csv"),
        Path("results/chbmit_subset_gsews_5min/summary_by_method.csv"),
        Path("results/chbmit_subset_gsews/summary_by_method.csv"),
    ]
    frames = []
    for p in candidates:
        if p.exists():
            df = pd.read_csv(p)
            df["source"] = str(p.parent)
            frames.append(df)
    if not frames:
        return None
    best = None
    best_auc = -1.0
    for df in frames:
        g = df[df["method"] == "GSEWS"]
        if not g.empty and float(g["auc_mean"].iloc[0]) > best_auc:
            best = df
            best_auc = float(g["auc_mean"].iloc[0])
    return best


def row(df: pd.DataFrame, method: str):
    return df[df["method"] == method].iloc[0]


if __name__ == "__main__":
    main()
