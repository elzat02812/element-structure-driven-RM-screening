"""
Reproduce Supplementary Figure 1 of the manuscript.

Panels:
    (a) Predicted vs. experimental oxidation potential (LOOCV), with the
        absolute prediction error encoded by point colour.
    (b) MDI-based feature importance ranking.

Inputs:
    results/loocv_predictions.csv
    results/feature_importance.csv
    results/metrics.json

Outputs:
    figures/supp_fig1a_parity.png
    figures/supp_fig1b_importance.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parity_plot(pred_csv: Path, metrics_json: Path, out_path: Path) -> None:
    """Predicted vs. experimental potential, coloured by absolute error."""
    df = pd.read_csv(pred_csv)
    with open(metrics_json) as fh:
        metrics = json.load(fh)

    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    sc = ax.scatter(
        df["E_ox_true"],
        df["E_ox_pred"],
        c=df["abs_error"],
        cmap="Reds",
        edgecolor="black",
        linewidth=0.5,
        s=80,
        vmin=0.0,
    )
    lo = min(df["E_ox_true"].min(), df["E_ox_pred"].min()) - 0.1
    hi = max(df["E_ox_true"].max(), df["E_ox_pred"].max()) + 0.1
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel(r"Experimental potential (V vs. Li$^+$/Li)")
    ax.set_ylabel(r"Predicted potential (V vs. Li$^+$/Li)")
    ax.set_title(
        rf"$R^2$ = {metrics['R2']:.3f},   MAE = {metrics['MAE']:.3f} V",
        fontsize=11,
    )
    cbar = fig.colorbar(sc, ax=ax, shrink=0.85)
    cbar.set_label("|Error| (V)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Wrote {out_path}")


def importance_plot(importance_csv: Path, out_path: Path) -> None:
    """Bar chart of MDI feature importances, sorted in descending order."""
    df = pd.read_csv(importance_csv).sort_values("importance_MDI", ascending=False)
    n = len(df)
    colors = plt.cm.RdYlBu_r((df["importance_MDI"].rank(ascending=False) - 1) / max(n - 1, 1))

    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    ax.bar(df["feature"], df["importance_MDI"], color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Importance score (MDI)")
    ax.tick_params(axis="x", rotation=45)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main(results_dir: Path, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    parity_plot(
        results_dir / "loocv_predictions.csv",
        results_dir / "metrics.json",
        figures_dir / "supp_fig1a_parity.png",
    )
    importance_plot(
        results_dir / "feature_importance.csv",
        figures_dir / "supp_fig1b_importance.png",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--figures-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()
    main(args.results_dir, args.figures_dir)
