"""Generate manuscript-ready figures from the corrected V3 result tables."""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config


def _save(fig, name):
    os.makedirs(config.RESULTS_FIGURES, exist_ok=True)
    fig.savefig(os.path.join(config.RESULTS_FIGURES, name + ".png"), dpi=400, bbox_inches="tight")
    fig.savefig(os.path.join(config.RESULTS_FIGURES, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)


def primary_performance():
    df = pd.read_csv(os.path.join(config.RESULTS_TABLES, "ablation_results_full.csv"))
    df = df[
        (df.split_type == "scaffold")
        & (df.test_subset == "test")
        & (df.feature_group != "baseline")
    ].copy()
    best = df.sort_values("roc_auc_mean", ascending=False).groupby("application", as_index=False).head(8)
    best["label"] = best["feature_group"] + "\n" + best["model"]

    for app, group in best.groupby("application"):
        group = group.sort_values("roc_auc_mean")
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        y = np.arange(len(group))
        ax.errorbar(
            group["roc_auc_mean"], y,
            xerr=group["roc_auc_std"], fmt="o", capsize=4, linewidth=1.6,
        )
        ax.set_yticks(y)
        ax.set_yticklabels(group["label"])
        ax.set_xlabel("ROC-AUC (mean ± SD across scaffold seeds)")
        ax.set_xlim(0.45, 1.0)
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        _save(fig, f"primary_auc_{app}")


def baseline_comparison():
    df = pd.read_csv(os.path.join(config.RESULTS_TABLES, "ablation_results_full.csv"))
    rows = []
    for app in config.APPLICATIONS:
        subset = df[(df.application == app) & (df.split_type == "scaffold") & (df.test_subset == "test")]
        learned = subset[subset.feature_group != "baseline"].sort_values("roc_auc_mean", ascending=False).iloc[0]
        rows.append({"application": app, "method": "best learned model", "auc": learned.roc_auc_mean})
        for model in ["class_prior", "linker_tanimoto_knn", "metal_aware_precursor_knn"]:
            hit = subset[subset.model == model]
            if len(hit):
                rows.append({"application": app, "method": model, "auc": hit.iloc[0].roc_auc_mean})
    plot = pd.DataFrame(rows)
    pivot = plot.pivot(index="method", columns="application", values="auc")
    fig, ax = plt.subplots(figsize=(8, 5))
    pivot.plot(kind="barh", ax=ax)
    ax.set_xlabel("ROC-AUC")
    ax.set_xlim(0.45, 1.0)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    _save(fig, "baseline_comparison")


def split_balance():
    df = pd.read_csv(os.path.join(config.RESULTS_AUDITS, "split_audit.csv"))
    scaffold = df[df.split_type == "scaffold"].sort_values("seed")
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    x = np.arange(len(scaffold))
    bottom = np.zeros(len(scaffold))
    for col, label in [("n_train", "Train"), ("n_valid", "Validation"), ("n_test", "Test")]:
        ax.bar(x, scaffold[col], bottom=bottom, label=label)
        bottom += scaffold[col].to_numpy()
    ax.set_xticks(x)
    ax.set_xticklabels([f"Seed {s}" for s in scaffold.seed])
    ax.set_ylabel("MOF rows")
    ax.legend()
    fig.tight_layout()
    _save(fig, "scaffold_split_balance")


def calibration_plot():
    path = os.path.join(config.RESULTS_TABLES, "calibration_curve.csv")
    if not os.path.exists(path):
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    for app, group in df.groupby("application"):
        ax.plot(group.mean_predicted_probability, group.observed_positive_fraction, marker="o", label=app)
    ax.set_xlabel("Mean calibrated probability")
    ax.set_ylabel("Observed positive fraction")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    _save(fig, "calibration_reliability")


def main():
    primary_performance()
    baseline_comparison()
    split_balance()
    calibration_plot()
    print(f"Figures saved to {config.RESULTS_FIGURES}")


if __name__ == "__main__":
    main()
