from __future__ import annotations

import argparse
import math
import shutil
import tempfile
import textwrap
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve


APP_LABELS = {
    "co2_capture": "CO₂ capture",
    "ch4_storage": "CH₄ storage",
}

FEATURE_GROUP_LABELS = {
    "precursor_only": "Precursor only",
    "descriptor_only": "Descriptor only",
    "precursor_descriptor": "Precursor + descriptor",
    "shap_selected": "SHAP-selected (15)",
    "baseline": "Baseline",
}

MODEL_LABELS = {
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "xgboost": "XGBoost",
    "class_prior": "Class-prior baseline",
    "linker_tanimoto_knn": "Linker-only Tanimoto kNN",
    "metal_aware_precursor_knn": "Metal-aware precursor kNN",
}

SUBSET_LABELS = {
    "test": "Combined holdout",
    "test_partial_unseen": "Partially unseen metals",
    "test_fully_unseen": "Fully unseen metals",
}


def _friendly_feature(feature: str, meaning: str) -> str:
    preferred = {
        "metal_electronegativity_avg": "Metal electronegativity",
        "metal_covalent_radius_avg": "Metal covalent radius",
        "metal_atomic_weight_avg": "Metal atomic weight",
        "metal_atomic_number_avg": "Metal atomic number",
        "num_metal_atoms": "Metal-node nuclearity",
        "MolLogP": "Linker hydrophobicity (MolLogP)",
        "MolWt": "Linker molecular weight",
        "NumCarboxylGroups": "Carboxyl-group count",
        "HeavyAtomCount": "Linker heavy-atom count",
        "TPSA": "Topological polar surface area",
        "NumRotatableBonds": "Rotatable-bond count",
        "RingCount": "Ring count",
        "NumAromaticRings": "Aromatic-ring count",
        "FractionCSP3": "sp³ carbon fraction",
        "n_linker_fragments": "Parsed linker-fragment count",
        "NumHeteroatoms": "Heteroatom count",
        "NumPyridylN": "Pyridyl-nitrogen count",
    }
    if feature in preferred:
        return preferred[feature]
    if feature.startswith("metal_") and len(feature.split("_")) == 2:
        return f"Metal identity: {feature.split('_', 1)[1]}"
    if feature.startswith("fp_"):
        if isinstance(meaning, str) and meaning.strip():
            cleaned = meaning.replace("substructure:", "Substructure:").strip()
            return f"{feature.upper()} — {cleaned}"
        return feature.upper()
    if isinstance(meaning, str) and meaning.strip():
        return meaning.strip().capitalize()
    return feature.replace("_", " ").title()


def _save(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}.png", dpi=400, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _find_root(path: Path) -> Path:
    if path.is_dir():
        candidates = [path] + [p for p in path.rglob("*") if p.is_dir()]
        for candidate in candidates:
            if (candidate / "result_csv").is_dir() and (candidate / "audits").is_dir():
                return candidate
            if (candidate / "results" / "tables").is_dir() and (candidate / "results" / "audits").is_dir():
                return candidate
    raise FileNotFoundError(
        "Could not locate result_csv/audits or results/tables/results/audits inside the input."
    )


def _resolve_dirs(root: Path) -> tuple[Path, Path]:
    if (root / "result_csv").is_dir():
        return root / "result_csv", root / "audits"
    return root / "results" / "tables", root / "results" / "audits"


def _ece(y_true: np.ndarray, probabilities: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ids = np.digitize(probabilities, bins[1:-1], right=True)
    total = 0.0
    for bin_index in range(n_bins):
        mask = ids == bin_index
        if mask.any():
            total += mask.mean() * abs(y_true[mask].mean() - probabilities[mask].mean())
    return float(total)


def plot_primary_auc(ablation: pd.DataFrame, app: str, out_dir: Path) -> None:
    data = ablation[
        (ablation["application"] == app)
        & (ablation["split_type"] == "scaffold")
        & (ablation["test_subset"] == "test")
        & (ablation["feature_group"] != "baseline")
    ].copy()
    data["label"] = data.apply(
        lambda row: f"{FEATURE_GROUP_LABELS[row['feature_group']]} | {MODEL_LABELS[row['model']]}",
        axis=1,
    )
    data = data.sort_values("roc_auc_mean", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10.5, 8.0))
    y_positions = np.arange(len(data))

    for model_name in ["logistic_regression", "random_forest", "xgboost"]:
        mask = data["model"] == model_name
        ax.errorbar(
            data.loc[mask, "roc_auc_mean"],
            y_positions[mask.to_numpy()],
            xerr=data.loc[mask, "roc_auc_std"],
            fmt="o",
            capsize=4,
            linewidth=1.4,
            markersize=6,
            label=MODEL_LABELS[model_name],
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(data["label"])
    ax.set_xlabel("Mean ROC–AUC ± SD across three scaffold-split seeds")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right")

    lower = max(0.48, float((data["roc_auc_mean"] - data["roc_auc_std"]).min()) - 0.04)
    upper = min(1.0, float((data["roc_auc_mean"] + data["roc_auc_std"]).max()) + 0.08)
    ax.set_xlim(lower, upper)

    for idx, row in data.iterrows():
        ax.text(
            row["roc_auc_mean"] + row["roc_auc_std"] + 0.006,
            idx,
            f"{row['roc_auc_mean']:.3f} ± {row['roc_auc_std']:.3f}",
            va="center",
            fontsize=8.5,
        )

    _save(fig, out_dir, f"primary_auc_{app}_revised")


def plot_baseline_comparison(ablation: pd.DataFrame, app: str, out_dir: Path) -> None:
    scaffold = ablation[
        (ablation["application"] == app)
        & (ablation["split_type"] == "scaffold")
        & (ablation["test_subset"] == "test")
    ].copy()

    learned = scaffold[scaffold["feature_group"] != "baseline"].sort_values(
        "roc_auc_mean", ascending=False
    ).iloc[0]
    baselines = scaffold[scaffold["feature_group"] == "baseline"].copy()

    rows = [
        {
            "label": (
                f"Best learned model\n"
                f"{FEATURE_GROUP_LABELS[learned['feature_group']]} + {MODEL_LABELS[learned['model']]}"
            ),
            "mean": learned["roc_auc_mean"],
            "std": learned["roc_auc_std"],
        }
    ]
    for baseline_name in [
        "metal_aware_precursor_knn",
        "linker_tanimoto_knn",
        "class_prior",
    ]:
        row = baselines[baselines["model"] == baseline_name].iloc[0]
        rows.append(
            {
                "label": MODEL_LABELS[baseline_name],
                "mean": row["roc_auc_mean"],
                "std": row["roc_auc_std"],
            }
        )

    frame = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    x = np.arange(len(frame))
    bars = ax.bar(
        x,
        frame["mean"],
        yerr=frame["std"],
        capsize=5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(frame["label"])
    ax.set_ylabel("Mean ROC–AUC ± SD")
    ax.set_ylim(0.45, min(1.0, float((frame["mean"] + frame["std"]).max()) + 0.12))
    ax.grid(axis="y", alpha=0.25)

    for bar, mean, std in zip(bars, frame["mean"], frame["std"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean + std + 0.012,
            f"{mean:.3f} ± {std:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    _save(fig, out_dir, f"baseline_comparison_{app}_revised")


def plot_calibration(tables_dir: Path, summary: pd.DataFrame, app: str, out_dir: Path) -> None:
    screen = pd.read_csv(tables_dir / f"confidence_screen_{app}.csv")
    y_true = screen["true_label"].to_numpy(dtype=int)
    raw = screen["raw_probability"].to_numpy(dtype=float)
    calibrated = screen["calibrated_probability"].to_numpy(dtype=float)

    raw_fraction, raw_mean = calibration_curve(
        y_true, raw, n_bins=10, strategy="quantile"
    )
    cal_fraction, cal_mean = calibration_curve(
        y_true, calibrated, n_bins=10, strategy="quantile"
    )

    row = summary[summary["application"] == app].iloc[0]

    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    ax.plot(
        raw_mean,
        raw_fraction,
        marker="o",
        label=f"Raw (Brier {row['brier_before']:.3f}; ECE {row['ece_before']:.3f})",
    )
    ax.plot(
        cal_mean,
        cal_fraction,
        marker="s",
        label=f"Platt-scaled (Brier {row['brier_after']:.3f}; ECE {row['ece_after']:.3f})",
    )
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive fraction")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left")

    _save(fig, out_dir, f"calibration_{app}_raw_vs_platt")


def plot_split_balance(split_audit: pd.DataFrame, out_dir: Path) -> None:
    data = split_audit[split_audit["split_type"] == "scaffold"].copy()
    data = data.sort_values("seed")
    seeds = [str(int(seed)) for seed in data["seed"]]
    x = np.arange(len(data))
    width = 0.24

    fig, ax = plt.subplots(figsize=(9.0, 6.2))
    bars_train = ax.bar(x - width, data["n_train"], width, label="Train")
    bars_valid = ax.bar(x, data["n_valid"], width, label="Validation")
    bars_test = ax.bar(x + width, data["n_test"], width, label="Test")

    ax.set_xticks(x)
    ax.set_xticklabels(seeds)
    ax.set_xlabel("Random seed")
    ax.set_ylabel("Number of MOFs")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    for bars in [bars_train, bars_valid, bars_test]:
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 70,
                f"{int(bar.get_height()):,}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_ylim(0, float(data[["n_train", "n_valid", "n_test"]].to_numpy().max()) * 1.15)
    _save(fig, out_dir, "scaffold_split_balance_revised")


def plot_overlap_audit(split_audit: pd.DataFrame, out_dir: Path) -> None:
    data = split_audit[split_audit["split_type"] == "scaffold"].copy()
    data = data.sort_values("seed")

    columns = [
        "precursor_overlap_train_test",
        "linker_overlap_train_test",
        "scaffold_overlap_train_test",
    ]
    labels = ["Precursor overlap", "Linker overlap", "Scaffold overlap"]

    cell_text = []
    for _, row in data.iterrows():
        cell_text.append([str(int(row[col])) for col in columns])

    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        rowLabels=[f"Seed {int(seed)}" for seed in data["seed"]],
        colLabels=labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.8)
    ax.text(
        0.5,
        0.12,
        "All train–test overlap counts are zero for the primary scaffold split.",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=10,
    )
    _save(fig, out_dir, "scaffold_leakage_audit_revised")


def plot_shap(tables_dir: Path, app: str, out_dir: Path) -> None:
    data = pd.read_csv(tables_dir / f"shap_trend_{app}.csv").copy()
    data = data.head(12).iloc[::-1].reset_index(drop=True)
    trend_symbol = {
        "positive": "+",
        "negative": "−",
        "nonlinear": "nonlinear",
    }
    data["label"] = data.apply(
        lambda row: f"{_friendly_feature(row['feature'], row['materials_meaning'])} ({trend_symbol.get(row['shap_trend'], row['shap_trend'])})",
        axis=1,
    )

    fig, ax = plt.subplots(figsize=(10.0, 7.2))
    bars = ax.barh(data["label"], data["mean_abs_shap"])
    ax.set_xlabel("Mean absolute SHAP value")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25)

    offset = max(float(data["mean_abs_shap"].max()) * 0.015, 0.0005)
    for bar, value in zip(bars, data["mean_abs_shap"]):
        ax.text(
            value + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            fontsize=8.5,
        )
    ax.set_xlim(0, float(data["mean_abs_shap"].max()) * 1.20)
    _save(fig, out_dir, f"shap_importance_{app}_revised")


def plot_metal_holdout(ablation: pd.DataFrame, app: str, out_dir: Path) -> None:
    data = ablation[
        (ablation["application"] == app)
        & (ablation["split_type"] == "metal_element_holdout")
        & (ablation["feature_group"] == "precursor_descriptor")
    ].copy()

    subsets = ["test", "test_partial_unseen", "test_fully_unseen"]
    models = ["logistic_regression", "random_forest", "xgboost"]
    x = np.arange(len(subsets))
    width = 0.24

    fig, ax = plt.subplots(figsize=(9.4, 6.2))
    for offset_index, model_name in enumerate(models):
        model_data = data[data["model"] == model_name].set_index("test_subset")
        values = [float(model_data.loc[subset, "roc_auc_mean"]) for subset in subsets]
        bars = ax.bar(
            x + (offset_index - 1) * width,
            values,
            width,
            label=MODEL_LABELS[model_name],
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.010,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([SUBSET_LABELS[subset] for subset in subsets])
    ax.set_ylabel("ROC–AUC (reference seed 42)")
    ax.set_ylim(0.65, min(1.0, float(data["roc_auc_mean"].max()) + 0.10))
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    _save(fig, out_dir, f"metal_holdout_{app}_revised")


def plot_compute(tables_dir: Path, out_dir: Path) -> None:
    data = pd.read_csv(tables_dir / "compute_comparison.csv").copy()
    data["label"] = data["model"].map(MODEL_LABELS)

    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    bars = ax.bar(
        data["label"],
        data["tuning_and_fit_time_sec_mean"],
        yerr=data["tuning_and_fit_time_sec_std"],
        capsize=5,
    )
    ax.set_ylabel("Tuning and fit time (s), mean ± SD")
    ax.grid(axis="y", alpha=0.25)
    for bar, mean, std in zip(
        bars,
        data["tuning_and_fit_time_sec_mean"],
        data["tuning_and_fit_time_sec_std"],
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean + std + 0.18,
            f"{mean:.2f} ± {std:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylim(0, float((data["tuning_and_fit_time_sec_mean"] + data["tuning_and_fit_time_sec_std"]).max()) * 1.22)
    _save(fig, out_dir, "low_compute_training_time_revised")

    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    bars = ax.bar(data["label"], data["inference_us_per_sample_mean"])
    ax.set_ylabel("Inference time (µs per sample)")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, data["inference_us_per_sample_mean"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.2,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylim(0, float(data["inference_us_per_sample_mean"].max()) * 1.22)
    _save(fig, out_dir, "low_compute_inference_time_revised")


def generate(input_path: Path, output_dir: Path) -> Path:
    temporary_dir: Path | None = None
    try:
        if input_path.is_file() and input_path.suffix.lower() == ".zip":
            temporary_dir = Path(tempfile.mkdtemp(prefix="paper2_figures_"))
            with zipfile.ZipFile(input_path, "r") as archive:
                archive.extractall(temporary_dir)
            root = _find_root(temporary_dir)
        else:
            root = _find_root(input_path)

        tables_dir, audits_dir = _resolve_dirs(root)
        output_dir.mkdir(parents=True, exist_ok=True)

        ablation = pd.read_csv(tables_dir / "ablation_results_full.csv")
        calibration_summary = pd.read_csv(tables_dir / "calibration_summary.csv")
        split_audit = pd.read_csv(audits_dir / "split_audit.csv")

        for app in ["co2_capture", "ch4_storage"]:
            plot_primary_auc(ablation, app, output_dir)
            plot_baseline_comparison(ablation, app, output_dir)
            plot_calibration(tables_dir, calibration_summary, app, output_dir)
            plot_shap(tables_dir, app, output_dir)
            plot_metal_holdout(ablation, app, output_dir)

        plot_split_balance(split_audit, output_dir)
        plot_overlap_audit(split_audit, output_dir)
        plot_compute(tables_dir, output_dir)

        readme = output_dir / "FIGURE_INDEX.txt"
        readme.write_text(
            "\n".join(
                [
                    "Paper 2 revised publication figures",
                    "",
                    "Each figure is provided separately in PNG and PDF.",
                    "No manuscript caption is embedded inside the figure.",
                    "",
                    *[path.name for path in sorted(output_dir.glob("*.png"))],
                ]
            ),
            encoding="utf-8",
        )

        zip_base = output_dir.parent / output_dir.name
        zip_path = Path(shutil.make_archive(str(zip_base), "zip", root_dir=output_dir))
        return zip_path
    finally:
        if temporary_dir is not None:
            shutil.rmtree(temporary_dir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Final result ZIP or extracted result folder")
    parser.add_argument("--output", required=True, help="Output folder for revised figures")
    args = parser.parse_args()

    zip_path = generate(Path(args.input), Path(args.output))
    print(f"Revised figures created: {zip_path}")


if __name__ == "__main__":
    main()
