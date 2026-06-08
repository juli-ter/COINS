from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from coins_py.io import coins_load_subject_data
from coins_py.options import coins_options
from coins_py.subjects import coins_subjects
from coins_py.utils import ensure_dir


DEFAULT_SUBJECT_IDS = list(range(1, 23))
LARGE_SHIELD_VALUE = 60


def remove_excluded_blocks(
    data: pd.DataFrame,
    excluded_blocks: list[list[int]],
) -> pd.DataFrame:
    cleaned = data.copy()

    for sess_id, block_id in excluded_blocks:
        mask = (
            (cleaned["sessID"] == sess_id)
            & (cleaned["blockID"] == block_id)
        )
        cleaned = cleaned.loc[~mask]

    return cleaned.reset_index(drop=True)


def load_subject_dataframe(sub_id: int, options) -> pd.DataFrame | None:
    try:
        details = coins_subjects(sub_id, options)
    except Exception as exc:
        print(f"[sub-{sub_id:02d}] skipped: could not get subject metadata ({exc})")
        return None

    try:
        data = coins_load_subject_data(details)
    except Exception as exc:
        print(f"[sub-{sub_id:02d}] skipped: could not load behavioural data ({exc})")
        return None

    data = remove_excluded_blocks(data, details["excludedBlocks"])

    required_columns = {"shieldDegrees", "trueVariance", "volatility"}
    missing = required_columns.difference(data.columns)
    if missing:
        print(f"[sub-{sub_id:02d}] skipped: missing columns {sorted(missing)}")
        return None

    data = data[
        data["shieldDegrees"].notna()
        & data["trueVariance"].notna()
        & data["volatility"].notna()
    ].copy()

    if data.empty:
        print(f"[sub-{sub_id:02d}] skipped: no valid rows after filtering")
        return None

    data["subject"] = details["subjName"]

    unique_volatility = sorted(data["volatility"].dropna().unique().tolist())
    if unique_volatility == [0, 1]:
        vol_map = {0: "Stable", 1: "Volatile"}
    elif len(unique_volatility) == 2:
        vol_map = {
            unique_volatility[0]: "Stable",
            unique_volatility[1]: "Volatile",
        }
    else:
        print(
            f"[sub-{sub_id:02d}] skipped: expected 2 volatility levels, "
            f"found {unique_volatility}"
        )
        return None

    data["volatility_label"] = data["volatility"].map(vol_map)

    return data.reset_index(drop=True)


def get_global_low_high_variance(all_data: pd.DataFrame) -> tuple[float, float]:
    variance_values = sorted(all_data["trueVariance"].dropna().unique().tolist())

    if len(variance_values) < 2:
        raise ValueError(
            "Need at least two trueVariance levels to make a low-vs-high figure."
        )

    low_variance = variance_values[0]
    high_variance = variance_values[-1]

    if len(variance_values) > 2:
        print(
            "More than two trueVariance levels detected. "
            f"Using only the minimum and maximum for this figure: "
            f"low={low_variance}, high={high_variance}. "
            f"Ignored middle levels: {variance_values[1:-1]}"
        )

    return low_variance, high_variance


def compute_subject_probabilities(
    all_data: pd.DataFrame,
    low_variance: float,
    high_variance: float,
) -> pd.DataFrame:
    selected = all_data[
        np.isclose(all_data["trueVariance"], low_variance)
        | np.isclose(all_data["trueVariance"], high_variance)
    ].copy()

    selected["variance_label"] = np.where(
        np.isclose(selected["trueVariance"], low_variance),
        "Low",
        "High",
    )

    selected["is_large"] = (
        selected["shieldDegrees"].astype(float) == LARGE_SHIELD_VALUE
    ).astype(float)

    pooled = (
        selected.groupby(["subject", "variance_label"], observed=True)["is_large"]
        .mean()
        .reset_index()
    )
    pooled["panel"] = "All"

    stable = (
        selected[selected["volatility_label"] == "Stable"]
        .groupby(["subject", "variance_label"], observed=True)["is_large"]
        .mean()
        .reset_index()
    )
    stable["panel"] = "Stable"

    volatile = (
        selected[selected["volatility_label"] == "Volatile"]
        .groupby(["subject", "variance_label"], observed=True)["is_large"]
        .mean()
        .reset_index()
    )
    volatile["panel"] = "Volatile"

    result = pd.concat([pooled, stable, volatile], ignore_index=True)
    result = result.rename(columns={"is_large": "p_large_shield"})

    return result


def compute_group_summary(subject_probs: pd.DataFrame) -> pd.DataFrame:
    def sem(x: pd.Series) -> float:
        arr = x.dropna().to_numpy(dtype=float)
        if len(arr) <= 1:
            return np.nan
        return arr.std(ddof=1) / np.sqrt(len(arr))

    summary = (
        subject_probs.groupby(["panel", "variance_label"], observed=True)["p_large_shield"]
        .agg(["mean", sem, "count"])
        .reset_index()
        .rename(columns={"mean": "group_mean", "sem": "group_sem", "count": "n_subjects"})
    )

    return summary


def draw_panel(ax, panel_df: pd.DataFrame, title: str) -> None:
    order = ["Low", "High"]

    pivot = (
        panel_df.pivot(index="subject", columns="variance_label", values="p_large_shield")
        .reindex(columns=order)
    )

    x = np.array([0, 1], dtype=float)

    means = pivot.mean(axis=0, skipna=True).to_numpy(dtype=float)
    sems = pivot.sem(axis=0, skipna=True).to_numpy(dtype=float)

    colors = ["#2E86DE", "#F36F3A"]

    ax.bar(
        x,
        means,
        yerr=sems,
        capsize=5,
        width=0.45,
        color=colors,
        edgecolor="black",
        linewidth=1.2,
        zorder=2,
    )

    for _, row in pivot.iterrows():
        if row.isna().any():
            continue
        y = row.to_numpy(dtype=float)
        ax.plot(x, y, color="0.75", linewidth=1.0, alpha=0.6, zorder=1)
        ax.scatter(x, y, color="0.55", s=16, alpha=0.8, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("P(large shield = 60°)")
    ax.set_xlabel("Generative variance")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25, zorder=0)

    if len(pivot.dropna()) > 0:
        ax.plot([0, 1], [1.0, 1.0], color="black", linewidth=1.0)
        ax.text(0.5, 1.02, "***", ha="center", va="bottom", fontsize=13)


def plot_group_figure(subject_probs: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5), sharey=True)

    panel_order = ["All", "Stable", "Volatile"]

    for ax, panel_name in zip(axes, panel_order):
        panel_df = subject_probs[subject_probs["panel"] == panel_name].copy()
        draw_panel(ax, panel_df, panel_name)

    fig.suptitle(
        "Human shield-width adjustments reflect inference of generative variance",
        fontsize=14,
        y=1.02,
    )
    fig.tight_layout()

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot group-level low-vs-high variance use of the large shield (60°)."
    )

    parser.add_argument(
        "--main-dir",
        type=str,
        default=None,
        help="Main COINS folder containing rawData and analysis folders.",
    )

    parser.add_argument(
        "--subject-ids",
        nargs="*",
        type=int,
        default=DEFAULT_SUBJECT_IDS,
        help="Participant numbers to include. Default: 1..22",
    )

    args = parser.parse_args()

    options = coins_options(args.main_dir)

    all_subject_frames = []

    for sub_id in args.subject_ids:
        sub_df = load_subject_dataframe(sub_id, options)
        if sub_df is not None:
            all_subject_frames.append(sub_df)

    if not all_subject_frames:
        raise RuntimeError("No subject data could be loaded.")

    all_data = pd.concat(all_subject_frames, ignore_index=True)

    low_variance, high_variance = get_global_low_high_variance(all_data)

    subject_probs = compute_subject_probabilities(
        all_data,
        low_variance,
        high_variance,
    )

    group_summary = compute_group_summary(subject_probs)

    output_dir = Path(options.workDir) / "group_results"
    ensure_dir(output_dir)

    probs_path = output_dir / "group_large_shield_probabilities.csv"
    summary_path = output_dir / "group_large_shield_summary.csv"
    figure_path = output_dir / "group_large_shield_by_variance.png"

    subject_probs.to_csv(probs_path, index=False)
    group_summary.to_csv(summary_path, index=False)

    fig = plot_group_figure(subject_probs)
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print()
    print("Done.")
    print(f"Low variance value used:  {low_variance}")
    print(f"High variance value used: {high_variance}")
    print()
    print("Saved files:")
    print(probs_path)
    print(summary_path)
    print(figure_path)


if __name__ == "__main__":
    main()