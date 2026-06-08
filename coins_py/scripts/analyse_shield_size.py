from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..io import coins_load_subject_data
from ..options import coins_options
from ..subjects import coins_subjects
from ..utils import ensure_dir


SHIELD_SIZES = (20, 40, 60)
NOISE_LABELS = ("low", "medium", "high")
VOLATILITY_LABELS = ("stable", "volatile")


def remove_excluded_blocks(
    sub_data: pd.DataFrame,
    excluded_blocks: list[list[int]],
) -> pd.DataFrame:
    data = sub_data.copy()

    for session_id, block_id in excluded_blocks:
        excluded_mask = (
            (data["sessID"] == session_id)
            & (data["blockID"] == block_id)
        )
        data = data.loc[~excluded_mask]

    return data.reset_index(drop=True)


def prepare_shield_size_data(
    sub_data: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    required_columns = {
        "shieldDegrees",
        "trueVariance",
        "volatility",
        "sessID",
        "blockID",
    }

    missing_columns = required_columns.difference(sub_data.columns)
    if missing_columns:
        raise ValueError(
            "The following required columns are missing from the loaded data: "
            f"{sorted(missing_columns)}"
        )

    data = sub_data.copy()

    data = data[
        data["shieldDegrees"].isin(SHIELD_SIZES)
        & data["trueVariance"].notna()
        & data["volatility"].notna()
    ].copy()

    if data.empty:
        raise ValueError(
            "No valid rows remained after filtering for shieldDegrees = 20, 40, 60."
        )

    variance_values = np.sort(data["trueVariance"].unique())

    if len(variance_values) != 3:
        raise ValueError(
            "Expected exactly three trueVariance values "
            f"(low, medium, high), but found {len(variance_values)}: "
            f"{variance_values}"
        )

    data["noise_category"] = pd.Series(index=data.index, dtype="object")

    for variance_value, noise_label in zip(variance_values, NOISE_LABELS):
        matching_rows = np.isclose(
            data["trueVariance"].to_numpy(dtype=float),
            variance_value,
            rtol=0.0,
            atol=1e-10,
        )
        data.loc[matching_rows, "noise_category"] = noise_label

    data["volatility_label"] = data["volatility"].map(
        {
            0: "stable",
            1: "volatile",
        }
    )

    data = data.dropna(
        subset=["noise_category", "volatility_label"]
    ).copy()

    data["shieldDegrees"] = data["shieldDegrees"].astype(int)

    return data, variance_values


def compute_shield_size_usage(
    data: pd.DataFrame,
    fsample: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    observed_counts = (
        data.groupby(
            ["volatility_label", "noise_category", "shieldDegrees"],
            observed=True,
        )
        .size()
        .reset_index(name="n_frames")
    )

    complete_index = pd.MultiIndex.from_product(
        [VOLATILITY_LABELS, NOISE_LABELS, SHIELD_SIZES],
        names=["volatility_label", "noise_category", "shieldDegrees"],
    )

    usage = (
        observed_counts
        .set_index(
            ["volatility_label", "noise_category", "shieldDegrees"]
        )
        .reindex(complete_index, fill_value=0)
        .reset_index()
    )

    usage["duration_seconds"] = usage["n_frames"] / float(fsample)

    total_frames = usage.groupby(
        ["volatility_label", "noise_category"]
    )["n_frames"].transform("sum")

    usage["proportion_time"] = np.where(
        total_frames > 0,
        usage["n_frames"] / total_frames,
        np.nan,
    )

    condition_summary = (
        data.groupby(
            ["volatility_label", "noise_category"],
            observed=True,
        )["shieldDegrees"]
        .agg(
            mean_shield_degrees="mean",
            median_shield_degrees="median",
            n_frames="size",
        )
        .reset_index()
    )

    condition_summary["duration_seconds"] = (
        condition_summary["n_frames"] / float(fsample)
    )

    return usage, condition_summary


def plot_shield_size_usage(
    usage: pd.DataFrame,
    subject_name: str,
) -> plt.Figure:
    condition_order = [
        ("stable", "low"),
        ("stable", "medium"),
        ("stable", "high"),
        ("volatile", "low"),
        ("volatile", "medium"),
        ("volatile", "high"),
    ]

    condition_labels = [
        "Stable\nLow",
        "Stable\nMedium",
        "Stable\nHigh",
        "Volatile\nLow",
        "Volatile\nMedium",
        "Volatile\nHigh",
    ]

    x_positions = np.arange(len(condition_order))
    bar_width = 0.25

    fig, ax = plt.subplots(figsize=(11, 5.5))

    offsets = (-bar_width, 0.0, bar_width)

    for shield_size, offset in zip(SHIELD_SIZES, offsets):
        values = []

        for volatility_label, noise_category in condition_order:
            current_row = usage[
                (usage["volatility_label"] == volatility_label)
                & (usage["noise_category"] == noise_category)
                & (usage["shieldDegrees"] == shield_size)
            ]

            if current_row.empty:
                values.append(np.nan)
            else:
                values.append(current_row["proportion_time"].iloc[0])

        ax.bar(
            x_positions + offset,
            values,
            width=bar_width,
            label=f"{shield_size}°",
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(condition_labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Proportion of time")
    ax.set_xlabel("Environmental condition")
    ax.set_title(
        f"{subject_name}: shield size use by noise and volatility condition"
    )
    ax.legend(title="Shield size")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()

    return fig


def analyse_subject_shield_size(
    sub_id: int,
    main_dir: str | Path | None = None,
) -> None:
    options = coins_options(main_dir)
    details = coins_subjects(sub_id, options)

    output_folder = Path(details["analysis"]["behav"]["folder"])
    ensure_dir(output_folder)

    sub_data = coins_load_subject_data(details)
    sub_data = remove_excluded_blocks(
        sub_data,
        details["excludedBlocks"],
    )

    clean_data, variance_values = prepare_shield_size_data(sub_data)

    usage, condition_summary = compute_shield_size_usage(
        clean_data,
        options.behav.fsample,
    )

    subject_name = details["subjName"]

    usage_path = output_folder / f"{subject_name}_shield_size_usage.csv"
    summary_path = output_folder / f"{subject_name}_shield_size_condition_summary.csv"
    figure_path = output_folder / f"{subject_name}_shield_size_by_condition.png"

    usage.to_csv(usage_path, index=False)
    condition_summary.to_csv(summary_path, index=False)

    fig = plot_shield_size_usage(usage, subject_name)
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print()
    print(f"Subject: {subject_name}")
    print("Detected trueVariance levels:")
    for noise_label, variance_value in zip(NOISE_LABELS, variance_values):
        print(f"  {noise_label}: {variance_value}")

    print()
    print("Mean shield size by condition:")
    print(condition_summary.to_string(index=False))

    print()
    print("Saved files:")
    print(f"  {usage_path}")
    print(f"  {summary_path}")
    print(f"  {figure_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyse use of shield sizes 20, 40 and 60 degrees "
            "for one COINS participant."
        )
    )

    parser.add_argument(
        "sub_id",
        type=int,
        help="Participant number, for example 1 for sub-01.",
    )

    parser.add_argument(
        "--main-dir",
        type=str,
        default=None,
        help=(
            "Main COINS folder containing rawData and analysis folders. "
            "For example: C:\\Users\\labri\\Uni\\Thesis\\COINS_Claude"
        ),
    )

    args = parser.parse_args()

    analyse_subject_shield_size(
        sub_id=args.sub_id,
        main_dir=args.main_dir,
    )


if __name__ == "__main__":
    main()