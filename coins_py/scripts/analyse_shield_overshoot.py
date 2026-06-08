
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


def remove_excluded_blocks(data: pd.DataFrame, excluded_blocks: list[list[int]]) -> pd.DataFrame:
    cleaned = data.copy()
    for sess_id, block_id in excluded_blocks:
        mask = (cleaned["sessID"] == sess_id) & (cleaned["blockID"] == block_id)
        cleaned = cleaned.loc[~mask]
    return cleaned.reset_index(drop=True)


def load_subject_data(sub_id: int, options) -> pd.DataFrame | None:
    try:
        details = coins_subjects(sub_id, options)
        data = coins_load_subject_data(details)
    except Exception as exc:
        print(f"[sub-{sub_id:02d}] skipped: {exc}")
        return None

    data = remove_excluded_blocks(data, details["excludedBlocks"])
    data["volatility_label"] = data["volatility"].map({0: "Stable", 1: "Volatile"})

    required = {"sessID", "blockID", "currentFrame", "trueVariance", "shieldDegrees", "volatility"}
    missing = required.difference(data.columns)

    if missing:
        print(f"[sub-{sub_id:02d}] skipped: missing columns {sorted(missing)}")
        return None
    data["subject"] = details["subjName"]
    data["volatility_label"] = data["volatility"].map({0: "Stable", 1: "Volatile"})
    return data.reset_index(drop=True)


def compute_subject_noise_targets(all_data: pd.DataFrame) -> pd.DataFrame:
    targets = (
        all_data.groupby(["subject", "trueVariance"], observed=True)["shieldDegrees"]
        .mean()
        .reset_index()
        .rename(columns={"shieldDegrees": "target_shield_for_new_noise"})
    )
    return targets


def extract_overshoot_for_block(
    block_data: pd.DataFrame,
    targets: pd.DataFrame,
    fsample: int,
    baseline_sec: float,
    post_sec: float,
) -> list[dict]:
    block_data = block_data.sort_values("currentFrame").reset_index(drop=True)

    subject = block_data["subject"].iloc[0]

    variance = block_data["trueVariance"].to_numpy(dtype=float)
    shield = block_data["shieldDegrees"].to_numpy(dtype=float)
    frames = block_data["currentFrame"].to_numpy(dtype=int)

    baseline_frames = int(round(baseline_sec * fsample))
    post_frames = int(round(post_sec * fsample))

    noise_change_idx = np.where(np.diff(variance) != 0)[0] + 1
    rows = []

    for cp_idx in noise_change_idx:
        old_noise = variance[cp_idx - 1]
        new_noise = variance[cp_idx]

        if np.isnan(old_noise) or np.isnan(new_noise):
            continue

        cp_frame = frames[cp_idx]

        baseline_mask = (frames >= cp_frame - baseline_frames) & (frames < cp_frame)
        post_mask = (frames >= cp_frame) & (frames <= cp_frame + post_frames)

        if baseline_mask.sum() < 2 or post_mask.sum() < 2:
            continue

        target_row = targets[
            (targets["subject"] == subject)
            & np.isclose(targets["trueVariance"], new_noise)
        ]

        if target_row.empty:
            continue

        target_shield = float(target_row["target_shield_for_new_noise"].iloc[0])

        baseline_shield = float(np.nanmean(shield[baseline_mask]))
        post_shield = shield[post_mask]

        if new_noise > old_noise:
            direction = "noise increase"
            extreme_shield = float(np.nanmax(post_shield))
            overshoot_relative_to_target = extreme_shield - target_shield
            response_amplitude = extreme_shield - baseline_shield
        elif new_noise < old_noise:
            direction = "noise decrease"
            extreme_shield = float(np.nanmin(post_shield))
            overshoot_relative_to_target = target_shield - extreme_shield
            response_amplitude = baseline_shield - extreme_shield
        else:
            continue

        rows.append(
            {
                "subject": subject,
                "sessID": int(block_data["sessID"].iloc[0]),
                "blockID": int(block_data["blockID"].iloc[0]),
                "noise_cp_frame": int(cp_frame),
                "old_noise": float(old_noise),
                "new_noise": float(new_noise),
                "noise_transition": f"{old_noise:g}->{new_noise:g}",
                "direction": direction,
                "baseline_shield": baseline_shield,
                "target_shield_for_new_noise": target_shield,
                "extreme_shield_after_cp": extreme_shield,
                "response_amplitude": response_amplitude,
                "overshoot_relative_to_target": overshoot_relative_to_target,
                "is_overshoot": bool(overshoot_relative_to_target > 0),
                "volatility_label": block_data["volatility_label"].iloc[0],
            }
        )

    return rows


def extract_all_overshoots(
    all_data: pd.DataFrame,
    targets: pd.DataFrame,
    fsample: int,
    baseline_sec: float,
    post_sec: float,
) -> pd.DataFrame:
    rows = []

    for _, block_data in all_data.groupby(["subject", "sessID", "blockID"], observed=True):
        rows.extend(
            extract_overshoot_for_block(
                block_data=block_data,
                targets=targets,
                fsample=fsample,
                baseline_sec=baseline_sec,
                post_sec=post_sec,
            )
        )

    return pd.DataFrame(rows)


def summarize_overshoots(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()

    summary = (
        events.groupby(["volatility_label", "noise_transition", "direction"], observed=True)
        .agg(
            n_events=("overshoot_relative_to_target", "size"),
            mean_response_amplitude=("response_amplitude", "mean"),
            mean_overshoot=("overshoot_relative_to_target", "mean"),
            median_overshoot=("overshoot_relative_to_target", "median"),
            p_overshoot=("is_overshoot", "mean"),
        )
        .reset_index()
    )

    return summary


def summarize_subject_overshoots(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()

    subject_summary = (
        events.groupby(["subject", "noise_transition", "direction"], observed=True)
        .agg(
            n_events=("overshoot_relative_to_target", "size"),
            mean_overshoot=("overshoot_relative_to_target", "mean"),
            p_overshoot=("is_overshoot", "mean"),
            mean_response_amplitude=("response_amplitude", "mean"),
        )
        .reset_index()
    )

    return subject_summary


def plot_overshoot_summary(summary: pd.DataFrame, output_path: Path) -> None:
    if summary.empty:
        return

    summary = summary.copy()
    summary = summary.sort_values(["volatility_label", "direction", "noise_transition"])

    labels = summary["noise_transition"].tolist()
    values = summary["mean_overshoot"].to_numpy(dtype=float)
    n_values = summary["n_events"].to_numpy(dtype=float)
    volatility_labels = summary["volatility_label"].tolist()

    n_min = np.nanmin(n_values)
    n_max = np.nanmax(n_values)

    if n_max == n_min:
        color_strength = np.ones_like(n_values) * 0.7
    else:
        color_strength = (n_values - n_min) / (n_max - n_min)

    cmap = plt.cm.Blues
    colors = cmap(0.25 + 0.75 * color_strength)

    fig, ax = plt.subplots(figsize=(13, 6.5))
    x = np.arange(len(labels))

    ax.bar(
        x,
        values,
        color=colors,
        edgecolor="black",
        linewidth=1.2,
    )

    ax.axhline(0, color="black", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Overshoot relative to target shield size (degrees)")
    ax.set_xlabel("Noise changepoint")
    ax.set_title("Shield-size overshoot after noise changepoints")
    ax.grid(axis="y", alpha=0.25)

    ymax = np.nanmax(values)
    ymin = np.nanmin(values)
    ax.set_ylim(
        min(0, ymin) - 1.0,
        ymax + 2.2,
    )

    for i, n in enumerate(n_values):
        y = values[i]
        ax.text(
            i,
            y + 0.45,
            f"n={int(n)}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    unique_vols = []
    for v in volatility_labels:
        if v not in unique_vols:
            unique_vols.append(v)

    y_top = ymax + 1.45

    for vol in unique_vols:
        idx = [i for i, v in enumerate(volatility_labels) if v == vol]

        if not idx:
            continue

        start = min(idx)
        end = max(idx)

        ax.plot(
            [start - 0.4, end + 0.4],
            [y_top, y_top],
            color="black",
            linewidth=1.2,
        )

        ax.text(
            (start + end) / 2,
            y_top + 0.15,
            vol,
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

        if end < len(labels) - 1:
            ax.axvline(
                end + 0.5,
                color="black",
                linestyle="--",
                linewidth=1.0,
                alpha=0.5,
            )

    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=plt.Normalize(vmin=n_min, vmax=n_max),
    )
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Number of noise changepoints")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse shield-size overshoot relative to target shield size for the new noise level."
    )

    parser.add_argument("--main-dir", type=str, default=None)
    parser.add_argument("--subject-ids", nargs="*", type=int, default=DEFAULT_SUBJECT_IDS)
    parser.add_argument("--baseline-sec", type=float, default=5.0)
    parser.add_argument("--post-sec", type=float, default=10.0)

    args = parser.parse_args()

    options = coins_options(args.main_dir)

    frames = []
    for sub_id in args.subject_ids:
        data = load_subject_data(sub_id, options)
        if data is not None:
            frames.append(data)

    if not frames:
        raise RuntimeError("No subject data could be loaded.")

    all_data = pd.concat(frames, ignore_index=True)

    targets = compute_subject_noise_targets(all_data)

    events = extract_all_overshoots(
        all_data=all_data,
        targets=targets,
        fsample=options.behav.fsample,
        baseline_sec=args.baseline_sec,
        post_sec=args.post_sec,
    )

    summary = summarize_overshoots(events)
    subject_summary = summarize_subject_overshoots(events)

    output_dir = Path(options.workDir) / "behav" / "group"
    ensure_dir(output_dir)

    suffix = f"baseline_{args.baseline_sec:g}s_post_{args.post_sec:g}s"

    targets_path = output_dir / f"shield_noise_targets_{suffix}.csv"
    events_path = output_dir / f"shield_overshoot_events_{suffix}.csv"
    summary_path = output_dir / f"shield_overshoot_summary_{suffix}.csv"
    subject_summary_path = output_dir / f"shield_overshoot_subject_summary_{suffix}.csv"
    figure_path = output_dir / f"shield_overshoot_summary_{suffix}.png"

    targets.to_csv(targets_path, index=False)
    events.to_csv(events_path, index=False)
    summary.to_csv(summary_path, index=False)
    subject_summary.to_csv(subject_summary_path, index=False)

    plot_overshoot_summary(summary, figure_path)

    print()
    print("Done.")
    print(f"Events analysed: {len(events)}")
    print()
    print("Summary:")
    print(summary.to_string(index=False))
    print()
    print("Saved files:")
    print(targets_path)
    print(events_path)
    print(summary_path)
    print(subject_summary_path)
    print(figure_path)


if __name__ == "__main__":
    main()