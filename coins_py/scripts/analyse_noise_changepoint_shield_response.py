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
    data["subject"] = details["subjName"]

    required = {"subject", "sessID", "blockID", "currentFrame", "trueVariance", "trueMean", "shieldDegrees"}
    missing = required.difference(data.columns)

    if missing:
        print(f"[sub-{sub_id:02d}] skipped: missing columns {sorted(missing)}")
        return None

    return data.reset_index(drop=True)


def find_noise_changepoints_for_block(
    block_data: pd.DataFrame,
    fsample: int,
    window_sec: float,
    mean_exclusion_sec: float,
) -> list[dict]:
    block_data = block_data.sort_values("currentFrame").reset_index(drop=True)

    variance = block_data["trueVariance"].to_numpy(dtype=float)
    mean = block_data["trueMean"].to_numpy(dtype=float)
    shield = block_data["shieldDegrees"].to_numpy(dtype=float)
    frames = block_data["currentFrame"].to_numpy(dtype=int)

    window_frames = int(round(window_sec * fsample))
    mean_exclusion_frames = int(round(mean_exclusion_sec * fsample))

    events = []

    noise_change_idx = np.where(np.diff(variance) != 0)[0] + 1

    for idx in noise_change_idx:
        old_noise = variance[idx - 1]
        new_noise = variance[idx]

        if np.isnan(old_noise) or np.isnan(new_noise):
            continue

        if new_noise > old_noise:
            expected_direction = "increase"
        elif new_noise < old_noise:
            expected_direction = "decrease"
        else:
            continue

        start_frame = frames[idx]
        end_frame = start_frame + window_frames

        window_mask = (frames >= start_frame) & (frames <= end_frame)
        window_indices = np.where(window_mask)[0]

        if len(window_indices) < 2:
            continue

        shield_window = shield[window_indices]
        shield_diff = np.diff(shield_window)

        increase_positions = np.where(shield_diff > 0)[0] + 1
        decrease_positions = np.where(shield_diff < 0)[0] + 1

        any_increase = len(increase_positions) > 0
        any_decrease = len(decrease_positions) > 0

        if expected_direction == "increase":
            correct_positions = increase_positions
            wrong_positions = decrease_positions
        else:
            correct_positions = decrease_positions
            wrong_positions = increase_positions

        correct_response = len(correct_positions) > 0
        wrong_response = len(wrong_positions) > 0
        any_size_change = any_increase or any_decrease

        if correct_response:
            first_correct_local_idx = correct_positions[0]
            first_correct_global_idx = window_indices[first_correct_local_idx]
            latency_frames = frames[first_correct_global_idx] - start_frame
            latency_sec = latency_frames / float(fsample)
        else:
            latency_frames = np.nan
            latency_sec = np.nan

        if mean_exclusion_frames > 0:
            mean_window_mask = (
                (frames >= start_frame - mean_exclusion_frames)
                & (frames <= start_frame + mean_exclusion_frames)
            )
            mean_near = mean[mean_window_mask]
            mean_change_nearby = bool(np.any(np.diff(mean_near) != 0)) if len(mean_near) > 1 else False
        else:
            mean_change_nearby = False

        events.append(
            {
                "subject": block_data["subject"].iloc[0],
                "sessID": int(block_data["sessID"].iloc[0]),
                "blockID": int(block_data["blockID"].iloc[0]),
                "noise_cp_frame": int(start_frame),
                "old_noise": float(old_noise),
                "new_noise": float(new_noise),
                "noise_transition": f"{old_noise:g}->{new_noise:g}",
                "expected_shield_response": expected_direction,
                "shield_at_cp": float(shield[idx]),
                "shield_end_window": float(shield_window[-1]),
                "net_shield_change": float(shield_window[-1] - shield_window[0]),
                "any_size_change": bool(any_size_change),
                "any_increase": bool(any_increase),
                "any_decrease": bool(any_decrease),
                "correct_response": bool(correct_response),
                "wrong_response": bool(wrong_response),
                "latency_frames": latency_frames,
                "latency_sec": latency_sec,
                "mean_change_nearby": bool(mean_change_nearby),
            }
        )

    return events


def extract_noise_changepoint_events(
    all_data: pd.DataFrame,
    fsample: int,
    window_sec: float,
    mean_exclusion_sec: float,
) -> pd.DataFrame:
    all_events = []

    group_cols = ["subject", "sessID", "blockID"]

    for _, block_data in all_data.groupby(group_cols, observed=True):
        events = find_noise_changepoints_for_block(
            block_data=block_data,
            fsample=fsample,
            window_sec=window_sec,
            mean_exclusion_sec=mean_exclusion_sec,
        )
        all_events.extend(events)

    return pd.DataFrame(all_events)


def summarize_events(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()

    summary = (
        events.groupby(["noise_transition", "expected_shield_response"], observed=True)
        .agg(
            n_events=("correct_response", "size"),
            n_correct=("correct_response", "sum"),
            n_wrong=("wrong_response", "sum"),
            n_any_change=("any_size_change", "sum"),
            p_correct=("correct_response", "mean"),
            p_wrong=("wrong_response", "mean"),
            p_any_change=("any_size_change", "mean"),
            median_latency_sec=("latency_sec", "median"),
            mean_latency_sec=("latency_sec", "mean"),
        )
        .reset_index()
    )

    return summary


def summarize_subject_events(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()

    subject_summary = (
        events.groupby(["subject", "noise_transition", "expected_shield_response"], observed=True)
        .agg(
            n_events=("correct_response", "size"),
            p_correct=("correct_response", "mean"),
            p_any_change=("any_size_change", "mean"),
            median_latency_sec=("latency_sec", "median"),
        )
        .reset_index()
    )

    return subject_summary


def plot_transition_summary(summary: pd.DataFrame, output_path: Path) -> None:
    if summary.empty:
        return

    summary = summary.copy()
    summary = summary.sort_values(["expected_shield_response", "noise_transition"])

    labels = summary["noise_transition"].tolist()
    values = summary["p_correct"].to_numpy(dtype=float)
    n_values = summary["n_events"].to_numpy(dtype=float)

    n_min = np.nanmin(n_values)
    n_max = np.nanmax(n_values)

    if n_max == n_min:
        color_values = np.ones_like(n_values) * 0.7
    else:
        color_values = (n_values - n_min) / (n_max - n_min)

    cmap = plt.cm.Blues
    colors = cmap(0.25 + 0.75 * color_values)

    fig, ax = plt.subplots(figsize=(10, 5.5))

    x = np.arange(len(labels))

    bars = ax.bar(
        x,
        values,
        color=colors,
        edgecolor="black",
        linewidth=1.2,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("P(correct shield-size response)")
    ax.set_xlabel("Noise changepoint")
    ax.set_title("Shield-size responses after noise changepoints")
    ax.grid(axis="y", alpha=0.25)

    for i, row in summary.reset_index(drop=True).iterrows():
        ax.text(
            i,
            row["p_correct"] + 0.03,
            f"n={int(row['n_events'])}",
            ha="center",
            va="bottom",
            fontsize=9,
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
        description="Count shield-size responses after trueVariance/noise changepoints."
    )

    parser.add_argument(
        "--main-dir",
        type=str,
        default=None,
        help="Main COINS folder.",
    )

    parser.add_argument(
        "--subject-ids",
        nargs="*",
        type=int,
        default=DEFAULT_SUBJECT_IDS,
        help="Subjects to include. Default: 1..22",
    )

    parser.add_argument(
        "--window-sec",
        type=float,
        default=10.0,
        help="Time window after noise changepoint in seconds. Default: 10.",
    )

    parser.add_argument(
        "--mean-exclusion-sec",
        type=float,
        default=0.0,
        help=(
            "If > 0, marks whether a trueMean changepoint occurred within this many "
            "seconds around the noise changepoint. Default: 0."
        ),
    )

    parser.add_argument(
        "--clean-only",
        action="store_true",
        help="If used, exclude events with nearby trueMean changepoints.",
    )

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

    events = extract_noise_changepoint_events(
        all_data=all_data,
        fsample=options.behav.fsample,
        window_sec=args.window_sec,
        mean_exclusion_sec=args.mean_exclusion_sec,
    )

    if args.clean_only:
        events = events[~events["mean_change_nearby"]].copy()

    summary = summarize_events(events)
    subject_summary = summarize_subject_events(events)

    output_dir = Path(options.workDir) / "behav" / "group"
    ensure_dir(output_dir)

    suffix = f"window_{args.window_sec:g}s"
    if args.clean_only:
        suffix += "_clean_only"

    events_path = output_dir / f"noise_changepoint_shield_events_{suffix}.csv"
    summary_path = output_dir / f"noise_changepoint_shield_summary_{suffix}.csv"
    subject_summary_path = output_dir / f"noise_changepoint_shield_subject_summary_{suffix}.csv"
    figure_path = output_dir / f"noise_changepoint_shield_response_{suffix}.png"

    events.to_csv(events_path, index=False)
    summary.to_csv(summary_path, index=False)
    subject_summary.to_csv(subject_summary_path, index=False)

    plot_transition_summary(summary, figure_path)

    print()
    print("Done.")
    print(f"Window after noise changepoint: {args.window_sec} seconds")
    print(f"Number of noise changepoint events: {len(events)}")

    print()
    print("Summary:")
    if not summary.empty:
        print(summary.to_string(index=False))
    else:
        print("No events found.")

    print()
    print("Saved files:")
    print(events_path)
    print(summary_path)
    print(subject_summary_path)
    print(figure_path)


if __name__ == "__main__":
    main()