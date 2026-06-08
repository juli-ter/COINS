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

    required = {"subject", "sessID", "blockID", "currentFrame", "trueVariance", "shieldDegrees"}
    missing = required.difference(data.columns)

    if missing:
        print(f"[sub-{sub_id:02d}] skipped: missing columns {sorted(missing)}")
        return None

    return data.reset_index(drop=True)


def extract_curves_for_block(
    block_data: pd.DataFrame,
    pre_sec: float,
    post_sec: float,
    fsample: int,
) -> list[dict]:
    block_data = block_data.sort_values("currentFrame").reset_index(drop=True)

    variance = block_data["trueVariance"].to_numpy(dtype=float)
    shield = block_data["shieldDegrees"].to_numpy(dtype=float)
    frames = block_data["currentFrame"].to_numpy(dtype=int)

    pre_frames = int(round(pre_sec * fsample))
    post_frames = int(round(post_sec * fsample))

    noise_change_idx = np.where(np.diff(variance) != 0)[0] + 1

    rows = []

    for cp_idx in noise_change_idx:
        old_noise = variance[cp_idx - 1]
        new_noise = variance[cp_idx]

        if np.isnan(old_noise) or np.isnan(new_noise):
            continue

        cp_frame = frames[cp_idx]

        window_start = cp_frame - pre_frames
        window_end = cp_frame + post_frames

        if frames[0] > window_start or frames[-1] < window_end:
            continue

        transition = f"{old_noise:g}->{new_noise:g}"

        if new_noise > old_noise:
            direction = "noise increase"
        elif new_noise < old_noise:
            direction = "noise decrease"
        else:
            continue

        for rel_frame in range(-pre_frames, post_frames + 1):
            target_frame = cp_frame + rel_frame

            nearest_idx = np.argmin(np.abs(frames - target_frame))

            if abs(frames[nearest_idx] - target_frame) > 1:
                continue

            rows.append(
                {
                    "subject": block_data["subject"].iloc[0],
                    "sessID": int(block_data["sessID"].iloc[0]),
                    "blockID": int(block_data["blockID"].iloc[0]),
                    "noise_cp_frame": int(cp_frame),
                    "old_noise": float(old_noise),
                    "new_noise": float(new_noise),
                    "noise_transition": transition,
                    "direction": direction,
                    "relative_frame": int(rel_frame),
                    "relative_sec": rel_frame / float(fsample),
                    "shieldDegrees": float(shield[nearest_idx]),
                    "baseline_shield": float(shield[cp_idx]),
                    "shield_change_from_cp": float(shield[nearest_idx] - shield[cp_idx]),
                }
            )

    return rows


def extract_all_curves(
    all_data: pd.DataFrame,
    pre_sec: float,
    post_sec: float,
    fsample: int,
) -> pd.DataFrame:
    rows = []

    for _, block_data in all_data.groupby(["subject", "sessID", "blockID"], observed=True):
        rows.extend(
            extract_curves_for_block(
                block_data=block_data,
                pre_sec=pre_sec,
                post_sec=post_sec,
                fsample=fsample,
            )
        )

    return pd.DataFrame(rows)


def summarize_curves(curves: pd.DataFrame) -> pd.DataFrame:
    if curves.empty:
        return pd.DataFrame()

    subject_curves = (
        curves.groupby(
            ["subject", "noise_transition", "direction", "relative_sec"],
            observed=True,
        )["shieldDegrees"]
        .mean()
        .reset_index()
    )

    summary = (
        subject_curves.groupby(
            ["noise_transition", "direction", "relative_sec"],
            observed=True,
        )["shieldDegrees"]
        .agg(
            mean_shield_degrees="mean",
            sem_shield_degrees=lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan,
            n_subjects="count",
        )
        .reset_index()
    )

    return summary


def plot_curves(summary: pd.DataFrame, output_path: Path) -> None:
    if summary.empty:
        return

    transitions_increase = ["10->20", "20->30", "10->30"]
    transitions_decrease = ["30->20", "20->10", "30->10"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

    for ax, transitions, title in zip(
        axes,
        [transitions_increase, transitions_decrease],
        ["Noise increase", "Noise decrease"],
    ):
        for transition in transitions:
            current = summary[summary["noise_transition"] == transition].copy()

            if current.empty:
                continue

            current = current.sort_values("relative_sec")

            x = current["relative_sec"].to_numpy(dtype=float)
            y = current["mean_shield_degrees"].to_numpy(dtype=float)
            sem = current["sem_shield_degrees"].to_numpy(dtype=float)

            ax.plot(x, y, linewidth=2.0, label=transition)
            ax.fill_between(x, y - sem, y + sem, alpha=0.18)

        ax.axvline(0, color="black", linewidth=1.2, linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("Time from noise changepoint (s)")
        ax.set_ylabel("Mean shield size (degrees)")
        ax.set_ylim(15, 65)
        ax.set_yticks([20, 40, 60])
        ax.grid(alpha=0.25)
        ax.legend(title="Transition")

    fig.suptitle("Shield size adaptation around noise changepoints", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot shieldDegrees trajectories around trueVariance/noise changepoints."
    )

    parser.add_argument("--main-dir", type=str, default=None)
    parser.add_argument("--subject-ids", nargs="*", type=int, default=DEFAULT_SUBJECT_IDS)
    parser.add_argument("--pre-sec", type=float, default=5.0)
    parser.add_argument("--post-sec", type=float, default=15.0)

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

    curves = extract_all_curves(
        all_data=all_data,
        pre_sec=args.pre_sec,
        post_sec=args.post_sec,
        fsample=options.behav.fsample,
    )

    summary = summarize_curves(curves)

    output_dir = Path(options.workDir) / "behav" / "group"
    ensure_dir(output_dir)

    suffix = f"pre_{args.pre_sec:g}s_post_{args.post_sec:g}s"

    curves_path = output_dir / f"noise_changepoint_shield_curves_long_{suffix}.csv"
    summary_path = output_dir / f"noise_changepoint_shield_curves_summary_{suffix}.csv"
    figure_path = output_dir / f"noise_changepoint_shield_curves_{suffix}.png"

    curves.to_csv(curves_path, index=False)
    summary.to_csv(summary_path, index=False)

    plot_curves(summary, figure_path)

    print()
    print("Done.")
    print(f"Extracted curve rows: {len(curves)}")
    print()
    print("Saved files:")
    print(curves_path)
    print(summary_path)
    print(figure_path)


if __name__ == "__main__":
    main()