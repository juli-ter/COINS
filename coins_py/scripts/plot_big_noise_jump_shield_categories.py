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
SHIELD_LEVELS = [20, 40, 60]


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

    required = {
        "sessID",
        "blockID",
        "currentFrame",
        "trueVariance",
        "shieldDegrees",
        "volatility",
    }

    missing = required.difference(data.columns)

    if missing:
        print(f"[sub-{sub_id:02d}] skipped: missing columns {sorted(missing)}")
        return None

    data["subject"] = details["subjName"]
    data["volatility_label"] = data["volatility"].map({0: "Stable", 1: "Volatile"})

    return data.reset_index(drop=True)


def classify_big_noise_jump_response(
    old_noise: float,
    new_noise: float,
    post_shield: np.ndarray,
    noise_levels: list[float],
) -> tuple[str, float]:
    low_noise = noise_levels[0]
    high_noise = noise_levels[-1]

    low_shield = SHIELD_LEVELS[0]
    medium_shield = SHIELD_LEVELS[1]
    high_shield = SHIELD_LEVELS[2]

    if old_noise == low_noise and new_noise == high_noise:
        extreme_shield = float(np.nanmax(post_shield))

        if extreme_shield <= low_shield:
            return "Stayed low", extreme_shield
        if extreme_shield < high_shield:
            return "Partial: only medium", extreme_shield
        return "Reached high", extreme_shield

    if old_noise == high_noise and new_noise == low_noise:
        extreme_shield = float(np.nanmin(post_shield))

        if extreme_shield >= high_shield:
            return "Stayed high", extreme_shield
        if extreme_shield > low_shield:
            return "Partial: only medium", extreme_shield
        return "Reached low", extreme_shield

    return "Other", np.nan


def extract_big_jump_events(
    all_data: pd.DataFrame,
    fsample: int,
    post_sec: float,
) -> pd.DataFrame:
    noise_levels = sorted(all_data["trueVariance"].dropna().unique().tolist())

    if len(noise_levels) < 3:
        raise ValueError(
            f"Expected at least three noise levels, found: {noise_levels}"
        )

    low_noise = noise_levels[0]
    high_noise = noise_levels[-1]

    post_frames = int(round(post_sec * fsample))
    rows = []

    for _, block_data in all_data.groupby(["subject", "sessID", "blockID"], observed=True):
        block_data = block_data.sort_values("currentFrame").reset_index(drop=True)

        variance = block_data["trueVariance"].to_numpy(dtype=float)
        shield = block_data["shieldDegrees"].to_numpy(dtype=float)
        frames = block_data["currentFrame"].to_numpy(dtype=int)

        noise_change_idx = np.where(np.diff(variance) != 0)[0] + 1

        for cp_idx in noise_change_idx:
            old_noise = variance[cp_idx - 1]
            new_noise = variance[cp_idx]

            if np.isnan(old_noise) or np.isnan(new_noise):
                continue

            is_big_jump = (
                (old_noise == low_noise and new_noise == high_noise)
                or (old_noise == high_noise and new_noise == low_noise)
            )

            if not is_big_jump:
                continue

            cp_frame = frames[cp_idx]
            window_mask = (frames >= cp_frame) & (frames <= cp_frame + post_frames)

            if window_mask.sum() < 2:
                continue

            post_shield = shield[window_mask]

            category, extreme_shield = classify_big_noise_jump_response(
                old_noise=old_noise,
                new_noise=new_noise,
                post_shield=post_shield,
                noise_levels=noise_levels,
            )

            if old_noise < new_noise:
                transition_type = "Noise increase"
            else:
                transition_type = "Noise decrease"

            rows.append(
                {
                    "subject": block_data["subject"].iloc[0],
                    "volatility_label": block_data["volatility_label"].iloc[0],
                    "sessID": int(block_data["sessID"].iloc[0]),
                    "blockID": int(block_data["blockID"].iloc[0]),
                    "noise_cp_frame": int(cp_frame),
                    "old_noise": float(old_noise),
                    "new_noise": float(new_noise),
                    "noise_transition": f"{old_noise:g}->{new_noise:g}",
                    "transition_type": transition_type,
                    "shield_at_cp": float(shield[cp_idx]),
                    "extreme_shield_after_cp": extreme_shield,
                    "response_category": category,
                }
            )

    return pd.DataFrame(rows)


def summarize_events(events: pd.DataFrame) -> pd.DataFrame:
    counts = (
        events.groupby(
            ["volatility_label", "noise_transition", "transition_type", "response_category"],
            observed=True,
        )
        .size()
        .reset_index(name="n")
    )

    total = counts.groupby(
        ["volatility_label", "noise_transition"],
        observed=True,
    )["n"].transform("sum")

    counts["percent"] = 100 * counts["n"] / total

    return counts


def plot_summary(summary: pd.DataFrame, output_path: Path) -> None:
    if summary.empty:
        return

    plot_order = [
        ("Stable", "10->30"),
        ("Volatile", "10->30"),
        ("Stable", "30->10"),
        ("Volatile", "30->10"),
    ]

    category_order_by_transition = {
        "10->30": ["Stayed low", "Partial: only medium", "Reached high"],
        "30->10": ["Stayed high", "Partial: only medium", "Reached low"],
    }

    labels = []
    data_rows = []

    for volatility_label, transition in plot_order:
        current = summary[
            (summary["volatility_label"] == volatility_label)
            & (summary["noise_transition"] == transition)
        ].copy()

        row = {
            "label": f"{transition}\n{volatility_label}",
            "total_n": int(current["n"].sum()) if not current.empty else 0,
        }

        for category in category_order_by_transition[transition]:
            value = current.loc[current["response_category"] == category, "percent"]
            row[category] = float(value.iloc[0]) if not value.empty else 0.0

        labels.append(row["label"])
        data_rows.append(row)

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(data_rows))
    bottom = np.zeros(len(data_rows))

    all_categories = [
        "Stayed low",
        "Stayed high",
        "Partial: only medium",
        "Reached high",
        "Reached low",
    ]

    for category in all_categories:
        values = np.array([row.get(category, 0.0) for row in data_rows])

        if np.all(values == 0):
            continue

        ax.bar(
            x,
            values,
            bottom=bottom,
            label=category,
            edgecolor="black",
            linewidth=0.8,
        )

        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Percent of big noise changepoints")
    ax.set_xlabel("Noise transition and block type")
    ax.set_title("Categorical shield-size responses to large noise jumps")
    ax.grid(axis="y", alpha=0.25)

    for i, row in enumerate(data_rows):
        ax.text(
            i,
            102,
            f"n={row['total_n']}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.legend(title="Response category", bbox_to_anchor=(1.02, 1), loc="upper left")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Categorise shield-size responses after large noise jumps 10->30 and 30->10."
    )

    parser.add_argument("--main-dir", type=str, default=None)
    parser.add_argument("--subject-ids", nargs="*", type=int, default=DEFAULT_SUBJECT_IDS)
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

    events = extract_big_jump_events(
        all_data=all_data,
        fsample=options.behav.fsample,
        post_sec=args.post_sec,
    )

    summary = summarize_events(events)

    output_dir = Path(options.workDir) / "behav" / "group"
    ensure_dir(output_dir)

    suffix = f"post_{args.post_sec:g}s"

    events_path = output_dir / f"big_noise_jump_shield_categories_events_{suffix}.csv"
    summary_path = output_dir / f"big_noise_jump_shield_categories_summary_{suffix}.csv"
    figure_path = output_dir / f"big_noise_jump_shield_categories_{suffix}.png"

    events.to_csv(events_path, index=False)
    summary.to_csv(summary_path, index=False)

    plot_summary(summary, figure_path)

    print()
    print("Done.")
    print(f"Events analysed: {len(events)}")
    print()
    print("Summary:")
    print(summary.to_string(index=False))
    print()
    print("Saved files:")
    print(events_path)
    print(summary_path)
    print(figure_path)


if __name__ == "__main__":
    main()