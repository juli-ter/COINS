from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .adjustments import coins_compute_block_adjustments, coins_compute_block_reaction_times, count_changepoints_by_noise
from .io import coins_load_subject_data
from .kernels import (
    coins_block_kernels_regression,
    coins_compute_block_kernels,
    coins_compute_sessionwise_regression_kernels,
)
from .options import Options, coins_options
from .performance import coins_compute_tracking_performance
from .plots import (
    coins_plot_block_changepoint_counts,
    coins_plot_group_changepoint_counts,
    coins_plot_block_data,
    coins_plot_group_adjustments,
    coins_plot_participant_performance,
    coins_plot_performance_overview,
    coins_plot_subject_adjustments,
    coins_plot_subject_betas_by_volatility,
    coins_plot_subject_kernels_by_volatility,
    coins_plot_subject_reg_kernels_session_wise,
    save_figure,
)
from .subjects import coins_subjects
from .utils import ensure_dir, load_pickle, save_pickle


def _excluded(perform: list[list[dict | None]], excluded_blocks: list[list[int]]) -> list[list[dict | None]]:
    out = [row[:] for row in perform]
    for sess, block in excluded_blocks:
        out[sess - 1][block - 1] = None
    return out


def coins_subject_performance(details: dict, sub_data: pd.DataFrame, options: Options) -> tuple[list[list[dict]], list[list[dict]]]:
    stim = [[None for _ in range(4)] for _ in range(details['nSessions'])]
    perform = [[None for _ in range(4)] for _ in range(details['nSessions'])]
    for i_sess in range(1, details['nSessions'] + 1):
        for i_block in range(1, 5):
            block_data = sub_data[(sub_data['sessID'] == i_sess) & (sub_data['blockID'] == i_block)]
            if block_data.empty:
                continue
            fig = coins_plot_block_data(block_data, options)
            save_figure(fig, details['analysis']['behav']['blockFigures'][(i_sess, i_block)])
            stim[i_sess - 1][i_block - 1], perform[i_sess - 1][i_block - 1] = coins_compute_tracking_performance(block_data, options)
    save_pickle(details['analysis']['behav']['performance'], {'stim': stim, 'perform': perform})
    return stim, perform


def coins_subject_kernels(details: dict, sub_data: pd.DataFrame, options: Options) -> None:
    betas_con, n_trials_con = coins_block_kernels_regression(sub_data, details['excludedBlocks'], options)
    save_pickle(details['analysis']['behav']['blockRegKernels'], {'betasCon': betas_con, 'nTrialsCon': n_trials_con})
    fig = coins_plot_subject_betas_by_volatility(betas_con, n_trials_con, options)
    save_figure(fig, details['analysis']['behav']['regBetaConditionPlot'])


def coins_subject_kernels_session_wise(details: dict, sub_data: pd.DataFrame, options: Options) -> None:
    for n_samp, out_key, fig_key in [(3, 'sessRegKernels3', 'figSessRegKernels3'), (5, 'sessRegKernels5', 'figSessRegKernels5')]:
        options.behav.flagRegKernelSamples = n_samp
        vol_betas = []
        sta_betas = []
        n_trials_vol = []
        n_trials_sta = []
        for i_sess in range(1, details['nSessions'] + 1):
            sess_data = sub_data[sub_data['sessID'] == i_sess]
            vb, sb, ntv, nts = coins_compute_sessionwise_regression_kernels(sess_data, options)
            vol_betas.append(vb)
            sta_betas.append(sb)
            n_trials_vol.append(ntv)
            n_trials_sta.append(nts)
        vol_betas_arr = np.vstack(vol_betas)
        sta_betas_arr = np.vstack(sta_betas)
        save_pickle(details['analysis']['behav'][out_key], {
            'volBetas': vol_betas_arr,
            'staBetas': sta_betas_arr,
            'nTrialsVol': np.asarray(n_trials_vol),
            'nTrialsSta': np.asarray(n_trials_sta),
        })
        fig = coins_plot_subject_reg_kernels_session_wise(vol_betas_arr, sta_betas_arr)
        save_figure(fig, details['analysis']['behav'][fig_key])


def coins_analyse_subject_behaviour(subID: int, options: Options | None = None) -> dict:
    block_noise_records = []
    if options is None:
        options = coins_options()
    details = coins_subjects(subID, options)
    ensure_dir(details['analysis']['behav']['folder'])

    if options.behav.flagLoadData:
        sub_data = coins_load_subject_data(details)
        save_pickle(details['analysis']['behav']['responseData'], {'subData': sub_data})
    else:
        sub_data = load_pickle(details['analysis']['behav']['responseData'])['subData']

    if options.behav.flagPerformance:
        stim, perform = coins_subject_performance(details, sub_data, options)
    else:
        perf_obj = load_pickle(details['analysis']['behav']['performance'])
        stim, perform = perf_obj['stim'], perf_obj['perform']

    masked_perform = _excluded(perform, details['excludedBlocks'])
    if options.behav.flagPerformance:
        save_figure(coins_plot_participant_performance(masked_perform), details['analysis']['behav']['performancePlot'])
        save_figure(coins_plot_performance_overview(masked_perform, 'reward'), details['analysis']['behav']['performRewardFig'])
        save_figure(coins_plot_performance_overview(masked_perform, 'meanPosPE'), details['analysis']['behav']['performAvgPeFig'])
        save_figure(coins_plot_performance_overview(masked_perform, 'meanDiff2mean'), details['analysis']['behav']['performAvgDevFromMeanFig'])

    if options.behav.flagKernels:
        kernel_len = options.behav.kernelPreSamples + options.behav.kernelPostSamples + 1
        avg_kernels = np.full((details['nSessions'], 4, 6, kernel_len), np.nan)
        n_kernels = np.zeros((details['nSessions'], 4, 6))
        vol_kernels = np.full_like(avg_kernels, np.nan)
        sta_kernels = np.full_like(avg_kernels, np.nan)
        condition_labels = np.full((details['nSessions'], 4), np.nan)
        block_moves = {}
        for i_sess in range(1, details['nSessions'] + 1):
            for i_block in range(1, 5):
                block_data = sub_data[(sub_data['sessID'] == i_sess) & (sub_data['blockID'] == i_block)]
                if block_data.empty:
                    continue
                if [i_sess, i_block] in details['excludedBlocks']:
                    nk = np.zeros(6)
                    ak = np.full((6, kernel_len), np.nan)
                    bm = None
                else:
                    ak, nk, bm = coins_compute_block_kernels(block_data, options)
                avg_kernels[i_sess - 1, i_block - 1] = ak
                n_kernels[i_sess - 1, i_block - 1] = nk
                block_moves[(i_sess, i_block)] = bm
                vol_value = block_data['volatility'].iloc[0]
                vol = None if pd.isna(vol_value) else int(vol_value)
                condition_labels[i_sess - 1, i_block - 1] = vol
                if vol == 1:
                    vol_kernels[i_sess - 1, i_block - 1] = ak
                else:
                    sta_kernels[i_sess - 1, i_block - 1] = ak
        save_pickle(details['analysis']['behav']['blockKernels'], {'avgKernels': avg_kernels, 'nKernels': n_kernels, 'volKernels': vol_kernels, 'staKernels': sta_kernels})
        save_pickle(details['analysis']['behav']['blockMoves'], {'blockMoves': block_moves})
        n_move = n_kernels[:, :, 0]
        n_size = n_kernels[:, :, 1]
        n_left = n_kernels[:, :, 2]
        n_right = n_kernels[:, :, 3]
        n_up = n_kernels[:, :, 4]
        n_down = n_kernels[:, :, 5]
        n_responses = {
            'move': float(np.nansum(n_move)),
            'size': float(np.nansum(n_size)),
            'sizeUp': float(np.nansum(n_up)),
            'sizeDown': float(np.nansum(n_down)),
            'volatile': {
                'move': float(np.nansum(n_move[condition_labels == 1])),
                'size': float(np.nansum(n_size[condition_labels == 1])),
                'sizeUp': float(np.nansum(n_up[condition_labels == 1])),
                'sizeDown': float(np.nansum(n_down[condition_labels == 1])),
            },
            'stable': {
                'move': float(np.nansum(n_move[condition_labels == 0])),
                'size': float(np.nansum(n_size[condition_labels == 0])),
                'sizeUp': float(np.nansum(n_up[condition_labels == 0])),
                'sizeDown': float(np.nansum(n_down[condition_labels == 0])),
            },
        }
        save_pickle(details['analysis']['behav']['nResponses'], {'nResponses': n_responses})
    else:
        kernel_obj = load_pickle(details['analysis']['behav']['blockKernels'])
        vol_kernels = kernel_obj['volKernels']
        sta_kernels = kernel_obj['staKernels']
        n_responses = load_pickle(details['analysis']['behav']['nResponses'])['nResponses']

    fig = coins_plot_subject_kernels_by_volatility(sta_kernels, vol_kernels, n_responses, details, options)
    save_figure(fig, details['analysis']['behav']['kernelConditionPlot'])

    mean_adjusts = None
    median_adjusts = None
    jump_sizes = np.asarray([])
    block_counts_df = pd.DataFrame()


    if options.behav.flagAdjustments:
        all_adjusts = []
        vol_adjusts = []
        sta_adjusts = []
        all_jumps = []
        vol_jumps = []
        sta_jumps = []
        all_vars = []
        vol_vars = []
        sta_vars = []
        all_conds = []
        all_adjustments = {}
        all_jump_sizes = {}
        all_variances = {}
        for i_sess in range(1, details['nSessions'] + 1):
            for i_block in range(1, 5):
                block_data = sub_data[(sub_data['sessID'] == i_sess) & (sub_data['blockID'] == i_block)]
                if block_data.empty:
                    continue
                if [i_sess, i_block] in details['excludedBlocks']:
                    adj = np.empty((0, options.behav.adjustPreSamples + options.behav.adjustPostSamples + 1))
                    jumps = np.empty((0,))
                    vars_ = np.empty((0,))
                else:
                    adj, jumps, vars_, _ = coins_compute_block_adjustments(block_data, options)
                    vol_value = block_data["volatility"].iloc[0]
                    if pd.isna(vol_value):
                        volatility_label = "unknown"
                    else:
                        volatility_label = "volatile" if int(vol_value) == 1 else "stable"
                    sess_id = int(block_data["sessID"].iloc[0])
                    block_id = int(block_data["blockID"].iloc[0])
                    for var in vars_:
                        block_noise_records.append({
                            "subject": subID,
                            "session": sess_id,
                            "block": block_id,
                            "volatility": volatility_label,
                            "noise_value": var,
                            "excluded": bool(np.isnan(var)),
                        })
                all_adjustments[(i_sess, i_block)] = adj
                all_jump_sizes[(i_sess, i_block)] = jumps
                all_variances[(i_sess, i_block)] = vars_
                if len(adj):
                    all_adjusts.append(adj)
                if len(jumps):
                    all_jumps.append(jumps)
                    all_vars.append(vars_)
                    vol_value = block_data['volatility'].iloc[0]
                    if pd.isna(vol_value):
                        continue

                    vol = int(vol_value)
                    all_conds.append(np.full(len(jumps), vol))
                
                vol_value = block_data['volatility'].iloc[0]
                if pd.isna(vol_value):
                    continue

                if int(vol_value) == 1:


                    if len(adj):
                        vol_adjusts.append(adj)
                    if len(jumps):
                        vol_jumps.append(jumps)
                        vol_vars.append(vars_)
                else:
                    if len(adj):
                        sta_adjusts.append(adj)
                    if len(jumps):
                        sta_jumps.append(jumps)
                        sta_vars.append(vars_)

        # ============================================================
        # Count changepoints by local within-block noise level
        # ============================================================
        noise_events_df = pd.DataFrame(block_noise_records)

        if noise_events_df.empty:
            print(f"\nSubject {subID}: no changepoint noise records were found.")
            block_counts_df = pd.DataFrame()
        else:
            valid_mask = ~noise_events_df["excluded"]
            valid_noise_values = np.sort(
                noise_events_df.loc[valid_mask, "noise_value"]
                .dropna()
                .unique()
            )

            print(f"\nSubject {subID}: observed valid noise values in radians:")
            print(valid_noise_values)

            if len(valid_noise_values) != 3:
                print(
                    f"WARNING: expected 3 noise levels, "
                    f"but found {len(valid_noise_values)}: {valid_noise_values}"
                )

            noise_events_df["noise_category"] = "excluded_nan"

            noise_labels = ["low", "medium", "high"]

            for value, label in zip(valid_noise_values, noise_labels):
                current_mask = (
                    valid_mask
                    & np.isclose(noise_events_df["noise_value"], value)
                )
                noise_events_df.loc[current_mask, "noise_category"] = label

            remaining_valid = (
                valid_mask
                & (noise_events_df["noise_category"] == "excluded_nan")
            )
            noise_events_df.loc[remaining_valid, "noise_category"] = "unmapped_valid"

            block_counts_df = (
                noise_events_df
                .groupby(
                    ["subject", "session", "block", "volatility", "noise_category"]
                )
                .size()
                .unstack(fill_value=0)
                .reset_index()
            )

            for column in [
                "low",
                "medium",
                "high",
                "excluded_nan",
                "unmapped_valid",
            ]:
                if column not in block_counts_df.columns:
                    block_counts_df[column] = 0

            block_counts_df["total_valid"] = (
                block_counts_df["low"]
                + block_counts_df["medium"]
                + block_counts_df["high"]
                + block_counts_df["unmapped_valid"]
            )

            block_counts_df["total_all"] = (
                block_counts_df["total_valid"]
                + block_counts_df["excluded_nan"]
            )

            block_counts_df = block_counts_df[
                [
                    "subject",
                    "session",
                    "block",
                    "volatility",
                    "low",
                    "medium",
                    "high",
                    "excluded_nan",
                    "unmapped_valid",
                    "total_valid",
                    "total_all",
                ]
            ].sort_values(["session", "block"])

            print("\nChangepoints by noise category for each block:")
            print(block_counts_df.to_string(index=False))

            condition_counts_df = (
                block_counts_df
                .groupby("volatility")[
                    [
                        "low",
                        "medium",
                        "high",
                        "excluded_nan",
                        "unmapped_valid",
                        "total_valid",
                        "total_all",
                    ]
                ]
                .sum()
                .reset_index()
            )

            print("\nTotal changepoints by volatility condition:")
            print(condition_counts_df.to_string(index=False))

            out_folder = Path(details["analysis"]["behav"]["folder"])

            events_csv_path = (
                out_folder
                / f"{details['subjName']}_changepoint_noise_events.csv"
            )

            counts_csv_path = (
                out_folder
                / f"{details['subjName']}_changepoints_by_block_and_noise.csv"
            )

            condition_csv_path = (
                out_folder
                / f"{details['subjName']}_changepoints_by_volatility_and_noise.csv"
            )

            noise_events_df.to_csv(events_csv_path, index=False)
            block_counts_df.to_csv(counts_csv_path, index=False)
            condition_counts_df.to_csv(condition_csv_path, index=False)

            print(f"\nSaved event-level table to: {events_csv_path}")
            print(f"Saved block-level table to: {counts_csv_path}")
            print(f"Saved condition-level table to: {condition_csv_path}")

            fig = coins_plot_block_changepoint_counts(block_counts_df)
            figure_path = (
                out_folder
                / f"{details['subjName']}_changepoints_by_block_and_noise.png"
            )
            save_figure(fig, figure_path)

            print(f"Saved changepoint count figure to: {figure_path}")


        all_adjusts_arr = np.vstack(all_adjusts) if all_adjusts else np.empty((0, options.behav.adjustPreSamples + options.behav.adjustPostSamples + 1))
        vol_adjusts_arr = np.vstack(vol_adjusts) if vol_adjusts else np.empty_like(all_adjusts_arr)
        sta_adjusts_arr = np.vstack(sta_adjusts) if sta_adjusts else np.empty_like(all_adjusts_arr)
        all_jumps_arr = np.concatenate(all_jumps) if all_jumps else np.empty((0,))
        vol_jumps_arr = np.concatenate(vol_jumps) if vol_jumps else np.empty((0,))
        sta_jumps_arr = np.concatenate(sta_jumps) if sta_jumps else np.empty((0,))
        all_vars_arr = np.concatenate(all_vars) if all_vars else np.empty((0,))
        vol_vars_arr = np.concatenate(vol_vars) if vol_vars else np.empty((0,))
        sta_vars_arr = np.concatenate(sta_vars) if sta_vars else np.empty((0,))
        all_conds_arr = np.concatenate(all_conds) if all_conds else np.empty((0,))
        save_pickle(details['analysis']['behav']['adjustments'], {'allAdjusts': all_adjusts_arr, 'allJumps': all_jumps_arr, 'allVars': all_vars_arr, 'allConds': all_conds_arr})
        save_pickle(details['analysis']['behav']['adjustmentsVolatility'], {'staAdjusts': sta_adjusts_arr, 'staJumps': sta_jumps_arr, 'staVars': sta_vars_arr, 'volAdjusts': vol_adjusts_arr, 'volJumps': vol_jumps_arr, 'volVars': vol_vars_arr})

        jump_sizes = np.unique(all_jumps_arr) if len(all_jumps_arr) else np.asarray([])
        variances = np.unique(all_vars_arr[~np.isnan(all_vars_arr)]) if len(all_vars_arr) else np.asarray([])
        mean_adjusts = np.full((3, 4, max(len(jump_sizes), 1), options.behav.adjustPreSamples + options.behav.adjustPostSamples + 1), np.nan)
        median_adjusts = np.full_like(mean_adjusts, np.nan)
        for i_jmp, jump in enumerate(jump_sizes):
            mask_all = all_jumps_arr == jump
            mask_sta = sta_jumps_arr == jump
            mask_vol = vol_jumps_arr == jump
            if np.any(mask_all):
                mean_adjusts[0, 0, i_jmp, :] = np.nanmean(all_adjusts_arr[mask_all], axis=0)
                median_adjusts[0, 0, i_jmp, :] = np.nanmedian(all_adjusts_arr[mask_all], axis=0)
            if np.any(mask_sta):
                mean_adjusts[1, 0, i_jmp, :] = np.nanmean(sta_adjusts_arr[mask_sta], axis=0)
                median_adjusts[1, 0, i_jmp, :] = np.nanmedian(sta_adjusts_arr[mask_sta], axis=0)
            if np.any(mask_vol):
                mean_adjusts[2, 0, i_jmp, :] = np.nanmean(vol_adjusts_arr[mask_vol], axis=0)
                median_adjusts[2, 0, i_jmp, :] = np.nanmedian(vol_adjusts_arr[mask_vol], axis=0)
            for i_var, var in enumerate(variances, start=1):
                ma = mask_all & (all_vars_arr == var)
                ms = mask_sta & (sta_vars_arr == var)
                mv = mask_vol & (vol_vars_arr == var)
                if np.any(ma):
                    mean_adjusts[0, i_var, i_jmp, :] = np.nanmean(all_adjusts_arr[ma], axis=0)
                    median_adjusts[0, i_var, i_jmp, :] = np.nanmedian(all_adjusts_arr[ma], axis=0)
                if np.any(ms):
                    mean_adjusts[1, i_var, i_jmp, :] = np.nanmean(sta_adjusts_arr[ms], axis=0)
                    median_adjusts[1, i_var, i_jmp, :] = np.nanmedian(sta_adjusts_arr[ms], axis=0)
                if np.any(mv):
                    mean_adjusts[2, i_var, i_jmp, :] = np.nanmean(vol_adjusts_arr[mv], axis=0)
                    median_adjusts[2, i_var, i_jmp, :] = np.nanmedian(vol_adjusts_arr[mv], axis=0)
        save_pickle(details['analysis']['behav']['meanAdjustments'], {'meanAdjusts': mean_adjusts, 'jumpSizes': jump_sizes})
        save_pickle(details['analysis']['behav']['medianAdjustments'], {'medianAdjusts': median_adjusts, 'jumpSizes': jump_sizes})
        fig1, fig2 = coins_plot_subject_adjustments(mean_adjusts, jump_sizes, options)
        save_figure(fig1, details['analysis']['behav']['adjustVolatilityFig'])
        save_figure(fig2, details['analysis']['behav']['adjustNoiseFig'])
        fig1, fig2 = coins_plot_subject_adjustments(median_adjusts, jump_sizes, options)
        save_figure(fig1, details['analysis']['behav']['adjustMedianVolatilityFig'])
        save_figure(fig2, details['analysis']['behav']['adjustMedianNoiseFig'])
    else:
        mean_obj = load_pickle(details['analysis']['behav']['meanAdjustments'])
        median_obj = load_pickle(details['analysis']['behav']['medianAdjustments'])

        mean_adjusts = mean_obj['meanAdjusts']
        median_adjusts = median_obj['medianAdjusts']
        jump_sizes = mean_obj['jumpSizes']

    if options.behav.flagReactionTimes:
        rt_per_block = {}
        for i_sess in range(1, details['nSessions'] + 1):
            for i_block in range(1, 5):
                block_data = sub_data[(sub_data['sessID'] == i_sess) & (sub_data['blockID'] == i_block)]
                if block_data.empty:
                    continue
                rt_per_block[(i_sess, i_block)] = coins_compute_block_reaction_times(block_data, options)
        save_pickle(details['analysis']['behav']['rtFig'].with_suffix('.pkl'), {'rtPerBlock': rt_per_block})

    coins_subject_kernels(details, sub_data, options)
    coins_subject_kernels_session_wise(details, sub_data, options)
    return {
        'details': details,
        'subData': sub_data,
        'meanAdjusts': mean_adjusts,
        'medianAdjusts': median_adjusts,
        'jumpSizes': jump_sizes,
        'blockCounts': block_counts_df,
    }


def _compute_group_mean_and_se(subject_arrays: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    stacked = np.stack(subject_arrays, axis=0)

    group_mean = np.nanmean(stacked, axis=0)

    n_valid = np.sum(~np.isnan(stacked), axis=0)
    group_sd = np.nanstd(stacked, axis=0, ddof=1)

    with np.errstate(divide='ignore', invalid='ignore'):
        group_se = group_sd / np.sqrt(n_valid)

    group_se[n_valid < 2] = np.nan

    return group_mean, group_se

def loop_coins_analyse_behaviour(options: Options | None = None) -> None:
    if options is None:
        options = coins_options()

    group_out = Path("group_results")
    ensure_dir(group_out)

    subject_mean_adjusts = []
    subject_median_adjusts = []
    subject_block_counts = []

    reference_jump_sizes = None
    included_subjects = []

    for sub_id in options.subjectIDs:
        print(f"\n========================================")
        print(f"Analysing subject {sub_id}")
        print(f"========================================")

        result = coins_analyse_subject_behaviour(sub_id, options)

        mean_adjusts = result['meanAdjusts']
        median_adjusts = result['medianAdjusts']
        jump_sizes = result['jumpSizes']
        block_counts_df = result['blockCounts']

        if mean_adjusts is None or median_adjusts is None:
            print(f"Skipping subject {sub_id}: no adjustment arrays available.")
            continue

        if reference_jump_sizes is None:
            reference_jump_sizes = jump_sizes
        else:
            if (
                len(jump_sizes) != len(reference_jump_sizes)
                or not np.allclose(jump_sizes, reference_jump_sizes)
            ):
                raise ValueError(
                    f"Subject {sub_id} has different jump sizes: "
                    f"{jump_sizes}; expected {reference_jump_sizes}"
                )

        subject_mean_adjusts.append(mean_adjusts)
        subject_median_adjusts.append(median_adjusts)
        included_subjects.append(sub_id)

        if not block_counts_df.empty:
            subject_block_counts.append(block_counts_df)

    if not subject_mean_adjusts:
        print("No participant adjustment results were collected.")
        return

    print(f"\nIncluded subjects for group adjustment plots: {included_subjects}")

    # ------------------------------------------------------------
    # Group-level mean adjustment curves
    # ------------------------------------------------------------
    group_mean_adjusts, group_mean_se = _compute_group_mean_and_se(
        subject_mean_adjusts
    )

    fig1, fig2 = coins_plot_group_adjustments(
        group_mean_adjusts,
        group_mean_se,
        reference_jump_sizes,
        options,
    )

    save_figure(fig1, group_out / "group_mean_adjustments_volatility.png")
    save_figure(fig2, group_out / "group_mean_adjustments_noise.png")

    # ------------------------------------------------------------
    # Group-level curves based on participant medians
    # ------------------------------------------------------------
    group_median_adjusts, group_median_se = _compute_group_mean_and_se(
        subject_median_adjusts
    )

    fig1, fig2 = coins_plot_group_adjustments(
        group_median_adjusts,
        group_median_se,
        reference_jump_sizes,
        options,
    )

    save_figure(fig1, group_out / "group_median_adjustments_volatility.png")
    save_figure(fig2, group_out / "group_median_adjustments_noise.png")

    # ------------------------------------------------------------
    # Save group adjustment arrays
    # ------------------------------------------------------------
    save_pickle(
        group_out / "group_adjustments.pkl",
        {
            "subjects": included_subjects,
            "jumpSizes": reference_jump_sizes,
            "groupMeanAdjusts": group_mean_adjusts,
            "groupMeanSE": group_mean_se,
            "groupMedianAdjusts": group_median_adjusts,
            "groupMedianSE": group_median_se,
        }
    )

    # ------------------------------------------------------------
    # Combine changepoint counts across subjects
    # ------------------------------------------------------------
    if subject_block_counts:
        all_block_counts_df = pd.concat(
            subject_block_counts,
            ignore_index=True,
        )

        all_block_counts_df.to_csv(
            group_out / "all_subjects_changepoints_by_block_and_noise.csv",
            index=False,
        )

        counts_by_subject_df = (
            all_block_counts_df
            .groupby("subject")[
                [
                    "low",
                    "medium",
                    "high",
                    "excluded_nan",
                    "unmapped_valid",
                    "total_valid",
                    "total_all",
                ]
            ]
            .sum()
            .reset_index()
        )

        counts_by_subject_df.to_csv(
            group_out / "changepoints_by_subject_and_noise.csv",
            index=False,
        )

        counts_by_condition_df = (
            all_block_counts_df
            .groupby("volatility")[
                [
                    "low",
                    "medium",
                    "high",
                    "excluded_nan",
                    "unmapped_valid",
                    "total_valid",
                    "total_all",
                ]
            ]
            .sum()
            .reset_index()
        )

        counts_by_condition_df.to_csv(
            group_out / "changepoints_by_volatility_and_noise.csv",
            index=False,
        )

        total_counts_df = pd.DataFrame(
            {
                "noise_category": [
                    "low",
                    "medium",
                    "high",
                    "excluded_nan",
                ],
                "n_changepoints": [
                    int(all_block_counts_df["low"].sum()),
                    int(all_block_counts_df["medium"].sum()),
                    int(all_block_counts_df["high"].sum()),
                    int(all_block_counts_df["excluded_nan"].sum()),
                ],
            }
        )

        total_counts_df.to_csv(
            group_out / "total_changepoints_by_noise.csv",
            index=False,
        )

        print("\nTotal changepoints across all participants:")
        print(total_counts_df.to_string(index=False))

        print("\nChangepoints separated by volatility condition:")
        print(counts_by_condition_df.to_string(index=False))

        fig = coins_plot_group_changepoint_counts(total_counts_df)

        save_figure(
            fig,
            group_out / "group_changepoint_counts_by_noise.png",
        )

    print(f"\nGroup results saved to: {group_out.resolve()}")