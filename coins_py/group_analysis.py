from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from .options import Options, coins_options
from .plots import coins_plot_group_adjustments, coins_plot_subject_reg_kernels_session_wise, save_figure
from .subjects import coins_subjects
from .utils import load_pickle, save_pickle, sem


def coins_group_post_jump_adjustments(options: Options | None = None) -> None:
    if options is None:
        options = coins_options()
    n_subjects = len(options.subjectIDs)
    results_dir = options.workDir / 'behav'
    group_adjusts = []
    for sub_id in options.subjectIDs:
        details = coins_subjects(sub_id, options)
        mean_adjusts = load_pickle(details['analysis']['behav']['meanAdjustments'])['meanAdjusts']
        group_adjusts.append(mean_adjusts)
    group_adjusts_arr = np.stack(group_adjusts)

    example = coins_subjects(options.subjectIDs[-1], options)
    jump_sizes = load_pickle(example['analysis']['behav']['meanAdjustments'])['jumpSizes']
    pre = options.behav.adjustPreSamples
    post = options.behav.adjustPostSamples
    time_axis = np.arange(-pre, post + 1) / options.behav.fsample

    if options.behav.flagNormaliseAdjustments:
        group_adjusts_arr = group_adjusts_arr - group_adjusts_arr[..., pre][..., None]

    avg_group_adjusts = np.nanmean(group_adjusts_arr, axis=0)
    se_group_adjusts = sem(group_adjusts_arr, axis=0)
    save_pickle(results_dir / f'n{n_subjects}_groupAdjustments.pkl', {
        'groupAdjusts': group_adjusts_arr,
        'avgGroupAdjusts': avg_group_adjusts,
        'seGroupAdjusts': se_group_adjusts,
        'timeAxis': time_axis,
        'jumpSizes': jump_sizes,
    })
    fig1, fig2 = coins_plot_group_adjustments(avg_group_adjusts, se_group_adjusts, jump_sizes, options)
    save_figure(fig1, results_dir / f'n{n_subjects}_adjustment_timecourse_volatility.png')
    save_figure(fig2, results_dir / f'n{n_subjects}_adjustment_timecourse_noise.png')


def coins_group_reaction_times(options: Options | None = None) -> None:
    if options is None:
        options = coins_options()
    n_subjects = len(options.subjectIDs)
    results_dir = options.workDir / 'behav'
    adj_obj = load_pickle(results_dir / f'n{n_subjects}_groupAdjustments.pkl')
    group_adjusts = adj_obj['groupAdjusts']
    time_axis = adj_obj['timeAxis']
    jump_sizes = adj_obj['jumpSizes']
    pre = options.behav.adjustPreSamples

    half_jumps = jump_sizes / 2
    all_adjusts = group_adjusts[..., pre:]
    all_times = time_axis[pre:]
    adjust_rts = np.full(all_adjusts.shape[:-1], np.nan)
    for i_sub in range(all_adjusts.shape[0]):
        for i_con in range(all_adjusts.shape[1]):
            for i_var in range(all_adjusts.shape[2]):
                for i_jmp in range(all_adjusts.shape[3]):
                    adjust_data = all_adjusts[i_sub, i_con, i_var, i_jmp, :]
                    idx = np.where(adjust_data >= half_jumps[i_jmp])[0]
                    if len(idx):
                        adjust_rts[i_sub, i_con, i_var, i_jmp] = all_times[idx[0]]
    save_pickle(results_dir / f'n{n_subjects}_groupAdjustment_RTs.pkl', {'adjustRTs': adjust_rts})

    data_rts = adjust_rts[:, 1:3, 1:4, :]
    rows = []
    for i_sub in range(data_rts.shape[0]):
        for i_con in range(data_rts.shape[1]):
            for i_var in range(data_rts.shape[2]):
                for i_jmp in range(data_rts.shape[3]):
                    rows.append({
                        'RT': data_rts[i_sub, i_con, i_var, i_jmp],
                        'volatility': -0.5 if i_con == 0 else 0.5,
                        'noise': [-1, 0, 1][i_var],
                        'jumpSize': [-2, -1, 0, 1, 2][i_jmp],
                        'ID': i_sub + 1,
                    })
    df = pd.DataFrame(rows).dropna()
    model = smf.ols('RT ~ jumpSize * volatility * noise', data=df).fit() if not df.empty else None
    save_pickle(results_dir / f'n{n_subjects}_reactionTimes_linearModel.pkl', {'model_summary': model.summary().as_text() if model else None})


def coins_group_regression_kernels_session_wise(options: Options | None = None) -> None:
    if options is None:
        options = coins_options()
    n_subjects = len(options.subjectIDs)
    results_dir = options.workDir / 'behav'
    grp_vol = []
    grp_sta = []
    grp_trials_vol = []
    grp_trials_sta = []
    for sub_id in options.subjectIDs:
        details = coins_subjects(sub_id, options)
        obj = load_pickle(details['analysis']['behav']['sessRegKernels5'])
        grp_vol.append(obj['volBetas'])
        grp_sta.append(obj['staBetas'])
        grp_trials_vol.append(obj['nTrialsVol'])
        grp_trials_sta.append(obj['nTrialsSta'])
    grp_vol_arr = np.stack(grp_vol)
    grp_sta_arr = np.stack(grp_sta)
    save_pickle(results_dir / f'n{n_subjects}_regressionBetas_sessionWise.pkl', {
        'grpVolBetas': grp_vol_arr,
        'grpStaBetas': grp_sta_arr,
        'grpTrialsSta': np.stack(grp_trials_sta),
        'grpTrialsVol': np.stack(grp_trials_vol),
    })
    fig = coins_plot_subject_reg_kernels_session_wise(np.nanmean(grp_vol_arr, axis=1), np.nanmean(grp_sta_arr, axis=1))
    save_figure(fig, results_dir / f'n{n_subjects}_regressionBetas.png')
