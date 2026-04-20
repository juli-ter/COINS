from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

from .options import Options
from .utils import compare_wrap_reference


def coins_compute_block_adjustments(block_data: pd.DataFrame, options: Options) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_before = options.behav.adjustPreSamples
    n_after = options.behav.adjustPostSamples

    shield = np.unwrap(block_data['shieldRotation'].to_numpy(dtype=float) * np.pi / 180)
    laser = np.unwrap(block_data['laserRotation'].to_numpy(dtype=float) * np.pi / 180)
    true_mean = np.unwrap(block_data['trueMean'].to_numpy(dtype=float) * np.pi / 180)
    laser = compare_wrap_reference(shield, laser)
    true_mean = compare_wrap_reference(shield, true_mean)

    dist2mean = shield - true_mean
    change_in_mean = np.r_[0.0, np.diff(true_mean)]
    cp_up = np.where(change_in_mean > 0)[0]
    cp_down = np.where(change_in_mean < 0)[0]

    all_adjust = []
    jump_size = []
    curr_var = []

    shield_pad = np.r_[np.full(n_before, np.nan), shield, np.full(n_after, np.nan)]
    true_mean_pad = np.r_[np.full(n_before, np.nan), true_mean, np.full(n_after, np.nan)]

    def handle_cp(indices: np.ndarray, sign: float) -> None:
        nonlocal all_adjust, jump_size, curr_var
        for cp in indices:
            if cp >= n_before and cp < len(dist2mean) - 120:
                trace = shield_pad[cp:cp + n_before + n_after + 1] - true_mean[cp - 1]
                if sign < 0:
                    trace = -trace
                all_adjust.append(trace)
                jump_size.append(sign * change_in_mean[cp])
                pre_var = np.unique(block_data['trueVariance'].iloc[max(cp - 30, 0):cp + 1])
                post_var = np.unique(block_data['trueVariance'].iloc[cp:min(cp + 91, len(block_data))])
                if len(pre_var) > 1:
                    warnings.warn('True noise level changed just before true mean')
                if len(post_var) > 1:
                    warnings.warn('True noise level changed within 1.5s after mean jump')
                    curr_var.append(np.nan)
                else:
                    curr_var.append(float(block_data['trueVariance'].iloc[cp]))

    handle_cp(cp_up, 1.0)
    handle_cp(cp_down, -1.0)

    all_adjust_arr = np.asarray(all_adjust, dtype=float) if all_adjust else np.empty((0, n_before + n_after + 1))
    jump_arr = np.round(np.asarray(jump_size, dtype=float), 4)
    curr_var_arr = np.round(np.asarray(curr_var, dtype=float) * np.pi / 180, 4)
    curr_sd = np.round(jump_arr / curr_var_arr, 1) if len(curr_var_arr) else np.empty((0,))
    return all_adjust_arr, jump_arr, curr_var_arr, curr_sd


def coins_compute_block_reaction_times(block_data: pd.DataFrame, options: Options) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_after = options.behav.maxRtSamples
    shield = np.unwrap(block_data['shieldRotation'].to_numpy(dtype=float) * np.pi / 180)
    true_mean = np.unwrap(block_data['trueMean'].to_numpy(dtype=float) * np.pi / 180)

    shield_move = np.r_[0.0, np.diff(shield)]
    bin_move = np.sign(shield_move)
    bin_2nd = np.r_[0.0, np.diff(bin_move)]

    up_on = np.where((bin_move > 0) & (bin_2nd > 0))[0]
    down_on = np.where((bin_move < 0) & (bin_2nd < 0))[0]
    up_off = np.where(((bin_move == 0) & (bin_2nd < 0)) | (bin_2nd < -1))[0]
    down_off = np.where(((bin_move == 0) & (bin_2nd > 0)) | (bin_2nd > 1))[0]
    if len(up_on) > len(up_off):
        up_off = np.r_[up_off, len(shield_move) - 1]
    if len(down_on) > len(down_off):
        down_off = np.r_[down_off, len(shield_move) - 1]
    up_dur = up_off[:len(up_on)] - up_on
    down_dur = down_off[:len(down_on)] - down_on

    change_in_mean = np.r_[0.0, np.diff(true_mean)]
    cp_up = np.where(change_in_mean > 0)[0]
    cp_down = np.where(change_in_mean < 0)[0]
    all_durs = np.full((len(cp_up) + len(cp_down), 10), np.nan)
    n_resp = []
    jump_size = []
    curr_var = []

    row = 0
    for cp in cp_up:
        response_onsets = np.where((up_on > cp) & (up_on < cp + n_after))[0]
        n_resp.append(len(response_onsets))
        for i_resp, idx in enumerate(response_onsets[:10]):
            all_durs[row, i_resp] = up_dur[idx]
        jump_size.append(change_in_mean[cp])
        curr_var.append(float(block_data['trueVariance'].iloc[cp]))
        row += 1

    for cp in cp_down:
        response_onsets = np.where((down_on > cp) & (down_on < cp + n_after))[0]
        n_resp.append(len(response_onsets))
        for i_resp, idx in enumerate(response_onsets[:10]):
            all_durs[row, i_resp] = down_dur[idx]
        jump_size.append(-change_in_mean[cp])
        curr_var.append(float(block_data['trueVariance'].iloc[cp]))
        row += 1

    curr_var_arr = np.round(np.asarray(curr_var, dtype=float) * np.pi / 180, 4)
    jump_arr = np.round(np.asarray(jump_size, dtype=float), 4)
    curr_sd = np.round(jump_arr / curr_var_arr, 1) if len(curr_var_arr) else np.empty((0,))
    return all_durs, np.asarray(n_resp), jump_arr, curr_var_arr, curr_sd
