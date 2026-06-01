from __future__ import annotations

import numpy as np
import pandas as pd

from .options import Options
from .utils import mat_pinv


def _prediction_error(block_data: pd.DataFrame) -> np.ndarray:
    return ((block_data['laserRotation'].to_numpy(dtype=float) - block_data['shieldRotation'].to_numpy(dtype=float) + 180) % 360) - 180


def coins_compute_block_movements(block_data: pd.DataFrame, options: Options) -> dict:
    prediction_error = _prediction_error(block_data)
    tol_area = float(np.unique(block_data['shieldDegrees'].to_numpy(dtype=float))[0]) / 2
    start_sample = 0
    for i, pe in enumerate(prediction_error):
        if abs(pe) <= tol_area:
            start_sample = i
            break
    trimmed = block_data.iloc[start_sample:].reset_index(drop=True)

    shield = trimmed['shieldRotation'].to_numpy(dtype=float)
    shield_1st = np.r_[0.0, np.diff(shield)]
    shield_2nd = np.r_[0.0, 0.0, np.diff(np.diff(shield))]

    left_onsets = np.where((shield_1st > 0) & (shield_2nd > 0))[0]
    right_onsets = np.where((shield_1st < 0) & (shield_2nd < 0))[0]

    left_offsets = []
    for onset in left_onsets:
        offset = onset
        while offset < len(shield_1st) and shield_1st[offset] > 0:
            offset += 1
        left_offsets.append(min(offset, len(shield_1st) - 1))
    right_offsets = []
    for onset in right_onsets:
        offset = onset
        while offset < len(shield_1st) and shield_1st[offset] < 0:
            offset += 1
        right_offsets.append(min(offset, len(shield_1st) - 1))

    left_offsets = np.asarray(left_offsets, dtype=int)
    right_offsets = np.asarray(right_offsets, dtype=int)
    left_step_sizes = left_offsets - left_onsets
    right_step_sizes = right_offsets - right_onsets
    orig_left_step_sizes = left_step_sizes.copy()
    orig_right_step_sizes = right_step_sizes.copy()

    left_discard: list[int] = []
    new_left_step_sizes = left_step_sizes.astype(float).copy()
    for i in range(1, len(left_onsets)):
        if (left_onsets[i] - left_offsets[i - 1]) < options.behav.minResponseDistance:
            left_discard.append(i)
            rel_step = i - 1
            while rel_step in left_discard and rel_step > 0:
                rel_step -= 1
            new_left_step_sizes[rel_step] = shield[left_offsets[i] - 1] - shield[left_onsets[rel_step] - 1]
    clean_left_step_sizes = new_left_step_sizes.copy()
    clean_left_onsets = left_onsets.copy()
    if left_discard:
        clean_left_step_sizes = np.delete(clean_left_step_sizes, left_discard)
        clean_left_onsets = np.delete(clean_left_onsets, left_discard)

    right_discard: list[int] = []
    new_right_step_sizes = right_step_sizes.astype(float).copy()
    for i in range(1, len(right_onsets)):
        if (right_onsets[i] - right_offsets[i - 1]) < options.behav.minResponseDistance:
            right_discard.append(i)
            rel_step = i - 1
            while rel_step in right_discard and rel_step > 0:
                rel_step -= 1
            new_right_step_sizes[rel_step] = shield[right_onsets[rel_step] - 1] - shield[right_offsets[i] - 1]
    clean_right_step_sizes = new_right_step_sizes.copy()
    clean_right_onsets = right_onsets.copy()
    if right_discard:
        clean_right_step_sizes = np.delete(clean_right_step_sizes, right_discard)
        clean_right_onsets = np.delete(clean_right_onsets, right_discard)

    small_left = np.where(clean_left_step_sizes < options.behav.minStepSize)[0]
    clean_left_onsets = np.delete(clean_left_onsets, small_left)
    clean_left_step_sizes = np.delete(clean_left_step_sizes, small_left)
    small_right = np.where(clean_right_step_sizes < options.behav.minStepSize)[0]
    clean_right_onsets = np.delete(clean_right_onsets, small_right)
    clean_right_step_sizes = np.delete(clean_right_step_sizes, small_right)

    return {
        'left': {
            'onsets': clean_left_onsets + start_sample,
            'stepSizes': clean_left_step_sizes,
            'smallSteps': int(len(small_left)),
            'unifiedSteps': int(len(left_discard)),
            'origStepSizes': orig_left_step_sizes,
        },
        'right': {
            'onsets': clean_right_onsets + start_sample,
            'stepSizes': clean_right_step_sizes,
            'smallSteps': int(len(small_right)),
            'unifiedSteps': int(len(right_discard)),
            'origStepSizes': orig_right_step_sizes,
        },
        'nMovements': int(len(clean_left_onsets) + len(clean_right_onsets)),
        'stepSizes': np.r_[clean_left_step_sizes, clean_right_step_sizes],
        'origStepSizes': np.r_[orig_left_step_sizes, orig_right_step_sizes],
        'nSmallSteps': int(len(small_left) + len(small_right)),
        'nUnifiedSteps': int(len(left_discard) + len(right_discard)),
    }


def coins_compute_block_kernels(block_data: pd.DataFrame, options: Options) -> tuple[np.ndarray, np.ndarray, dict]:
    n_before = options.behav.kernelPreSamples
    n_after = options.behav.kernelPostSamples
    n_total = n_before + n_after

    prediction_error = _prediction_error(block_data).astype(float)
    tol_area = float(block_data['shieldDegrees'].iloc[0]) / 2
    for i, pe in enumerate(prediction_error):
        if abs(pe) > tol_area:
            prediction_error[i] = np.nan
        else:
            break
    abs_pe = np.abs(prediction_error)
    prediction_error = prediction_error - np.nanmean(prediction_error)
    abs_pe = abs_pe - np.nanmean(abs_pe)

    block_move = coins_compute_block_movements(block_data, options)
    pe_trace = np.r_[np.full(n_before, np.nan), prediction_error, np.full(n_after, np.nan)]

    left_kernels = np.vstack([
        pe_trace[start:start + n_total + 1] for start in block_move['left']['onsets']
    ]) if len(block_move['left']['onsets']) else np.empty((0, n_total + 1))
    right_kernels = np.vstack([
        pe_trace[start:start + n_total + 1] for start in block_move['right']['onsets']
    ]) if len(block_move['right']['onsets']) else np.empty((0, n_total + 1))

    shield_size_1st = np.r_[0.0, np.diff(block_data['shieldDegrees'].to_numpy(dtype=float))]
    size_up_idx = np.where(shield_size_1st > 0)[0]
    size_down_idx = np.where(shield_size_1st < 0)[0]
    abs_pe_trace = np.r_[np.full(n_before, np.nan), abs_pe, np.full(n_after, np.nan)]

    up_kernels = np.vstack([
        abs_pe_trace[idx:idx + n_total + 1] for idx in size_up_idx
    ]) if len(size_up_idx) else np.full((1, n_total + 1), np.nan)
    down_kernels = np.vstack([
        abs_pe_trace[idx:idx + n_total + 1] for idx in size_down_idx
    ]) if len(size_down_idx) else np.full((1, n_total + 1), np.nan)

    move_kernels = np.vstack([left_kernels, -right_kernels]) if len(left_kernels) or len(right_kernels) else np.full((1, n_total + 1), np.nan)
    size_kernels = np.vstack([up_kernels, -down_kernels])

    avg_kernels = np.vstack([
        np.nanmean(move_kernels, axis=0),
        np.nanmean(size_kernels, axis=0),
        np.nanmean(left_kernels, axis=0) if len(left_kernels) else np.full(n_total + 1, np.nan),
        np.nanmean(right_kernels, axis=0) if len(right_kernels) else np.full(n_total + 1, np.nan),
        np.nanmean(up_kernels, axis=0),
        np.nanmean(down_kernels, axis=0),
    ])
    n_kernels = np.asarray([
        move_kernels.shape[0], size_kernels.shape[0], left_kernels.shape[0], right_kernels.shape[0], up_kernels.shape[0], down_kernels.shape[0]
    ], dtype=float)
    return avg_kernels, n_kernels, block_move


def coins_compute_glm(X: np.ndarray, Y: np.ndarray, options: Options) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.size == 0:
        return np.full((X.shape[1] if X.ndim == 2 else 0,), np.nan)
    if options.behav.flagNormaliseEvidence:
        x_mean = np.nanmean(X, axis=0)
        x_std = np.nanstd(X, axis=0)
        x_std[~np.isfinite(x_std) | (x_std == 0)] = 1.0
        y_mean = np.nanmean(Y)
        y_std = np.nanstd(Y)
        if not np.isfinite(y_std) or y_std == 0:
            y_std = 1.0
        X = (X - x_mean) / x_std
        Y = (Y - y_mean) / y_std
    valid = np.all(np.isfinite(X), axis=1) & np.isfinite(Y)
    X = X[valid]
    Y = Y[valid]
    if X.size == 0:
        return np.full((X.shape[1] if X.ndim == 2 else 0,), np.nan)
    return mat_pinv(X) @ Y


def coins_compute_regression_kernels(block_data: pd.DataFrame, options: Options) -> tuple[np.ndarray, int, np.ndarray, int, np.ndarray, np.ndarray]:
    n_before = options.behav.kernelPreSamplesEvi
    prediction_error = _prediction_error(block_data).astype(float)
    tol_area = float(np.unique(block_data['shieldDegrees'])[0]) / 2
    for i, pe in enumerate(prediction_error):
        if abs(pe) > tol_area:
            prediction_error[i] = np.nan
        else:
            break
    abs_pe = np.abs(prediction_error)

    block_move = coins_compute_block_movements(block_data, options)
    pe_trace = np.r_[np.nan, prediction_error]
    change_idx = np.where(np.r_[0.0, np.diff(block_data['laserRotation'].to_numpy(dtype=float))] != 0)[0]

    def recent_matrix(starts: np.ndarray, trace: np.ndarray) -> np.ndarray:
        rows = []
        for start in starts:
            prior = change_idx[change_idx < start]
            if len(prior) >= n_before:
                recent = prior[-n_before:]
                non_avail = 0
            else:
                recent = prior
                non_avail = n_before - len(prior)
            idx = np.r_[np.ones(non_avail, dtype=int), recent + 1]
            rows.append(trace[idx])
        return np.asarray(rows, dtype=float) if rows else np.empty((0, n_before))

    left = recent_matrix(block_move['left']['onsets'], pe_trace)
    right = recent_matrix(block_move['right']['onsets'], pe_trace)
    move_kernels = np.vstack([left, -right]) if len(left) or len(right) else np.empty((0, n_before))

    abs_pe = abs_pe - np.nanmean(abs_pe)
    abs_trace = np.r_[np.nan, abs_pe]
    size_1st = np.r_[0.0, np.diff(block_data['shieldDegrees'].to_numpy(dtype=float))]
    up = recent_matrix(np.where(size_1st > 0)[0], abs_trace)
    down = recent_matrix(np.where(size_1st < 0)[0], abs_trace)

    avg_kernels = np.nanmean(move_kernels, axis=0) if move_kernels.size else np.full(n_before, np.nan)
    n_kernels = move_kernels.shape[0]
    Ybin = np.r_[np.ones(len(block_move['left']['onsets'])), -np.ones(len(block_move['right']['onsets']))]
    Y = np.r_[block_move['left']['stepSizes'], -block_move['right']['stepSizes']]
    X = np.vstack([left, right]) if len(left) or len(right) else np.empty((0, n_before))

    if X.size == 0:
        betas = np.full(n_before, np.nan)
        n_trials = 0
    else:
        if options.behav.flagNormaliseEvidence:
            x_mean = np.nanmean(X, axis=0)
            x_std = np.nanstd(X, axis=0)
            x_std[~np.isfinite(x_std) | (x_std == 0)] = 1.0
            y_mean = np.nanmean(Y)
            y_std = np.nanstd(Y)
            if not np.isfinite(y_std) or y_std == 0:
                y_std = 1.0
            X = (X - x_mean) / x_std
            Y = (Y - y_mean) / y_std
        valid = np.all(np.isfinite(X), axis=1) & np.isfinite(Y)
        X = X[valid]
        Y = Y[valid]
        Ybin = Ybin[valid]
        n_trials = int(X.shape[0])
        p_dm = mat_pinv(X)
        betas = p_dm @ (Ybin if options.behav.flagUseBinaryRegression else Y)
    return betas, n_trials, avg_kernels, n_kernels, up, down


def coins_block_kernels_regression(sub_data: pd.DataFrame, excluded_blocks: list[list[int]], options: Options) -> tuple[list[np.ndarray], list[np.ndarray]]:
    n_sessions = int(sub_data['sessID'].max())
    n_blocks = 4
    n_samples = options.behav.kernelPreSamplesEvi
    counts = np.zeros(2, dtype=int)
    betas_con = [np.empty((0, n_samples)), np.empty((0, n_samples))]
    trials_con = [np.empty((0,), dtype=float), np.empty((0,), dtype=float)]

    for i_sess in range(1, n_sessions + 1):
        for i_block in range(1, n_blocks + 1):
            block_data = sub_data[(sub_data['sessID'] == i_sess) & (sub_data['blockID'] == i_block)]
            if block_data.empty:
                continue
            if [i_sess, i_block] in excluded_blocks:
                beta = np.full(n_samples, np.nan)
                n_trials = 0.0
            else:
                beta, n_trials, *_ = coins_compute_regression_kernels(block_data, options)
            
            vol_value = block_data['volatility'].iloc[0]
            if pd.isna(vol_value):
                continue
            vol = int(vol_value)

            betas_con[vol] = np.vstack([betas_con[vol], beta])
            trials_con[vol] = np.r_[trials_con[vol], n_trials]
            counts[vol] += 1
    return betas_con, trials_con


def coins_compute_sessionwise_regression_kernels(sess_data: pd.DataFrame, options: Options) -> tuple[np.ndarray, np.ndarray, int, int]:
    pre_smp = options.behav.kernelPreSamplesEvi
    X_parts: list[np.ndarray] = []
    Y_parts: list[np.ndarray] = []
    vol_reg: list[np.ndarray] = []
    noise_reg: list[np.ndarray] = []
    uncert_reg: list[np.ndarray] = []

    for i_block in range(1, int(sess_data['blockID'].max()) + 1):
        block_data = sess_data[sess_data['blockID'] == i_block]
        if block_data.empty:
            continue
        block_vol = block_data['volatility'].to_numpy(dtype=float)
        noise_trace = block_data['trueVariance'].to_numpy(dtype=float) / 10 - 2
        shield_size = block_data['shieldDegrees'].to_numpy(dtype=float) / 20 - 2

        prediction_error = _prediction_error(block_data).astype(float)
        tol_area = float(block_data['shieldDegrees'].iloc[0]) / 2
        for i, pe in enumerate(prediction_error):
            if abs(pe) > tol_area:
                prediction_error[i] = np.nan
            else:
                break

        block_move = coins_compute_block_movements(block_data, options)
        pe_trace = np.r_[np.nan, prediction_error]
        change_idx = np.where(np.r_[0.0, np.diff(block_data['laserRotation'].to_numpy(dtype=float))] != 0)[0]

        def recent_matrix(starts: np.ndarray) -> np.ndarray:
            rows = []
            for start in starts:
                prior = change_idx[change_idx < start]
                if len(prior) >= pre_smp:
                    recent = prior[-pre_smp:]
                    non_avail = 0
                else:
                    recent = prior
                    non_avail = pre_smp - len(prior)
                idx = np.r_[np.ones(non_avail, dtype=int), recent + 1]
                rows.append(pe_trace[idx])
            return np.asarray(rows, dtype=float) if rows else np.empty((0, pre_smp))

        left = recent_matrix(block_move['left']['onsets'])
        right = recent_matrix(block_move['right']['onsets'])
        X_parts.append(np.vstack([left, -right]) if len(left) or len(right) else np.empty((0, pre_smp)))
        Y_parts.append(np.r_[block_move['left']['stepSizes'], block_move['right']['stepSizes']])
        vol_reg.append(np.r_[block_vol[block_move['left']['onsets']], block_vol[block_move['right']['onsets']]])
        noise_reg.append(np.r_[noise_trace[block_move['left']['onsets']], noise_trace[block_move['right']['onsets']]])
        uncert_reg.append(np.r_[shield_size[block_move['left']['onsets']], shield_size[block_move['right']['onsets']]])

    X = np.vstack(X_parts) if X_parts else np.empty((0, pre_smp))
    Y = np.concatenate(Y_parts) if Y_parts else np.empty((0,))
    vol_reg_arr = np.concatenate(vol_reg) if vol_reg else np.empty((0,))
    noise_reg_arr = np.concatenate(noise_reg) if noise_reg else np.empty((0,))
    uncert_reg_arr = np.concatenate(uncert_reg) if uncert_reg else np.empty((0,))

    if X.shape[0] == 0:
        nbetas = 11 if options.behav.flagRegKernelSamples == 5 else 7
        return np.full(nbetas, np.nan), np.full(nbetas, np.nan), 0, 0

    Xred = X[:, 2:5]
    if options.behav.flagRegKernelSamples == 5:
        interact = np.column_stack([X[:, i] * noise_reg_arr for i in range(X.shape[1])])
        Xnoi_inter = np.column_stack([X, noise_reg_arr, interact])
    else:
        interact = np.column_stack([Xred[:, i] * noise_reg_arr for i in range(Xred.shape[1])])
        Xnoi_inter = np.column_stack([Xred, noise_reg_arr, interact])

    Xvol = Xnoi_inter[vol_reg_arr == 1]
    Xsta = Xnoi_inter[vol_reg_arr == 0]
    Yvol = Y[vol_reg_arr == 1]
    Ysta = Y[vol_reg_arr == 0]
    n_trials_vol = int(Xvol.shape[0])
    n_trials_sta = int(Xsta.shape[0])

    betas_vol = coins_compute_glm(Xvol, Yvol, options) if len(Xvol) else np.full(Xnoi_inter.shape[1], np.nan)
    betas_sta = coins_compute_glm(Xsta, Ysta, options) if len(Xsta) else np.full(Xnoi_inter.shape[1], np.nan)
    return betas_vol, betas_sta, n_trials_vol, n_trials_sta
