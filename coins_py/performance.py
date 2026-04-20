from __future__ import annotations

import numpy as np
import pandas as pd

from .options import Options
from .utils import moving_mean, compare_wrap_reference


def _exclude_until_first_hit(block_data: pd.DataFrame) -> pd.DataFrame:
    prediction_error = ((block_data['laserRotation'] - block_data['shieldRotation'] + 180) % 360) - 180
    tol_area = float(block_data['shieldDegrees'].iloc[0]) / 2
    smp2excl: list[int] = []
    for i, pe in enumerate(prediction_error.to_numpy()):
        if abs(pe) > tol_area:
            smp2excl.append(i)
        else:
            break
    if smp2excl:
        return block_data.iloc[len(smp2excl):].reset_index(drop=True)
    return block_data.reset_index(drop=True)


def coins_compute_tracking_performance(block_data: pd.DataFrame, options: Options) -> tuple[dict, dict]:
    block_data = _exclude_until_first_hit(block_data)

    stim: dict = {}
    perf: dict = {}
    vol_value = block_data['volatility'].iloc[0]
    volatility = None if pd.isna(vol_value) else int(vol_value)

    stim['volatility'] = volatility
    perf['volatility'] = volatility

    shield = np.unwrap(block_data['shieldRotation'].to_numpy(dtype=float) * np.pi / 180)
    laser = np.unwrap(block_data['laserRotation'].to_numpy(dtype=float) * np.pi / 180)
    true_mean = np.unwrap(block_data['trueMean'].to_numpy(dtype=float) * np.pi / 180)
    laser = compare_wrap_reference(shield, laser)
    true_mean = compare_wrap_reference(shield, true_mean)
    true_variance = block_data['trueVariance'].to_numpy(dtype=float) * np.pi / 180
    shield_width = block_data['shieldDegrees'].to_numpy(dtype=float) * np.pi / 180

    stim['position'] = laser
    stim['genMean'] = true_mean
    stim['movAvg'] = moving_mean(laser, options.behav.movAvgWin)
    stim['genStd'] = true_variance
    stim['nMeanCPs'] = int(np.sum(np.diff(true_mean) != 0))
    stim['sumDeltaMean'] = float(np.sum(np.abs(np.diff(true_mean))))
    stim['nStdCPs'] = int(np.sum(np.diff(true_variance) != 0))
    stim['moveBias'] = float(np.sum(np.diff(laser)))
    stim['meanBias'] = float(np.sum(np.diff(true_mean)))
    stim['stdBias'] = float(np.sum(np.diff(true_variance)))

    shield_1st = np.r_[0.0, np.diff(shield)]
    shield_2nd = np.r_[0.0, 0.0, np.diff(np.diff(shield))]
    left_turn_onsets = (shield_1st > 0) & (shield_2nd > 0)
    right_turn_onsets = (shield_1st < 0) & (shield_2nd < 0)
    is_movement = shield_1st != 0

    shield_size_1st = np.r_[0.0, np.diff(shield_width)]
    shield_up = shield_size_1st > 0
    shield_down = shield_size_1st < 0

    perf['position'] = shield
    perf['positionPE'] = laser - shield
    perf['overallPosPE'] = float(np.sum(np.abs(perf['positionPE'])))
    perf['diff2genMean'] = true_mean - shield
    perf['meanPosPE'] = float(np.mean(np.abs(perf['positionPE'])))
    perf['medianPosPE'] = float(np.median(np.abs(perf['positionPE'])))
    perf['sumDiff2mean'] = float(np.sum(np.abs(perf['diff2genMean'])))
    perf['meanDiff2mean'] = float(np.mean(np.abs(perf['diff2genMean'])))
    perf['medianDiff2mean'] = float(np.median(np.abs(perf['diff2genMean'])))
    perf['nMoveOnsets'] = int(np.sum(left_turn_onsets) + np.sum(right_turn_onsets))
    perf['nMoveFrames'] = int(np.sum(is_movement))
    perf['overallMove'] = float(np.sum(np.abs(shield_1st)))
    perf['moveBias'] = float(np.sum(shield_1st))
    perf['relMoveBias'] = perf['moveBias'] - stim['moveBias']
    perf['trackBias'] = float(np.sum(perf['positionPE']))
    perf['relTrackBias'] = perf['trackBias'] - stim['moveBias']
    perf['infBias'] = float(np.sum(perf['diff2genMean']))
    perf['relInfBias'] = perf['infBias'] - stim['meanBias']
    perf['stdev'] = 0.5 * shield_width
    perf['meanStdev'] = float(np.mean(perf['stdev']))
    perf['absPositionPE'] = np.abs(perf['positionPE'])
    perf['diff2absPE'] = perf['absPositionPE'] - perf['stdev']
    perf['sumDiff2absPE'] = float(np.sum(np.abs(perf['diff2absPE'])))
    perf['diff2genStd'] = perf['stdev'] - stim['genStd']
    perf['sumDiff2std'] = float(np.sum(np.abs(perf['diff2genStd'])))
    perf['nSizeOnsets'] = int(np.sum(shield_up) + np.sum(shield_down))
    perf['sizeTrackBias'] = float(np.sum(perf['diff2absPE']))
    perf['sizeInfBias'] = float(np.sum(perf['diff2genStd']))
    perf['sizeUpdateBias'] = float(np.sum(shield_size_1st))
    perf['relSizeUpdateBias'] = perf['sizeUpdateBias'] - stim['stdBias']
    perf['diff2genMeanCPs'] = perf['nMoveOnsets'] - stim['nMeanCPs']
    perf['diff2sumDeltaMean'] = perf['overallMove'] - stim['sumDeltaMean']
    perf['diff2genStdCPs'] = perf['nSizeOnsets'] - stim['nStdCPs']
    perf['reward'] = float(block_data['totalReward'].iloc[-1])
    return stim, perf
