from __future__ import annotations

from pathlib import Path
from typing import Any

from .options import Options


def coins_subjects(subID: int, options: Options) -> dict[str, Any]:
    subj_name = f'sub-{subID:02d}'
    details: dict[str, Any] = {
        'subjName': subj_name,
        'nSessions': 4,
        'excludedBlocks': [],
    }
    excluded = {
        3: [[1, 3]],
        10: [[3, 1]],
        15: [[2, 2]],
        21: [[2, 2]],
    }
    details['excludedBlocks'] = excluded.get(subID, [])

    raw_beh_folder = options.rawDir / subj_name / 'ses-2-meg' / 'beh'
    raw_meg_folder = options.rawDir / subj_name / 'ses-2-meg' / 'meg'
    session_files = [
        raw_beh_folder / f'{subj_name}_ses-2-meg_task-coinsmeg_run-{i}.csv'
        for i in range(1, 5)
    ]

    analysis_folder = options.workDir / 'behav' / subj_name
    block_figures = {
        (sess, block): analysis_folder / f'{subj_name}_sess{sess}_block{block}_blockPlot.png'
        for sess in range(1, 5)
        for block in range(1, 5)
    }

    details['raw'] = {
        'behav': {'folder': raw_beh_folder, 'sessionFileNames': session_files},
        'meg': {'folder': raw_meg_folder},
    }
    details['analysis'] = {
        'behav': {
            'folder': analysis_folder,
            'responseData': analysis_folder / f'{subj_name}_responseData.pkl',
            'blockFigures': block_figures,
            'performance': analysis_folder / f'{subj_name}_perform.pkl',
            'performancePlot': analysis_folder / f'{subj_name}_performance.png',
            'performRewardFig': analysis_folder / f'{subj_name}_performOverviewReward.png',
            'performAvgPeFig': analysis_folder / f'{subj_name}_performOverviewAvgPE.png',
            'performAvgDevFromMeanFig': analysis_folder / f'{subj_name}_performOverviewDevFromMean.png',
            'blockKernels': analysis_folder / f'{subj_name}_blockKernels.pkl',
            'blockMoves': analysis_folder / f'{subj_name}_blockMoves.pkl',
            'nResponses': analysis_folder / f'{subj_name}_nResponses.pkl',
            'kernelConditionPlot': analysis_folder / f'{subj_name}_kernelsVolatility.png',
            'blockRegKernels': analysis_folder / f'{subj_name}_blockKernels_regression.pkl',
            'regBetaConditionPlot': analysis_folder / f'{subj_name}_betasVolatility.png',
            'sessRegKernels3': analysis_folder / f'{subj_name}_sessionKernels_regression_3samples.pkl',
            'figSessRegKernels3': analysis_folder / f'{subj_name}_sessionKernels_3samples.png',
            'sessRegKernels5': analysis_folder / f'{subj_name}_sessionKernels_regression_5samples.pkl',
            'figSessRegKernels5': analysis_folder / f'{subj_name}_sessionKernels_5samples.png',
            'adjustments': analysis_folder / f'{subj_name}_adjustments.pkl',
            'adjustmentsVolatility': analysis_folder / f'{subj_name}_adjustmentsSplitByVola.pkl',
            'meanAdjustments': analysis_folder / f'{subj_name}_meanAdjustments.pkl',
            'medianAdjustments': analysis_folder / f'{subj_name}_medianAdjustments.pkl',
            'movements': analysis_folder / f'{subj_name}_movements.pkl',
            'adjustVolatilityFig': analysis_folder / f'{subj_name}_meanAdjust2Volatility.png',
            'adjustMedianVolatilityFig': analysis_folder / f'{subj_name}_medianAdjust2Volatility.png',
            'adjustNoiseFig': analysis_folder / f'{subj_name}_meanAdjust2Noise.png',
            'adjustMedianNoiseFig': analysis_folder / f'{subj_name}_medianAdjust2Noise.png',
            'rtFig': analysis_folder / f'{subj_name}_rts.png',
        }
    }
    return details
