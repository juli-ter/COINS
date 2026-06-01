from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import math
import os

from .sub_meg.triggers import coins_meg_trigger_list


@dataclass
class BehavOptions:
    flagLoadData: bool = True
    flagPerformance: bool = True
    flagKernels: bool = True
    flagAdjustments: bool = True
    flagReactionTimes: bool = True
    fsample: int = 60
    movAvgWin: int = 100
    minResponseDistance: int = 20
    minStepSize: int = 10
    kernelPreSamples: int = 6 * 60
    kernelPostSamples: int = 1 * 60
    kernelPreSamplesEvi: int = 5
    flagUseBinaryRegression: bool = False
    flagNormaliseEvidence: bool = True
    flagBaselineCorrectKernels: bool = False
    nSamplesKernelBaseline: int = int(1.5 * 60)
    adjustPreSamples: int = 100
    adjustPostSamples: int = 500
    flagNormaliseAdjustments: bool = True
    meanJumpSet: tuple[float, ...] = field(default_factory=lambda: tuple(v * 20 * math.pi / 180 for v in [-3, -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 3]))
    varianceSet: tuple[float, ...] = field(default_factory=lambda: tuple(v * math.pi / 180 for v in [10, 20, 30]))
    maxRtSamples: int = 2 * 60
    flagRegKernelSamples: int = 5


@dataclass
class Options:
    codeDir: Path
    mainDir: Path
    rawDir: Path
    workDir: Path
    subjectIDs: list[int]
    pilotIDs: list[str]
    conditionLabels: list[str]
    conditionIDs: list[int]
    behav: BehavOptions
    meg: dict


def coins_options(main_dir: str | os.PathLike[str] | None = None) -> Options:
    code_dir = Path(__file__).resolve().parent
    if main_dir is None:
        env_main = os.environ.get('COINS_MAIN_DIR')
        if env_main:
            main_dir = Path(env_main)
        else:
            main_dir = code_dir.parent
    main_dir = Path(main_dir).resolve()
    raw_dir = main_dir / 'rawData'
    work_dir = main_dir / 'analysis'
    return Options(
        codeDir=code_dir,
        mainDir=main_dir,
        rawDir=raw_dir,
        workDir=work_dir,
        subjectIDs=list(range(1, 23)),
        pilotIDs=['Ryan', 'Caroline', 'Karen', 'CarolineY'],
        conditionLabels=['Stable', 'Volatile'],
        conditionIDs=[0, 1],
        behav=BehavOptions(),
        meg=coins_meg_trigger_list(),
    )
