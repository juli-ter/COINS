from __future__ import annotations

from scipy.io import loadmat
from pathlib import Path


def load_stim_sequence(mat_file: str | Path) -> dict:
    return loadmat(mat_file)


def set_sim_parameters() -> dict:
    return {
        'status': 'placeholder',
        'note': 'MATLAB model code was not fully ported. Behavioural analysis pipeline was ported first.'
    }
