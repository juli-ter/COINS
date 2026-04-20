from __future__ import annotations

from pathlib import Path
from typing import Any
import pickle

import numpy as np
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_pickle(path: str | Path, obj: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open('wb') as f:
        pickle.dump(obj, f)


def load_pickle(path: str | Path) -> Any:
    with Path(path).open('rb') as f:
        return pickle.load(f)


def mat_pinv(x: np.ndarray) -> np.ndarray:
    return np.linalg.pinv(np.asarray(x, dtype=float))


def sem(x: np.ndarray, axis: int = 0) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return np.nanstd(arr, axis=axis, ddof=0) / np.sqrt(np.sum(~np.isnan(arr), axis=axis))


def moving_mean(x: np.ndarray, window: int) -> np.ndarray:
    ser = pd.Series(np.asarray(x, dtype=float))
    return ser.rolling(window=window, min_periods=1).mean().to_numpy()


def compare_wrap_reference(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    ref = np.asarray(reference, dtype=float)
    cand = np.asarray(candidate, dtype=float)
    if np.nansum(np.abs(ref - cand)) > np.nansum(np.abs(ref - (cand + 2 * np.pi))):
        return cand + 2 * np.pi
    return cand


def nanmean_safe(x: np.ndarray, axis: int = 0) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.size == 0:
        return np.array([])
    return np.nanmean(arr, axis=axis)


def nanmedian_safe(x: np.ndarray, axis: int = 0) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.size == 0:
        return np.array([])
    return np.nanmedian(arr, axis=axis)


def row_mean_or_nan(data: np.ndarray, width: int) -> np.ndarray:
    arr = np.asarray(data, dtype=float)
    if arr.size == 0:
        return np.full(width, np.nan)
    if arr.ndim == 1:
        arr = arr[None, :]
    return np.nanmean(arr, axis=0)
