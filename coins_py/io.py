from __future__ import annotations

from pathlib import Path
import csv
from typing import Iterable

import numpy as np
import pandas as pd

from .utils import save_pickle

EXPECTED_COLUMNS = [
    'blockID', 'currentFrame', 'laserRotation', 'shieldRotation', 'shieldDegrees',
    'currentHit', 'totalReward', 'sendTrigger', 'triggerValue', 'trueMean',
    'trueVariance', 'volatility', 'eyepositionX', 'eyepositionY'
]

NUMERIC_COLUMNS = [
    'blockID', 'currentFrame', 'laserRotation', 'shieldRotation', 'shieldDegrees',
    'totalReward', 'triggerValue', 'trueMean', 'trueVariance', 'volatility'
]


def _normalise_row(row: list[str]) -> list[str]:
    if not row:
        return [''] * len(EXPECTED_COLUMNS)
    row[0] = row[0].lstrip('[').strip()
    if len(row) < len(EXPECTED_COLUMNS):
        row = row + [''] * (len(EXPECTED_COLUMNS) - len(row))
    if len(row) > len(EXPECTED_COLUMNS):
        row = row[:len(EXPECTED_COLUMNS) - 2] + [','.join(row[len(EXPECTED_COLUMNS) - 2:-1]), row[-1]]
    return row


def coins_load_saved_data(file_name: str | Path, data_flag: str | None = None) -> pd.DataFrame:
    file_name = Path(file_name)
    rows: list[list[str]] = []
    with file_name.open('r', encoding='utf-8', errors='ignore', newline='') as f:
        reader = csv.reader(f)
        header_skipped = False
        for row in reader:
            if not header_skipped:
                header_skipped = True
                continue
            if not row or all(cell.strip() == '' for cell in row):
                continue
            rows.append(_normalise_row([cell.strip() for cell in row]))

    df = pd.DataFrame(rows, columns=EXPECTED_COLUMNS)
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    for col in ['currentHit', 'sendTrigger', 'eyepositionX', 'eyepositionY']:
        df[col] = df[col].astype(str)

    if df['volatility'].dropna().isin([0, 1]).all():
        pass
    else:
        df['volatility'] = pd.to_numeric(df['volatility'].replace({'stable': 0, 'volatile': 1}), errors='coerce')
    return df


def coins_load_subject_data(details: dict) -> pd.DataFrame:
    subj_suffix = details['subjName'][-2:]
    data_flag = 'initial' if int(subj_suffix) < 10 else 'later'

    all_sessions: list[pd.DataFrame] = []
    for i_sess, file_name in enumerate(details['raw']['behav']['sessionFileNames'], start=1):
        sess_data = coins_load_saved_data(file_name, data_flag=data_flag)
        sess_data['sessID'] = i_sess
        all_sessions.append(sess_data)

    sub_data = pd.concat(all_sessions, ignore_index=True)
    sub_data['blockIDorg'] = sub_data['blockID']

    new_ids = np.full(len(sub_data), np.nan)
    block_count = 1
    if len(sub_data):
        new_ids[0] = np.nan
    for i in range(1, len(sub_data)):
        if sub_data.loc[i, 'blockID'] != sub_data.loc[i - 1, 'blockID']:
            block_count += 1
        new_ids[i] = block_count

    sub_data['blockIDall'] = new_ids
    block_ids = new_ids.copy()
    block_ids[block_ids > 12] -= 12
    block_ids[block_ids > 8] -= 8
    block_ids[block_ids > 4] -= 4
    sub_data['blockID'] = block_ids
    return sub_data
