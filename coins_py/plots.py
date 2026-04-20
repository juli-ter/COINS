from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .colors import coins_colours
from .options import Options


def save_figure(fig: plt.Figure, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def coins_plot_block_data(block_data: pd.DataFrame, options: Options) -> plt.Figure:
    shield = np.unwrap(block_data['shieldRotation'].to_numpy(dtype=float) * np.pi / 180)
    laser = np.unwrap(block_data['laserRotation'].to_numpy(dtype=float) * np.pi / 180)
    shield_degrees = block_data['shieldDegrees'].to_numpy(dtype=float) * np.pi / 180
    true_mean = np.unwrap(block_data['trueMean'].to_numpy(dtype=float) * np.pi / 180)
    true_variance = block_data['trueVariance'].to_numpy(dtype=float) * np.pi / 180

    if np.sum(np.abs(shield - laser)) > np.sum(np.abs(shield - (laser + 2 * np.pi))):
        laser = laser + 2 * np.pi
        true_mean = true_mean + 2 * np.pi
    if np.sum(np.abs(shield - true_mean)) > np.sum(np.abs(shield - (true_mean + 2 * np.pi))):
        true_mean = true_mean + 2 * np.pi

    prediction_error = laser - shield
    abs_pe = np.abs(prediction_error)
    time_idx = np.arange(1, len(abs_pe) + 1) / options.behav.fsample / 60
    vol_value = block_data['volatility'].iloc[0]

    if pd.isna(vol_value):
        con = 'UNKNOWN'
    else:
        con = 'VOLATILE' if int(vol_value) == 1 else 'STABLE'

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    ax = axes[0]
    ax.plot(time_idx, shield, label='shield position')
    ax.fill_between(time_idx, shield - 0.5 * shield_degrees, shield + 0.5 * shield_degrees, alpha=0.2)
    ax.plot(time_idx, laser, label='laser location')
    ax.plot(time_idx, true_mean, '--', linewidth=2, label='true mean')
    ax.set_title(f'session{int(block_data["sessID"].iloc[0])}, block{int(block_data["blockID"].iloc[0])} | {con}')
    ax.set_xlabel('Time (min) across block')
    ax.set_ylabel('location in radians')
    ax.legend()

    ax = axes[1]
    ax.plot(time_idx, abs_pe, label='abs PE')
    ax.plot(time_idx, 0.5 * shield_degrees, label='shield size')
    ax.plot(time_idx, true_variance, '--', linewidth=2, label='true variance')
    ax.set_title('Absolute prediction error and shield size over time')
    ax.set_xlabel('Time (min) across block')
    ax.set_ylabel('Radians')
    ax.legend()
    return fig


def coins_plot_participant_performance(perform: list[list[dict | None]]) -> plt.Figure:
    sample = next(p for row in perform for p in row if p)
    fields = [k for k, v in sample.items() if np.isscalar(v)]
    fig, axes = plt.subplots(5, 6, figsize=(18, 12))
    axes = axes.ravel()
    for i, field in enumerate(fields[:30]):
        vol_values = []
        sta_values = []
        for row in perform:
            for p in row:
                if not p:
                    continue
                (vol_values if p['volatility'] == 1 else sta_values).append(p[field])
        ax = axes[i]
        ax.plot(np.ones(len(vol_values)), vol_values, 'o')
        ax.plot(np.ones(len(sta_values)) * 2, sta_values, 'o')
        if vol_values:
            ax.plot(1, np.mean(vol_values), 'sk')
        if sta_values:
            ax.plot(2, np.mean(sta_values), 'sk')
        if vol_values and sta_values:
            ax.plot([1, 2], [np.mean(vol_values), np.mean(sta_values)], '-k')
        ax.set_xlim(0, 3)
        ax.set_xticks([1, 2], ['volatile', 'stable'])
        ax.set_title(field)
    return fig


def coins_plot_performance_overview(perform: list[list[dict | None]], field_name: str) -> plt.Figure:
    col = coins_colours()
    data = {0: [], 1: []}
    for row in perform:
        for p in row:
            if p:
                vol = p.get('volatility', None)
                if vol in data and p.get(field_name) is not None:
                    data[vol].append(p[field_name])
    fig, ax = plt.subplots(figsize=(8, 4))
    for vol, label, color in [(0, 'stable', col['stable']), (1, 'volatile', col['volatile'])]:
        xs = np.arange(1, len(data[vol]) + 1)
        ax.plot(xs, data[vol], 'o--', color=color, label=label)
    if field_name != 'reward':
        ax.axhline(0.1745, color='gray')
    ax.set_xlabel('block')
    ax.set_ylabel(field_name)
    ax.set_title(f'{field_name} across blocks')
    ax.legend()
    return fig


def coins_plot_subject_kernels_by_volatility(sta_kernels: np.ndarray, vol_kernels: np.ndarray, n_responses: dict, details: dict, options: Options) -> plt.Figure:
    col = coins_colours()
    time_axis = np.arange(-options.behav.kernelPreSamples, options.behav.kernelPostSamples + 1) / options.behav.fsample

    vola = np.nanmean(np.nanmean(vol_kernels[:, :, 0, :], axis=0), axis=0)
    stab = np.nanmean(np.nanmean(sta_kernels[:, :, 0, :], axis=0), axis=0)
    if options.behav.flagBaselineCorrectKernels:
        vola = vola - np.nanmean(vola[:options.behav.nSamplesKernelBaseline])
        stab = stab - np.nanmean(stab[:options.behav.nSamplesKernelBaseline])

    vola_up = np.nanmean(np.nanmean(vol_kernels[:, :, 4, :], axis=0), axis=0)
    stab_up = np.nanmean(np.nanmean(sta_kernels[:, :, 4, :], axis=0), axis=0)
    vola_down = np.nanmean(np.nanmean(vol_kernels[:, :, 5, :], axis=0), axis=0)
    stab_down = np.nanmean(np.nanmean(sta_kernels[:, :, 5, :], axis=0), axis=0)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    ax = axes[0]
    ax.plot(time_axis, vola, color=col['volatile'])
    ax.plot(time_axis, stab, color=col['stable'])
    ax.axhline(0, color='gray')
    ax.set_xlabel('time (s) around button press')
    ax.set_ylabel('signed PE')
    ax.set_title(f"{details['subjName']}: Average signed PEs leading up to shield movement onset")
    ax.legend([
        f"volatile blocks (N={n_responses['volatile']['move']})",
        f"stable blocks (N={n_responses['stable']['move']})",
    ])

    ax = axes[1]
    ax.plot(time_axis, vola_up)
    ax.plot(time_axis, vola_down)
    ax.plot(time_axis, stab_up)
    ax.plot(time_axis, stab_down)
    ax.set_xlabel('time (s) around button press')
    ax.set_ylabel('absolute PE')
    ax.set_title('Average absolute PEs leading up to shield size update')
    ax.legend([
        f"volatile up (N={n_responses['volatile']['sizeUp']})",
        f"volatile down (N={n_responses['volatile']['sizeDown']})",
        f"stable up (N={n_responses['stable']['sizeUp']})",
        f"stable down (N={n_responses['stable']['sizeDown']})",
    ])
    return fig


def coins_plot_subject_betas_by_volatility(betas_con: list[np.ndarray], trials_con: list[np.ndarray], options: Options) -> plt.Figure:
    col = coins_colours()
    time_axis = np.arange(-options.behav.kernelPreSamplesEvi, 0)
    vola = np.nanmean(betas_con[1], axis=0)
    stab = np.nanmean(betas_con[0], axis=0)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(time_axis, vola, color=col['volatile'])
    ax.plot(time_axis, stab, color=col['stable'])
    ax.axhline(0, color=col['medNoise'])
    ax.set_xlabel('evidence samples leading up to button press')
    ax.set_ylabel('avg weight on decision')
    ax.set_title('Avg weight of evidence samples leading up to shield movement onset')
    ax.legend([
        f'volatile blocks (N={int(np.nansum(trials_con[1]))})',
        f'stable blocks (N={int(np.nansum(trials_con[0]))})',
    ])
    return fig


def coins_plot_subject_reg_kernels_session_wise(vol_betas: np.ndarray, sta_betas: np.ndarray) -> plt.Figure:
    col = coins_colours()
    n_betas = vol_betas.shape[1]
    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(n_betas):
        for j in range(vol_betas.shape[0]):
            ax.plot([i + 1 - 0.2, i + 1 + 0.2], [sta_betas[j, i], vol_betas[j, i]], '-', color=col['highNoise'])
        ax.axhline(0, color='gray')
        ax.plot(np.full(vol_betas.shape[0], i + 1 - 0.2), sta_betas[:, i], 'o', color=col['stable'])
        ax.plot(np.full(vol_betas.shape[0], i + 1 + 0.2), vol_betas[:, i], 'o', color=col['volatile'])
    if n_betas == 11:
        labels = ['-5', '-4', '-3', '-2', '-1', 'noise', 'noise*(-5)', 'noise*(-4)', 'noise*(-3)', 'noise*(-2)', 'noise*(-1)']
    else:
        labels = ['-3', '-2', '-1', 'noise', 'noise*(-3)', 'noise*(-2)', 'noise*(-1)']
    ax.set_xticks(range(1, n_betas + 1), labels, rotation=45)
    ax.set_ylabel('weight on response')
    ax.set_xlabel('regressor')
    ax.set_title(f'betas: {n_betas} regressors')
    return fig


def _normalize_adjusts(adjusts: np.ndarray, pre: int) -> np.ndarray:
    out = adjusts.copy()
    for i_con in range(out.shape[0]):
        for i_var in range(out.shape[1]):
            for i_jmp in range(out.shape[2]):
                out[i_con, i_var, i_jmp, :] = out[i_con, i_var, i_jmp, :] - out[i_con, i_var, i_jmp, pre]
    return out


def coins_plot_subject_adjustments(adjusts: np.ndarray, jump_sizes: np.ndarray, options: Options) -> tuple[plt.Figure, plt.Figure]:
    pre = options.behav.adjustPreSamples
    post = options.behav.adjustPostSamples
    fsmp = options.behav.fsample
    time_axis = np.arange(-pre, post + 1) / fsmp
    if options.behav.flagNormaliseAdjustments:
        adjusts = _normalize_adjusts(adjusts, pre)

    fig1, ax1 = plt.subplots(figsize=(8, 5))

    sta = adjusts[1, 0, :, :]   # shape: (n_pairs, time)
    vol = adjusts[2, 0, :, :]

    n_pairs = sta.shape[0]

    # get a colormap with enough distinct colors
    colors = plt.cm.tab10(range(n_pairs))  # or plt.cm.viridis(...)

    for i in range(n_pairs):
        ax1.plot(time_axis, sta[i, :], linewidth=2, color=colors[i],
                label='stable' if i == 0 else None)
        ax1.plot(time_axis, vol[i, :], '--', linewidth=2, color=colors[i],
                label='volatile' if i == 0 else None)

    for j in jump_sizes:
        ax1.plot([0, 8], [j, j], color='gray')

    ax1.axhline(0, color='gray')
    ax1.axvline(0, color='gray')

    ax1.set_xlim(-1, 4)
    ax1.set_xlabel('time from change point (s)')
    ax1.set_ylabel('shield adjustment (rad)')
    ax1.set_title('Effect of volatility')
    ax1.legend()

    fig2, ax2 = plt.subplots(figsize=(8, 5))

    low  = adjusts[0, 1, :, :]
    med  = adjusts[0, 2, :, :]
    high = adjusts[0, 3, :, :]
    n_pairs = low.shape[0]
    colors = plt.cm.tab10(range(n_pairs))
    for i in range(n_pairs):
        ax2.plot(time_axis, low[i, :],  linewidth=2, color=colors[i],
                label='low noise' if i == 0 else None)
        ax2.plot(time_axis, med[i, :],  '--', linewidth=2, color=colors[i],
                label='medium noise' if i == 0 else None)
        ax2.plot(time_axis, high[i, :], '-.', linewidth=2, color=colors[i],
                label='high noise' if i == 0 else None)

    for j in jump_sizes:
        ax2.plot([0, 8], [j, j], color='gray')

    ax2.axhline(0, color='gray')
    ax2.axvline(0, color='gray')
    ax2.set_xlim(-1, 4)
    ax2.set_xlabel('time from change point (s)')
    ax2.set_ylabel('shield adjustment (rad)')
    ax2.set_title('Effect of noise')
    ax2.legend()
    return fig1, fig2


def coins_plot_group_adjustments(avg_adjusts: np.ndarray, se_adjusts: np.ndarray, jump_sizes: np.ndarray, options: Options) -> tuple[plt.Figure, plt.Figure]:
    pre = options.behav.adjustPreSamples
    post = options.behav.adjustPostSamples
    time_axis = np.arange(-pre, post + 1) / options.behav.fsample
    col = coins_colours()

    def plot_pair(data_a, err_a, data_b, err_b, color_a, color_b, title):
        fig, ax = plt.subplots(figsize=(8, 5))
        for i in range(data_a.shape[0]):
            ax.plot(time_axis, data_a[i], color=color_a, alpha=0.6)
            ax.fill_between(time_axis, data_a[i] - err_a[i], data_a[i] + err_a[i], color=color_a, alpha=0.1)
            ax.plot(time_axis, data_b[i], color=color_b, alpha=0.6, linestyle='--')
            ax.fill_between(time_axis, data_b[i] - err_b[i], data_b[i] + err_b[i], color=color_b, alpha=0.1)
        for j in jump_sizes:
            ax.plot([0, time_axis[-1]], [j, j], color='gray', alpha=0.3)
        ax.axhline(0, color='gray')
        ax.axvline(0, color='gray')
        ax.set_title(title)
        ax.set_xlabel('time from change point (s)')
        ax.set_ylabel('adjustment (rad)')
        return fig

    fig1 = plot_pair(avg_adjusts[1, 0, :, :], se_adjusts[1, 0, :, :], avg_adjusts[2, 0, :, :], se_adjusts[2, 0, :, :], col['stable'], col['volatile'], 'Effect of volatility')
    fig2 = plot_pair(avg_adjusts[0, 1, :, :], se_adjusts[0, 1, :, :], avg_adjusts[0, 3, :, :], se_adjusts[0, 3, :, :], col['lowNoise'], col['highNoise'], 'Effect of noise level')
    return fig1, fig2
