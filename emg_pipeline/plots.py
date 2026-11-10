"""Graphs for the notebook. Each function returns a matplotlib Figure."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .config import CHANNEL_NAMES, CLASS_LABELS, NUM_CHANNELS, PLOT_LEVELS, PipelineConfig
from .data import Session
from .evaluation import MetricsResult
from .pipeline import LiveResult

CLASS_COLORS = {
    "Rest": "#7f7f7f",
    "Left": "#1f77b4",
    "Right": "#ff7f0e",
    "Rotate": "#2ca02c",
    "Drop": "#d62728",
}


def plot_calibration_signals(session: Session, cfg: PipelineConfig) -> Figure:
    """Plot every channel of every calibration file (4 channels x 5 movements).

    Each row shares one y-axis, so a channel's amplitude can be compared across
    movements.
    """

    fig, axes = plt.subplots(
        NUM_CHANNELS,
        len(CLASS_LABELS),
        figsize=(15, 8),
        sharex=True,
        sharey="row",
        squeeze=False,
    )
    for column, label in enumerate(CLASS_LABELS):
        signal = session.calibration[label]
        seconds = np.arange(signal.shape[0]) / cfg.fs
        for row in range(NUM_CHANNELS):
            axis = axes[row, column]
            axis.plot(seconds, signal[:, row], color=CLASS_COLORS[label], linewidth=0.5)
            if row == 0:
                axis.set_title(label)
            if column == 0:
                axis.set_ylabel(f"Ch {row + 1}: {CHANNEL_NAMES[row]}")
            if row == NUM_CHANNELS - 1:
                axis.set_xlabel("Time (s)")
    fig.suptitle("Calibration data (already filtered): one column per movement")
    fig.tight_layout()
    return fig


def plot_lda_scatter(model: object, X: np.ndarray, y: np.ndarray) -> Figure:
    """Plot the calibration windows in the first two LDA dimensions."""

    features = np.asarray(X, dtype=np.float64)
    labels = np.asarray(y, dtype=str)
    if features.shape[0] != labels.shape[0]:
        raise ValueError("X and y must have the same number of rows.")

    projected = model.named_steps["lda"].transform(features)
    if projected.shape[1] < 2:
        raise ValueError("The LDA projection needs at least 2 dimensions to plot.")

    fig, axis = plt.subplots(figsize=(7, 6))
    for label in CLASS_LABELS:
        mask = labels == label
        axis.scatter(
            projected[mask, 0],
            projected[mask, 1],
            s=14,
            alpha=0.7,
            color=CLASS_COLORS[label],
            label=label,
        )
    axis.set_xlabel("LDA dimension 1 (a.u.)")
    axis.set_ylabel("LDA dimension 2 (a.u.)")
    axis.set_title("Calibration windows in the first two LDA dimensions")
    axis.legend()
    axis.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_confusion_matrix(metrics: MetricsResult) -> Figure:
    """Heatmap of what we decided for each of the company's decisions.

    Each row adds up to 100%, so the diagonal is the recall of that movement.
    """

    counts = metrics.confusion
    row_totals = counts.sum(axis=1, keepdims=True)
    share = counts / np.where(row_totals == 0, 1, row_totals)

    fig, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(share, cmap="Blues", vmin=0, vmax=1)
    ticks = np.arange(len(metrics.labels))
    axis.set_xticks(ticks, metrics.labels)
    axis.set_yticks(ticks, metrics.labels)
    axis.set_xlabel("Our decision")
    axis.set_ylabel("Company decision (reference)")
    axis.set_title("Our decisions for each company decision (rows add to 100%)", fontsize=11)

    for row in range(counts.shape[0]):
        for column in range(counts.shape[1]):
            axis.text(
                column,
                row,
                f"{share[row, column]:.0%}\n({counts[row, column]})",
                ha="center",
                va="center",
                color="white" if share[row, column] > 0.5 else "black",
                fontsize=9,
            )
    fig.colorbar(image, ax=axis, label="Share of the company's decisions")
    fig.tight_layout()
    return fig


def _set_decision_axis(axis: Axes) -> None:
    """Label an axis with the movement names at their PLOT_LEVELS heights."""

    ordered_names = sorted(PLOT_LEVELS, key=PLOT_LEVELS.get)
    positions = [PLOT_LEVELS[name] for name in ordered_names]
    axis.set_yticks(positions, ordered_names)
    axis.set_ylim(min(positions) - 0.6, max(positions) + 0.6)
    axis.grid(alpha=0.3)


def _decision_levels(names: np.ndarray) -> np.ndarray:
    return np.array([PLOT_LEVELS[name] for name in names])


def plot_decision_timelines(
    times_s: np.ndarray,
    actual: np.ndarray,
    predicted: np.ndarray,
    start_s: float | None = None,
    duration_s: float | None = None,
) -> Figure:
    """Plot the company's and our decisions as step lines over time.

    Two panels share one time axis: the company's decisions on top and ours
    below. Each movement is drawn at the height given by PLOT_LEVELS. Leave
    start_s and duration_s as None for the whole session, or set them to zoom
    in on a stretch of it.
    """

    times = np.asarray(times_s, dtype=np.float64)
    actual_array = np.asarray(actual, dtype=str)
    predicted_array = np.asarray(predicted, dtype=str)
    if not (times.shape == actual_array.shape == predicted_array.shape) or times.ndim != 1:
        raise ValueError("times_s, actual, and predicted must be 1-D and equal in length.")

    first = times[0] if start_s is None else start_s
    last = times[-1] if duration_s is None else first + duration_s
    selected = (times >= first) & (times <= last)
    if not selected.any():
        raise ValueError("No decisions fall inside the selected time range.")
    times = times[selected]
    actual_array = actual_array[selected]
    predicted_array = predicted_array[selected]

    fig, axes = plt.subplots(2, 1, figsize=(15, 6), sharex=True)
    for axis in axes:
        _set_decision_axis(axis)

    axes[0].step(times, _decision_levels(actual_array), where="post", color="tab:blue", linewidth=1.3)
    axes[0].set_title("Company decisions")
    axes[1].step(times, _decision_levels(predicted_array), where="post", color="tab:orange", linewidth=1.3)
    axes[1].set_title("Our decisions")
    axes[1].set_xlabel("Time (s)")

    fig.tight_layout()
    return fig


def plot_decision_windows(
    live: LiveResult,
    signal: np.ndarray,
    cfg: PipelineConfig,
    start_s: float,
    duration_s: float = 3.0,
    highlight_disagreements: bool = True,
    max_windows: int = 400,
) -> Figure:
    """Zoomed view of the decisions above the amplitude of all four channels.

    Rows from the top: the company's decisions, our decisions, then channels 1
    to 4. A vertical line marks the end of each decision window on every row,
    and windows where we disagree with the company are shaded red.

    Each column shows our decision for a window, the company's decision it is
    compared with (cfg.decision_lag steps later in the company's own record),
    and the signal that window was classified from. Pass the filtered live
    recording as `signal`, since that is what the model sees.
    """

    array = np.asarray(signal, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != NUM_CHANNELS:
        raise ValueError(f"signal must have shape (samples, {NUM_CHANNELS}).")
    if duration_s <= 0:
        raise ValueError("duration_s must be positive.")

    window_ends = np.asarray(live.window_end_s, dtype=np.float64)
    actual_array = np.asarray(live.actual, dtype=str)
    predicted_array = np.asarray(live.predicted, dtype=str)

    selected = np.flatnonzero((window_ends >= start_s) & (window_ends <= start_s + duration_s))
    if selected.size == 0:
        raise ValueError("No decisions fall inside the selected time range.")
    if selected.size > max_windows:
        raise ValueError(
            f"The selected range holds {selected.size} decision windows, more than "
            f"max_windows ({max_windows}). Shorten duration_s."
        )

    # Each decision owns the stretch of time between the end of the previous
    # window and the end of its own; `edges` are the boundaries of those stretches.
    sample_period = 1.0 / cfg.fs
    typical_step = (
        float(np.median(np.diff(window_ends))) if window_ends.size > 1 else cfg.window_ms / 1_000
    )
    first_index = selected[0]
    if first_index > 0:
        left_edge = window_ends[first_index - 1] + sample_period
    else:
        left_edge = window_ends[first_index] + sample_period - typical_step
    edges = np.concatenate(([left_edge], window_ends[selected] + sample_period))

    actual_levels = _decision_levels(actual_array[selected])
    predicted_levels = _decision_levels(predicted_array[selected])
    mismatch = actual_levels != predicted_levels

    fig, axes = plt.subplots(
        2 + NUM_CHANNELS,
        1,
        figsize=(15, 12),
        sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1.2, 1, 1, 1, 1]},
    )
    for axis in axes[:2]:
        _set_decision_axis(axis)

    axes[0].step(edges, np.append(actual_levels, actual_levels[-1]), where="post",
                 color="tab:blue", linewidth=1.5)
    axes[0].set_ylabel("Company")
    axes[1].step(edges, np.append(predicted_levels, predicted_levels[-1]), where="post",
                 color="tab:orange", linewidth=1.5)
    axes[1].set_ylabel("Ours")

    first_sample = max(int(np.rint(edges[0] * cfg.fs)), 0)
    last_sample = min(int(np.rint(edges[-1] * cfg.fs)), array.shape[0])
    sample_times = np.arange(first_sample, last_sample) / cfg.fs
    for channel in range(NUM_CHANNELS):
        axis = axes[2 + channel]
        axis.plot(sample_times, array[first_sample:last_sample, channel], color="black", linewidth=0.8)
        axis.set_ylabel(f"Ch {channel + 1}\n{CHANNEL_NAMES[channel]}")
        axis.grid(alpha=0.3)

    for axis in axes:
        axis.vlines(edges, 0, 1, transform=axis.get_xaxis_transform(),
                    colors="gray", linewidth=0.6, alpha=0.7)
        if highlight_disagreements:
            for index in np.flatnonzero(mismatch):
                axis.axvspan(edges[index], edges[index + 1], color="red", alpha=0.12, linewidth=0)

    axes[0].set_xlim(edges[0], edges[-1])
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(
        f"Decision windows from {edges[0]:.2f} s to {edges[-1]:.2f} s: "
        f"we disagree with the company in {int(mismatch.sum())} of {mismatch.size} windows"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig