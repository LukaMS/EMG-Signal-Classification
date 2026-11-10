"""Filtering, windowing, and feature extraction."""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import butter, sosfilt, sosfilt_zi, sosfiltfilt

from .config import NUM_CHANNELS, PipelineConfig

# Feature extraction is done in batches to keep memory use modest.
_BATCH_SIZE = 4_096


def _validate_signal(signal: np.ndarray, *, name: str = "signal") -> np.ndarray:
    array = np.asarray(signal, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != NUM_CHANNELS:
        raise ValueError(
            f"{name} must have shape (samples, {NUM_CHANNELS}); "
            f"received {array.shape}."
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values.")
    return array


def filter_signal(signal: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    """Band-pass raw data with a high-pass filter followed by a low-pass filter.

    By default the filter is causal (it only uses past samples), as it would be
    on the ESP32. It starts from the steady state for the first sample, so the
    large constant offset in the raw data does not cause a start-up spike.
    """

    array = _validate_signal(signal, name="signal")
    high = butter(
        cfg.filter_order, cfg.highpass_hz, btype="highpass", fs=cfg.fs, output="sos"
    )
    low = butter(
        cfg.filter_order, cfg.lowpass_hz, btype="lowpass", fs=cfg.fs, output="sos"
    )
    sos = np.vstack((high, low))

    if not cfg.causal_filter:
        return sosfiltfilt(sos, array, axis=0)

    initial_state = sosfilt_zi(sos)[:, :, None] * array[0][None, None, :]
    filtered, _ = sosfilt(sos, array, axis=0, zi=initial_state)
    return filtered


def window_signal(
    signal: np.ndarray, window_samples: int, hop_samples: int
) -> np.ndarray:
    """Cut a signal into windows of window_samples, starting every hop_samples.

    Returns an array of shape (windows, window_samples, 4).
    """

    array = _validate_signal(signal, name="signal")
    if window_samples <= 0 or hop_samples <= 0:
        raise ValueError("window_samples and hop_samples must be positive.")
    if array.shape[0] < window_samples:
        raise ValueError(
            f"Signal has {array.shape[0]} samples, fewer than the "
            f"window length of {window_samples}."
        )

    windows = sliding_window_view(array, window_samples, axis=0)  # (n, 4, window)
    return windows[::hop_samples].transpose(0, 2, 1)


def windows_ending_at(
    signal: np.ndarray, end_indices: np.ndarray, window_samples: int
) -> tuple[np.ndarray, np.ndarray]:
    """Cut one window ending at each of the given sample indices (inclusive).

    Returns (windows, valid). `valid` has one entry per requested index and is
    False where a full window does not fit inside the signal; no window is
    returned for those entries.
    """

    array = _validate_signal(signal, name="signal")
    ends = np.asarray(end_indices, dtype=np.int64)
    if ends.ndim != 1:
        raise ValueError("end_indices must be one-dimensional.")
    if window_samples <= 0:
        raise ValueError("window_samples must be positive.")

    valid = (ends - window_samples + 1 >= 0) & (ends < array.shape[0])
    offsets = np.arange(-window_samples + 1, 1)
    windows = array[ends[valid][:, None] + offsets[None, :]]
    return windows, valid


def compute_features(windows: np.ndarray) -> np.ndarray:
    """Return a (windows, 16) feature matrix from (windows, samples, 4) data.

    Per channel: RMS, waveform length, zero crossings, and slope sign changes,
    laid out channel by channel.
    """

    array = np.asarray(windows, dtype=np.float64)
    if array.ndim != 3 or array.shape[2] != NUM_CHANNELS:
        raise ValueError(
            f"windows must have shape (windows, samples, {NUM_CHANNELS}); "
            f"received {array.shape}."
        )
    if array.shape[1] < 3:
        raise ValueError("Each window needs at least 3 samples.")
    if not np.isfinite(array).all():
        raise ValueError("windows contain NaN or infinite values.")

    batches = []
    for start in range(0, array.shape[0], _BATCH_SIZE):
        batch = array[start : start + _BATCH_SIZE]
        differences = np.diff(batch, axis=1)

        rms = np.sqrt(np.mean(np.square(batch), axis=1))
        waveform_length = np.sum(np.abs(differences), axis=1)
        # The paper gives no noise threshold for ZC or SSC, so sign changes are
        # counted directly.
        zero_crossings = np.sum(batch[:, :-1] * batch[:, 1:] < 0, axis=1)
        slope_sign_changes = np.sum(
            differences[:, :-1] * differences[:, 1:] < 0, axis=1
        )

        stacked = np.stack(
            (rms, waveform_length, zero_crossings, slope_sign_changes), axis=2
        )  # (batch, channels, 4 features)
        batches.append(stacked.reshape(batch.shape[0], -1))

    if not batches:
        return np.empty((0, NUM_CHANNELS * 4))
    return np.vstack(batches)


def compute_window_features(window: np.ndarray) -> np.ndarray:
    """Return the 16 features for a single (samples, 4) window."""

    array = _validate_signal(window, name="window")
    return compute_features(array[None, :, :])[0]