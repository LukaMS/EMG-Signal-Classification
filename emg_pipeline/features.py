"""Filtering, windowing, and feature extraction for the EMG data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.signal import butter, sosfiltfilt

from .data import EXPECTED_CHANNELS, Trial

LOW_CUTOFF_HZ = 20.0
HIGH_CUTOFF_HZ = 400.0
FILTER_ORDER = 2
WINDOW_MILLISECONDS = 64
HOP_MILLISECONDS = 32


@dataclass(frozen=True)
class FeatureSet:
    """Feature rows and their movement labels."""

    X: np.ndarray
    y: np.ndarray


def _validate_signal(signal: np.ndarray, *, name: str = "signal") -> np.ndarray:
    array = np.asarray(signal, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != EXPECTED_CHANNELS:
        raise ValueError(
            f"{name} must have shape (samples, {EXPECTED_CHANNELS}); "
            f"received {array.shape}."
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values.")
    return array


def filter_recording(signal: np.ndarray, fs: int) -> np.ndarray:
    """Apply an offline zero-phase, second-order 20-400 Hz Butterworth filter."""

    array = _validate_signal(signal, name="recording")
    if fs <= 2 * HIGH_CUTOFF_HZ:
        raise ValueError(
            f"Sampling rate must be greater than {2 * HIGH_CUTOFF_HZ:g} Hz "
            f"for a {HIGH_CUTOFF_HZ:g} Hz cutoff; received {fs} Hz."
        )

    sos = butter(
        FILTER_ORDER,
        [LOW_CUTOFF_HZ, HIGH_CUTOFF_HZ],
        btype="bandpass",
        fs=fs,
        output="sos",
    )
    return sosfiltfilt(sos, array, axis=0)


def _milliseconds_to_samples(milliseconds: int, fs: int, *, name: str) -> int:
    samples = milliseconds * fs / 1_000
    rounded = int(round(samples))
    if not np.isclose(samples, rounded):
        raise ValueError(
            f"{name} ({milliseconds} ms) is not an integer number of samples at {fs} Hz."
        )
    return rounded


def window_signal(
    signal: np.ndarray,
    fs: int,
    *,
    window_ms: int = WINDOW_MILLISECONDS,
    hop_ms: int = HOP_MILLISECONDS,
) -> tuple[np.ndarray, np.ndarray]:
    """Split one trial into overlapping windows without crossing its boundaries."""

    array = _validate_signal(signal, name="trial signal")
    window_samples = _milliseconds_to_samples(window_ms, fs, name="Window length")
    hop_samples = _milliseconds_to_samples(hop_ms, fs, name="Hop length")
    if window_samples <= 0 or hop_samples <= 0:
        raise ValueError("Window and hop lengths must be positive.")
    if array.shape[0] < window_samples:
        raise ValueError(
            f"Trial contains {array.shape[0]} samples, fewer than the "
            f"{window_samples}-sample window."
        )

    starts = np.arange(
        0, array.shape[0] - window_samples + 1, hop_samples, dtype=np.int64
    )
    windows = np.stack([array[start : start + window_samples] for start in starts])
    return windows, starts


def compute_window_features(window: np.ndarray) -> np.ndarray:
    """Return channel-major RMS, WL, ZC, and SSC features for one window."""

    array = _validate_signal(window, name="window")

    differences = np.diff(array, axis=0)
    rms = np.sqrt(np.mean(np.square(array), axis=0))
    waveform_length = np.sum(np.abs(differences), axis=0)

    # The paper does not give a noise threshold for ZC or SSC, so sign changes
    # are counted directly.
    zero_crossings = np.sum(array[:-1] * array[1:] < 0, axis=0)

    first_slopes = differences[:-1]
    second_slopes = differences[1:]
    slope_sign_changes = np.sum(first_slopes * second_slopes < 0, axis=0)

    # Rows are channels; columns are features, then flattened channel-by-channel.
    features = np.column_stack(
        (rms, waveform_length, zero_crossings, slope_sign_changes)
    ).reshape(-1)
    return np.asarray(features, dtype=np.float64)


def extract_features(
    trials: Sequence[Trial],
    fs: int,
) -> FeatureSet:
    """Extract 16-element feature vectors from a sequence of labelled trials."""

    if not trials:
        raise ValueError("At least one trial is required for feature extraction.")

    feature_rows: list[np.ndarray] = []
    labels: list[str] = []

    for trial in trials:
        windows, _ = window_signal(trial.signal, fs)
        for window in windows:
            feature_rows.append(compute_window_features(window))
            labels.append(trial.label)

    X = np.vstack(feature_rows)
    if X.shape[1] != EXPECTED_CHANNELS * 4:
        raise RuntimeError(
            f"Expected 16 features per window, generated {X.shape[1]}."
        )
    if not np.isfinite(X).all():
        raise ValueError("Extracted feature matrix contains NaN or infinite values.")

    return FeatureSet(
        X=X,
        y=np.asarray(labels, dtype=str),
    )
