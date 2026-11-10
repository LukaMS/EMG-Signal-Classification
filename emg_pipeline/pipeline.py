"""Train on the calibration files, then classify the live recording."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import CLASS_LABELS, PipelineConfig
from .data import Session
from .features import compute_features, filter_signal, window_signal, windows_ending_at


@dataclass(frozen=True)
class TrainingSet:
    """Feature windows from the calibration files and their movement labels."""

    X: np.ndarray  # (windows, 16)
    y: np.ndarray  # (windows,) movement name for each window


@dataclass(frozen=True)
class LiveResult:
    """Our classification of the live data next to the company's decisions."""

    times_s: np.ndarray  # time of each company decision we compare with, in seconds
    window_end_s: np.ndarray  # time of the last sample in the window each of our decisions used
    predicted: np.ndarray  # our movement for each decision
    actual: np.ndarray  # the company's movement for each decision
    skipped: int  # decisions with no full window before them (not compared)


def build_training_set(session: Session, cfg: PipelineConfig) -> TrainingSet:
    """Cut each calibration file into windows and extract their features.

    The calibration files are already filtered, so they are not filtered again.
    """

    feature_blocks: list[np.ndarray] = []
    labels: list[str] = []

    for label in CLASS_LABELS:
        signal = session.calibration[label][cfg.calibration_trim_samples :]
        if signal.shape[0] < cfg.window_samples:
            raise ValueError(
                f"Calibration data for '{label}' has {signal.shape[0]} samples after "
                f"trimming, fewer than the window length of {cfg.window_samples}."
            )
        windows = window_signal(signal, cfg.window_samples, cfg.train_hop_samples)
        features = compute_features(windows)
        if features.shape[0] < 2:
            raise ValueError(f"Calibration data for '{label}' gives fewer than 2 windows.")
        feature_blocks.append(features)
        labels.extend([label] * features.shape[0])

    return TrainingSet(X=np.vstack(feature_blocks), y=np.asarray(labels, dtype=str))


def classify_live(session: Session, model: object, cfg: PipelineConfig) -> LiveResult:
    """Classify the live data at each of the company's decision times.

    Each decision timestamp is taken as the time of the last sample in that
    decision's window (the first decision, at 0.063 s, ends at sample 63). We
    cut a window of cfg.window_samples ending at that sample, so every one of
    our decisions covers the same stretch of signal as the company's.
    """

    filtered = filter_signal(session.live, cfg)
    end_indices = np.rint(session.decision_times_s * cfg.fs).astype(np.int64)
    windows, valid = windows_ending_at(filtered, end_indices, cfg.window_samples)
    if not valid.any():
        raise ValueError("No decision has a full window of live data before it.")

        predicted = np.asarray(model.predict(compute_features(windows)), dtype=str)
    predicted = np.asarray(model.predict(compute_features(windows)), dtype=str)
    times = session.decision_times_s[valid]
    actual = session.decisions[valid]

    lag = cfg.decision_lag
    if lag >= len(predicted):
        raise ValueError(
            f"decision_lag ({lag}) must be smaller than the number of decisions "
            f"compared ({len(predicted)})."
        )
    
    window_ends = times
    if lag > 0:
        predicted = predicted[:-lag]
        window_ends = times[:-lag]
        actual = actual[lag:]
        times = times[lag:]

    return LiveResult(
        times_s=times,
        window_end_s=window_ends,
        predicted=predicted,
        actual=actual,
        skipped=int((~valid).sum()),
    )