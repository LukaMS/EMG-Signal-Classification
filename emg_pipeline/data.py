"""Load one Ozdemir dataset recording and split it into gesture trials."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat

EXPECTED_FS = 2_000
EXPECTED_CHANNELS = 4
MINIMUM_RECORDING_SECONDS = 640
TRIAL_SECONDS = 6

# Start of the first (rest) gesture in each published repetition.
REPETITION_START_SECONDS = (4, 138, 272, 406, 540)

# Gesture numbers are the slots used by the dataset's published segmentation code.
# This tuple is chronological within each repetition.
GESTURES = (
    (0, "rest", "no_input"),
    (1, "wrist_extension", "right"),
    (2, "wrist_flexion", "left"),
    (6, "finger_abduction", "drop"),
    (8, "supination", "rotate_right"),
    (9, "pronation", "rotate_left"),
)

# Paper control order, used for metrics and display.
CLASS_LABELS = (
    "rest",
    "wrist_flexion",
    "wrist_extension",
    "supination",
    "pronation",
    "finger_abduction",
)
@dataclass(frozen=True)
class Recording:
    """One continuous four-channel participant recording."""

    signal: np.ndarray
    fs: int
    participant_id: str


@dataclass(frozen=True)
class Trial:
    """One labelled six-second gesture interval."""

    signal: np.ndarray
    label: str
    control: str
    gesture_id: int
    repetition: int
    start_sample: int


def _read_scalar(mat: dict[str, object], key: str) -> float:
    if key not in mat:
        raise ValueError(f"MAT file is missing required variable '{key}'.")
    value = np.asarray(mat[key]).squeeze()
    if value.size != 1:
        raise ValueError(f"MAT variable '{key}' must contain exactly one value.")
    return float(value)


def _normalise_signal(data: object) -> np.ndarray:
    signal = np.asarray(data)
    if signal.ndim != 2:
        raise ValueError(
            "MAT variable 'data' must be a two-dimensional samples-by-channels array."
        )

    if signal.shape[1] == EXPECTED_CHANNELS:
        normalised = signal
    elif signal.shape[0] == EXPECTED_CHANNELS:
        normalised = signal.T
    else:
        raise ValueError(
            "Expected exactly 4 EMG channels in MAT variable 'data'; "
            f"received shape {signal.shape}."
        )

    if not np.issubdtype(normalised.dtype, np.number):
        raise ValueError("MAT variable 'data' must contain numeric EMG samples.")

    normalised = np.asarray(normalised, dtype=np.float64)
    if not np.isfinite(normalised).all():
        raise ValueError("EMG recording contains NaN or infinite values.")
    return normalised


def load_recording(path: str | Path) -> Recording:
    """Load and validate one raw participant MAT file.

    The returned signal always has shape ``(samples, 4)`` even when the source
    file stores its channels as rows.
    """

    mat_path = Path(path)
    if not mat_path.is_file():
        raise FileNotFoundError(
            f"EMG participant file was not found at '{mat_path}'. Download one raw "
            "participant .mat file and save it as data/raw/participant.mat."
        )

    mat = loadmat(mat_path)
    if "data" not in mat:
        raise ValueError("MAT file is missing required variable 'data'.")

    signal = _normalise_signal(mat["data"])
    fs_value = _read_scalar(mat, "fs")
    if not np.isclose(fs_value, EXPECTED_FS):
        raise ValueError(
            f"Expected a {EXPECTED_FS} Hz recording, but MAT variable 'fs' is "
            f"{fs_value:g} Hz. Use a raw file from the documented dataset."
        )

    minimum_samples = MINIMUM_RECORDING_SECONDS * EXPECTED_FS
    if signal.shape[0] < minimum_samples:
        seconds = signal.shape[0] / EXPECTED_FS
        raise ValueError(
            f"Recording is too short ({seconds:.2f} s). The published segmentation "
            f"schedule requires at least {MINIMUM_RECORDING_SECONDS} s."
        )

    participant_id = mat_path.stem
    if "iD" in mat:
        participant_value = np.asarray(mat["iD"]).squeeze()
        if participant_value.size == 1:
            participant_id = str(participant_value.item())

    return Recording(signal=signal, fs=EXPECTED_FS, participant_id=participant_id)


def segment_trials(recording: Recording) -> list[Trial]:
    """Extract the six selected gestures from all five repetitions."""

    if recording.fs != EXPECTED_FS:
        raise ValueError(
            f"Segmentation expects {EXPECTED_FS} Hz, received {recording.fs} Hz."
        )
    if recording.signal.ndim != 2 or recording.signal.shape[1] != EXPECTED_CHANNELS:
        raise ValueError(
            f"Segmentation expects samples x {EXPECTED_CHANNELS} channels; "
            f"received {recording.signal.shape}."
        )
    if not np.isfinite(recording.signal).all():
        raise ValueError("EMG recording contains NaN or infinite values.")

    trials: list[Trial] = []
    trial_samples = TRIAL_SECONDS * recording.fs

    for repetition, repetition_start in enumerate(REPETITION_START_SECONDS, start=1):
        for gesture_id, label, control in GESTURES:
            start_second = repetition_start + gesture_id * 10
            start_sample = start_second * recording.fs
            end_sample = start_sample + trial_samples
            if end_sample > recording.signal.shape[0]:
                available_seconds = recording.signal.shape[0] / recording.fs
                raise ValueError(
                    "Recording is too short for the published segmentation schedule: "
                    f"gesture '{label}' in repetition {repetition} ends at "
                    f"{end_sample / recording.fs:.1f} s, but only "
                    f"{available_seconds:.1f} s are available."
                )
            trials.append(
                Trial(
                    signal=recording.signal[start_sample:end_sample],
                    label=label,
                    control=control,
                    gesture_id=gesture_id,
                    repetition=repetition,
                    start_sample=start_sample,
                )
            )

    return trials
