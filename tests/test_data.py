from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import emg_pipeline.data as data_module
from emg_pipeline.data import (
    CLASS_LABELS,
    EXPECTED_FS,
    Recording,
    load_recording,
    segment_trials,
)


def _touch(path: Path) -> None:
    path.write_bytes(b"placeholder")


def test_load_recording_normalises_channel_first_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "participant.mat"
    _touch(path)
    monkeypatch.setattr(data_module, "MINIMUM_RECORDING_SECONDS", 0.01)
    source = np.arange(4 * 40, dtype=float).reshape(4, 40)
    monkeypatch.setattr(
        data_module,
        "loadmat",
        lambda _: {"data": source, "fs": np.array([[2_000]]), "iD": [[7]]},
    )

    recording = load_recording(path)

    assert recording.signal.shape == (40, 4)
    np.testing.assert_array_equal(recording.signal, source.T)
    assert recording.fs == EXPECTED_FS
    assert recording.participant_id == "7"


def test_load_recording_reports_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="data/raw/participant.mat"):
        load_recording(tmp_path / "missing.mat")


@pytest.mark.parametrize(
    ("mat", "message"),
    [
        ({"data": np.zeros((100, 3)), "fs": [[2_000]]}, "exactly 4"),
        ({"data": np.zeros((100, 4)), "fs": [[1_000]]}, "2000 Hz"),
        ({"data": np.zeros((100, 4)), "fs": [[2_000]]}, "too short"),
        (
            {"data": np.full((100, 4), np.nan), "fs": [[2_000]]},
            "NaN or infinite",
        ),
    ],
)
def test_load_recording_validation_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mat: dict[str, object],
    message: str,
) -> None:
    path = tmp_path / "participant.mat"
    _touch(path)
    monkeypatch.setattr(data_module, "loadmat", lambda _: mat)

    with pytest.raises(ValueError, match=message):
        load_recording(path)


def test_segment_trials_reproduces_published_schedule() -> None:
    # A zero-stride view avoids allocating the full 20 MB synthetic recording.
    signal = np.broadcast_to(np.zeros((1, 4)), (640 * EXPECTED_FS, 4))
    recording = Recording(signal=signal, fs=EXPECTED_FS, participant_id="synthetic")

    trials = segment_trials(recording)

    assert len(trials) == 30
    assert {trial.repetition for trial in trials} == {1, 2, 3, 4, 5}
    assert {trial.label for trial in trials} == set(CLASS_LABELS)
    assert all(trial.signal.shape == (6 * EXPECTED_FS, 4) for trial in trials)
    assert trials[0].start_sample == 4 * EXPECTED_FS
    assert trials[-1].start_sample == (540 + 9 * 10) * EXPECTED_FS


def test_segment_trials_rejects_short_recording() -> None:
    recording = Recording(
        signal=np.zeros((10_000, 4)), fs=EXPECTED_FS, participant_id="short"
    )
    with pytest.raises(ValueError, match="too short"):
        segment_trials(recording)

