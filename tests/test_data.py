"""Tests for the settings and for loading the CSV files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from emg_pipeline import CLASS_LABELS, PipelineConfig, load_session
from emg_pipeline.data import DECISIONS_FILENAME, LIVE_FILENAME, calibration_filename


def write_session_files(folder: Path, *, decisions_text: str | None = None) -> None:
    """Write a tiny, valid set of the 7 CSV files into a folder."""

    rng = np.random.default_rng(0)
    for label in CLASS_LABELS:
        np.savetxt(folder / calibration_filename(label), rng.normal(size=(200, 4)), delimiter=",")
    np.savetxt(folder / LIVE_FILENAME, rng.normal(size=(1_000, 4)), delimiter=",")
    if decisions_text is None:
        decisions_text = (
            "Time_s,Decision,Speed_index\n"
            "0.063,0,1\n"
            "0.127,1,1\n"
            "0.191,2,1\n"
            "0.255,3,1\n"
            "0.319,5,1\n"
        )
    (folder / DECISIONS_FILENAME).write_text(decisions_text)


# ---- PipelineConfig ----------------------------------------------------------


def test_config_defaults_give_64_sample_windows_at_1_khz() -> None:
    cfg = PipelineConfig()
    assert cfg.fs == 1_000
    assert cfg.window_samples == 64
    assert cfg.train_hop_samples == 32
    assert cfg.calibration_trim_samples == 0
    assert cfg.decision_lag == 5


def test_config_window_length_is_one_setting() -> None:
    assert PipelineConfig(window_ms=128).window_samples == 128


@pytest.mark.parametrize(
    "settings",
    [
        {"fs": 700},  # too low for a 400 Hz cutoff
        {"highpass_hz": 500.0},  # high-pass above low-pass
        {"window_ms": 0},
        {"filter_order": 0},
        {"calibration_trim_ms": -5},
        {"decision_lag": -1},
        {"fs": 1_500, "window_ms": 1},  # 1.5 samples is not a whole number
    ],
)
def test_config_rejects_bad_settings(settings: dict) -> None:
    with pytest.raises(ValueError):
        PipelineConfig(**settings)


# ---- load_session ------------------------------------------------------------


def test_load_session_reads_all_files(tmp_path: Path) -> None:
    write_session_files(tmp_path)
    session = load_session(tmp_path)

    assert set(session.calibration) == set(CLASS_LABELS)
    assert all(signal.shape == (200, 4) for signal in session.calibration.values())
    assert session.live.shape == (1_000, 4)
    assert session.decision_times_s.tolist() == [0.063, 0.127, 0.191, 0.255, 0.319]
    # Codes 0, 1, 2, 3, 5 become names; the speed column is ignored.
    assert session.decisions.tolist() == ["Rest", "Left", "Right", "Rotate", "Drop"]
    assert session.source == tmp_path


def test_load_session_lists_every_missing_file(tmp_path: Path) -> None:
    write_session_files(tmp_path)
    (tmp_path / calibration_filename("Left")).unlink()
    (tmp_path / LIVE_FILENAME).unlink()

    with pytest.raises(FileNotFoundError) as error:
        load_session(tmp_path)
    assert calibration_filename("Left") in str(error.value)
    assert LIVE_FILENAME in str(error.value)


def test_load_session_rejects_wrong_column_count(tmp_path: Path) -> None:
    write_session_files(tmp_path)
    np.savetxt(tmp_path / LIVE_FILENAME, np.ones((50, 3)), delimiter=",")
    with pytest.raises(ValueError, match="4 columns"):
        load_session(tmp_path)


def test_load_session_rejects_nan_in_a_signal(tmp_path: Path) -> None:
    write_session_files(tmp_path)
    bad = np.ones((50, 4))
    bad[3, 2] = np.nan
    np.savetxt(tmp_path / calibration_filename("Rest"), bad, delimiter=",")
    with pytest.raises(ValueError, match="NaN or infinite"):
        load_session(tmp_path)


def test_load_session_rejects_unknown_decision_code(tmp_path: Path) -> None:
    write_session_files(
        tmp_path, decisions_text="Time_s,Decision,Speed_index\n0.063,0,1\n0.127,4,1\n"
    )
    with pytest.raises(ValueError, match="decision codes"):
        load_session(tmp_path)


def test_load_session_rejects_timestamps_that_do_not_increase(tmp_path: Path) -> None:
    write_session_files(
        tmp_path, decisions_text="Time_s,Decision,Speed_index\n0.127,0,1\n0.063,1,1\n"
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        load_session(tmp_path)


def test_load_session_requires_decision_columns(tmp_path: Path) -> None:
    write_session_files(tmp_path, decisions_text="Time,Choice\n0.063,0\n")
    with pytest.raises(ValueError, match="Time_s"):
        load_session(tmp_path)