"""Tests for filtering, windowing, and feature extraction."""

from __future__ import annotations

import numpy as np
import pytest

from emg_pipeline import (
    PipelineConfig,
    compute_features,
    compute_window_features,
    filter_signal,
    window_signal,
    windows_ending_at,
)
from emg_pipeline import features as features_module


def ramp_signal(samples: int = 400) -> np.ndarray:
    """A signal whose value at (row, channel) identifies the row."""

    return np.repeat(np.arange(samples, dtype=float)[:, None], 4, axis=1)


# ---- filter_signal -----------------------------------------------------------


def test_filter_removes_constant_offset_and_keeps_the_passband() -> None:
    seconds = np.arange(5_000) / 1_000
    tone = 100 * np.sin(2 * np.pi * 100 * seconds)
    raw = np.column_stack([3_000 + tone] * 4)

    filtered = filter_signal(raw, PipelineConfig())

    settled = filtered[1_500:]
    assert np.all(np.abs(settled.mean(axis=0)) < 1.0)
    # A 100 Hz sine of amplitude 100 has an RMS of about 70.7.
    assert np.allclose(settled.std(axis=0), 70.7, rtol=0.05)


def test_filter_has_no_start_up_spike_on_a_large_offset() -> None:
    raw = np.full((2_000, 4), 5_000.0)
    filtered = filter_signal(raw, PipelineConfig())
    assert np.abs(filtered).max() < 1e-6


def test_filter_strongly_reduces_slow_drift() -> None:
    seconds = np.arange(5_000) / 1_000
    drift = np.column_stack([200 * np.sin(2 * np.pi * 2 * seconds)] * 4)
    filtered = filter_signal(drift, PipelineConfig())
    assert filtered[1_000:].std() < 0.05 * drift.std()


def test_filter_zero_phase_option_keeps_the_shape() -> None:
    raw = np.random.default_rng(0).normal(size=(1_000, 4))
    filtered = filter_signal(raw, PipelineConfig(causal_filter=False))
    assert filtered.shape == raw.shape


def test_filter_rejects_wrong_shape_and_nan() -> None:
    with pytest.raises(ValueError, match="shape"):
        filter_signal(np.zeros((100, 3)), PipelineConfig())
    bad = np.zeros((100, 4))
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        filter_signal(bad, PipelineConfig())


# ---- window_signal -----------------------------------------------------------


def test_window_signal_count_and_positions() -> None:
    signal = ramp_signal(400)
    windows = window_signal(signal, window_samples=64, hop_samples=32)

    assert windows.shape == ((400 - 64) // 32 + 1, 64, 4)
    assert np.array_equal(windows[0], signal[0:64])
    assert np.array_equal(windows[1], signal[32:96])
    assert np.array_equal(windows[-1], signal[320:384])


def test_window_signal_without_overlap() -> None:
    windows = window_signal(ramp_signal(256), window_samples=64, hop_samples=64)
    assert windows.shape == (4, 64, 4)


def test_window_signal_rejects_short_signals_and_bad_sizes() -> None:
    with pytest.raises(ValueError, match="fewer than"):
        window_signal(ramp_signal(50), window_samples=64, hop_samples=32)
    with pytest.raises(ValueError, match="positive"):
        window_signal(ramp_signal(200), window_samples=64, hop_samples=0)


# ---- windows_ending_at -------------------------------------------------------


def test_windows_end_exactly_at_the_requested_samples() -> None:
    signal = ramp_signal(400)
    windows, valid = windows_ending_at(signal, np.array([63, 64, 100, 399]), 64)

    assert valid.all()
    assert windows.shape == (4, 64, 4)
    assert np.array_equal(windows[0], signal[0:64])  # ends at sample 63
    assert np.array_equal(windows[1], signal[1:65])  # ends at sample 64
    assert windows[2][-1, 0] == 100
    assert windows[3][-1, 0] == 399


def test_windows_ending_at_flags_windows_that_do_not_fit() -> None:
    signal = ramp_signal(400)
    windows, valid = windows_ending_at(signal, np.array([10, 63, 399, 400]), 64)

    assert valid.tolist() == [False, True, True, False]
    assert windows.shape == (2, 64, 4)  # only the valid ones are returned


def test_longer_windows_skip_the_first_decision() -> None:
    signal = ramp_signal(400)
    ends = np.array([63, 127, 191])
    _, valid_64 = windows_ending_at(signal, ends, 64)
    _, valid_128 = windows_ending_at(signal, ends, 128)
    assert valid_64.all()
    assert valid_128.tolist() == [False, True, True]


# ---- features ----------------------------------------------------------------


def test_features_match_hand_calculated_values() -> None:
    window = np.zeros((8, 4))
    window[:, 0] = [1, -1, 1, -1, 1, -1, 1, -1]  # alternating
    window[:, 1] = 3.0  # constant
    window[:, 2] = [0, 1, 2, 3, 4, 5, 6, 7]  # straight ramp
    window[:, 3] = [1, 2, 1, 2, 1, 2, 1, 2]  # zigzag above zero

    features = compute_window_features(window)

    assert features.shape == (16,)
    channel_0, channel_1, channel_2, channel_3 = features.reshape(4, 4)
    # [RMS, waveform length, zero crossings, slope sign changes]
    assert np.allclose(channel_0, [1.0, 14.0, 7.0, 6.0])
    assert np.allclose(channel_1, [3.0, 0.0, 0.0, 0.0])
    assert np.allclose(channel_2[1:], [7.0, 0.0, 0.0])  # ramp: no crossings or turns
    assert np.allclose(channel_3[1:], [7.0, 0.0, 6.0])  # zigzag: 7 steps, 6 turns


def test_batch_features_match_single_window_features() -> None:
    windows = np.random.default_rng(1).normal(size=(30, 64, 4))
    batch = compute_features(windows)

    assert batch.shape == (30, 16)
    for index in (0, 7, 29):
        assert np.allclose(batch[index], compute_window_features(windows[index]))


def test_features_are_the_same_when_processed_in_small_batches(monkeypatch) -> None:
    windows = np.random.default_rng(2).normal(size=(25, 32, 4))
    expected = compute_features(windows)
    monkeypatch.setattr(features_module, "_BATCH_SIZE", 4)
    assert np.allclose(compute_features(windows), expected)


def test_features_reject_bad_input() -> None:
    with pytest.raises(ValueError, match="shape"):
        compute_features(np.zeros((5, 64, 3)))
    with pytest.raises(ValueError, match="at least 3"):
        compute_features(np.zeros((5, 2, 4)))
    bad = np.zeros((2, 8, 4))
    bad[0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="NaN"):
        compute_features(bad)