from __future__ import annotations

import numpy as np
import pytest

from emg_pipeline.data import CLASS_LABELS, EXPECTED_FS, Trial
from emg_pipeline.features import (
    compute_window_features,
    extract_features,
    filter_recording,
    window_signal,
)


def test_filter_keeps_passband_and_attenuates_out_of_band_frequencies() -> None:
    duration = 4.0
    time = np.arange(int(duration * EXPECTED_FS)) / EXPECTED_FS
    signal = np.column_stack(
        (
            np.sin(2 * np.pi * 10 * time),
            np.sin(2 * np.pi * 100 * time),
            np.sin(2 * np.pi * 600 * time),
            np.zeros_like(time),
        )
    )

    filtered = filter_recording(signal, EXPECTED_FS)
    middle = slice(EXPECTED_FS, -EXPECTED_FS)
    rms = np.sqrt(np.mean(filtered[middle] ** 2, axis=0))

    assert rms[0] < 0.05
    assert rms[1] > 0.60
    assert rms[2] < 0.05
    assert rms[3] == pytest.approx(0.0, abs=1e-12)


def test_filter_rejects_incompatible_sampling_rate() -> None:
    with pytest.raises(ValueError, match="greater than 800"):
        filter_recording(np.zeros((1_000, 4)), 800)


def test_feature_formulas_have_channel_major_order() -> None:
    values = np.array([1.0, -1.0, 1.0, -1.0])
    window = np.column_stack((values, values, values, values))

    features = compute_window_features(window)

    expected_channel = np.array([1.0, 6.0, 3.0, 2.0])
    np.testing.assert_allclose(features, np.tile(expected_channel, 4))


def test_window_count_for_six_second_trial() -> None:
    windows, starts = window_signal(np.zeros((6 * EXPECTED_FS, 4)), EXPECTED_FS)

    assert windows.shape == (186, 128, 4)
    assert starts.shape == (186,)
    assert starts[0] == 0
    assert starts[-1] == 11_840


def test_full_trial_set_produces_expected_balanced_feature_matrix() -> None:
    shared_signal = np.zeros((6 * EXPECTED_FS, 4))
    trials = [
        Trial(
            signal=shared_signal,
            label=label,
            control="unused",
            gesture_id=gesture_id,
            repetition=repetition,
            start_sample=(repetition * 100 + gesture_id * 10) * EXPECTED_FS,
        )
        for repetition in range(1, 6)
        for gesture_id, label in enumerate(CLASS_LABELS)
    ]

    features = extract_features(trials, EXPECTED_FS)

    assert features.X.shape == (5_580, 16)
    assert np.isfinite(features.X).all()
    assert features.y.shape == (5_580,)
    labels, counts = np.unique(features.y, return_counts=True)
    assert set(labels) == set(CLASS_LABELS)
    assert set(counts) == {930}
