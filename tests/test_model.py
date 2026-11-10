"""Tests for the classifier, the train/classify pipeline, metrics, and plots."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # draw without opening windows

import matplotlib.pyplot as plt
import numpy as np
import pytest

from emg_pipeline import (
    CLASS_LABELS,
    PLOT_LEVELS,
    PipelineConfig,
    Session,
    build_classifier,
    build_training_set,
    classify_live,
    compute_metrics,
    lag_sweep,
    load_session,
    plot_calibration_signals,
    plot_confusion_matrix,
    plot_decision_timelines,
    plot_lda_scatter,
    predict_with_voting,
    filter_signal,
    plot_decision_windows,
)
from emg_pipeline.data import DATA_DIR, LIVE_FILENAME

LOUD, QUIET = 100.0, 10.0


def make_session(seed: int = 0, live_seconds: int = 30) -> Session:
    """A small synthetic session.

    Movement k (Left, Right, Rotate, Drop) is loud on channel k - 1; Rest is
    quiet everywhere. Calibration data is zero-mean (already filtered); the live
    data is raw, with a large constant offset added.
    """

    rng = np.random.default_rng(seed)

    def signal_for(label: str, samples: int) -> np.ndarray:
        amplitudes = np.full(4, QUIET)
        if label != "Rest":
            amplitudes[CLASS_LABELS.index(label) - 1] = LOUD
        return rng.normal(size=(samples, 4)) * amplitudes

    calibration = {label: signal_for(label, 4_000) for label in CLASS_LABELS}

    block_labels = [CLASS_LABELS[i] for i in rng.integers(0, 5, size=live_seconds)]
    live = np.vstack([signal_for(label, 1_000) for label in block_labels]) + 2_000.0

    times = np.round(np.arange(1, live.shape[0] // 64 + 1) * 0.064 - 0.001, 3)
    end_samples = np.rint(times * 1_000).astype(int)
    decisions = np.array([block_labels[end // 1_000] for end in end_samples])

    return Session(
        calibration=calibration,
        live=live,
        decision_times_s=times,
        decisions=decisions,
        source=DATA_DIR,
    )


# ---- classifier --------------------------------------------------------------


def test_classifier_uses_one_fewer_lda_dimension_than_classes() -> None:
    assert build_classifier().named_steps["lda"].n_components == 4
    assert build_classifier(n_classes=3).named_steps["lda"].n_components == 2


# ---- build_training_set ------------------------------------------------------


def test_training_set_has_equal_windows_per_movement() -> None:
    training = build_training_set(make_session(), PipelineConfig())

    expected_per_class = (4_000 - 64) // 32 + 1
    assert training.X.shape == (5 * expected_per_class, 16)
    for label in CLASS_LABELS:
        assert np.sum(training.y == label) == expected_per_class


def test_trimming_removes_windows_from_each_calibration_file() -> None:
    cfg = PipelineConfig(calibration_trim_ms=1_000)
    training = build_training_set(make_session(), cfg)
    assert np.sum(training.y == "Rest") == (3_000 - 64) // 32 + 1


def test_training_set_rejects_a_trim_that_leaves_too_little_data() -> None:
    cfg = PipelineConfig(calibration_trim_ms=3_990)
    with pytest.raises(ValueError, match="fewer than the window length"):
        build_training_set(make_session(), cfg)


# ---- classify_live -----------------------------------------------------------


def test_synthetic_session_is_classified_accurately() -> None:
    session = make_session()
    cfg = PipelineConfig(decision_lag=0)
    training = build_training_set(session, cfg)
    model = build_classifier().fit(training.X, training.y)

    live = classify_live(session, model, cfg)

    assert live.skipped == 0
    assert len(live.predicted) == len(live.actual) == len(live.times_s) == len(session.decisions)
    assert np.mean(live.predicted == live.actual) > 0.9


def test_longer_windows_skip_decisions_without_a_full_window() -> None:
    session = make_session()
    cfg = PipelineConfig(window_ms=128, decision_lag=0)
    training = build_training_set(session, cfg)
    model = build_classifier().fit(training.X, training.y)

    live = classify_live(session, model, cfg)

    assert live.skipped == 1
    assert len(live.predicted) == len(session.decisions) - 1
    assert live.times_s[0] == session.decision_times_s[1]


def test_decision_lag_pairs_each_decision_with_a_later_company_decision() -> None:
    session = make_session()
    training = build_training_set(session, PipelineConfig())
    model = build_classifier().fit(training.X, training.y)

    unshifted = classify_live(session, model, PipelineConfig(decision_lag=0))
    shifted = classify_live(session, model, PipelineConfig(decision_lag=3))

    assert np.array_equal(shifted.predicted, unshifted.predicted[:-3])
    assert np.array_equal(shifted.actual, unshifted.actual[3:])
    assert np.array_equal(shifted.times_s, unshifted.times_s[3:])
    assert np.array_equal(shifted.window_end_s, unshifted.window_end_s[:-3])
    assert np.array_equal(unshifted.window_end_s, unshifted.times_s)


def test_decision_lag_larger_than_the_session_is_rejected() -> None:
    session = make_session()
    training = build_training_set(session, PipelineConfig())
    model = build_classifier().fit(training.X, training.y)

    with pytest.raises(ValueError, match="decision_lag"):
        classify_live(session, model, PipelineConfig(decision_lag=100_000))


# ---- compute_metrics ---------------------------------------------------------


def test_metrics_on_a_hand_checked_example() -> None:
    actual = np.array(["Rest", "Left", "Left", "Right"])
    predicted = np.array(["Rest", "Left", "Rest", "Right"])

    metrics = compute_metrics(actual, predicted)

    assert metrics.accuracy == 0.75
    assert metrics.recall["Left"] == 0.5
    assert metrics.precision["Rest"] == 0.5
    assert metrics.support == {"Rest": 1, "Left": 2, "Right": 1, "Rotate": 0, "Drop": 0}
    # Balanced accuracy averages only movements that occur: (1 + 0.5 + 1) / 3.
    assert metrics.balanced_accuracy == pytest.approx(5 / 6)
    assert metrics.confusion.sum() == 4
    assert metrics.confusion[CLASS_LABELS.index("Left"), CLASS_LABELS.index("Rest")] == 1


def test_metrics_table_lists_every_movement_in_order() -> None:
    labels = np.array(CLASS_LABELS)
    table = compute_metrics(labels, labels).table()

    assert list(table.index) == list(CLASS_LABELS)
    assert list(table.columns) == ["Precision", "Recall", "F1", "Decisions"]
    assert (table["F1"] == 1.0).all()


def test_metrics_reject_unknown_names_and_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="Unexpected"):
        compute_metrics(np.array(["Rest"]), np.array(["Jump"]))
    with pytest.raises(ValueError, match="equal in length"):
        compute_metrics(np.array(["Rest", "Left"]), np.array(["Rest"]))
    with pytest.raises(ValueError, match="no decisions"):
        compute_metrics(np.array([], dtype=str), np.array([], dtype=str))


# ---- lag_sweep ---------------------------------------------------------------


def test_lag_sweep_finds_a_known_delay() -> None:
    rng = np.random.default_rng(3)
    predicted = np.array(CLASS_LABELS)[rng.integers(0, 5, size=300)]
    # The company's decisions repeat ours three decisions later.
    actual = np.concatenate([np.array(CLASS_LABELS)[rng.integers(0, 5, size=3)], predicted[:-3]])

    lags, agreement = lag_sweep(actual, predicted, max_lag=6)

    assert lags[np.argmax(agreement)] == 3
    assert agreement.max() == 1.0


def test_lag_sweep_rejects_an_oversized_shift() -> None:
    labels = np.array(["Rest", "Left", "Rest"])
    with pytest.raises(ValueError, match="max_lag"):
        lag_sweep(labels, labels, max_lag=3)


# ---- predict_with_voting -----------------------------------------------------


class FakeModel:
    """Returns fixed predictions and probabilities."""

    classes_ = np.array(["Left", "Rest"])

    def __init__(self, predictions: list[str], probabilities: np.ndarray) -> None:
        self._predictions = np.array(predictions)
        self._probabilities = probabilities

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self._predictions

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return self._probabilities


def test_voting_takes_the_majority_and_drops_leftovers() -> None:
    predictions = ["Rest", "Rest", "Left", "Rest", "Left", "Left", "Left", "Rest", "Left", "Left", "Rest"]
    model = FakeModel(predictions, np.full((11, 2), 0.5))

    result = predict_with_voting(model, np.zeros((11, 16)))

    assert result.voted_predictions.tolist() == ["Rest", "Left"]
    assert result.agreement.tolist() == [0.6, 0.8]
    assert result.discarded_windows == 1
    assert result.output_rate_hz == pytest.approx(15.625 / 5)


def test_voting_breaks_ties_with_probabilities() -> None:
    # Two votes each for Left and Rest (vote_size=4), so the probabilities decide.
    predictions = ["Left", "Left", "Rest", "Rest"]
    probabilities = np.array([[0.4, 0.6], [0.4, 0.6], [0.2, 0.8], [0.2, 0.8]])
    model = FakeModel(predictions, probabilities)

    result = predict_with_voting(model, np.zeros((4, 16)), vote_size=4)

    assert result.voted_predictions.tolist() == ["Rest"]


# ---- plots -------------------------------------------------------------------


@pytest.fixture(scope="module")
def trained():
    session = make_session()
    cfg = PipelineConfig()
    training = build_training_set(session, cfg)
    model = build_classifier().fit(training.X, training.y)
    live = classify_live(session, model, cfg)
    return session, cfg, training, model, live


def test_plots_return_figures_with_the_expected_panels(trained) -> None:
    session, cfg, training, model, live = trained
    metrics = compute_metrics(live.actual, live.predicted)

    figures = {
        "calibration": plot_calibration_signals(session, cfg),
        "scatter": plot_lda_scatter(model, training.X, training.y),
        "confusion": plot_confusion_matrix(metrics),
        "timeline": plot_decision_timelines(live.times_s, live.actual, live.predicted),
        "zoom": plot_decision_timelines(
            live.times_s, live.actual, live.predicted, start_s=5, duration_s=10
        ),
        "windows": plot_decision_windows(
            live, filter_signal(session.live, cfg), cfg, start_s=5, duration_s=2
        ),
    }

    assert len(figures["calibration"].axes) == 20  # 4 channels x 5 movements
    assert len(figures["timeline"].axes) == 2
    assert len(figures["zoom"].axes) == 2
    assert len(figures["windows"].axes) == 6  # company, ours, 4 channels
    for figure in figures.values():
        plt.close(figure)


def test_timeline_axis_labels_follow_plot_levels(trained) -> None:
    _, _, _, _, live = trained
    figure = plot_decision_timelines(live.times_s, live.actual, live.predicted)

    labels = [tick.get_text() for tick in figure.axes[0].get_yticklabels()]
    assert labels == sorted(PLOT_LEVELS, key=PLOT_LEVELS.get)
    plt.close(figure)


def test_timeline_rejects_an_empty_time_range(trained) -> None:
    _, _, _, _, live = trained
    with pytest.raises(ValueError, match="No decisions"):
        plot_decision_timelines(
            live.times_s, live.actual, live.predicted, start_s=10_000, duration_s=5
        )

def test_decision_windows_rejects_bad_ranges(trained) -> None:
      session, cfg, _, _, live = trained
      filtered = filter_signal(session.live, cfg)
      with pytest.raises(ValueError, match="No decisions"):
          plot_decision_windows(live, filtered, cfg, start_s=10_000, duration_s=2)
      with pytest.raises(ValueError, match="max_windows"):
          plot_decision_windows(live, filtered, cfg, start_s=0, duration_s=25, max_windows=10)


# ---- the real data (skipped if the CSV files are not present) -----------------


@pytest.mark.skipif(not (DATA_DIR / LIVE_FILENAME).exists(), reason="data files not present")
def test_real_session_runs_end_to_end() -> None:
    session = load_session()
    cfg = PipelineConfig()
    training = build_training_set(session, cfg)
    model = build_classifier().fit(training.X, training.y)

    live = classify_live(session, model, cfg)
    metrics = compute_metrics(live.actual, live.predicted)

    assert len(live.predicted) + live.skipped + cfg.decision_lag == len(session.decisions)
    assert 0.0 <= metrics.accuracy <= 1.0
    assert metrics.confusion.sum() == len(live.predicted)