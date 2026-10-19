from __future__ import annotations

import numpy as np
import pytest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

from emg_pipeline.data import CLASS_LABELS
from emg_pipeline.model import (
    build_classifier,
    evaluate_classifier,
    predict_with_voting,
)


def _separable_six_class_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1234)
    rows: list[np.ndarray] = []
    labels: list[str] = []
    for index, label in enumerate(CLASS_LABELS):
        centre = np.zeros(16)
        centre[index] = 8.0
        rows.append(rng.normal(loc=centre, scale=0.4, size=(25, 16)))
        labels.extend([label] * 25)
    return np.vstack(rows), np.asarray(labels)


def test_classifier_is_exactly_lda_then_gaussian_nb() -> None:
    classifier = build_classifier()

    assert list(classifier.named_steps) == ["lda", "gnb"]
    assert isinstance(classifier.named_steps["lda"], LinearDiscriminantAnalysis)
    assert classifier.named_steps["lda"].n_components == 5
    assert isinstance(classifier.named_steps["gnb"], GaussianNB)


def test_five_fold_evaluation_returns_accuracy_and_precision() -> None:
    X, y = _separable_six_class_data()

    result = evaluate_classifier(X, y)

    assert result.fold_accuracies.shape == (5,)
    assert result.mean_accuracy > 0.95
    assert result.std_accuracy >= 0
    assert result.fold_precisions.shape == (5, 6)
    assert set(result.per_class_precision) == set(CLASS_LABELS)
    assert set(result.per_class_precision_std) == set(CLASS_LABELS)


class _FixedClassifier:
    classes_ = np.asarray(["A", "B", "C"])

    def __init__(self) -> None:
        self._predictions = np.asarray(
            ["A", "A", "B", "B", "C", "C", "C", "C", "A", "B", "A", "B"]
        )
        self._probabilities = np.asarray(
            [
                [0.60, 0.35, 0.05],
                [0.60, 0.35, 0.05],
                [0.10, 0.85, 0.05],
                [0.10, 0.85, 0.05],
                [0.10, 0.20, 0.70],
                [0.05, 0.05, 0.90],
                [0.05, 0.05, 0.90],
                [0.05, 0.05, 0.90],
                [0.80, 0.10, 0.10],
                [0.10, 0.80, 0.10],
                [0.80, 0.10, 0.10],
                [0.10, 0.80, 0.10],
            ]
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._predictions[: X.shape[0]]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._probabilities[: X.shape[0]]


def test_voting_uses_probability_tie_break_and_discards_incomplete_group() -> None:
    result = predict_with_voting(_FixedClassifier(), np.zeros((12, 16)))

    # First group is tied 2-2-1 by count; B has the greater summed probability.
    assert result.voted_predictions.tolist() == ["B", "C"]
    np.testing.assert_allclose(result.agreement, [0.4, 0.6])
    assert result.discarded_windows == 2
    assert result.output_rate_hz == pytest.approx(6.25)


@pytest.mark.parametrize(
    ("X", "y", "message"),
    [
        (np.zeros((30, 15)), np.asarray(["x"] * 30), "windows, 16"),
        (np.zeros((30, 16)), np.asarray(["x"] * 29), "one label"),
        (np.zeros((30, 16)), np.asarray(["x"] * 30), "requires 6 classes"),
    ],
)
def test_evaluation_rejects_invalid_inputs(
    X: np.ndarray, y: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        evaluate_classifier(X, y)
