"""LDA, Gaussian Naive Bayes, cross-validation, and majority voting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

from .data import CLASS_LABELS


@dataclass(frozen=True)
class EvaluationResult:
    fold_accuracies: np.ndarray
    fold_precisions: np.ndarray
    mean_accuracy: float
    std_accuracy: float
    class_labels: tuple[str, ...]
    per_class_precision: dict[str, float]
    per_class_precision_std: dict[str, float]


@dataclass(frozen=True)
class VotingResult:
    voted_predictions: np.ndarray
    agreement: np.ndarray
    output_rate_hz: float
    discarded_windows: int


def build_classifier() -> Pipeline:
    """Build the paper's five-dimensional LDA followed by GaussianNB."""

    return Pipeline(
        steps=(
            (
                "lda",
                LinearDiscriminantAnalysis(n_components=5, solver="svd"),
            ),
            ("gnb", GaussianNB()),
        )
    )


def _validate_training_data(
    features: np.ndarray, labels: np.ndarray, n_splits: int
) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=str)
    if X.ndim != 2 or X.shape[1] != 16:
        raise ValueError(f"Features must have shape (windows, 16); received {X.shape}.")
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError("Labels must be one-dimensional with one label per feature row.")
    if not np.isfinite(X).all():
        raise ValueError("Feature matrix contains NaN or infinite values.")

    classes, counts = np.unique(y, return_counts=True)
    if classes.size != 6:
        raise ValueError(
            f"This classifier expects 6 movement classes; received {classes.size}."
        )
    if np.any(counts < n_splits):
        raise ValueError(
            f"Every class needs at least {n_splits} windows for {n_splits}-fold validation."
        )
    return X, y


def evaluate_classifier(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    n_splits: int = 5,
    random_state: int = 42,
) -> EvaluationResult:
    """Run seeded, shuffled, stratified cross-validation over feature windows."""

    X, y = _validate_training_data(features, labels, n_splits)
    observed = set(np.unique(y))
    if observed == set(CLASS_LABELS):
        class_labels = CLASS_LABELS
    else:
        class_labels = tuple(sorted(observed))

    splitter = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    fold_accuracies: list[float] = []
    fold_precisions: list[np.ndarray] = []

    for train_indices, test_indices in splitter.split(X, y):
        model = build_classifier()
        model.fit(X[train_indices], y[train_indices])
        predicted = model.predict(X[test_indices])

        fold_accuracies.append(accuracy_score(y[test_indices], predicted))
        fold_precisions.append(
            precision_score(
                y[test_indices],
                predicted,
                labels=class_labels,
                average=None,
                zero_division=0,
            )
        )

    fold_array = np.asarray(fold_accuracies, dtype=np.float64)
    precision_array = np.asarray(fold_precisions, dtype=np.float64)
    precision_means = precision_array.mean(axis=0)
    precision_stds = precision_array.std(axis=0)

    return EvaluationResult(
        fold_accuracies=fold_array,
        fold_precisions=precision_array,
        mean_accuracy=float(np.mean(fold_array)),
        std_accuracy=float(np.std(fold_array)),
        class_labels=class_labels,
        per_class_precision={
            label: float(value)
            for label, value in zip(class_labels, precision_means)
        },
        per_class_precision_std={
            label: float(value)
            for label, value in zip(class_labels, precision_stds)
        },
    )


def predict_with_voting(
    model: object,
    features: np.ndarray,
    *,
    vote_size: int = 5,
    feature_rate_hz: float = 31.25,
) -> VotingResult:
    """Reduce consecutive model decisions to one probability-tie-broken vote."""

    X = np.asarray(features, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != 16:
        raise ValueError(f"Features must have shape (windows, 16); received {X.shape}.")
    if not np.isfinite(X).all():
        raise ValueError("Feature matrix contains NaN or infinite values.")
    if vote_size <= 0:
        raise ValueError("vote_size must be positive.")
    if feature_rate_hz <= 0:
        raise ValueError("feature_rate_hz must be positive.")

    predictions = np.asarray(model.predict(X), dtype=str)
    probabilities = np.asarray(model.predict_proba(X), dtype=np.float64)
    classes = np.asarray(model.classes_, dtype=str)
    if probabilities.shape != (X.shape[0], classes.size):
        raise ValueError("Classifier returned an unexpected probability matrix shape.")

    full_groups = X.shape[0] // vote_size
    voted: list[str] = []
    agreements: list[float] = []

    for group_index in range(full_groups):
        start = group_index * vote_size
        stop = start + vote_size
        group_predictions = predictions[start:stop]
        labels, counts = np.unique(group_predictions, return_counts=True)
        maximum_count = int(np.max(counts))
        candidates = labels[counts == maximum_count]

        if candidates.size == 1:
            winner = str(candidates[0])
        else:
            probability_sums = probabilities[start:stop].sum(axis=0)
            candidate_columns = [
                column
                for column, class_label in enumerate(classes)
                if class_label in candidates
            ]
            # np.argmax is deterministic and preserves model.classes_ order on an exact tie.
            winning_column = candidate_columns[
                int(np.argmax(probability_sums[candidate_columns]))
            ]
            winner = str(classes[winning_column])

        voted.append(winner)
        agreements.append(maximum_count / vote_size)

    return VotingResult(
        voted_predictions=np.asarray(voted, dtype=str),
        agreement=np.asarray(agreements, dtype=np.float64),
        output_rate_hz=feature_rate_hz / vote_size,
        discarded_windows=X.shape[0] - full_groups * vote_size,
    )
