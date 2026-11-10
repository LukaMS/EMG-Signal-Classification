"""LDA, Gaussian Naive Bayes, and optional majority voting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

from .config import CLASS_LABELS


@dataclass(frozen=True)
class VotingResult:
    voted_predictions: np.ndarray
    agreement: np.ndarray
    output_rate_hz: float
    discarded_windows: int


def build_classifier(n_classes: int = len(CLASS_LABELS)) -> Pipeline:
    """Build LDA (n_classes - 1 dimensions) followed by GaussianNB."""

    return Pipeline(
        steps=(
            (
                "lda",
                LinearDiscriminantAnalysis(n_components=n_classes - 1, solver="svd"),
            ),
            ("gnb", GaussianNB()),
        )
    )


def predict_with_voting(
    model: object,
    features: np.ndarray,
    *,
    vote_size: int = 5,
    feature_rate_hz: float = 15.625,
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
