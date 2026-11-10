"""Accuracy metrics for comparing our decisions with the company's."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from .config import CLASS_LABELS


@dataclass(frozen=True)
class MetricsResult:
    """Agreement between our decisions and the company's.

    The company's decisions are treated as the reference ("actual"), so recall
    is the share of a movement they recorded that we also recorded, and
    precision is the share of our decisions for a movement that they agree with.
    """

    labels: tuple[str, ...]
    accuracy: float  # share of all decisions that match
    balanced_accuracy: float  # average of the per-movement recalls
    precision: dict[str, float]
    recall: dict[str, float]
    f1: dict[str, float]
    support: dict[str, int]  # number of company decisions for each movement
    confusion: np.ndarray  # rows = company's decision, columns = ours (counts)

    def table(self) -> pd.DataFrame:
        """Per-movement results as a table for display."""

        return pd.DataFrame(
            {
                "Precision": [self.precision[label] for label in self.labels],
                "Recall": [self.recall[label] for label in self.labels],
                "F1": [self.f1[label] for label in self.labels],
                "Decisions": [self.support[label] for label in self.labels],
            },
            index=pd.Index(self.labels, name="Movement"),
        )


def _check_sequences(actual: np.ndarray, predicted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    actual_array = np.asarray(actual, dtype=str)
    predicted_array = np.asarray(predicted, dtype=str)
    if actual_array.ndim != 1 or actual_array.shape != predicted_array.shape:
        raise ValueError("actual and predicted must be one-dimensional and equal in length.")
    if actual_array.size == 0:
        raise ValueError("There are no decisions to compare.")
    return actual_array, predicted_array


def compute_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    labels: tuple[str, ...] = CLASS_LABELS,
) -> MetricsResult:
    """Compare our predicted movements with the company's decisions."""

    actual_array, predicted_array = _check_sequences(actual, predicted)
    unknown = (set(actual_array) | set(predicted_array)) - set(labels)
    if unknown:
        raise ValueError(f"Unexpected movement names: {sorted(unknown)}.")

    precision, recall, f1, support = precision_recall_fscore_support(
        actual_array, predicted_array, labels=list(labels), zero_division=0
    )
    present = support > 0

    return MetricsResult(
        labels=tuple(labels),
        accuracy=float(np.mean(actual_array == predicted_array)),
        balanced_accuracy=float(np.mean(recall[present])),
        precision=dict(zip(labels, precision.astype(float))),
        recall=dict(zip(labels, recall.astype(float))),
        f1=dict(zip(labels, f1.astype(float))),
        support={label: int(count) for label, count in zip(labels, support)},
        confusion=confusion_matrix(actual_array, predicted_array, labels=list(labels)),
    )


def lag_sweep(
    actual: np.ndarray, predicted: np.ndarray, max_lag: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """Agreement when our decisions are shifted against the company's.

    A lag of k compares our decision i with the company's decision i + k, so a
    positive best lag means the company's decisions arrive later than ours.
    One step is one decision interval (about 64 ms). Returns (lags, agreement).
    """

    actual_array, predicted_array = _check_sequences(actual, predicted)
    if max_lag < 0 or max_lag >= actual_array.size:
        raise ValueError("max_lag must be between 0 and the number of decisions - 1.")

    lags = np.arange(-max_lag, max_lag + 1)
    agreement = np.empty(lags.size, dtype=np.float64)
    for index, lag in enumerate(lags):
        if lag >= 0:
            ours, theirs = predicted_array[: predicted_array.size - lag], actual_array[lag:]
        else:
            ours, theirs = predicted_array[-lag:], actual_array[: actual_array.size + lag]
        agreement[index] = np.mean(ours == theirs)
    return lags, agreement