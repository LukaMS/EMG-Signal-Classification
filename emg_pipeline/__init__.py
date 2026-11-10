"""Functions used by the EMG classification notebook."""

from .config import CLASS_LABELS, DECISION_CODES, PLOT_LEVELS, PipelineConfig
from .data import Session, load_session
from .evaluation import MetricsResult, compute_metrics, lag_sweep
from .features import (
    compute_features,
    compute_window_features,
    filter_signal,
    window_signal,
    windows_ending_at,
)
from .model import VotingResult, build_classifier, predict_with_voting
from .pipeline import LiveResult, TrainingSet, build_training_set, classify_live
from .plots import (
    plot_calibration_signals,
    plot_confusion_matrix,
    plot_decision_timelines,
    plot_lda_scatter,
    plot_decision_windows,
)