"""Functions used by the EMG classification notebook."""

from .data import (
    CLASS_LABELS,
    EXPECTED_CHANNELS,
    EXPECTED_FS,
    Recording,
    Trial,
    load_recording,
    segment_trials,
)
from .features import (
    FeatureSet,
    compute_window_features,
    extract_features,
    filter_recording,
    window_signal,
)
from .model import (
    EvaluationResult,
    VotingResult,
    build_classifier,
    evaluate_classifier,
    predict_with_voting,
)

