"""Central settings for the EMG pipeline.

Change values here (or pass a modified PipelineConfig) instead of editing the
other modules.
"""

from __future__ import annotations

from dataclasses import dataclass

# The five movements, in the order used for tables and plots.
CLASS_LABELS = ("Rest", "Left", "Right", "Rotate", "Drop")

NUM_CHANNELS = 4

# Column order of the sensors. This was inferred from which channel is most
# active in each calibration file, so confirm it with the company.
CHANNEL_NAMES = ("Left forearm", "Left quad", "Right quad", "Right forearm")

# The numbers used in decision_times.csv. Code 4 is not used.
DECISION_CODES = {0: "Rest", 1: "Left", 2: "Right", 3: "Rotate", 5: "Drop"}

# Heights used by the decision timeline graph. The distance between levels is
# arbitrary, so it does not measure how wrong a mistake is.
PLOT_LEVELS = {"Drop": -2, "Left": -1, "Rest": 0, "Right": 1, "Rotate": 2}


def _milliseconds_to_samples(milliseconds: int, fs: int, name: str) -> int:
    samples = milliseconds * fs / 1_000
    rounded = int(round(samples))
    if abs(samples - rounded) > 1e-9:
        raise ValueError(
            f"{name} ({milliseconds} ms) is not a whole number of samples at {fs} Hz."
        )
    return rounded


@dataclass(frozen=True)
class PipelineConfig:
    """Settings that control filtering, windowing, and training."""

    fs: int = 1_000  # sampling rate in Hz
    window_ms: int = 64  # window length; try 128 to test overlapping windows
    train_hop_ms: int = 32  # spacing between training windows (more rows, same data)
    highpass_hz: float = 20.0
    lowpass_hz: float = 400.0
    filter_order: int = 2
    causal_filter: bool = True  # True = filter as a live system would
    calibration_trim_ms: int = 0  # cut this much off the start of each calibration file
    decision_lag: int = 5  # compare our decisions with the company's this many steps later (0 = no shift)

    def __post_init__(self) -> None:
        if self.fs <= 2 * self.lowpass_hz:
            raise ValueError(
                f"fs ({self.fs} Hz) must be more than twice the "
                f"low-pass cutoff ({self.lowpass_hz:g} Hz)."
            )
        if not 0 < self.highpass_hz < self.lowpass_hz:
            raise ValueError("Cutoffs must satisfy 0 < highpass_hz < lowpass_hz.")
        if self.filter_order < 1:
            raise ValueError("filter_order must be at least 1.")
        if self.window_samples <= 0 or self.train_hop_samples <= 0:
            raise ValueError("window_ms and train_hop_ms must be positive.")
        if self.calibration_trim_samples < 0:
            raise ValueError("calibration_trim_ms cannot be negative.")
        if self.decision_lag < 0:
            raise ValueError("decision_lag cannot be negative.")

    @property
    def window_samples(self) -> int:
        return _milliseconds_to_samples(self.window_ms, self.fs, "window_ms")

    @property
    def train_hop_samples(self) -> int:
        return _milliseconds_to_samples(self.train_hop_ms, self.fs, "train_hop_ms")

    @property
    def calibration_trim_samples(self) -> int:
        return _milliseconds_to_samples(
            self.calibration_trim_ms, self.fs, "calibration_trim_ms"
        )