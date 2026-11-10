"""Loading the calibration, live, and decision CSV files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CLASS_LABELS, DECISION_CODES, NUM_CHANNELS

# The repository's data folder, found relative to this file so it works no
# matter where the notebook is launched from.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

LIVE_FILENAME = "live_samples.csv"
DECISIONS_FILENAME = "decision_times.csv"


def calibration_filename(label: str) -> str:
    return f"calibration_data_{label}.csv"


@dataclass(frozen=True)
class Session:
    """Everything recorded in one session."""

    calibration: dict[str, np.ndarray]  # movement -> (samples, 4), already filtered
    live: np.ndarray  # (samples, 4), raw and unfiltered
    decision_times_s: np.ndarray  # (decisions,) time of each decision in seconds
    decisions: np.ndarray  # (decisions,) movement name for each decision
    source: Path  # folder the files were loaded from


def _read_signal(path: Path) -> np.ndarray:
    """Read a CSV of four sensor columns with no header."""

    try:
        values = pd.read_csv(path, header=None, dtype=np.float64).to_numpy()
    except ValueError as error:
        raise ValueError(
            f"{path.name} could not be read as four numeric columns without "
            f"a header: {error}"
        ) from error
    if values.ndim != 2 or values.shape[1] != NUM_CHANNELS:
        raise ValueError(
            f"{path.name} must have {NUM_CHANNELS} columns; found shape {values.shape}."
        )
    if values.shape[0] == 0:
        raise ValueError(f"{path.name} is empty.")
    if not np.isfinite(values).all():
        raise ValueError(f"{path.name} contains NaN or infinite values.")
    return values


def _read_decisions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read decision_times.csv and convert the numeric codes to movement names."""

    table = pd.read_csv(path)
    for column in ("Time_s", "Decision"):
        if column not in table.columns:
            raise ValueError(
                f"{path.name} must have a '{column}' column; "
                f"found {list(table.columns)}."
            )

    times = table["Time_s"].to_numpy(dtype=np.float64)
    codes = table["Decision"].to_numpy(dtype=np.float64)
    if times.size == 0:
        raise ValueError(f"{path.name} is empty.")
    if not np.isfinite(times).all() or not np.isfinite(codes).all():
        raise ValueError(f"{path.name} contains NaN or infinite values.")
    if np.any(np.diff(times) <= 0):
        raise ValueError(f"Timestamps in {path.name} must be strictly increasing.")

    int_codes = codes.astype(int)
    unknown = sorted(set(int_codes.tolist()) - set(DECISION_CODES))
    if unknown or not np.array_equal(int_codes, codes):
        raise ValueError(
            f"{path.name} contains decision codes {unknown or 'that are not whole numbers'}; "
            f"expected only {sorted(DECISION_CODES)}."
        )

    labels = np.array([DECISION_CODES[code] for code in int_codes], dtype=str)
    return times, labels


def load_session(data_dir: str | Path = DATA_DIR) -> Session:
    """Load the five calibration files, the live recording, and the decisions."""

    folder = Path(data_dir)
    paths = {label: folder / calibration_filename(label) for label in CLASS_LABELS}
    live_path = folder / LIVE_FILENAME
    decisions_path = folder / DECISIONS_FILENAME

    expected = [*paths.values(), live_path, decisions_path]
    missing = [path.name for path in expected if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing file(s) in '{folder}': {', '.join(missing)}."
        )

    calibration = {label: _read_signal(path) for label, path in paths.items()}
    live = _read_signal(live_path)
    times, labels = _read_decisions(decisions_path)

    return Session(
        calibration=calibration,
        live=live,
        decision_times_s=times,
        decisions=labels,
        source=folder,
    )