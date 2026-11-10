# EMG Signal Classification Prototype

This is the software proof-of-concept for our engineering capstone project.
It classifies five movements (Rest, Left, Right, Rotate, Drop) from four-channel
surface EMG recorded during a Tetris session. The model is trained on short
calibration recordings, then classifies the live session, and its decisions are
compared with the decisions recorded by the company's system. The method is
based on *An EMG-Based Biofeedback System for Tailored Interventions Involving
Distributed Muscles*:

1. 20-400 Hz band-pass filtering of the raw live data (a causal high-pass and
   low-pass filter; the calibration files arrive already filtered);
2. 64 ms windows, one ending at each recorded decision time (the length is set
   by `window_ms` in `PipelineConfig`);
3. RMS, waveform length, zero-crossing, and slope-sign-change features; and
4. four-dimensional multiclass LDA followed by Gaussian Naive Bayes.

At this stage the code only works with recorded data. Hardware input and
Tetris control are outside the scope of this prototype. Majority voting
(`predict_with_voting`) is implemented but not used.

## Data

Place these seven files directly in the `data/` folder:

| File | Contents |
| --- | --- |
| `calibration_data_Rest.csv`, `_Left.csv`, `_Right.csv`, `_Rotate.csv`, `_Drop.csv` | One 4-second, already-filtered, four-column recording per movement (1 kHz, no header) |
| `live_samples.csv` | The raw, unfiltered four-column recording of the game session (1 kHz, no header) |
| `decision_times.csv` | The company's decisions: `Time_s`, `Decision`, `Speed_index` (header row) |

Decision codes: 0 = Rest, 1 = Left, 2 = Right, 3 = Rotate, 5 = Drop. The speed
column is ignored. The channels appear to be left forearm, left quad, right
quad, and right forearm (inferred from the calibration files).

The data files are excluded from Git because they are patient recordings
supplied by the company.

## Setup

From the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the tests:

```powershell
python -m pytest
```

Start the demonstration notebook:

```powershell
python -m jupyter notebook emg_classifier_demo.ipynb
```

Run the notebook from the repository root so that the local `emg_pipeline`
package resolves correctly. The seven CSV files must be in `data/`.

## Expected output

With the default settings, the notebook produces:

- 620 training windows, 124 per movement (64 ms windows every 32 ms from each
  4-second calibration file), with 16 features per window;
- one classification for each of the company's decisions (about 28,500), using
  the window that ends at that decision's timestamp;
- overall and balanced accuracy, plus precision, recall, and F1 for each movement;
- a confusion matrix;
- a step-line timeline of the company's decisions and ours for the whole session;
- a zoomed view of a few seconds showing both sets of decisions above the
  amplitude of all four channels, with the windows where we disagree shaded.

Change `window_ms` in the notebook's `PipelineConfig(...)` to test other window
lengths.

## Limitations

- The data is one session from one patient, with only 4 seconds of calibration
  per movement.
- The company's decisions come from their own classifier, so the results
  measure agreement with it, not correctness.
- The filter used on the calibration data is assumed to be a 20-400 Hz band-pass;
  the exact filter is not known.
- Agreement is highest when our decisions are compared with the company's
  decisions about 320 ms later, which suggests their output is smoothed. We do
  not reproduce that yet.
- The channel order is inferred from the calibration files.
- This is an engineering prototype, not a medical device or clinical tool.
