# EMG Signal Classification Prototype

This is the software proof-of-concept for our engineering capstone project.
It classifies six movements from a recorded four-channel surface EMG signal.
The method is based on *An EMG-Based Biofeedback System for Tailored
Interventions Involving Distributed Muscles*:

1. second-order 20-400 Hz Butterworth band-pass filtering;
2. 64 ms windows with a 32 ms hop;
3. RMS, waveform length, zero-crossing, and slope-sign-change features;
4. five-dimensional multiclass LDA followed by Gaussian Naive Bayes; and
5. one majority-voted decision for every five feature windows.

At this stage the code only works with recorded data. Hardware input and
Tetris control are outside the scope of this prototype.

## Dataset

Use version 2 of the open [Dataset for multi-channel surface electromyography
(sEMG) signals of hand gestures](https://data.mendeley.com/datasets/ckwc76xr2z/2)
by Ozdemir, Kisa, Guren, and Akan. It is licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

1. Open the dataset page and download one file from `sEMG-dataset/raw/mat`.
   The pipeline intentionally uses the **raw**, not already-filtered, data.
2. Rename the selected file to `participant.mat`.
3. Place it at `data/raw/participant.mat`.

Dataset files are excluded from Git. The MAT file must contain the variables
`data` and `fs`; the optional `iD` variable is used as the participant label.

The notebook retains the six gestures used by the reference paper:

| Dataset gesture | Class label | Intended control |
| --- | --- | --- |
| Rest | `rest` | No input |
| Wrist flexion | `wrist_flexion` | Left |
| Wrist extension | `wrist_extension` | Right |
| Supination | `supination` | Rotate right |
| Pronation | `pronation` | Rotate left |
| Finger abduction | `finger_abduction` | Drop |

The timing schedule comes from the dataset authors' published
[segmentation procedure](https://www.mathworks.com/matlabcentral/answers/1719545-emg-gesture-segmentation-issues):
five repetitions start at 4, 138, 272, 406, and 540 seconds; each gesture is
the first six seconds of its ten-second slot.

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
package and `data/raw/participant.mat` resolve correctly.

## Expected output

For one complete participant file, the pipeline produces:

- 30 trials: six classes by five repetitions;
- 186 overlapping windows per six-second trial;
- 5,580 windows total, balanced at 930 per class; and
- 16 features per window: four features from each of four channels.

The notebook reports shuffled, stratified five-fold accuracy and per-class
precision as mean (standard deviation).
Its visuals mirror the paper's calibration-signal panel and two-dimensional
LDA projection.

Five-window majority voting remains implemented in `predict_with_voting` for
future use on a chronological hardware/game stream.

## Limitations

- The example uses one healthy participant, so the result cannot be treated as
  performance on a person with MS.
- The cross-validation split is made at the window level to match the paper.
  Since neighbouring windows overlap, the reported accuracy may be optimistic.
- The filter is offline and zero-phase. A live version will need a causal
  implementation.
- This is an engineering prototype, not a medical device or clinical tool.
