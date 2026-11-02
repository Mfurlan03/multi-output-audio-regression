# Multi-Output Audio Feature Regression

[![CI](https://github.com/Mfurlan03/multi-output-audio-regression/actions/workflows/ci.yml/badge.svg)](https://github.com/Mfurlan03/multi-output-audio-regression/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

How well can musical attributes be predicted directly from full-length audio? This project
builds an end-to-end pipeline that turns raw audio into frame-level features and trains
models to predict **11 Spotify attributes** at once: danceability, energy, valence, tempo,
liveness, speechiness, instrumentalness, acousticness, loudness, time signature and key.

The goal isn't one low error number. It's understanding **which attributes are learnable
from audio, which dominate the error, and how evaluation choices can make a model look much
better or worse than it is.**

---

## Pipeline

```
audio (mp3) ──librosa @ 22.05 kHz──▶ frames × 39 features ──┬─▶ mean/std pooling (78-d) ──▶ Random Forest (one per target)
                                                            └─▶ standardized sequence ─────▶ GRU ──▶ 11 outputs
  13 MFCC · 12 chroma · 7 spectral contrast · 6 tonnetz · 1 ZCR
```

| Module | What it does |
|---|---|
| `features.py` | Frame-level feature extraction (39-d), song-level pooling |
| `data.py` | Loads `.npy` tensors + metadata per split, Unicode-safe filename matching, synthetic sanity dataset |
| `targets.py` | Target list, train-only `TargetScaler` |
| `metrics.py` | Per-target MSE / MAE / **R²** plus a predict-the-mean baseline |
| `models.py` | Per-target Random Forest; GRU with packed variable-length sequences, input and target standardization, early stopping |
| `cli.py` | `extract`, `evaluate`, `demo` |

```bash
pip install -e ".[gru,dev]"
python -m audio_regression extract  --audio-dir songs/ --out data/train/
python -m audio_regression evaluate --data-dir data/ --model rf --out results/
python -m audio_regression evaluate --data-dir data/ --model gru --out results/
python -m audio_regression demo     # synthetic end-to-end run, no dataset needed
```

---

## Original results (course project, APS360) and what they actually show

The original experiments live in [`notebooks/`](notebooks), unchanged. Their saved outputs show:

| Model | Reported metric | Value (from notebook outputs) |
|---|---|---|
| Random Forest (mean/std pooled features) | 5-fold CV MSE (train+val) | 83.68 |
| Random Forest | test MSE averaged over 11 targets | ≈ 79.1 |
| GRU, iteration 1 | test MSE | 90.17 |
| GRU, iteration 2, exp 1 (hidden 256, lr 1e-3, dropout 0.3) | test MSE | 90.15 |
| GRU, iteration 2, exp 2 (hidden 512, lr 5e-4, dropout 0.3) | test MSE | 90.17 |

**Raw MSE averaged over targets is the wrong headline metric here.** The RF's per-target test
MSEs (saved in `APS360BaseModel_V2.ipynb`) were:

| tempo | loudness | key | time_sig | instrumentalness | valence | acousticness | liveness | energy | danceability | speechiness |
|---|---|---|---|---|---|---|---|---|---|---|
| 843.9 | 13.86 | 11.36 | 0.242 | 0.084 | 0.049 | 0.047 | 0.027 | 0.026 | 0.022 | 0.007 |

**Tempo is 97% of the averaged MSE.** Three consequences:

1. The aggregate number says almost nothing about the other 10 attributes.
2. Three GRU configurations landing at 90.15–90.17 is the signature of a model
   predicting close to the training mean. The targets were never standardized, so the MSE
   loss was effectively a tempo loss.
3. The original RF was a single joint multi-output forest on raw targets. Every split was
   chosen mostly by tempo's variance reduction, and the other attributes inherited those splits.

> An earlier version of this README quoted GRU test MSEs of 24.67 and 3.97. Those values
> don't appear in any saved notebook output, and the second-iteration notebook records
> 90.17 for the configuration labelled 24.67. They've been removed until they can be
> reproduced.

## What v0.2 changes

| Problem | Fix |
|---|---|
| Targets on scales from 0.01 to 800 | `TargetScaler` fit on the training split; models train on standardized targets |
| Raw MSE hides per-target quality | Report per-target **R²** and MSE against a **predict-the-mean baseline**; mean R² as the headline |
| Joint forest lets one target pick every split | One forest per target (`MultiOutputRegressor`) |
| GRU fed raw MFCCs (hundreds) next to chroma (0–1) | Per-dimension input standardization fit on training frames |
| Padded batches | Packed variable-length sequences, gradient clipping, early stopping on validation loss |
| Unicode filenames renamed on disk (and failing) | Match normalized names in memory; never mutate the dataset |

### Sanity check: synthetic audio with known answers

`python -m audio_regression demo` synthesizes click tracks where **tempo** (click rate) and
**loudness** (amplitude) are encoded in the audio and the other nine targets are random noise.
A correct pipeline should learn the first two and score R² ≈ 0 or below on the rest. This check
is in the test suite, and it's how the joint-forest problem was found:

| Random Forest variant (default settings) | loudness R² | tempo R² |
|---|---:|---:|
| Joint multi-output forest (original approach) | 0.50 | 0.72 |
| One forest per target (v0.2) | **0.87** | **0.85** |

*80 synthetic clips (68 train+val, 12 test), seed 0, 300 trees, standardized targets. With only
12 test clips these are rough estimates; the direction is consistent across seeds and is
enforced by `test_joint_forest_is_hijacked_by_noise_targets`.*

**Next step:** re-run `evaluate` on the original Spotify/FMA dataset to get per-target R² for both
models. The audio itself isn't redistributable, so it isn't in this repo.

---

## Development

```bash
pip install -e ".[dev]"     # add [gru] for the PyTorch model
ruff check . && pytest -q
```

CI runs lint and the test suite, including the synthetic end-to-end checks and a short GRU
training run on CPU.

## Limitations

- Spotify's attributes are themselves model outputs, so they put a ceiling on what audio can recover.
- Third-party audio sources reflect genre and popularity biases.
- `key` and `time_signature` are categorical. Treating them as regression targets is a
  simplification; classification heads would be the principled fix.

## Ethics

Respect copyright and platform terms when collecting audio, and document data provenance.
This repo ships no audio.

## Contact

Michael Furlano · [linkedin.com/in/michaelfurlano](https://linkedin.com/in/michaelfurlano) · [github.com/mfurlan03](https://github.com/mfurlan03)
