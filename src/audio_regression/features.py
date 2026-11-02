"""Frame-level audio features, matching the original data-processing notebook.

Per frame (librosa defaults, 22,050 Hz, hop 512):
  13 MFCC + 12 chroma + 7 spectral contrast + 6 tonnetz + 1 zero-crossing rate = 39 dims.
"""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 22050
FEATURE_GROUPS = {"mfcc": 13, "chroma": 12, "spectral_contrast": 7, "tonnetz": 6, "zcr": 1}
N_FEATURES = sum(FEATURE_GROUPS.values())  # 39


def extract_features(waveform: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Return a (time_steps, 39) float32 array."""
    import librosa  # heavy import, keep lazy

    y = np.asarray(waveform, dtype=np.float32)
    if y.ndim != 1:
        raise ValueError("expected a mono waveform")
    feats = np.vstack(
        [
            librosa.feature.mfcc(y=y, sr=sr, n_mfcc=FEATURE_GROUPS["mfcc"]),
            librosa.feature.chroma_stft(y=y, sr=sr),
            librosa.feature.spectral_contrast(y=y, sr=sr),
            librosa.feature.tonnetz(y=y, sr=sr),
            librosa.feature.zero_crossing_rate(y=y),
        ]
    )
    return feats.T.astype(np.float32)


def load_and_extract(path: str, sr: int = SAMPLE_RATE, max_seconds: float | None = None) -> np.ndarray:
    import librosa

    y, _ = librosa.load(path, sr=sr, duration=max_seconds)
    return extract_features(y, sr)


def summarize(frames: np.ndarray) -> np.ndarray:
    """Fixed-length song vector for non-sequential models: per-dim mean and std (78 dims)."""
    frames = np.asarray(frames, dtype=np.float32)
    return np.concatenate([frames.mean(axis=0), frames.std(axis=0)])
