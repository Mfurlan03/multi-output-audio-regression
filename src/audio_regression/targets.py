"""Target definitions and scaling.

The 11 Spotify attributes live on wildly different scales: tempo has a variance of roughly
800 (BPM²) while danceability has a variance of about 0.03. Training on, or reporting, a
plain MSE averaged over raw targets means tempo dominates both the loss and the headline
number. `TargetScaler` standardizes each target with statistics from the training split
only, so every attribute contributes equally.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TARGETS = [
    "valence",
    "liveness",
    "time_signature",
    "energy",
    "speechiness",
    "instrumentalness",
    "acousticness",
    "tempo",
    "loudness",
    "danceability",
    "key",
]  # fmt: skip  (column order of the original metadata CSVs)


@dataclass
class TargetScaler:
    mean_: np.ndarray | None = None
    std_: np.ndarray | None = None

    def fit(self, y: np.ndarray) -> TargetScaler:
        y = np.asarray(y, dtype=np.float64)
        self.mean_ = y.mean(axis=0)
        std = y.std(axis=0)
        self.std_ = np.where(std > 0, std, 1.0)
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        self._check()
        return (np.asarray(y, dtype=np.float64) - self.mean_) / self.std_

    def inverse_transform(self, z: np.ndarray) -> np.ndarray:
        self._check()
        return np.asarray(z, dtype=np.float64) * self.std_ + self.mean_

    def _check(self) -> None:
        if self.mean_ is None:
            raise RuntimeError("TargetScaler is not fitted")
