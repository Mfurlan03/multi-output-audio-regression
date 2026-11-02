"""Loading feature tensors + metadata, and a synthetic dataset for end-to-end tests.

On-disk layout (same as the original project):

    data_dir/
      train/*.npy   val/*.npy   test/*.npy        # (time_steps, 39) per song
      train.csv     val.csv     test.csv          # file_path + 11 target columns
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .features import SAMPLE_RATE, extract_features
from .targets import TARGETS

SPLITS = ("train", "val", "test")


def normalize_name(name: str) -> str:
    """Unicode-normalize file names in memory.

    The notebooks renamed files on disk with unidecode, which failed for some mis-encoded
    names (e.g. 'BÃ¼cher...'); matching on normalized names avoids touching the files.
    """
    return unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()


@dataclass
class Split:
    names: list[str]
    sequences: list[np.ndarray]  # each (T_i, 39)
    y: np.ndarray  # (n, 11)


def load_split(data_dir: Path, split: str, max_len: int | None = None) -> Split:
    data_dir = Path(data_dir)
    meta = pd.read_csv(data_dir / f"{split}.csv")
    missing_cols = [t for t in TARGETS if t not in meta.columns]
    if missing_cols:
        raise ValueError(f"{split}.csv missing target columns {missing_cols}")
    files = {normalize_name(p.name): p for p in (data_dir / split).glob("*.npy")}
    meta["_key"] = meta["file_path"].map(lambda s: normalize_name(Path(str(s)).name))
    meta = meta[meta["_key"].isin(files)].sort_values("_key").reset_index(drop=True)
    if meta.empty:
        raise ValueError(f"no tensors in {data_dir / split} match {split}.csv")
    seqs = []
    for key in meta["_key"]:
        arr = np.load(files[key], mmap_mode="r")
        seqs.append(np.asarray(arr[:max_len] if max_len else arr, dtype=np.float32))
    return Split(meta["_key"].tolist(), seqs, meta[TARGETS].to_numpy(dtype=np.float64))


def make_synthetic_dataset(out_dir: Path, n_songs: int = 60, seconds: float = 4.0, seed: int = 0) -> Path:
    """Write a tiny dataset where *some* targets are genuinely learnable from the audio.

    Each clip is a click track over a tone. ``tempo`` sets the click rate and ``loudness``
    sets the amplitude (dB), so a working pipeline should achieve R² > 0 on those two, and
    R² ≈ 0 on the remaining targets, which are pure noise. It's a sanity check that the
    model learns real signal rather than memorizing.
    """
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    n = int(seconds * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    rows = {s: [] for s in SPLITS}
    for i in range(n_songs):
        split = SPLITS[0] if i < 0.7 * n_songs else SPLITS[1] if i < 0.85 * n_songs else SPLITS[2]
        tempo = rng.uniform(70, 180)
        loudness = rng.uniform(-25, -3)
        amp = 10 ** (loudness / 20)
        y = 0.2 * np.sin(2 * np.pi * rng.uniform(150, 600) * t)
        click_every = int(SAMPLE_RATE * 60 / tempo)
        for start in range(0, n, click_every):
            y[start : start + 200] += np.hanning(min(200, n - start)) * 1.5
        y = (amp * y / np.max(np.abs(y))).astype(np.float32)
        (out_dir / split).mkdir(parents=True, exist_ok=True)
        name = f"song_{i:04d}.npy"
        np.save(out_dir / split / name, extract_features(y))
        noise = rng.uniform(0, 1, size=len(TARGETS))
        record = dict(zip(TARGETS, noise, strict=True))
        record.update(tempo=tempo, loudness=loudness, key=float(rng.integers(0, 12)), time_signature=4.0)
        rows[split].append({"file_path": name, **record})
    for split, recs in rows.items():
        pd.DataFrame(recs)[["file_path", *TARGETS]].to_csv(out_dir / f"{split}.csv", index=False)
    return out_dir
