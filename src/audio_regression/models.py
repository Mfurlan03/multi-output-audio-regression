"""Models: a song-level Random Forest baseline and a sequence GRU."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

from .features import summarize
from .targets import TargetScaler


def song_vectors(sequences: list[np.ndarray]) -> np.ndarray:
    return np.vstack([summarize(s) for s in sequences])


def build_random_forest(seed: int = 0, per_target: bool = True, **kw):
    """Song-level Random Forest.

    per_target=True (default) fits one forest per attribute. A single *joint* multi-output
    forest chooses every split by the summed variance reduction across all 11 targets, so
    noisy or hard-to-learn targets (and, on raw scales, tempo) dictate the splits that every
    other target must use. On the synthetic sanity set this drops loudness R² from ~0.98
    (per-target) to ~0.07 (joint); see tests/test_audio_regression.py.
    """
    params = dict(n_estimators=300, max_features=0.5, min_samples_leaf=2, n_jobs=-1, random_state=seed)
    params.update(kw)
    rf = RandomForestRegressor(**params)
    return MultiOutputRegressor(rf) if per_target else rf


# ---------------------------------------------------------------- GRU (PyTorch)


@dataclass
class GRUConfig:
    input_size: int = 39
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.3
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 16
    max_epochs: int = 30
    patience: int = 5
    max_len: int = 2000  # frames (~46 s at hop 512); long songs are cropped
    seed: int = 0


def _torch():
    import torch

    return torch


class FeatureScaler:
    """Per-dimension standardization of frame features, fit on training frames only.

    The original GRU consumed raw MFCCs (values in the hundreds) next to chroma in [0, 1].
    """

    def fit(self, seqs: list[np.ndarray]) -> FeatureScaler:
        frames = np.concatenate(seqs, axis=0)
        self.mean_ = frames.mean(axis=0)
        self.std_ = frames.std(axis=0) + 1e-6
        return self

    def transform(self, seq: np.ndarray) -> np.ndarray:
        return ((seq - self.mean_) / self.std_).astype(np.float32)


def build_gru(cfg: GRUConfig, n_outputs: int):
    torch = _torch()
    nn = torch.nn

    class MusicGRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(
                cfg.input_size, cfg.hidden_size, cfg.num_layers, batch_first=True,
                dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            )  # fmt: skip
            self.norm = nn.LayerNorm(cfg.hidden_size)
            self.head = nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(cfg.hidden_size, n_outputs))

        def forward(self, packed):
            _, h = self.gru(packed)
            return self.head(self.norm(h[-1]))

    return MusicGRU()


def _batches(seqs, y, batch_size, shuffle, rng):
    torch = _torch()
    from torch.nn.utils.rnn import pack_sequence

    idx = rng.permutation(len(seqs)) if shuffle else np.arange(len(seqs))
    for s in range(0, len(idx), batch_size):
        b = idx[s : s + batch_size]
        packed = pack_sequence([torch.from_numpy(seqs[i]) for i in b], enforce_sorted=False)
        yield packed, (torch.from_numpy(y[b]).float() if y is not None else None)


class GRURegressor:
    """Fit/predict wrapper: scales inputs and targets, early-stops on validation loss."""

    def __init__(self, cfg: GRUConfig | None = None):
        self.cfg = cfg or GRUConfig()
        self.history: list[dict] = []

    def _prep(self, seqs):
        return [self.x_scaler.transform(s[: self.cfg.max_len]) for s in seqs]

    def fit(self, train_seqs, y_train, val_seqs, y_val):
        torch = _torch()
        cfg = self.cfg
        torch.manual_seed(cfg.seed)
        rng = np.random.default_rng(cfg.seed)
        self.x_scaler = FeatureScaler().fit([s[: cfg.max_len] for s in train_seqs])
        self.y_scaler = TargetScaler().fit(y_train)
        xtr, xva = self._prep(train_seqs), self._prep(val_seqs)
        ytr, yva = self.y_scaler.transform(y_train), self.y_scaler.transform(y_val)

        self.model = build_gru(cfg, y_train.shape[1])
        opt = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        loss_fn = torch.nn.MSELoss()
        best, best_state, bad = float("inf"), None, 0
        for epoch in range(cfg.max_epochs):
            self.model.train()
            tr_losses = []
            for xb, yb in _batches(xtr, ytr, cfg.batch_size, True, rng):
                opt.zero_grad()
                loss = loss_fn(self.model(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                tr_losses.append(loss.item())
            val = float(np.mean((self._predict_scaled(xva) - yva) ** 2))
            self.history.append({"epoch": epoch, "train_loss": float(np.mean(tr_losses)), "val_loss": val})
            if val < best - 1e-4:
                best, bad = val, 0
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
            else:
                bad += 1
                if bad >= cfg.patience:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def _predict_scaled(self, prepped):
        torch = _torch()
        self.model.eval()
        out = []
        with torch.no_grad():
            for xb, _ in _batches(prepped, None, self.cfg.batch_size, False, None):
                out.append(self.model(xb).numpy())
        return np.vstack(out)

    def predict(self, seqs) -> np.ndarray:
        return self.y_scaler.inverse_transform(self._predict_scaled(self._prep(seqs)))
