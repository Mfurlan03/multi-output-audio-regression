"""Scale-aware evaluation for multi-output regression.

`report()` returns one row per target with MSE, MAE and R², plus the same numbers for a
predict-the-training-mean baseline, and two aggregates:

* raw_mean_mse: the average of raw per-target MSEs (what the original notebooks reported;
  about 95% of it is tempo)
* mean_r2: the average per-target R². Scale-free: 0 = no better than predicting the mean,
  1 = perfect. Comparable across targets and across models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .targets import TARGETS


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ss_res = ((y_true - y_pred) ** 2).sum(axis=0)
    ss_tot = ((y_true - y_true.mean(axis=0)) ** 2).sum(axis=0)
    return 1.0 - ss_res / np.where(ss_tot > 0, ss_tot, np.nan)


def report(
    y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray, targets: list[str] = TARGETS
) -> tuple[pd.DataFrame, dict]:
    y_true, y_pred, y_train = (np.asarray(a, float) for a in (y_true, y_pred, y_train))
    if y_true.shape != y_pred.shape or y_true.shape[1] != len(targets):
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}, {len(targets)} targets")
    baseline = np.broadcast_to(y_train.mean(axis=0), y_true.shape)

    mse = ((y_true - y_pred) ** 2).mean(axis=0)
    base_mse = ((y_true - baseline) ** 2).mean(axis=0)
    table = pd.DataFrame(
        {
            "target": targets,
            "mse": mse,
            "mae": np.abs(y_true - y_pred).mean(axis=0),
            "r2": r2(y_true, y_pred),
            "baseline_mse": base_mse,
            "share_of_raw_mse": mse / mse.sum(),
        }
    )
    summary = {
        "raw_mean_mse": float(mse.mean()),
        "baseline_raw_mean_mse": float(base_mse.mean()),
        "mean_r2": float(np.nanmean(table.r2)),
        "n_targets_beating_baseline": int((mse < base_mse).sum()),
    }
    return table, summary
