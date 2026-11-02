"""Command line entry points.

python -m audio_regression extract  --audio-dir songs/ --out tensors/   # mp3/wav -> .npy
python -m audio_regression evaluate --data-dir data/ --model rf|gru     # train + report
python -m audio_regression demo                                          # synthetic end-to-end
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from .data import load_split, make_synthetic_dataset
from .features import load_and_extract
from .metrics import report
from .models import GRUConfig, GRURegressor, build_random_forest, song_vectors


def evaluate(data_dir: Path, model: str, out: Path | None, gru_cfg: GRUConfig | None = None) -> dict:
    cfg = gru_cfg or GRUConfig()
    tr = load_split(data_dir, "train", cfg.max_len)
    va = load_split(data_dir, "val", cfg.max_len)
    te = load_split(data_dir, "test", cfg.max_len)
    if model == "rf":
        # One forest per target (see build_random_forest). Standardizing is a no-op for
        # per-target trees but keeps the joint variant fair if it is selected.
        from .targets import TargetScaler

        ys = TargetScaler().fit(tr.y)
        rf = build_random_forest().fit(song_vectors(tr.sequences + va.sequences), ys.transform(np.vstack([tr.y, va.y])))
        pred = ys.inverse_transform(rf.predict(song_vectors(te.sequences)))
    elif model == "gru":
        gru = GRURegressor(cfg).fit(tr.sequences, tr.y, va.sequences, va.y)
        pred = gru.predict(te.sequences)
    else:
        raise ValueError(model)
    table, summary = report(te.y, pred, tr.y)
    summary.update(model=model, n_train=len(tr.y), n_val=len(va.y), n_test=len(te.y))
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(json.dumps(summary, indent=2))
    if out:
        out.mkdir(parents=True, exist_ok=True)
        table.to_csv(out / f"{model}_per_target.csv", index=False)
        (out / f"{model}_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def extract_dir(audio_dir: Path, out_dir: Path, max_seconds: float | None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(p for p in audio_dir.iterdir() if p.suffix.lower() in {".mp3", ".wav", ".flac", ".ogg"}):
        target = out_dir / (path.stem + ".npy")
        if target.exists():
            continue
        try:
            np.save(target, load_and_extract(str(path), max_seconds=max_seconds))
            print(f"ok   {path.name}")
        except Exception as e:  # keep going on corrupt files, like the original pipeline
            print(f"fail {path.name}: {e}")


def main() -> None:
    p = argparse.ArgumentParser(prog="audio_regression")
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--audio-dir", type=Path, required=True)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--max-seconds", type=float, default=None)
    v = sub.add_parser("evaluate")
    v.add_argument("--data-dir", type=Path, required=True)
    v.add_argument("--model", choices=["rf", "gru"], default="rf")
    v.add_argument("--out", type=Path, default=None)
    d = sub.add_parser("demo", help="Build a synthetic dataset and run both models on it")
    d.add_argument("--songs", type=int, default=80)
    a = p.parse_args()

    if a.cmd == "extract":
        extract_dir(a.audio_dir, a.out, a.max_seconds)
    elif a.cmd == "evaluate":
        evaluate(a.data_dir, a.model, a.out)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            make_synthetic_dataset(Path(tmp), n_songs=a.songs)
            for m in ("rf", "gru"):
                print(f"\n===== {m} =====")
                evaluate(Path(tmp), m, None, GRUConfig(hidden_size=32, num_layers=1, max_epochs=40, patience=8))


if __name__ == "__main__":
    main()
