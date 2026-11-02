import numpy as np
import pandas as pd
import pytest

from audio_regression.data import load_split, make_synthetic_dataset, normalize_name
from audio_regression.features import N_FEATURES, SAMPLE_RATE, extract_features, summarize
from audio_regression.metrics import r2, report
from audio_regression.targets import TARGETS, TargetScaler


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    return make_synthetic_dataset(tmp_path_factory.mktemp("synth"), n_songs=60, seconds=3.0)


def test_extract_features_shape():
    y = np.sin(2 * np.pi * 440 * np.arange(SAMPLE_RATE) / SAMPLE_RATE).astype(np.float32)
    f = extract_features(y)
    assert f.shape[1] == N_FEATURES == 39
    assert f.dtype == np.float32 and np.isfinite(f).all()


def test_extract_rejects_stereo():
    with pytest.raises(ValueError):
        extract_features(np.zeros((2, 1000)))


def test_summarize_is_mean_and_std():
    frames = np.array([[1.0, 2.0], [3.0, 6.0]])
    np.testing.assert_allclose(summarize(frames), [2.0, 4.0, 1.0, 2.0])


def test_target_scaler_roundtrip_and_constant_column():
    y = np.column_stack([np.linspace(60, 180, 50), np.full(50, 4.0)])
    s = TargetScaler().fit(y)
    z = s.transform(y)
    np.testing.assert_allclose(z[:, 0].mean(), 0, atol=1e-9)
    np.testing.assert_allclose(z[:, 1], 0)  # constant target: no divide-by-zero
    np.testing.assert_allclose(s.inverse_transform(z), y)


def test_target_scaler_requires_fit():
    with pytest.raises(RuntimeError):
        TargetScaler().transform(np.zeros((1, 1)))


def test_r2_perfect_and_mean_predictor():
    y = np.random.default_rng(0).normal(size=(100, 3))
    np.testing.assert_allclose(r2(y, y), 1.0)
    np.testing.assert_allclose(r2(y, np.broadcast_to(y.mean(0), y.shape)), 0.0, atol=1e-12)


def test_report_exposes_tempo_dominance():
    """Raw mean-MSE is dominated by the high-variance target; R² is not."""
    rng = np.random.default_rng(0)
    y = rng.normal(size=(200, len(TARGETS))) * 0.1
    y[:, TARGETS.index("tempo")] = rng.normal(120, 29, size=200)
    pred = y + rng.normal(size=y.shape) * 0.05
    pred[:, TARGETS.index("tempo")] = 120  # tempo: mean predictor
    table, summary = report(y, pred, y)
    tempo = table.set_index("target").loc["tempo"]
    assert tempo.share_of_raw_mse > 0.99
    assert abs(tempo.r2) < 0.05
    assert summary["n_targets_beating_baseline"] == len(TARGETS) - 1


def test_report_shape_mismatch():
    with pytest.raises(ValueError):
        report(np.zeros((3, 11)), np.zeros((3, 10)), np.zeros((3, 11)))


def test_normalize_name_handles_mojibake():
    assert normalize_name("Für_immer.npy") == "Fur_immer.npy"
    assert normalize_name("BÃ¼cher.npy").isascii()


def test_synthetic_dataset_layout(synth):
    for split in ("train", "val", "test"):
        meta = pd.read_csv(synth / f"{split}.csv")
        assert list(meta.columns) == ["file_path", *TARGETS]
        assert len(list((synth / split).glob("*.npy"))) == len(meta)


def test_load_split_matches_metadata(synth):
    s = load_split(synth, "train", max_len=50)
    assert len(s.sequences) == len(s.y) == len(s.names)
    assert all(seq.shape[0] <= 50 and seq.shape[1] == 39 for seq in s.sequences)


def _rf_r2(synth, per_target):
    from audio_regression.models import build_random_forest, song_vectors

    tr, te = load_split(synth, "train"), load_split(synth, "test")
    ys = TargetScaler().fit(tr.y)
    rf = build_random_forest(n_estimators=100, per_target=per_target)
    rf.fit(song_vectors(tr.sequences), ys.transform(tr.y))
    table, _ = report(te.y, ys.inverse_transform(rf.predict(song_vectors(te.sequences))), tr.y)
    return table.set_index("target").r2


def test_per_target_forest_learns_signal_not_noise(synth):
    r2s = _rf_r2(synth, per_target=True)
    assert r2s["loudness"] > 0.6 and r2s["tempo"] > 0.3
    assert r2s["valence"] < 0.3  # pure noise target


def test_joint_forest_is_hijacked_by_noise_targets(synth):
    """Documents why per-target forests are the default."""
    assert _rf_r2(synth, per_target=True)["loudness"] > _rf_r2(synth, per_target=False)["loudness"] + 0.3


def test_gru_trains_and_predicts(synth):
    pytest.importorskip("torch")
    from audio_regression.models import GRUConfig, GRURegressor

    tr, va = load_split(synth, "train", 200), load_split(synth, "val", 200)
    cfg = GRUConfig(hidden_size=16, num_layers=1, max_epochs=3, patience=2, batch_size=8)
    gru = GRURegressor(cfg).fit(tr.sequences, tr.y, va.sequences, va.y)
    pred = gru.predict(va.sequences)
    assert pred.shape == va.y.shape and np.isfinite(pred).all()
    assert 1 <= len(gru.history) <= 3
