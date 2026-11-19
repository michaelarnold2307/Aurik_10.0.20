"""Tests für F12 · BANQUET Real-Vinyl-Finetune (`scripts/train_banquet_vinyl_finetune.py`).

Sichern ab: Paarung/Alignment (deterministisch), Feature-Parität zum
Produktions-Pfad (plugins/banquet_vinyl_plugin.py) und die
Differenzierbarkeit der Trainingskette (feat → Kern → Maske → Loss).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_spec = importlib.util.spec_from_file_location(
    "train_banquet_vinyl_finetune", _ROOT / "scripts" / "train_banquet_vinyl_finetune.py"
)
assert _spec is not None and _spec.loader is not None
tbv = importlib.util.module_from_spec(_spec)
sys.modules["train_banquet_vinyl_finetune"] = tbv
_spec.loader.exec_module(tbv)


# ---------------------------------------------------------------------------
# Datenseite: Paarung, Alignment, Segmente
# ---------------------------------------------------------------------------


def test_pair_key_extracts_song_stem():
    assert tbv.pair_key(Path("vinyl_blues_1950s_crackle_hiss.wav")) == "vinyl_blues_1950s"
    assert tbv.pair_key(Path("vinyl_rock_1970s_crackle_chain_vinyl_cassette.wav")) == "vinyl_rock_1970s"
    # Fallback: zu kurze Namen bleiben intakt (kein stilles Abschneiden)
    assert tbv.pair_key(Path("rare_take.wav")) == "rare_take"


def test_discover_pairs_matches_by_key_and_warns_on_orphans(tmp_path: Path):
    (tmp_path / "clean").mkdir()
    (tmp_path / "damaged").mkdir()
    (tmp_path / "clean" / "vinyl_blues_1950s_clean.wav").touch()
    (tmp_path / "damaged" / "vinyl_blues_1950s_crackle.wav").touch()
    (tmp_path / "damaged" / "vinyl_blues_1950s_hiss_hum.wav").touch()
    # Waise ohne Referenz — muss übersprungen werden, nicht verschwinden
    (tmp_path / "damaged" / "vinyl_jazz_1960s_clicks.wav").touch()

    pairs = tbv.discover_pairs(tmp_path)
    assert [p.key for p in pairs] == ["vinyl_blues_1950s", "vinyl_blues_1950s"]
    assert all(p.clean.name == "vinyl_blues_1950s_clean.wav" for p in pairs)
    # deterministische Reihenfolge
    again = tbv.discover_pairs(tmp_path)
    assert [p.damaged.name for p in again] == [p.damaged.name for p in pairs]


def test_align_pair_recovers_integer_shift():
    rng = np.random.default_rng(7)
    n = 5 * tbv.SEG_N
    clean = np.zeros((2, n), dtype=np.float32)
    for k in range(2, n, 24000):
        clean[:, k] = rng.standard_normal(2).astype(np.float32)
    shift = 137
    damaged = np.zeros((2, n), dtype=np.float32)
    damaged[:, shift:] = clean[:, : n - shift]

    d, c, lag = tbv.align_pair(damaged, clean)
    assert lag == shift
    assert d.shape == c.shape
    assert float(np.abs(d - c).max()) < 1e-6


def test_make_segments_is_seeded_deterministic():
    rng_a = np.random.default_rng(11)
    rng_b = np.random.default_rng(11)
    pair = np.random.default_rng(1).standard_normal((2, 3 * tbv.SEG_N)).astype(np.float32)
    segs_a = tbv.make_segments(pair, pair, rng_a, n_segments=4)
    segs_b = tbv.make_segments(pair, pair, rng_b, n_segments=4)
    assert len(segs_a) == 4
    for (d1, c1), (d2, c2) in zip(segs_a, segs_b):
        assert np.array_equal(d1, d2) and np.array_equal(c1, c2)
    assert segs_a[0][0].shape == (2, tbv.SEG_N)


# ---------------------------------------------------------------------------
# Feature-Parität zum Produktions-Pfad
# ---------------------------------------------------------------------------


def test_prepare_features_torch_matches_np():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(3)
    chunk = rng.standard_normal((2, tbv.SEG_N)).astype(np.float32) * 0.1

    feat_np, ctx_np = tbv.prepare_features_np(chunk)
    feat_t, ctx_t = tbv.prepare_features_torch(chunk, "cpu", torch.float32)

    assert feat_np.shape == (1, tbv.N_BANDS, tbv.N_FRAMES, tbv.FEAT_DIM)
    np.testing.assert_allclose(feat_t.numpy(), feat_np, rtol=1e-3, atol=1e-5)
    np.testing.assert_allclose(
        ctx_t.numpy().view(np.float32).reshape(ctx_np.shape + (2,)),
        ctx_np.view(np.float32).reshape(ctx_np.shape + (2,)),
        rtol=1e-3,
        atol=1e-6,
    )


def test_reconstruct_torch_matches_plugin_extract_output():
    torch = pytest.importorskip("torch")
    try:
        from plugins.banquet_vinyl_plugin import BanquetVinylPlugin
    except Exception as exc:  # pragma: no cover - Plugin-Import hängt von der Env ab
        pytest.skip(f"Plugin nicht importierbar: {exc}")

    rng = np.random.default_rng(5)
    chunk = rng.standard_normal((2, tbv.SEG_N)).astype(np.float32) * 0.1
    raw = rng.standard_normal((1, tbv.N_BANDS, tbv.N_FRAMES, tbv.FEAT_DIM)).astype(np.float32)

    feat_np, ctx_np = tbv.prepare_features_np(chunk)
    plugin_out = BanquetVinylPlugin._extract_output(raw, 2, tbv.SEG_N, ctx_np)

    mask = tbv.core_output_to_mask(torch.from_numpy(raw))
    ctx_t = tbv.stft_ctx_torch(chunk, "cpu", torch.complex64)
    audio_t = tbv.reconstruct_torch(mask, ctx_t, length=tbv.SEG_N)

    assert plugin_out.shape == (2, tbv.SEG_N)
    np.testing.assert_allclose(audio_t[0].detach().numpy(), plugin_out[0], rtol=1e-3, atol=1e-4)


def test_wiener_target_mask_bounds():
    torch = pytest.importorskip("torch")
    clean = torch.rand(128, 128) + 0.5
    # identisches Paar → Ziel ≈ 1 (auf 0.99 gedeckelt)
    g_same = tbv.wiener_target_mask(clean, clean)
    assert torch.all(g_same <= 0.99) and torch.allclose(g_same, torch.full_like(g_same, 0.99))
    # starker Defekt (D = 5C → SNR 0 dB unterhalb) → Ziel klein, aber nie 0
    g_bad = tbv.wiener_target_mask(5.0 * clean, clean)
    assert torch.all(g_bad < 0.3) and torch.all(g_bad >= 0.01)


def test_a1_masking_loss_loader_exposes_uniform_callable():
    torch = pytest.importorskip("torch")
    fn = tbv.load_a1_masking_loss()
    # EAR-VAE-Rezept oder Produktcode-Fallback (PsychoacousticMaskingLoss) —
    # in dieser Repo MUSS einer der beiden da sein (Fail-closed-Assert).
    assert fn is not None
    pred = torch.zeros(1, 1, tbv.SEG_N)
    target = torch.zeros(1, 1, tbv.SEG_N)
    target[:, :, 1000:1100] = 0.3
    loss = fn(pred, target)
    assert torch.isfinite(loss)
    assert loss.ndim == 0


def test_snr_db_is_scale_invariant():
    rng = np.random.default_rng(2)
    ref = rng.standard_normal(48000).astype(np.float32)
    noise = rng.standard_normal(48000).astype(np.float32) * 0.01
    est = ref + noise
    s1 = tbv.snr_db(est, ref)
    s2 = tbv.snr_db(3.7 * est, 3.7 * ref)
    assert abs(s1 - s2) < 1e-6
    assert 30.0 < s1 < 50.0
    # NaN/Inf-Schutz (§0a (copilot-instructions.md))
    assert np.isfinite(tbv.snr_db(np.full(100, np.nan), ref[:100]))


# ---------------------------------------------------------------------------
# Trainingskette: Differenzierbarkeit + Kern-Shape
# ---------------------------------------------------------------------------


def test_trainable_chain_backpropagates():
    torch = pytest.importorskip("torch")

    class StubCore(torch.nn.Module):
        """Minimaler Kern-Stellvertreter mit echtem Parameter (Gradientenfluss)."""

        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x):
            return x * 0.0 + self.p

    rng = np.random.default_rng(9)
    d_chunk = rng.standard_normal((1, tbv.SEG_N)).astype(np.float32) * 0.1
    c_chunk = rng.standard_normal((1, tbv.SEG_N)).astype(np.float32) * 0.1

    feat, _ = tbv.prepare_features_torch(d_chunk, "cpu", torch.float32)
    d_ctx = tbv.stft_ctx_torch(d_chunk, "cpu", torch.complex64)
    c_ctx = tbv.stft_ctx_torch(c_chunk, "cpu", torch.complex64)
    target = tbv.wiener_target_mask(d_ctx, c_ctx)

    core = StubCore()
    opt = torch.optim.SGD(core.parameters(), lr=0.05)
    before = float(core.p)
    for _ in range(3):
        opt.zero_grad()
        mask = tbv.core_output_to_mask(core(feat))
        loss = torch.mean(torch.abs(mask - target))
        loss.backward()
        assert core.p.grad is not None and torch.isfinite(core.p.grad).all()
        opt.step()
    assert float(core.p) != before
    assert torch.isfinite(loss)


def test_load_core_matches_production_shapes():
    torch = pytest.importorskip("torch")
    if not (_ROOT / "models" / "banquet" / "banquet_vinyl_final.onnx").is_file():
        pytest.skip("Zero-Shot-ONNX nicht vorhanden (models/ ist gitignored)")

    core = tbv.load_core("cpu")
    n_params = sum(p.numel() for p in core.parameters())
    assert n_params > 1_000_000  # 24 Zellen BiLSTM + FC
    x = torch.zeros(1, tbv.N_BANDS, tbv.N_FRAMES, tbv.FEAT_DIM)
    with torch.no_grad():
        y = core(x)
    assert y.shape == (1, tbv.N_BANDS, tbv.N_FRAMES, tbv.FEAT_DIM)
    assert torch.isfinite(y).all()


def test_load_channels_first_wavfile_fallback_und_bug12_guard(monkeypatch, tmp_path):
    """scipy-Fallback ohne soundfile lädt channels-first; Bug 12 (§V36 (VERBOTEN.md)):
    kein blindes Tuple-Unpack von wavfile.read()."""
    import sys

    import scipy.io.wavfile as wavfile

    rng = np.random.default_rng(3)
    data = (rng.standard_normal((tbv.SEG_N, 2)) * 3000).astype(np.int16)
    wav_path = tmp_path / "probe.wav"
    wavfile.write(str(wav_path), tbv.TARGET_SR, data)

    monkeypatch.setitem(sys.modules, "soundfile", None)  # ImportError → scipy-Pfad
    audio = tbv.load_channels_first(wav_path)
    assert audio.shape == (2, tbv.SEG_N)  # channels-first (C, N)
    assert audio.dtype == np.float32

    # Bug 12 (§V36 (VERBOTEN.md)): defektes wavfile.read-Ergebnis muss hart fehlschlagen
    monkeypatch.setattr(wavfile, "read", lambda p: ("kaputt",))
    with pytest.raises(ValueError):
        tbv.load_channels_first(wav_path)
