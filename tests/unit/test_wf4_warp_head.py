"""SOTA-M12 (D-K4-7) — Vertragstests des WF-V4-Klassen-Warp-Head (Challenger zu Phase 12).

Verträge (Aufgaben-Scope):
- Flag ``use_wf4_warp_head`` Default AUS (``backend/core/music_model_flags.py``)
  ⇒ bit-identisches Bestandsverhalten: die Stage gibt ``(None, False)`` zurück.
- Determinismus (§G5 (copilot-instructions.md)): 2× gleicher Input ⇒ bit-identisch.
- NaN/Inf-Sanitisierung der Eingaben (§0a (copilot-instructions.md)): Ausgaben
  endlich, Warp-Kurve innerhalb WARP_LIMIT_LOG2.
- Warp-Schätzung auf Synth-Signal plausibel-tolerant gegen bekannte GT-Kurve
  (2**(gt_warp) ≈ stretch_factors).
"""

from __future__ import annotations

import numpy as np
import pytest

import backend.core.music_model_flags as mmf
from backend.core.dsp.wf4_warp_head import (
    N_SPEC_FEAT,
    T_MAX_FRAMES,
    TARGET_FPS,
    WARP_LIMIT_LOG2,
    WF4WarpHead,
    apply_wf4_warp_stage,
    build_wf4_warp_model,
    compute_spectral_features,
    warp_to_stretch_factors,
)

SR = 48_000


# ── Signal-Helfer ─────────────────────────────────────────────────────────────


def _pitch_conf(n: int, f0: float = 220.0) -> tuple[np.ndarray, np.ndarray]:
    return np.full(n, f0, dtype=np.float64), np.ones(n, dtype=np.float64)


class _FixedHead:
    """Deterministischer Test-Head (Muster SOTA-M9-Test-Injektion)."""

    def is_available(self) -> bool:
        return True

    def predict(self, x_f0, x_conf, x_feat):
        t = np.arange(x_f0.shape[-1], dtype=np.float32)
        warp = 0.01 * np.sin(t * 0.1).astype(np.float32)
        conf = np.full_like(x_f0, 0.9, dtype=np.float32)
        return warp, conf


class _NanEchoHead:
    """Gibt Eingaben zurück, um NaN-Lecks sichtbar zu machen."""

    def is_available(self) -> bool:
        return True

    def predict(self, x_f0, x_conf, x_feat):
        return np.asarray(x_f0, dtype=np.float32), np.asarray(x_conf, dtype=np.float32)


# ── Flag Default AUS ⇒ bit-identisches Bestandsverhalten ──────────────────────


def test_flag_default_off() -> None:
    """``use_wf4_warp_head`` Default False — Challenger läuft nie ohne Sign-off
    (§v10.802 (copilot-instructions.md))."""
    assert mmf.use_wf4_warp_head is False


def test_stage_flag_off_returns_none_false(monkeypatch) -> None:
    """Flag AUS ⇒ die Stage liefert (None, False): keine Verhaltensänderung,
    der heuristische Phase-12-Pfad bleibt unangetastet."""

    class _Bomb:
        def is_available(self) -> bool:
            raise AssertionError("Stage darf bei Flag AUS nicht auf den Head zugreifen")

        def predict(self, *a, **k):
            raise AssertionError("Stage darf bei Flag AUS nicht inferieren")

    pitch, conf = _pitch_conf(64)
    res, used = apply_wf4_warp_stage(pitch, conf, SR, head=_Bomb())
    assert res is None
    assert used is False


# ── Determinismus (§G5 (copilot-instructions.md)) ─────────────────────────────


def test_inference_two_runs_bit_identical(monkeypatch) -> None:
    """2× gleicher Input ⇒ bit-identische Stretch-Faktoren (§G5 (copilot-instructions.md))."""
    monkeypatch.setattr(mmf, "use_wf4_warp_head", True)
    head = _FixedHead()
    pitch, conf = _pitch_conf(200)

    r1, u1 = apply_wf4_warp_stage(pitch, conf, SR, head=head)
    r2, u2 = apply_wf4_warp_stage(pitch, conf, SR, head=head)

    assert u1 is True and u2 is True
    assert r1 is not None and r2 is not None
    np.testing.assert_array_equal(r1, r2)  # bit-identisch
    assert r1.dtype == np.float32
    assert r1.shape == (200,)
    assert bool(np.isfinite(r1).all())


def test_torch_model_forward_deterministic(monkeypatch) -> None:
    """Torch-Ref (eval/inference_mode): 2× Forward bit-identisch, Warp begrenzt."""
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(mmf, "use_wf4_warp_head", True)
    torch.manual_seed(1234)
    m1 = build_wf4_warp_model()
    torch.manual_seed(1234)
    m2 = build_wf4_warp_model()
    sd1, sd2 = m1.state_dict(), m2.state_dict()
    assert all(torch.equal(sd1[k], sd2[k]) for k in sd1)

    m1.eval()
    x_f0 = torch.zeros(1, 50)
    x_conf = torch.ones(1, 50)
    x_feat = torch.zeros(1, 50, N_SPEC_FEAT)
    with torch.no_grad():
        o1a, c1a = m1(x_f0, x_conf, x_feat)
        o1b, c1b = m1(x_f0, x_conf, x_feat)
    assert torch.equal(o1a, o1b)
    assert torch.equal(c1a, c1b)
    assert float(o1a.abs().max()) <= WARP_LIMIT_LOG2 + 1e-6


# ── NaN/Inf-Sanitisierung (§0a (copilot-instructions.md)) ─────────────────────


def test_nan_inf_sanitize_features(monkeypatch) -> None:
    """compute_spectral_features: NaN/Inf im Audio ⇒ Ausgabe endlich."""
    monkeypatch.setattr(mmf, "use_wf4_warp_head", True)
    n = SR
    mono = np.sin(np.linspace(0, 2 * np.pi * 220 * 3, n, dtype=np.float64)).astype(np.float32)
    bad = mono.copy()
    bad[::1000] = np.nan
    bad[::500] = np.inf
    feats = compute_spectral_features(bad, SR, 50)
    assert feats.shape == (50, N_SPEC_FEAT)
    assert feats.dtype == np.float32
    assert bool(np.isfinite(feats).all())


def test_nan_inf_sanitize_stage_output(monkeypatch) -> None:
    """apply_wf4_warp_stage: NaN/Inf in pitch/confidence ⇒ Ausgabe endlich
    (Head-Echo-Modus macht ein Leck sichtbar, dann saniert die Stage)."""
    monkeypatch.setattr(mmf, "use_wf4_warp_head", True)
    pitch = np.array([220.0, np.nan, np.inf, -np.inf, 220.0, 220.0], dtype=np.float64)
    conf = np.array([1.0, np.nan, 0.5, 0.5, 1.0, 1.0], dtype=np.float64)
    res, used = apply_wf4_warp_stage(pitch, conf, SR, head=_NanEchoHead())
    assert used is True
    assert res is not None
    assert bool(np.isfinite(res).all())
    assert res.min() >= 0.90 - 1e-6 and res.max() <= 1.10 + 1e-6


def test_warp_to_stretch_factors_identity_and_bounds() -> None:
    """warp_to_stretch_factors: log2→sf (sf>1 = langsamer), NaN/Inf wird gesäubert."""
    sf = warp_to_stretch_factors(np.array([0.0, 0.01, -0.01], dtype=np.float32))
    np.testing.assert_allclose(sf, [1.0, 2**0.01, 2 ** (-0.01)], rtol=1e-6)
    bad = warp_to_stretch_factors(np.array([0.01, np.nan], dtype=np.float32))
    assert bool(np.isfinite(bad).all())


# ── Synth-Warp-Schätzung plausibel-tolerant gegen GT ───────────────────────────


def test_synth_warp_estimate_near_gt(monkeypatch) -> None:
    """Synth-Signal mit bekannter GT-Warp-Kurve: Stretch-Faktoren liegen
    tolerant in der Nähe der GT (nicht bit-identisch, nur plausibel —
    die GT kommt hier aus dem Test-Head, der echte Challenger liefert Näherungen)."""
    monkeypatch.setattr(mmf, "use_wf4_warp_head", True)
    t_frames = 200
    t = np.arange(t_frames, dtype=np.float64) / TARGET_FPS
    gt_warp = (0.01 * np.sin(2 * np.pi * 1.0 * t)).astype(np.float32)  # ≈ ±12 cents
    gt_stretch = warp_to_stretch_factors(gt_warp)

    class _GtHead:
        def is_available(self) -> bool:
            return True

        def predict(self, x_f0, x_conf, x_feat):
            return gt_warp, np.ones_like(x_f0, dtype=np.float32)

    pitch, conf = _pitch_conf(t_frames)
    res, used = apply_wf4_warp_stage(pitch, conf, SR, head=_GtHead())
    assert used is True
    assert res is not None
    assert res.shape == (t_frames,)
    # tolerant: max Abweichung ≤ 5e-3 (~6 cents) gegen die GT-Kurve
    assert float(np.max(np.abs(res.astype(np.float64) - gt_stretch.astype(np.float64)))) < 5e-3
    # und generell: Warp-Delta bleibt in der Warp-Kappe (§G188/§G189 (GEBOTE.md))
    assert float(np.max(np.abs(res - 1.0))) <= 0.10 + 1e-6


def test_stage_handles_long_curve_chunked(monkeypatch) -> None:
    """T > T_MAX_FRAMES ⇒ Overlap-Add-Chunking, Ausgabe-Shape bleibt vollständig."""
    monkeypatch.setattr(mmf, "use_wf4_warp_head", True)
    t = T_MAX_FRAMES + 150
    pitch, conf = _pitch_conf(t)
    res, used = apply_wf4_warp_stage(pitch, conf, SR, head=_FixedHead())
    assert used is True
    assert res is not None
    assert res.shape == (t,)
    assert bool(np.isfinite(res).all())


# ── Ladepfad: Trainings-Checkpoint statt ONNX (D-K3-85) ───────────────────────


def _save_ckpt(path, *, hidden: int = 8) -> None:
    """Legt einen Checkpoint im Vertrag von `scripts/train_wf4_warp_head.py` an."""
    torch = pytest.importorskip("torch")
    torch.manual_seed(7)
    net = build_wf4_warp_model(hidden=hidden)
    torch.save(
        {
            "model_state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()},
            "wf4_warp_config": {"n_feat": N_SPEC_FEAT, "hidden": hidden, "target_fps": TARGET_FPS},
            "seed": 7,
            "loss": 0.25,
        },
        path,
    )


def test_ladepfad_checkpoint_wenn_onnx_fehlt(monkeypatch, tmp_path) -> None:
    """Ohne ONNX-Export muss der Torch-Checkpoint des Trainings tragen.

    Vor dem Fix war `is_available()` immer False (ONNX existiert nie) ⇒ der
    Flag-Pfad war tot und jede AN/AUS-Messung verglich den heuristischen Pfad
    mit sich selbst (D-K3-85).
    """
    pytest.importorskip("torch")
    monkeypatch.setattr("backend.core.ml_memory_budget.try_allocate", lambda *a, **k: True)
    ckpt = tmp_path / "wf4_warp_head_v1.pt"
    _save_ckpt(ckpt)

    head = WF4WarpHead(model_path=tmp_path / "fehlt.onnx", checkpoint_path=ckpt)
    assert head.is_available() is True
    assert head.source == "checkpoint:wf4_warp_head_v1.pt"

    n = 64
    f0 = np.full(n, 0.01, dtype=np.float32)
    conf = np.ones(n, dtype=np.float32)
    feat = np.zeros((n, N_SPEC_FEAT), dtype=np.float32)
    w1, c1 = head.predict(f0, conf, feat)
    w2, c2 = head.predict(f0, conf, feat)

    np.testing.assert_array_equal(w1, w2)  # §G5 (copilot-instructions.md): bit-identisch
    np.testing.assert_array_equal(c1, c2)
    assert w1.shape == (n,) and w1.dtype == np.float32
    assert bool(np.isfinite(w1).all())
    assert float(np.abs(w1).max()) <= WARP_LIMIT_LOG2 + 1e-6
    assert float(np.min(c1)) >= 0.0 and float(np.max(c1)) <= 1.0


def test_ladepfad_ohne_artefakt_bleibt_inaktiv(tmp_path) -> None:
    """Fehlt beides, bleibt der Challenger inaktiv und bricht ehrlich ab (§V6 (copilot-instructions.md))."""
    head = WF4WarpHead(model_path=tmp_path / "fehlt.onnx", checkpoint_path=tmp_path / "fehlt.pt")
    assert head.is_available() is False
    assert head.source == "nicht_geladen"
    with pytest.raises(RuntimeError):
        head.predict(
            np.zeros(8, dtype=np.float32),
            np.ones(8, dtype=np.float32),
            np.zeros((8, N_SPEC_FEAT), dtype=np.float32),
        )


def test_echtes_trainingsartefakt_ist_erreichbar() -> None:
    """Regression D-K3-85: der deployte Checkpoint muss ladbar sein (kein Phantom)."""
    pytest.importorskip("torch")
    import pathlib

    import backend.core.dsp.wf4_warp_head as wf4mod

    repo = pathlib.Path(wf4mod.__file__).resolve().parents[3]
    if not (repo / wf4mod.DEFAULT_CHECKPOINT_SUBPATH).is_file():
        pytest.skip("WF4-Trainingsartefakt nicht vorhanden (CI ohne models/)")
    head = WF4WarpHead()
    assert head.is_available() is True
    assert head.source.startswith("checkpoint:")
