"""Unit-Tests für den TF-GridNet-Klassen-Vollband-Denoiser (SOTA-M9, D-K4-2).

Verträge (Aufgaben-Scope):
- Shapes: (N,), (2, N), (N, 2) — Layout bleibt erhalten, float32, endlich.
- Determinismus (§G5 (copilot-instructions.md)): 2× gleicher Input ⇒ bit-identisch.
- Never-worsen-Harness (§v10.26 (.github/specs/v10.26_autonomer_never_worsen_arbiter.md)):
  Verschlechterung ⇒ bit-identischer Rücksprung auf die Eingabe.
- Layout: (2, N) und (N, 2) liefern äquivalente Kanal-Ergebnisse.
- Flag AUS ⇒ keine Verhaltensänderung (music_model_flags.use_tf_gridnet_quality
  Default False; Stage ist reiner Passthrough, der Phase-03-Hook ruft sie nicht).
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.core import music_model_flags
from backend.core.dsp.tf_gridnet_denoiser import (
    NEVER_WORSEN_MARGIN,
    TFGridNetDenoiser,
    apply_tf_gridnet_quality_stage,
    build_tf_gridnet_model,
    default_quality_scorer,
    never_worsen_select,
)

SR = 48_000


# ── Signal-Helfer ─────────────────────────────────────────────────────────────


def _tone(n: int = SR, freq: float = 440.0, amp: float = 0.5) -> np.ndarray:
    t = np.arange(n, dtype=np.float32) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _tone_noisy(n: int = SR) -> np.ndarray:
    rng = np.random.RandomState(7)
    return (_tone(n) + 0.05 * rng.randn(n)).astype(np.float32)


def _stereo_pair(n: int = SR) -> np.ndarray:
    return np.stack([_tone(n), _tone(n, freq=660.0, amp=0.4)], axis=0)


def _has_warning(caplog, needle: str) -> bool:
    return any(needle in rec.getMessage() for rec in caplog.records if rec.levelname == "WARNING")


# ── Synthetische ONNX-Doubles (Konstant-Maske) ────────────────────────────────


def _save_const_mask_onnx(path, value: float, dims=("b", "f", "t")) -> None:
    """Schreibt ein elementweises ONNX-Double: mask = log_mag * 0 + value."""
    pytest.importorskip("onnx")
    import onnx
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("log_mag", TensorProto.FLOAT, list(dims))
    y = helper.make_tensor_value_info("mask", TensorProto.FLOAT, list(dims))
    zero = helper.make_tensor("zero", TensorProto.FLOAT, [], [0.0])
    val = helper.make_tensor("val", TensorProto.FLOAT, [], [float(value)])
    mul = helper.make_node("Mul", ["log_mag", "zero"], ["zx"])
    add = helper.make_node("Add", ["zx", "val"], ["mask"])
    graph = helper.make_graph([mul, add], "tfg_const_mask", [x], [y], initializer=[zero, val])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, str(path))


@pytest.fixture(autouse=True)
def _ml_budget(monkeypatch):
    """ML-Budget-Gate ist nicht Testgegenstand (Systemgrenze wird gemockt)."""
    monkeypatch.setattr("backend.core.ml_memory_budget.try_allocate", lambda *a, **k: True)


def _make_denoiser(tmp_path, value: float = 1.0, dims=("b", "f", "t"), **kwargs) -> TFGridNetDenoiser:
    model_path = tmp_path / "tf_gridnet_quality.onnx"
    _save_const_mask_onnx(model_path, value, dims=dims)
    return TFGridNetDenoiser(model_path=model_path, **kwargs)


# ── Flag AUS ⇒ keine Verhaltensänderung ───────────────────────────────────────


def test_flag_default_off_keeps_dfn_default():
    """§III.9 (copilot-instructions.md): DFN bleibt Default; Flag Default False."""
    assert music_model_flags.use_tf_gridnet_quality is False


def test_stage_passthrough_identical_object_when_flag_off():
    audio = _tone()
    result, used = apply_tf_gridnet_quality_stage(audio, SR)
    assert result is audio  # identisches Objekt — keine Verhaltensänderung
    assert used is False


def test_stage_flag_off_never_touches_denoiser():
    class _Bomb:
        def is_available(self):
            raise AssertionError("Stage darf bei Flag AUS nicht auf den Denoiser zugreifen")

        def denoise(self, audio, sample_rate, **kw):
            raise AssertionError("Stage darf bei Flag AUS nicht inferieren")

    audio = _tone()
    result, used = apply_tf_gridnet_quality_stage(audio, SR, denoiser=_Bomb())
    assert result is audio
    assert used is False


# ── Fallback-Verträge (§V6 (copilot-instructions.md)) ─────────────────────────


def test_unavailable_without_model_warns_and_returns_none(tmp_path, caplog):
    den = TFGridNetDenoiser(model_path=tmp_path / "missing.onnx")
    assert den.is_available() is False
    assert den.denoise(_tone(), SR) is None
    assert _has_warning(caplog, "nicht gefunden")


def test_broken_onnx_file_falls_back_unavailable(tmp_path, caplog):
    path = tmp_path / "tf_gridnet_quality.onnx"
    path.write_bytes(b"kein ONNX")
    den = TFGridNetDenoiser(model_path=path)
    assert den.is_available() is False
    assert _has_warning(caplog, "nicht ladbar")


def test_contract_mismatch_static_dims_rejected(tmp_path, caplog):
    den = _make_denoiser(tmp_path, value=1.0, dims=(1, 17, 100))
    assert den.is_available() is False
    assert _has_warning(caplog, "Vertrags-Bruch")


def test_stage_flag_on_unavailable_falls_back_with_warning(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(music_model_flags, "use_tf_gridnet_quality", True)
    den = TFGridNetDenoiser(model_path=tmp_path / "missing.onnx")
    audio = _tone()
    result, used = apply_tf_gridnet_quality_stage(audio, SR, denoiser=den)
    assert used is False
    assert result is audio
    assert _has_warning(caplog, "Fallback")


def test_stage_flag_on_error_falls_back_with_warning(monkeypatch, caplog):
    monkeypatch.setattr(music_model_flags, "use_tf_gridnet_quality", True)

    class _Broken:
        def is_available(self):
            return True

        def denoise(self, audio, sample_rate, **kw):
            raise RuntimeError("Boom")

    audio = _tone()
    result, used = apply_tf_gridnet_quality_stage(audio, SR, denoiser=_Broken())
    assert used is False
    assert result is audio
    assert _has_warning(caplog, "fehlgeschlagen")


def test_stage_uses_denoiser_and_reports_used(monkeypatch):
    monkeypatch.setattr(music_model_flags, "use_tf_gridnet_quality", True)

    class _Stub:
        def is_available(self):
            return True

        def denoise(self, audio, sample_rate, **kw):
            return np.asarray(audio, dtype=np.float32) * 0.5

    audio = _tone()
    result, used = apply_tf_gridnet_quality_stage(audio, SR, denoiser=_Stub())
    assert used is True
    np.testing.assert_allclose(result, audio * 0.5)


# ── Shapes + Layout (C, N)-Invariante (AGENTS.md §3) ──────────────────────────


@pytest.mark.parametrize("shape", [(SR,), (2, SR), (SR, 2)])
def test_denoise_preserves_shape_layout_and_finiteness(tmp_path, shape):
    den = _make_denoiser(tmp_path, value=1.0)
    rng = np.random.RandomState(3)
    x = (0.3 * rng.randn(*shape)).astype(np.float32)
    out = den.denoise(x, SR)
    assert out is not None
    assert out.shape == x.shape
    assert out.dtype == np.float32
    assert np.isfinite(out).all()
    assert float(np.sqrt(np.mean(out**2))) > 1e-6  # nicht stumm


def test_layout_transpose_equivalence(tmp_path):
    """(2, N) und (N, 2) liefern bit-identische Kanal-Ergebnisse (C, N)."""
    den = _make_denoiser(tmp_path, value=0.8)
    x = _stereo_pair(SR)
    out_cn = den.denoise(x, SR)  # (2, N)
    out_nc = den.denoise(x.T, SR)  # (N, 2)
    assert out_cn.shape == (2, SR)
    assert out_nc.shape == (SR, 2)
    np.testing.assert_array_equal(out_nc.T, out_cn)


# ── Determinismus (§G5 (copilot-instructions.md)) ─────────────────────────────


def test_determinism_two_runs_bit_identical(tmp_path):
    den = _make_denoiser(tmp_path, value=0.7)
    x = _stereo_pair(240_000)  # > 4 s ⇒ Overlap-Add-Mehrfensterpfad
    r1 = den.denoise(x, SR)
    r2 = den.denoise(x, SR)
    np.testing.assert_array_equal(r1, r2)


def test_determinism_fresh_instance_bit_identical(tmp_path):
    x = _stereo_pair(240_000)
    r1 = _make_denoiser(tmp_path, value=0.7).denoise(x, SR)
    r2 = _make_denoiser(tmp_path, value=0.7).denoise(x, SR)
    np.testing.assert_array_equal(r1, r2)


def test_identity_mask_preserves_signal(tmp_path):
    den = _make_denoiser(tmp_path, value=1.0)
    x = _tone_noisy()
    out = den.denoise(x, SR)
    assert out is not None
    np.testing.assert_allclose(out, x, atol=1e-4)  # STFT-Roundtrip


# ── Never-worsen-Harness (§v10.26) ────────────────────────────────────────────


def test_never_worsen_reverts_silent_candidate(tmp_path, caplog):
    """Null-Maske ⇒ Kandidat stumm ⇒ bit-identischer Rücksprung + Warnung."""
    den = _make_denoiser(tmp_path, value=0.0)
    x = _tone_noisy()
    out = den.denoise(x, SR)
    np.testing.assert_array_equal(out, x)  # bit-identisch
    assert _has_warning(caplog, "Never-worsen")


def test_never_worsen_select_accepts_equal_candidate():
    x = _tone()
    cand = x * 1.1
    chosen = never_worsen_select(x, cand, SR, scorer=lambda m, s: 5.0)
    np.testing.assert_array_equal(chosen, cand)  # gleicher Score ⇒ akzeptiert


def test_never_worsen_select_reverts_worse_candidate(caplog):
    x = _tone(amp=0.5)
    cand = x * 0.5

    def scorer(mono, sample_rate):
        rms = float(np.sqrt(np.mean(np.asarray(mono) ** 2)))
        return 10.0 if rms > 0.2 else 0.0

    chosen = never_worsen_select(x, cand, SR, scorer=scorer)
    np.testing.assert_array_equal(chosen, x)  # bit-identischer Rücksprung
    assert _has_warning(caplog, "Never-worsen")


def test_never_worsen_harness_battery_default_scorer(tmp_path):
    """Harness: gedämpfter Kandidat ⇒ Ergebnis nie schlechter als die Eingabe."""
    den = _make_denoiser(tmp_path, value=0.5)  # Kandidat dämpft ⇒ Guard muss greifen
    rng = np.random.RandomState(11)
    signals = [
        _tone(),
        _tone_noisy(),
        (0.02 * rng.randn(SR)).astype(np.float32),
        np.clip(_tone(amp=1.2), -1.0, 1.0).astype(np.float32),
    ]
    for x in signals:
        out = den.denoise(x, SR)
        assert out is not None
        assert default_quality_scorer(out, SR) >= default_quality_scorer(x, SR) - NEVER_WORSEN_MARGIN


# ── NaN/Inf-Schutz (§III.6 (copilot-instructions.md)) ─────────────────────────


def test_nan_inf_input_guard_finite_output(tmp_path):
    den = _make_denoiser(tmp_path, value=1.0)
    x = _tone()
    x[100] = np.nan
    x[200] = np.inf
    out = den.denoise(x, SR)
    assert out is not None
    assert np.isfinite(out).all()


# ── Torch-Architektur: Shapes, Determinismus, Nicht-Kausalität ────────────────


def test_torch_model_shapes_determinism_and_noncausality():
    torch = pytest.importorskip("torch")
    model = build_tf_gridnet_model(n_freq=17, hidden=8, n_blocks=2)
    model.eval()
    x = torch.randn(1, 17, 12)
    with torch.no_grad():
        y1 = model(x)
        y2 = model(x)
    assert tuple(y1.shape) == tuple(x.shape)
    assert torch.isfinite(y1).all()
    assert torch.equal(y1, y2)  # §G5 (copilot-instructions.md)
    # Nicht-Kausalität: späte Frames verändern die Ausgabe früher Frames.
    # Kurze Sequenz: der rekurrente Einfluss zerfällt über lange Distanzen unter
    # die Float-Toleranz — der Nachweis misst deshalb auf kurzer Distanz.
    x2 = x.clone()
    x2[:, :, -4:] += 2.0
    with torch.no_grad():
        y3 = model(x2)
    assert not torch.allclose(y1[:, :, :2], y3[:, :, :2])


# ── Restoration-Pfad (Phase 03): Flag-Gate + §V6-Fallback ─────────────────────


def test_phase_flag_off_stage_never_runs(monkeypatch):
    """Flag AUS ⇒ der Phase-03-Hook ruft die Stage nie auf — keine Verhaltensänderung."""
    calls: list = []

    def _bomb(*args, **kwargs):
        calls.append(args)
        raise AssertionError("Stage darf bei Flag AUS nicht laufen")

    monkeypatch.setattr("backend.core.dsp.tf_gridnet_denoiser.apply_tf_gridnet_quality_stage", _bomb)
    from backend.core.phases.phase_03_denoise import DenoisePhase

    result = DenoisePhase(sample_rate=SR).process(_tone_noisy(), sample_rate=SR, material_type="vinyl")
    assert calls == []
    assert np.isfinite(result.audio).all()


def test_phase_flag_on_error_keeps_dfn_result_with_warning(monkeypatch, caplog):
    """§V6 (copilot-instructions.md): Fehler hinter dem Flag ⇒ Warnung + DFN/Bestand bleibt."""
    monkeypatch.setattr(music_model_flags, "use_tf_gridnet_quality", True)

    def _boom(*args, **kwargs):
        raise RuntimeError("Boom")

    monkeypatch.setattr("backend.core.dsp.tf_gridnet_denoiser.apply_tf_gridnet_quality_stage", _boom)
    from backend.core.phases.phase_03_denoise import DenoisePhase

    result = DenoisePhase(sample_rate=SR).process(_tone_noisy(), sample_rate=SR, material_type="vinyl")
    assert np.isfinite(result.audio).all()
    assert _has_warning(caplog, "TF-GridNet")


# ── Trainer: Gradient-Akkumulation (Speicherbudget 24 GiB, Befund 2026-11-06) ──


def _trainer_lauf(tmp_path, monkeypatch, *, batch_size: int, grad_accum: int, steps: int) -> dict:
    """Kurz-Training auf synthetischen Paaren (CPU), Checkpoints in tmp_path."""
    import torch

    from scripts import train_tf_gridnet_denoiser as trainer

    latest = tmp_path / f"latest_b{batch_size}_a{grad_accum}.pt"
    best = tmp_path / f"best_b{batch_size}_a{grad_accum}.pt"
    monkeypatch.setattr(trainer, "LATEST_PT", latest)
    monkeypatch.setattr(trainer, "BEST_PT", best)
    monkeypatch.setattr(trainer, "CHECKPOINT_DIR", tmp_path)

    trainer.train(
        epochs=1,
        batch_size=batch_size,
        lr=1e-3,
        steps_per_epoch=steps,
        resume=None,
        seed=42,
        clean_dir=None,
        noisy_dir=None,
        synthetic=4,
        device="cpu",
        hidden=8,
        blocks=1,
        skip_onnx=True,
        grad_accum=grad_accum,
    )
    return torch.load(latest, map_location="cpu", weights_only=False)["model_state_dict"]


def test_grad_accum_lehnt_null_ab(tmp_path) -> None:
    """``--grad-accum 0`` darf nicht still zu „kein Optimizer-Schritt“ werden."""
    from scripts import train_tf_gridnet_denoiser as trainer

    with pytest.raises(ValueError):
        trainer.train(
            epochs=1,
            batch_size=1,
            lr=1e-3,
            steps_per_epoch=1,
            resume=None,
            seed=42,
            clean_dir=None,
            noisy_dir=None,
            synthetic=2,
            device="cpu",
            hidden=4,
            blocks=1,
            skip_onnx=True,
            grad_accum=0,
        )


def test_grad_accum_entspricht_grossem_batch(tmp_path, monkeypatch) -> None:
    """Micro-Batch 1 × Akkumulation 2 == Batch 2 (gleiche Samples, gleiche Schritte)."""
    import torch

    gross = _trainer_lauf(tmp_path, monkeypatch, batch_size=2, grad_accum=1, steps=2)
    # steps zaehlt Optimizer-Schritte: 2 × Akkumulation 2 = 4 Micro-Batches mit
    # Batch 1 — dieselben 4 Samples wie 2 Optimizer-Schritte mit Batch 2.
    klein = _trainer_lauf(tmp_path, monkeypatch, batch_size=1, grad_accum=2, steps=2)

    assert set(gross) == set(klein)
    for key in gross:
        assert torch.allclose(gross[key], klein[key], atol=1e-5, rtol=1e-4), (
            f"Gewicht {key} weicht ab — Akkumulation ist nicht äquivalent"
        )
