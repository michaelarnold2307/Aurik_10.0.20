"""SOTA-M8 (D-K4-1) — Vertragstests des neuronalen A-SPADE-Klassen-Declipper (CHALLENGER zu Phase 07).

Verträge (Auftrag SOTA-M8 D-K4-1):
- Architektur-Shapes (A-SPADE-Klasse: unrolled ADMM + 1-D-CNN-Proximal)
- Determinismus: 2× gleicher Input ⇒ bit-identisch (§G5 (copilot-instructions.md));
  Seed-Fähigkeit: gleicher Seed ⇒ gleiche Gewichte/gleicher Output
- Never-worsen-Harness auf geclipptem Signal (Orakel akzeptiert,
  Verschlechterung bit-identisch abgelehnt)
- Stereo-Layout-Invariante: (2, N) ⇄ (N, 2) bit-identisch (AGENTS.md §3)
- Flag `use_aspade_declipper` AUS ⇒ bit-identisches Bestandsverhalten
  (Challenger wird nie gerufen; keine neuen Metrik-Schlüssel)
- Fehlerpfade: logger.warning + Ersatzpfad (§V6 (copilot-instructions.md)),
  kein stiller Fallback; ungültige Argumente werfen ValueError (laut)
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
import torch

import backend.core.music_model_flags as mmf
from backend.core.dsp import aspade_declipper as asp
from backend.core.dsp.aspade_declipper import AspadeClassDeclipper, AspadeClassNet, AspadeProximalNet
from backend.core.phases.phase_07_declipper import DeclipperPhase

SR = 48000


def _bursts_with_clean(amp: float = 6.0, dur_s: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """0.8-Sinus mit drei 25-ms-Bursts ×amp — (clean, hart bei ±1.0 geclippt).

    MUSTER: tests/unit/test_phase_07_declipper.py::_clipped_bursts; hier mit
    Ground-Truth-clean für den Never-worsen-Harness (SDR gegen clean).
    """
    n = int(SR * dur_s)
    t = np.linspace(0, 1, n, endpoint=False, dtype=np.float64)
    x = np.sin(2 * np.pi * 440 * t) * 0.8
    burst = int(SR * 0.025)
    for onset_s in (0.25, 0.5, 0.75):
        s0 = int(onset_s * n)
        x[s0 : s0 + burst] *= amp
    clean = x.astype(np.float32)
    clipped = np.clip(x, -1.0, 1.0).astype(np.float32)
    return clean, clipped


def _sdr_db(est: np.ndarray, ref: np.ndarray) -> float:
    """Test-Harness-Metrik (unabhängig vom Modul-Witness): SDR gegen Ground Truth."""
    err = float(np.mean((est.astype(np.float64) - ref.astype(np.float64)) ** 2))
    sig = float(np.mean(ref.astype(np.float64) ** 2))
    return 10.0 * np.log10((sig + 1e-12) / (err + 1e-12))


def _thd(y: np.ndarray) -> float:
    """THD-Verhältnis (Harmonische 2–10 vs. Grundton 440 Hz) — MUSTER test_phase_07_declipper."""
    spec = np.abs(np.fft.rfft(y.astype(np.float64)))
    freqs = np.fft.rfftfreq(len(y), 1.0 / SR)
    fund = float(spec[np.argmin(np.abs(freqs - 440))]) + 1e-12
    harm = 0.0
    for h in range(2, 11):
        i = int(np.argmin(np.abs(freqs - 440 * h)))
        harm += float(spec[i]) ** 2
    return float(np.sqrt(harm) / fund)


# ── Architektur-Shapes ──────────────────────────────────────────────────────


def test_proximal_net_shape_contract() -> None:
    """T_θ (1-D-CNN-Proximal): (B, 1, T) ⇒ (B, 1, T), Parameter vorhanden."""
    prox = AspadeProximalNet(channels=8, kernel_size=7, dilations=(1, 2, 4))
    x = torch.linspace(-1.0, 1.0, 2 * 1 * 2048).reshape(2, 1, 2048)
    out = prox(x)
    assert out.shape == x.shape
    assert sum(p.numel() for p in prox.parameters()) > 0
    assert bool(torch.isfinite(out).all())


def test_klassennetz_shape_und_stufen() -> None:
    """Unrolled ADMM: Stufenzahl == K, Forward (B, 1, T) ⇒ (B, 1, T) (auch ungerade T)."""
    net = AspadeClassNet(stages=3, channels=8, kernel_size=7, dilations=(1, 2, 4))
    assert len(net.stages) == 3
    y = torch.linspace(-1.0, 1.0, 1 * 1 * 2049).reshape(1, 1, 2049)
    theta = torch.tensor([[[0.5]]], dtype=torch.float32)
    out = net(y, theta)
    assert out.shape == y.shape
    assert bool(torch.isfinite(out).all())


def test_klassennetz_datenkonsistenz_ungeclippt() -> None:
    """Ungeclippte Samples bleiben exakt auf dem Eingang (Π_Γ-Projektion)."""
    net = AspadeClassNet(stages=2, channels=8, kernel_size=7, dilations=(1, 2))
    y = torch.linspace(-0.4, 0.4, 1024).reshape(1, 1, 1024)  # |y| < θ=0.5
    theta = torch.tensor([[[0.5]]], dtype=torch.float32)
    out = net(y, theta)
    torch.testing.assert_close(out, y, rtol=0.0, atol=0.0)  # exakt (bit-identisch)


# ── Determinismus (§G5 (copilot-instructions.md)) ───────────────────────────


def test_build_model_seed_reproduzierbar() -> None:
    """Gleicher Seed ⇒ bit-identische Gewichte; anderer Seed ⇒ andere Gewichte."""
    a = asp.build_model(stages=2, channels=8, seed=7)
    b = asp.build_model(stages=2, channels=8, seed=7)
    c = asp.build_model(stages=2, channels=8, seed=8)
    sd_a, sd_b, sd_c = a.state_dict(), b.state_dict(), c.state_dict()
    for key in sd_a:
        assert torch.equal(sd_a[key], sd_b[key]), f"Gewichte weichen bei {key} ab"
    assert any(not torch.equal(sd_a[key], sd_c[key]) for key in sd_a)


def test_inferenz_zwei_laeufe_bit_identisch() -> None:
    """2× gleicher Input ⇒ bit-identischer Output (§G5 (copilot-instructions.md))."""
    _, clipped = _bursts_with_clean(6.0)
    d = AspadeClassDeclipper(stages=2, channels=8, seed=5)
    r1 = d.declip(clipped, SR, 1.0)
    r2 = d.declip(clipped, SR, 1.0)
    np.testing.assert_array_equal(r1, r2)
    assert r1.shape == clipped.shape
    assert r1.dtype == clipped.dtype


def test_seed_faehigkeit_gleicher_seed_gleicher_output() -> None:
    """Zwei Instanzen mit demselben Seed liefern bit-identische Ergebnisse."""
    _, clipped = _bursts_with_clean(6.0)
    d1 = AspadeClassDeclipper(stages=2, channels=8, seed=11)
    d2 = AspadeClassDeclipper(stages=2, channels=8, seed=11)
    np.testing.assert_array_equal(d1.declip(clipped, SR, 1.0), d2.declip(clipped, SR, 1.0))


# ── Never-worsen-Harness auf geclipptem Signal ──────────────────────────────


def test_never_worsen_harness_geclippt() -> None:
    """Der Kandidat darf das Signal nie verschlechtern (SDR gegen Ground Truth)."""
    clean, clipped = _bursts_with_clean(6.0)
    d = AspadeClassDeclipper(stages=2, channels=8, seed=5)
    out = d.declip(clipped, SR, 1.0)
    assert np.isfinite(out).all()
    assert out.shape == clipped.shape
    assert _sdr_db(out, clean) >= _sdr_db(clipped, clean) - 1e-9  # Never-worsen


def test_never_worsen_akzeptiert_orakel(monkeypatch) -> None:
    """Orakel-Kandidat (clean) wird übernommen — der Guard akzeptiert Gewinne."""
    clean, clipped = _bursts_with_clean(6.0)
    d = AspadeClassDeclipper(stages=2, channels=8, seed=5)
    torch_clean = torch.from_numpy(clean.astype(np.float32)).reshape(1, 1, -1)
    monkeypatch.setattr(d, "_net", lambda y_t, theta_t: torch_clean)
    out = d.declip(clipped, SR, 1.0)
    np.testing.assert_array_equal(out, clean)  # Orakel bit-identisch übernommen
    assert _sdr_db(out, clean) > _sdr_db(clipped, clean) + 20.0


def test_never_worsen_lehnt_verschlechterung_ab(monkeypatch, caplog) -> None:
    """Verschlechternder Kandidat wird abgelehnt — Eingang bleibt bit-identisch.

    Harness-Präzedenz: der Kandidat muss den Harmonik-Witness GARANTIERT
    erhöhen. Gewählt wird die 6. Harmonische (2640 Hz = 6 × 440) — außerhalb
    der Grundtonsuche (60–2000 Hz), aber in den gezählten Harmonischen (2–10)
    des Witnesses; ein Kandidat, der den Grundton selbst verschiebt (z. B.
    1320 Hz), würde den heuristischen f0-Suchlauf austricksen.
    """
    _, clipped = _bursts_with_clean(6.0)
    d = AspadeClassDeclipper(stages=2, channels=8, seed=5)
    t = torch.arange(clipped.size, dtype=torch.float32).reshape(1, 1, -1) / SR
    harmonic = 0.9 * torch.sin(2 * np.pi * 2640.0 * t)

    def _worse(y_t: torch.Tensor, theta_t: torch.Tensor) -> torch.Tensor:
        return y_t + harmonic

    # Präzedenz des Harness: der Stub-Kandidat ist per Witness messbar schlechter.
    cand_np = clipped + harmonic.numpy().reshape(-1)
    assert asp.harmonic_distortion_witness(cand_np, SR) > asp.harmonic_distortion_witness(clipped, SR)

    monkeypatch.setattr(d, "_net", _worse)
    with caplog.at_level(logging.INFO, logger="backend.core.dsp.aspade_declipper"):
        out = d.declip(clipped, SR, 1.0)
    np.testing.assert_array_equal(out, clipped)  # bit-identischer Passthrough
    assert any("abgelehnt" in r.getMessage() for r in caplog.records)


# ── Stereo-Layout-Invariante (C, N) (AGENTS.md §3) ─────────────────────────


def test_layout_stereo_2n_und_n2_bit_identisch() -> None:
    """(2, N) und (N, 2) liefern transponiert bit-identische Ergebnisse."""
    clean0, clipped0 = _bursts_with_clean(6.0)
    clean1, clipped1 = _bursts_with_clean(2.5)
    x_2n = np.stack([clipped0, clipped1])  # (2, N)
    x_n2 = np.ascontiguousarray(x_2n.T)  # (N, 2)
    d = AspadeClassDeclipper(stages=2, channels=8, seed=5)
    r_2n = d.declip(x_2n, SR, 1.0)
    r_n2 = d.declip(x_n2, SR, 1.0)
    assert r_2n.shape == x_2n.shape
    assert r_n2.shape == x_n2.shape
    np.testing.assert_array_equal(r_2n, r_n2.T)  # Layout-invariant, bit-identisch


def test_layout_mono_shape_und_dtype() -> None:
    """(N,) float64 ⇒ (N,) float64; (1, N) ⇒ (1, N)."""
    _, clipped = _bursts_with_clean(6.0)
    d = AspadeClassDeclipper(stages=2, channels=8, seed=5)
    out1 = d.declip(clipped.astype(np.float64), SR, 1.0)
    assert out1.shape == clipped.shape
    assert out1.dtype == np.float64
    out2 = d.declip(clipped.reshape(1, -1), SR, 1.0)
    assert out2.shape == (1, clipped.size)
    np.testing.assert_array_equal(out2[0], out1.astype(np.float32))


# ── Flag-Gate: use_aspade_declipper AUS ⇒ bit-identisches Bestandsverhalten ─


def test_flag_default_aus() -> None:
    """Flag ist aus (music_model_flags.py) — Challenger-Pfad läuft nie ohne Sign-off."""
    assert mmf.use_aspade_declipper is False


def test_flag_aus_challenger_nie_gerufen_und_bit_identisch(monkeypatch) -> None:
    """Flag AUS: die Phase 07 ruft den Challenger nie, Ausgabe/Metrik unverändert."""
    _, x = _bursts_with_clean(6.0)
    ref = DeclipperPhase().process(x, sample_rate=SR)

    calls: list[int] = []

    def _bomb(*_args, **_kwargs):
        calls.append(1)
        raise AssertionError("Challenger darf bei Flag AUS nicht gerufen werden")

    monkeypatch.setattr(asp, "declip_aspade_class", _bomb)
    out = DeclipperPhase().process(x, sample_rate=SR)

    assert calls == []  # nie aufgerufen
    np.testing.assert_array_equal(out.audio, ref.audio)  # bit-identisch
    assert out.metrics == ref.metrics  # keine neuen Metrik-Schlüssel
    assert "aspade_class_used" not in out.metrics


def test_flag_an_abstinenz_bit_identisch_zu_flag_aus(monkeypatch) -> None:
    """Flag AN, Challenger abstinent (None): PCHIP-Kaskade bleibt bit-identisch."""
    _, x = _bursts_with_clean(6.0)
    ref = DeclipperPhase().process(x, sample_rate=SR)

    monkeypatch.setattr(mmf, "use_aspade_declipper", True)
    monkeypatch.setattr(asp, "declip_aspade_class", lambda *_a, **_k: None)
    out = DeclipperPhase().process(x, sample_rate=SR)

    np.testing.assert_array_equal(out.audio, ref.audio)
    assert out.metrics["aspade_class_used"] is False


def test_flag_an_orakel_wird_uebernommen(monkeypatch) -> None:
    """Flag AN, gewinnender Kandidat: Challenger übernimmt, Harmonik-Proxy sinkt."""
    clean, clipped = _bursts_with_clean(6.0)
    monkeypatch.setattr(mmf, "use_aspade_declipper", True)
    monkeypatch.setattr(
        asp,
        "declip_aspade_class",
        lambda audio, sample_rate, clip_threshold: clean.astype(np.float64).copy(),
    )
    out = DeclipperPhase().process(clipped, sample_rate=SR)
    assert out.metrics["aspade_class_used"] is True
    assert _thd(out.audio) < _thd(clipped)


def test_flag_an_fehler_pchip_kaskade_mit_warnung(monkeypatch, caplog) -> None:
    """Flag AN, Fehler im Challenger: §V6-Warnung + PCHIP-Kaskade bleibt bit-identisch (§V6, copilot-instructions.md)."""
    _, x = _bursts_with_clean(6.0)
    ref = DeclipperPhase().process(x, sample_rate=SR)

    monkeypatch.setattr(mmf, "use_aspade_declipper", True)

    def _boom(*_args, **_kwargs):
        raise RuntimeError("kaputt")

    monkeypatch.setattr(asp, "declip_aspade_class", _boom)
    with caplog.at_level(logging.WARNING, logger="backend.core.phases.phase_07_declipper"):
        out = DeclipperPhase().process(x, sample_rate=SR)

    np.testing.assert_array_equal(out.audio, ref.audio)  # PCHIP-Kaskade bleibt
    assert out.metrics["aspade_class_used"] is False
    warn_text = " ".join(r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING)
    assert "A-SPADE-Klassen-Declipper" in warn_text
    assert "PCHIP-Kaskade bleibt" in warn_text
    assert "§V6" in warn_text  # Textmarke des Logs (copilot-instructions.md)


# ── Fehlerpfade (§V6 (copilot-instructions.md): Warnung statt stiller Fallback) ─


def test_fassade_ohne_artefakt_none_mit_warnung(monkeypatch, caplog, tmp_path) -> None:
    """Ohne trainiertes Artefakt: None + §V6-Warnung — der Aufrufer fällt auf PCHIP zurück."""
    monkeypatch.setattr(asp, "DEFAULT_CHECKPOINT", tmp_path / "fehlt.pt")
    monkeypatch.setattr(asp, "_declipper_singleton", None)
    with caplog.at_level(logging.WARNING, logger="backend.core.dsp.aspade_declipper"):
        res = asp.declip_aspade_class(np.zeros(1024, dtype=np.float32), SR, 0.5)
    assert res is None
    assert any("§V6" in r.getMessage() for r in caplog.records)  # Textmarke des Logs (copilot-instructions.md)


def test_fehlerpfad_leeres_audio_warnung(caplog) -> None:
    """Leeres Audio: Warnung + unveränderter Passthrough (kein stilles Schweigen)."""
    d = AspadeClassDeclipper(stages=2, channels=8, seed=5)
    empty = np.zeros(0, dtype=np.float32)
    with caplog.at_level(logging.WARNING, logger="backend.core.dsp.aspade_declipper"):
        out = d.declip(empty, SR, 0.5)
    assert out.shape == (0,)
    assert any("leeres Audio" in r.getMessage() for r in caplog.records)


def test_fehlerpfad_nan_eingang_sanitisiert(caplog) -> None:
    """NaN/Inf im Eingang: Warnung + sanitisierte, finite Ausgabe (§0a/§V6, copilot-instructions.md)."""
    _, clipped = _bursts_with_clean(6.0)
    dirty = clipped.copy()
    dirty[100] = np.nan
    dirty[200] = np.inf
    d = AspadeClassDeclipper(stages=2, channels=8, seed=5)
    with caplog.at_level(logging.WARNING, logger="backend.core.dsp.aspade_declipper"):
        out = d.declip(dirty, SR, 1.0)
    assert out.shape == dirty.shape
    assert np.isfinite(out).all()
    assert any("NaN/Inf" in r.getMessage() for r in caplog.records)


def test_fehlerpfad_modelloutput_nicht_finit_warnung(monkeypatch, caplog) -> None:
    """Nicht-finites Modell-Ergebnis: §V6-Warnung + bit-identischer Eingangs-Passthrough."""
    _, clipped = _bursts_with_clean(6.0)
    d = AspadeClassDeclipper(stages=2, channels=8, seed=5)
    monkeypatch.setattr(d, "_net", lambda y_t, theta_t: torch.full_like(y_t, float("nan")))
    with caplog.at_level(logging.WARNING, logger="backend.core.dsp.aspade_declipper"):
        out = d.declip(clipped, SR, 1.0)
    np.testing.assert_array_equal(out, clipped)  # Eingang bleibt bit-identisch
    assert any("nicht-finites Modell-Ergebnis" in r.getMessage() for r in caplog.records)


def test_value_error_bei_falschen_argumenten() -> None:
    """Ungültige Argumente werfen laut ValueError — kein stilles Erraten."""
    d = AspadeClassDeclipper(stages=2, channels=8, seed=5)
    with pytest.raises(ValueError):
        d.declip(np.zeros((2, 2, 2), dtype=np.float32), SR, 0.5)  # falsches Layout
    with pytest.raises(ValueError):
        d.declip(np.zeros(16, dtype=np.int16), SR, 0.5)  # falscher dtype
    with pytest.raises(ValueError):
        d.declip(np.zeros(16, dtype=np.float32), SR, 0.0)  # θ muss > 0 sein
    with pytest.raises(ValueError):
        d.declip(np.zeros(16, dtype=np.float32), 0, 0.5)  # sample_rate muss > 0 sein


def test_checkpoint_laden_bit_identisch(tmp_path) -> None:
    """Trainiertes Artefakt (aspade_config + model_state_dict) lädt bit-identisch."""
    net = asp.build_model(stages=2, channels=8, dilations=(1, 2), seed=9)
    ckpt = tmp_path / "aspade_class_v1.pt"
    torch.save(
        {
            "aspade_config": {"stages": 2, "channels": 8, "kernel_size": 7, "dilations": (1, 2)},
            "model_state_dict": net.state_dict(),
        },
        ckpt,
    )
    d_loaded = AspadeClassDeclipper(checkpoint=ckpt)
    d_seeded = AspadeClassDeclipper(stages=2, channels=8, dilations=(1, 2), seed=9)
    _, clipped = _bursts_with_clean(6.0)
    np.testing.assert_array_equal(
        d_loaded.declip(clipped, SR, 1.0),
        d_seeded.declip(clipped, SR, 1.0),
    )
