"""D-K3-38 Phase-12-Teilbehebung (2026-10-09): gebündelter Flutter-Fit + Kalman-JIT.

Pinnt die Wirkung des Flutter-Scans (Zeugen-Gate, 6-Hz-Synthetik) und die
Verdrahtung der gebündelten Normalengleichungen (np.linalg.solve gestapelt;
lstsq nur noch als Singular-Fallback). Hintergrund: Die Phase-12-Zerlegung
(Register D-K3-38) maß 37 677 Einzel-lstsq-SVD-Aufrufe (2,63 s je 15-s-Lauf)
im Flutter-Scan — ersetzt durch den Batch-Solve; der direkte A/B gegen die
Vorversion (git HEAD) war auf synthetischem Flutter bit-identisch
(virtual_pitch max|Δ| = 0.0, Rauschfall identisch verworfen).
"""

from __future__ import annotations

import inspect

import numpy as np

from backend.core.phases.phase_12_wow_flutter_fix import WowFlutterFix

SR = 48000


def _synthetic_flutter_stack(*, seed: int = 23, with_flutter: bool = True) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = SR  # 1 s
    b = 41
    t = np.arange(n) / SR
    dev = rng.standard_normal((b, n)) * 3.0
    if with_flutter:
        dev[9] += 22.9 * np.sin(2.0 * np.pi * 6.0 * t)
        dev[17] += 8.0 * np.sin(2.0 * np.pi * 6.0 * t + 0.7)
    val = np.ones((b, n), dtype=np.float64)
    return dev, val


def test_flutter_track_detects_synthetic_6hz_flutter() -> None:
    """6-Hz-Flutter (22,9 cents) passiert das Zeugen-Gate und wird als Pitch-Track erkannt."""
    dev, val = _synthetic_flutter_stack()
    vp, conf = WowFlutterFix()._flutter_track_from_stacks(dev, val, SR)
    assert vp.shape == (200,)  # 1 s auf dem 200-Hz-Raster
    amp_cents = 1200.0 * np.log2(float(np.max(vp)) / 440.0)
    assert 20.0 < amp_cents < 26.0, f"Peak-Amplitude unerwartet: {amp_cents:.2f} cents"
    assert float(np.min(conf)) > 0.5
    # 6-Hz-Sinusstruktur: ≈ 6 lokale Maxima je Sekunde auf dem 200-Hz-Raster.
    peaks = int(np.sum((vp[1:-1] > vp[:-2]) & (vp[1:-1] > vp[2:])))
    assert 5 <= peaks <= 7, f"Modulationsfrequenz unerwartet: {peaks} Maxima/s"


def test_flutter_track_rejects_pure_noise() -> None:
    """Nur Rauschen ⇒ flache Spur (440.0, conf 0) — „keine signifikante Modulation“."""
    dev, val = _synthetic_flutter_stack(seed=24, with_flutter=False)
    vp, conf = WowFlutterFix()._flutter_track_from_stacks(dev, val, SR)
    assert np.all(vp == 440.0)
    assert not np.any(conf)


def test_flutter_fit_uses_batched_solve_with_lstsq_fallback_only() -> None:
    """Wiring-Pin: Hot-Path nutzt np.linalg.solve (gebündelt); lstsq nur im Singular-Fallback.

    §D-K3-38 (Register): Ein Rückfall auf den 37 677-Einzel-lstsq-Pfad senkt die
    Phase-12-Zeit messbar (gemessen +2,6 s je 15-s-Lauf, cProfile 2026-10-09) —
    dieser Pin bricht, sobald der Batch-Solve aus dem Rumpf verschwindet.
    """
    src = inspect.getsource(WowFlutterFix._flutter_track_from_stacks)
    assert src.count("np.linalg.lstsq") == 1, "lstsq darf nur noch im Singular-Fallback stehen"
    assert "np.linalg.solve(_N, _rhs)" in src
    assert 'np.einsum("nc,bn,nd->bcd"' in src


def test_flutter_batched_normal_equations_match_lstsq_formula() -> None:
    """Formel-Pin: gestapelte Normalengleichungen == gewichtete lstsq je Band (Referenz)."""
    rng = np.random.default_rng(31)
    n = 1500
    t = np.arange(n) / 200.0
    des = np.column_stack([np.sin(2 * np.pi * 6.0 * t), np.cos(2 * np.pi * 6.0 * t), np.ones(n)])
    w = rng.uniform(0.0, 1.0, size=(17, n))
    y = rng.standard_normal((17, n))
    nrm = np.einsum("nc,bn,nd->bcd", des, w, des, optimize=True)
    rhs = np.einsum("nc,bn,bn->bc", des, w, y, optimize=True)
    cb_batched = np.linalg.solve(nrm, rhs)
    for b in range(17):
        sw = np.sqrt(w[b])
        cb_ref, *_ = np.linalg.lstsq(des * sw[:, None], y[b] * sw, rcond=None)
        np.testing.assert_allclose(cb_batched[b], cb_ref, rtol=1e-8, atol=1e-10)


def test_subband_scan_detects_coherent_modulation() -> None:
    """§D-K3-38: kohärente 1,1-Hz-Modulation (10 in-phase Bänder, 6 cents) wird erkannt.

    A/B gegen die Vorversion (git HEAD, 10.12.37-Stand): Frequenz-Δ = 3,9e-14 Hz
    bei 7,4× Speedup auf der Synthetik; E2E-Audio bit-identisch.
    """
    rng = np.random.default_rng(41)
    sr = 48000
    n = 96000  # 2 s → 40 Blöcke auf dem 20-Hz-Raster (Gate: ≥ 32)
    b = 41
    t = np.arange(n) / sr
    dev = rng.standard_normal((b, n)) * 3.0
    for k in range(10):
        dev[k] = 6.0 * np.sin(2.0 * np.pi * 1.1 * t)
    val = np.ones((b, n), dtype=np.float64)
    f_peak = WowFlutterFix._scan_subband_modulation_frequency(dev, val, sr)
    assert f_peak is not None
    assert abs(f_peak - 1.1) < 0.02, f"Kohärenz-Peak unerwartet: {f_peak:.4f} Hz"


def test_subband_scan_uses_batched_solve_with_lstsq_fallback_only() -> None:
    """Wiring-Pin: Auch der Subband-Scan rechnet gebündelt; lstsq nur im Singular-Ausweichpfad.

    §D-K3-38 (Register): Der Einzel-lstsq-Pfad kostete gemessen 15 619 Aufrufe
    / ~1,4 s je 15-s-Lauf (cProfile 2026-10-09); dieser Pin bricht, sobald der
    Batch-Solve aus dem Rumpf verschwindet.
    """
    src = inspect.getsource(WowFlutterFix._scan_subband_modulation_frequency)
    assert src.count("np.linalg.lstsq") == 1, "lstsq darf nur noch im Singular-Ausweichpfad stehen"
    assert "np.linalg.solve(_N, _rhs)" in src
    assert 'np.einsum("nc,bn,nd->bcd"' in src


def test_flutter_singular_fallback_matches_batched(monkeypatch) -> None:
    """Audit-Finding 2026-10-09 (Fix 10.12.39): Der bisher NIE ausgeführte lstsq-Ausweichpfad
    des Flutter-Fits ist wirkungsgeprüft — er liefert bit-identische Ergebnisse (simuliert
    durch werfendes np.linalg.solve).
    """
    dev, val = _synthetic_flutter_stack()
    ph = WowFlutterFix()
    ref_vp, ref_conf = ph._flutter_track_from_stacks(dev.copy(), val.copy(), SR)

    def _boom(_nrm, _rhs, *_a, **_k):
        raise np.linalg.LinAlgError("simuliert singuläre Normalengleichung (Test)")

    monkeypatch.setattr(np.linalg, "solve", _boom)
    got_vp, got_conf = ph._flutter_track_from_stacks(dev.copy(), val.copy(), SR)
    assert np.array_equal(ref_vp, got_vp), "Ausweichpfad muss bit-identisch zum Batch-Solve sein"
    assert np.array_equal(ref_conf, got_conf)


def test_subband_singular_fallback_matches_batched(monkeypatch) -> None:
    """Audit-Finding 2026-10-09 (Fix 10.12.39): Der Subband-Ausweichpfad liefert denselben Peak.

    Ehrlich benannte Grenze: Der lstsq-Ausweichpfad ist NICHT bit-gleich zum
    Batch-Solve (3×3-Inversions-Reihenfolge LAPACK-LU vs. lstsq-SVD) — gemessen
    Δ = 3,4e-14 Hz auf der Parabel-Verfeinerung (≪ 0,002-Hz-Feinraster, keine
    Hörrelevanz). Der Pin fordert < 1e-12 Hz.
    """
    rng = np.random.default_rng(41)
    sr = 48000
    n = 96000
    b = 41
    t = np.arange(n) / sr
    dev = rng.standard_normal((b, n)) * 3.0
    for k in range(10):
        dev[k] = 6.0 * np.sin(2.0 * np.pi * 1.1 * t)
    val = np.ones((b, n), dtype=np.float64)
    ref = WowFlutterFix._scan_subband_modulation_frequency(dev.copy(), val.copy(), sr)

    def _boom(_nrm, _rhs, *_a, **_k):
        raise np.linalg.LinAlgError("simuliert singulär (Test)")

    monkeypatch.setattr(np.linalg, "solve", _boom)
    got = WowFlutterFix._scan_subband_modulation_frequency(dev.copy(), val.copy(), sr)
    assert ref is not None and got is not None
    assert abs(ref - got) < 1e-12, f"Subband-Ausweichpfad weicht ab: Δ = {abs(ref - got):.3e} Hz (> 1e-12)"
