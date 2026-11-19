"""§Witness-SOTA WF-V2: Akzeptanztests für die Spektral-Warp-Versorgung in phase_12.

Roadmap-Akzeptanz (docs/TODOS_SOTA_ROADMAP.md, SOTA-WF-V2):
Test mit synthetisch gewarptem Musik-Segment (bekannter Warp) —
Konsens-Trajektorie weicht < 10 % vom Soll ab.

Verdrahtung (phase_12_wow_flutter_fix.py):
- `_spectral_warp_supply_or_consensus()` schätzt F0-unabhängig (Log-f-Zentroid,
  Capstan-Prinzip) und versorgt den Zero-Consensus-Fall (flache pYIN/CREPE-
  Trajektorie) bzw. verfeinert per Konsens-Gate.
- Sie läuft EINMAL im Hauptfluss auf der Mono-Referenz (safe_to_mono) VOR dem
  M/S-Split — Mid/Side erhalten identische Faktoren (§2.51 L/R-Timing-
  Invariante, kein per-Kanal-Schätzer).
"""

from __future__ import annotations

import numpy as np

from backend.core.dsp.bandlimited_resampler import bandlimited_warp
from backend.core.dsp.warp_estimator import spectral_warp_estimate
from backend.core.phases.phase_12_wow_flutter_fix import WowFlutterFix

SR = 44100


def _ratio_fn(t: np.ndarray) -> np.ndarray:
    """Bekannte Wow-Warp-Ratio: 0,5 Hz, ±0,8 % — 4 s Signal = 2 volle Perioden
    (Median-Referenz des Schätzers liegt dann exakt bei 1,0)."""
    return 1.0 + 0.008 * np.sin(2 * np.pi * 0.5 * t)


def _harmonic_music_like(n_samples: int, seed: int = 42) -> np.ndarray:
    """Deterministisches, harmonisch reiches Musik-Segment (6 Teiltöne,
    leichte Verstimmung, langsame Amplituden-Modulation)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / SR
    x = np.zeros(n_samples, dtype=np.float64)
    amps = [0.5, 0.28, 0.22, 0.14, 0.10, 0.07]
    phases = rng.uniform(0, 2 * np.pi, len(amps))
    for k, (a, p) in enumerate(zip(amps, phases), start=1):
        x += a * np.sin(2 * np.pi * (220.0 * k) * t + p)
    x *= 0.6 + 0.4 * np.sin(2 * np.pi * 0.37 * t)
    return (0.5 * x / np.max(np.abs(x))).astype(np.float32)


def _apply_known_warp(x: np.ndarray) -> np.ndarray:
    """Wendet die bekannte, zeitvariierende Warp-Ratio an (positions = ∫ ratio dt)."""
    n = len(x)
    t_axis = np.arange(n) / SR
    pos = np.cumsum(_ratio_fn(t_axis)) / SR
    pos = pos - pos[0]
    return bandlimited_warp(x, np.clip(pos * SR, 0, n - 1)).astype(np.float32)


def _deviation_vs_target(est: np.ndarray, times: np.ndarray) -> float:
    """Relativer Trajektorienfehler: mean|est − Soll| / mean|Soll − 1|."""
    target = 1.0 / _ratio_fn(times)
    err = float(np.mean(np.abs(np.asarray(est, dtype=np.float64) - target)))
    amp = float(np.mean(np.abs(target - 1.0)))
    return err / max(amp, 1e-9)


def test_spectral_warp_supply_recovers_known_warp_within_10_percent() -> None:
    """Roadmap-Akzeptanz: Zero-Consensus-Versorgung trifft den bekannten Warp
    mit < 10 % Trajektorienabweichung (harmonisch reiches Musik-Segment)."""
    n = SR * 4
    warped = _apply_known_warp(_harmonic_music_like(n))
    times, _, _ = spectral_warp_estimate(warped, SR)
    flat = np.ones(len(times), dtype=np.float32)

    supplied = WowFlutterFix()._spectral_warp_supply_or_consensus(warped, flat, SR)

    assert supplied.shape == flat.shape  # Raster der Eingabe-Trajektorie
    assert float(np.max(np.abs(supplied - 1.0))) >= 0.004  # Versorgung hat stattgefunden
    rel_err = _deviation_vs_target(supplied, times)
    assert rel_err < 0.10, f"Trajektorienabweichung {rel_err:.4f} >= 10 %"


def test_spectral_warp_estimate_matches_known_warp_single_tone() -> None:
    """Schätzer-Ebene: Einzelton mit bekanntem Warp — < 5 % Abweichung."""
    n = SR * 4
    t = np.arange(n) / SR
    tone = (0.5 * np.sin(2 * np.pi * 1000.0 * t)).astype(np.float32)
    warped = _apply_known_warp(tone)
    times, warp_est, quality = spectral_warp_estimate(warped, SR)
    assert float(np.median(quality)) >= 0.55
    rel_err = _deviation_vs_target(1.0 / np.asarray(warp_est, dtype=np.float64), times)
    assert rel_err < 0.05, f"Trajektorienabweichung {rel_err:.4f} >= 5 %"


def test_supply_preserves_input_grid_length() -> None:
    """Die Versorgung interpoliert auf das Raster der Eingabe-Trajektorie."""
    n = SR * 4
    warped = _apply_known_warp(_harmonic_music_like(n))
    odd_grid = np.ones(101, dtype=np.float32)
    supplied = WowFlutterFix()._spectral_warp_supply_or_consensus(warped, odd_grid, SR)
    assert supplied.shape == odd_grid.shape


def test_noise_input_quality_gate_blocks_supply() -> None:
    """Rauschen: Self-Consistency-Qualität ≈ 0 → Eingabe bleibt unverändert."""
    rng = np.random.default_rng(3)
    noise = (rng.standard_normal(SR * 3) * 0.1).astype(np.float32)
    flat = np.ones(64, dtype=np.float32)
    out = WowFlutterFix()._spectral_warp_supply_or_consensus(noise, flat, SR)
    assert np.array_equal(out, flat)


def test_supply_short_input_returns_unchanged() -> None:
    """Signale < n_fft: Schätzer liefert nichts → Trajektorie unverändert."""
    short = np.zeros(1024, dtype=np.float32)
    flat = np.ones(8, dtype=np.float32)
    out = WowFlutterFix()._spectral_warp_supply_or_consensus(short, flat, SR)
    assert np.array_equal(out, flat)


def test_supply_deterministic() -> None:
    """Determinismus §G5 (copilot-instructions.md): zwei Aufrufe → identisch."""
    n = SR * 4
    warped = _apply_known_warp(_harmonic_music_like(n))
    flat = np.ones(100, dtype=np.float32)
    fix = WowFlutterFix()
    a = fix._spectral_warp_supply_or_consensus(warped, flat, SR)
    b = fix._spectral_warp_supply_or_consensus(warped, flat, SR)
    assert np.array_equal(a, b)


def test_consensus_refinement_averages_agreement(monkeypatch) -> None:
    """Konsens-Gate: Spektral-Schätzer stimmt der F0-Trajektorie zu → Mittelung
    (kontrolliert per Fake-Schätzer, damit der Pfad deterministisch testbar ist)."""
    n = 200
    t = np.arange(n, dtype=np.float64)
    sf = (1.0 + 0.004 * np.sin(2 * np.pi * 0.02 * t)).astype(np.float32)  # nicht flach

    def _fake_estimate(audio, sr, **kwargs):
        return np.arange(n, dtype=np.float64), (1.0 / sf).astype(np.float64), np.full(n, 0.9)

    monkeypatch.setattr("backend.core.dsp.warp_estimator.spectral_warp_estimate", _fake_estimate)
    audio = np.zeros(SR, dtype=np.float32)  # Dummy — Fake ignoriert Inhalt
    out = WowFlutterFix()._spectral_warp_supply_or_consensus(audio, sf, SR)
    assert np.allclose(out, sf, atol=1e-6)


def test_consensus_disagreement_keeps_f0_trajectory(monkeypatch) -> None:
    """Konsens-Gate: Widerspruch (> tol) → F0-Trajektorie bleibt unverändert."""
    n = 200
    t = np.arange(n, dtype=np.float64)
    sf = (1.0 + 0.004 * np.sin(2 * np.pi * 0.02 * t)).astype(np.float32)

    def _fake_estimate(audio, sr, **kwargs):
        wrong = (1.0 / (sf + 0.03)).astype(np.float64)  # 3 % daneben → nie Konsens
        return np.arange(n, dtype=np.float64), wrong, np.full(n, 0.9)

    monkeypatch.setattr("backend.core.dsp.warp_estimator.spectral_warp_estimate", _fake_estimate)
    audio = np.zeros(SR, dtype=np.float32)
    out = WowFlutterFix()._spectral_warp_supply_or_consensus(audio, sf, SR)
    assert np.array_equal(out, sf)


def test_vibrato_like_modulation_not_supplied() -> None:
    """WF-V2-Härtung (§v10.709-Befund): Musikalische Modulation (Vibrato-artig,
    6 Hz) liegt außerhalb des mechanischen Wow-Bands (< 4 Hz) — die Versorgung
    darf sie NICHT korrigieren (sonst artikulation/tonal_center-Degradation)."""
    n = SR * 4
    t = np.arange(n) / SR
    rng = np.random.default_rng(42)
    x = np.zeros(n)
    for k, a in enumerate([0.5, 0.28, 0.22, 0.14], start=1):
        x += a * np.sin(2 * np.pi * 220.0 * k * t + rng.uniform(0, 2 * np.pi))
    x = (0.5 * x / np.max(np.abs(x))).astype(np.float32)
    _ratio_6hz = 1.0 + 0.02 * np.sin(2 * np.pi * 6.0 * t)
    pos = np.cumsum(_ratio_6hz) / SR
    pos = pos - pos[0]
    warped = bandlimited_warp(x, np.clip(pos * SR, 0, n - 1)).astype(np.float32)

    times, warp_est, quality = spectral_warp_estimate(warped, SR)
    # Sanity: Der Schätzer SIEHT die Modulation (Deviation über dem Gate) —
    # die Wow-Band-Härtung muss die Versorgung trotzdem verweigern.
    assert float(np.max(np.abs(warp_est - 1.0))) >= 0.004
    assert float(np.median(quality)) >= 0.55

    flat = np.ones(len(times), dtype=np.float32)
    out = WowFlutterFix()._spectral_warp_supply_or_consensus(warped, flat, SR)
    assert np.array_equal(out, flat)


def test_consensus_requires_majority(monkeypatch) -> None:
    """Konsens erst ab ≥ 50 % Übereinstimmung — 30 % bleiben bei der F0-Trajektorie."""
    n = 200
    t = np.arange(n, dtype=np.float64)
    sf = (1.0 + 0.004 * np.sin(2 * np.pi * 0.02 * t)).astype(np.float32)

    def _fake_estimate(audio, sr, **kwargs):
        # 30 % Übereinstimmung: 60 Frames exakt, 140 um 0,03 versetzt
        warp = np.empty(n, dtype=np.float64)
        warp[:60] = 1.0 / sf[:60]
        warp[60:] = 1.0 / (sf[60:] + 0.03)
        return np.arange(n, dtype=np.float64), warp, np.full(n, 0.9)

    monkeypatch.setattr("backend.core.dsp.warp_estimator.spectral_warp_estimate", _fake_estimate)
    audio = np.zeros(SR, dtype=np.float32)
    out = WowFlutterFix()._spectral_warp_supply_or_consensus(audio, sf, SR)
    assert np.array_equal(out, sf)
