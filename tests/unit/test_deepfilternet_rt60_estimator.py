"""Unit-Tests für die RT60-Schätzung via DeepFilterNet-Trocken-Zerlegung (§SOTA-DR-V1).

plugins/deepfilternet_v3_ii_plugin.estimate_rt60_sec: DFN entfernt Hall anteilig;
die Differenz Eingang − trocken ≈ Hallanteil; Schröder-Rückwärtsintegration ⇒
T30 ⇒ RT60. Confidence aus dem Hall-Energie-Anteil (kein Hall ⇒ conf=0 ⇒ Aufrufer
ignoriert den Wert). Deterministisch (§G5 (GEBOTE.md)), kein Modell ⇒ None
(§V6 (copilot-instructions.md)-Fallback).

Abgedeckt: Ground-Truth-Rekonstruktion (synthetischer Hall, RT60≈0,6 s),
trockenes Material ⇒ conf=0, fehlendes Modell ⇒ None, Layout-Sicherheit,
Determinismus, NaN/Inf-Robustheit.
"""

from __future__ import annotations

import numpy as np
import pytest

from plugins.deepfilternet_v3_ii_plugin import DeepFilterNetV3Plugin


class _FakeDfn(DeepFilterNetV3Plugin):
    """enhance() liefert das injizierte Trocken-Signal (perfekte Trennung)."""

    def __init__(self, dry: np.ndarray | None = None) -> None:
        self._enc = object()  # Modell-Verfügbarkeit simulieren
        self._dry = dry

    def enhance(self, audio: np.ndarray, sr: int, energy_bias_db: float = 0.0) -> np.ndarray:
        if self._dry is None:
            return np.asarray(audio, dtype=np.float32)
        n = min(audio.size, self._dry.size)
        return np.asarray(self._dry[:n], dtype=np.float32)


def _dry_noise(sr: int, seconds: float, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.randn(int(sr * seconds)).astype(np.float32) * 0.05


def _synthetic_reverb(dry: np.ndarray, sr: int, rt60: float) -> np.ndarray:
    """Echt abklingender Hall: Amplituden-Envelope exp(−6.9078·t/rt60) ⇒ Energie
    −60 dB bei rt60; Hall-RMS auf 0,03 normiert (conf-Gate schlägt sicher an)."""
    rng = np.random.RandomState(99)
    n = len(dry)
    t_axis = np.arange(n, dtype=np.float64) / sr
    env = np.exp(-6.9078 * t_axis / rt60)
    rev = rng.randn(n).astype(np.float64) * env
    rev = rev / float(np.sqrt(np.mean(rev**2)) + 1e-12) * 0.03
    out = dry.astype(np.float64) + rev  # kein Skalenfaktor — perfekte Trennung muss exakt gelten
    return out.astype(np.float32)


def test_rt60_recovers_ground_truth() -> None:
    """Perfekte Trennung + Schröder-T30 ⇒ RT60 ≈ Ground Truth (0,6 s ± 25 %)."""
    sr = 8000
    dry = _dry_noise(sr, 4.0, seed=1)
    wet = _synthetic_reverb(dry, sr, rt60=0.6)
    p = _FakeDfn(dry)
    res = p.estimate_rt60_sec(wet, sr)
    assert res is not None
    rt60, conf = res
    assert 0.45 <= rt60 <= 0.75
    assert conf >= 0.3


def test_rt60_dry_signal_zero_confidence() -> None:
    """Ohne Hall (enhance = Identität) ⇒ rev≈0 ⇒ conf=0, RT60 am Floor."""
    sr = 8000
    x = _dry_noise(sr, 3.0, seed=2)
    p = _FakeDfn(None)  # enhance = Passthrough
    res = p.estimate_rt60_sec(x, sr)
    assert res is not None
    rt60, conf = res
    assert conf == 0.0
    assert rt60 == pytest.approx(0.05, abs=1e-3)


def test_rt60_none_without_model() -> None:
    """Kein Modell geladen ⇒ None (Aufrufer nutzt DSP-Heuristik, §V6 (copilot-instructions.md))."""
    p = DeepFilterNetV3Plugin.__new__(DeepFilterNetV3Plugin)
    p._enc = None
    assert p.estimate_rt60_sec(_dry_noise(8000, 1.0, seed=3), 8000) is None


def test_rt60_stereo_layout_safe() -> None:
    """(C,N)-Stereo kollabiert nicht auf C Samples — gleiche Schätzung wie Mono-Mix."""
    sr = 8000
    dry = _dry_noise(sr, 2.0, seed=4)
    wet_mono = _synthetic_reverb(dry, sr, rt60=0.5)
    wet_stereo = np.stack([wet_mono, wet_mono], axis=0)  # (2, N)
    p = _FakeDfn(dry)
    res_mono = p.estimate_rt60_sec(wet_mono, sr)
    res_stereo = p.estimate_rt60_sec(wet_stereo, sr)
    assert res_mono is not None and res_stereo is not None
    assert res_stereo[0] == pytest.approx(res_mono[0], abs=1e-6)


def test_rt60_deterministic() -> None:
    """Gleicher Input ⇒ bit-identisches Ergebnis (§G5 (GEBOTE.md))."""
    sr = 8000
    dry = _dry_noise(sr, 2.0, seed=5)
    wet = _synthetic_reverb(dry, sr, rt60=0.4)
    p = _FakeDfn(dry)
    r1 = p.estimate_rt60_sec(wet, sr)
    r2 = p.estimate_rt60_sec(wet, sr)
    assert r1 == r2


def test_rt60_nan_inf_robust() -> None:
    """NaN/Inf im Eingang werden bereinigt, Ergebnis bleibt finit."""
    sr = 8000
    dry = _dry_noise(sr, 2.0, seed=6)
    wet = _synthetic_reverb(dry, sr, rt60=0.5)
    wet = wet.copy()
    wet[::500] = np.nan
    wet[1::500] = np.inf
    p = _FakeDfn(dry)
    res = p.estimate_rt60_sec(wet, sr)
    assert res is not None
    assert np.isfinite(res[0]) and np.isfinite(res[1])


def test_rt60_strength_delta_gates() -> None:
    """§SOTA-DR-V1-Delta: neutral < 0,8 s, conf < 0,5 ⇒ 0, Cap bei +0,35."""
    from plugins.deepfilternet_v3_ii_plugin import rt60_strength_delta

    assert rt60_strength_delta(0.5, 0.9) == 0.0
    assert rt60_strength_delta(1.2, 0.9) == pytest.approx(0.14, abs=1e-6)
    assert rt60_strength_delta(1.2, 0.4) == 0.0
    assert rt60_strength_delta(5.0, 1.0) == pytest.approx(0.35, abs=1e-6)
    assert rt60_strength_delta(float("nan"), 1.0) == 0.0
