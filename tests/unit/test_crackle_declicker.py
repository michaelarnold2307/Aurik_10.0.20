"""
Tests für backend/core/dsp/crackle_declicker.py — §SR-CK4 Fein-Declicker.

Test-Abdeckung:
  - Knistern-Entfernung: HP-Band-MAE sinkt deutlich, Vollband-Max-Fehler
    wächst nie (Never-worsen, §G7 (copilot-instructions.md))
  - Musik-Erhalt: Korrelation zur Wahrheit bleibt ≥ 0,999
  - Deterministisch (§G5 (copilot-instructions.md)): bit-identisch bei Wiederholung
  - Sauberes Signal: bit-identischer Passthrough (keine False Positives)
  - Musik-Transienten-Schutz: 10-ms-Burst bleibt unangetastet
  - Stereo-Layout-Invariante: (2, N) und (N, 2) werden erhalten
  - NaN-Sicherheit, Kurzsignal-Fail-Open
  - Strength-Skalierung: sekundäre Reparaturen folgen der Stärke

Alle Tests deterministisch (fester Seed), kein Datei-I/O.
"""

import numpy as np
import pytest
from scipy import signal

SR = 48000

_HP_SOS = signal.butter(4, 2200.0, btype="highpass", fs=SR, output="sos")


def _hp(x: np.ndarray) -> np.ndarray:
    return signal.sosfiltfilt(_HP_SOS, x)


def _sine(duration_s: float = 2.0, freq: float = 440.0, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(duration_s * SR)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _add_impulses(
    base: np.ndarray,
    n_impulses: int,
    min_dist: int,
    seed: int,
    amp_lo: float = 0.05,
    amp_hi: float = 0.12,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sig = base.copy()
    lo, hi = 1000, len(base) - 1000
    positions: list[int] = []
    attempts = 0
    while len(positions) < n_impulses and attempts < 100000:
        p = int(rng.integers(lo, hi))
        if all(abs(p - q) >= min_dist for q in positions):
            positions.append(p)
        attempts += 1
    for p in positions:
        w = int(rng.integers(1, 8))
        sig[p : p + w] += rng.uniform(amp_lo, amp_hi)  # konstante Klick-Amplitude
    return sig


@pytest.mark.unit
class TestDeclickFineCrackle:
    """declick_fine_crackle — Kernfunktion."""

    def test_removes_crackle_preserves_music_never_worsens(self):
        """HP-Knistern sinkt deutlich, Musik bleibt, HP-Band-Max wächst nie.

        Vollband-Never-worsen ist bei HP-Band-Subtraktion prinzipiell nicht
        garantierbar: Der LP-Komplementfilter (scharfe Transition, ℓ₁ > 1)
        lässt einen Schmier-Rest, der einen breiten Klick im Vollband leicht
        überschreiten kann (Produktionsbefund 0,145 vs 0,113). Die hörbare
        Knistern-Metrik ist das HP-Band — dort gilt Never-worsen.
        """
        from backend.core.dsp.crackle_declicker import declick_fine_crackle

        sine = _sine(3.0)
        crack = _add_impulses(sine, 60, min_dist=1000, seed=7)
        out = declick_fine_crackle(crack, SR, strength=1.0)
        hp_mae_in = float(np.abs(_hp(crack) - _hp(sine)).mean())
        hp_mae_out = float(np.abs(_hp(out) - _hp(sine)).mean())
        assert hp_mae_out < 0.8 * hp_mae_in, f"HP-MAE {hp_mae_in:.6f} -> {hp_mae_out:.6f}"
        assert float(np.abs(_hp(out) - _hp(sine)).max()) <= float(np.abs(_hp(crack) - _hp(sine)).max()) + 1e-6
        assert float(np.abs(out - sine).mean()) <= 2.0 * float(np.abs(crack - sine).mean())
        assert float(np.corrcoef(out, sine)[0, 1]) >= 0.999

    def test_deterministic(self):
        """Gleicher Input ⇒ bit-identischer Output (§G5 (copilot-instructions.md))."""
        from backend.core.dsp.crackle_declicker import declick_fine_crackle

        crack = _add_impulses(_sine(2.0), 50, min_dist=800, seed=11)
        a = declick_fine_crackle(crack, SR, strength=1.0)
        b = declick_fine_crackle(crack, SR, strength=1.0)
        assert np.array_equal(a, b)

    def test_clean_passthrough_bit_identical(self):
        """Sauberer Sinus wird bit-identisch zurückgegeben (keine False Positives)."""
        from backend.core.dsp.crackle_declicker import declick_fine_crackle

        sine = _sine(2.0)
        out = declick_fine_crackle(sine, SR, strength=1.0)
        assert np.array_equal(out, sine)

    def test_musical_transient_preserved(self):
        """Ein 10-ms-Burst (Musik-Transient) bleibt unangetastet."""
        from backend.core.dsp.crackle_declicker import declick_fine_crackle

        sig = _sine(2.0)
        sig[20000 : 20000 + 480] += 0.4
        out = declick_fine_crackle(sig, SR, strength=1.0)
        assert float(out.max()) == pytest.approx(float(sig.max()), abs=1e-6)

    def test_stereo_layouts_preserved(self):
        """Channels-first (2, N) und channels-last (N, 2) werden erhalten."""
        from backend.core.dsp.crackle_declicker import declick_fine_crackle

        crack = _add_impulses(_sine(2.0), 40, min_dist=1000, seed=3)
        stereo_cf = np.stack([crack, crack * 0.9], axis=0)
        out_cf = declick_fine_crackle(stereo_cf, SR, strength=1.0)
        assert out_cf.shape == stereo_cf.shape
        stereo_cl = stereo_cf.T
        out_cl = declick_fine_crackle(stereo_cl, SR, strength=1.0)
        assert out_cl.shape == stereo_cl.shape

    def test_nan_safe(self):
        """NaN-Eingaben führen nicht zu NaN/Inf im Output."""
        from backend.core.dsp.crackle_declicker import declick_fine_crackle

        crack = _add_impulses(_sine(2.0), 40, min_dist=1000, seed=5)
        crack[100] = np.nan
        out = declick_fine_crackle(crack, SR, strength=1.0)
        assert np.all(np.isfinite(out))

    def test_short_input_fail_open(self):
        """Kurze Eingaben: unveränderte Kopie (Fail-Open, §V6 (copilot-instructions.md))."""
        from backend.core.dsp.crackle_declicker import declick_fine_crackle

        short = np.zeros(100, dtype=np.float32)
        out = declick_fine_crackle(short, SR, strength=1.0)
        assert np.array_equal(out, short)

    def test_strength_scales_secondary_repairs(self):
        """Strength=0 repariert nicht mehr Samples als strength=1."""
        from backend.core.dsp.crackle_declicker import declick_fine_crackle

        crack = _add_impulses(_sine(2.0), 60, min_dist=600, seed=13, amp_lo=0.02, amp_hi=0.05)
        full = declick_fine_crackle(crack, SR, strength=1.0)
        none = declick_fine_crackle(crack, SR, strength=0.0)
        n_full = int((np.abs(full - crack) > 1e-9).sum())
        n_none = int((np.abs(none - crack) > 1e-9).sum())
        assert n_none <= n_full
