"""§SOTA-PSY-A7 (wahrnehmungs-basierter Loudness-Cap, phase_47) — Tests.

Deckt den 2026-09-15-Folge-Schritt: Der True-Peak-Limiter darf die Kurzzeit-
Lautheit (peak STL, Sone) nicht über den Input hinaus anheben; Überschreitung
> Marge ⇒ proportionale Rücknahme Richtung Input (Never-worsen, Hörordnung §4/§8a).

Autor: Aurik Testing Team
"""

import numpy as np

from backend.core.phases.phase_47_truepeak_limiter import _perceptual_loudness_cap

_CEIL = 10 ** (-0.5 / 20.0)


def _audio(n: int = 48000) -> np.ndarray:
    t = np.linspace(0, n / 48000, n, endpoint=False, dtype=np.float32)
    return (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


class TestPerceptualLoudnessCap:
    def test_no_cap_when_loudness_not_raised(self):
        audio = _audio()
        processed = audio * 0.8  # leiser → nie cappen
        psy7 = {"peak_stl_before_sone": 2.0, "peak_stl_after_sone": 1.5}
        out, psy7 = _perceptual_loudness_cap(audio, processed, psy7, _CEIL)
        assert psy7["loudness_cap_applied"] is False
        assert np.array_equal(out, processed)

    def test_cap_backs_off_when_loudness_raised(self):
        audio = _audio()
        processed = audio * 1.5  # lauter → Cap greift
        psy7 = {"peak_stl_before_sone": 2.0, "peak_stl_after_sone": 4.0}
        out, psy7 = _perceptual_loudness_cap(audio, processed, psy7, _CEIL)
        assert psy7["loudness_cap_applied"] is True
        assert 0.0 < psy7["loudness_cap_wet"] < 1.0
        # Rücknahme Richtung Input: |out − audio| < |processed − audio|
        d_capped = float(np.abs(out - audio).mean())
        d_processed = float(np.abs(processed - audio).mean())
        assert d_capped < d_processed

    def test_small_raise_within_margin_is_kept(self):
        audio = _audio()
        processed = audio * 1.02
        psy7 = {"peak_stl_before_sone": 10.0, "peak_stl_after_sone": 10.2}  # < 0,5-Marge
        out, psy7 = _perceptual_loudness_cap(audio, processed, psy7, _CEIL)
        assert psy7["loudness_cap_applied"] is False
        assert np.array_equal(out, processed)

    def test_deterministic(self):
        audio = _audio(24000)
        processed = audio * 1.6
        psy = {"peak_stl_before_sone": 2.0, "peak_stl_after_sone": 4.5}
        o1, _ = _perceptual_loudness_cap(audio, processed, dict(psy), _CEIL)
        o2, _ = _perceptual_loudness_cap(audio, processed, dict(psy), _CEIL)
        assert np.array_equal(o1, o2)

    def test_nan_safe_and_clipped(self):
        audio = _audio(8000)
        processed = (audio * 2.0).copy()
        processed[100] = np.nan
        psy = {"peak_stl_before_sone": 2.0, "peak_stl_after_sone": 6.0}
        out, _ = _perceptual_loudness_cap(audio, processed, psy, _CEIL)
        assert np.isfinite(out).all()
        assert float(np.abs(out).max()) <= _CEIL + 1e-6


class TestPhase47Metadata:
    def test_process_reports_loudness_cap_flag(self):
        from backend.core.phases.phase_47_truepeak_limiter import TruePeakLimiterPhase

        audio = _audio(24000)
        res = TruePeakLimiterPhase().process(audio, sample_rate=48000, strength=1.0)
        psy = res.metadata.get("short_term_loudness", {})
        # Witness läuft (oder fehlt nur bei Kurzsignalen) — wenn vorhanden,
        # ist das Cap-Flag ein bool.
        if psy:
            assert isinstance(psy.get("loudness_cap_applied", False), bool)
        assert np.isfinite(res.audio).all()
        assert len(res.audio) == len(audio)
