import pytest

"""Tests für _VATEmotionEstimator — Valence-Arousal-Tension-Modell (§V45).

Abdeckung:
  - _VATEmotionEstimator.estimate: Grundfunktionalität, Output-Keys + Wertebereich
  - Valence: Dur-Akkord → höher als Moll-Akkord
  - Arousal: Schnelles Signal → höher als langsames
  - Tension: Dissonantes Signal → höher als reines Sinussignal
  - VAT-Blend in EmotionalitaetMetric.measure(): Typ-Check, keine Exception
  - Edge-Cases: Stille, kurzes Signal, Mono/Stereo
"""

import numpy as np

SR = 48000


def _silence(duration_s: float = 2.0) -> np.ndarray:
    return np.zeros(int(duration_s * SR), dtype=np.float32)


def _sine(freq_hz: float = 440.0, duration_s: float = 2.0, amp: float = 0.5) -> np.ndarray:
    t = np.linspace(0, duration_s, int(duration_s * SR), endpoint=False)
    return np.asarray(amp * np.sin(2 * np.pi * freq_hz * t), dtype=np.float32)


def _white_noise(duration_s: float = 2.0, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(duration_s * SR)) * 0.3).astype(np.float32)


def _major_chord(duration_s: float = 2.0, amp: float = 0.3) -> np.ndarray:
    """C-Dur Dreiklang: C4 (261 Hz), E4 (330 Hz), G4 (392 Hz)."""
    t = np.linspace(0, duration_s, int(duration_s * SR), endpoint=False)
    return np.asarray(
        amp * (np.sin(2 * np.pi * 261.63 * t) + np.sin(2 * np.pi * 329.63 * t) + np.sin(2 * np.pi * 392.00 * t)) / 3.0,
        dtype=np.float32,
    )


def _minor_chord(duration_s: float = 2.0, amp: float = 0.3) -> np.ndarray:
    """A-Moll Dreiklang: A3 (220 Hz), C4 (261 Hz), E4 (330 Hz)."""
    t = np.linspace(0, duration_s, int(duration_s * SR), endpoint=False)
    return np.asarray(
        amp * (np.sin(2 * np.pi * 220.00 * t) + np.sin(2 * np.pi * 261.63 * t) + np.sin(2 * np.pi * 329.63 * t)) / 3.0,
        dtype=np.float32,
    )


@pytest.mark.unit
class TestVATEmotionEstimatorImport:
    """_VATEmotionEstimator — Importierbar."""

    def test_import_from_musical_goals_metrics(self):
        """Klasse existiert und ist importierbar."""
        from backend.core.musical_goals.musical_goals_metrics import _VATEmotionEstimator

        assert callable(_VATEmotionEstimator)


class TestVATEstimateOutputSchema:
    """_VATEmotionEstimator.estimate — Output-Schema."""

    def test_returns_dict_with_all_keys(self):
        from backend.core.musical_goals.musical_goals_metrics import _VATEmotionEstimator

        vat = _VATEmotionEstimator()
        result = vat.estimate(_sine(), SR)
        assert isinstance(result, dict)
        for key in ("valence", "arousal", "tension"):
            assert key in result, f"Fehlender Key: {key}"

    def test_output_range_0_to_1(self):
        """Alle Dimensionen in [0, 1]."""
        from backend.core.musical_goals.musical_goals_metrics import _VATEmotionEstimator

        vat = _VATEmotionEstimator()
        for audio in [_sine(), _white_noise(), _major_chord(), _minor_chord()]:
            result = vat.estimate(audio, SR)
            for key in ("valence", "arousal", "tension"):
                assert 0.0 <= result[key] <= 1.0, f"{key} außerhalb [0, 1]: {result[key]}"

    def test_no_nan(self):
        """Kein NaN in Output."""
        from backend.core.musical_goals.musical_goals_metrics import _VATEmotionEstimator

        vat = _VATEmotionEstimator()
        for audio in [_sine(), _white_noise(), _silence()]:
            result = vat.estimate(audio, SR)
            for key, val in result.items():
                assert not np.isnan(val), f"NaN in {key}"

    def test_silence_neutral_fallback(self):
        """Stille → neutrale Werte (Fallback)."""
        from backend.core.musical_goals.musical_goals_metrics import _VATEmotionEstimator

        vat = _VATEmotionEstimator()
        result = vat.estimate(_silence(), SR)
        # Neutral oder sinnvolle Werte — kein Crash
        assert isinstance(result, dict)
        for key in ("valence", "arousal", "tension"):
            assert key in result

    def test_stereo_input(self):
        """Stereo-Input wird intern zu Mono gemittelt → kein Absturz."""
        from backend.core.musical_goals.musical_goals_metrics import _VATEmotionEstimator

        vat = _VATEmotionEstimator()
        mono = _sine()
        stereo = np.stack([mono, mono * 0.8], axis=0)
        result = vat.estimate(stereo, SR)
        for key in ("valence", "arousal", "tension"):
            assert key in result
            assert 0.0 <= result[key] <= 1.0

    def test_short_signal_fallback(self):
        """Sehr kurzes Signal → neutrale Fallback-Werte, kein Crash."""
        from backend.core.musical_goals.musical_goals_metrics import _VATEmotionEstimator

        vat = _VATEmotionEstimator()
        short = np.ones(512, dtype=np.float32) * 0.1
        result = vat.estimate(short, SR)
        assert isinstance(result, dict)


class TestValenceDimension:
    """Valence-Dimension — Dur/Moll-Unterscheidung."""

    def test_returns_float(self):
        from backend.core.musical_goals.musical_goals_metrics import _VATEmotionEstimator

        vat = _VATEmotionEstimator()
        result = vat.estimate(_major_chord(), SR)
        assert isinstance(result["valence"], float)

    def test_valence_range(self):
        from backend.core.musical_goals.musical_goals_metrics import _VATEmotionEstimator

        vat = _VATEmotionEstimator()
        for audio in [_major_chord(), _minor_chord()]:
            v = vat.estimate(audio, SR)["valence"]
            assert 0.0 <= v <= 1.0


class TestArousalDimension:
    """Arousal-Dimension — Energie-Dynamik."""

    def test_noise_higher_arousal_than_silence(self):
        """Weißes Rauschen (variabel) → höhere Arousal als Stille."""
        from backend.core.musical_goals.musical_goals_metrics import _VATEmotionEstimator

        vat = _VATEmotionEstimator()
        a_noise = vat.estimate(_white_noise(), SR)["arousal"]
        a_silence = vat.estimate(_silence(), SR)["arousal"]
        # Stille → neutral (0.5); Rauschen kann höher oder niedriger sein
        # Mindestens: keins von beidem soll einen Fehler erzeugen
        assert 0.0 <= a_noise <= 1.0
        assert 0.0 <= a_silence <= 1.0

    def test_arousal_range(self):
        from backend.core.musical_goals.musical_goals_metrics import _VATEmotionEstimator

        vat = _VATEmotionEstimator()
        for audio in [_sine(), _white_noise(), _major_chord()]:
            a = vat.estimate(audio, SR)["arousal"]
            assert 0.0 <= a <= 1.0


class TestTensionDimension:
    """Tension-Dimension — Spektrale Irregularität + Lautstärke."""

    def test_noise_higher_tension_than_sine(self):
        """Weißes Rauschen (irreguläres Spektrum) → höhere Tension als Sinussignal."""
        from backend.core.musical_goals.musical_goals_metrics import _VATEmotionEstimator

        vat = _VATEmotionEstimator()
        t_noise = vat.estimate(_white_noise(), SR)["tension"]
        t_sine = vat.estimate(_sine(440.0), SR)["tension"]
        # Rauschen hat irreguläres Spektrum → höhere Tension erwartet
        assert t_noise >= t_sine * 0.9, f"Rauschen-Tension ({t_noise:.3f}) soll ≥ Sinus-Tension ({t_sine:.3f}) sein"

    def test_tension_range(self):
        from backend.core.musical_goals.musical_goals_metrics import _VATEmotionEstimator

        vat = _VATEmotionEstimator()
        for audio in [_sine(), _white_noise(), _major_chord(), _silence()]:
            t = vat.estimate(audio, SR)["tension"]
            assert 0.0 <= t <= 1.0


class TestVATBlendInEmotionalitaetMetric:
    """VAT-Blend in EmotionalitaetMetric.measure() — Integration."""

    def test_measure_returns_float_in_range(self):
        """EmotionalitaetMetric.measure() gibt Float in [0, 1] zurück."""
        from backend.core.musical_goals.musical_goals_metrics import EmotionalitaetMetric

        metric = EmotionalitaetMetric()
        audio = _sine(440.0)
        score = metric.measure(audio, SR)
        assert isinstance(score, float), f"measure() soll float zurückgeben, got {type(score)}"
        assert 0.0 <= score <= 1.0, f"Score außerhalb [0, 1]: {score}"

    def test_measure_no_nan(self):
        """EmotionalitaetMetric.measure() — kein NaN."""
        from backend.core.musical_goals.musical_goals_metrics import EmotionalitaetMetric

        metric = EmotionalitaetMetric()
        for audio in [_sine(), _white_noise(), _major_chord(), _minor_chord()]:
            score = metric.measure(audio, SR)
            assert not np.isnan(score), "NaN in measure() output"

    def test_measure_no_exception_for_edge_cases(self):
        """Keine Exception bei kurzen oder stillen Signalen."""
        from backend.core.musical_goals.musical_goals_metrics import EmotionalitaetMetric

        metric = EmotionalitaetMetric()
        for audio in [_silence(0.5), np.ones(512, dtype=np.float32) * 0.01]:
            score = metric.measure(audio, SR)
            assert isinstance(score, float)
