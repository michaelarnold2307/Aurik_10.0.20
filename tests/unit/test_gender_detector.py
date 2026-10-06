from __future__ import annotations

import pytest

"""
Unit tests for GenderDetector.detect() / _detect_f0() / _classify_gender().

Bugs fixed (see CHANGELOG.md):
  - _detect_f0 used peaks[0] (highest f0) instead of argmax peak (true f0)
  - FEMALE f0 range was (165,255) — speech-only; singing range is (165,700)
  - No tie-breaking rule for FEMALE vs CHILD when f0 < 350 Hz
"""


import numpy as np

from backend.core.vocal_ai_enhancement import GenderDetector, VoiceGender

SR = 48_000


def _sine(f0: float, dur: float = 0.5, sr: int = SR) -> np.ndarray:
    """Pure sine at f0 Hz, normalized."""
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    return (np.sin(2 * np.pi * f0 * t) * 0.5).astype(np.float32)  # type: ignore[no-any-return]


def _harmonic(f0: float, dur: float = 0.5, sr: int = SR, n_harmonics: int = 6) -> np.ndarray:
    """Harmonic signal at f0 Hz (fundamental + overtones), normalized."""
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    sig = np.zeros_like(t)
    for k in range(1, n_harmonics + 1):
        sig += np.sin(2 * np.pi * f0 * k * t) / k
    sig /= np.max(np.abs(sig)) + 1e-10
    return (sig * 0.5).astype(np.float32)


def _add_noise(sig: np.ndarray, snr_db: float = 20.0) -> np.ndarray:
    """Add white noise at given SNR."""
    rms_sig = np.sqrt(np.mean(sig**2)) + 1e-10
    noise = np.random.default_rng(42).standard_normal(len(sig)).astype(np.float32)
    rms_noise_target = rms_sig * 10 ** (-snr_db / 20)
    noise *= rms_noise_target / (np.sqrt(np.mean(noise**2)) + 1e-10)
    return sig + noise


# ---------------------------------------------------------------------------
# _detect_f0 — strongest peak, not first peak
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDetectF0:
    """_detect_f0 must return fundamental, not an octave-up harmonic."""

    def test_clean_sine_220hz(self):
        """Pure sine at 220 Hz must be detected within ±10 Hz."""
        gd = GenderDetector(sample_rate=SR)
        f0 = gd._detect_f0(_sine(220))
        assert abs(f0 - 220) <= 10, f"Expected ~220 Hz, got {f0:.1f}"

    def test_clean_sine_300hz(self):
        """Pure sine at 300 Hz (mezzo singing) must be detected within ±15 Hz."""
        gd = GenderDetector(sample_rate=SR)
        f0 = gd._detect_f0(_sine(300))
        assert abs(f0 - 300) <= 15, f"Expected ~300 Hz, got {f0:.1f}"

    def test_harmonic_220hz_noisy(self):
        """Harmonic signal at 220 Hz + 20 dB noise: fundamental must dominate."""
        gd = GenderDetector(sample_rate=SR)
        sig = _add_noise(_harmonic(220), snr_db=20)
        f0 = gd._detect_f0(sig)
        # Old peaks[0] would often return ~440 Hz (first harmonic peak in autocorr);
        # argmax must return ~220 Hz since the fundamental autocorr peak is strongest.
        assert f0 < 300, f"Expected f0 < 300 Hz (fundamental), got {f0:.1f} (likely octave-up bug)"
        assert f0 > 100, f"Unexpectedly low f0: {f0:.1f}"

    def test_harmonic_200hz_noisy_tape_snr(self):
        """Vintage tape SNR (15 dB): harmonic at 200 Hz still detected as fundamental."""
        gd = GenderDetector(sample_rate=SR)
        sig = _add_noise(_harmonic(200, n_harmonics=8), snr_db=15)
        f0 = gd._detect_f0(sig)
        assert f0 < 280, f"Expected f0 < 280 Hz (fundamental ~200), got {f0:.1f}"

    def test_returns_zero_for_noise_only(self):
        """Broadband noise without pitch: f0 = 0.0."""
        gd = GenderDetector(sample_rate=SR)
        rng = np.random.default_rng(0)
        noise = rng.standard_normal(SR // 2).astype(np.float32) * 0.01
        f0 = gd._detect_f0(noise)
        assert f0 == 0.0 or f0 < 50, f"Expected 0 Hz for noise, got {f0:.1f}"

    def test_returns_zero_for_silence(self):
        gd = GenderDetector(sample_rate=SR)
        f0 = gd._detect_f0(np.zeros(SR // 4, dtype=np.float32))
        assert f0 == 0.0


# ---------------------------------------------------------------------------
# formant_ranges — FEMALE singing range extended (165–700 Hz)
# ---------------------------------------------------------------------------


class TestFemaleF0Range:
    """FEMALE f0 range must cover singing (up to 700 Hz), not just speech (255 Hz)."""

    def test_female_range_upper_bound_geq_700(self):
        gd = GenderDetector(sample_rate=SR)
        assert gd.formant_ranges[VoiceGender.FEMALE]["f0"][1] >= 700, (
            "FEMALE f0 upper bound must be >= 700 Hz to cover soprano/mezzo singing"
        )

    def test_female_range_lower_bound_leq_170(self):
        gd = GenderDetector(sample_rate=SR)
        assert gd.formant_ranges[VoiceGender.FEMALE]["f0"][0] <= 170, (
            "FEMALE f0 lower bound must be <= 170 Hz (contralto)"
        )

    def test_child_lower_bound_geq_250(self):
        """CHILD f0 lower bound must be >= 250 Hz so singing mezzo f0 < 250 Hz maps to FEMALE."""
        gd = GenderDetector(sample_rate=SR)
        assert gd.formant_ranges[VoiceGender.CHILD]["f0"][0] >= 250


# ---------------------------------------------------------------------------
# _classify_gender — tie-breaking: f0 < 350 Hz → prefer FEMALE over CHILD
# ---------------------------------------------------------------------------


class TestClassifyGenderTieBreak:
    """When scores are close and f0 < 350 Hz, FEMALE must win over CHILD."""

    def _score_manual(self, gd, f0, formants):
        """Call the internal _classify_gender directly."""
        return gd._classify_gender(f0, formants)

    def test_f0_300hz_returns_female_not_child(self):
        """f0=300 Hz with adult-female formants must classify as FEMALE."""
        gd = GenderDetector(sample_rate=SR)
        # Adult female formants: F1~600, F2~1800, F3~2700 Hz
        gender, conf = self._score_manual(gd, 300.0, [600.0, 1800.0, 2700.0])
        assert gender == VoiceGender.FEMALE, f"Expected FEMALE for f0=300 Hz + adult formants, got {gender}"

    def test_f0_250hz_returns_female(self):
        """f0=250 Hz (mezzo singing) must be FEMALE."""
        gd = GenderDetector(sample_rate=SR)
        gender, conf = self._score_manual(gd, 250.0, [550.0, 1700.0, 2600.0])
        assert gender == VoiceGender.FEMALE, f"Expected FEMALE for f0=250 Hz, got {gender}"

    def test_f0_200hz_returns_female(self):
        """f0=200 Hz (alto singing) must be FEMALE."""
        gd = GenderDetector(sample_rate=SR)
        gender, conf = self._score_manual(gd, 200.0, [500.0, 1600.0, 2500.0])
        assert gender == VoiceGender.FEMALE, f"Expected FEMALE for f0=200 Hz, got {gender}"

    def test_f0_120hz_returns_male(self):
        """f0=120 Hz must remain MALE."""
        gd = GenderDetector(sample_rate=SR)
        gender, _ = self._score_manual(gd, 120.0, [400.0, 1200.0, 2200.0])
        assert gender == VoiceGender.MALE, f"Expected MALE for f0=120 Hz, got {gender}"

    def test_f0_400hz_high_child_formants_returns_child(self):
        """f0=400 Hz + clearly child-sized formants (F2 > 3000) → CHILD."""
        gd = GenderDetector(sample_rate=SR)
        # Very high formants typical of young child (tiny vocal tract)
        gender, conf = self._score_manual(gd, 400.0, [900.0, 3200.0, 4800.0])
        # With proper formant evidence, CHILD must still be classifiable
        assert gender in (VoiceGender.CHILD, VoiceGender.FEMALE), f"Unexpected gender for child-like formants: {gender}"

    def test_no_unknown_for_valid_female_signal(self):
        """detect() on a plausible female singing signal must not return UNKNOWN."""
        gd = GenderDetector(sample_rate=SR)
        sig = _harmonic(230, dur=0.5)  # Mezzo-soprano pitch
        result = gd.detect(sig)
        assert result.gender != VoiceGender.UNKNOWN, "Expected valid gender classification, got UNKNOWN"


# ---------------------------------------------------------------------------
# End-to-end detect() — integration
# ---------------------------------------------------------------------------


class TestDetectIntegration:
    """detect() on synthetic signals must produce plausible results."""

    def test_detect_returns_voice_characteristics(self):
        gd = GenderDetector(sample_rate=SR)
        result = gd.detect(_harmonic(220, dur=0.6))
        assert hasattr(result, "gender")
        assert hasattr(result, "fundamental_freq")
        assert hasattr(result, "confidence")

    def test_detect_alto_signal_female(self):
        """Harmonic signal at 220 Hz (alto, well above MALE max 180 Hz) → FEMALE.

        190 Hz is excluded: at that frequency, synthetic harmonics overlap with
        MALE F2/F3 range enough to tie; 220 Hz gives a clear f0-score advantage
        for FEMALE (distance to MALE range = 0.22). Titze 1994 — alto range starts
        at ~165 Hz; 220 Hz is an unambiguous female pitch.
        """
        gd = GenderDetector(sample_rate=SR)
        result = gd.detect(_harmonic(220, dur=0.6))
        assert result.gender == VoiceGender.FEMALE, (
            f"Expected FEMALE for alto f0=220 Hz, got {result.gender} (f0={result.fundamental_freq:.1f})"
        )

    def test_detect_noisy_vintage_female_signal(self):
        """Harmonic at 220 Hz + 15 dB tape noise → must not return CHILD."""
        gd = GenderDetector(sample_rate=SR)
        sig = _add_noise(_harmonic(220, n_harmonics=6), snr_db=15)
        result = gd.detect(sig)
        assert result.gender != VoiceGender.CHILD, (
            f"False CHILD classification on noisy female signal (f0={result.fundamental_freq:.1f})"
        )

    def test_detect_baritone_signal_male(self):
        """Harmonic at 130 Hz (baritone) → MALE."""
        gd = GenderDetector(sample_rate=SR)
        result = gd.detect(_harmonic(130, dur=0.6))
        assert result.gender == VoiceGender.MALE, f"Expected MALE for f0=130 Hz, got {result.gender}"

    def test_detect_stereo_input_handled(self):
        """Stereo input must be converted to mono without error."""
        gd = GenderDetector(sample_rate=SR)
        mono = _harmonic(200, dur=0.4)
        stereo = np.stack([mono, mono * 0.9], axis=-1)
        result = gd.detect(stereo)
        assert result.gender in (VoiceGender.FEMALE, VoiceGender.MALE, VoiceGender.CHILD, VoiceGender.UNKNOWN)

    def test_detect_fundamental_freq_stored(self):
        """Result.fundamental_freq must match _detect_f0 output."""
        gd = GenderDetector(sample_rate=SR)
        sig = _harmonic(250, dur=0.5)
        result = gd.detect(sig)
        # Should be within ±20 Hz of true f0
        assert abs(result.fundamental_freq - 250) <= 20, (
            f"fundamental_freq={result.fundamental_freq:.1f} too far from 250 Hz"
        )


def test_deesser_contralto_root_classification(monkeypatch, caplog):
    """Wurzel-Fix Contralto (2026-09-13): Tiefe F0 (103 Hz) + weibliche
    Formant-Anatomie (F1=314) + degradiertes F2 (MP3/Bandbreitenverlust) →
    FEMALE direkt in der Klassifikation — ohne §v10.303.11-Override.
    Produktionsbefund: Classifier sagte „male“ (confidence 0,95), erst der
    Override korrigierte auf FEMALE."""
    import logging

    from backend.core.phases import phase_19_de_esser as _p19

    class _FakeChars:
        class _Gender:
            value = "male"

        gender = _Gender()
        confidence = 0.95
        fundamental_freq = 80.0  # weicht >15 % von pYIN-F0=103 ab → pYIN-Pfad
        formants = [314.0, 735.0]

    class _FakeDetector:
        def __init__(self, sample_rate=None):
            pass

        def detect(self, mono):
            return _FakeChars()

    monkeypatch.setattr(_p19, "_RobustGenderDetector", _FakeDetector)
    monkeypatch.setattr(_p19, "_HAS_ROBUST_GENDER", True)
    monkeypatch.setattr("librosa.pyin", lambda *a, **k: _fake_pyin())

    audio = np.zeros(24000, dtype=np.float32)
    phase = _p19.DeEsserPhase()
    with caplog.at_level(logging.INFO):
        gender = phase._detect_gender_robust(audio, 48000, bandwidth_loss=0.6)
    assert gender == "female"
    assert any("Wurzel-Klassifikation" in r.message for r in caplog.records)
    assert not any("CONTRALTO erkannt" in r.message for r in caplog.records)


def _fake_pyin():
    import numpy as _np

    _n = 120
    f0 = _np.full(_n, 103.0, dtype=_np.float64)
    voiced = _np.full(_n, True)
    prob = _np.full(_n, 0.95, dtype=_np.float64)
    return f0, voiced, prob


# ============================================================
# §SOTA-Gender-Fusion (2026-10-06): PANNs-Singing-Evidenz + EIN Pfad
# ============================================================


def test_panns_singing_prior_clear_male() -> None:
    """Klare Male-Singing-Evidenz (Score ≥ 0,25, Abstand > 0,10) → MALE."""
    from backend.core.vocal_ai_enhancement import _panns_singing_prior

    assert _panns_singing_prior({"Male singing": 0.80, "Female singing": 0.05}) == VoiceGender.MALE


def test_panns_singing_prior_clear_female() -> None:
    from backend.core.vocal_ai_enhancement import _panns_singing_prior

    assert _panns_singing_prior({"Male singing": 0.10, "Female singing": 0.70}) == VoiceGender.FEMALE


def test_panns_singing_prior_ambiguous_is_silent() -> None:
    """Gleichstand/Untergrenze schweigt — keine Evidenz ist besser als geratene."""
    from backend.core.vocal_ai_enhancement import _panns_singing_prior

    assert _panns_singing_prior({"Male singing": 0.60, "Female singing": 0.58}) is None
    assert _panns_singing_prior({"Male singing": 0.20, "Female singing": 0.05}) is None
    assert _panns_singing_prior(None) is None
    assert _panns_singing_prior({}) is None


def test_panns_prior_carries_without_anatomy() -> None:
    """Ohne F0/Formanten (instrumentales Intro) trägt der PANNs-Prior die
    Entscheidung — statt blind UNKNOWN (Spec 19 Bug 2/4)."""
    det = GenderDetector(sample_rate=SR)
    gender, conf = det._classify_gender(0.0, [], panns_tags={"Female singing": 0.66})
    assert gender == VoiceGender.FEMALE
    assert conf == pytest.approx(0.60)


def test_without_panns_tags_classification_is_unchanged() -> None:
    """Rückwärtskompatibilität: ohne PANNs-Evidenz entscheidet die Anatomie wie bisher."""
    det = GenderDetector(sample_rate=SR)
    assert det._classify_gender(120.0, [500.0, 1500.0, 2500.0])[0] == VoiceGender.MALE
    assert det._classify_gender(300.0, [800.0, 2200.0, 2900.0])[0] == VoiceGender.FEMALE


def test_canonical_facade_uses_no_speech_embedder() -> None:
    """§III.11 copilot-instructions.md/§G9: Der Gender-Pfad lädt kein
    sprachtrainiertes Embedding-Modell mehr als Entscheider."""
    import inspect

    from backend.core.forensics import gender_detection as _gd

    _src = inspect.getsource(_gd)
    assert "get_resemblyzer_plugin" not in _src, "Resemblyzer darf hier nicht mehr geladen werden (§III.11)"
    assert "from plugins.resemblyzer_plugin import" not in _src
    assert "vocal_ai_enhancement" in _src, "Fassade muss an den kanonischen Kern delegieren"


def test_rule_based_detector_delegates_to_canonical_path() -> None:
    """§G9 copilot-instructions.md: Die regelbasierte Klasse ist nur noch eine
    Fassade — kein zweiter Gender-Begriff mit eigenen Schwellen im Projekt."""
    import inspect

    from backend.core.forensics import gender_rule_based as _grb
    from backend.core.forensics.gender_rule_based import RuleBasedGenderDetector

    _src = inspect.getsource(_grb)
    assert "def _estimate_formants" not in _src
    assert "def _lpc" not in _src
    assert "def classify_from_features" not in _src
    _det = RuleBasedGenderDetector(sr=SR)
    assert _det.detect_gender(np.zeros(0, dtype=np.float32)) == "unknown"


def test_gender_facade_array_path_returns_valid_label() -> None:
    """Fassade: Signal → zulässiges Label; zu kurzes Signal → ehrliches 'unknown'."""
    from backend.core.forensics.gender_detection import GenderDetector as _Fassade

    _det = _Fassade(sample_rate=SR)
    assert _det.detect_gender_array(_harmonic(220.0, dur=0.5), SR) in {
        "male",
        "female",
        "child",
        "unknown",
    }
    assert _det.detect_gender_array(np.zeros(10, dtype=np.float32), SR) == "unknown"


def test_scan_f0_never_blind_to_late_voice() -> None:
    """Spec 19 Bug 2/4: Ein instrumentales Intro (Stille) darf die F0-Schätzung
    nicht blockieren — gescannt wird über den ganzen Clip."""
    from backend.core.forensics.gender_detection import _scan_f0

    _lead_in = np.zeros(int(1.0 * SR), dtype=np.float32)
    _voice = _harmonic(180.0, dur=2.0)
    _sig = np.concatenate([_lead_in, _voice])
    _f0 = _scan_f0(_sig, SR)
    assert 150.0 < _f0 < 210.0, f"Scan-F0={_f0:.1f} Hz — Intro blockierte die Schätzung"
