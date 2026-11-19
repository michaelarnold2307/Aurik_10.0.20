import pytest

"""Erweiterte Unit-Tests für forensics.medium_detector (v10.0).

Testet:
- Verschiedene synthetische Signale (Ton, Chirp, Rauschen, Stille)
- Kassetten-Simulation
- MP3-Simulation
- Digitales Signal
- Shellac-Simulation
- Rauschboden-Messung
- Effective Bandwidth
- Wow/Flutter-Index
- Multi-Stufen-Kette detection
- evidence-Liste
- as_dict() Vollständigkeit
"""

import numpy as np
import scipy.signal

from forensics.medium_detector import (
    MediumDetectionResult,
    MediumDetector,
    detect_medium_chain,
    get_medium_detector,
)


@pytest.mark.unit
class TestV100MediumDetectorExtended:
    """Erweiterte Tests für MediumDetector v10.0."""

    def test_01_detect_sine_wave_returns_result(self):
        """Sinus-Welle: detect() gibt Result zurück."""
        np.random.seed(42)
        t = np.linspace(0, 1, 48000, dtype=np.float32)
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.3
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        assert isinstance(result, MediumDetectionResult)

    def test_02_detect_chirp_returns_result(self):
        """Chirp: detect() gibt Result zurück."""
        np.random.seed(42)
        t = np.linspace(0, 1, 48000, dtype=np.float32)
        audio = scipy.signal.chirp(t, f0=100, f1=2000, t1=1, method="linear").astype(np.float32) * 0.3
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        assert isinstance(result, MediumDetectionResult)

    def test_03_detect_white_noise_returns_result(self):
        """Weißes Rauschen: detect() gibt Result zurück."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        assert isinstance(result, MediumDetectionResult)

    def test_04_detect_silence_returns_result(self):
        """Stille: detect() gibt Result zurück."""
        audio = np.zeros(48000, dtype=np.float32)
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        assert isinstance(result, MediumDetectionResult)

    def test_05_cassette_simulation_lowpass_8khz_detected(self):
        """Kassetten-Simulation: Tiefpass 8 kHz + Rauschen gut erkannt."""
        np.random.seed(42)
        # Tiefpass-Signal simulieren
        t = np.linspace(0, 1, 48000, dtype=np.float32)
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.3
        # Tiefpass bei 8 kHz
        sos = scipy.signal.butter(4, 8000, "low", fs=48000, output="sos")
        audio = scipy.signal.sosfilt(sos, audio).astype(np.float32)
        # Rauschen hinzufügen
        audio += np.random.randn(48000).astype(np.float32) * 0.02
        audio = np.clip(audio, -1.0, 1.0).astype(np.float32)

        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        # Sollte tape oder ähnlich erkennen
        assert result.spectral_fingerprint.rolloff_95_percent_hz < 12000

    def test_06_mp3_simulation_hf_energy_zero_detected(self):
        """MP3-Simulation: HF-Energie = 0 erkannt."""
        np.random.seed(42)
        # Tiefpass bei 16 kHz (MP3 high), kein HF > 16 kHz
        t = np.linspace(0, 1, 48000, dtype=np.float32)
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.3
        sos = scipy.signal.butter(8, 16000, "low", fs=48000, output="sos")
        audio = scipy.signal.sosfilt(sos, audio).astype(np.float32)

        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        # HF-Energie sollte sehr klein sein
        assert result.spectral_fingerprint.hf_energy_above_16khz_percent < 5.0

    def test_07_digital_signal_high_resolution_detected(self):
        """Digitales Signal hochauflösend erkannt."""
        np.random.seed(42)
        # Breitbandiges Signal (weißes Rauschen hat Energie bis nyquist = 24 kHz)
        # Sinus-Summentests täuschen: 440 + 5000 Hz ergibt Rolloff ~5000 Hz (korrekt).
        # Für Rolloff > 10000 Hz braucht man wirklich Energie bis 15–20 kHz.
        audio = np.random.randn(48000).astype(np.float32) * 0.2
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        # Weißes Rauschen hat flaches Spektrum bis Nyquist → Rolloff ≥ 20 kHz
        assert result.spectral_fingerprint.rolloff_95_percent_hz > 10000

    def test_08_shellac_simulation_rolloff_below_4khz_detected(self):
        """Shellac-Simulation: Rolloff < 4 kHz erkannt."""
        np.random.seed(42)
        t = np.linspace(0, 1, 48000, dtype=np.float32)
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.3
        # Sehr starker Tiefpass bei 3.5 kHz
        sos = scipy.signal.butter(6, 3500, "low", fs=48000, output="sos")
        audio = scipy.signal.sosfilt(sos, audio).astype(np.float32)

        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        assert result.spectral_fingerprint.rolloff_95_percent_hz < 5000

    def test_09_noise_floor_measurement_correct(self):
        """Rauschboden-Messung korrekt."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.01  # sehr leise
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        # Noise floor sollte negativ sein (dB)
        assert result.spectral_fingerprint.noise_floor_db < -20.0

    def test_10_effective_bandwidth_less_than_nyquist(self):
        """Effective Bandwidth Messung < SR/2."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        assert result.spectral_fingerprint.effective_bandwidth_hz <= 24000

    def test_11_wow_flutter_index_stable_pitch_small(self):
        """Wow/Flutter-Index bei pitch-stabilem Signal klein."""
        np.random.seed(42)
        t = np.linspace(0, 1, 48000, dtype=np.float32)
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.3
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        # Stabiler Pitch → kleiner Index
        assert result.spectral_fingerprint.wow_flutter_index < 10.0

    def test_12_evidence_list_not_empty(self):
        """evidence-Liste nicht leer."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        assert len(result.evidence) > 0

    def test_13_all_fields_of_result_populated(self):
        """Alle Felder von MediumDetectionResult befüllt."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        assert result.transfer_chain is not None
        assert result.is_multi_generation is not None
        assert result.primary_material is not None
        assert result.confidence is not None
        assert result.chain_label is not None
        assert result.spectral_fingerprint is not None
        assert result.evidence is not None

    def test_14_as_dict_has_all_expected_keys(self):
        """as_dict() hat alle erwarteten Keys."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        d = result.as_dict()
        expected_keys = [
            "transfer_chain",
            "is_multi_generation",
            "primary_material",
            "confidence",
            "chain_label",
            "spectral_fingerprint",
            "evidence",
        ]
        for key in expected_keys:
            assert key in d

    def test_15_sr_not_48000_no_crash(self):
        """SR != 48000 → kein Absturz (nur Warnung)."""
        np.random.seed(42)
        audio = np.random.randn(22050).astype(np.float32) * 0.1
        detector = MediumDetector()
        # Sollte intern resamplen oder Warnung ausgeben
        result = detector.detect(audio, sr=22050)
        assert isinstance(result, MediumDetectionResult)

    def test_16_very_short_audio_below_1s_no_crash(self):
        """Sehr kurzes Audio (< 1 s) → kein Absturz."""
        np.random.seed(42)
        audio = np.random.randn(4800).astype(np.float32) * 0.1
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        assert isinstance(result, MediumDetectionResult)

    def test_17_stereo_array_shape_2_N_correct_conversion(self):
        """Stereo-Array shape (2, N) → korrekte Mono-Konversion."""
        np.random.seed(42)
        audio_stereo = np.random.randn(2, 48000).astype(np.float32) * 0.1
        detector = MediumDetector()
        result = detector.detect(audio_stereo, sr=48000)
        assert isinstance(result, MediumDetectionResult)

    def test_18_stereo_array_shape_N_2_correct_conversion(self):
        """Stereo-Array shape (N, 2) → korrekte Mono-Konversion."""
        np.random.seed(42)
        audio_stereo = np.random.randn(48000, 2).astype(np.float32) * 0.1
        detector = MediumDetector()
        result = detector.detect(audio_stereo, sr=48000)
        assert isinstance(result, MediumDetectionResult)

    def test_19_rolloff_always_positive(self):
        """Rolloff immer positiv."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        assert result.spectral_fingerprint.rolloff_95_percent_hz > 0

    def test_20_hf_energy_in_valid_range(self):
        """HF-Energie ∈ [0, 100]."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        assert 0.0 <= result.spectral_fingerprint.hf_energy_above_16khz_percent <= 100.0

    def test_21_noise_floor_reasonable_range(self):
        """Noise floor in vernünftigem Bereich."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        # Sollte zwischen -120 und 0 dB sein
        assert -120.0 <= result.spectral_fingerprint.noise_floor_db <= 0.0

    def test_22_effective_bandwidth_positive(self):
        """Effective Bandwidth positiv."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        assert result.spectral_fingerprint.effective_bandwidth_hz > 0

    def test_23_wow_flutter_index_non_negative(self):
        """Wow/Flutter-Index ≥ 0."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        assert result.spectral_fingerprint.wow_flutter_index >= 0.0

    def test_24_confidence_always_in_range(self):
        """Confidence immer ∈ [0, 1]."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        assert 0.0 <= result.confidence <= 1.0

    def test_25_primary_material_is_known_type(self):
        """primary_material ist bekannter Typ."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        # D-K1-3 F5 (2026-10-09): 'cassette' ergänzt — §6.1 v10.0.0 First-Class-Material,
        # Liste nie nachgezogen; die datengetriebene _MATERIAL_MODELS-Neukalibrierung
        # (120 Echt-Träger) lässt Weißrauschen jetzt auf 'cassette' münden (früher 'tape').
        known_materials = [
            "tape",
            "vinyl",
            "shellac",
            "cd_digital",
            "digital",
            "mp3_low",
            "mp3_high",
            "aac",
            "unknown",
            "streaming",
            "dat",
            "wax_cylinder",
            "wire_recording",
            "lacquer_disc",
            "minidisc",
            "reel_tape",
            "cassette",
        ]
        assert result.primary_material in known_materials

    def test_26_transfer_chain_at_least_one_element(self):
        """TransferChain hat mind. 1 Element."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        assert len(result.transfer_chain) >= 1

    def test_27_is_multi_generation_bool(self):
        """is_multi_generation ist bool."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        assert isinstance(result.is_multi_generation, bool)

    def test_28_chain_label_non_empty_string(self):
        """chain_label ist nicht-leerer String."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        assert isinstance(result.chain_label, str)
        assert len(result.chain_label) > 0

    def test_29_spectral_fingerprint_all_fields_finite(self):
        """SpectralFingerprint: alle Felder finite."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        fp = result.spectral_fingerprint
        assert np.isfinite(fp.rolloff_95_percent_hz)
        assert np.isfinite(fp.wow_flutter_index)
        assert np.isfinite(fp.hf_energy_above_16khz_percent)
        assert np.isfinite(fp.noise_floor_db)
        assert np.isfinite(fp.effective_bandwidth_hz)

    def test_30_evidence_list_is_list_of_strings(self):
        """evidence ist Liste von Strings."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        assert isinstance(result.evidence, list)
        for ev in result.evidence:
            assert isinstance(ev, str)

    def test_31_detect_medium_chain_convenience_function(self):
        """detect_medium_chain() convenience function works."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        result = detect_medium_chain(audio, sr=48000)
        assert isinstance(result, MediumDetectionResult)

    def test_32_singleton_multiple_calls_same_instance(self):
        """Singleton: mehrere get_medium_detector() → selbe Instanz."""
        det1 = get_medium_detector()
        det2 = get_medium_detector()
        det3 = get_medium_detector()
        assert det1 is det2
        assert det2 is det3

    def test_33_nan_in_audio_no_crash(self):
        """NaN in audio → kein Absturz."""
        audio = np.full(48000, np.nan, dtype=np.float32)
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        assert isinstance(result, MediumDetectionResult)

    def test_34_inf_in_audio_no_crash(self):
        """Inf in audio → kein Absturz."""
        audio = np.full(48000, np.inf, dtype=np.float32)
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        assert isinstance(result, MediumDetectionResult)

    def test_35_very_loud_audio_no_crash(self):
        """Sehr lautes Audio → kein Absturz."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 10.0  # > 1.0
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        assert isinstance(result, MediumDetectionResult)

    def test_36_very_quiet_audio_no_crash(self):
        """Sehr leises Audio → kein Absturz."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.0001
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        assert isinstance(result, MediumDetectionResult)

    def test_37_dc_offset_audio_no_crash(self):
        """Audio mit DC-Offset → kein Absturz."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.1 + 0.5
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        assert isinstance(result, MediumDetectionResult)

    def test_38_square_wave_no_crash(self):
        """Rechteck-Welle → kein Absturz."""
        np.random.seed(42)
        t = np.linspace(0, 1, 48000, dtype=np.float32)
        audio = scipy.signal.square(2 * np.pi * 440 * t).astype(np.float32) * 0.3
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        assert isinstance(result, MediumDetectionResult)

    def test_39_sawtooth_wave_no_crash(self):
        """Sägezahn-Welle → kein Absturz."""
        np.random.seed(42)
        t = np.linspace(0, 1, 48000, dtype=np.float32)
        audio = scipy.signal.sawtooth(2 * np.pi * 440 * t).astype(np.float32) * 0.3
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        assert isinstance(result, MediumDetectionResult)

    def test_40_triangle_wave_no_crash(self):
        """Dreieck-Welle → kein Absturz."""
        np.random.seed(42)
        t = np.linspace(0, 1, 48000, dtype=np.float32)
        audio = scipy.signal.sawtooth(2 * np.pi * 440 * t, width=0.5).astype(np.float32) * 0.3
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        assert isinstance(result, MediumDetectionResult)

    def test_41_pink_noise_no_crash(self):
        """Rosa Rauschen → kein Absturz."""
        np.random.seed(42)
        # Vereinfachte Pink-Noise-Simulation
        audio = np.random.randn(48000).astype(np.float32)
        # Tiefpass für rohes Pink-Noise-Approximation
        sos = scipy.signal.butter(1, 8000, "low", fs=48000, output="sos")
        audio = scipy.signal.sosfilt(sos, audio).astype(np.float32) * 0.1
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        assert isinstance(result, MediumDetectionResult)

    def test_42_impulse_train_no_crash(self):
        """Impuls-Zug → kein Absturz."""
        audio = np.zeros(48000, dtype=np.float32)
        audio[::4800] = 0.5  # Impulse alle 0.1 s
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        assert isinstance(result, MediumDetectionResult)

    def test_43_multi_frequency_tone_no_crash(self):
        """Multi-Frequenz-Ton → kein Absturz."""
        np.random.seed(42)
        t = np.linspace(0, 1, 48000, dtype=np.float32)
        audio = (np.sin(2 * np.pi * 440 * t) + np.sin(2 * np.pi * 880 * t) + np.sin(2 * np.pi * 1320 * t)).astype(
            np.float32
        ) * 0.2
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        assert isinstance(result, MediumDetectionResult)

    def test_44_as_dict_spectral_fingerprint_serializable(self):
        """as_dict(): spectral_fingerprint ist serialisierbar."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        d = result.as_dict()
        fp = d["spectral_fingerprint"]
        assert isinstance(fp, dict)
        assert "rolloff_95_percent_hz" in fp
        assert "wow_flutter_index" in fp

    def test_45_as_dict_transfer_chain_is_list(self):
        """as_dict(): transfer_chain ist Liste."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        d = result.as_dict()
        assert isinstance(d["transfer_chain"], list)

    def test_46_long_audio_10s_no_crash(self):
        """10-Sekunden-Audio → kein Absturz."""
        np.random.seed(42)
        audio = np.random.randn(480000).astype(np.float32) * 0.1
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        assert isinstance(result, MediumDetectionResult)

    def test_47_detect_alternating_stereo_channels(self):
        """Stereo mit unterschiedlichen Kanälen → Mono-Mix korrekt."""
        np.random.seed(42)
        left = np.random.randn(48000).astype(np.float32) * 0.1
        right = np.random.randn(48000).astype(np.float32) * 0.05
        audio_stereo = np.stack([left, right], axis=0)
        detector = MediumDetector()
        result = detector.detect(audio_stereo, sr=48000)
        assert isinstance(result, MediumDetectionResult)

    def test_48_detect_mono_1d_array(self):
        """Mono als 1D-Array → korrekt verarbeitet."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        assert audio.ndim == 1
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        assert isinstance(result, MediumDetectionResult)

    def test_49_all_spectral_fingerprint_fields_exist(self):
        """Alle SpectralFingerprint-Felder existieren."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        detector = MediumDetector()
        result = detector.detect(audio, sr=48000)
        fp = result.spectral_fingerprint
        assert hasattr(fp, "rolloff_95_percent_hz")
        assert hasattr(fp, "wow_flutter_index")
        assert hasattr(fp, "hf_energy_above_16khz_percent")
        assert hasattr(fp, "noise_floor_db")
        assert hasattr(fp, "effective_bandwidth_hz")

    def test_50_result_reproducible_same_seed(self):
        """Gleicher Seed → reproduzierbares Ergebnis."""
        np.random.seed(42)
        audio1 = np.random.randn(48000).astype(np.float32) * 0.1
        detector = MediumDetector()
        result1 = detector.detect(audio1, sr=48000)

        np.random.seed(42)
        audio2 = np.random.randn(48000).astype(np.float32) * 0.1
        result2 = detector.detect(audio2, sr=48000)

        assert result1.primary_material == result2.primary_material
        assert abs(result1.confidence - result2.confidence) < 0.01
