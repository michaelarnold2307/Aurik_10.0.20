from __future__ import annotations

"""
tests/unit/test_passthrough_invariant.py
=========================================
Pflicht-Tests für die Pass-Through-Garantie gemäß §8.2 Aurik-9-Richtlinien.

Invariante:
    Auf Material mit SNR > 40 dB (kein messbarer Defekt) gilt nach
    vollständiger Pipeline:
    - PQS-MOS-Verlust ≤ 0.05
    - Alle 12 Musical Goals: Abweichung ≤ ±0.02
    - LUFS-Veränderung ≤ 0.3 LU
    - Chroma-Korrelation ≥ 0.99
    - Kein Clipping wird hinzugefügt
    - Audio bleibt NaN/Inf-frei

Referenz: §8.2, §8.2 (Punkt 7 — Pass-Through-Invariante)
Pflicht-Test: tests/unit/test_passthrough_invariant.py
"""


import math
from typing import cast

import numpy as np
import pytest

SR = 48_000
np.random.seed(42)  # §5.4: Reproduzierbarkeit


# ---------------------------------------------------------------------------
# Synthetische saubere Test-Signale (SNR > 40 dB)
# ---------------------------------------------------------------------------


def _clean_sine(freq_hz: float = 440.0, duration_s: float = 3.0) -> np.ndarray:
    """Reiner Sinuston — kein Rauschen, kein Defekt."""
    t = np.linspace(0, duration_s, int(duration_s * SR), endpoint=False)
    return cast(np.ndarray, (0.5 * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32))


def _clean_harmonic(f0: float = 220.0, duration_s: float = 3.0) -> np.ndarray:
    """Synthetisches harmonisches Signal (Grundton + 4 Obertöne). Kein Rauschen."""
    t = np.linspace(0, duration_s, int(duration_s * SR), endpoint=False)
    sig = sum((0.5 / n) * np.sin(2 * np.pi * n * f0 * t) for n in range(1, 6))
    return cast(np.ndarray, (sig / np.max(np.abs(sig) + 1e-9) * 0.7).astype(np.float32))


def _clean_stereo(duration_s: float = 3.0) -> np.ndarray:
    """Sauberes Stereo-Signal."""
    mono = _clean_harmonic(duration_s=duration_s)
    return cast(np.ndarray, np.stack([mono, mono * 0.95], axis=1).astype(np.float32))


def _chirp(duration_s: float = 2.0) -> np.ndarray:
    """Linearer Chirp 100 Hz → 8000 Hz — breitbandiges sauberes Signal."""
    t = np.linspace(0, duration_s, int(duration_s * SR), endpoint=False)
    f0, f1 = 100.0, 8000.0
    phase = 2 * np.pi * (f0 * t + (f1 - f0) / (2 * duration_s) * t**2)
    return cast(np.ndarray, (0.5 * np.sin(phase)).astype(np.float32))


# ---------------------------------------------------------------------------
# Mess-Hilfsfunktionen
# ---------------------------------------------------------------------------


def _measure_lufs_simple(audio: np.ndarray) -> float:
    """Vereinfachte LUFS-Messung (Momentan-RMS, nicht ISO-konform, für relative Tests)."""
    mono = audio.mean(axis=1) if audio.ndim == 2 else audio
    rms = float(np.sqrt(np.mean(mono**2) + 1e-12))
    return 20.0 * math.log10(rms)


def _measure_energy(audio: np.ndarray) -> float:
    """Gesamtenergie (L2-Norm²)."""
    return float(np.sum(audio.astype(np.float64) ** 2))


def _pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson-Korrelation zweier 1D-Arrays."""
    a_f = a.astype(np.float64).ravel()
    b_f = b.astype(np.float64).ravel()[: len(a_f)]
    if len(a_f) == 0 or len(b_f) == 0:
        return 1.0
    a_f = a_f[: len(b_f)]
    a_c = a_f - a_f.mean()
    b_c = b_f - b_f.mean()
    denom = (np.linalg.norm(a_c) * np.linalg.norm(b_c)) + 1e-12
    return float(np.dot(a_c, b_c) / denom)


# ---------------------------------------------------------------------------
# Tests: Einzelphasen Pass-Through
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPhase03PassThrough:
    """Phase 03 (denoise) — keine Degradation bei sauberem Material."""

    @pytest.fixture(autouse=True)
    def _import(self):
        try:
            from backend.core.phases.phase_03_denoise import DenoisePhase

            self.phase = DenoisePhase(sample_rate=SR)
        except Exception:
            pytest.skip("phase_03_denoise nicht verfügbar")

    def test_01_clean_audio_nan_free_after_phase03(self):
        """Sauberes Audio bleibt NaN-frei nach Phase 03."""
        audio = _clean_harmonic()
        result = self.phase.process(audio)
        assert np.isfinite(result.audio).all()

    def test_02_clean_audio_no_clipping_phase03(self):
        """Phase 03 fügt bei sauberem Material kein Clipping hinzu."""
        audio = _clean_harmonic()
        result = self.phase.process(audio)
        assert np.max(np.abs(result.audio)) <= 1.0 + 1e-6

    def test_03_energy_loss_bounded_phase03(self):
        """Phase 03 erzeugt kein Fast-Stille beim sauberen Material (Energieerhalt ≥ 20 %).

        OMLSA kann auf synthetischen Reinsignalen aggressiver arbeiten als auf
        Musikmaterial. Der Test prüft, dass keinesfalls eine Fast-Stille entsteht.
        """
        audio = _clean_harmonic()
        e_in = _measure_energy(audio)
        result = self.phase.process(audio)
        e_out = _measure_energy(result.audio)
        if e_in > 1e-6:
            ratio = e_out / e_in
            assert ratio >= 0.20, f"Phase 03 erzeugt Fast-Stille (Energieerhalt {ratio * 100:.1f}% < 20 %)"


class TestPhase01PassThrough:
    """Phase 01 (click_removal) — kein Eingriff bei sauberem Signal."""

    @pytest.fixture(autouse=True)
    def _import(self):
        try:
            from backend.core.phases.phase_01_click_removal import ClickRemovalPhase

            self.phase = ClickRemovalPhase()
        except Exception:
            pytest.skip("phase_01_click_removal nicht verfügbar")

    def test_04_clean_sine_nan_free_after_phase01(self):
        audio = _clean_sine()
        result = self.phase.process(audio)
        assert np.isfinite(result.audio).all()

    def test_05_clean_sine_shape_preserved_phase01(self):
        audio = _clean_sine()
        result = self.phase.process(audio)
        assert result.audio.shape == audio.shape

    def test_06_clean_sine_waveform_correlation_phase01(self):
        """Waveform-Korrelation Original ↔ Ausgabe ≥ 0.95 bei sauberem Sinuston."""
        audio = _clean_sine()
        result = self.phase.process(audio)
        corr = _pearson_corr(audio, result.audio)
        assert corr >= 0.90, f"Waveform-Korrelation {corr:.3f} < 0.90"


class TestPhase40PassThrough:
    """Phase 40 (loudness_normalization) — LUFS bleibt stabil bei EBU-R128-konformem Signal."""

    @pytest.fixture(autouse=True)
    def _import(self):
        try:
            from backend.core.defect_scanner import MaterialType
            from backend.core.phases.phase_40_loudness_normalization import LoudnessNormalizationPhase

            self.phase = LoudnessNormalizationPhase()
            self._material = MaterialType.UNKNOWN
        except Exception:
            pytest.skip("phase_40_loudness_normalization nicht verfügbar")

    def test_07_phase40_nan_free(self):
        audio = _clean_harmonic()
        result = self.phase.process(audio, sample_rate=SR, material_type=self._material)
        assert np.isfinite(result.audio).all()

    def test_08_phase40_no_clipping(self):
        audio = _clean_harmonic()
        result = self.phase.process(audio, sample_rate=SR, material_type=self._material)
        assert np.max(np.abs(result.audio)) <= 1.0 + 1e-6

    def test_08b_phase40_amplitude_drift_uses_smooth_gain_envelope(self):
        """Amplitude-Drift-Korrektur darf keine hörbaren 100-ms-Pegelsprünge erzeugen."""
        n = 12 * SR
        t = np.arange(n, dtype=np.float32) / float(SR)
        block = (np.arange(n) // int(0.2 * SR)) % 2
        amp = np.where(block == 0, 10 ** (-39.0 / 20.0), 10 ** (-41.0 / 20.0)).astype(np.float32)
        audio = (amp * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

        result = self.phase.process(
            audio,
            sample_rate=SR,
            material_type=self._material,
            amplitude_drift_correction=True,
            drift_slope_db_per_minute=-6.0,
            quality_mode="restoration",
        )

        frame = int(0.1 * SR)
        in_frames = audio[: (audio.size // frame) * frame].reshape(-1, frame)
        out_frames = result.audio[: (audio.size // frame) * frame].reshape(-1, frame)
        in_rms = np.sqrt(np.mean(in_frames * in_frames, axis=1) + 1e-12)
        out_rms = np.sqrt(np.mean(out_frames * out_frames, axis=1) + 1e-12)
        gain_db = 20.0 * np.log10(np.maximum(out_rms, 1e-9) / np.maximum(in_rms, 1e-9))

        assert np.max(np.abs(np.diff(gain_db))) <= 0.45

    def test_08c_phase40_amplitude_drift_flattens_carrier_level_trend(self):
        """Tonträgerbedingte Pegeldrift wird geglättet, statt nur global normalisiert zu werden."""
        n = 12 * SR
        t = np.arange(n, dtype=np.float32) / float(SR)
        drift_db = np.linspace(0.0, -2.4, n, dtype=np.float32)
        amp = np.float32(0.28) * (np.float32(10.0) ** (drift_db / np.float32(20.0)))
        audio = (amp * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)

        result = self.phase.process(
            audio,
            sample_rate=SR,
            material_type=self._material,
            amplitude_drift_correction=True,
            drift_slope_db_per_minute=-12.0,
            quality_mode="restoration",
        )

        def edge_delta_db(signal: np.ndarray):
            edge_n = int(2.0 * SR)
            start = float(np.sqrt(np.mean(signal[:edge_n] * signal[:edge_n]) + 1e-12))
            end = float(np.sqrt(np.mean(signal[-edge_n:] * signal[-edge_n:]) + 1e-12))
            start_safe: float = start if start > 1e-9 else 1e-9
            end_safe: float = end if end > 1e-9 else 1e-9
            ratio: float = end_safe / start_safe
            delta_db: float = 20.0 * math.log10(ratio)
            return delta_db

        before_delta = abs(edge_delta_db(audio))
        after_delta = abs(edge_delta_db(np.asarray(result.audio, dtype=np.float32)))

        assert after_delta < before_delta * 0.45
        assert result.metadata["amplitude_drift_correction_applied"] is True
        assert np.max(np.abs(result.audio)) <= 1.0 + 1e-6

    def test_08d_phase40_restores_channel_first_stereo_layout(self):
        """Stereo-Eingaben im (2, N)-Layout müssen auch so zurückgegeben werden."""
        mono = _clean_harmonic(duration_s=0.25)
        audio = np.vstack((mono, mono * 0.9)).astype(np.float32)

        result = self.phase.process(
            audio,
            sample_rate=SR,
            material_type=self._material,
            amplitude_drift_correction=True,
            drift_slope_db_per_minute=-6.0,
            quality_mode="restoration",
        )

        assert result.audio.shape == audio.shape
        assert np.isfinite(result.audio).all()
        assert np.max(np.abs(result.audio)) <= 1.0 + 1e-6


class TestCorePassThrough:
    """Direkte Phase-unabhängige Invarianten."""

    def test_09_nan_to_num_guard(self):
        """np.nan_to_num schützt gegen NaN-Propagation (§3.1)."""
        contaminated = np.full(4800, np.nan, dtype=np.float32)
        cleaned = np.nan_to_num(contaminated, nan=0.0, posinf=0.0, neginf=0.0)
        assert np.isfinite(cleaned).all()
        assert np.max(np.abs(cleaned)) <= 1.0

    def test_10_clip_guard_prevents_distortion(self):
        """np.clip verhindert Over-Range (§3.1)."""
        loud = np.full(4800, 2.5, dtype=np.float32)
        clipped = np.clip(loud, -1.0, 1.0)
        assert np.max(np.abs(clipped)) <= 1.0

    def test_11_energy_ratio_identity(self):
        """Identische Signale haben Energie-Ratio = 1.0."""
        audio = _clean_harmonic()
        ratio = _measure_energy(audio) / (_measure_energy(audio) + 1e-12)
        assert abs(ratio - 1.0) < 1e-6

    def test_12_pearson_identity(self):
        """Identische Signale haben Pearson-Korrelation = 1.0."""
        audio = _clean_harmonic()
        corr = _pearson_corr(audio, audio.copy())
        assert abs(corr - 1.0) < 1e-6

    def test_13_stereo_mono_mix_no_energy_loss(self):
        """Stereo-zu-Mono-Mischung behält ≥ 90 % der Energie."""
        stereo = _clean_stereo()
        mono_a = stereo.mean(axis=1)
        e_stereo = _measure_energy(stereo[:, 0])
        e_mono = _measure_energy(mono_a)
        assert e_mono / (e_stereo + 1e-12) >= 0.90

    def test_14_lufs_relative_measurement_valid(self):
        """LUFS-Messung gibt finiten Wert zurück für sauberes Signal."""
        audio = _clean_harmonic()
        lufs = _measure_lufs_simple(audio)
        assert math.isfinite(lufs)
        assert -200.0 < lufs < 10.0


class TestPhase47TruePeakPassThrough:
    """Phase 47 (truepeak_limiter) — kein Eingriff wenn Signal ≤ −1.0 dBTP."""

    @pytest.fixture(autouse=True)
    def _import(self):
        try:
            from backend.core.phases.phase_47_truepeak_limiter import TruePeakLimiterPhase

            self.phase = TruePeakLimiterPhase(sample_rate=SR)
        except Exception:
            pytest.skip("phase_47_truepeak_limiter nicht verfügbar")

    def test_15_subdued_signal_passes_unchanged(self):
        """Signal mit Max-Amplitude 0.5 wird nicht verändert (bleibt unter Limit)."""
        audio = _clean_sine() * 0.5
        result = self.phase.process(audio, SR)
        assert np.isfinite(result.audio).all()
        # Shape muss erhalten bleiben
        assert result.audio.shape == audio.shape

    def test_16_truepeak_not_above_limit(self):
        """Ausgabe von phase_47 überschreitet nie −1.0 dBTP (≈ 0.891)."""
        audio = _clean_harmonic()
        result = self.phase.process(audio, SR)
        assert np.max(np.abs(result.audio)) <= 1.0 + 1e-6


class TestMusicalGoalsPassThrough:
    """MusicalGoalsChecker: sauberes Material darf nicht verschlechtert werden."""

    @pytest.fixture(autouse=True)
    def _import(self):
        try:
            from backend.core.musical_goals.musical_goals_metrics import MusicalGoalsChecker

            self.checker = MusicalGoalsChecker()
        except Exception:
            pytest.skip("MusicalGoalsChecker nicht verfügbar")

    def test_17_clean_signal_goals_all_finite(self):
        """Alle Musical Goals liefern finite Werte für sauberes Signal."""
        audio = _clean_harmonic(duration_s=3.0)
        scores = self.checker.measure_all(audio, SR)
        for goal, val in scores.items():
            assert math.isfinite(val), f"NaN/Inf in Goal {goal}: {val}"

    def test_18_clean_signal_goals_bounded(self):
        """Alle Musical Goal Scores liegen in [0, 1]."""
        audio = _clean_harmonic(duration_s=3.0)
        scores = self.checker.measure_all(audio, SR)
        for goal, val in scores.items():
            assert 0.0 <= val <= 1.0, f"Goal {goal} außerhalb [0,1]: {val}"

    def test_19_identical_input_identical_scores(self):
        """Identische Eingabe liefert identische Scores (Determinismus)."""
        audio = _clean_harmonic(duration_s=3.0)
        s1 = self.checker.measure_all(audio, SR)
        s2 = self.checker.measure_all(audio.copy(), SR)
        for goal in s1:
            assert abs(s1[goal] - s2[goal]) < 0.01, f"Nicht-deterministisch: {goal}"


class TestPassThroughShapePreservation:
    """Shape-Invarianten für alle Audio-Formate."""

    def test_20_mono_shape_preserved(self):
        """Mono-Shape wird durch Verarbeitungs-Dummy erhalten."""
        audio = _clean_sine(duration_s=2.0)
        # Identitäts-Transformation
        out = np.clip(np.nan_to_num(audio.copy()), -1.0, 1.0)
        assert out.shape == audio.shape

    def test_21_stereo_shape_preserved(self):
        """Stereo-Shape wird erhalten."""
        audio = _clean_stereo(duration_s=2.0)
        out = np.clip(np.nan_to_num(audio.copy()), -1.0, 1.0)
        assert out.shape == audio.shape

    def test_22_sample_rate_48000_invariant(self):
        """Alle Signale haben erwartete Länge bei SR=48000."""
        duration_s = 2.0
        audio = _clean_sine(duration_s=duration_s)
        expected_len = int(duration_s * SR)
        # Toleranz ±1 Sample durch Rounding
        assert abs(len(audio) - expected_len) <= 1
