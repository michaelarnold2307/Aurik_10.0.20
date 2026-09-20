from __future__ import annotations

"""
Tests für dsp/pghi.py, dsp/psola.py und dsp/dtw_groove.py

Tests folgen §5.1–§5.4:
- Shape/Dtype-Tests
- NaN/Inf-Tests
- Bounds-Tests (alle Ausgaben in [-1, 1] oder [0, 1])
- Edge-Cases (Stille, weißes Rauschen, Dirac-Impuls)
- Mono + Stereo
- Konsistenz (gleiche Eingabe → gleiche Ausgabe)

Konvention: np.random.seed(42) in jedem Test, nur synthetische Signale.
"""


import math
import threading

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# PGHI-Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPghiReconstructor:
    """Tests für dsp/pghi.py — Phase Gradient Heap Integration."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    # ------------------------------------------------------------------
    # Import & Singleton
    # ------------------------------------------------------------------

    def test_01_import_pghi(self):
        """pghi-Modul ist importierbar."""
        from dsp.pghi import get_pghi_reconstructor

        rec = get_pghi_reconstructor()
        assert rec is not None

    def test_02_singleton_identity(self):
        """get_pghi_reconstructor() gibt dasselbe Objekt zurück."""
        from dsp.pghi import get_pghi_reconstructor

        a = get_pghi_reconstructor()
        b = get_pghi_reconstructor()
        assert a is b

    def test_03_singleton_thread_safe(self):
        """Parallele Zugriffe liefern identisches Singleton."""
        from dsp.pghi import get_pghi_reconstructor

        instances = []
        errors = []

        def fetch():
            try:
                instances.append(get_pghi_reconstructor())
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=fetch) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert all(inst is instances[0] for inst in instances)

    # ------------------------------------------------------------------
    # Hilfs-Methode: Magnitude-Spektrogramm erzeugen
    # ------------------------------------------------------------------

    def _make_magnitude(self, n_samples: int = 48000, n_fft: int = 2048, hop: int = 256) -> np.ndarray:
        """Erzeugt ein zufälliges Magnitude-Spektrogramm [n_bins, n_frames]."""
        n_bins = n_fft // 2 + 1
        n_frames = (n_samples - n_fft) // hop + 1
        return np.abs(np.random.randn(n_bins, n_frames).astype(np.float32)) * 0.3

    # ------------------------------------------------------------------
    # Grundfunktionen
    # ------------------------------------------------------------------

    def test_04_reconstruct_shape(self):
        """reconstruct() gibt 1D-float32-Array zurück."""
        from dsp.pghi import pghi_reconstruct

        mag = self._make_magnitude()
        result = pghi_reconstruct(mag, sr=48000)
        assert result.ndim == 1
        assert len(result) > 0

    def test_05_reconstruct_dtype_float32(self):
        """Ausgabe ist immer float32."""
        from dsp.pghi import pghi_reconstruct

        mag = self._make_magnitude().astype(np.float64)
        result = pghi_reconstruct(mag, sr=48000)
        assert result.dtype == np.float32

    def test_06_reconstruct_no_nan(self):
        """Rekonstruktion ist NaN-frei."""
        from dsp.pghi import pghi_reconstruct

        mag = self._make_magnitude()
        result = pghi_reconstruct(mag, sr=48000)
        assert np.isfinite(result).all(), "NaN/Inf in pghi_reconstruct output"

    def test_07_reconstruct_bounds(self):
        """Ausgabe liegt in [-1, 1]."""
        from dsp.pghi import pghi_reconstruct

        mag = self._make_magnitude()
        result = pghi_reconstruct(mag, sr=48000)
        assert np.max(np.abs(result)) <= 1.0 + 1e-5

    def test_08_silence_passthrough(self):
        """Null-Magnitude → Stille im Ausgabe-Audio."""
        from dsp.pghi import pghi_reconstruct

        mag = np.zeros((1025, 100), dtype=np.float32)  # 2048-FFT
        result = pghi_reconstruct(mag, sr=48000)
        assert np.max(np.abs(result)) < 1e-4

    def test_09_dirac_impulse_stable(self):
        """Einzelner Peak im Spektrogramm führt nicht zu Absturz."""
        from dsp.pghi import pghi_reconstruct

        mag = np.zeros((1025, 50), dtype=np.float32)
        mag[10, 25] = 1.0  # einzelner Frequenz-Peak
        result = pghi_reconstruct(mag, sr=48000)
        assert np.isfinite(result).all()

    def test_10_white_noise_stable(self):
        """Zufälliges Magnitude-Spektrogramm wird stabil verarbeitet."""
        from dsp.pghi import pghi_reconstruct

        mag = np.abs(np.random.randn(1025, 200).astype(np.float32)) * 0.1
        result = pghi_reconstruct(mag, sr=48000)
        assert np.isfinite(result).all()
        assert np.max(np.abs(result)) <= 1.0 + 1e-5

    def test_11_reconstruct_from_stft(self):
        """pghi_reconstruct_from_stft() gibt NaN-freies float32-Ergebnis zurück."""
        from dsp.pghi import pghi_reconstruct_from_stft

        n_fft = 1024
        hop = n_fft // 4
        # Erzeuge komplexes STFT-Array [n_bins, n_frames]
        n_bins = n_fft // 2 + 1
        n_frames = 100
        stft = (np.random.randn(n_bins, n_frames) + 1j * np.random.randn(n_bins, n_frames)).astype(np.complex64) * 0.2
        result = pghi_reconstruct_from_stft(stft, sr=48000, win_size=n_fft, hop=hop)
        assert np.isfinite(result).all()
        assert result.dtype == np.float32

    def test_12_griffin_lim_fallback(self):
        """griffin_lim_reconstruct() ist verfügbar und stabil."""
        from dsp.pghi import griffin_lim_reconstruct

        mag = np.abs(np.random.randn(1025, 100).astype(np.float32)) * 0.2
        result = griffin_lim_reconstruct(mag, sr=48000)
        assert np.isfinite(result).all()
        assert result.ndim == 1

    def test_13_consistency(self):
        """Identische Eingabe → identische Ausgabe."""
        from dsp.pghi import pghi_reconstruct

        mag = self._make_magnitude(2048)
        r1 = pghi_reconstruct(mag.copy(), sr=48000)
        r2 = pghi_reconstruct(mag.copy(), sr=48000)
        np.testing.assert_array_equal(r1, r2)

    def test_14_sr_assertion(self):
        """SR != 48000 wirft AssertionError."""

        with pytest.raises((AssertionError, ValueError)):
            from dsp.pghi import PghiReconstructor

            PghiReconstructor(sr=44100)

    def test_15_numba_heapq_bit_identical(self):
        """Numba-Heap-Kern reproduziert den heapq-Pfad bit-identisch (§PERF-R15).

        Der Kern in `_pghi` (aktiv wenn numba importierbar ist) muss exakt
        dieselben rekonstruierten Phasen liefern wie `_pghi_heapq`. Dies
        pinnt die bit-Identität beider Pfade dauerhaft in CI.
        """
        import math

        import dsp.pghi as _pg
        from dsp.pghi import PghiReconstructor

        if _pg._numba is None:
            pytest.skip("numba nicht verfuegbar")

        rec = PghiReconstructor(sr=48000, win_size=2048, hop=256)

        def _ref_heapq(mag_f64: np.ndarray) -> np.ndarray:
            n_bins, n_frames = mag_f64.shape
            log_mag = np.log(mag_f64 + rec.tol)
            k_idx = np.arange(n_bins)
            dt = np.tile(
                2.0 * math.pi * rec.hop * k_idx[:, np.newaxis] / rec.win_size,
                (1, n_frames),
            )
            dlog = np.zeros_like(log_mag)
            dlog[1:-1, :] = (log_mag[2:, :] - log_mag[:-2, :]) / 2.0
            dlog[0, :] = log_mag[1, :] - log_mag[0, :]
            dlog[-1, :] = log_mag[-1, :] - log_mag[-2, :]
            do = -rec.gamma * dlog * rec.win_size / (2.0 * math.pi)
            phase = np.zeros((n_bins, n_frames), dtype=np.float64)
            rng = np.random.default_rng(seed=42)
            phase[:, 0] = rng.uniform(-math.pi, math.pi, n_bins)
            visited = np.zeros((n_bins, n_frames), dtype=bool)
            max_frame = int(np.argmax(np.max(mag_f64, axis=0)))
            max_bin = int(np.argmax(mag_f64[:, max_frame]))
            rec._pghi_heapq(mag_f64, phase, visited, dt, do, max_frame, max_bin)
            return np.angle(np.exp(1j * phase)).astype(np.float64)

        for bins, frames, seed in [
            (64, 64, 1),
            (512, 256, 3),
            (1025, 120, 5),
            (2, 50, 6),  # 2 Bins (Minimum für Zentraldifferenz in _pghi)
            (50, 1, 7),  # 1 Frame
            (8, 8, 8),  # kleine Matrix
        ]:
            mag = np.abs(np.random.default_rng(seed).standard_normal((bins, frames)) * 0.3).astype(np.float32)
            out_kernel = rec._pghi(mag.astype(np.float64), rec.win_size, rec.hop, None)
            out_heapq = _ref_heapq(mag.astype(np.float64))
            np.testing.assert_array_equal(
                out_kernel,
                out_heapq,
                err_msg=f"Kern/heapq-Drift bei bins={bins} frames={frames}",
            )

        # §PERF-R17: Tie-lastige und konstante Magnituden (Push-Dedup-Pfad)
        for bins, frames, seed in [(100, 100, 9), (100, 100, 10), (50, 300, 11), (30, 500, 12)]:
            mag = np.round(np.abs(np.random.default_rng(seed).standard_normal((bins, frames)) * 0.3), 2).astype(
                np.float32
            )
            out_kernel = rec._pghi(mag.astype(np.float64), rec.win_size, rec.hop, None)
            out_heapq = _ref_heapq(mag.astype(np.float64))
            np.testing.assert_array_equal(
                out_kernel,
                out_heapq,
                err_msg=f"Kern/heapq-Drift (tie) bei bins={bins} frames={frames}",
            )

        # Konstante Magnituden (alle Energien gleich => maximale Tie-Situation)
        mag = np.full((40, 40), 0.5, dtype=np.float32)
        out_kernel = rec._pghi(mag.astype(np.float64), rec.win_size, rec.hop, None)
        out_heapq = _ref_heapq(mag.astype(np.float64))
        np.testing.assert_array_equal(out_kernel, out_heapq, err_msg="Kern/heapq-Drift bei konstantem mag")


# ---------------------------------------------------------------------------
# PSOLA-Tests
# ---------------------------------------------------------------------------


class TestPsolaPitchShifter:
    """Tests für dsp/psola.py — PSOLA formanterhaltender Pitch-Shifter."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    def _make_sine(self, freq_hz: float = 220.0, duration_s: float = 1.0, sr: int = 48000) -> np.ndarray:
        t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
        return (np.sin(2 * np.pi * freq_hz * t) * 0.5).astype(np.float32)  # type: ignore[no-any-return]

    def test_01_import(self):
        """psola-Modul ist importierbar."""
        from dsp.psola import get_psola_shifter

        assert get_psola_shifter() is not None

    def test_02_singleton_identity(self):
        """get_psola_shifter() gibt identisches Objekt zurück."""
        from dsp.psola import get_psola_shifter

        assert get_psola_shifter() is get_psola_shifter()

    def test_03_passthrough_zero_semitones(self):
        """0 Halbtöne → identisches Audio (Passthrough)."""
        from dsp.psola import psola_shift

        audio = self._make_sine(220.0)
        result = psola_shift(audio, semitones=0.0)
        np.testing.assert_array_equal(result, audio)

    def test_04_output_shape(self):
        """Ausgabe hat gleiche Länge wie Eingabe."""
        from dsp.psola import psola_shift

        audio = self._make_sine(220.0, 0.5)
        result = psola_shift(audio, semitones=2.0, f0_hz=220.0)
        assert result.shape == audio.shape

    def test_05_output_dtype_float32(self):
        """Ausgabe ist float32."""
        from dsp.psola import psola_shift

        audio = self._make_sine(220.0)
        result = psola_shift(audio, semitones=1.0, f0_hz=220.0)
        assert result.dtype == np.float32

    def test_06_no_nan(self):
        """Keine NaN/Inf im Ausgabe-Audio."""
        from dsp.psola import psola_shift

        audio = self._make_sine(440.0)
        result = psola_shift(audio, semitones=-2.0, f0_hz=440.0)
        assert np.isfinite(result).all()

    def test_07_bounds(self):
        """Ausgabe liegt in [-1, 1]."""
        from dsp.psola import psola_shift

        audio = self._make_sine(220.0) * 0.9
        result = psola_shift(audio, semitones=4.0, f0_hz=220.0)
        assert np.max(np.abs(result)) <= 1.0 + 1e-5

    def test_08_silence_stable(self):
        """Stille führt nicht zu Absturz und bleibt leise."""
        from dsp.psola import psola_shift

        audio = np.zeros(48000, dtype=np.float32)
        result = psola_shift(audio, semitones=3.0)
        assert np.isfinite(result).all()
        assert np.max(np.abs(result)) < 1e-4

    def test_09_white_noise_stable(self):
        """Weißes Rauschen als Eingabe → kein Absturz."""
        from dsp.psola import psola_shift

        audio = np.random.randn(48000).astype(np.float32) * 0.1
        result = psola_shift(audio, semitones=1.0)
        assert np.isfinite(result).all()

    def test_10_positive_shift(self):
        """Positive Halbton-Verschiebung verarbeitet ohne Fehler."""
        from dsp.psola import psola_shift

        audio = self._make_sine(220.0)
        result = psola_shift(audio, semitones=7.0, f0_hz=220.0)
        assert result.shape == audio.shape
        assert np.isfinite(result).all()

    def test_11_negative_shift(self):
        """Negative Halbton-Verschiebung verarbeitet ohne Fehler."""
        from dsp.psola import psola_shift

        audio = self._make_sine(440.0)
        result = psola_shift(audio, semitones=-5.0, f0_hz=440.0)
        assert result.shape == audio.shape
        assert np.isfinite(result).all()

    def test_12_result_dataclass(self):
        """shift_pitch() gibt PsolaResult mit allen Feldern zurück."""
        from dsp.psola import PsolaResult, get_psola_shifter

        audio = self._make_sine(220.0, 0.5)
        result = get_psola_shifter().shift_pitch(audio, semitones=2.0, f0_hz=220.0)
        assert isinstance(result, PsolaResult)
        assert math.isfinite(result.pitch_shift_semitones)
        assert isinstance(result.n_epochs, int)
        assert isinstance(result.formant_preserved, bool)
        assert isinstance(result.method_used, str)

    def test_13_wow_flutter_correction_shape(self):
        """correct_wow_flutter() gibt Array gleicher Länge zurück."""
        from dsp.psola import psola_correct_wow_flutter

        audio = self._make_sine(220.0, 2.0)
        n_frames = 200
        f0_traj = np.full(n_frames, 218.0)  # leichter Drift
        result = psola_correct_wow_flutter(audio, f0_traj, target_f0=220.0)
        assert result.shape == audio.shape
        assert np.isfinite(result).all()

    def test_14_clamp_extreme_shift(self):
        """Extreme Halbton-Verschiebung (±24) wird geclamppt — kein Absturz."""
        from dsp.psola import psola_shift

        audio = self._make_sine(220.0, 0.5)
        result = psola_shift(audio, semitones=24.0, f0_hz=220.0)
        assert np.isfinite(result).all()
        assert result.shape == audio.shape

    def test_15_sr_assertion(self):
        """SR != 48000 wirft AssertionError."""
        from dsp.psola import PsolaPitchShifter

        with pytest.raises(AssertionError):
            PsolaPitchShifter(sr=44100)

    def test_16_short_audio_stable(self):
        """Sehr kurzes Audio (< 1000 Samples) → kein Absturz."""
        from dsp.psola import psola_shift

        audio = self._make_sine(220.0, 0.01)  # 480 Samples
        result = psola_shift(audio, semitones=2.0, f0_hz=220.0)
        assert np.isfinite(result).all()

    def test_17_f0_trajectory_used(self):
        """Mit f0_trajectory: kein Absturz, korrekte Shape."""
        from dsp.psola import psola_shift

        audio = self._make_sine(220.0, 1.0)
        n_frames = 100
        traj = np.linspace(215.0, 225.0, n_frames)
        result = psola_shift(audio, semitones=1.0, f0_hz=220.0, f0_trajectory=traj)
        assert result.shape == audio.shape
        assert np.isfinite(result).all()

    def test_18_thread_safe_singleton(self):
        """Parallele Zugriffe auf Singleton sind sicher."""
        from dsp.psola import get_psola_shifter

        results = []

        def fetch():
            results.append(get_psola_shifter())

        threads = [threading.Thread(target=fetch) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert all(r is results[0] for r in results)


# ---------------------------------------------------------------------------
# DTW-Groove-Tests
# ---------------------------------------------------------------------------


class TestDtwGrooveMeasurer:
    """Tests für dsp/dtw_groove.py — DTW Groove-Messung."""

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(42)

    def _make_signal(self, duration_s: float = 3.0, sr: int = 48000) -> np.ndarray:
        """Synthetisches Signal mit erkennbaren Onsets."""
        n = int(sr * duration_s)
        audio = np.random.randn(n).astype(np.float32) * 0.05
        # Onsets bei 0.5, 1.0, 1.5, 2.0, 2.5 s
        for t in [0.5, 1.0, 1.5, 2.0, 2.5]:
            idx = int(t * sr)
            if idx < n:
                decay = np.exp(-np.arange(min(2400, n - idx)) * 0.003)
                audio[idx : idx + len(decay)] += 0.8 * decay.astype(np.float32)
        return np.clip(audio, -1.0, 1.0)

    def test_01_import(self):
        """dtw_groove-Modul ist importierbar."""
        from dsp.dtw_groove import get_groove_measurer

        assert get_groove_measurer() is not None

    def test_02_singleton_identity(self):
        """get_groove_measurer() gibt identisches Objekt zurück."""
        from dsp.dtw_groove import get_groove_measurer

        assert get_groove_measurer() is get_groove_measurer()

    def test_03_measure_same_signal(self):
        """Gleiches Signal → groove_score nahe 1.0, RMS nahe 0 ms."""
        from dsp.dtw_groove import measure_groove

        audio = self._make_signal(3.0)
        result = measure_groove(audio, audio.copy())
        assert result.groove_score >= 0.95
        assert result.dtw_rms_ms <= 1.0

    def test_04_result_fields_finite(self):
        """Alle numerischen Felder sind finite."""
        from dsp.dtw_groove import measure_groove

        orig = self._make_signal()
        rest = self._make_signal() + np.random.randn(len(self._make_signal())).astype(np.float32) * 0.01
        result = measure_groove(orig, rest)
        assert math.isfinite(result.dtw_distance_ms)
        assert math.isfinite(result.dtw_rms_ms)
        assert math.isfinite(result.groove_score)

    def test_05_score_bounds(self):
        """groove_score liegt in [0, 1]."""
        from dsp.dtw_groove import measure_groove

        orig = self._make_signal()
        rest = np.random.randn(len(orig)).astype(np.float32) * 0.3
        result = measure_groove(orig, rest)
        assert 0.0 <= result.groove_score <= 1.0

    def test_06_silence_no_crash(self):
        """Zwei Stille-Signale → kein Absturz, groove_score = 1.0."""
        from dsp.dtw_groove import measure_groove

        silence = np.zeros(48000, dtype=np.float32)
        result = measure_groove(silence, silence.copy())
        assert result.groove_score == 1.0
        assert math.isfinite(result.dtw_rms_ms)

    def test_07_passes_threshold_same_signal(self):
        """Gleiches Signal besteht den 8-ms-Schwellwert."""
        from dsp.dtw_groove import measure_groove

        audio = self._make_signal()
        result = measure_groove(audio, audio.copy())
        assert result.passes_threshold

    def test_08_onset_counts_nonneg(self):
        """Onset-Counts sind ≥ 0."""
        from dsp.dtw_groove import measure_groove

        orig = self._make_signal()
        rest = self._make_signal()
        result = measure_groove(orig, rest)
        assert result.n_onsets_original >= 0
        assert result.n_onsets_restored >= 0

    def test_09_deviations_array_finite(self):
        """onset_deviations_ms ist finite (wenn nicht leer)."""
        from dsp.dtw_groove import measure_groove

        orig = self._make_signal()
        rest = self._make_signal()
        result = measure_groove(orig, rest)
        if len(result.onset_deviations_ms) > 0:
            assert np.isfinite(result.onset_deviations_ms).all()

    def test_10_sr_assertion(self):
        """SR != 48000 wirft AssertionError."""
        from dsp.dtw_groove import DtwGrooveMeasurer

        with pytest.raises(AssertionError):
            DtwGrooveMeasurer(sr=44100)

    def test_11_onset_detection_shape(self):
        """detect_onsets() gibt OnsetDetectionResult mit konsistenten Feldern zurück."""
        from dsp.dtw_groove import detect_onsets

        audio = self._make_signal()
        result = detect_onsets(audio)
        assert len(result.onset_samples) == result.n_onsets
        assert len(result.onset_times_ms) == result.n_onsets
        assert len(result.onset_strengths) == result.n_onsets

    def test_12_onset_detection_silence(self):
        """Stille → 0 Onsets, kein Fehler."""
        from dsp.dtw_groove import detect_onsets

        audio = np.zeros(48000, dtype=np.float32)
        result = detect_onsets(audio)
        assert result.n_onsets == 0

    def test_13_dtw_align_empty(self):
        """dtw_align mit leeren Sequenzen → kein Absturz."""
        from dsp.dtw_groove import dtw_align

        pair_arr, dist = dtw_align(np.array([]), np.array([]))
        assert len(pair_arr) == 0
        assert math.isfinite(dist)

    def test_14_dtw_align_simple(self):
        """dtw_align mit einfacher Sequenz: Distanz ist finite."""
        from dsp.dtw_groove import dtw_align

        a = np.array([0.0, 100.0, 200.0])
        b = np.array([5.0, 105.0, 205.0])  # 5 ms verschoben
        pair_arr, dist = dtw_align(a, b)
        assert math.isfinite(dist)
        assert len(pair_arr) > 0

    def test_15_quick_measure_bounds(self):
        """measure_groove_quick() gibt Wert in [0, 1]."""
        from dsp.dtw_groove import measure_groove_quick

        audio = self._make_signal(5.0)
        score = measure_groove_quick(audio)
        assert 0.0 <= score <= 1.0, f"Score außerhalb [0,1]: {score}"

    def test_16_quick_measure_no_nan(self):
        """measure_groove_quick() gibt keinen NaN zurück."""
        from dsp.dtw_groove import measure_groove_quick

        audio = self._make_signal(5.0)
        score = measure_groove_quick(audio)
        assert math.isfinite(score)

    def test_17_onset_times_ascending(self):
        """Onset-Zeiten sind aufsteigend sortiert."""
        from dsp.dtw_groove import detect_onsets

        audio = self._make_signal()
        result = detect_onsets(audio)
        if result.n_onsets > 1:
            assert np.all(np.diff(result.onset_times_ms) >= 0)

    def test_18_method_field(self):
        """method_used ist ein nicht-leerer String."""
        from dsp.dtw_groove import measure_groove

        orig = self._make_signal()
        rest = self._make_signal()
        result = measure_groove(orig, rest)
        assert isinstance(result.method_used, str)
        assert len(result.method_used) > 0

    def test_19_thread_safe_singleton(self):
        """Parallele Zugriffe auf Singleton sind sicher."""
        from dsp.dtw_groove import get_groove_measurer

        results = []

        def fetch():
            results.append(get_groove_measurer())

        threads = [threading.Thread(target=fetch) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert all(r is results[0] for r in results)

    def test_20_white_noise_no_crash(self):
        """Weißes Rauschen → kein Absturz."""
        from dsp.dtw_groove import measure_groove

        orig = np.random.randn(48000).astype(np.float32) * 0.1
        rest = np.random.randn(48000).astype(np.float32) * 0.1
        result = measure_groove(orig, rest)
        assert math.isfinite(result.groove_score)
        assert 0.0 <= result.groove_score <= 1.0


class TestDtwGrooveOnsetLossGuard:
    """Regressionstests für den Onset-Verlust-Guard (§v10.x, Befund 2026-08-22).

    Der alte symmetrische "no_onsets"-Pfad lieferte groove_score=1.0, wenn EINE
    Seite keine Onsets hatte — ein False-Pass, der kaputtes Audio (0 Onsets)
    als "perfekter Groove" durchließ (Log: onsets_orig=185, onsets_rest=0 →
    Wert=1.000 trotz katastrophalem Onset-Verlust).
    """

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(7)

    def _signal(self, duration_s: float = 3.0, sr: int = 48000) -> np.ndarray:
        n = int(sr * duration_s)
        audio = np.random.randn(n).astype(np.float32) * 0.05
        for t in [0.5, 1.0, 1.5, 2.0, 2.5]:
            idx = int(t * sr)
            if idx < n:
                decay = np.exp(-np.arange(min(2400, n - idx)) * 0.003)
                audio[idx : idx + len(decay)] += 0.8 * decay.astype(np.float32)
        return np.clip(audio, -1.0, 1.0)

    def test_01_restored_silent_reports_onset_loss(self):
        """Original hat Onsets, Restauriert ist still → 0.0 statt False-Pass."""
        from dsp.dtw_groove import measure_groove

        orig = self._signal()
        rest = np.zeros(48000 * 3, dtype=np.float32)
        result = measure_groove(orig, rest)
        assert result.n_onsets_original >= 4
        assert result.n_onsets_restored == 0
        assert result.groove_score == 0.0
        assert result.passes_threshold is False
        assert result.method_used == "onset_loss_restored"

    def test_02_original_silent_reports_onset_loss(self):
        """Restauriert hat Onsets, Original ist still → symmetrisch gemeldet."""
        from dsp.dtw_groove import measure_groove

        orig = np.zeros(48000 * 3, dtype=np.float32)
        rest = self._signal()
        result = measure_groove(orig, rest)
        assert result.n_onsets_restored >= 4
        assert result.groove_score == 0.0
        assert result.method_used == "onset_loss_original"

    def test_03_both_silent_stays_neutral(self):
        """Beide still → weiterhin neutral (1.0), kein False-Pass-Umkehrfehler."""
        from dsp.dtw_groove import measure_groove

        silence = np.zeros(48000, dtype=np.float32)
        result = measure_groove(silence, silence.copy())
        assert result.groove_score == 1.0
        assert result.method_used == "no_onsets"

    def test_04_nan_input_deterministic_no_false_pass(self):
        """NaN im Signal → deterministisch, kein Absturz, kein False-Pass."""
        from dsp.dtw_groove import measure_groove

        orig = self._signal()
        rest = self._signal().copy()
        rest[::997] = np.nan  # verstreute NaNs
        result = measure_groove(orig, rest)
        assert math.isfinite(result.groove_score)
        assert 0.0 <= result.groove_score <= 1.0
        # Gleicher Input ⇒ gleiches Ergebnis (Determinismus, §G5 (GEBOTE.md)).
        result2 = measure_groove(orig, rest)
        assert result.groove_score == result2.groove_score

    def test_05_channels_first_detect_parity(self):
        """detect_onsets((2, N)) == detect_onsets(mono) — Zeitachse korrekt."""
        from dsp.dtw_groove import detect_onsets

        mono = self._signal()
        stereo_cf = np.stack([mono, mono], axis=0)  # (2, N) channels-first
        stereo_sf = np.stack([mono, mono], axis=1)  # (N, 2) samples-first
        r_cf = detect_onsets(stereo_cf)
        r_sf = detect_onsets(stereo_sf)
        r_mono = detect_onsets(mono)
        assert r_cf.n_onsets == r_mono.n_onsets
        assert r_sf.n_onsets == r_mono.n_onsets
        assert r_cf.n_onsets >= 4, "Layout-Fehler würde 0 Onsets ergeben"
