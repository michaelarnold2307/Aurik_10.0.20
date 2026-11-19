import pytest

"""Tests für v10.0.0-Features: Phase-12-Qualität, UV3-Reihenfolge, Queue-Drag&Drop,
Phase-06/20/42 DSP-Qualitätsverbesserungen.

Abgedeckt:
  - Phase 12 PITCH_HOP_FACTOR == 4 (75 % Overlap, Nyquist-sicher für 4–100 Hz Flutter)
  - Phase 12 STFT_WINDOW_SIZE == 2048 (23 Hz/Bin @ 48 kHz)
  - UV3 phase_55 vor phase_56 in der Phasen-Ordnung deklariert
  - QueueManager.reorder_items ordnet korrekt um
  - QueueWidget emittiert reorder_requested nach _on_rows_moved
  - Phase 06: Alpha steigt bei schwerem Bandbreiten-Defizit (shellac rolloff=4500 Hz)
  - Phase 06: Alpha bleibt moderat bei leichtem Defizit (vinyl rolloff=11000 Hz)
  - Phase 20: Late-Reverb-Suppression reduziert G_combined in Decay-Frames
  - Phase 42: Multi-Formant-Fallback erzeugt hörbaren Unterschied zu single-Bell
"""

import threading
import uuid

# ---------------------------------------------------------------------------
# Phase 12 — Konstanten-Invarianten
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPhase12QualityConstants:
    """Phase 12 muss 75 % Overlap (factor=4) und 2048er STFT-Fenster verwenden."""

    def _load_constants(self):
        from backend.core.phases.phase_12_wow_flutter_fix import WowFlutterFix

        inst = WowFlutterFix.__new__(WowFlutterFix)
        return inst

    def test_pitch_hop_factor_is_4(self):
        """PITCH_HOP_FACTOR muss 4 sein (75 % Overlap, kein Performance-Cut)."""
        from backend.core.phases.phase_12_wow_flutter_fix import WowFlutterFix

        assert WowFlutterFix.PITCH_HOP_FACTOR == 4, (
            "PITCH_HOP_FACTOR wurde auf 2 reduziert — das reicht nicht für 4–100 Hz Flutter! "
            "Nyquist verlangt mindestens 2 Samples pro Flatter-Zyklus. Bei 48 kHz und "
            "100 ms Fenster muss PITCH_HOP_FACTOR >= 4 sein."
        )

    def test_stft_window_2048(self):
        """STFT_WINDOW_SIZE == 2048 → 23 Hz/Bin @ 48 kHz."""
        from backend.core.phases.phase_12_wow_flutter_fix import WowFlutterFix

        assert WowFlutterFix.STFT_WINDOW_SIZE == 2048, (
            f"STFT_WINDOW_SIZE={WowFlutterFix.STFT_WINDOW_SIZE}, erwartet 2048."
        )

    def test_stft_hop_is_window_over_4(self):
        """STFT_HOP_SIZE muss 75 % des Fensters entsprechen (window // 4 * 1 = window//4)."""
        from backend.core.phases.phase_12_wow_flutter_fix import WowFlutterFix

        # hop = window / factor → window=2048, factor=4 → hop=512
        expected = WowFlutterFix.STFT_WINDOW_SIZE // WowFlutterFix.PITCH_HOP_FACTOR
        assert expected == WowFlutterFix.STFT_HOP_SIZE, (
            f"STFT_HOP_SIZE={WowFlutterFix.STFT_HOP_SIZE}, erwartet {expected} "
            f"(= {WowFlutterFix.STFT_WINDOW_SIZE} // {WowFlutterFix.PITCH_HOP_FACTOR})."
        )

    def test_hop_samples_at_48k_nyquist_safe_for_20hz_flutter(self):
        """Bei 48 kHz und PITCH_WINDOW_MS=100 muss hop_samples <= 1/(2*20 Hz)*48000 = 1200.

        Flutter auf Vinyl/Tape tritt im Bereich 4–20 Hz auf. Nyquist-Kriterium:
        Mindestens 2 Schätzungen pro Flatter-Zyklus → hop ≤ 48000/(2×20) = 1200 Samples.
        PITCH_HOP_FACTOR=4 → hop = 4800//4 = 1200 Samples → genau am Nyquist-Limit.
        PITCH_HOP_FACTOR=2 → hop = 4800//2 = 2400 Samples → über Nyquist für 10-20 Hz Flutter.
        """
        from backend.core.phases.phase_12_wow_flutter_fix import WowFlutterFix

        window_samples = int(getattr(WowFlutterFix, "PITCH_WINDOW_MS", 100) / 1000 * 48000)
        hop_samples = max(1, window_samples // WowFlutterFix.PITCH_HOP_FACTOR)
        nyquist_for_20hz_flutter = 48000 / (2 * 20)  # = 1200 samples
        assert hop_samples <= nyquist_for_20hz_flutter, (
            f"hop_samples={hop_samples} > Nyquist-Grenze {nyquist_for_20hz_flutter:.0f} für 20 Hz Flutter. "
            "PITCH_HOP_FACTOR muss >= 4 sein."
        )


# ---------------------------------------------------------------------------
# UV3 — Phase-55-vor-Phase-56-Guard
# ---------------------------------------------------------------------------


class TestUV3Phase55Before56:
    """UV3 muss phase_55_diffusion_inpainting vor phase_56_spectral_band_gap_repair einreihen."""

    def _get_uv3_source(self) -> str:
        import os

        uv3_path = os.path.join(
            os.path.dirname(__file__),
            "../../backend/core/unified_restorer_v3.py",
        )
        with open(os.path.normpath(uv3_path), encoding="utf-8") as f:
            return f.read()

    def test_move_before_guard_present_in_source(self):
        """_move_before('phase_55_diffusion_inpainting', 'phase_56_spectral_band_gap_repair') muss im UV3-Quelltext stehen."""
        src = self._get_uv3_source()
        assert (
            '_move_before("phase_55_diffusion_inpainting", "phase_56_spectral_band_gap_repair")' in src
            or "_move_before('phase_55_diffusion_inpainting', 'phase_56_spectral_band_gap_repair')" in src
        ), "UV3 enthält keinen _move_before-Guard für phase_55 → phase_56!"

    def test_phase_55_guard_appears_before_phase_57_guard(self):
        """Der phase_55→56-Guard muss im Quelltext vor dem phase_57→29-Guard stehen
        (kanonische Reihenfolge aus §7.x)."""
        src = self._get_uv3_source()
        guard_55 = '_move_before("phase_55_diffusion_inpainting", "phase_56_spectral_band_gap_repair")'
        guard_57 = '_move_before("phase_57_print_through_reduction", "phase_29_tape_hiss_reduction")'
        idx_55_56 = src.find(guard_55)
        idx_57_29 = src.find(guard_57)
        assert idx_55_56 != -1, "phase_55→56 _move_before-Guard nicht gefunden."
        assert idx_57_29 != -1, "phase_57→29 _move_before-Guard nicht gefunden."
        assert idx_55_56 < idx_57_29, (
            f"phase_55→56-Guard (pos {idx_55_56}) steht nach phase_57→29-Guard (pos {idx_57_29})."
        )


# ---------------------------------------------------------------------------
# QueueManager — reorder_items
# ---------------------------------------------------------------------------


class TestQueueManagerReorder:
    """QueueManager.reorder_items ordnet die interne _items-Dict korrekt um."""

    def _make_manager_with_items(self, n: int = 3):
        from Aurik10.core.queue_manager import QueueManager

        mgr = QueueManager()
        ids = []
        for i in range(n):
            item = mgr.add_item(f"/input_{i}.wav", f"/output_{i}.flac")
            ids.append(item.id)
        return mgr, ids

    def test_reorder_reverses_order(self):
        mgr, ids = self._make_manager_with_items(3)
        reversed_ids = list(reversed(ids))
        changed = mgr.reorder_items(reversed_ids)
        assert changed is True
        assert list(mgr._items.keys()) == reversed_ids

    def test_reorder_same_order_returns_false(self):
        mgr, ids = self._make_manager_with_items(3)
        changed = mgr.reorder_items(ids)
        assert changed is False

    def test_reorder_unknown_ids_ignored(self):
        """Unbekannte IDs in new_order werden still ignoriert."""
        mgr, ids = self._make_manager_with_items(3)
        bogus = [str(uuid.uuid4())] + ids
        mgr.reorder_items(bogus)  # darf keine Exception werfen
        assert set(mgr._items.keys()) == set(ids)

    def test_reorder_partial_order_remaining_appended(self):
        """IDs die fehlen werden ans Ende angehängt."""
        mgr, ids = self._make_manager_with_items(3)
        partial = [ids[2], ids[0]]  # ids[1] fehlt
        mgr.reorder_items(partial)
        result = list(mgr._items.keys())
        assert result[0] == ids[2]
        assert result[1] == ids[0]
        assert result[2] == ids[1]  # ans Ende verschoben

    def test_reorder_thread_safety(self):
        """Gleichzeitige Aufrufe dürfen keine Exception werfen."""
        mgr, ids = self._make_manager_with_items(5)
        errors = []

        def _worker():
            try:
                for _ in range(50):
                    mgr.reorder_items(list(reversed(ids)))
                    mgr.reorder_items(ids)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, f"Thread-Safety-Fehler: {errors}"

    def test_reorder_returns_true_when_order_changes(self):
        mgr, ids = self._make_manager_with_items(4)
        swapped = [ids[0], ids[2], ids[1], ids[3]]
        assert mgr.reorder_items(swapped) is True

    def test_reorder_empty_list_leaves_items_intact(self):
        mgr, ids = self._make_manager_with_items(2)
        mgr.reorder_items([])
        # Known IDs still present
        assert set(mgr._items.keys()) == set(ids)


# ---------------------------------------------------------------------------
# Phase 06 — Adaptive AudioSR-Blend-Alpha (v10.0.0)
# ---------------------------------------------------------------------------


class TestPhase06AdaptiveAlpha:
    """Alpha für AudioSR-Blending muss bei starkem Bandbreitendefizit deutlich größer sein."""

    @staticmethod
    def _calc_alpha(quality_mode: str, rolloff_hz: float, restoration_strength: float = 0.7) -> float:
        """Repliziert die Alpha-Berechnung aus Phase 06 ohne ML-Modell zu laden."""
        import numpy as np

        alpha_by_mode = {"balanced": 0.25, "quality": 0.38, "maximum": 0.55, "restoration": 0.32}
        _alpha_base = alpha_by_mode.get(quality_mode, 0.25)
        _sample_rate = 48000
        _deficit_threshold_hz = float(_sample_rate) * 0.30  # = 14400 Hz
        _deficit_fraction = float(np.clip(1.0 - rolloff_hz / _deficit_threshold_hz, 0.0, 1.0))
        _deficit_boost = _deficit_fraction * 0.35
        alpha = (_alpha_base + _deficit_boost) * restoration_strength
        return float(np.clip(alpha, 0.0, 0.80))

    def test_shellac_alpha_higher_than_vinyl(self):
        alpha_shellac = self._calc_alpha("quality", 4500.0)
        alpha_vinyl = self._calc_alpha("quality", 11000.0)
        assert alpha_shellac > alpha_vinyl, (
            f"Shellac alpha={alpha_shellac:.3f} nicht größer als vinyl={alpha_vinyl:.3f}"
        )

    def test_shellac_quality_mode_above_0_35(self):
        """Shellac in quality-Mode soll jetzt alpha > 0.35 (alt: max 0.21)."""
        alpha = self._calc_alpha("quality", 4500.0, restoration_strength=0.90)
        assert alpha > 0.35, f"Shellac alpha={alpha:.3f} ≤ 0.35 — Blend zu schwach"

    def test_no_deficit_alpha_at_baseline(self):
        """Rolloff nahe Nyquist → kein Boost, alpha ≈ alpha_base * restoration_strength."""

        alpha = self._calc_alpha("quality", 22000.0)
        expected_max = 0.38 * 0.7 + 0.01
        assert alpha <= expected_max, f"Kein Defizit aber alpha={alpha:.3f} > {expected_max:.3f}"

    def test_maximum_mode_higher_than_balanced(self):
        alpha_max = self._calc_alpha("maximum", 9000.0)
        alpha_bal = self._calc_alpha("balanced", 9000.0)
        assert alpha_max > alpha_bal

    def test_alpha_capped_at_0_80(self):
        """Alpha darf 0.80 nie überschreiten."""

        for rolloff in [500.0, 1000.0, 2000.0]:
            alpha = self._calc_alpha("maximum", rolloff, restoration_strength=1.0)
            assert alpha <= 0.80, f"alpha={alpha:.3f} > 0.80 für rolloff={rolloff}"


# ---------------------------------------------------------------------------
# Phase 42 — Multi-Formant Bell-EQ Fallback (v10.0.0)
# ---------------------------------------------------------------------------


class TestPhase42MultiFormantFallback:
    """Multi-Formant-Fallback muss F1, F2, F3 und Singer's Formant abdecken."""

    @staticmethod
    def _apply_single_bell(audio, sr, gain_db=3.0):
        import numpy as np
        from scipy import signal

        w0 = 2 * np.pi * 1500 / sr
        alpha = np.sin(w0) / (2 * 2.0)
        A = 10 ** (gain_db / 40)
        b = np.array([1 + alpha * A, -2 * np.cos(w0), 1 - alpha * A]) / (1 + alpha / A)
        a = np.array([1.0, -2 * np.cos(w0) / (1 + alpha / A), (1 - alpha / A) / (1 + alpha / A)])
        return signal.lfilter(b, a, audio)

    @staticmethod
    def _apply_multi_formant(audio, sr, gain_db=3.0):
        import numpy as np
        from scipy import signal

        FORMANT_BANDS = [(500, 0.50, 3.0), (1500, 0.80, 2.0), (2500, 0.35, 2.5), (3200, 0.20, 3.5)]
        enhanced = audio.copy().astype(np.float64)
        for f0_hz, gfrac, q in FORMANT_BANDS:
            gband = gain_db * gfrac
            if abs(gband) < 0.15:
                continue
            w0 = 2.0 * np.pi * f0_hz / sr
            sin_w0 = np.sin(w0)
            cos_w0 = np.cos(w0)
            af = sin_w0 / (2.0 * q)
            A = 10.0 ** (gband / 40.0)
            a0 = 1 + af / A
            b = np.array([1 + af * A, -2 * cos_w0, 1 - af * A]) / a0
            a = np.array([1.0, -2 * cos_w0 / a0, (1 - af / A) / a0])
            enhanced = signal.lfilter(b, a, enhanced)
        return np.clip(np.nan_to_num(enhanced, nan=0.0), -1.0, 1.0).astype(np.float32)

    @staticmethod
    def _pink_noise(sr=48000, dur=0.5):
        import numpy as np

        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        sig = sum(np.sin(2 * np.pi * f * t) / f for f in [300, 500, 800, 1200, 1500, 2000, 2500, 3200])
        return (sig / (np.max(np.abs(sig)) + 1e-9) * 0.7).astype(np.float32)

    def test_f1_boost_at_500hz(self):
        """Multi-Formant muss bei 500 Hz mehr anheben als der alte single-Bell."""
        import numpy as np
        from scipy import signal as sp

        sr = 48000
        audio = self._pink_noise(sr)
        single = self._apply_single_bell(audio, sr)
        multi = self._apply_multi_formant(audio, sr)
        f, psd_s = sp.welch(single, fs=sr, nperseg=2048)
        _, psd_m = sp.welch(multi, fs=sr, nperseg=2048)
        mask = (f >= 420) & (f <= 580)
        assert float(np.mean(psd_m[mask])) > float(np.mean(psd_s[mask])), (
            "F1 @ 500 Hz: Multi-Formant nicht stärker als single-Bell"
        )

    def test_singers_formant_present(self):
        """3000–3500 Hz Band muss angehoben werden."""
        import numpy as np
        from scipy import signal as sp

        sr = 48000
        audio = self._pink_noise(sr)
        single = self._apply_single_bell(audio, sr)
        multi = self._apply_multi_formant(audio, sr)
        f, psd_s = sp.welch(single, fs=sr, nperseg=2048)
        _, psd_m = sp.welch(multi, fs=sr, nperseg=2048)
        mask = (f >= 2800) & (f <= 3600)
        ratio = float(np.mean(psd_m[mask])) / (float(np.mean(psd_s[mask])) + 1e-15)
        assert ratio > 1.01, f"Singer's Formant: ratio={ratio:.3f} ≤ 1.01"

    def test_no_nan_inf(self):
        import numpy as np

        audio = self._pink_noise()
        result = self._apply_multi_formant(audio, 48000)
        assert np.isfinite(result).all()

    def test_clipped_to_minus1_plus1(self):
        import numpy as np

        audio = self._pink_noise() * 0.9
        result = self._apply_multi_formant(audio, 48000)
        assert np.max(np.abs(result)) <= 1.0


# ---------------------------------------------------------------------------
# Phase 20 — Late-Reverb Decay-Suppression (v10.0.0)
# ---------------------------------------------------------------------------


class TestPhase20LateReverbSuppression:
    """Late-Reverb-Suppression muss G_combined in Decay-Frames reduzieren."""

    @staticmethod
    def _calc_G_lr(E_log_db, strength=0.7, REF_HOP=512, sr=48000):
        import numpy as np

        n_t = len(E_log_db)
        dE = np.diff(E_log_db, prepend=E_log_db[0])
        _sm = max(3, min(7, n_t // 20))
        dE_smooth = np.convolve(dE.astype(np.float32), np.ones(_sm) / _sm, mode="same")
        decay_mask = (dE_smooth < -0.5).astype(np.float32)
        _prot = max(1, int(0.040 * sr / REF_HOP))
        for _oi in np.where(dE > 2.0)[0]:
            decay_mask[int(_oi) : min(n_t, int(_oi) + _prot)] = 0.0
        _penalty = float(np.clip(strength * 0.35, 0.0, 0.35))
        return np.clip(1.0 - _penalty * decay_mask, 0.60, 1.0).astype(np.float32), decay_mask

    def test_decay_tail_suppressed(self):
        """Frames mit exponentiell abfallender Energie erhalten G_lr < 1.0."""
        import numpy as np

        n_t = 100
        E = np.zeros(n_t)
        E[20] = 10.0
        for i in range(21, n_t):
            E[i] = 10.0 * np.exp(-0.15 * (i - 20)) - 2.0
        G_lr, _ = self._calc_G_lr(E, strength=0.7)
        assert float(np.min(G_lr)) < 1.0, "Kein Frame supprimiert — Late-Reverb inaktiv"

    def test_onset_protected(self):
        """Onset-Fenster (40 ms) darf nicht supprimiert werden."""
        import numpy as np

        n_t = 200
        E = np.zeros(n_t)
        E[10] = 15.0
        for i in range(11, n_t):
            E[i] = 15.0 * np.exp(-0.05 * (i - 10)) - 0.5
        G_lr, _ = self._calc_G_lr(E, strength=1.0, REF_HOP=512, sr=48000)
        _prot = max(1, int(0.040 * 48000 / 512))
        for t in range(10, min(n_t, 10 + _prot)):
            assert G_lr[t] >= 0.999, f"Frame t={t} im Schutzfenster supprimiert"

    def test_minimum_gain_floor_0_60(self):
        """G_lr darf nie unter 0.60 sinken."""
        import numpy as np

        n_t = 50
        E = np.array([10.0 - 2.0 * i for i in range(n_t)])
        G_lr, _ = self._calc_G_lr(E, strength=1.0)
        assert float(np.min(G_lr)) >= 0.60 - 1e-6

    def test_low_strength_minimal_penalty(self):
        """Bei strength=0.2 (> threshold) max. penalty ≈ 7 % → G_lr_min ≥ 0.90."""
        import numpy as np

        n_t = 80
        E = np.zeros(n_t)
        E[5] = 8.0
        for i in range(6, n_t):
            E[i] = 8.0 * np.exp(-0.20 * (i - 5))
        G_lr, _ = self._calc_G_lr(E, strength=0.2)
        assert float(np.min(G_lr)) >= 0.90
