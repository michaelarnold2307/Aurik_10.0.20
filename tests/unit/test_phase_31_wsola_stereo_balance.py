"""Unit-Tests für §2.51 Linked-Stereo WSOLA in phase_31.

Prüft dass die OLA-Window-Sum-Normalisierung die L/R-Amplitudenbeziehung
erhält (keine per-Kanal-Normalisierung die bis zu 6 dB Mismatch erzeugt).
"""

import numpy as np
import pytest


@pytest.fixture
def phase31_instance():
    from backend.core.phases.phase_31_speed_pitch_correction import SpeedPitchCorrectionPhase

    phase = SpeedPitchCorrectionPhase()
    phase.sample_rate = 48000
    return phase


def _make_stereo_sine(sr=48000, duration=0.5, freq_L=440.0, freq_R=440.0, amp_L=0.5, amp_R=0.2):
    """Stereo-Sinus mit unterschiedlichen Amplituden."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    L = (amp_L * np.sin(2 * np.pi * freq_L * t)).astype(np.float32)
    R = (amp_R * np.sin(2 * np.pi * freq_R * t)).astype(np.float32)
    return np.column_stack([L, R])  # (samples, 2)


@pytest.mark.unit
class TestWSOLAStereoBalance:
    """WSOLA darf die L/R-Amplitudenbeziehung nicht zerstören."""

    def test_wsola_mono_no_peak_normalization(self, phase31_instance):
        """_wsola_mono gibt OLA-normalisiertes Signal zurück, NICHT peak-normalisiert.

        Wenn L und R mit unterschiedlichen Amplituden verarbeitet werden,
        muss die Amplitude-Ratio im Ergebnis erhalten bleiben.
        """
        p = phase31_instance
        sr = p.sample_rate
        window_size = int(0.02 * sr)
        hop_a = window_size // 2
        hop_s = hop_a  # ratio = 1.0, kein Zeitstretch

        # Signale mit klar unterschiedlichen Amplituden
        t = np.linspace(0, 1.0, sr, endpoint=False).astype(np.float32)
        sig_L = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        sig_R = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)  # 5× leiser

        out_L = p._wsola_mono(sig_L, window_size, hop_a, hop_s)
        out_R = p._wsola_mono(sig_R, window_size, hop_a, hop_s)

        # Wenn per-Kanal-Peak-Normalisierung vorhanden wäre, hätten beide ~1.0 Peak
        # Stattdessen muss RMS-Verhältnis erhalten bleiben (ca. 5:1)
        rms_L = float(np.sqrt(np.mean(out_L**2) + 1e-12))
        rms_R = float(np.sqrt(np.mean(out_R**2) + 1e-12))
        ratio = rms_L / (rms_R + 1e-12)

        # Mit alter Peak-Normalisierung wäre ratio ≈ 1.0 (beide auf Peak ~1.0 normiert)
        # Mit COLA-Normalisierung muss ratio ≈ 5.0 sein (Amplitude-Verhältnis erhalten)
        assert ratio > 3.0, (
            f"WSOLA per-Kanal-Normalisierung zerstört L/R-Balance: rms_L={rms_L:.3f}, "
            f"rms_R={rms_R:.3f}, ratio={ratio:.2f} (erwartet > 3.0)"
        )

    def test_correct_wsola_stereo_balance_preserved(self, phase31_instance):
        """_correct_wsola erhält die L/R-Amplitudenbeziehung bei Stereo-Input.

        L ist 5× lauter als R — nach WSOLA muss diese Relation erhalten bleiben.
        """
        p = phase31_instance
        audio = _make_stereo_sine(sr=p.sample_rate, amp_L=0.5, amp_R=0.1, duration=0.3)

        rms_in_L = float(np.sqrt(np.mean(audio[:, 0] ** 2) + 1e-12))
        rms_in_R = float(np.sqrt(np.mean(audio[:, 1] ** 2) + 1e-12))
        ratio_in = rms_in_L / (rms_in_R + 1e-12)

        out = p._correct_wsola(audio, ratio=1.0, params={})

        assert out.ndim == 2, "Output muss Stereo sein"
        rms_out_L = float(np.sqrt(np.mean(out[:, 0] ** 2) + 1e-12))
        rms_out_R = float(np.sqrt(np.mean(out[:, 1] ** 2) + 1e-12))
        ratio_out = rms_out_L / (rms_out_R + 1e-12)

        # L/R-Ratio muss erhalten bleiben (max. ±1 dB Toleranz)
        ratio_diff_db = abs(20 * np.log10(ratio_out / (ratio_in + 1e-12) + 1e-12))
        assert ratio_diff_db < 1.0, (
            f"WSOLA zerstört Stereo-Balance: ratio_in={ratio_in:.2f}, "
            f"ratio_out={ratio_out:.2f}, Abweichung={ratio_diff_db:.1f} dB (max. 1 dB)"
        )

    def test_correct_wsola_stereo_length_equal(self, phase31_instance):
        """L und R haben nach WSOLA die gleiche Länge (Linked-Invariante §2.51)."""
        p = phase31_instance
        audio = _make_stereo_sine(sr=p.sample_rate, duration=0.5)
        out = p._correct_wsola(audio, ratio=0.9, params={})  # Zeitstretch

        assert out.shape[1] == 2, "Output muss 2 Kanäle haben"
        assert out.shape[0] > 0, "Output darf nicht leer sein"

    def test_correct_wsola_no_channel_phase_shift(self, phase31_instance):
        """L und R dürfen keinen relativen Zeitversatz aufweisen.

        Bei ratio=1.0 (kein Stretch) und identischem Signal auf L+R muss
        die Korrelation nach WSOLA nahe 1.0 sein.
        """
        p = phase31_instance
        # Identisches Signal auf L und R
        t = np.linspace(0, 0.5, int(p.sample_rate * 0.5), endpoint=False).astype(np.float32)
        sig = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        audio = np.column_stack([sig, sig])

        out = p._correct_wsola(audio, ratio=1.0, params={})

        min_len = min(len(out[:, 0]), len(out[:, 1]))
        corr = float(np.corrcoef(out[:min_len, 0], out[:min_len, 1])[0, 1])
        assert corr > 0.99, (
            f"L/R-Korrelation nach WSOLA zu niedrig: {corr:.4f} (erwartet > 0.99) — deutet auf Zeitversatz hin"
        )

    def test_wsola_mono_output_finite(self, phase31_instance):
        """_wsola_mono gibt keine NaN/Inf-Werte zurück."""
        p = phase31_instance
        sr = p.sample_rate
        window_size = int(0.02 * sr)
        hop_a = window_size // 2
        hop_s = int(hop_a * 0.9)

        rng = np.random.default_rng(42)
        audio = rng.uniform(-0.3, 0.3, sr).astype(np.float32)
        out = p._wsola_mono(audio, window_size, hop_a, hop_s)

        assert np.all(np.isfinite(out)), "WSOLA-Output enthält NaN/Inf"

    def test_preventive_damage_shield_caps_rms_explosion(self, phase31_instance):
        """DSP-Präventionsschicht muss starke Pegelsteigerung lokal begrenzen."""
        p = phase31_instance
        sr = p.sample_rate
        t = np.linspace(0, 1.0, sr, endpoint=False).astype(np.float32)
        orig = (0.02 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
        exploded = (orig * 12.0).astype(np.float32)

        safe, meta = p._apply_preventive_damage_shield(orig, exploded, sr)

        rms_in = float(np.sqrt(np.mean(orig**2)) + 1e-12)
        rms_out = float(np.sqrt(np.mean(safe**2)) + 1e-12)
        delta_db = float(20.0 * np.log10(rms_out / rms_in))

        assert meta.get("phase31_damage_shield_applied") is True
        assert meta.get("phase31_rms_cap_applied") is True
        assert delta_db <= 2.8, f"RMS-Anstieg zu hoch trotz Shield: {delta_db:.2f} dB"
        assert float(meta.get("phase31_peak99_after", 1.0)) <= 0.981

    def test_preventive_damage_shield_corrects_lr_delay(self, phase31_instance):
        """DSP-Präventionsschicht muss interkanalen Zeitversatz korrigieren."""
        p = phase31_instance
        sr = p.sample_rate
        t = np.linspace(0, 1.0, sr, endpoint=False).astype(np.float32)
        left = (0.3 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
        delay = 240
        right = np.pad(left, (delay, 0))[: len(left)]
        drifted = np.column_stack([left, right]).astype(np.float32)

        safe, meta = p._apply_preventive_damage_shield(drifted, drifted, sr)

        corr = np.correlate(safe[:, 0], safe[:, 1], mode="full")
        lag = int(np.argmax(corr) - (len(safe[:, 0]) - 1))

        assert meta.get("phase31_stereo_delay_corrected") is True
        assert abs(lag) <= 2, f"L/R-Zeitversatz nicht korrigiert: lag={lag} samples"
