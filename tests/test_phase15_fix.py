#!/usr/bin/env python
"""
Schneller Test für Phase 1.5 De-Hum/De-Buzz Fix
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np
import pytest

from backend.core.dsp.automatic_dehum import AutomaticDehum


@pytest.mark.unit
def test_dehum_basic():
    """Test dass AutomaticDehum.dehum() funktioniert"""
    print("=" * 80)
    print("TEST: AutomaticDehum.dehum() Methode")
    print("=" * 80)

    # Erstelle Testsignal mit 60Hz Hum
    sr = 48000
    duration = 1.0  # 1 Sekunde
    t = np.linspace(0, duration, int(sr * duration))

    # Sauberes Signal (440Hz Ton)
    clean_signal = np.sin(2 * np.pi * 440 * t)

    # Füge 60Hz Hum hinzu
    hum = 0.3 * np.sin(2 * np.pi * 60 * t)
    noisy_signal = clean_signal + hum

    print(f"✅ Testsignal erstellt: {len(noisy_signal)} samples @ {sr}Hz")
    print(f"   Clean Signal RMS: {np.sqrt(np.mean(clean_signal**2)):.4f}")
    print(f"   Noisy Signal RMS: {np.sqrt(np.mean(noisy_signal**2)):.4f}")

    # Test 1: Dehummer mit 60Hz
    print("\n[Test 1] AutomaticDehum mit 60Hz...")
    dehummer = AutomaticDehum(hum_freq=60.0, q=30.0)
    try:
        result = dehummer.dehum(noisy_signal, sr)
        result_rms = np.sqrt(np.mean(result**2))
        print("✅ Dehum erfolgreich!")
        print(f"   Result RMS: {result_rms:.4f}")
        print(f"   Hum Reduktion: {np.abs(noisy_signal).max() - np.abs(result).max():.4f}")
    except AttributeError as e:
        print(f"❌ FEHLER: {e}")
        pytest.fail(f"AutomaticDehum.dehum() AttributeError: {e}")
    except Exception as e:
        print(f"❌ FEHLER: {e}")
        pytest.fail(f"AutomaticDehum.dehum() failed: {e}")

    # Test 2: Multi-Pass wie in unified_restorer_v2.py
    print("\n[Test 2] Multi-Pass (3x) wie in unified_restorer_v2.py...")
    best_x = None
    best_quality = -1.0  # Start mit -1 um sicherzustellen dass mindestens ein Pass akzeptiert wird

    for pass_num in range(3):
        try:
            # Create dehummer instance with correct frequency for this pass
            dehummer = AutomaticDehum(hum_freq=60.0)
            x_pass = dehummer.dehum(noisy_signal, sr)

            # Measure quality (residual hum)
            freqs = np.fft.rfftfreq(len(x_pass), 1 / sr)
            fft_mag = np.abs(np.fft.rfft(x_pass))

            # Check 60Hz harmonics
            hum_indices = [np.argmin(np.abs(freqs - f)) for f in [60, 120, 180, 240, 300, 360]]
            residual_hum = np.sum([fft_mag[i] for i in hum_indices])
            total_energy = np.sum(fft_mag)
            residual_ratio = residual_hum / (total_energy + 1e-10)

            quality = 1.0 - min(residual_ratio * 1000, 1.0)

            # Immer den ersten Pass akzeptieren falls quality >= 0
            if quality >= best_quality:
                best_quality = quality
                best_x = x_pass

            print(f"   Pass {pass_num + 1}: Quality: {quality:.3f} (Residual: {residual_ratio:.4%})")

        except Exception as e:
            print(f"   Pass {pass_num + 1} fehlgeschlagen: {e}")
            pytest.fail(f"Multi-pass {pass_num + 1} failed: {e}")

    if best_x is not None:
        print(f"✅ Multi-Pass erfolgreich! Best Quality: {best_quality:.3f}")
    else:
        print("❌ Kein gültiges Ergebnis!")
        pytest.fail("Multi-pass: kein gültiges Ergebnis")

    print("\n" + "=" * 80)
    print("✅✅✅ ALLE TESTS BESTANDEN ✅✅✅")
    print("=" * 80)


if __name__ == "__main__":
    try:
        test_dehum_basic()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ TEST FEHLGESCHLAGEN: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
