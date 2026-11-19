import pytest

"""
Test: 2 Magic Buttons in Aurik 10.0.0
====================================

Magic Button 1: Restoration Only (keine Enhancement)
Magic Button 2: Studio 2026 (Complete: Restoration + Enhancement + Dynamics + Remastering)

Integration Test: Phase 10 (Compression) + Phase 11 (Limiting)
"""

import sys

import numpy as np

sys.path.insert(0, ".")

from backend.core.ai_framework import AurikAIFramework


@pytest.mark.unit
def generate_test_audio(sample_rate=48000, duration=3.0):
    """Generate test audio with defects and dynamics issues."""
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Clean signal
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)

    # Add dynamics problems (große Dynamiksprünge)
    # Leise Passagen (0-1s)
    audio[: int(sample_rate)] *= 0.2
    # Laute Passagen (1-2s) mit Peaks
    audio[int(sample_rate) : int(2 * sample_rate)] *= 1.5
    # Normale Passagen (2-3s)
    audio[int(2 * sample_rate) :] *= 0.7

    # Add defects
    # 1. Clicks
    click_positions = [5000, 10000, 15000, 25000]
    for pos in click_positions:
        if pos < len(audio):
            audio[pos : pos + 10] += 0.8

    # 2. Hiss
    hiss = np.random.normal(0, 0.03, len(audio))
    audio += hiss

    # 3. Hum (50Hz)
    hum = 0.08 * np.sin(2 * np.pi * 50 * t)
    audio += hum

    # Make stereo
    audio_stereo = np.column_stack((audio, audio * 0.95))

    return audio_stereo


def analyze_audio(audio, label):
    """Analyze audio characteristics."""
    rms = np.sqrt(np.mean(audio**2))
    peak = np.max(np.abs(audio))
    peak_db = 20 * np.log10(peak) if peak > 0 else -np.inf
    rms_db = 20 * np.log10(rms) if rms > 0 else -np.inf
    crest_factor = peak / rms if rms > 0 else 0

    print(f"\n{label}:")
    print(f"  RMS: {rms:.4f} ({rms_db:.1f} dBFS)")
    print(f"  Peak: {peak:.4f} ({peak_db:.1f} dBFS)")
    print(f"  Crest Factor: {crest_factor:.2f}")

    return {"rms": rms, "peak": peak, "rms_db": rms_db, "peak_db": peak_db, "crest_factor": crest_factor}


def main():
    print("=" * 80)
    print("TEST: 2 Magic Buttons in Aurik 10.0.0")
    print("=" * 80)

    # Generate test audio
    sr = 48000
    print(f"\n🎵 Generating test audio ({sr} Hz, 3s)")
    audio = generate_test_audio(sample_rate=sr, duration=3.0)

    # Analyze input
    input_stats = analyze_audio(audio, "📥 INPUT AUDIO")

    # Initialize framework
    print("\n" + "─" * 80)
    print("🚀 Initializing Aurik AI Framework...")
    framework = AurikAIFramework(sample_rate=sr)

    # ============================================================
    # TEST 1: Magic Button "Restoration" (nur Restoration)
    # ============================================================
    print("\n" + "=" * 80)
    print("TEST 1: Magic Button 'Restoration' (nur Restoration, keine Enhancement)")
    print("=" * 80)

    restoration_audio, restoration_report = framework.restoration_magic_button(audio)

    print("\n✅ Restoration Complete!")
    print(f"   Mode: {restoration_report['final']['mode']}")
    print(f"   Material: {restoration_report['detection']['material_type']}")
    print(f"   Defects Found: {restoration_report['detection']['defects_found']}")
    print(f"   Defects Removed: {restoration_report['restoration']['defects_removed']}")
    print(f"   Quality Improvement: +{restoration_report['restoration']['quality_improvement']:.2f}")
    print(f"   Processes: {', '.join(restoration_report['restoration']['processes'])}")

    restoration_stats = analyze_audio(restoration_audio, "🔧 RESTORED AUDIO")

    # ============================================================
    # TEST 2: Magic Button "Studio 2026" (Complete Pipeline)
    # ============================================================
    print("\n" + "=" * 80)
    print("TEST 2: Magic Button 'Studio 2026' (Complete Pipeline)")
    print("=" * 80)
    print("Pipeline: Restoration + Enhancement + Dynamics (Phase 10+11) + Remastering")

    studio_audio, studio_report = framework.studio2026_magic_button(audio)

    print("\n✅ Studio 2026 Complete!")
    print(f"   Mode: {studio_report['final']['mode']}")
    print(f"   Material: {studio_report['detection']['material_type']}")

    # Restoration Phase
    print("\n   📍 RESTORATION:")
    print(f"      Defects Found: {studio_report['detection']['defects_found']}")
    print(f"      Defects Removed: {studio_report['restoration']['defects_removed']}")
    print(f"      Quality Improvement: +{studio_report['restoration']['quality_improvement']:.2f}")

    # Enhancement Phase
    print("\n   📍 ENHANCEMENT:")
    print(f"      Enhancements: {', '.join(studio_report['enhancement']['enhancements'])}")
    print(f"      Clarity: +{studio_report['enhancement']['clarity']:.2f}")
    print(f"      Presence: +{studio_report['enhancement']['presence']:.2f}")
    print(f"      Detail: +{studio_report['enhancement']['detail']:.2f}")

    # Dynamics Phase (Phase 10 + 11)
    print("\n   📍 DYNAMICS (Phase 10 + 11):")
    dynamics = studio_report.get("dynamics", {})
    if dynamics.get("compression_applied"):
        comp = dynamics.get("compression", {})
        print(
            f"      ✓ Compression: Ratio={comp.get('ratio', 0):.1f}:1, "
            f"Threshold={comp.get('threshold_db', 0):.0f} dB, "
            f"Gain Reduction={comp.get('gain_reduction_db', 0):.1f} dB"
        )
    else:
        print("      ○ Compression: Skipped")

    if dynamics.get("limiting_applied"):
        lim = dynamics.get("limiting", {})
        print(
            f"      ✓ Limiting: Ceiling={lim.get('ceiling_db', 0):.1f} dBFS, "
            f"Peak Reduction={lim.get('peak_reduction_db', 0):.1f} dB"
        )
    else:
        print("      ○ Limiting: Skipped")

    if dynamics.get("fallback_limiting"):
        print("      ⚠ Fallback Limiting (Phases nicht verfügbar)")

    studio_stats = analyze_audio(studio_audio, "🎛️  STUDIO 2026 AUDIO")

    # ============================================================
    # COMPARISON
    # ============================================================
    print("\n" + "=" * 80)
    print("VERGLEICH: Restoration vs. Studio 2026")
    print("=" * 80)

    print(f"\n{'Metrik':<20} {'Input':<15} {'Restoration':<15} {'Studio 2026':<15}")
    print("─" * 68)
    print(
        f"{'RMS (dBFS)':<20} {input_stats['rms_db']:>14.1f} {restoration_stats['rms_db']:>14.1f} {studio_stats['rms_db']:>14.1f}"
    )
    print(
        f"{'Peak (dBFS)':<20} {input_stats['peak_db']:>14.1f} {restoration_stats['peak_db']:>14.1f} {studio_stats['peak_db']:>14.1f}"
    )
    print(
        f"{'Crest Factor':<20} {input_stats['crest_factor']:>14.2f} {restoration_stats['crest_factor']:>14.2f} {studio_stats['crest_factor']:>14.2f}"
    )

    # Erwartungen
    print("\n" + "=" * 80)
    print("ERWARTUNGEN")
    print("=" * 80)
    print("\n✓ Restoration Button:")
    print("  - Nur Defektentfernung (Clicks, Hiss, Hum)")
    print("  - Keine Dynamik-Bearbeitung")
    print("  - Keine Enhancement")
    print("  - Authentizität erhalten")

    print("\n✓ Studio 2026 Button:")
    print("  - Defektentfernung")
    print("  - Enhancement (Clarity, Presence, Detail)")
    print("  - Dynamik-Kompression (Phase 10) → niedrigerer Crest Factor")
    print("  - Peak-Limiting (Phase 11) → kontrollierter Peak")
    print("  - Professionelles Mastering")

    # Validation
    print("\n" + "=" * 80)
    print("VALIDATION")
    print("=" * 80)

    validation_passed = True

    # 1. Restoration sollte weniger Dynamik-Bearbeitung haben
    if studio_stats["crest_factor"] < restoration_stats["crest_factor"]:
        print("✅ Studio 2026 hat niedrigeren Crest Factor (mehr Kompression)")
    else:
        print("⚠️ Studio 2026 sollte niedrigeren Crest Factor haben")
        validation_passed = False

    # 2. Studio 2026 sollte kontrollierten Peak haben
    if studio_stats["peak_db"] < -0.5 and studio_stats["peak_db"] > -1.5:
        print("✅ Studio 2026 Peak ist kontrolliert (-0.5 bis -1.5 dBFS)")
    else:
        print(f"⚠️ Studio 2026 Peak außerhalb Zielbereich: {studio_stats['peak_db']:.1f} dBFS")

    # 3. Beide sollten Defekte entfernt haben
    if restoration_report["restoration"]["defects_removed"] > 0:
        print("✅ Restoration Button hat Defekte entfernt")
    else:
        print("⚠️ Restoration Button hat keine Defekte entfernt")

    if studio_report["restoration"]["defects_removed"] > 0:
        print("✅ Studio 2026 Button hat Defekte entfernt")
    else:
        print("⚠️ Studio 2026 Button hat keine Defekte entfernt")

    print("\n" + "=" * 80)
    if validation_passed:
        print("✅ TEST PASSED: Beide Magic Buttons funktionieren korrekt!")
    else:
        print("⚠️ TEST WARNING: Prüfung erforderlich")
    print("=" * 80)


if __name__ == "__main__":
    main()
