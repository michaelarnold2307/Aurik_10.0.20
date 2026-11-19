"""
Standalone-Messung: UnifiedRestorerV3 mit QualityMode.QUALITY auf Testkünstlerin (Schlager).mp3
Vergleicht gemessene Goal-Scores gegen kanonische §0-Schwellen.
Aurik 10.0.0 — 2026-04-19
"""

# pylint: disable=wrong-import-position
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

AUDIO_PATH = (
    ROOT / "test_audio" / "Testkünstlerin (Schlager) - Du wolltest nur ein Abenteuer, aber ich suchte einen Freund.mp3"
)

from backend.core.calibration_matrix import CANONICAL_THRESHOLDS_RESTORATION as CANONICAL

PRIO = {
    "P1": ["natuerlichkeit", "authentizitaet"],
    "P2": ["tonal_center", "timbre_authentizitaet", "artikulation", "transient_energie"],
    "P3": ["emotionalitaet", "micro_dynamics", "groove"],
    "P4": ["transparenz", "waerme", "bass_kraft", "separation_fidelity"],
    "P5": ["brillanz", "spatial_depth"],
}


def _lufs(audio, sr):
    """Berechnet die integrierte Lautheit (LUFS) des Audio-Arrays."""
    try:
        import pyloudnorm as pyln  # pylint: disable=import-outside-toplevel

        meter = pyln.Meter(sr)
        mono = audio[:, 0] if audio.ndim == 2 else audio
        return float(meter.integrated_loudness(mono.astype(np.float64)))
    except Exception:
        rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)) + 1e-12)
        return float(20 * np.log10(rms))


def main():
    """Führt die Qualitätsmessung durch und gibt die Ergebnisse auf der Konsole aus."""
    print("=" * 70)
    print("Aurik 10.0.0 — QualityMode.QUALITY Messung")
    print(f"Song: {AUDIO_PATH.name}")
    print("=" * 70)

    if not AUDIO_PATH.exists():
        print(f"FEHLER: Audio-Datei nicht gefunden: {AUDIO_PATH}")
        sys.exit(1)

    # pylint: disable=import-outside-toplevel
    from backend.core.musical_goals.musical_goals_metrics import MusicalGoalsChecker
    from backend.core.performance_guard import QualityMode
    from backend.core.unified_restorer_v3 import RestorationConfig, UnifiedRestorerV3
    from backend.file_import import load_audio_file
    # pylint: enable=import-outside-toplevel

    print("\n[1/4] Lade Audio...")
    loaded = load_audio_file(str(AUDIO_PATH), target_sr=None, mono=False, do_carrier_analysis=False)
    if loaded is None:
        print("FEHLER: Audio-Datei konnte nicht geladen werden.")
        sys.exit(1)
    audio = np.asarray(loaded["audio"], dtype=np.float32)
    sr = int(loaded.get("sr") or 48_000)

    # samples-first
    if audio.ndim == 2 and audio.shape[0] in (1, 2) and audio.shape[1] > audio.shape[0]:
        audio = audio.T
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=1)

    # 20s zentrierter Clip
    max_n = int(sr * 20)
    if audio.shape[0] > max_n:
        start = (audio.shape[0] - max_n) // 2
        audio = audio[start : start + max_n]

    if sr != 48_000:
        import librosa  # pylint: disable=import-outside-toplevel

        audio = librosa.resample(audio.T, orig_sr=sr, target_sr=48_000).T.astype(np.float32)
        sr = 48_000

    print(f"   Shape: {audio.shape}, SR: {sr}, Länge: {audio.shape[0] / sr:.1f}s")

    print("\n[2/4] Goals VOR Restaurierung messen...")
    checker = MusicalGoalsChecker(mode="restoration")
    mono_orig = (audio[:, 0] + audio[:, 1]) / 2.0
    goals_before = checker.measure_all(mono_orig, sr)
    lufs_before = _lufs(audio, sr)
    print(f"   LUFS vorher: {lufs_before:.1f} LUFS")

    print("\n[3/4] Restaurierung mit QualityMode.QUALITY starten...")
    t0 = time.time()
    cfg = RestorationConfig(
        mode=QualityMode.QUALITY,
        enable_performance_guard=True,
        enable_phase_gate=True,
        enable_phase_skipping=True,
    )
    restorer = UnifiedRestorerV3(config=cfg)
    result = restorer.restore(
        audio.T,
        sample_rate=sr,
        mode="quality",
        ml_runtime_budget_s=120.0,
    )
    elapsed = time.time() - t0
    print(f"   Fertig in {elapsed:.1f}s")

    restored = np.asarray(result.audio, dtype=np.float32)
    if restored.ndim == 2 and restored.shape[0] in (1, 2) and restored.shape[1] > restored.shape[0]:
        restored = restored.T
    if restored.ndim == 1:
        restored = np.stack([restored, restored], axis=1)

    n = min(audio.shape[0], restored.shape[0])
    rest_clip = restored[:n]

    print("\n[4/4] Goals NACH Restaurierung messen...")
    mono_rest = (rest_clip[:, 0] + rest_clip[:, 1]) / 2.0
    goals_after = checker.measure_all(mono_rest, sr, reference=mono_orig[:n])
    lufs_after = _lufs(rest_clip, sr)

    # ── Auswertung ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ERGEBNISSE — Musical Goals (15 Ziele)")
    print("=" * 70)
    print(f"{'Goal':<28} {'Vorher':>8} {'Nachher':>8} {'Kanon.':>8} {'∆':>7}  Status")
    print("-" * 70)

    passed = 0
    failed_goals = []
    for prio_label, goals in PRIO.items():
        for goal in goals:
            before = float(goals_before.get(goal, 0.0))
            after = float(goals_after.get(goal, 0.0))
            threshold = CANONICAL.get(goal, 0.70)
            delta = after - before
            ok = after >= threshold
            status = "✅" if ok else "❌"
            if ok:
                passed += 1
            else:
                failed_goals.append((goal, after, threshold))
            print(
                f"  [{prio_label}] {goal:<24} {before:>8.3f} {after:>8.3f} {threshold:>8.2f} {delta:>+7.3f}  {status}"
            )

    print("-" * 70)
    print(f"\n  LUFS: {lufs_before:.1f} → {lufs_after:.1f} LUFS  (Δ {lufs_after - lufs_before:+.1f})")
    print(f"  Material: {getattr(getattr(result, 'material_type', 'unknown'), 'value', 'unknown')}")
    print(f"  Laufzeit: {elapsed:.1f}s")
    total_goals = len(CANONICAL)
    print(f"\n  Goals erfüllt: {passed}/{total_goals}  (kanonische §0-Schwellen)")

    if failed_goals:
        print("\n  Nicht erfüllt:")
        for g, v, t in failed_goals:
            print(f"    {g}: {v:.3f} < {t:.2f}")

    # NaN/Inf Check
    has_nan = not np.isfinite(rest_clip).all()
    has_clip = float(np.max(np.abs(rest_clip))) > 1.0 + 1e-6
    print(f"\n  NaN/Inf im Export: {'JA ⚠️' if has_nan else 'Nein ✅'}")
    print(f"  Clipping im Export: {'JA ⚠️' if has_clip else 'Nein ✅'}")

    print("\n" + "=" * 70)
    if passed == total_goals:
        print("🎯 ALLE 15 GOALS ERFÜLLT — Studio-Klangtreue erreicht")
    elif passed >= 10:
        print(f"⚠️  {total_goals - passed} Goal(s) unter Schwelle — gute Restaurierung, aber nicht vollständig")
    else:
        print(f"❌ {total_goals - passed} Goals unter Schwelle — weitere Arbeit nötig")
    print("=" * 70)


if __name__ == "__main__":
    main()
