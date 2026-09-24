"""Audio-Evidenz-Harness fuer die Defect-Scanner-Messung (§SR-Audit, Spec 06 §7.2d).

Prueft pro Defekt-Familie: deterministische Synthese-Signale -> DefectScanner ->
gemeldete Scores gegen Erwartungswerte. Der Harness dokumentiert Mess-Luecken
als fehlschlagende Faelle, bis die Detektoren kalibriert/ergaenzt sind
(Arbeitsauftrag Spec 06 §7.2d).

Deterministisch (§G5, copilot-instructions.md): feste Seeds, keine Zufallsquellen
ausser den dokumentierten Generatoren.

Nutzung: python scripts/defect_evidence_harness.py noise
Exit-Code 0 = alle Erwartungen erfuellt, 1 = mindestens eine Luecke.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable

import numpy as np

logging.disable(logging.CRITICAL)

from backend.core.defect_scanner import DefectScanner, DefectType, MaterialType

SR = 48000
DUR = 15


# ------------------------------------------------------------------ Synthese
def _tape_hiss() -> np.ndarray:
    rng = np.random.default_rng(1)
    x = rng.normal(0, 0.003, SR * DUR)
    x = x + 0.4 * np.diff(x, prepend=0.0)  # HF-Anhebung (Tape-Hiss-Charakter)
    _out: np.ndarray = np.stack([x, x], axis=1).astype(np.float32)
    return _out


def _modulation_noise() -> np.ndarray:
    rng = np.random.default_rng(2)
    t = np.arange(SR * DUR) / SR
    tone = 0.2 * np.sin(2 * np.pi * 220.0 * t)
    # Modulationsindex ~10 (Basis 0,02): Rauschflur folgt dem Signal deutlich —
    # mit Basis 0,05 war das Verhaeltnis nach Frame-RMS-Mittelung nur ~1,1.
    x = tone + rng.normal(0, 1.0, SR * DUR) * (np.abs(tone) + 0.02) * 0.03
    _out: np.ndarray = np.stack([x, x], axis=1).astype(np.float32)
    return _out


def _clean_tone() -> np.ndarray:
    t = np.arange(SR * DUR) / SR
    x = 0.2 * np.sin(2 * np.pi * 220.0 * t)
    _out: np.ndarray = np.stack([x, x], axis=1).astype(np.float32)
    return _out


def _broadband() -> np.ndarray:
    rng = np.random.default_rng(3)
    x = rng.normal(0, 0.01, SR * DUR)
    _out: np.ndarray = np.stack([x, x], axis=1).astype(np.float32)
    return _out


def _fm_tone(mod_freq: float, depth_pct: float) -> np.ndarray:
    """Ton mit Frequenzmodulation (Wow/Flutter-Synthese)."""
    t = np.arange(SR * DUR) / SR
    phase = 2 * np.pi * 220.0 * t + (depth_pct / 100.0) * (220.0 / mod_freq) * np.sin(2 * np.pi * mod_freq * t)
    x = 0.2 * np.sin(phase)
    _out: np.ndarray = np.stack([x, x], axis=1).astype(np.float32)
    return _out


def _wow() -> np.ndarray:
    return _fm_tone(0.3, 0.5)  # 0,3 Hz, ±0,5 % (IEC 60386 Wow)


def _flutter() -> np.ndarray:
    return _fm_tone(6.0, 0.2)  # 6 Hz, ±0,2 % (IEC 60386 Flutter)


def _speed_offset() -> np.ndarray:
    t = np.arange(SR * DUR) / SR
    x = 0.2 * np.sin(2 * np.pi * 220.0 * 1.004 * t)  # konstant +0,4 %
    _out: np.ndarray = np.stack([x, x], axis=1).astype(np.float32)
    return _out


# ------------------------------------------------------------------ Familien
# Jeder Fall: (Name, Generator, Material, {DefectType: (min, max)})
CASES: dict[str, list[tuple[str, Callable[[], np.ndarray], MaterialType, dict[DefectType, tuple[float, float]]]]] = {
    "noise": [
        (
            "Tape-Hiss wird als HISS typisiert",
            _tape_hiss,
            MaterialType.TAPE,
            {DefectType.HISS: (0.3, 1.0)},
        ),
        (
            "Modulations-Rauschen wird erkannt (TAPE)",
            _modulation_noise,
            MaterialType.TAPE,
            {DefectType.MODULATION_NOISE: (0.2, 1.0)},
        ),
        (
            "Reiner Ton loest kein quantization_noise aus",
            _clean_tone,
            MaterialType.TAPE,
            {DefectType.QUANTIZATION_NOISE: (0.0, 0.2)},
        ),
        (
            "Breitband-Rauschen wird als high_freq_noise erkannt",
            _broadband,
            MaterialType.TAPE,
            {DefectType.HIGH_FREQ_NOISE: (0.8, 1.0)},
        ),
    ],
    "wow_flutter": [
        (
            "Wow (0,3 Hz FM ±0,5 %) wird erkannt",
            _wow,
            MaterialType.TAPE,
            {DefectType.WOW: (0.3, 1.0)},
        ),
        (
            "Flutter (6 Hz FM ±0,2 %) wird erkannt",
            _flutter,
            MaterialType.TAPE,
            {DefectType.FLUTTER: (0.3, 1.0)},
        ),
        (
            "Sauberer Ton loest kein wow/flutter aus",
            _clean_tone,
            MaterialType.TAPE,
            {DefectType.WOW: (0.0, 0.2), DefectType.FLUTTER: (0.0, 0.2)},
        ),
        (
            "Konstanter Speed-Offset wird als speed_calibration_error erkannt",
            _speed_offset,
            MaterialType.TAPE,
            {DefectType.SPEED_CALIBRATION_ERROR: (0.3, 1.0)},
        ),
    ],
}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in CASES:
        print(f"Familien: {', '.join(sorted(CASES))}")
        return 2
    family = sys.argv[1]
    failed = 0
    for name, gen, material, expects in CASES[family]:
        sc = DefectScanner(sample_rate=SR, material_type=material)
        res = sc.scan(gen())
        line = []
        for dt, (lo, hi) in expects.items():
            score = res.scores.get(dt)
            v = float(score.severity) if score is not None else 0.0
            ok = lo <= v <= hi
            if not ok:
                failed += 1
            line.append(f"{dt.value}={v:.3f} (erwartet {lo:.2f}-{hi:.2f}) {'OK' if ok else 'FEHLT'}")
        print(f"[{'OK' if all('OK' in l for l in line) else 'LUECKE'}] {name}: {'; '.join(line)}")
    print(f"Familie {family}: {failed} Luecke(n)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
