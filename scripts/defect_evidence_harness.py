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
from scipy import signal

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


def _dropout_oxide() -> np.ndarray:
    t = np.arange(SR * DUR) / SR
    x = 0.2 * np.sin(2 * np.pi * 220.0 * t)
    x[SR * 7 : SR * 7 + int(0.006 * SR)] = 0.0  # 6-ms-Signalabriss (Oxid-Dropout)
    _out: np.ndarray = np.stack([x, x], axis=1).astype(np.float32)
    return _out


def _dropout_head_contact() -> np.ndarray:
    t = np.arange(SR * DUR) / SR
    x = 0.2 * np.sin(2 * np.pi * 220.0 * t)
    dip = np.ones(SR * DUR)
    dip[SR * 7 : SR * 7 + int(0.2 * SR)] = 0.3  # 200-ms-Kontaktverlust (-10 dB)
    _out: np.ndarray = np.stack([x * dip, x * dip], axis=1).astype(np.float32)
    return _out


def _dropout_splice() -> np.ndarray:
    t = np.arange(SR * DUR) / SR
    x = 0.2 * np.sin(2 * np.pi * 220.0 * t)
    # Echter Bandschnitt: Pegelsprung + Phasenversatz (die Bandenden sind
    # nicht phasen-aligned) + Klick an der Klebestelle.
    x[SR * 7 :] = 0.2 * np.sin(2 * np.pi * 220.0 * t[SR * 7 :] + 0.8) * 0.7
    x[SR * 7 : SR * 7 + 12] += 0.15  # Klick an der Klebestelle
    _out: np.ndarray = np.stack([x, x], axis=1).astype(np.float32)
    return _out


def _vinyl_igd() -> np.ndarray:
    t = np.arange(SR * DUR) / SR
    ramp = np.linspace(0.0, 1.0, SR * DUR)
    x = 0.2 * np.sin(2 * np.pi * 440.0 * t)
    x = x + ramp * 0.1 * np.sin(2 * np.pi * 880.0 * t)  # H2 waechst zur Innenseite
    x = x + ramp * 0.05 * np.sin(2 * np.pi * 1320.0 * t)  # H3 waechst
    _out: np.ndarray = np.stack([x, x], axis=1).astype(np.float32)
    return _out


def _vinyl_groove_echo() -> np.ndarray:
    t = np.arange(SR * DUR) / SR
    x = 0.05 * np.sin(2 * np.pi * 440.0 * t)  # leise Grundlage
    burst = np.zeros(SR * DUR)
    burst[SR * 10 : SR * 11] = 0.5 * np.sin(2 * np.pi * 880.0 * t[SR * 10 : SR * 11])
    # Ghost 1,8 s VOR dem lauten Durchgang (Nachbarrillen-Eindruck, -10 dB)
    ghost = np.zeros(SR * DUR)
    ghost_start = SR * 10 - int(1.8 * SR)
    ghost[ghost_start : ghost_start + SR] = 0.15 * burst[SR * 10 : SR * 11]
    sig = x + burst + ghost
    _out: np.ndarray = np.stack([sig, sig], axis=1).astype(np.float32)
    return _out


def _vinyl_riaa() -> np.ndarray:
    t = np.arange(SR * DUR) / SR
    x = 0.2 * np.sin(2 * np.pi * 440.0 * t)
    sos = signal.butter(1, 1000, btype="high", fs=SR, output="sos")
    x = x + 1.2 * signal.sosfiltfilt(sos, x)  # falsche RIAA: HF-Anhebung
    _out: np.ndarray = np.stack([x, x], axis=1).astype(np.float32)
    return _out


def _vinyl_motor() -> np.ndarray:
    t = np.arange(SR * DUR) / SR
    x = 0.2 * np.sin(2 * np.pi * 440.0 * t)
    x = x + 0.03 * np.sin(2 * np.pi * 100.0 * t)
    x = x + 0.02 * np.sin(2 * np.pi * 200.0 * t)
    x = x + 0.01 * np.sin(2 * np.pi * 300.0 * t)
    _out: np.ndarray = np.stack([x, x], axis=1).astype(np.float32)
    return _out


def _vinyl_motor_dense() -> np.ndarray:
    t = np.arange(SR * DUR) / SR
    x = 0.2 * np.sin(2 * np.pi * 440.0 * t)
    for f, a in ((80, 0.03), (160, 0.025), (240, 0.02), (300, 0.015)):
        x = x + a * np.sin(2 * np.pi * f * t)
    _out: np.ndarray = np.stack([x, x], axis=1).astype(np.float32)
    return _out


def _vinyl_stylus() -> np.ndarray:
    t = np.arange(SR * DUR) / SR
    x = 0.2 * np.sin(2 * np.pi * 440.0 * t)
    x[x > 0] = x[x > 0] * 0.4  # asymmetrische Abtastverzerrung
    _out: np.ndarray = np.stack([x, x], axis=1).astype(np.float32)
    return _out


def _spectral_aliasing() -> np.ndarray:
    t = np.arange(SR * DUR) / SR
    # Near-Nyquist-Energie (21 kHz) ohne musikalische Quelle: AA-Filter-Versagen
    x = 0.12 * np.sin(2 * np.pi * 12000.0 * t)
    x = x + 0.15 * np.sin(2 * np.pi * 21000.0 * t)
    _out: np.ndarray = np.stack([x, x], axis=1).astype(np.float32)
    return _out


def _spectral_imd() -> np.ndarray:
    t = np.arange(SR * DUR) / SR
    # Zwei-Ton-IMD: 3 kHz + 4 kHz durch quadratische Nichtlinearitaet
    # -> Differenz-/Summentoene (1/2/5/7 kHz)
    x = np.sin(2 * np.pi * 3000.0 * t) + np.sin(2 * np.pi * 4000.0 * t)
    x = x + 0.2 * x**2
    _out: np.ndarray = np.stack([0.15 * x, 0.15 * x], axis=1).astype(np.float32)
    return _out


def _spectral_quantization() -> np.ndarray:
    t = np.arange(SR * DUR) / SR
    x = 0.2 * np.sin(2 * np.pi * 220.0 * t)
    q = np.round(x * 127.0) / 127.0  # 8-Bit-Quantisierung
    err = x - q
    _out: np.ndarray = np.stack([q + 0.8 * err, q + 0.8 * err], axis=1).astype(np.float32)
    return _out


def _spectral_dc() -> np.ndarray:
    t = np.arange(SR * DUR) / SR
    x = 0.2 * np.sin(2 * np.pi * 220.0 * t) + 0.15  # konstanter DC-Anteil
    _out: np.ndarray = np.stack([x, x], axis=1).astype(np.float32)
    return _out


def _spectral_phase_rotation() -> np.ndarray:
    t = np.arange(SR * DUR) / SR
    # Mehrtoene noetig: Dispersion ist frequenzabhaengig - ein Einzelton
    # traegt keine Gruppenlaufzeit-Varianz-Signatur.
    x = 0.2 * np.sin(2 * np.pi * 220.0 * t)
    x = x + 0.15 * np.sin(2 * np.pi * 880.0 * t)
    x = x + 0.1 * np.sin(2 * np.pi * 1760.0 * t)
    a = 0.7  # Allpass 1. Ordnung: frequenzabhaengige Phasendrehung
    y = np.zeros_like(x)
    y[0] = x[0]
    for n in range(1, len(x)):
        y[n] = -a * x[n] + x[n - 1] + a * y[n - 1]
    _out: np.ndarray = np.stack([y, y], axis=1).astype(np.float32)
    return _out


def _spectral_phase_issues() -> np.ndarray:
    t = np.arange(SR * DUR) / SR
    x = 0.2 * np.sin(2 * np.pi * 220.0 * t)
    _out: np.ndarray = np.stack([x, -x], axis=1).astype(np.float32)  # Kanal gegenpolig
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
    "dropout": [
        (
            "Oxid-Dropout (6 ms) — Severity ist Dauer-Anteil, ein Abriss in 15 s ~0,007",
            _dropout_oxide,
            MaterialType.TAPE,
            {DefectType.DROPOUTS: (0.005, 0.1)},
        ),
        (
            "Head-Contact-Dip (200 ms, -10 dB) wird erkannt",
            _dropout_head_contact,
            MaterialType.TAPE,
            {DefectType.DROPOUTS: (0.3, 1.0)},
        ),
        (
            "Splice-Pegelsprung wird als tape_splice_artifact erkannt",
            _dropout_splice,
            MaterialType.TAPE,
            {DefectType.TAPE_SPLICE_ARTIFACT: (0.3, 1.0)},
        ),
        (
            "Sauberer Ton loest keine Dropouts aus",
            _clean_tone,
            MaterialType.TAPE,
            {DefectType.DROPOUTS: (0.0, 0.2)},
        ),
    ],
    "vinyl": [
        (
            "IGD (zur Innenseite wachsende H2/H3) wird erkannt",
            _vinyl_igd,
            MaterialType.VINYL,
            {DefectType.INNER_GROOVE_DISTORTION: (0.3, 1.0)},
        ),
        (
            "Groove-Echo (1,8-s-Pre-Echo) wird erkannt",
            _vinyl_groove_echo,
            MaterialType.VINYL,
            {DefectType.GROOVE_ECHO: (0.3, 1.0)},
        ),
        (
            "Falsche RIAA (HF-Anhebung) wird erkannt",
            _vinyl_riaa,
            MaterialType.VINYL,
            {DefectType.RIAA_CURVE_ERROR: (0.3, 1.0)},
        ),
        (
            "Motor-Interferenz (100/200/300 Hz) wird erkannt",
            _vinyl_motor,
            MaterialType.VINYL,
            {DefectType.MOTOR_INTERFERENCE: (0.3, 1.0)},
        ),
        (
            "Dichter Motor-Kamm (80/160/240/300 Hz) wird erkannt",
            _vinyl_motor_dense,
            MaterialType.VINYL,
            {DefectType.MOTOR_INTERFERENCE: (0.3, 1.0)},
        ),
        (
            "Asymmetrische Abtastverzerrung wird erkannt",
            _vinyl_stylus,
            MaterialType.VINYL,
            {DefectType.STYLUS_DAMAGE: (0.3, 1.0)},
        ),
    ],
    "spectral": [
        (
            "Aliasing (Near-Nyquist-Energie) wird erkannt",
            _spectral_aliasing,
            MaterialType.TAPE,
            {DefectType.ALIASING: (0.3, 1.0)},
        ),
        (
            "Intermodulationsverzerrung (3+4 kHz) wird erkannt",
            _spectral_imd,
            MaterialType.VINYL,
            {DefectType.INTERMODULATION_DISTORTION: (0.3, 1.0)},
        ),
        (
            "8-Bit-Quantisierungsrauschen wird erkannt",
            _spectral_quantization,
            MaterialType.TAPE,
            {DefectType.QUANTIZATION_NOISE: (0.3, 1.0)},
        ),
        (
            "DC-Offset wird erkannt",
            _spectral_dc,
            MaterialType.TAPE,
            {DefectType.DC_OFFSET: (0.3, 1.0)},
        ),
        (
            "Phasendrehung (Allpass) wird erkannt",
            _spectral_phase_rotation,
            MaterialType.TAPE,
            {DefectType.PHASE_ROTATION: (0.3, 1.0)},
        ),
        (
            "Gegenpoliger Kanal wird erkannt",
            _spectral_phase_issues,
            MaterialType.TAPE,
            {DefectType.PHASE_ISSUES: (0.3, 1.0)},
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
