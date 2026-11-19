"""Audio-Evidenz-Harness fuer die Defect-Scanner-Messung (§SR-Audit, Spec 06 §7.2d).

Prueft pro Defekt-Familie: deterministische Synthese-Signale -> DefectScanner ->
gemeldete Scores gegen Erwartungswerte. Der Harness dokumentiert Mess-Luecken
als fehlschlagende Faelle, bis die Detektoren kalibriert/ergaenzt sind
(Arbeitsauftrag Spec 06 §7.2d).

Pfad-Aufteilung der Enum-Typen (gemessen 2026-09-24):
- DISTORTION und DROPOUT (Singular) werden NICHT vom DefectScanner.scan()
  befuellt, sondern vom UnifiedDefectDetector (backend/core/ai_framework.py,
  _detect_distortion/_detect_dropout) - der Harness misst sie in der
  "framework"-Familie.
- DROPOUT_SPLICE ist der Scanner-Subtyp fuer abrupte Abrisse mit >95 %
  Pegelverlust (_classify_dropout_subtype); Bandstoesse deckt der
  eigenstaendige TAPE_SPLICE_ARTIFACT-Detektor ab (gemessen: 0,589 auf dem
  Splice-Fall, Subtyp 0 - anderes Profil).

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

from backend.core.defect_scanner import DefectScanner, DefectScore, DefectType, MaterialType

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
    # Echter Bandschnitt: Pegelsprung (-6 dB) + Phasenversatz (die Bandenden sind
    # nicht phasen-aligned) + Klick an der Klebestelle.
    x[SR * 7 :] = 0.2 * np.sin(2 * np.pi * 220.0 * t[SR * 7 :] + 0.8) * 0.5
    x[SR * 7 : SR * 7 + 3] += 0.35  # Bandstoss-Klick: Sub-ms-Impuls (breitbandig)
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
    rng = np.random.default_rng(7)
    # Musik-artiger Inhalt statt Rein-Ton: der Tonal-Guard des Detektors
    # nullt Reintoene korrekt (Sinus-Steigung mimt LSB-Stufen); Quantisierung
    # auf realistischem Inhalt ist der nachweisbare Fall.
    x = 0.12 * np.sin(2 * np.pi * 220.0 * t)
    x = x + 0.09 * np.sin(2 * np.pi * 587.0 * t)
    x = x + 0.07 * np.sin(2 * np.pi * 1318.0 * t)
    x = x + 0.02 * rng.standard_normal(len(t))
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


def _stereo_crosstalk() -> np.ndarray:
    t = np.arange(SR * DUR) / SR
    left = 0.2 * np.sin(2 * np.pi * 440.0 * t)
    # Uebersprechen L->R: 440-Hz-Geistbild bei nur ~10 dB Kanaltrennung
    right = 0.2 * np.sin(2 * np.pi * 630.0 * t) + 0.06 * np.sin(2 * np.pi * 440.0 * t)
    _out: np.ndarray = np.stack([left, right], axis=1).astype(np.float32)
    return _out


def _stereo_imbalance() -> np.ndarray:
    t = np.arange(SR * DUR) / SR
    left = 0.2 * np.sin(2 * np.pi * 440.0 * t)
    right = 0.14 * np.sin(2 * np.pi * 440.0 * t)  # R = -3 dB
    _out: np.ndarray = np.stack([left, right], axis=1).astype(np.float32)
    return _out


def _stereo_collapse() -> np.ndarray:
    t = np.arange(SR * DUR) / SR
    x = 0.2 * np.sin(2 * np.pi * 440.0 * t)
    _out: np.ndarray = np.stack([x, x], axis=1).astype(np.float32)  # L == R: Feld kollabiert
    return _out


def _stereo_decorrelated() -> np.ndarray:
    t = np.arange(SR * DUR) / SR
    left = 0.2 * np.sin(2 * np.pi * 440.0 * t)
    right = 0.2 * np.sin(2 * np.pi * 630.0 * t)  # voll dekorreliert
    _out: np.ndarray = np.stack([left, right], axis=1).astype(np.float32)
    return _out


def _dyn_clipping() -> np.ndarray:
    t = np.arange(SR * DUR) / SR
    x = 1.4 * np.sin(2 * np.pi * 220.0 * t)
    x = np.clip(x, -1.0, 1.0)  # Full-Scale-Hard-Clipping: Flat-Tops, ungerade Harmonische
    _out: np.ndarray = np.stack([x, x], axis=1).astype(np.float32)
    return _out


def _dyn_saturation() -> np.ndarray:
    t = np.arange(SR * DUR) / SR
    x = 0.2 * np.sin(2 * np.pi * 220.0 * t)
    x = x + 0.3 * x**2  # quadratische Kennlinie: gerade Harmonische (Tube/Tape)
    _out: np.ndarray = np.stack([x, x], axis=1).astype(np.float32)
    return _out


def _dyn_compression_artifacts() -> np.ndarray:
    t = np.arange(SR * DUR) / SR
    rng = np.random.default_rng(17)
    # Codec-Kompressions-Artefakte: HF-Kappung (~15 kHz) + Spektralloch +
    # gleichfoermig niedrige spektrale Flachheit auf breitbandigem Inhalt
    x = 0.2 * rng.standard_normal(len(t))  # rauschdominiert (breitbandig)
    x = x + 0.1 * np.sin(2 * np.pi * 440.0 * t)
    sos_lp = signal.butter(6, 14500, btype="lowpass", fs=SR, output="sos")
    x = signal.sosfiltfilt(sos_lp, x)
    sos_notch = signal.butter(4, (3000, 3800), btype="bandstop", fs=SR, output="sos")
    x = signal.sosfiltfilt(sos_notch, x)  # Spektralloch (Codec-Signatur)
    x = x / (np.max(np.abs(x)) + 1e-12) * 0.2
    _out: np.ndarray = np.stack([x, x], axis=1).astype(np.float32)
    return _out


def _dyn_compression_excess() -> np.ndarray:
    t = np.arange(SR * DUR) / SR
    x = 0.2 * np.sin(2 * np.pi * 220.0 * t)
    # Totkomprimiert: sanfte Huelle 0,9..1,0 statt musikalischer Dynamik
    env = 0.95 + 0.05 * np.sin(2 * np.pi * 0.3 * t)
    _out: np.ndarray = np.stack([x * env, x * env], axis=1).astype(np.float32)
    return _out


def _dyn_sibilance() -> np.ndarray:
    t = np.arange(SR * DUR) / SR
    rng = np.random.default_rng(11)
    x = 0.15 * np.sin(2 * np.pi * 220.0 * t)
    # „S“-Bursts: 100 ms gefiltertes HF-Rauschen (5-10 kHz) jede Sekunde
    noise = rng.standard_normal(len(t))
    sos = signal.butter(4, (5000, 10000), btype="bandpass", fs=SR, output="sos")
    hf = signal.sosfiltfilt(sos, noise)
    hf = hf / (np.max(np.abs(hf)) + 1e-12)
    burst_mask = ((t % 1.0) < 0.1).astype(float)
    x = x + 0.6 * hf * burst_mask  # harsche Sibilanz dominiert die Mischung
    _out: np.ndarray = np.stack([x, x], axis=1).astype(np.float32)
    return _out


def _tape_bias_error() -> np.ndarray:
    t = np.arange(SR * DUR) / SR
    rng = np.random.default_rng(23)
    # Pathologischer Over-Bias: Empfindlichkeitsabbruch bereits ab ~5,5 kHz
    # (Bias-Mismatch verschiebt die Eckfrequenz nach unten) -> Baender
    # 5-8/8-14 kHz praktisch leer, 2-5 kHz normal. Ein Abbruch exakt an der
    # 8-kHz-Bandgrenze bleibt wegen Welch-Fenster-Leckage unter der
    # Detektorschwelle (-16 dB/Okt) - das ist die ehrliche Messgrenze.
    x = 0.15 * rng.standard_normal(len(t))
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1.0 / SR)
    spec[freqs > 5500.0] *= 0.001  # -60 dB oberhalb des Abbruchs
    x = np.fft.irfft(spec, len(x))
    _out: np.ndarray = np.stack([x, x], axis=1).astype(np.float32)
    return _out


def _tape_print_through() -> np.ndarray:
    t = np.arange(SR * DUR) / SR
    # Ruhige Passage vor dem Einsatz: genau dann wird Print-Through hoerbar
    x = 0.01 * np.sin(2 * np.pi * 440.0 * t)  # sehr leise Grundlage
    burst = np.zeros(SR * DUR)
    burst[SR * 10 : SR * 11] = 0.5 * np.sin(2 * np.pi * 880.0 * t[SR * 10 : SR * 11])
    # Print-Through: Geist-Echo 200 ms VOR dem Onset bei -20 dB (IEC 60094-3)
    ghost = np.zeros(SR * DUR)
    ghost_start = SR * 10 - int(0.2 * SR)
    ghost[ghost_start : ghost_start + SR] = 0.1 * burst[SR * 10 : SR * 11]
    sig = x + burst + ghost
    _out: np.ndarray = np.stack([sig, sig], axis=1).astype(np.float32)
    return _out


def _tape_azimuth() -> np.ndarray:
    t = np.arange(SR * DUR) / SR
    rng = np.random.default_rng(29)
    # Azimuth-Fehler: Spalt-Verkantung = Zeitversatz zwischen L und R
    # -> Phasendifferenz waechst LINEAR mit der Frequenz (PHD-Slope, IEC 60386)
    left = 0.15 * rng.standard_normal(len(t))
    left = left + 0.2 * np.sin(2 * np.pi * 440.0 * t)
    left = left + 0.08 * np.sin(2 * np.pi * 8000.0 * t)
    delay = 10  # 0,2 ms Zeitversatz -> ~72 Grad/kHz Phasenslope
    right = np.concatenate((np.zeros(delay), left[:-delay]))
    _out: np.ndarray = np.stack([left, right], axis=1).astype(np.float32)
    return _out


def _tape_hf_remanence() -> np.ndarray:
    t = np.arange(SR * DUR) / SR
    x = 0.15 * np.sin(2 * np.pi * 440.0 * t)
    # HF-Remanenzverlust: HF-Anteil nimmt mit der Zeit ab (Bandalterung)
    env = np.linspace(1.0, 0.05, SR * DUR)
    x = x + env * 0.1 * np.sin(2 * np.pi * 10000.0 * t)
    _out: np.ndarray = np.stack([x, x], axis=1).astype(np.float32)
    return _out


def _tape_modulation_noise() -> np.ndarray:
    t = np.arange(SR * DUR) / SR
    rng = np.random.default_rng(19)
    x = 0.2 * np.sin(2 * np.pi * 220.0 * t)
    # Modulationsrauschen: Rauschpegel folgt der Signalhuellkurve
    noise = rng.standard_normal(len(t))
    x = x + 0.04 * noise * (1.0 + 2.0 * np.abs(np.sin(2 * np.pi * 220.0 * t)))
    _out: np.ndarray = np.stack([x, x], axis=1).astype(np.float32)
    return _out


def _tape_bandwidth_loss() -> np.ndarray:
    t = np.arange(SR * DUR) / SR
    x = 0.2 * np.sin(2 * np.pi * 440.0 * t) + 0.1 * np.sin(2 * np.pi * 8000.0 * t)
    sos = signal.butter(4, 6000, btype="lowpass", fs=SR, output="sos")
    x = signal.sosfiltfilt(sos, x)  # Bandbreitenverlust: alles ueber 6 kHz weg
    _out: np.ndarray = np.stack([x, x], axis=1).astype(np.float32)
    return _out


def _tape_dolby_mismatch() -> np.ndarray:
    t = np.arange(SR * DUR) / SR
    rng = np.random.default_rng(31)
    # Dolby-Encoder an, Decoder aus: HF-Shelf (+6 dB ueber ~1,5 kHz) auf
    # breitbandigem Inhalt -> E(2-16k)/E(300-2k) steigt deutlich
    x = 0.1 * rng.standard_normal(len(t))
    x = x + 0.15 * np.sin(2 * np.pi * 500.0 * t)
    x = x + 0.08 * np.sin(2 * np.pi * 4000.0 * t)
    sos = signal.butter(1, 1500, btype="high", fs=SR, output="sos")
    x = x + 1.0 * signal.sosfiltfilt(sos, x)  # +6 dB HF-Shelf
    _out: np.ndarray = np.stack([x, x], axis=1).astype(np.float32)
    return _out


def _tape_head_clog() -> np.ndarray:
    t = np.arange(SR * DUR) / SR
    rng = np.random.default_rng(37)
    # Kopfverschmutzung: periodisch wiederkehrende HF-Ausloeschungen
    x = 0.12 * rng.standard_normal(len(t))
    x = x + 0.15 * np.sin(2 * np.pi * 440.0 * t)
    sos = signal.butter(4, 4500, btype="highpass", fs=SR, output="sos")
    hf_part = signal.sosfiltfilt(sos, x)
    dip = 1.0 - 0.9 * (((t % 2.0) < 0.3).astype(float))  # alle 2 s: 300 ms HF-Dip
    x = x - hf_part * (1.0 - dip)
    _out: np.ndarray = np.stack([x, x], axis=1).astype(np.float32)
    return _out


def _tape_head_wear() -> np.ndarray:
    t = np.arange(SR * DUR) / SR
    rng = np.random.default_rng(41)
    # Kopfverschleiss: glatter progressiver HF-Rolloff ab ~4 kHz (monoton),
    # Baender 8-16 kHz >30 dB unter Referenz (Pegel-Kriterium des Detektors)
    x = 0.15 * rng.standard_normal(len(t))
    x = x + 0.15 * np.sin(2 * np.pi * 500.0 * t)
    sos = signal.butter(3, 4200, btype="lowpass", fs=SR, output="sos")
    x = signal.sosfiltfilt(sos, x)
    _out: np.ndarray = np.stack([x, x], axis=1).astype(np.float32)
    return _out


def _tape_sticky_shed() -> np.ndarray:
    t = np.arange(SR * DUR) / SR
    rng = np.random.default_rng(43)
    # Sticky-Shed: kurze Pegel-Dips (10-100 ms) + Modulationsrauschen
    x = 0.2 * np.sin(2 * np.pi * 440.0 * t)
    noise = rng.standard_normal(len(t))
    x = x + 0.03 * noise * (1.0 + 2.0 * np.abs(np.sin(2 * np.pi * 440.0 * t)))
    dip = 1.0 - 0.8 * (((t % 1.0) < 0.05).astype(float))  # alle 1 s: 50-ms-Dip
    x = x * dip
    _out: np.ndarray = np.stack([x, x], axis=1).astype(np.float32)
    return _out


def _tape_head_level_dip() -> np.ndarray:
    t = np.arange(SR * DUR) / SR
    x = 0.2 * np.sin(2 * np.pi * 440.0 * t)
    # Kopfkontakt-Variation: 60 ms Ramp auf -15 dB, 200 ms halten, Snap-back
    phase = (t % 1.0) / 1.0  # 1 Ereignis pro Sekunde
    dip = np.ones_like(t)
    ramp_mask = (phase >= 0.0) & (phase < 0.06)
    hold_mask = (phase >= 0.06) & (phase < 0.26)
    dip[ramp_mask] = 1.0 - 0.82 * (phase[ramp_mask] / 0.06)
    dip[hold_mask] = 0.18  # -15 dB
    x = x * dip
    _out: np.ndarray = np.stack([x, x], axis=1).astype(np.float32)
    return _out


def _tape_transport_bump() -> np.ndarray:
    t = np.arange(SR * DUR) / SR
    x = 0.2 * np.sin(2 * np.pi * 440.0 * t)
    # Transport-Stoss: Kopf hebt kurz ab (Pflicht-Feature Pegel-Drop) +
    # gleichzeitiger LF-Thump (60 Hz) + Fluss-/Zentroid-Disruption
    phase = (t % 1.5) / 1.5
    drop_mask = (phase >= 0.0) & (phase < 0.03)  # 45 ms Drop
    dip = np.ones_like(t)
    dip[drop_mask] = 0.25  # -12 dB
    x = x * dip
    thump = 0.05 * np.sin(2 * np.pi * 60.0 * t) * drop_mask.astype(float)  # leiser mechanischer Stoss
    x = x + thump
    _out: np.ndarray = np.stack([x, x], axis=1).astype(np.float32)
    return _out


def _tape_generation_loss() -> np.ndarray:
    t = np.arange(SR * DUR) / SR
    rng = np.random.default_rng(47)
    # Mehrfach-Ueberspielung: Rauschen + starke Bandbreitenverengung +
    # Phasen-Randomisierung in den HF-Baendern
    x = 0.12 * rng.standard_normal(len(t))
    x = x + 0.15 * np.sin(2 * np.pi * 440.0 * t)
    sos = signal.butter(3, 7000, btype="lowpass", fs=SR, output="sos")
    x = signal.sosfiltfilt(sos, x)
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1.0 / SR)
    phase = np.exp(2j * np.pi * rng.standard_normal(len(spec)))  # HF-Phase jittern
    blend = np.clip((freqs - 4000.0) / 8000.0, 0.0, 0.8)
    spec = spec * (1.0 - blend + blend * phase)
    x = np.fft.irfft(spec, len(x))
    _out: np.ndarray = np.stack([x, x], axis=1).astype(np.float32)
    return _out


def _tape_nr_breathing() -> np.ndarray:
    t = np.arange(SR * DUR) / SR
    rng = np.random.default_rng(53)
    # NR-Atmung: HF-Rauschboden ist ANTI-korreliert zum Signalpegel
    # (leise Passage -> Rauschen laut; laute Passage -> Rauschen leise)
    signal_env = 0.5 * (1.0 + np.sign(np.sin(2 * np.pi * 0.5 * t)))  # 0/1 alle 2 s
    x = 0.2 * np.sin(2 * np.pi * 440.0 * t) * signal_env
    sos = signal.butter(4, 4000, btype="highpass", fs=SR, output="sos")
    noise = signal.sosfiltfilt(sos, rng.standard_normal(len(t)))
    noise = noise / (np.max(np.abs(noise)) + 1e-12)
    x = x + 0.08 * noise * (1.0 - signal_env)  # Rauschen nur in der Stille
    _out: np.ndarray = np.stack([x, x], axis=1).astype(np.float32)
    return _out


def _tape_pre_echo() -> np.ndarray:
    t = np.arange(SR * DUR) / SR
    # Pre-Echo: Geist 150 ms VOR dem Transienten, spektral identisch
    x = 0.01 * np.sin(2 * np.pi * 440.0 * t)
    burst = np.zeros(SR * DUR)
    burst[SR * 10 : SR * 11] = 0.5 * np.sin(2 * np.pi * 880.0 * t[SR * 10 : SR * 11])
    ghost = np.zeros(SR * DUR)
    ghost_start = SR * 10 - int(0.15 * SR)
    ghost[ghost_start : ghost_start + SR] = 0.08 * burst[SR * 10 : SR * 11]
    sig = x + burst + ghost
    _out: np.ndarray = np.stack([sig, sig], axis=1).astype(np.float32)
    return _out


def _tape_dropout_oxide() -> np.ndarray:
    t = np.arange(SR * DUR) / SR
    x = 0.2 * np.sin(2 * np.pi * 440.0 * t)
    # Oxid-Abrieb: langsamer Pegel-Einbruch mit sanfter Erholung
    drop = np.ones_like(t)
    seg = int(0.5 * SR)
    start = 7 * SR
    drop[start : start + seg] = np.linspace(1.0, 0.1, seg)
    drop[start + seg : start + 2 * seg] = 0.1
    drop[start + 2 * seg : start + 3 * seg] = np.linspace(0.1, 1.0, seg)
    x = x * drop
    _out: np.ndarray = np.stack([x, x], axis=1).astype(np.float32)
    return _out


def _tape_dropout_head_contact() -> np.ndarray:
    t = np.arange(SR * DUR) / SR
    x = 0.2 * np.sin(2 * np.pi * 440.0 * t)
    # Kopfkontakt-Verlust: Drops von ~40 ms auf -15 dB alle 300 ms
    # (<=20 ms faellt in den Oxide-Fallback des Subtyp-Klassifizierers)
    dip = 1.0 - 0.82 * (((t % 0.3) < 0.04).astype(float))
    x = x * dip
    _out: np.ndarray = np.stack([x, x], axis=1).astype(np.float32)
    return _out


def _wow_scrape_flutter() -> np.ndarray:
    t = np.arange(SR * DUR) / SR
    # Tape-Scrape: FM-Seitenbaender im 40-120-Hz-Bereich um den Traeger.
    # 85 Hz statt 80 Hz: das Detektor-Raster prueft [40, 55, 70, 85, 100, 120].
    mod = np.sin(2 * np.pi * 85.0 * t)
    x = 0.2 * np.sin(2 * np.pi * 2000.0 * t + 0.8 * mod)
    _out: np.ndarray = np.stack([x, x], axis=1).astype(np.float32)
    return _out


def _scrape_am_iec() -> np.ndarray:
    """§D-K3-28 (2026-10-07): IEC-60386-Scrape — 0,7 % AM @ 60 Hz auf einem
    breitbandigen Träger (Grundton + Harmonische + dezentes Rauschen). Die vier
    Bestands-Kanäle waren dafür blind (Messung: AM 20–150 Hz @ 0,7 % → 0.000 in
    allen); der kohärente Mehrband-AM-Kanal erkennt Rate/Tiefe kohärenz-gekoppelt
    (0,69–0,75 @ 0,7 %), saubere Signale liefern 0,0 (Anti-FP)."""
    t = np.arange(SR * DUR) / SR
    rng = np.random.default_rng(42)
    carrier = (
        0.20 * np.sin(2 * np.pi * 440.0 * t)
        + 0.10 * np.sin(2 * np.pi * 880.0 * t)
        + 0.05 * np.sin(2 * np.pi * 1760.0 * t)
        + 0.02 * rng.standard_normal(SR * DUR)
    )
    x = carrier * (1.0 + 0.007 * np.sin(2 * np.pi * 60.0 * t))
    _out: np.ndarray = np.stack([x, x], axis=1).astype(np.float32)
    return _out


def _wow_multiband() -> np.ndarray:
    t = np.arange(SR * DUR) / SR
    # Band-abhaengiges Flutter: nur der 1,5-kHz-Ton flattert (0,4 Hz, ±0,5 %).
    # Der Detektor untersucht Oktavbaender um 250/500/1000/2000 Hz und braucht
    # >= 3 aktive Baender - deshalb liegen alle Traeger unterhalb 2,8 kHz.
    low = 0.2 * np.sin(2 * np.pi * 300.0 * t)
    mid = 0.15 * np.sin(2 * np.pi * 750.0 * t)
    mod = np.sin(2 * np.pi * 0.4 * t)
    high = 0.12 * np.sin(2 * np.pi * 1500.0 * t + 0.5 * mod)
    _out: np.ndarray = np.stack([low + mid + high, low + mid + high], axis=1).astype(np.float32)
    return _out


def _wow_sidebands() -> np.ndarray:
    t = np.arange(SR * DUR) / SR
    # Flutter-Seitenbaender: tonale ±1-8-Hz-Seitenbaender um einen Traeger
    mod = np.sin(2 * np.pi * 3.0 * t)
    x = 0.2 * np.sin(2 * np.pi * 1000.0 * t + 0.6 * mod)
    _out: np.ndarray = np.stack([x, x], axis=1).astype(np.float32)
    return _out


def _env_reverb() -> np.ndarray:
    t = np.arange(SR * DUR) / SR
    rng = np.random.default_rng(59)
    # Uebermaessiger Nachhall: JEDER Impuls bekommt seinen eigenen exponentiellen
    # Nachhall-Schwanz (tau ~0,36 s -> RT60 ~2,5 s). Ein globaler Decay ist
    # nicht an die Onsets gekoppelt und wird nicht als Nachhall gemessen.
    tail = signal.sosfiltfilt(
        signal.butter(2, 4000, btype="lowpass", fs=SR, output="sos"), rng.standard_normal(SR * DUR)
    )
    tail = tail / (np.max(np.abs(tail)) + 1e-12)
    x = np.zeros(SR * DUR)
    for k in range(1, 6):
        idx = k * 2 * SR
        seg = np.arange(SR * DUR - idx)
        x[idx:] = x[idx:] + 0.3 * tail[idx:] * np.exp(-seg / (SR * 0.36))
        x[idx : idx + 200] += 0.5 * np.sin(2 * np.pi * 800.0 * t[idx : idx + 200])
    _out: np.ndarray = np.stack([x, x], axis=1).astype(np.float32)
    return _out


def _env_room_mode() -> np.ndarray:
    t = np.arange(SR * DUR) / SR
    rng = np.random.default_rng(61)
    # Stehende Welle: schmalbandiger Resonanz-Peak bei 73 Hz (Q~15),
    # bewusst KEINE 50/60-Hz-Harmonische (Brumm-Ausschluss des Detektors)
    x = 0.08 * rng.standard_normal(len(t))
    sos = signal.butter(2, (70, 76), btype="bandpass", fs=SR, output="sos")
    x = x + 0.4 * signal.sosfiltfilt(sos, rng.standard_normal(len(t)))
    _out: np.ndarray = np.stack([x, x], axis=1).astype(np.float32)
    return _out


def _env_proximity() -> np.ndarray:
    t = np.arange(SR * DUR) / SR
    # Nahbesprechung: LF-Band (80-250 Hz) stark angehoben, kein Sub-Bass
    # (Anti-Rumble-Guard), Mitten normal
    x = 0.25 * np.sin(2 * np.pi * 150.0 * t)
    x = x + 0.12 * np.sin(2 * np.pi * 700.0 * t)
    x = x + 0.08 * np.sin(2 * np.pi * 1500.0 * t)
    _out: np.ndarray = np.stack([x, x], axis=1).astype(np.float32)
    return _out


def _dig_jitter() -> np.ndarray:
    t = np.arange(SR * DUR) / SR
    # Clock-Jitter: periodisches Sample-Timing-Jitter -> FM-Seitenbaender + ZC-Irregularitaet
    jitter = 0.5 * np.sin(2 * np.pi * 50.0 * t)
    x = 0.2 * np.sin(2 * np.pi * 1000.0 * (t + jitter / SR))
    _out: np.ndarray = np.stack([x, x], axis=1).astype(np.float32)
    return _out


def _dig_digital_artifacts() -> np.ndarray:
    t = np.arange(SR * DUR) / SR
    rng = np.random.default_rng(67)
    # Sammel-Signatur: Near-Nyquist-Energie + 8-Bit-Quantisierung
    x = 0.1 * rng.standard_normal(len(t))
    x = x + 0.12 * np.sin(2 * np.pi * 21000.0 * t)
    x = np.round(x * 127.0) / 127.0
    _out: np.ndarray = np.stack([x, x], axis=1).astype(np.float32)
    return _out


def _dig_mpeg_frame_loss() -> np.ndarray:
    t = np.arange(SR * DUR) / SR
    x = 0.2 * np.sin(2 * np.pi * 440.0 * t)
    # MPEG-Frame-Verlust: 26-ms-Blöcke (1152 Samples) periodisch genullt
    frame = 1152
    for k in range(1, 12):
        idx = k * SR
        x[idx : idx + frame] = 0.0
    _out: np.ndarray = np.stack([x, x], axis=1).astype(np.float32)
    return _out


def _dig_amplitude_drift() -> np.ndarray:
    n = SR * 45  # Detektor verlangt >= 30 s fuer Trend-Erkennung
    t = np.arange(n) / SR
    # Traeger-Drift: Pegel steigt linear, Aktivitaet konstant (kein Crescendo)
    env = np.linspace(0.5, 1.5, n)
    x = 0.2 * np.sin(2 * np.pi * 440.0 * t) * env
    _out: np.ndarray = np.stack([x, x], axis=1).astype(np.float32)
    return _out


def _dig_pitch_drift() -> np.ndarray:
    t = np.arange(SR * DUR) / SR
    # Pitch-Drift: lineare Frequenzdrift 440 -> 462 Hz ueber 15 s (W&F).
    # 32 cents (440->450) fielen noch in den Glissando-Penalty (Segment-Jumps
    # > 50 % des Drifts); 85 cents liegen klar ueber der Detektorskala.
    phase = 2 * np.pi * (440.0 * t + 0.5 * (22.0 / DUR) * t**2)
    x = 0.2 * np.sin(phase)
    _out: np.ndarray = np.stack([x, x], axis=1).astype(np.float32)
    return _out


def _dig_transient_smearing() -> np.ndarray:
    t = np.arange(SR * DUR) / SR
    # Verschmierte Transienten: Huelle steigt linear ueber ~15 ms an
    # (Detektor: mediane Anstiegszeit > 10 ms = verschmiert)
    x = np.zeros(SR * DUR)
    for k in range(1, 28):
        idx = int(k * 0.5 * SR)
        ramp_len = int(0.015 * SR)
        ramp = np.linspace(0.0, 1.0, ramp_len)
        x[idx : idx + ramp_len] = 0.5 * np.sin(2 * np.pi * 3000.0 * t[idx : idx + ramp_len]) * ramp
        x[idx + ramp_len : idx + 240] = 0.5 * np.sin(2 * np.pi * 3000.0 * t[idx + ramp_len : idx + 240])
    _out: np.ndarray = np.stack([x, x], axis=1).astype(np.float32)
    return _out


def _dig_vocal_harshness() -> np.ndarray:
    t = np.arange(SR * DUR) / SR
    rng = np.random.default_rng(71)
    # Vokal-aehnlich mit harter 2-6-kHz-Presence: Multiton + HF-Band + Soft-Clip
    x = 0.2 * np.sin(2 * np.pi * 300.0 * t)
    x = x + 0.15 * np.sin(2 * np.pi * 900.0 * t)
    x = x + 0.12 * np.sin(2 * np.pi * 2500.0 * t)
    sos = signal.butter(2, (2000, 6000), btype="bandpass", fs=SR, output="sos")
    hf = signal.sosfiltfilt(sos, rng.standard_normal(len(t)))
    hf = hf / (np.max(np.abs(hf)) + 1e-12)
    x = x + 0.15 * hf
    x = 0.35 * np.tanh(2.0 * x)  # Odd-Harmonics-Dominanz + niedriger Crest
    _out: np.ndarray = np.stack([x, x], axis=1).astype(np.float32)
    return _out


def _dig_overload() -> np.ndarray:
    t = np.arange(SR * DUR) / SR
    # Analoge Eingangs-Üebersteuerung: weiche Saettigung, THD im Prozentbereich
    x = 0.4 * np.tanh(2.5 * np.sin(2 * np.pi * 440.0 * t))
    _out: np.ndarray = np.stack([x, x], axis=1).astype(np.float32)
    return _out


def _fw_distortion() -> np.ndarray:
    t = np.arange(SR * DUR) / SR
    # THD-Test fuer den UnifiedDefectDetector (ai_framework): tanh-Kennlinie
    # erzeugt Harmonische 2-5 um den 440-Hz-Grundton
    x = 0.4 * np.tanh(2.5 * np.sin(2 * np.pi * 440.0 * t))
    _out: np.ndarray = np.stack([x, x], axis=1).astype(np.float32)
    return _out


def _fw_dropout() -> np.ndarray:
    t = np.arange(SR * DUR) / SR
    # Huellkurven-Drops fuer den UnifiedDefectDetector: drei 300-ms-Abrisse
    x = 0.2 * np.sin(2 * np.pi * 440.0 * t)
    for k in (3, 7, 11):
        x[k * SR : k * SR + int(0.3 * SR)] = 0.02 * np.sin(2 * np.pi * 440.0 * t[k * SR : k * SR + int(0.3 * SR)])
    _out: np.ndarray = np.stack([x, x], axis=1).astype(np.float32)
    return _out


def _dropout_splice_deep() -> np.ndarray:
    t = np.arange(SR * DUR) / SR
    # Scanner-Subtyp DROPOUT_SPLICE: abrupte Abrisse mit >95 % Pegelverlust
    # (_classify_dropout_subtype: loss_ratio > 0,95)
    x = 0.2 * np.sin(2 * np.pi * 440.0 * t)
    dip = np.ones_like(t)
    for k in range(1, 30):
        idx = int(k * 0.5 * SR)
        dip[idx : idx + int(0.010 * SR)] = 0.03  # -30 dB
    x = x * dip
    _out: np.ndarray = np.stack([x, x], axis=1).astype(np.float32)
    return _out


def _noise_clicks() -> np.ndarray:
    t = np.arange(SR * DUR) / SR
    rng = np.random.default_rng(73)
    x = 0.15 * np.sin(2 * np.pi * 440.0 * t)
    # Irregulaere Click-Abstaende (Periodicity-Guard) UND saliente Impulse:
    # der Scan wendet eine Perceptual-Salience-Kalibrierung an - Clicks nur
    # ~7 dB ueber dem Traeger wurden als maskiert eingestuft (26/26).
    idx = 8000
    while idx < SR * DUR - 200:
        x[idx : idx + 3] += 0.8
        idx += int(0.3 * SR + 0.7 * SR * rng.random())
    _out: np.ndarray = np.stack([x, x], axis=1).astype(np.float32)
    return _out


def _noise_crackle() -> np.ndarray:
    t = np.arange(SR * DUR) / SR
    rng = np.random.default_rng(79)
    x = 0.15 * np.sin(2 * np.pi * 440.0 * t)
    # Knistern in CLUSTERN mit Luecken: zu dichte Impulse werden vom
    # tonal_or_dense_hf_guard als kontinuierliches HF genullt (gemessen:
    # crackle_percentage 99,8 %, Guard aktiv). Echte Knistern-Huelle pulsiert.
    idx = 8000
    while idx < SR * DUR - 2000:
        n_pops = int(2 + 4 * rng.random())
        for _ in range(n_pops):
            x[idx : idx + 2] += (0.05 + 0.12 * rng.random()) * rng.choice((-1.0, 1.0))
            idx += int(2 + 8 * rng.random())
        idx += int(0.05 * SR + 0.15 * SR * rng.random())
    _out: np.ndarray = np.stack([x, x], axis=1).astype(np.float32)
    return _out


def _noise_hum() -> np.ndarray:
    t = np.arange(SR * DUR) / SR
    # Netzfrequenz-Brumm: 50 Hz + Harmonische
    x = 0.12 * np.sin(2 * np.pi * 50.0 * t)
    x = x + 0.06 * np.sin(2 * np.pi * 100.0 * t)
    x = x + 0.04 * np.sin(2 * np.pi * 150.0 * t)
    x = x + 0.08 * np.sin(2 * np.pi * 440.0 * t)
    _out: np.ndarray = np.stack([x, x], axis=1).astype(np.float32)
    return _out


def _noise_rumble() -> np.ndarray:
    t = np.arange(SR * DUR) / SR
    # Sub-Bass-Rumpeln < 80 Hz (Plattenspieler/Gebaeude)
    x = 0.15 * np.sin(2 * np.pi * 30.0 * t) + 0.1 * np.sin(2 * np.pi * 55.0 * t)
    x = x + 0.08 * np.sin(2 * np.pi * 440.0 * t)
    _out: np.ndarray = np.stack([x, x], axis=1).astype(np.float32)
    return _out


def _vinyl_lacquer() -> np.ndarray:
    t = np.arange(SR * DUR) / SR
    rng = np.random.default_rng(83)
    # Acetat-Zersetzung (Hess 1988): dichte Substrat-Riss-Clicks +
    # deutlicher HF-Verlust (Rolloff ab ~6 kHz -> hf_loss > 15 dB) +
    # stark nicht-stationaeres Substrat-Rauschen (170-ms-Welch-Skala).
    x = 0.15 * np.sin(2 * np.pi * 400.0 * t)
    base_noise = rng.standard_normal(len(t))
    for seg in range(0, SR * DUR, int(0.3 * SR)):
        level = 0.02 + 0.16 * rng.random()
        x[seg : seg + int(0.3 * SR)] += level * base_noise[seg : seg + int(0.3 * SR)]
    sos = signal.butter(4, 6000, btype="lowpass", fs=SR, output="sos")
    x = signal.sosfiltfilt(sos, x)
    for k in range(1, 56):
        idx = int(k * 0.25 * SR)
        x[idx : idx + 3] += 1.0
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
        (
            # Scan-Skalierung: severity = click_rate/45 (1,73 Clicks/s -> 0,038).
            # Die Aktivierungs-Schwelle gehoert in die Pipeline (Praezedenzfall
            # Wow); der Harness prueft hier die DETEKTION (Score > 0).
            "Isolierte Impulse werden als Clicks erkannt",
            _noise_clicks,
            MaterialType.VINYL,
            {DefectType.CLICKS: (0.02, 1.0)},
        ),
        (
            "Sauberer Ton loest keine Clicks aus",
            _clean_tone,
            MaterialType.VINYL,
            {DefectType.CLICKS: (0.0, 0.02)},
        ),
        (
            "Dichtes Knistern (Mikro-Impulse) wird als Crackle erkannt",
            _noise_crackle,
            MaterialType.VINYL,
            {DefectType.CRACKLE: (0.3, 1.0)},
        ),
        (
            "Netzfrequenz-Brumm (50 Hz + Harmonische) wird erkannt",
            _noise_hum,
            MaterialType.TAPE,
            {DefectType.HUM: (0.3, 1.0)},
        ),
        (
            "Sub-Bass-Rumpeln (< 80 Hz) wird erkannt",
            _noise_rumble,
            MaterialType.TAPE,
            {DefectType.LOW_FREQ_RUMBLE: (0.3, 1.0)},
        ),
    ],
    "wow_flutter": [
        (
            "Wow (0,3 Hz FM ±0,5 %) wird erkannt",
            _wow,
            MaterialType.TAPE,
            # Nacht-Befund (defect_scanner._detect_wow, 2026-09-24): Reines
            # FM-Wow liefert Detektor-seitig ehrlich 0,272; die Aktivierungs-
            # Schwelle gehoert in die Pipeline (Spec 03), nicht in den Detektor.
            # Untergrenze 0,25 haelt die Trennung zum sauberen Fall (0,037 < 0,20).
            {DefectType.WOW: (0.25, 1.0)},
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
        (
            "Tape-Scrape-Flutter (FM-Seitenbaender ~80 Hz) wird erkannt",
            _wow_scrape_flutter,
            MaterialType.TAPE,
            {DefectType.SCRAPE_FLUTTER: (0.3, 1.0)},
        ),
        (
            "IEC-60386-Scrape (0,7 % AM @ 60 Hz) wird erkannt",
            _scrape_am_iec,
            MaterialType.TAPE,
            {DefectType.FLUTTER: (0.3, 1.0)},
        ),
        (
            "Band-abhaengiges Flutter (nur hohes Band) wird erkannt",
            _wow_multiband,
            MaterialType.TAPE,
            {DefectType.MULTIBAND_WOW_FLUTTER: (0.3, 1.0)},
        ),
        (
            "Flutter-Seitenbaender (±3-Hz-Seitenbaender um Traeger) werden erkannt",
            _wow_sidebands,
            MaterialType.TAPE,
            {DefectType.FLUTTER_SPECTRAL_SIDEBANDS: (0.3, 1.0)},
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
            "Tiefer kurzer Abriss (>95 %) wird als DROPOUT_SPLICE klassifiziert",
            _dropout_splice_deep,
            MaterialType.TAPE,
            {DefectType.DROPOUT_SPLICE: (0.3, 1.0)},
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
        (
            "Acetat-Zersetzung (Clicks + HF-Verlust + Substrat-Rauschen) wird erkannt",
            _vinyl_lacquer,
            MaterialType.LACQUER_DISC,
            {DefectType.LACQUER_DISC_DEGRADATION: (0.3, 1.0)},
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
    "stereo": [
        (
            "Uebersprechen L->R (Geistbild bei ~10 dB Trennung) wird erkannt",
            _stereo_crosstalk,
            MaterialType.TAPE,
            {DefectType.CROSSTALK: (0.3, 1.0)},
        ),
        (
            "Stereo-Imbalance (-3 dB rechts) wird erkannt",
            _stereo_imbalance,
            MaterialType.TAPE,
            {DefectType.STEREO_IMBALANCE: (0.3, 1.0)},
        ),
        (
            "Stereo-Feld-Kollaps (L==R) wird erkannt",
            _stereo_collapse,
            MaterialType.TAPE,
            {DefectType.STEREO_FIELD_COLLAPSE: (0.3, 1.0)},
        ),
        (
            "Dekorrelierte Kanaele loesen keinen Kollaps aus",
            _stereo_decorrelated,
            MaterialType.TAPE,
            {DefectType.STEREO_FIELD_COLLAPSE: (0.0, 0.2)},
        ),
    ],
    "dynamics": [
        (
            "Hard-Clipping (Flat-Tops) wird erkannt",
            _dyn_clipping,
            MaterialType.TAPE,
            {DefectType.CLIPPING: (0.3, 1.0)},
        ),
        (
            "Soft-Saturation (gerade Harmonische) wird erkannt",
            _dyn_saturation,
            MaterialType.TAPE,
            {DefectType.SOFT_SATURATION: (0.3, 1.0)},
        ),
        (
            "Codec-Kompressions-Artefakte (HF-Kappung + Spektralloch) werden erkannt",
            _dyn_compression_artifacts,
            MaterialType.TAPE,
            {DefectType.COMPRESSION_ARTIFACTS: (0.3, 1.0)},
        ),
        (
            "Ueberkompression (kaum Dynamik) wird erkannt",
            _dyn_compression_excess,
            MaterialType.TAPE,
            {DefectType.DYNAMIC_COMPRESSION_EXCESS: (0.3, 1.0)},
        ),
        (
            "Exzessive Sibilanz (HF-Bursts) wird erkannt",
            _dyn_sibilance,
            MaterialType.TAPE,
            {DefectType.SIBILANCE: (0.3, 1.0)},
        ),
    ],
    "tape_media": [
        (
            "Bias-Fehler (Empfindlichkeitsabbruch ab ~5,5 kHz) wird erkannt",
            _tape_bias_error,
            MaterialType.TAPE,
            {DefectType.BIAS_ERROR: (0.3, 1.0)},
        ),
        (
            "Print-Through (Geist-Echo 200 ms vor dem Onset) wird erkannt",
            _tape_print_through,
            MaterialType.TAPE,
            {DefectType.PRINT_THROUGH: (0.3, 1.0)},
        ),
        (
            "Azimuth-Fehler (L/R-Zeitversatz = Phasenslope) wird erkannt",
            _tape_azimuth,
            MaterialType.TAPE,
            {DefectType.AZIMUTH_ERROR: (0.3, 1.0)},
        ),
        (
            "HF-Remanenzverlust (HF nimmt mit der Zeit ab) wird erkannt",
            _tape_hf_remanence,
            MaterialType.TAPE,
            {DefectType.HF_REMANENCE_LOSS: (0.3, 1.0)},
        ),
        (
            "Modulationsrauschen (Rauschen folgt Signalhuellkurve) wird erkannt",
            _tape_modulation_noise,
            MaterialType.TAPE,
            {DefectType.MODULATION_NOISE: (0.3, 1.0)},
        ),
        (
            "Bandbreitenverlust (Tiefpass 6 kHz) wird erkannt",
            _tape_bandwidth_loss,
            MaterialType.TAPE,
            {DefectType.BANDWIDTH_LOSS: (0.3, 1.0)},
        ),
        (
            "Dolby-NR-Mismatch (+6-dB-HF-Shelf) wird erkannt",
            _tape_dolby_mismatch,
            MaterialType.TAPE,
            {DefectType.DOLBY_NR_MISMATCH: (0.3, 1.0)},
        ),
        (
            "Kopfverschmutzung (periodische HF-Dips) wird erkannt",
            _tape_head_clog,
            MaterialType.TAPE,
            {DefectType.TAPE_HEAD_CLOG: (0.3, 1.0)},
        ),
        (
            "Kopfverschleiss (progressiver Rolloff ab 4 kHz) wird erkannt",
            _tape_head_wear,
            MaterialType.TAPE,
            {DefectType.HEAD_WEAR: (0.3, 1.0)},
        ),
        (
            "Sticky-Shed (kurze Pegel-Dips + Modulationsrauschen) wird erkannt",
            _tape_sticky_shed,
            MaterialType.TAPE,
            {DefectType.STICKY_SHED_RESIDUE: (0.3, 1.0)},
        ),
        (
            "Kopfkontakt-Pegeldip (Ramp -15 dB + Snap-back) wird erkannt",
            _tape_head_level_dip,
            MaterialType.TAPE,
            {DefectType.TAPE_HEAD_LEVEL_DIP: (0.3, 1.0)},
        ),
        (
            "Transport-Stoss (LF-Thump + RMS-Spike) wird erkannt",
            _tape_transport_bump,
            MaterialType.TAPE,
            {DefectType.TRANSPORT_BUMP: (0.3, 1.0)},
        ),
        (
            "Generationenverlust (Rauschen + Verengung + Phasen-Jitter) wird erkannt",
            _tape_generation_loss,
            MaterialType.TAPE,
            {DefectType.GENERATION_LOSS: (0.3, 1.0)},
        ),
        (
            "NR-Atmung (anti-korrelierter Rauschboden) wird erkannt",
            _tape_nr_breathing,
            MaterialType.TAPE,
            {DefectType.NR_BREATHING_ARTIFACT: (0.3, 1.0)},
        ),
        (
            "Pre-Echo (Geist 150 ms vor dem Transienten) wird erkannt",
            _tape_pre_echo,
            MaterialType.TAPE,
            {DefectType.PRE_ECHO: (0.3, 1.0)},
        ),
        (
            "Oxid-Abrieb (langsamer Pegel-Einbruch) wird erkannt",
            _tape_dropout_oxide,
            MaterialType.TAPE,
            {DefectType.DROPOUT_OXIDE: (0.3, 1.0)},
        ),
        (
            "Kopfkontakt-Verlust (schnelle kurze Drops) wird erkannt",
            _tape_dropout_head_contact,
            MaterialType.TAPE,
            {DefectType.DROPOUT_HEAD_CONTACT: (0.3, 1.0)},
        ),
    ],
    "environment": [
        (
            "Uebermaessiger Nachhall (RT60 ~2,5 s) wird erkannt",
            _env_reverb,
            MaterialType.TAPE,
            {DefectType.REVERB_EXCESS: (0.3, 1.0)},
        ),
        (
            "Stehende Welle (73-Hz-Resonanz, Q~15) wird erkannt",
            _env_room_mode,
            MaterialType.TAPE,
            {DefectType.ROOM_MODE_RESONANCE: (0.3, 1.0)},
        ),
        (
            "Nahbesprechungseffekt (LF-Ueberhoehung 80-250 Hz) wird erkannt",
            _env_proximity,
            MaterialType.TAPE,
            {DefectType.PROXIMITY_EFFECT_EXCESS: (0.3, 1.0)},
        ),
    ],
    "digital_other": [
        (
            "Clock-Jitter (Sample-Timing-Jitter) wird erkannt",
            _dig_jitter,
            MaterialType.CD_DIGITAL,
            {DefectType.JITTER_ARTIFACTS: (0.3, 1.0)},
        ),
        (
            "Digitale Sammel-Artefakte (Near-Nyquist + 8-Bit) werden erkannt",
            _dig_digital_artifacts,
            MaterialType.CD_DIGITAL,
            {DefectType.DIGITAL_ARTIFACTS: (0.3, 1.0)},
        ),
        (
            "MPEG-Frame-Verlust (26-ms-Blöcke) wird erkannt",
            _dig_mpeg_frame_loss,
            MaterialType.CD_DIGITAL,
            {DefectType.MPEG_FRAME_LOSS: (0.3, 1.0)},
        ),
        (
            "Amplituden-Drift (linearer Pegelanstieg) wird erkannt",
            _dig_amplitude_drift,
            MaterialType.TAPE,
            {DefectType.AMPLITUDE_DRIFT: (0.3, 1.0)},
        ),
        (
            "Pitch-Drift (440->450 Hz ueber 15 s) wird erkannt",
            _dig_pitch_drift,
            MaterialType.TAPE,
            {DefectType.PITCH_DRIFT: (0.3, 1.0)},
        ),
        (
            "Transienten-Verschmierung (Anstiegsflanken verrundet) wird erkannt",
            _dig_transient_smearing,
            MaterialType.TAPE,
            {DefectType.TRANSIENT_SMEARING: (0.3, 1.0)},
        ),
        (
            "Vokale Haerte (2-6-kHz-Presence + Odd-Harmonics) wird erkannt",
            _dig_vocal_harshness,
            MaterialType.TAPE,
            {DefectType.VOCAL_HARSHNESS: (0.3, 1.0)},
        ),
        (
            "Analoge Uebersteuerung (weiche Saettigung) wird erkannt",
            _dig_overload,
            MaterialType.TAPE,
            {DefectType.OVERLOAD_DISTORTION: (0.3, 1.0)},
        ),
    ],
    "framework": [
        (
            "Framework-DISTORTION (THD-basiert, UnifiedDefectDetector) wird erkannt",
            _fw_distortion,
            MaterialType.TAPE,
            {DefectType.DISTORTION: (0.3, 1.0)},
        ),
        (
            "Framework-DROPOUT (Envelope-Drops, UnifiedDefectDetector) wird erkannt",
            _fw_dropout,
            MaterialType.TAPE,
            {DefectType.DROPOUT: (0.3, 1.0)},
        ),
    ],
}


def _report(name: str, expects: dict[DefectType, tuple[float, float]], scores: dict) -> int:
    failed = 0
    line = []
    for dt, (lo, hi) in expects.items():
        score = scores.get(dt)
        v = float(score.severity) if score is not None else 0.0
        ok = lo <= v <= hi
        if not ok:
            failed += 1
        line.append(f"{dt.value}={v:.3f} (erwartet {lo:.2f}-{hi:.2f}) {'OK' if ok else 'FEHLT'}")
    print(f"[{'OK' if not failed else 'LUECKE'}] {name}: {'; '.join(line)}")
    return failed


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in CASES:
        print(f"Familien: {', '.join(sorted(CASES))}")
        return 2
    family = sys.argv[1]
    failed = 0
    if family == "framework":
        # Der UnifiedDefectDetector (ai_framework) befuellt DISTORTION/DROPOUT
        # (Singular) - eigener Detektionspfad neben dem DefectScanner.
        from backend.core.ai_framework import UnifiedDefectDetector

        _fwd = UnifiedDefectDetector(sample_rate=SR)
        for name, gen, material, expects in CASES[family]:
            audio = gen()
            mono = audio.mean(axis=1) if audio.ndim == 2 else audio
            scores: dict = {}
            for dt in expects:
                if dt == DefectType.DISTORTION:
                    conf, sev, _ = _fwd._detect_distortion(mono, False)
                    scores[dt] = DefectScore(dt, float(sev), float(conf))
                elif dt == DefectType.DROPOUT:
                    conf, sev, _ = _fwd._detect_dropout(mono, False)
                    scores[dt] = DefectScore(dt, float(sev), float(conf))
            failed += _report(name, expects, scores)
    else:
        for name, gen, material, expects in CASES[family]:
            sc = DefectScanner(sample_rate=SR, material_type=material)
            res = sc.scan(gen())
            failed += _report(name, expects, res.scores)
    print(f"Familie {family}: {failed} Luecke(n)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
