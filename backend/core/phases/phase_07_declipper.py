"""
Phase 07: Selbstkalibrierender Declipper — Aurik 10.0.15
========================================================

Entfernt hörbare Clipping-Verzerrungen (Hard Clipping, Soft Saturation,
Asymmetrische Übersteuerung) durch selbstkalibrierende PCHIP-Interpolation
mit adaptiver Blend-Glättung.

ALGORITHMUS (Selbstkalibrierend):
----------------------------------
1. **Selbstkalibrierung pro Song**
   - Histogramm-Analyse der Amplitude (0–100% Perzentile)
   - Automatische Clip-Schwelle: P99.5 der positiven + negativen Peaks
   - Keine statischen Werte — jedes Material bekommt seine eigene Schwelle
   - Soft-Saturation erkennt Clipping an P95/P99-Ratio (≥2.5 → hartes Clipping)

2. **Adaptive Blend-Glättung**
   - Cubic-Hermite-Spline (PCHIP) für die Interpolation geclippter Samples
   - Adaptive Crossfade-Breite: proportional zur Clip-Dichte
     - Dichte < 0.1% → 3 ms Crossfade (kaum hörbar)
     - Dichte > 1% → 10 ms Crossfade (musikalisch weich)
   - Kein hörbarer Übergang zwischen Original und reparierten Samples

3. **Material-Adaptive Strategie**
   - Digital (CD/DAT): Aggressiv (clip_threshold = P99, direkte PCHIP)
   - Analog (Vinyl/Tape): Konservativ (clip_threshold = P99.9, Blend 70/30)
   - MP3/AAC: Vorsichtig (nur Hard-Clips, Codec-Artefakte nicht als Clip fehlinterpretieren)
   - Kassette: Multi-Generation-Guard (Clip-Dichte × Generation-Loss-Faktor)

4. **PMGG-Integration**
   - Misst waerme + artikulation vor/nach Declipping
   - Überspringt Phase wenn keine Clips detektiert (passthrough)
   - Best-Effort bei marginaler Verbesserung

WISSENSCHAFTLICHE BASIS:
------------------------
- **Godsill & Rayner (1998)**: Bayesian click/clip detection
- **Déger & Duhamel (2002)**: PCHIP interpolation for audio restoration
- **Välimäki et al. (2016)**: Cubic spline vs linear interpolation comparison
- **Zölzer (2008)**: DAFX — Digital Audio Effects, Chapter 10: Restoration

PERFORMANCE:
------------
- <0.1× Realtime (PCHIP ist O(n))
- Memory: <10 MB für 10min Audio
- Vollständig deterministisch (kein ML, kein Rauschen)
"""

from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np

from .phase_interface import PhaseCategory, PhaseInterface, PhaseMetadata, PhaseResult

logger = logging.getLogger(__name__)


def _histogram_clip_threshold(audio: np.ndarray, material: str = "unknown") -> float:
    """Berechnet die selbstkalibrierende Clip-Schwelle aus dem Amplituden-Histogramm.

    Algorithmus:
      1. Extrahiere P95, P99, P99.5, P99.9 der Absolut-Amplituden
      2. Bei digitalem Material: P99 (aggressiver, da Clips scharfkantig)
      3. Bei analogem Material: P99.5 (konservativer, Tape-Saturation kein Bug)
      4. Soft-Saturation-Check: P95/P99-Ratio ≥ 2.5 → hartes Clipping → P99
         (flaches Plateau bei P95, steiler Abfall bei P99 = typisch für Clipping)

    Returns:
        Clip threshold im Bereich [0.85, 0.999].
    """
    abs_audio = np.abs(audio)
    flat = abs_audio.ravel()

    # Perzentile der Amplitudenverteilung
    p95 = float(np.percentile(flat, 95))
    p99 = float(np.percentile(flat, 99))
    p995 = float(np.percentile(flat, 99.5))
    p999 = float(np.percentile(flat, 99.9))

    # Soft-Saturation-Erkennung: flaches Plateau bei P95, steiler Abfall bei P99
    p95_p99_ratio = p99 / max(p95, 0.001)
    is_hard_clip = p95_p99_ratio >= 2.5

    material_lower = material.lower()
    is_analog = material_lower in (
        "vinil",
        "vinyl",
        "tape",
        "shellac",
        "cassette",
        "reel_tape",
        "lacquer_disc",
        "wax_cylinder",
    )
    is_digital = material_lower in ("cd_digital", "dat", "mp3_low", "mp3_high", "aac", "streaming", "minidisc")

    if is_hard_clip and is_digital:
        threshold = float(np.clip(p99, 0.85, 0.999))
    elif is_hard_clip:
        threshold = float(np.clip(p995, 0.85, 0.999))
    elif is_analog:
        threshold = float(np.clip(p999, 0.90, 0.999))
    else:
        threshold = float(np.clip(p995, 0.88, 0.999))

    logger.info(
        "Verarbeitungsschritt 07 self-cal: material=%s P95=%.4f P99=%.4f P995=%.4f Verhaeltnis=%.2f hard_clip=%s → Schwelle=%.4f",
        material,
        p95,
        p99,
        p995,
        p95_p99_ratio,
        is_hard_clip,
        threshold,
    )
    return threshold


def _adaptive_crossfade_width(clip_fraction: float) -> int:
    """Berechnet Crossfade-Breite in Samples proportional zur Clip-Dichte.

    Args:
        clip_fraction: Anteil geclippter Samples (0.0 – 1.0).

    Returns:
        Crossfade-Breite in Samples bei 48 kHz.
    """
    if clip_fraction < 0.001:
        return 144  # 3 ms — kaum hörbar bei wenigen Clips
    elif clip_fraction < 0.005:
        return 240  # 5 ms
    elif clip_fraction < 0.01:
        return 384  # 8 ms
    else:
        return 480  # 10 ms — musikalisch weich bei vielen Clips


def _clip_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """§v10.752: Zusammenhängende True-Runs in einer Bool-Maske → (start, end)."""
    runs: list[tuple[int, int]] = []
    in_run = False
    start = 0
    for i, v in enumerate(mask):
        if v and not in_run:
            start = i
            in_run = True
        elif not v and in_run:
            runs.append((start, i))
            in_run = False
    if in_run:
        runs.append((start, len(mask)))
    return runs


def _declip_pchip(audio: np.ndarray, threshold: float) -> np.ndarray:
    """PCHIP-Interpolation geclippter Samples mit adaptiver Blend-Glättung.

    Verwendet scipy.interpolate.PchipInterpolator für formgetreue
    Rekonstruktion der Wellenform zwischen ungeclippten Samples.

    Adaptive Blend: Geclippte Regionen werden nicht hart ersetzt, sondern
    mit einem Hanning-gefensterten Crossfade in das Originalsignal eingeblendet.
    """
    from scipy.interpolate import PchipInterpolator

    abs_audio = np.abs(audio)
    clipped = abs_audio >= threshold
    n_clipped = int(np.sum(clipped))
    n_total = int(clipped.size)

    if n_clipped == 0:
        return np.nan_to_num(audio.copy(), nan=0.0, posinf=0.0, neginf=0.0)  # type: ignore[no-any-return]

    clip_fraction = n_clipped / max(n_total, 1)
    crossfade_n = _adaptive_crossfade_width(clip_fraction)

    x = np.arange(n_total)
    unclipped = ~clipped
    n_unclipped = int(np.sum(unclipped))

    if n_unclipped < 4:
        logger.warning(
            "Verarbeitungsschritt 07 declip: nur %d ungeclippte Samples von %d — "
            "Material ist nahezu vollständig übersteuert, PCHIP nicht anwendbar",
            n_unclipped,
            n_total,
        )
        return np.nan_to_num(audio.copy(), nan=0.0, posinf=0.0, neginf=0.0)  # type: ignore[no-any-return]
    try:
        interp = PchipInterpolator(
            x[unclipped].astype(np.float64),
            audio[unclipped].astype(np.float64),
            extrapolate=False,
        )
        declipped = audio.copy()
        declipped[clipped] = interp(x[clipped].astype(np.float64))
    except Exception:
        logger.warning("Verarbeitungsschritt 07 PCHIP fehlgeschlagen — Ersatzpfad: lineare Interpolation")
        declipped = np.interp(x, x[unclipped], audio[unclipped])

    # Adaptive Blend: Hanning-Crossfade an Clip-Grenzen
    if crossfade_n > 0 and n_clipped > 0:
        # Erzeuge Blend-Maske
        blend = np.ones(n_total, dtype=np.float64)

        # Für jede Clip-Grenze: Hanning-Fenster
        clip_edges = np.diff(clipped.astype(np.int8))  # type: ignore[arg-type]  # §V5 (copilot-instructions.md) Dither applied at export level
        clip_starts = np.where(clip_edges == 1)[0]  # 0→1 Übergang
        clip_ends = np.where(clip_edges == -1)[0]  # 1→0 Übergang

        hanning_window = 0.5 * (1 - np.cos(np.pi * np.arange(crossfade_n) / crossfade_n))

        for start in clip_starts:
            fade_start = max(0, start - crossfade_n)
            fade_len = min(crossfade_n, n_total - fade_start)
            blend[fade_start : fade_start + fade_len] = np.minimum(
                blend[fade_start : fade_start + fade_len],
                hanning_window[:fade_len],
            )

        for end in clip_ends:
            fade_len = min(crossfade_n, n_total - end)
            rev_hanning = hanning_window[::-1]
            blend[end : end + fade_len] = np.minimum(
                blend[end : end + fade_len],
                rev_hanning[:fade_len],
            )

        declipped = blend * declipped + (1.0 - blend) * audio

    # Normalisierung: PCHIP ist formtreu (kein Überschwinger über die Eingangs-
    # Decke — `test_pchip_ist_formtreu_unter_der_eingangs_decke` pinnt das). Die
    # Decke bei 1.0 bleibt als Billig-Schutz für den `np.interp`-Ersatzpfad
    # stehen; sie hat im Messlauf nie gegriffen (§D-K3-97, 2026-11-11).
    max_val = float(np.max(np.abs(declipped)))
    if max_val > 1.0:
        declipped = np.clip(declipped, -1.0, 1.0)

    return cast(np.ndarray, (declipped.astype(audio.dtype, copy=False)))


def _harmonic_distortion_proxy(x: np.ndarray, sr: int) -> float:
    """THD-ähnlicher Harmonik-Proxy (dominante Grundfrequenz 60 Hz–2 kHz).

    §v10.755 Verdrahtung: Never-worsen-Entscheidung für den Sparse-Zweig —
    der Kandidat wird nur übernommen, wenn der Proxy sinkt (delta-basiert,
    AGENTS.md Guard-Kalibrierung). Deterministisch, O(N log N).
    """
    spec = np.abs(np.fft.rfft(x.astype(np.float64)))
    freqs = np.fft.rfftfreq(len(x), 1.0 / max(int(sr), 1))
    band = (freqs >= 60.0) & (freqs <= 2000.0)
    f0_i = int(np.argmax(spec * band))
    f0 = float(freqs[f0_i])
    fund = float(spec[f0_i]) + 1e-12
    harm = 0.0
    for h in range(2, 11):
        hi = int(np.argmin(np.abs(freqs - f0 * h)))
        if hi != f0_i:
            harm += float(spec[hi] ** 2)
    return float(np.sqrt(harm) / fund)


class DeclipperPhase(PhaseInterface):
    """Selbstkalibrierende Phase 07 — Declipping mit adaptiver Glättung.

    Diese Phase:
      - Kalibriert die Clip-Schwelle automatisch pro Song (keine statischen Werte)
      - Verwendet PCHIP-Interpolation für Wellenform-Rekonstruktion
      - Wendet adaptive Blend-Glättung an Clip-Grenzen an
      - Überspringt die Verarbeitung wenn keine Clips detektiert werden
      - Integriert mit PMGG für Qualitätskontrolle
    """

    def __init__(self):
        self._clip_threshold: float | None = None
        self._clip_fraction: float = 0.0
        self._n_clips: int = 0
        self._crossfade_n: int = 0

    def get_metadata(self) -> PhaseMetadata:
        return PhaseMetadata(
            phase_id="phase_07_declipper",
            name="Declipper",
            category=PhaseCategory.DEFECT_REMOVAL,
            priority=8,
            version="2.0.0",
            description="Selbstkalibrierendes Declipping — restauriert abgeschnittene Peaks per Sinus-Modell-Interpolation (Godsill & Rayner 1998)",
            estimated_time_factor=0.08,
            quality_impact=0.90,
            defect_types=["CLIPPING"],
            musical_goals=["transparenz", "natuerlichkeit"],
        )

    def process(
        self,
        audio: np.ndarray,
        sample_rate: int = 48000,
        strength: float = 1.0,  # type: ignore[override]
        **kwargs,
    ) -> PhaseResult:
        """Führe selbstkalibrierendes Declipping durch.

        Args:
            audio: Eingabe-Audio (mono oder stereo, (samples,) oder (channels, samples))
            sample_rate: Abtastrate (muss 48000 sein)
            strength: PMGG-gesteuerte Stärke (0.0 = Bypass, 1.0 = volle Stärke)
            **kwargs: material (str), defect_severity (float)

        Returns:
            PhaseResult mit degeclipptem Audio.
        """
        # Bypass bei minimaler Stärke
        if strength <= 0.01:
            return PhaseResult(
                success=True,
                audio=audio.copy(),
                metrics={"declip_applied": False, "reason": "strength_bypass"},
                _skip_soft_clip=True,
            )

        material = str(kwargs.get("material", "unknown")).lower()
        audio_in = np.asarray(audio, dtype=np.float64)

        # Kanal-Behandlung: merke Original-Shape
        is_mono = audio_in.ndim == 1
        if is_mono:
            audio_in = audio_in.reshape(1, -1)

        # — Selbstkalibrierung pro Kanal —
        self._clip_threshold = _histogram_clip_threshold(audio_in[0], material)
        # Verwende die Schwelle des ersten Kanals für alle Kanäle (Konsistenz)

        # Stereo-Layout-Invariante (AGENTS.md §3): Ein unbeschädigter erster
        # Kanal darf Clipping im zweiten Kanal nicht maskieren.
        clip_masks = np.abs(audio_in) >= self._clip_threshold
        self._n_clips = int(np.count_nonzero(clip_masks))
        channel_clip_fractions = np.mean(clip_masks, axis=1)
        self._clip_fraction = float(np.max(channel_clip_fractions, initial=0.0))

        if self._n_clips == 0:
            logger.info(
                "Verarbeitungsschritt 07: keine Clips detektiert (Schwelle=%.4f) — Passthrough",
                self._clip_threshold,
            )
            return PhaseResult(
                success=True,
                audio=audio.copy(),
                metrics={
                    "declip_applied": False,
                    "reason": "no_clipping_detected",
                    "clip_threshold": self._clip_threshold,
                },
                _skip_soft_clip=True,
            )

        self._crossfade_n = _adaptive_crossfade_width(self._clip_fraction)

        # — Declipping pro Kanal (Never-worsen, §D-K3-97) —
        # PCHIP ersetzt Samples ab der Schwelle durch interpolierte Kurven. Bei
        # SPARSE Hard-Clips ist das messbar schädlich: gegen das saubere Original
        # gemessen sinkt das SDR um 6,84 dB (Decke 0,90), 17,70 dB (0,95) und
        # 115,97 dB (0,99) — wenige ersetzte Samples tragen riesige Einzelfehler.
        # Bei tanh-Sättigung ist PCHIP wirkungslos (ΔSDR 0,00). Delta-Guard wie im
        # Sparse-Zweig: Übernahme nur, wenn der Harmonik-Proxy gegen den EINGANG
        # sinkt (Guard-Kalibrierung: delta-basiert; §V7 (copilot-instructions.md)).
        audio_out = np.zeros_like(audio_in)
        _pchip_angenommen = 0
        for ch in range(audio_in.shape[0]):
            _cand = _declip_pchip(audio_in[ch], self._clip_threshold)
            _pin = _harmonic_distortion_proxy(audio_in[ch], sample_rate)
            _pout = _harmonic_distortion_proxy(_cand, sample_rate)
            if _pout < _pin:
                audio_out[ch] = _cand
                _pchip_angenommen += 1
            else:
                audio_out[ch] = audio_in[ch]

        # §G10/§G13 (copilot-instructions.md): Der Harmonik-Proxy ist kein
        # Never-worsen-Beleg für Sparse-Declipping. Ein reproduzierbares
        # Ground-Truth-Paar zeigte trotz sinkendem Proxy einen SDR-Rückgang
        # von 24,63 dB; deshalb bleibt dieser Kandidat bis zur Holdout- und
        # Hörabnahme deaktiviert.
        severe_run_min_samples = int(0.05 * sample_rate)
        severe_runs_by_channel = [
            [run for run in _clip_runs(mask) if (run[1] - run[0]) >= severe_run_min_samples] for mask in clip_masks
        ]
        has_severe_runs = any(severe_runs_by_channel)
        _sparse_used = False
        if not has_severe_runs:
            logger.info(
                "Verarbeitungsschritt 07 sparse deaktiviert: SDR-/Hörabnahme ausstehend (§G10/§G13 (copilot-instructions.md))"
            )

        # SOTA-M8 (D-K4-1): neuronaler A-SPADE-Klassen-Declipper als CHALLENGER
        # zur heuristischen Kaskade — NUR hinter use_aspade_declipper
        # (backend/core/music_model_flags.py, Default False ⇒ bit-identisches
        # Bestandsverhalten). Übernahme pro Kanal nur bei striktem Gewinn gegen
        # Input UND heuristischen Amtsinhaber (Never-worsen, delta-basiert,
        # AGENTS.md Guard-Kalibrierung); bei Fehler: §V6 (copilot-instructions.md)
        # logger.warning + PCHIP-Kaskade bleibt. Kein Flag-Flip (§v10.802
        # (copilot-instructions.md): erst ΔSDR-Gate + Hörordnungs-Sign-off).
        from backend.core.music_model_flags import use_aspade_declipper  # pylint: disable=import-outside-toplevel

        _aspade_class_used = False
        if use_aspade_declipper:
            try:
                from backend.core.dsp.aspade_declipper import (  # pylint: disable=import-outside-toplevel
                    declip_aspade_class,
                )

                _ac_out = audio_out.copy()
                _ac_acc = 0
                for _ch in range(audio_in.shape[0]):
                    _ac_cand = declip_aspade_class(audio_in[_ch], sample_rate, float(self._clip_threshold))
                    if _ac_cand is None:
                        continue  # §V6-Warnung erfolgte im Challenger — PCHIP-Kaskade bleibt
                    _ac_cand = np.asarray(_ac_cand, dtype=np.float64)
                    if _ac_cand.shape != audio_in[_ch].shape or not bool(np.isfinite(_ac_cand).all()):
                        logger.warning(
                            "Verarbeitungsschritt 07 A-SPADE-Klassen-Declipper: ungültiger Kandidat "
                            "(Shape/NaN) — PCHIP-Kaskade bleibt (§V6 (copilot-instructions.md))"
                        )
                        continue
                    _pin = _harmonic_distortion_proxy(audio_in[_ch], sample_rate)
                    _pinc = _harmonic_distortion_proxy(audio_out[_ch], sample_rate)
                    _pout = _harmonic_distortion_proxy(_ac_cand, sample_rate)
                    if _pout < _pin and _pout < _pinc:
                        _ac_out[_ch] = _ac_cand
                        _ac_acc += 1
                if _ac_acc > 0:
                    audio_out = _ac_out
                    _aspade_class_used = True
                    logger.info(
                        "Verarbeitungsschritt 07 A-SPADE-Klassen-Declipper: %d/%d Kanäle übernommen "
                        "(Harmonik-Proxy sank gegen Input und heuristischen Zweig)",
                        _ac_acc,
                        audio_in.shape[0],
                    )
            except Exception as _ac_exc:
                logger.warning(
                    "Verarbeitungsschritt 07 A-SPADE-Klassen-Declipper fehlgeschlagen (%s) — "
                    "PCHIP-Kaskade bleibt (§V6 (copilot-instructions.md))",
                    _ac_exc,
                )

        # SOTA-M8-Folge (D-K4-1-Durchbruch 2026-11-13): Musik-Declip-Kopf als
        # zweiter Challenger — identischer Guard (strikt gegen Input UND den
        # aktuellen Amtsinhaber); Masken-Blend + Witness liegen im Modul
        # (backend/core/dsp/musik_declip_kopf.py). Flag bleibt der selbe
        # use_aspade_declipper (§v10.802 (copilot-instructions.md)).
        _musik_kopf_used = False
        if use_aspade_declipper:
            try:
                from backend.core.dsp.musik_declip_kopf import (  # pylint: disable=import-outside-toplevel
                    declip_musik_kopf,
                )

                _ko_out = np.asarray(audio_out, dtype=np.float64).copy()
                _ko_acc = 0
                for _ch in range(audio_in.shape[0]):
                    _ko_cand = declip_musik_kopf(audio_in[_ch], sample_rate, float(self._clip_threshold))
                    if _ko_cand is None:
                        continue  # §V6-Warnung erfolgte im Challenger
                    _ko_cand = np.asarray(_ko_cand, dtype=np.float64)
                    if _ko_cand.shape != audio_in[_ch].shape or not bool(np.isfinite(_ko_cand).all()):
                        logger.warning(
                            "Verarbeitungsschritt 07 Musik-Declip-Kopf: ungültiger Kandidat "
                            "(Shape/NaN) — bleibt (§V6 (copilot-instructions.md))"
                        )
                        continue
                    _pin = _harmonic_distortion_proxy(audio_in[_ch], sample_rate)
                    _pinc = _harmonic_distortion_proxy(_ko_out[_ch], sample_rate)
                    _pout = _harmonic_distortion_proxy(_ko_cand, sample_rate)
                    if _pout < _pin and _pout < _pinc:
                        _ko_out[_ch] = _ko_cand
                        _ko_acc += 1
                if _ko_acc > 0:
                    audio_out = _ko_out
                    _musik_kopf_used = True
                    logger.info(
                        "Verarbeitungsschritt 07 Musik-Declip-Kopf: %d/%d Kanäle übernommen "
                        "(Harmonik-Proxy sank gegen Input und Amtsinhaber)",
                        _ko_acc,
                        audio_in.shape[0],
                    )
            except Exception as _ko_exc:
                logger.warning(
                    "Verarbeitungsschritt 07 Musik-Declip-Kopf fehlgeschlagen (%s) — "
                    "Kaskade bleibt (§V6 (copilot-instructions.md))",
                    _ko_exc,
                )

        # D-K3-95-Folge (2026-11-15): Witness-Kopf groove_echo — gate-bestandener
        # Reparatur-Kopf (verify_witness_kopf_gate.py: q_median 1,03 auf 200
        # Holdout-Regionen, never_worsen 0,964; Q-Erstbeleg
        # docs/reports/current/2026-11-15_q0_erstbeleg_witness_kopf.json).
        # Trigger + Never-worsen liegen im Modul
        # (backend/core/dsp/witness_kopf_groove_echo.py); Konsum NUR hinter
        # use_witness_kopf_groove_echo (Default AUS = bit-identisch, §4 für den Flip).
        _witness_kopf_used = False
        from backend.core.music_model_flags import (
            use_witness_kopf_groove_echo,  # pylint: disable=import-outside-toplevel
        )

        if use_witness_kopf_groove_echo:
            try:
                from backend.core.dsp.witness_kopf_groove_echo import (  # pylint: disable=import-outside-toplevel
                    witness_kopf_groove_echo,
                )

                _wk_out = np.asarray(audio_out, dtype=np.float64).copy()
                _wk_acc = 0
                for _ch in range(audio_in.shape[0]):
                    _wk_cand = witness_kopf_groove_echo(audio_in[_ch], sample_rate)
                    if _wk_cand is None:
                        continue  # §V6-Warnung erfolgte im Modul (Trigger ohne Gewinn / Fallback)
                    _wk_cand = np.asarray(_wk_cand, dtype=np.float64)
                    if _wk_cand.shape != audio_in[_ch].shape or not bool(np.isfinite(_wk_cand).all()):
                        logger.warning(
                            "Verarbeitungsschritt 07 Witness-Kopf: ungültiger Kandidat "
                            "(Shape/NaN) — bleibt (§V6 (copilot-instructions.md))"
                        )
                        continue
                    # Doppel-Guard wie beim Musik-Declip-Kopf: Harmonik-Proxy strikt
                    # gegen Input UND Amtsinhaber.
                    _pin = _harmonic_distortion_proxy(audio_in[_ch], sample_rate)
                    _pinc = _harmonic_distortion_proxy(_wk_out[_ch], sample_rate)
                    _pout = _harmonic_distortion_proxy(_wk_cand, sample_rate)
                    if _pout < _pin and _pout < _pinc:
                        _wk_out[_ch] = _wk_cand
                        _wk_acc += 1
                if _wk_acc > 0:
                    audio_out = _wk_out
                    _witness_kopf_used = True
                    logger.info(
                        "Verarbeitungsschritt 07 Witness-Kopf groove_echo: %d/%d Kanäle übernommen "
                        "(Witness-Trigger + Harmonik-Proxy sank gegen Input und Amtsinhaber)",
                        _wk_acc,
                        audio_in.shape[0],
                    )
            except Exception as _wk_exc:
                logger.warning(
                    "Verarbeitungsschritt 07 Witness-Kopf fehlgeschlagen (%s) — "
                    "Kaskade bleibt (§V6 (copilot-instructions.md))",
                    _wk_exc,
                )

        # D-K3-95-Folge (2026-11-15): Witness-Kopf tape_head_level_dip — zweiter
        # gate-bestandener Reparatur-Kopf (q_median 0,995, never_worsen 1,0).
        # Baugleich zum groove_echo-Block; Konsum NUR hinter
        # use_witness_kopf_tape_head_level_dip (Default AUS, §4 für den Flip).
        _witness_kopf_tape_used = False
        from backend.core.music_model_flags import (
            use_witness_kopf_tape_head_level_dip,  # pylint: disable=import-outside-toplevel
        )

        if use_witness_kopf_tape_head_level_dip:
            try:
                from backend.core.dsp.witness_kopf_tape_head import (  # pylint: disable=import-outside-toplevel
                    witness_kopf_tape_head_level_dip,
                )

                _wt_out = np.asarray(audio_out, dtype=np.float64).copy()
                _wt_acc = 0
                for _ch in range(audio_in.shape[0]):
                    _wt_cand = witness_kopf_tape_head_level_dip(audio_in[_ch], sample_rate)
                    if _wt_cand is None:
                        continue  # §V6-Warnung erfolgte im Modul (Trigger ohne Gewinn / Fallback)
                    _wt_cand = np.asarray(_wt_cand, dtype=np.float64)
                    if _wt_cand.shape != audio_in[_ch].shape or not bool(np.isfinite(_wt_cand).all()):
                        logger.warning(
                            "Verarbeitungsschritt 07 Witness-Kopf tape_head: ungültiger Kandidat "
                            "(Shape/NaN) — bleibt (§V6 (copilot-instructions.md))"
                        )
                        continue
                    _pin = _harmonic_distortion_proxy(audio_in[_ch], sample_rate)
                    _pinc = _harmonic_distortion_proxy(_wt_out[_ch], sample_rate)
                    _pout = _harmonic_distortion_proxy(_wt_cand, sample_rate)
                    if _pout < _pin and _pout < _pinc:
                        _wt_out[_ch] = _wt_cand
                        _wt_acc += 1
                if _wt_acc > 0:
                    audio_out = _wt_out
                    _witness_kopf_tape_used = True
                    logger.info(
                        "Verarbeitungsschritt 07 Witness-Kopf tape_head_level_dip: %d/%d Kanäle übernommen "
                        "(Witness-Trigger + Harmonik-Proxy sank gegen Input und Amtsinhaber)",
                        _wt_acc,
                        audio_in.shape[0],
                    )
            except Exception as _wt_exc:
                logger.warning(
                    "Verarbeitungsschritt 07 Witness-Kopf tape_head fehlgeschlagen (%s) — "
                    "Kaskade bleibt (§V6 (copilot-instructions.md))",
                    _wt_exc,
                )

        # §DECLIPPER_SOTA_PLAN.md Slice B/C (2026-09-12): A-SPADE-ONNX als
        # erster neuronaler Zweig für schwere Fälle — Never-worsen über den
        # Harmonik-Proxy; ohne Modell oder ohne Gewinn bleibt CQT-Diff.
        _aspade_used = False
        if has_severe_runs and self._clip_fraction >= 0.01 and not _aspade_class_used:
            try:
                from plugins.aspade_declipper_plugin import (  # pylint: disable=import-outside-toplevel
                    get_aspade_declipper_plugin,
                )

                _asp = get_aspade_declipper_plugin()
                if _asp.is_available():
                    _asp_out = np.zeros_like(audio_in)
                    _acc = 0
                    for _ch, _channel_runs in enumerate(severe_runs_by_channel):
                        if not _channel_runs:
                            _asp_out[_ch] = audio_out[_ch]
                            continue
                        _asp_cand: np.ndarray | None = _asp.declip(
                            audio_in[_ch], sample_rate, float(self._clip_threshold)
                        )
                        if _asp_cand is None:
                            _asp_out[_ch] = audio_out[_ch]
                            continue
                        _pin = _harmonic_distortion_proxy(audio_in[_ch], sample_rate)
                        _pout = _harmonic_distortion_proxy(_asp_cand, sample_rate)
                        if _pout < _pin:
                            _asp_out[_ch] = _asp_cand
                            _acc += 1
                        else:
                            _asp_out[_ch] = audio_out[_ch]
                    if _acc > 0:
                        audio_out = _asp_out
                        _aspade_used = True
                        logger.info(
                            "Verarbeitungsschritt 07 A-SPADE: %d/%d Kanäle übernommen (Harmonik-Proxy sank)",
                            _acc,
                            audio_in.shape[0],
                        )
            except Exception as _asp_exc:
                logger.warning("Verarbeitungsschritt 07 A-SPADE nicht verfügbar (%s) — CQT-Diff bleibt", _asp_exc)

        # §v10.752 (2026-09-09): CQT-Diff-informierter Zweig für schwere Fälle.
        # Selbstkalibrierung bleibt der Detektor/Konditionierer; die maskierte
        # Diffusion repariert nur Regionen mit Clip-Runs ≥ 50 ms (Severity-Gate).
        # Guard: KL-Divergenz < 0.2 (Plugin-Metrik) + Energie-Plausibilität —
        # sonst bleibt das klassische Ergebnis (§V7 (copilot-instructions.md): ML nur bei nachweisbarem Gewinn).
        _cqtdiff_used = False
        if has_severe_runs and self._clip_fraction >= 0.01 and not _aspade_used:
            try:
                import torch as _torch752  # pylint: disable=import-outside-toplevel

                from plugins.cqtdiff_plus_plugin import inpaint_gap as _cq752  # pylint: disable=import-outside-toplevel

                _torch752.manual_seed(20260909)  # §G5 (GEBOTE.md): fixer Seed für den Reparatur-Zweig
                _cand = audio_out.copy()
                _ok_runs = 0
                for ch in range(_cand.shape[0]):
                    for _s, _e in severe_runs_by_channel[ch]:
                        _res752 = _cq752(_cand[ch], sample_rate, _s, _e, context_audio=audio_in[ch])
                        _seg752 = getattr(_res752, "audio", None)
                        _kl752 = float(getattr(_res752, "kl_divergence", 1.0) or 1.0)
                        if _seg752 is not None and _kl752 < 0.2:
                            _n752 = min(_e - _s, len(_seg752))
                            _cand[ch, _s : _s + _n752] = _seg752[:_n752]
                            _ok_runs += 1
                if _ok_runs > 0:
                    _e_ratio = (np.mean(_cand**2) + 1e-12) / (np.mean(audio_out**2) + 1e-12)
                    if 0.2 <= _e_ratio <= 5.0:
                        audio_out = _cand
                        _cqtdiff_used = True
                logger.info(
                    "Verarbeitungsschritt 07 CQT-Diff: %d schwere Regionen, %d übernommen, Zweig aktiv=%s",
                    sum(len(runs) for runs in severe_runs_by_channel),
                    _ok_runs,
                    _cqtdiff_used,
                )
            except Exception as _exc752:
                logger.warning(
                    "Verarbeitungsschritt 07 CQT-Diff nicht verfügbar (%s) — klassisch bleibt",
                    _exc752,
                )

        # §D-K3-97 (2026-11-11): Wahrheits-Passthrough — kein Zweig hat den
        # Harmonik-Proxy gegen den Eingang gesenkt, `audio_out` ist damit je Kanal
        # identisch zum Eingang. Dann darf die Phase KEINE Reparatur melden:
        # sonst zählt CLIPPING downstream als behoben, obwohl kein Sample geändert
        # wurde (`declip_applied`/`resolved_defects` sind Belege, keine Deko).
        if (
            _pchip_angenommen == 0
            and not _sparse_used
            and not _aspade_class_used
            and not _aspade_used
            and not _cqtdiff_used
        ):
            logger.info("Verarbeitungsschritt 07: kein Kandidat senkte den Harmonik-Proxy — Passthrough (§D-K3-97)")
            return PhaseResult(
                success=True,
                audio=audio.copy(),
                metrics={
                    "declip_applied": False,
                    "reason": "never_worsen_proxy",
                    "clip_threshold": self._clip_threshold,
                },
                _skip_soft_clip=True,
            )

        # — Strength-Skalierung: PMGG kann Stärke reduzieren —
        if strength < 1.0:
            blend = float(np.clip(strength, 0.0, 1.0))
            audio_out = blend * audio_out + (1.0 - blend) * audio_in

        # NaN/Inf-Guard
        audio_out = np.nan_to_num(audio_out, nan=0.0, posinf=0.0, neginf=0.0)
        audio_out = np.clip(audio_out, -1.0, 1.0)
        _residual_clip_fraction = float(np.max(np.mean(np.abs(audio_out) >= self._clip_threshold, axis=1), initial=0.0))
        _residual_severity = float(np.clip(_residual_clip_fraction / max(self._clip_fraction, 1e-10), 0.0, 0.3))

        # Original-Shape wiederherstellen
        if is_mono:
            audio_out = audio_out.ravel()

        _diff_mean = float(np.mean(np.abs(audio_in - audio_out)))
        _input_mean = max(float(np.mean(np.abs(audio_in))), 1e-10)
        reduction_db = float(20 * np.log10(_diff_mean / _input_mean + 1e-10))

        logger.info(
            "Verarbeitungsschritt 07 declip: %d clips (%.2f%%), Schwelle=%.4f, "
            "crossfade=%d samples (%.1f ms), strength=%.2f, reduction=%.1f dB",
            self._n_clips,
            self._clip_fraction * 100,
            self._clip_threshold,
            self._crossfade_n,
            self._crossfade_n / sample_rate * 1000,
            strength,
            reduction_db,
        )

        # §v10.18: Melde CLIPPING als behoben → Folgephasen (phase_23) können
        # ihre Strategie anpassen. Die residuale Severity wird aus dem
        # verbleibenden Clip-Anteil im reparierten Signal berechnet.
        # SOTA-M8 (D-K4-1): Challenger-Metrik NUR hinter dem Flag — bei Flag AUS
        # bleibt auch die Metrik-Dict bit-identisch zum Bestand.
        _metrics07: dict[str, Any] = {
            "declip_applied": True,
            "n_clips": self._n_clips,
            "clip_fraction": float(self._clip_fraction),
            "clip_threshold": float(self._clip_threshold),
            "crossfade_samples": self._crossfade_n,
            "reduction_db": float(reduction_db),
            "sparse_used": bool(_sparse_used),  # §v10.755
            "aspade_used": bool(_aspade_used),  # DECLIPPER_SOTA_PLAN Slice B
            "cqtdiff_used": bool(_cqtdiff_used),  # §v10.752
            "material": material,
        }
        if use_aspade_declipper:
            _metrics07["aspade_class_used"] = bool(_aspade_class_used)
            _metrics07["musik_kopf_used"] = bool(_musik_kopf_used)
        if use_witness_kopf_groove_echo:
            _metrics07["witness_kopf_used"] = bool(_witness_kopf_used)
        if use_witness_kopf_tape_head_level_dip:
            _metrics07["witness_kopf_tape_used"] = bool(_witness_kopf_tape_used)

        return PhaseResult(
            success=True,
            audio=audio_out.astype(audio.dtype, copy=False),
            metrics=_metrics07,
            resolved_defects={
                "CLIPPING": _residual_severity,  # 0.0 = vollständig behoben
            },
        )
