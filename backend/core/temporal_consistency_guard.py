"""backend/core/temporal_consistency_guard.py — §v10.700 J3.

Erkennt zeitliche Inkonsistenzen zwischen aufeinanderfolgenden Phasen:
  - Energie-Sprünge >6 dB zwischen 100ms-Fenstern
  - Wiedereinführung von Rauschen nach Denoise-Phasen
  - Stereo-Bild-Kollaps (M/S-Ratio >30%)

Integration in _profiled_phase_call(): nach JEDER Phase (§2.69b, v10.0.8) —
Stereo-/Rausch-Verstöße lösen eine konservative Dry/Wet-Rescue aus,
Energie-Sprünge speisen einen dämpfenden Folgephasen-Scalar (≤ 1.0,
Do-No-Harm). Am Export (bridge_export.py) läuft der Guard weiterhin als
rein beobachtende Transparenz-Metrik (Hörordnung §1: Zeuge, kein Richter).

Layout-Invariante §V7 (copilot-instructions.md): Alle Messungen normalisieren
über backend.core.audio_layout — kein hartes (N, 2)/(2, N)-Annehmen.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


def compute_dampening_scalar(pending_violations: int) -> float:
    """Dämpfender Folgephasen-Scalar aus offenen Konsistenz-Verstößen (≤ 1.0).

    §2.69b (UV3): Nur dämpfend, nie verstärkend (Do-No-Harm, Muster
    §8.6g-II). 0 Verstöße → 1.0 (kein Eingriff); 4+ Verstöße → 0.4 (Floor).
    """
    return float(np.clip(1.0 - 0.15 * max(0, int(pending_violations)), 0.4, 1.0))


@dataclass
class TemporalConsistencyResult:
    """Ergebnis der Temporal-Consistency-Prüfung."""

    passed: bool = True
    energy_jumps: int = 0
    noise_reintroduced: bool = False
    stereo_collapse: bool = False
    warnings: list[str] = field(default_factory=list)


class TemporalConsistencyGuard:
    """Prüft zeitliche Konsistenz zwischen Audio vor/nach einer Phase."""

    def __init__(
        self,
        energy_threshold_db: float = 6.0,
        noise_threshold_db: float = -60.0,
        stereo_threshold: float = 0.30,
        window_ms: int = 100,
    ):
        self.energy_threshold_db = energy_threshold_db
        self.noise_threshold_db = noise_threshold_db
        self.stereo_threshold = stereo_threshold
        self.window_ms = int(window_ms)
        # 48kHz-Default (Export-Pfad-Historie); check() berechnet das Fenster
        # abhängig von sr — dieser Wert bleibt nur für Rückwärtskompatibilität.
        self.window_samples = int(self.window_ms * 48 / 1000)

    def check(
        self,
        audio_before: np.ndarray,
        audio_after: np.ndarray,
        phase_id: str = "",
        *,
        sr: int = 48000,
        relative_to_median: bool = False,
    ) -> TemporalConsistencyResult:
        """Prüft Konsistenz zwischen vor/nach einer Phase.

        Args:
            audio_before/audio_after: Pre-/Post-Signal (beliebiges Layout —
                wird über backend.core.audio_layout normalisiert).
            phase_id: Phasen-ID (steuert die Rausch-Wiedereinführungs-Prüfung).
            sr: Sample-Rate für das 100ms-Fenster (Default 48000).
            relative_to_median: True = Energie-Sprung-Zählung relativ zum
                Median-Delta der Phase (UV3-Phasen-Pfad §2.69b; robust gegen
                uniforme Pegeländerungen). False = absolutes Delta (Export).

        Returns:
            TemporalConsistencyResult mit passed=True wenn konsistent.
        """
        result = TemporalConsistencyResult()

        try:
            from backend.core.audio_layout import mono_mix, to_channels_first

            before_arr = np.asarray(audio_before)
            after_arr = np.asarray(audio_after)
            mono_before = mono_mix(before_arr)
            mono_after = mono_mix(after_arr)
            win = max(int(self.window_ms * sr / 1000.0), 16)

            # 1. Energie-Sprünge zwischen 100ms-Fenstern
            result.energy_jumps = self._count_energy_jumps(
                mono_before, mono_after, win, relative_to_median
            )

            # 2. Rausch-Wiedereinführung (nur nach Denoise-Phasen)
            if "denoise" in phase_id.lower() or "noise" in phase_id.lower():
                result.noise_reintroduced = self._check_noise_reintroduction(mono_before, mono_after)

            # 3. Stereo-Kollaps (nur bei 2-Kanal-Signalen)
            result.stereo_collapse = self._check_stereo_collapse(
                to_channels_first(before_arr), to_channels_first(after_arr)
            )

            # Warnings generieren
            if result.energy_jumps > 0:
                result.warnings.append(
                    f"{result.energy_jumps} Energie-Sprünge >{self.energy_threshold_db}dB "
                    f"zwischen 100ms-Fenstern in Phase {phase_id}"
                )
            if result.noise_reintroduced:
                result.warnings.append(
                    f"Rauschen wurde in Phase {phase_id} wiedereingeführt (nach vorheriger Denoise-Phase)"
                )
            if result.stereo_collapse:
                result.warnings.append(
                    f"Stereo-Bild-Kollaps in Phase {phase_id} (M/S-Ratio >{self.stereo_threshold:.0%})"
                )

            result.passed = len(result.warnings) == 0

            if not result.passed:
                logger.warning(
                    "TemporalConsistencyGuard %s: %d Verletzungen — %s",
                    phase_id,
                    len(result.warnings),
                    "; ".join(result.warnings),
                )

        except Exception as exc:
            # Guard-Ausfall darf die Phase nie blockieren (§V6 (copilot-instructions.md):
            # Logging mit Begründung statt Stille).
            logger.warning("TemporalConsistencyGuard Fehler in %s — Pass-Through: %s", phase_id, exc)

        return result

    def _count_energy_jumps(
        self,
        before: np.ndarray,
        after: np.ndarray,
        win: int,
        relative_to_median: bool,
    ) -> int:
        """Zählt Energie-Sprünge >threshold zwischen 100ms-Fenstern (vektorisiert).

        relative_to_median=True: Zählt nur Fenster, deren Delta vom
        Median-Delta der Phase abweicht — uniforme Pegeländerungen
        (z. B. Loudness-Normalisierung) erzeugen dann keine False-Positives.
        """
        n = min(len(before), len(after))
        win = max(int(win), 16)
        n_win = n // win
        if n_win < 1:
            return 0
        b = np.asarray(before[: n_win * win], dtype=np.float64).reshape(n_win, win)
        a = np.asarray(after[: n_win * win], dtype=np.float64).reshape(n_win, win)
        rms_before = np.sqrt(np.mean(b**2, axis=1)) + 1e-12
        rms_after = np.sqrt(np.mean(a**2, axis=1)) + 1e-12
        delta_db = np.abs(20.0 * np.log10(rms_after / rms_before))
        if relative_to_median:
            delta_db = np.abs(delta_db - float(np.median(delta_db)))
        return int(np.count_nonzero(delta_db > self.energy_threshold_db))

    def _check_noise_reintroduction(self, before: np.ndarray, after: np.ndarray) -> bool:
        """Prüft ob nach einer Denoise-Phase wieder Rauschen hinzugefügt wurde."""
        rms_before = float(np.sqrt(np.mean(before**2)) + 1e-12)
        rms_after = float(np.sqrt(np.mean(after**2)) + 1e-12)

        # Nur relevant wenn Input sehr leise war (erfolgreiches Denoising)
        if 20 * np.log10(rms_before) < self.noise_threshold_db:
            return False

        # Prüfe ob Output signifikant mehr Rauschen hat (>3dB)
        return bool(20 * np.log10(rms_after / rms_before) > 3.0)

    def _check_stereo_collapse(self, before: np.ndarray, after: np.ndarray) -> bool:
        """Prüft ob das Stereo-Bild kollabiert ist (M/S-Ratio-Änderung >30%).

        Erwartet channels-first (C, N) — Normalisierung erfolgt im Aufrufer
        über backend.core.audio_layout.to_channels_first. Mono und
        <2-Kanal-Signale: kein Befund (kein Stereo-Bild).
        """
        if before.ndim != 2 or after.ndim != 2 or before.shape[0] < 2 or after.shape[0] < 2:
            return False

        n = min(before.shape[1], after.shape[1])

        # Mid/Side-Berechnung
        mid_before = (before[0, :n] + before[1, :n]) / 2
        side_before = (before[0, :n] - before[1, :n]) / 2
        mid_after = (after[0, :n] + after[1, :n]) / 2
        side_after = (after[0, :n] - after[1, :n]) / 2

        ms_ratio_before = float(np.sqrt(np.mean(side_before**2)) / (np.sqrt(np.mean(mid_before**2)) + 1e-12))
        ms_ratio_after = float(np.sqrt(np.mean(side_after**2)) / (np.sqrt(np.mean(mid_after**2)) + 1e-12))

        if ms_ratio_before < 0.01:  # Mono-Signal
            return False

        change = abs(ms_ratio_after - ms_ratio_before) / ms_ratio_before
        return bool(change > self.stereo_threshold)
