"""Remaster-Detektor — erkennt digitale Neuauflagen historischer Aufnahmen.

Aurik 10.0.0 — §2.14 EraClassifier-Guard: Verhindert falsche Ära-Zuweisung,
wenn eine historische Aufnahme digital neu gemastert wurde (z. B. 1928er
Shellac mit 22 kHz Bandbreite und −90 dBFS Rauschboden).

Erkennungsmerkmale digitaler Remaster:
  - Rauschboden < −80 dBFS (professionelle digitale Stille; Analogband ≈ −60 dBFS)
  - HF-Rolloff > 18 kHz  (volle CD/digital-Bandbreite; Originalband rollt bei 6–14 kHz ab)
  - Zusammen fingerprinting ein digital remastertes Artefakt

Singleton-Pattern §3.2: Thread-sicher via Double-Checked Locking.

Referenz: §2.14 Aurik-9-Spec (v10.0.0)
Datum: Februar 2026
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Datenklasse
# ---------------------------------------------------------------------------


@dataclass
class RemasterResult:
    """Ergebnis der Remaster-Erkennung.

    Attributes:
        is_remaster:       True wenn starke Hinweise auf eine digitale Neuauflage.
        confidence:        Gesamt-Konfidenz ∈ [0.0, 1.0].
        noise_floor_db:    Rauschboden in dBFS (5. Perzentil der Frame-Energien).
        hf_rolloff_khz:    Bandbreite in kHz bei 90 % kumulativer Spektralenergie.
        evidence:          Menschenlesbare Begründungen (Deutsch).
    """

    is_remaster: bool
    confidence: float
    noise_floor_db: float
    hf_rolloff_khz: float
    evidence: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Hauptklasse
# ---------------------------------------------------------------------------


class RemasterDetector:
    """Erkennt digitale Remaster anhand Rauschboden und Bandbreite.

    Algorithmus:
        1. Mono-Downmix (falls Stereo)
        2. Rauschboden: Frame-Energien (100 ms, 50 % Überlappung),
           5. Perzentil → noise_floor_db [dBFS]
        3. HF-Rolloff: rfft-PSD kumulativ → Frequenz bei 90 % Gesamtenergie
        4. is_remaster = noise_floor_db < NOISE_FLOOR_THRESHOLD
                         AND hf_rolloff_khz > HF_ROLLOFF_THRESHOLD
        5. confidence = floor_score × bw_score   (beide ∈ [0, 1])
        6. evidence: Deutsch-laienverständliche Begründungsliste

    Invarianten:
        - NaN/Inf-sicher: nan_to_num am Eingang, Ausgaben geclamppt
        - math.isfinite() für alle float-Felder im Ergebnis
        - Laufzeit: ≤ 0.5 s / Minute Audio (reines DSP, kein ML)
        - Thread-sicher: Singleton via Double-Checked Locking (§3.2)
    """

    NOISE_FLOOR_THRESHOLD: float = -80.0  # dBFS  — digitale Stille
    HF_ROLLOFF_THRESHOLD: float = 18.0  # kHz   — volle CD/digital-Bandbreite
    FRAME_DURATION_S: float = 0.10  # 100 ms Frame-Länge
    FRAME_HOP_S: float = 0.05  # 50 % Überlappung

    def analyse(self, audio: np.ndarray, sr: int) -> RemasterResult:
        """Analysiert Audio auf Remaster-Merkmale.

        Algorithmus:
            1. Mono-Downmix + NaN/Inf-Bereinigung
            2. Rauschboden = 5. Pct der Frame-RMS-Energien → dBFS
            3. HF-Rolloff = Frequenz bei 90 % kumulativer rfft-Energie → kHz
            4. Entscheidung: (floor < −80) AND (rolloff > 18 kHz)
            5. Konfidenz = floor_score × bw_score ∈ [0, 1]

        Args:
            audio: float32/64 ndarray, mono oder stereo.
            sr:    Abtastrate in Hz (≥ 8000 erforderlich).

        Returns:
            RemasterResult — alle Felder sind math.isfinite().

        Raises:
            Kein Ausnahme-Propagation — robuste Fallback-Pfade für jeden Fehlerfall.
        """
        if sr < 8000:
            return RemasterResult(
                is_remaster=False,
                confidence=0.0,
                noise_floor_db=0.0,
                hf_rolloff_khz=0.0,
                evidence=["Abtastrate zu niedrig für zuverlässige Analyse."],
            )

        # ── Vorverarbeitung ─────────────────────────────────────────────
        audio_f = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        # Stereo → Mono
        if audio_f.ndim == 2:
            audio_f = audio_f.mean(axis=0) if audio_f.shape[0] <= audio_f.shape[1] else audio_f.mean(axis=1)
        audio_f = audio_f.ravel()

        if len(audio_f) < 256:
            return RemasterResult(
                is_remaster=False,
                confidence=0.0,
                noise_floor_db=0.0,
                hf_rolloff_khz=0.0,
                evidence=["Signal zu kurz für Remaster-Erkennung."],
            )

        # ── Merkmale berechnen ──────────────────────────────────────────
        noise_floor_db = self._compute_noise_floor(audio_f, sr)
        hf_rolloff_khz = self._compute_hf_rolloff(audio_f, sr)

        # ── Entscheidung ────────────────────────────────────────────────
        floor_score = self._floor_score(noise_floor_db)
        bw_score = self._bw_score(hf_rolloff_khz)
        confidence = float(np.clip(floor_score * bw_score, 0.0, 1.0))

        is_remaster = noise_floor_db < self.NOISE_FLOOR_THRESHOLD and hf_rolloff_khz > self.HF_ROLLOFF_THRESHOLD

        # ── Begründungen (Deutsch) ──────────────────────────────────────
        evidence: list[str] = []
        if noise_floor_db < self.NOISE_FLOOR_THRESHOLD:
            evidence.append(
                f"Rauschboden {noise_floor_db:.1f} dBFS — typisch für digitale "
                f"Stille (Schwelle: {self.NOISE_FLOOR_THRESHOLD:.0f} dBFS)."
            )
        else:
            evidence.append(
                f"Rauschboden {noise_floor_db:.1f} dBFS — "
                "analoges Bandrauschen erkannt (kein digitaler Remaster-Hinweis)."
            )

        if hf_rolloff_khz > self.HF_ROLLOFF_THRESHOLD:
            evidence.append(
                f"Bandbreite {hf_rolloff_khz:.1f} kHz — volle digitale Bandbreite "
                f"(Schwelle: {self.HF_ROLLOFF_THRESHOLD:.0f} kHz)."
            )
        else:
            evidence.append(
                f"Bandbreite {hf_rolloff_khz:.1f} kHz — "
                "analogbegrenzte Bandbreite erkannt (kein digitaler Remaster-Hinweis)."
            )

        if is_remaster:
            evidence.append(f"Remaster erkannt: beide Kriterien erfüllt (Konfidenz {confidence:.2f}).")
        else:
            evidence.append("Kein Remaster erkannt — mindestens ein Kriterium nicht erfüllt.")

        logger.debug(
            "RemasterDetector: floor=%.1f dBFS  hf_rolloff=%.1f kHz  is_remaster=%s  conf=%.2f",
            noise_floor_db,
            hf_rolloff_khz,
            is_remaster,
            confidence,
        )

        return RemasterResult(
            is_remaster=is_remaster,
            confidence=confidence,
            noise_floor_db=float(noise_floor_db),
            hf_rolloff_khz=float(hf_rolloff_khz),
            evidence=evidence,
        )

    # ------------------------------------------------------------------
    # Interne Hilfsmethoden
    # ------------------------------------------------------------------

    def _compute_noise_floor(self, audio: np.ndarray, sr: int) -> float:
        """5. Perzentil der Frame-Energien als Rauschboden-Schätzung [dBFS].

        Formel:
            frame_energy[k] = mean(audio[k*hop : k*hop + frame_size] ** 2)
            noise_floor_db  = 10 · log10(percentile(frame_energy, 5))

        Returns:
            noise_floor_db ∈ [−140, 0] dBFS (geclamppt).
        """
        frame_size = max(64, int(self.FRAME_DURATION_S * sr))
        hop_size = max(32, int(self.FRAME_HOP_S * sr))
        n = len(audio)

        energies: list[float] = []
        i = 0
        while i + frame_size <= n:
            frame = audio[i : i + frame_size]
            e = float(np.mean(frame**2))
            energies.append(e)
            i += hop_size

        if not energies:
            return -120.0

        floor_energy = float(np.percentile(energies, 5))
        if floor_energy < 1e-30:
            return -120.0

        noise_floor_db = 10.0 * math.log10(floor_energy)
        return float(np.clip(noise_floor_db, -140.0, 0.0))

    def _compute_hf_rolloff(self, audio: np.ndarray, sr: int) -> float:
        """Frequenz in kHz bei der 90 % der rfft-Spektralenergie akkumuliert ist.

        Formel:
            psd    = |rfft(audio)|²
            cumsum = cumulative_sum(psd) / sum(psd)
            idx    = first index where cumsum ≥ 0.90
            rolloff = freqs[idx]

        Returns:
            rolloff_khz ≥ 0.0  (0.0 bei Stille oder Fehler).
        """
        # Nächste Zweierpotenz ≤ Signallänge (max 65536 für Geschwindigkeit)
        n_raw = len(audio)
        n_pow = 1
        while n_pow * 2 <= min(n_raw, 65536):
            n_pow *= 2
        n_fft = max(n_pow, 512)

        spectrum = np.abs(np.fft.rfft(audio[:n_fft], n=n_fft)) ** 2
        freqs_hz = np.fft.rfftfreq(n_fft, d=1.0 / sr)

        total = float(spectrum.sum())
        if total < 1e-30:
            return 0.0

        cumsum = np.cumsum(spectrum) / total
        idx_90 = int(np.searchsorted(cumsum, 0.90))
        idx_90 = min(idx_90, len(freqs_hz) - 1)

        return float(freqs_hz[idx_90]) / 1000.0

    def _floor_score(self, noise_floor_db: float) -> float:
        """Skaliert den Rauschboden-Befund auf [0, 1].

        score = clip((THRESHOLD − floor_db) / 40, 0, 1)
        Beispiel: floor = −100 dBFS → score = (−80 − (−100)) / 40 = 0.5
                  floor =  −80 dBFS → score = 0.0 (exakt an Schwelle)
                  floor = −120 dBFS → score = 1.0
        """
        if noise_floor_db >= self.NOISE_FLOOR_THRESHOLD:
            return 0.0
        return float(
            np.clip(
                (self.NOISE_FLOOR_THRESHOLD - noise_floor_db) / 40.0,
                0.0,
                1.0,
            )
        )

    def _bw_score(self, hf_rolloff_khz: float) -> float:
        """Skaliert die Bandbreiten-Messung auf [0, 1].

        score = clip((rolloff − THRESHOLD) / 4, 0, 1)
        Beispiel: rolloff = 20 kHz → score = (20 − 18) / 4 = 0.5
                  rolloff = 18 kHz → score = 0.0 (exakt an Schwelle)
                  rolloff = 22 kHz → score = 1.0
        """
        if hf_rolloff_khz <= self.HF_ROLLOFF_THRESHOLD:
            return 0.0
        return float(
            np.clip(
                (hf_rolloff_khz - self.HF_ROLLOFF_THRESHOLD) / 4.0,
                0.0,
                1.0,
            )
        )


# ---------------------------------------------------------------------------
# Singleton-Accessor (§3.2 — Thread-sicher via Double-Checked Locking)
# ---------------------------------------------------------------------------

_instance: RemasterDetector | None = None
_lock = threading.Lock()


def get_remaster_detector() -> RemasterDetector:
    """Thread-sicherer Singleton-Accessor (Double-Checked Locking).

    Returns:
        Die gemeinsame RemasterDetector-Instanz.
    """
    global _instance
    if _instance is None:  # Schnellpfad ohne Lock
        with _lock:
            if _instance is None:  # Zweiter Check unter Lock (Race Condition sicher)
                _instance = RemasterDetector()
    return _instance


def analyse_remaster(audio: np.ndarray, sr: int) -> RemasterResult:
    """Convenience-Wrapper: erkennt digitale Remaster ohne Klassen-Instantiierung.

    Args:
        audio: float32/64 ndarray, mono oder stereo.
        sr:    Abtastrate in Hz.

    Returns:
        RemasterResult mit Befund (alle Felder sind math.isfinite()).
    """
    return get_remaster_detector().analyse(audio, sr)
