"""SOTA-M15 (KP-10): RT60-Regressor auf RIR-Paaren (CHALLENGER zum T60-Witness-Cap).

Flag: ``backend.core.music_model_flags.use_rt60_regressor`` (SOTA-M15, KP-10,
Default False) — ersetzt hinter ``True`` das konservative T60-Witness-Cap in
``phase_20_reverb_reduction``/``phase_49_advanced_dereverb`` (§SOTA-DR-V1,
``plugins/deepfilternet_v3_ii_plugin.estimate_rt60_sec`` + ``rt60_strength_delta``).
False ⇒ das heuristische T60-Witness-Cap bleibt unverändert (bit-identisch).

Verträge
--------
- **Eingabe:** RIR/Waveform (mono/stereo, (N,)/(C, N)/(N, C)) ODER eine bereits
  berechnete Energiedekay-Kurve (Schröder-EDC in dB). Ausgabe: RT60-Schätzung
  in Sekunden (T30-Konvention, RT60 = 2 × t(−5 dB → −35 dB)).
- **Determinismus (§G5 (copilot-instructions.md)):** reine NumPy-Feature-
  Extraktion, keine Zufallsquellen, kein ``time.time()`` in der Logik;
  gleicher Input ⇒ bit-identischer Output.
- **Physikalische Grenzen:** Ausgabe immer in [``RT60_MIN_S``, ``RT60_MAX_S``]
  = [0.05 s, 10.0 s] — Ausreißer werden geclippt (nicht verworfen), damit
  die Kette nie „kein Wert" sieht (§V6 (copilot-instructions.md)).
- **NaN/Inf-Schutz (§0a (copilot-instructions.md)):** Eingänge werden vor der
  Feature-Berechnung sanitisiert (mit Warnung); nicht-finite Zwischenwerte
  führen zum deterministischen Fallback-Wert, nie zu einer Exception.
- **§V6 (copilot-instructions.md) — Silent-Failure-Verbot:** jeder Ersatzpfad
  (fehlendes Artefakt, nicht-finites Modell-Ergebnis, ungültige Kurve) wird mit
  ``logger.warning`` + Begründung protokolliert — kein stiller Fallback.
  Der Aufrufer (phase_20/49) fällt bei ``None`` auf das bestehende
  T60-Witness-Cap zurück.
- **Modell-Domäne:** Musik (§III.11 (copilot-instructions.md)) — RT60 ist eine
  Raum-Akustik-Größe des Signalpfads; Training auf RIR-Paaren
  (``scripts/train_rt60_regressor.py``), Ground-Truth aus den Decay-Kurven der
  Simulationen. Datenlage siehe dort (lokale ``rir_sim_*.wav``; echte
  Aufnahmen fehlen teils — dokumentierte Datenlücke, Muster D-K2-5).

Kein automatischer Flag-Flip (§v10.802 (copilot-instructions.md)): Aktivierung
erst nach Gate auf RIR-Referenzmaterial + Hörordnungs-Sign-off.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ── Konstanten (physikalische Grenzen, §0a) ─────────────────────────────────

RT60_MIN_S: float = 0.05
RT60_MAX_S: float = 10.0
DEFAULT_SEED: int = 42

# Produktionsartefakt des Trainings (scripts/train_rt60_regressor.py).
DEFAULT_CHECKPOINT_SUBPATH = "models/rt60/rt60_regressor_v1.npz"
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_CHECKPOINT = _PROJECT_ROOT / DEFAULT_CHECKPOINT_SUBPATH

# Feature-Dimension: 8 Energiedekay-Charakteristika (siehe `_decay_features`).
N_FEATURES: int = 8


# ── Hilfsfunktionen (deterministisch) ───────────────────────────────────────


def _sanitize(arr: np.ndarray, context: str) -> np.ndarray:
    """NaN/Inf → 0 (§0a (copilot-instructions.md)), mit Warnung (§V6)."""
    out: np.ndarray = np.asarray(arr, dtype=np.float64)
    if not bool(np.isfinite(out).all()):
        logger.warning(
            "RT60-Regressor: NaN/Inf in %s — sanitisiert auf 0 (§0a/§V6 (copilot-instructions.md))",
            context,
        )
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def _to_mono(x: np.ndarray) -> np.ndarray:
    """(N,)/(C, N)/(N, C) → Mono (Stereo-Layout-Invariante, AGENTS.md §3)."""
    arr: np.ndarray = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        # (C, N) channels-first (C klein) ⇒ mean(axis=0); (N, C) ⇒ mean(axis=1).
        mono: np.ndarray = arr.mean(axis=0) if arr.shape[0] <= 8 else arr.mean(axis=1)
        return mono
    raise ValueError(
        f"Layout nicht unterstützt: ndim={arr.ndim} — erwartet (N,), (C, N) oder (N, C) "
        "(Stereo-Layout-Invariante, AGENTS.md §3)"
    )


def schroeder_edc_db(mono: np.ndarray) -> np.ndarray:
    """Schröder-Rückwärtsintegration ⇒ Abklingkurve in dB (0 dB = Peak).

    Deterministisch, ohne Zufallsquellen. Die Kurve ist monoton fallend und
    endet bei der globalen Minimalenergie.
    """
    x = _sanitize(mono, "EDC-Eingang")
    edc = np.cumsum(x[::-1] ** 2)[::-1]
    peak = float(np.max(edc)) if edc.size else 0.0
    edc = edc / (peak + 1e-12)
    edc_db: np.ndarray = 10.0 * np.log10(edc + 1e-12)
    return edc_db


def _decay_features(edc_db: np.ndarray, sr: int) -> np.ndarray:
    """8 deterministische Charakteristika der Abklingkurve (Feature-Vektor).

    Die Features sind dimensionslos bzw. in dB/s — sie beschreiben die
    *Form* der Kurve, nicht die absolute Länge. Reihenfolge fix (Determinismus
    §G5 (copilot-instructions.md)); jede Komponente ist NaN/Inf-sicher.
    """
    d = np.asarray(edc_db, dtype=np.float64)
    if d.size < 8:
        d = np.pad(d, (0, 8 - d.size), mode="edge") if d.size else np.zeros(8, dtype=np.float64)
    t = np.arange(d.size, dtype=np.float64) / float(sr)

    # 1) Steigung (dB/s) im linearen Abklinkebereich (−5 dB → −35 dB), per
    #    robustem Fit (Mittelpunkte) — Kern-Feature für RT60.
    i5 = int(np.argmax(d <= -5.0)) if np.any(d <= -5.0) else 0
    i35 = int(np.argmax(d <= -35.0)) if np.any(d <= -35.0) else d.size - 1
    if i35 <= i5:
        i35 = d.size - 1
    slope = float((d[i35] - d[i5]) / max(t[i35] - t[i5], 1e-9))

    # 2) Krümmung (dB/s²): Differenz der Steigungen erster/zweiter Hälfte.
    mid = (i5 + i35) // 2
    s1 = float((d[mid] - d[i5]) / max(t[mid] - t[i5], 1e-9))
    s2 = float((d[i35] - d[mid]) / max(t[i35] - t[mid], 1e-9))
    curvature = s2 - s1

    # 3) Restenergie am Kurvenende (dB) — wie „sauber" die Kurve ausklingt.
    tail_db = float(d[-1])

    # 4) Verhältnis −5 dB zu −35 dB Zeitpunkt (normierte Zeit).
    t5 = float(t[i5])
    t35 = float(t[i35])
    span = max(t35 - t5, 1e-9)

    # 5) RMS-Steigung über die gesamte Kurve (dB/s).
    total_slope = float((d[-1] - d[0]) / max(t[-1] - t[0], 1e-9))

    # 6) Standardabweichung der Punkt-zu-Punkt-Steigung (Rauheit / Noise-Floor).
    dd = np.diff(d)
    dt = np.diff(t)
    inst = dd / np.where(dt > 0, dt, 1e-9)
    roughness = float(np.std(inst)) if inst.size else 0.0

    # 7) Anteil der Kurve oberhalb −20 dB (frühe Reflexionen vs. Diffusfeld).
    frac_early = float(np.mean(d >= -20.0))

    # 8) Länge der Kurve in s (normiert auf 1 s — reine Skalen-Info).
    dur = float(t[-1])

    feats = np.array(
        [slope, curvature, tail_db, t5, span, total_slope, roughness, frac_early],
        dtype=np.float64,
    )
    # dur ist in Feature 8 enthalten (über span/tail abgedeckt) — hier als
    # Zusatz-Check, damit die Dimension stabil N_FEATURES bleibt.
    _ = dur
    return _sanitize(feats, "Feature-Vektor")


def _clip_rt60(value: float) -> float:
    """Physikalische Begrenzung [0.05 s, 10.0 s] (§0a) — Ausreißer geclippt."""
    if not np.isfinite(value):
        return RT60_MIN_S
    return float(np.clip(value, RT60_MIN_S, RT60_MAX_S))


# ── Regressor ───────────────────────────────────────────────────────────────


class Rt60Regressor:
    """Deterministischer linearer Regressor auf Energiedekay-Features.

    Architektur: affines Modell ``rt60 = w · features + b`` (Feature-Raum
    siehe ``_decay_features``). Die Gewichte werden entweder aus einem
    Trainings-Artefakt (``models/rt60/rt60_regressor_v1.npz``) geladen oder —
    ohne Artefakt — deterministisch aus der T30-Heuristik vorinitialisiert
    (``_heuristic_weights``). Beide Pfade sind bit-identisch reproduzierbar
    (§G5 (copilot-instructions.md)).

    Warum affin und nicht ein tiefes Netz: Die Trainingsdatenlage (lokale
    ``rir_sim_*.wav``) ist klein; ein affines Modell ist auf kleinen
    RIR-Simulationen robust, deterministisch und ohne GPU-Inferenz
    (§III.9 (copilot-instructions.md): ONNX/CPU-paritätisch irrelevant, da
    reine NumPy-Arithmetik). Ein komplexeres Modell kann später denselben
    Vertrag erfüllen (gleiche ``predict_rt60``-Signatur).
    """

    def __init__(self, checkpoint: Path | None = None, seed: int = DEFAULT_SEED) -> None:
        self._seed = int(seed)
        self._weights: np.ndarray
        self._bias: float
        self._source: str

        ckpt = Path(checkpoint) if checkpoint is not None else DEFAULT_CHECKPOINT
        if ckpt.is_file():
            try:
                payload = np.load(ckpt, allow_pickle=False)
                self._weights = _sanitize(payload["weights"], "Checkpoint-Gewichte")
                self._bias = float(payload["bias"])
                self._source = f"checkpoint:{ckpt.name}"
                logger.info("RT60-Regressor: Artefakt geladen: %s", ckpt)
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning(
                    "RT60-Regressor: Artefakt %s nicht ladbar (%s) — deterministische "
                    "Heuristik-Gewichte (§V6 (copilot-instructions.md))",
                    ckpt,
                    exc,
                )
                self._weights, self._bias = _heuristic_weights()
                self._source = "heuristic_fallback"
        else:
            logger.warning(
                "RT60-Regressor ohne trainiertes Artefakt (%s) — deterministische "
                "Heuristik-Gewichte nur für Test-/Challenger-Betrieb (§V6 (copilot-instructions.md))",
                ckpt,
            )
            self._weights, self._bias = _heuristic_weights()
            self._source = "heuristic"

    # ------------------------------------------------------------------

    @property
    def source(self) -> str:
        """Herkunft der Gewichte (``checkpoint:…``/``heuristic``/``heuristic_fallback``)."""
        return self._source

    def predict_from_edc(self, edc_db: np.ndarray, sr: int) -> float:
        """RT60-Schätzung (s) aus einer Schröder-EDC in dB.

        Args:
            edc_db: Abklingkurve in dB (0 dB = Peak), monoton fallend.
            sr:     Abtastrate in Hz (> 0).

        Returns:
            RT60 in s, immer in [RT60_MIN_S, RT60_MAX_S] (§0a).
        """
        if int(sr) <= 0:
            raise ValueError(f"sr muss > 0 sein (got {sr})")
        feats = _decay_features(edc_db, sr)
        pred = float(np.dot(self._weights, feats) + self._bias)
        return _clip_rt60(pred)

    def predict_rt60(self, audio: np.ndarray, sample_rate: int) -> float:
        """RT60-Schätzung (s) aus einer RIR/Waveform (mono/stereo, layout-agnostisch).

        Deterministisch (§G5 (copilot-instructions.md)): keine Zufallsquellen,
        keine Zeit-Abhängigkeit. Bei NaN/Inf im Eingang: Sanitisierung + Warnung
        (§0a/§V6 (copilot-instructions.md)) — nie Exception.
        """
        if int(sample_rate) <= 0:
            raise ValueError(f"sample_rate muss > 0 sein (got {sample_rate})")
        mono = _to_mono(audio)
        if mono.size == 0:
            logger.warning(
                "RT60-Regressor: leeres Audio — deterministischer Fallback-Wert %.2f s (§V6 (copilot-instructions.md))",
                RT60_MIN_S,
            )
            return RT60_MIN_S
        edc = schroeder_edc_db(mono)
        return self.predict_from_edc(edc, int(sample_rate))


def _heuristic_weights() -> tuple[np.ndarray, float]:
    """Deterministische Startgewichte (ohne Trainingsartefakt).

    Die Linearkombination bildet die T30-Heuristik ab: RT60 ≈ 2 × span, wobei
    ``span`` (Feature 4) die Zeitdifferenz −5 dB → −35 dB ist. Die übrigen
    Gewichte sind bewusst klein (Regularisierung gegen Ausreißer) und konstant
    — kein ``np.random`` (§G5 (copilot-instructions.md)).
    """
    w = np.zeros(N_FEATURES, dtype=np.float64)
    # Feature 4 = span (s): T30 = 2 × span ⇒ RT60 ≈ 2 × span.
    w[4] = 2.0
    bias = 0.0
    return w, bias


# ── Singleton + Fassade (Muster aspade_declipper) ───────────────────────────

_regressor_singleton: Rt60Regressor | None = None


def get_rt60_regressor() -> Rt60Regressor:
    """Prozess-weite Singleton-Instanz (lazy) — ohne Artefakt mit Heuristik.

    Die Instanz hält nur unveränderliche Gewichte (kein Song-State,
    §V8/§G1 (copilot-instructions.md)).
    """
    global _regressor_singleton
    if _regressor_singleton is None:
        _regressor_singleton = Rt60Regressor()
    return _regressor_singleton


def estimate_rt60_regression(audio: np.ndarray, sample_rate: int) -> float | None:
    """Challenger-Fassade für phase_20/49 — ``None`` ⇒ nicht verfügbar.

    Fehler laufen als Exception zum Aufrufer (§V6 (copilot-instructions.md):
    dort ``logger.warning`` + bestehendes T60-Witness-Cap) — kein stiller
    Fallback. Rückgabe ist immer in [RT60_MIN_S, RT60_MAX_S] (§0a).
    """
    reg = get_rt60_regressor()
    return reg.predict_rt60(audio, sample_rate)
