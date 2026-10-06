"""SOTA Vocal-Gender-Erkennung — kanonische Fassade (Spec 19, §2.8/§2.11).

Es gibt genau EINEN Detektor-Pfad: die Multi-Evidenz-Fusion des kanonischen
Kerns ``backend.core.vocal_ai_enhancement.GenderDetector``. Diese Datei ist
dessen Fassade für datei- und arraybasierte Aufrufer und enthält selbst KEINE
eigene Klassifikationslogik mehr (§G9 copilot-instructions.md).

Evidenz im Kern (absteigend nach Domänen-Tauglichkeit):

1. **PANNs-AudioSet „Male/Female singing"** (Klassen 32/33) — die einzige
   Geschlechts-Evidenz, die auf GESUNGENEM Material trainiert wurde; geht
   additiv in die Fusion ein (mind. 0,25 Score UND 0,10 Abstand, sonst
   schweigt sie).
2. **F0**: Scanning-Autokorrelation plus pYIN mit Voicing-Confidence (Mauch &
   Dixon 2014) — Vibrato- und oktavfehler-robust, kein Intro-Blindflug.
3. **Formanten F1–F4** via Burg-LPC, gegated auf stimmhafte Frames, mit
   WORLD-Kreuzvalidierung (Morise et al. 2016).
4. **Anatomie-Override** für Contralto/Alt: tiefe F0, aber weiblicher
   Vokaltrakt (F1/F2) → FEMALE statt MALE.

Ein sprachtrainiertes Embedding-Modell (Resemblyzer/LibriSpeech) entscheidet
hier bewusst NICHT mit (§III.11 copilot-instructions.md): Ohne Musik-Fine-Tune
ist es kein zulässiger Richter über Musik/Gesang — und es hat in diesem Pfad
historisch nachweislich nichts zur Entscheidung beigetragen (Embedding wurde
berechnet, aber verworfen; fehlte es, blockierte es die Pitch-Auswertung).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class GenderDetector:
    """Fassade über den kanonischen SOTA-Gender-Kern (Multi-Evidenz-Fusion)."""

    def __init__(self, sample_rate: int = 48000, use_auth_token: Any = None) -> None:
        del use_auth_token  # historischer Resemblyzer-Auth-Parameter — ungenutzt
        self.sample_rate = int(sample_rate)

    # ── Kanonischer Pfad ────────────────────────────────────────────────
    def detect_gender_array(self, audio: Any, sample_rate: int | None = None, panns_tags: Any = None) -> str:
        """Erkennt ``male``/``female``/``child``/``unknown`` aus einem Audio-Array.

        Args:
            audio: mono oder Stereo (channels-first (C, N) bevorzugt, (N, C) wird
                ebenfalls bedient — Stereo-Layout-Invariante).
            sample_rate: Abtastrate; sonst die der Instanz.
            panns_tags: Optionale PANNs-Tags aus der Einmal-Analyse (Shortcut/Fusion).
        """
        sr = int(sample_rate or self.sample_rate)
        _mono = _to_mono(audio)
        if _mono is None or _mono.size < int(0.2 * sr):
            logger.debug("Gender: Signal zu kurz/leer — unknown")
            return "unknown"
        try:
            from backend.core.vocal_ai_enhancement import GenderDetector as _KanonKern

            _chars = _KanonKern(sample_rate=sr).detect(_mono, panns_tags=panns_tags)
            _gender = str(getattr(getattr(_chars, "gender", None), "value", "") or "").lower()
            if _gender in {"male", "female", "child"}:
                logger.debug(
                    "Gender (kanonischer Kern): %s (conf=%.2f, F0=%.0f Hz)",
                    _gender,
                    float(getattr(_chars, "confidence", 0.0) or 0.0),
                    float(getattr(_chars, "fundamental_freq", 0.0) or 0.0),
                )
                return _gender
            logger.debug("Gender: kanonischer Kern meldete '%s' — unknown", _gender or "?")
            return "unknown"
        except Exception as _exc:  # pylint: disable=broad-except
            logger.warning(
                "§19 (19_sota_gender_detection.md) kanonischer Gender-Kern nicht nutzbar (%s) — "
                "eingeschränkter DSP-Ersatzpfad (§V6 copilot-instructions.md)",
                _exc,
            )
            return self._fallback_scan_gender(_mono, sr)

    def detect_gender(self, audio_file: Any, panns_tags: Any = None) -> str:
        """Dateibasierter Wrapper (API-kompatibel) auf denselben kanonischen Pfad."""
        if isinstance(audio_file, np.ndarray):
            return self.detect_gender_array(audio_file, self.sample_rate, panns_tags=panns_tags)
        try:
            from backend.file_import import load_audio_file as _laf

            _ld = _laf(audio_file)
            if _ld is None or _ld.get("audio") is None:
                return "unknown"
            return self.detect_gender_array(_ld["audio"], int(_ld["sr"]), panns_tags=panns_tags)
        except Exception as _exc:  # pylint: disable=broad-except
            logger.warning("Gender-Dateipfad konnte nicht gelesen werden: %s (§V6 copilot-instructions.md)", _exc)
            return "unknown"

    # ── Eingeschränkter Fallback (nur wenn der kanonische Kern nicht ladbar ist) ──
    def _fallback_scan_gender(self, mono: np.ndarray, sr: int) -> str:
        """Scan-F0 über den ganzen Clip (Spec 19 Bug 2/4: nie nur der Anfang).

        Bewusst konservativ: F0 ist als Einzelmerkmal domänenschwach, deshalb
        wird nur bei klarer Lage entschieden — sonst ehrliches ``unknown``
        statt einer Fehlklassifikation (§V7 copilot-instructions.md, Hörordnung §1).
        """
        _f0 = _scan_f0(mono, sr)
        if _f0 <= 0:
            return "unknown"
        if _f0 < 155.0:
            return "male"
        if 200.0 < _f0 < 330.0:
            return "female"
        return "unknown"


# ============================================================
# MODUL-HELFER (Layout, Scan-F0) — domänenneutral, kein Training
# ============================================================


def _to_mono(audio: Any) -> np.ndarray | None:
    """Layout-Normalisierung → float32 mono (Stereo = Mittel beider Kanäle)."""
    if audio is None:
        return None
    try:
        _arr = np.asarray(audio, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if _arr.ndim == 2:
        # channels-first (C, N) ist die Pipeline-Invariante; (N, C) mit kleinem
        # C wird ebenfalls bedient (Stereo-Layout-Invariante).
        if _arr.shape[0] <= 8 and _arr.shape[0] < _arr.shape[1]:
            _arr = np.mean(_arr, axis=0)
        else:
            _arr = np.mean(_arr, axis=1)
    elif _arr.ndim != 1:
        return None
    _clean: np.ndarray = np.nan_to_num(_arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return _clean


def _scan_f0(mono: np.ndarray, sr: int, window_s: float = 2.0, hop_s: float = 1.0) -> float:
    """Median-F0 über 2-s-Fenster (FFT-Autokorrelation) — nie nur der Clip-Anfang.

    Spec 19 (Bug 2/4): Instrumentale Intros ≥ 100 ms blockierten die Detektion;
    deshalb wird über den gesamten Clip gescannt und der Median der stimmhaften
    Fenster gebildet. Deterministisch (§G5 GEBOTE.md).
    """
    _win = max(int(window_s * sr), 256)
    _hop = max(int(hop_s * sr), 1)
    if mono.size < _win:
        _win = mono.size
        _hop = max(_win // 2, 1)
    _est: list[float] = []
    for _start in range(0, max(mono.size - _win + 1, 1), _hop):
        _seg = mono[_start : _start + _win]
        if _seg.size < 256:
            break
        if float(np.sqrt(np.mean(_seg**2))) < 1e-4:
            continue  # Stille/Pause überspringen
        _f0 = _autocorr_f0(_seg, sr)
        if _f0 > 0:
            _est.append(_f0)
    return float(np.median(_est)) if _est else 0.0


def _autocorr_f0(seg: np.ndarray, sr: int, fmin: float = 60.0, fmax: float = 700.0) -> float:
    """FFT-Autokorrelation mit parabolischer Peak-Interpolation (O(N log N))."""
    _seg = seg - float(np.mean(seg))
    if _seg.size < 256 or not np.any(_seg):
        return 0.0
    _n = int(2 ** int(np.ceil(np.log2(2 * _seg.size))))
    _spec = np.fft.rfft(_seg, n=_n)
    _acf = np.fft.irfft(_spec * np.conj(_spec), n=_n)[: _seg.size]
    if _acf.size < 4 or _acf[0] <= 0:
        return 0.0
    _lo = max(int(sr / fmax), 1)
    _hi = min(int(sr / fmin), _acf.size - 2)
    if _hi <= _lo:
        return 0.0
    _peak = int(np.argmax(_acf[_lo:_hi])) + _lo
    # Nur echte Periodizität akzeptieren (normiert auf die Energie bei Lag 0)
    if _acf[_peak] < 0.30 * _acf[0]:
        return 0.0
    _y0, _y1, _y2 = float(_acf[_peak - 1]), float(_acf[_peak]), float(_acf[_peak + 1])
    _denom = 2.0 * (2.0 * _y1 - _y0 - _y2)
    _shift = 0.0 if abs(_denom) < 1e-12 else (_y2 - _y0) / _denom
    _lag = _peak + _shift
    return float(sr / _lag) if _lag > 0 else 0.0
