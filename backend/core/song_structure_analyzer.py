"""
SongStructureAnalyzer — §2.52b [RELEASE_MUST]
=============================================

Segment-bewusste Pipeline: Erkennt Intro/Vers/Chorus/Bridge/Outro/Instrumental
und liefert segment-adaptive Strength-Skalare für jede Phase.

Spec: 02_pipeline_architecture.md §2.52b (v10.0.0)
"""

import logging
import threading
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

_instance: "SongStructureAnalyzer | None" = None
_lock = threading.Lock()


def get_song_structure_analyzer() -> "SongStructureAnalyzer":
    """Singleton-Getter (thread-safe, double-checked locking)."""
    global _instance  # pylint: disable=global-statement
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = SongStructureAnalyzer()
    return _instance


# ---------------------------------------------------------------------------
# Datenmodell
# ---------------------------------------------------------------------------


@dataclass
class SongSegment:
    """Ein erkanntes Song-Segment (§2.52b)."""

    start_s: float
    end_s: float
    label: str  # "intro", "verse", "chorus", "bridge", "outro", "instrumental"
    energy_level: float  # [0, 1] normiert auf Ø RMS
    has_vocals: bool
    is_climax: bool


# Strength-Skalare pro Segment-Typ (§2.52b Tabelle) — bounded [0.70, 1.30]
_STRENGTH_SCALARS: dict[str, dict[str, float]] = {
    "verse": {
        "nr_strength": 1.15,
        "dereverb": 1.10,
        "default": 1.05,
    },
    "chorus": {
        "nr_strength": 0.85,
        "compression": 0.70,
        "default": 0.90,
    },
    "intro": {
        "default": 1.00,
    },
    "outro": {
        "default": 1.00,
    },
    "bridge": {
        "default": 0.95,
    },
    "instrumental": {
        "nr_strength": 1.00,
        "default": 1.00,
    },
    "silence": {
        "default": 0.70,  # nur passiv
    },
    "unknown": {
        "default": 1.00,
    },
}

# Zusätzliche Climax-Overrides (überschreiben Segment-Label)
_CLIMAX_SCALAR = 0.85

# §SOTA-Upgrade 2026-09-17 (Wiederholungs-Evidenz, deterministisch):
# Der Refrain ist per Definition der WIEDERKEHRENDE Abschnitt — ohne
# Wiederholungs-Nachweis kann ein Analyzer keinen Chorus erkennen
# (Produktionsbefund: Test-Track-225s — 0 Chorus/0 Klimax trotz klarem
# Refrain-Motiv bei ≈44/100/156/192 s). Segment-MITTELWERTE sind dafür
# unbrauchbar (schlüsseldominiert, alle Korrelationen ≈0,65 — gemessen);
# die Wiederholung zeigt sich erst auf 8-s-FENSTER-Niveau.
_WINDOW_LEN_S = 8.0  # Fensterlänge für Wiederholungs-Scan
_WINDOW_HOP_S = 4.0  # Fenster-Schritt
_WINDOW_CORR = 0.70  # Kosinus-Ähnlichkeit zweier Fenster-Mittel-Chromas
_CHORUS_MIN_REPEATS = 2  # wiederholte Fenster je Segment → Chorus-Kandidat
_INTRO_MAX_FRAC = 0.15  # Intro darf max. 15 % der Songdauer sein
_OUTRO_MAX_FRAC = 0.15  # Outro darf max. 15 % der Songdauer sein
_CLIMAX_TOP_FRAC = 0.98  # Klimax = Chorus mit p90-RMS ≥ 98 % des Song-Maximums


# ---------------------------------------------------------------------------
# §SOTA-Upgrade 2026-09-17: Wiederholungs-Evidenz (modul-frei testbar)
# ---------------------------------------------------------------------------


def _segment_index_at(segments: list[SongSegment], time_s: float) -> int:
    """Index des Segments, das time_s enthält (0-basiert, deterministisch)."""
    for _i, _seg in enumerate(segments):
        if _seg.start_s <= time_s < _seg.end_s:
            return _i
    return len(segments) - 1 if segments else 0


def _window_repetition_counts(
    chroma_norm: np.ndarray,
    hop: int,
    sr: int,
    segments: list[SongSegment],
) -> list[int]:
    """Wiederholte 8-s-Chroma-Fenster je Segment (Schritt 4 s, Kosinus ≥ 0,70).

    Ein Fenster gilt als „wiederholt“, wenn ein stark ähnliches Fenster in
    einem ANDEREN Segment liegt — der Refrain ist per Definition der
    wiederkehrende Abschnitt. Determinismus: rein numpy (§G5 (GEBOTE.md)).
    """
    _n_seg = len(segments)
    if _n_seg == 0:
        return []
    _win_len = max(1, int(_WINDOW_LEN_S * sr / hop))
    _win_hop = max(1, int(_WINDOW_HOP_S * sr / hop))
    _win_starts = list(range(0, max(chroma_norm.shape[1] - _win_len, 0) + 1, _win_hop))
    _win_means: list[np.ndarray] = []
    for _w0 in _win_starts:
        _wm = chroma_norm[:, _w0 : _w0 + _win_len].mean(axis=1)
        _wn = float(np.linalg.norm(_wm)) + 1e-12
        _win_means.append((_wm / _wn).astype(np.float32))
    _win_times = [float(_w0 * hop / sr) for _w0 in _win_starts]
    _win_repeated = [False] * len(_win_starts)
    for _i in range(len(_win_starts)):
        for _j in range(_i + 1, len(_win_starts)):
            if float(np.dot(_win_means[_i], _win_means[_j])) < _WINDOW_CORR:
                continue
            if _segment_index_at(segments, _win_times[_i]) != _segment_index_at(segments, _win_times[_j]):
                _win_repeated[_i] = True
                _win_repeated[_j] = True
    _repeats = [0] * _n_seg
    for _wi, _rep in enumerate(_win_repeated):
        if _rep:
            _repeats[_segment_index_at(segments, _win_times[_wi])] += 1
    return _repeats


# ---------------------------------------------------------------------------
# Hauptklasse
# ---------------------------------------------------------------------------


class SongStructureAnalyzer:
    """Erkennt Song-Segmente und liefert segment-adaptive Strength-Skalare."""

    def analyze_structure(
        self,
        audio: np.ndarray,
        sr: int,
        panns_singing_confidence: float = 0.0,
        vocal_scorer=None,
    ) -> list[SongSegment]:
        """Analysiert die Song-Struktur via librosa Boundary-Erkennung.

        Args:
            audio: Float32-Audio (mono oder stereo).
            sr:    Sample-Rate in Hz.
            panns_singing_confidence: Ø PANNs Singing-Score für das Stück (global).
            vocal_scorer: Optionaler Callable (mono, sr, t0_s, t1_s) -> float | None
                — §SOTA-Analogie-Korrektur 2026-09-17 (ANA-2): die definierende
                per-Segment-Evidenz (PANNs-Singing) ersetzt den
                Flatness-Proxy; None ⇒ DSP-Proxy unverändert.

        Returns:
            Liste von SongSegment-Objekten, sortiert nach start_s.
            Laufzeit: ≤ 2 s / Minute Audio (Pflicht §2.52b).
        """
        try:
            return self._analyze_librosa(audio, sr, panns_singing_confidence, vocal_scorer=vocal_scorer)
        except Exception as exc:
            logger.warning(
                "SongStructureAnalyzer.analyze_structure fehlgeschlagen: %s — Ersatzpfad: single segment", exc
            )
            duration_s = len(audio[0] if audio.ndim == 2 else audio) / sr
            return [
                SongSegment(
                    start_s=0.0,
                    end_s=float(duration_s),
                    label="unknown",
                    energy_level=0.5,
                    has_vocals=panns_singing_confidence >= 0.35,
                    is_climax=False,
                )
            ]

    def _analyze_librosa(
        self,
        audio: np.ndarray,
        sr: int,
        panns_singing_confidence: float,
        vocal_scorer=None,
    ) -> list[SongSegment]:
        import librosa  # pylint: disable=import-outside-toplevel

        # Mono für Analyse
        if audio.ndim == 2:
            mono = audio.mean(axis=0) if audio.shape[0] == 2 else audio.mean(axis=1)
        else:
            mono = audio
        mono = np.nan_to_num(mono.astype(np.float32), nan=0.0)

        duration_s = len(mono) / sr

        # Boundary-Erkennung: MFCC + Chroma (≤ 2 s / min)
        hop = 512
        mfcc = librosa.feature.mfcc(y=mono, sr=sr, n_mfcc=12, hop_length=hop)
        chroma = librosa.feature.chroma_stft(y=mono, sr=sr, hop_length=hop)
        features = np.vstack([mfcc, chroma])  # (24, T)

        # Boundary-Erkennung: SSM + Checkerboard-Novelty (Foote 2000) — die
        # kanonische Methode aus dsp/ssm_segmentation.py, geteilt mit dem
        # §2.17-MusicalStructureAnalyzer (§SOTA-Analogie-Korrektur 2026-09-17:
        # die agglomerative k-Heuristik „1 Grenze / 30 s“ war dieselbe
        # Fehlerklasse wie die Label-Heuristik — die definierende
        # Novelty-Evidenz existierte bereits im §2.17-Analysator,
        # §V7 (copilot-instructions.md): EINE Grenz-Methode pro Rolle).
        _hop_ssm = max(1, int(sr * 0.5))
        boundary_times: np.ndarray
        try:
            _chroma_ssm = librosa.feature.chroma_cqt(y=mono, sr=sr, hop_length=_hop_ssm, bins_per_octave=36).astype(
                np.float32
            )
            from backend.core.dsp.ssm_segmentation import (  # pylint: disable=import-outside-toplevel
                ssm_boundaries_from_chroma,
            )

            _bounds_ssm, _ = ssm_boundaries_from_chroma(_chroma_ssm, _hop_ssm, sr, duration_s)
            if len(_bounds_ssm) < 3 or (len(_bounds_ssm) - 1) > 60:
                raise RuntimeError("SSM liefert keine brauchbaren Grenzen — agglomerativer Ersatzpfad")
            boundary_times = np.asarray([float(_b) / sr for _b in _bounds_ssm], dtype=np.float64)
        except Exception:
            # Fallback: bestehende agglomerative/uniform-Heuristik
            k = max(2, min(12, int(duration_s / 30) + 1))
            try:
                boundaries = librosa.segment.agglomerative(features, k)  # type: ignore[attr-defined]
                boundary_times = librosa.frames_to_time(boundaries, sr=sr, hop_length=hop)
            except Exception:
                # Fallback: gleichmäßige Aufteilung
                n = max(2, int(duration_s / 30))
                boundary_times = np.linspace(0, duration_s, n + 1)[1:-1]

        # Segment-Grenzen aufbauen
        times = [0.0, *boundary_times.tolist(), duration_s]
        times = sorted(set(times))

        # RMS-Profil für Energie
        rms = librosa.feature.rms(y=mono, hop_length=hop)[0]
        rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)

        # §SOTA-Upgrade 2026-09-17: normiertes Chroma für die
        # Wiederholungs-Evidenz (Fenster-Mittelwerte).
        chroma_norm = librosa.util.normalize(chroma, norm=2, axis=0)

        global_rms_mean = float(np.mean(rms) + 1e-12)

        segments = []
        for i in range(len(times) - 1):
            t0, t1 = times[i], times[i + 1]
            if t1 - t0 < 0.5:
                continue

            # Segment-Energie
            seg_mask = (rms_times >= t0) & (rms_times < t1)
            seg_rms = rms[seg_mask]
            energy_norm = float(np.clip(np.mean(seg_rms) / global_rms_mean, 0.0, 2.0) / 2.0)

            # Vocal-Aktivität im Segment (Proxy: spectral flatness niedrig = tonales Material)
            i0 = max(0, int(t0 * sr))
            i1 = min(len(mono), int(t1 * sr))
            seg_audio = mono[i0:i1]
            has_vocals = self._estimate_vocal_activity(
                seg_audio, sr, panns_singing_confidence, scorer=vocal_scorer, t0=float(t0), t1=float(t1)
            )

            segments.append(
                SongSegment(
                    start_s=float(t0),
                    end_s=float(t1),
                    label="unknown",  # Labels folgen im Wiederholungs-Durchgang
                    energy_level=float(energy_norm),
                    has_vocals=has_vocals,
                    is_climax=False,
                )
            )

        # ----------------------------------------------------------------
        # §SOTA-Upgrade 2026-09-17: Wiederholungs-Evidenz auf FENSTER-Niveau.
        # 8-s-Chroma-Fenster (Schritt 4 s) werden paarweise verglichen
        # (Kosinus); Fenster mit starkem Match AUSSERHALB des eigenen
        # Segments gelten als „wiederholt“. Der Refrain ist der Abschnitt
        # mit ≥ 2 wiederholten Fenstern (Hörordnung-konform: strukturelle
        # Wiederholung statt nur Position/Energie).
        # ----------------------------------------------------------------
        _n_seg = len(segments)
        _repeats_per_seg = _window_repetition_counts(chroma_norm, hop, sr, segments)
        _seg_p90: list[float] = []
        _seg_max_p90 = 0.0
        for _seg in segments:
            _sm = (rms_times >= _seg.start_s) & (rms_times < _seg.end_s)
            _p90 = float(np.percentile(rms[_sm], 90)) if np.any(_sm) else 0.0
            _seg_p90.append(_p90)
            _seg_max_p90 = max(_seg_max_p90, _p90)
        _energy_median = float(np.median([s.energy_level for s in segments]))
        for _i, _seg in enumerate(segments):
            _seg.label = self._assign_label(
                _i,
                _n_seg,
                _seg.energy_level,
                _seg.has_vocals,
                duration_s,
                _seg.start_s,
                matches=(_repeats_per_seg[_i] // _CHORUS_MIN_REPEATS)
                if _repeats_per_seg[_i] >= _CHORUS_MIN_REPEATS
                else 0,
                energy_median=_energy_median,
                seg_duration=_seg.end_s - _seg.start_s,
            )
            _seg.is_climax = bool(
                _seg.has_vocals
                and _seg.label == "chorus"
                and _seg_p90[_i] >= _CLIMAX_TOP_FRAC * max(_seg_max_p90, 1e-12)
            )

        return (
            segments
            if segments
            else [
                SongSegment(
                    start_s=0.0,
                    end_s=float(duration_s),
                    label="unknown",
                    energy_level=0.5,
                    has_vocals=panns_singing_confidence >= 0.35,
                    is_climax=False,
                )
            ]
        )

    def _estimate_vocal_activity(
        self,
        seg_audio: np.ndarray,
        sr: int,
        panns_confidence: float,
        scorer=None,
        t0: float = 0.0,
        t1: float = 0.0,
    ) -> bool:
        """Vokal-Aktivitäts-Schätzer: per-Segment-Scorer (PANNs) als
        definierende Evidenz, spectral flatness + globale PANNs-Konfidenz als
        DSP-Proxy (§SOTA-Analogie-Korrektur 2026-09-17, ANA-2)."""
        if scorer is not None:
            try:
                _score = scorer(seg_audio, sr, t0, t1)
                if isinstance(_score, (int, float)):
                    _s = float(_score)
                    if np.isfinite(_s):
                        return _s >= 0.30
            except Exception as _scorer_exc:
                logger.debug("song_structure_analyzer.py::vocal_scorer Ersatzpfad: %s", _scorer_exc)
        if len(seg_audio) < 512:
            return panns_confidence >= 0.35
        try:
            from scipy.signal import welch  # pylint: disable=import-outside-toplevel

            _, psd = welch(seg_audio, fs=sr, nperseg=min(512, len(seg_audio)))
            psd = psd + 1e-12
            # Spectral flatness: hoch = rauschähnlich (kein Vokal); niedrig = tonal (Vokal)
            flatness = float(np.exp(np.mean(np.log(psd))) / (np.mean(psd) + 1e-12))
            # Vokal typisch: flatness < 0.15 UND globalem PANNs-Vertrauen
            return flatness < 0.20 and panns_confidence >= 0.20
        except Exception as e:
            logger.warning("song_structure_analyzer.py::_estimate_vocal_activity Ersatzpfad: %s", e)
            return panns_confidence >= 0.35

    def _assign_label(
        self,
        idx: int,
        n_segments: int,
        energy: float,
        has_vocals: bool,
        duration_s: float,
        t0: float,
        matches: int = 0,
        energy_median: float = 0.5,
        seg_duration: float = 0.0,
    ) -> str:
        """Label-Zuweisung: Position + Energie + WIEDERHOLUNGS-Evidenz
        (§SOTA-Upgrade 2026-09-17). Der Refrain ist per Definition der
        wiederkehrende Abschnitt — matches ≥ 1 + Energie ≥ Median ⇒ chorus.
        Intro/Outro nur für KURZE Rand-Segmente (vorher wurde das gesamte
        erste Segment als „intro“ etikettiert, Produktionsbefund: 42 s „intro“).
        """
        relative_pos = idx / max(1, n_segments - 1)  # [0, 1]

        if not has_vocals:
            return "instrumental"

        # Intro/Outro nur für das ERSTE/letzte Segment UND kurze Dauer
        # (§SOTA-Analogie-Korrektur 2026-09-17: relative_pos ≥ 0,85
        # etikettierte vorletzte Segmente als outro und verschluckte den
        # 156-s-Refrain — Positionsregeln ohne Längen-/Klammer-Evidenz sind
        # dieselbe Fehlerklasse wie die alte Label-Heuristik).
        if idx == 0 and seg_duration <= _INTRO_MAX_FRAC * duration_s:
            return "intro"
        if idx == n_segments - 1 and seg_duration <= _OUTRO_MAX_FRAC * duration_s:
            return "outro"

        # Wiederholungs-Evidenz: Chorus = wiederkehrend + energiereich
        # (strikt über dem Median — wiederholte STROPHEN bleiben Strophen;
        # Median-Gleichstand ist kein Energie-Argument, Produktionsbefund
        # Test-Track: Segment 116,6–155,1 mit en=Median wurde fälschlich chorus).
        if matches >= 1:
            if energy > energy_median:
                return "chorus"
            return "verse"

        # Einmalige Abschnitte: kontrastierender Mittelteil = Bridge.
        if 0.25 <= relative_pos <= 0.80 and energy < energy_median:
            return "bridge"
        return "verse"

    def get_strength_scalar(
        self,
        segment: SongSegment | None,
        phase_type: str = "default",
    ) -> float:
        """Gibt den Strength-Skalar [0.70, 1.30] für ein Segment zurück.

        Args:
            segment:    Aktuelles SongSegment (None → 1.0).
            phase_type: "nr_strength", "dereverb", "compression", oder "default".

        Returns:
            Strength-Skalar; immer bounded [0.70, 1.30].
        """
        if segment is None:
            return 1.0

        # Climax überschreibt Label
        if segment.is_climax:
            scalar = _CLIMAX_SCALAR
        else:
            label_scalars = _STRENGTH_SCALARS.get(segment.label, _STRENGTH_SCALARS["unknown"])
            scalar = label_scalars.get(phase_type, label_scalars.get("default", 1.0))

        # Hard-Bound (§2.52b Invariante)
        return float(np.clip(scalar, 0.70, 1.30))

    def find_segment_at(
        self,
        segments: list[SongSegment],
        time_s: float,
    ) -> SongSegment | None:
        """Gibt das Segment zurück, in dem time_s liegt."""
        for seg in segments:
            if seg.start_s <= time_s < seg.end_s:
                return seg
        # Fallback: letztes Segment
        return segments[-1] if segments else None
