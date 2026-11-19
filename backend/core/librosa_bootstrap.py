"""Librosa-Bootstrap — thread-sichere, idempotente Voll-Initialisierung im Hauptthread.

Root-Fix 2026-08-16 (Spec 24): Im GUI-Prozess crashten librosa-Erstzugriffe aus
mehreren Threads gleichzeitig — bridge-Warmup-Daemon-Thread vs. Hauptthread
(Pre-Analyse/Denker). Symptome im Log:

    - AttributeError: 'function' object has no attribute 'get_call_template'
      (numba-GUfunc-Kompilierung korrumpiert durch parallelen Erstzugriff)
    - KeyError: 'scipy.sparse._construct' (lazy_loader-Import-Race)
    - „chroma_cqt() takes 0 positional arguments" (halb aufgelöstes Modul)

Repro (frischer NUMBA_CACHE_DIR, zwei Threads importieren parallel
librosa.core.constantq / librosa.onset) schlägt reproduzierbar fehl.

Wurzel-Lösung: Alle Submodule und numba-Kompilierungen EINMAL im Hauptthread
vollständig auflösen, BEVOR Worker-Threads starten. Danach sind alle Zugriffe
reine sys.modules-Lookups — kein Import, keine Kompilierung, kein Race.

``ensure_librosa_ready()`` ist idempotent und thread-sicher (Lock + Flag).
Ein Aufruf aus einem Worker-Thread ist ein No-Op, wenn der Hauptthread bereits
fertig war. Ein fehlgeschlagenes Submodul degradiert einzeln (§V6 (copilot-instructions.md)), nie das
Ganze — die DSP-Ersatzpfade der Aufrufer bleiben intakt.
"""

from __future__ import annotations

import importlib
import logging
import threading
import warnings

import numpy as np

logger = logging.getLogger(__name__)

_LIBROSA_LOCK = threading.Lock()
_LIBROSA_READY = False

# Alle in Aurik genutzten librosa-Submodule — in Import-Abhängigkeits-Reihenfolge.
# Jedes Modul löst beim Erstimport lazy_loader-Ketten aus (numba-GUfuncs in
# core.audio/_zc_wrapper, constantq.vqt → pitch.piptrack → util.expand_to).
_LIBROSA_SUBMODULES: tuple[str, ...] = (
    "librosa.core.audio",
    "librosa.core.spectrum",
    "librosa.core.constantq",
    "librosa.core.pitch",
    "librosa.filters",
    "librosa.util",
    "librosa.util.utils",
    "librosa.feature",
    "librosa.onset",
    "librosa.beat",
    "librosa.segment",
)


def _warmup_calls(librosa) -> list[tuple]:
    """Baut die funktionalen Warmup-Aufrufe (numba-Kompilierung erzwingen).

    Jeder Eintrag wird einzeln in try/except ausgeführt (Muster aus
    musical_goals_metrics._warm_up_librosa) — ein Fehler degradiert nur
    diesen einen Pfad, nie die übrigen.
    """
    _dummy = np.zeros(4096, dtype=np.float32)
    _dummy[::4] = 0.1
    _dummy_cqt = np.zeros(int(22050 * 0.5), dtype=np.float32) + 0.1
    # librosa 0.11: Feature-APIs sind keyword-only (y=...) — positional
    # wirft TypeError (Befund 2026-08-16).
    return [
        (librosa.stft, (), {"y": _dummy, "n_fft": 512, "hop_length": 128}),
        (librosa.feature.mfcc, (), {"y": _dummy, "sr": 8000, "n_mfcc": 13}),
        (librosa.feature.spectral_centroid, (), {"y": _dummy, "sr": 8000}),
        (librosa.feature.spectral_rolloff, (), {"y": _dummy, "sr": 8000}),
        (librosa.feature.zero_crossing_rate, (), {"y": _dummy}),
        (librosa.feature.chroma_stft, (), {"y": _dummy, "sr": 8000}),
        (librosa.feature.rms, (), {"y": _dummy}),
        # CQT-Pfad: constantq.vqt → pitch.piptrack → util.expand_to
        (librosa.feature.chroma_cqt, (), {"y": _dummy_cqt, "sr": 22050}),
        (librosa.onset.onset_strength, (), {"y": _dummy, "sr": 8000}),
        (librosa.beat.beat_track, (), {"y": _dummy, "sr": 8000}),
        (librosa.resample, (), {"y": _dummy, "orig_sr": 8000, "target_sr": 22050}),
    ]


def ensure_librosa_ready() -> bool:
    """Löst librosa vollständig und serialisiert auf (idempotent, thread-sicher).

    Returns:
        True wenn der Bootstrap vollständig (oder bereits zuvor) durchlief.
        False nur wenn librosa selbst nicht importierbar ist — dann bleiben
        alle librosa-Nutzer ohnehin auf ihren Import-Fallbacks.
    """
    global _LIBROSA_READY
    if _LIBROSA_READY:
        return True
    with _LIBROSA_LOCK:
        if _LIBROSA_READY:
            return True
        try:
            import librosa
        except ImportError as exc:
            logger.warning("librosa-Bootstrap: librosa nicht verfügbar (%s) — Ersatzpfade aktiv", exc)
            _LIBROSA_READY = True  # nicht endlos wiederholen
            return False

        # 1) Submodule serialisiert auflösen (lazy_loader-Import-Race ausschließen)
        for _sub in _LIBROSA_SUBMODULES:
            try:
                importlib.import_module(_sub)
            except (ImportError, AttributeError, KeyError) as exc:
                logger.warning(
                    "librosa-Bootstrap: Submodul %s nicht ladbar (%s) — DSP-Ersatzpfade aktiv (Spec 24)",
                    _sub,
                    exc,
                )

        # 2) numba-GUfunc-Kompilierung erzwingen — jede Operation einzeln abgesichert
        for _fn, _args, _kwargs in _warmup_calls(librosa):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    _fn(*_args, **_kwargs)
            except Exception as exc:
                logger.warning("librosa-Bootstrap: Warmup %s fehlgeschlagen (%s)", getattr(_fn, "__name__", _fn), exc)

        # 3) EIGENE numba-Kerne erzwingen (Exit-139-Fix 2026-11-19): die
        # njit-Kerne in pghi/warp_kalman/phase_defect_verifier kompilieren
        # lazy beim Erstaufruf — passiert der im Denker-Thread, segfaultet
        # der Prozess (Faulthandler: <no Python frame> + numba-Module).
        _warmup_eigene_numba()

        _LIBROSA_READY = True
        logger.debug("librosa-Bootstrap: vollständig aufgelöst (Hauptthread, vor Worker-Start)")
        return True


def _warmup_eigene_numba() -> None:
    """JIT-Kompilierung der eigenen numba-Kerne im Hauptthread erzwingen.

    Jeder Kern wird einzeln mit Mini-Eingaben kompiliert; Fehler degradieren
    nur diesen Kern (§V6 (copilot-instructions.md)) — die Aufrufer haben jeweils Python-Fallbacks.
    """
    try:
        from backend.core.dsp import pghi as _pghi_mod

        _mag = np.ones((4, 3), dtype=np.float64)
        _phase = np.zeros((4, 3), dtype=np.float64)
        _visited = np.zeros((4, 3), dtype=np.bool_)
        _in_heap = np.zeros((4, 3), dtype=np.bool_)
        _dpt = np.zeros((4, 3), dtype=np.float64)
        _dpo = np.zeros((4, 3), dtype=np.float64)
        _pghi_mod._pghi_heap_kernel_nb(_mag, _phase, _visited, _in_heap, _dpt, _dpo, 0, 0)
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.warning("numba-Bootstrap: pghi-Kern nicht warmgelaufen (%s) — Python-Ersatzpfad bleibt", exc)

    try:
        from backend.core.phase_defect_verifier import _qp_envelope_numba as _qpw

        _env = np.zeros(16, dtype=np.float32)
        _out = np.zeros(16, dtype=np.float32)
        _qpw(_env, 0.1, 0.2, _out)
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.warning("numba-Bootstrap: QP-Hüllen-Kern nicht warmgelaufen (%s) — Python-Ersatzpfad bleibt", exc)

    try:
        from backend.core.dsp.warp_kalman import kalman_smooth_warp as _ksw

        _ksw(np.zeros(16, dtype=np.float64), q=1e-2, r=1e-2)
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.warning("numba-Bootstrap: warp_kalman-Kern nicht warmgelaufen (%s) — Python-Ersatzpfad bleibt", exc)
