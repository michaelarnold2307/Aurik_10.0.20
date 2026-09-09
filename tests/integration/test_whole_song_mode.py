"""§v10.720 (2026-09-08, Lücke-4 Stufe 1): Ganzsong-Modus — Integrationstest.

End-to-End auf 121 s synthetischem Audio (FAST-Modus) mit BEGRENZTER Laufzeit:
Die ML-lastige Vor-Analyse (Era-Klassifikation, Defect-Scan, QualityAnalyzer,
AST-Classifier) und die Pipeline werden gepatcht, damit der Test die reale
restore()-Verdrahtung des Chunk-Gates prüft, ohne 12-GB-Restores zu tragen
(gemessen 2026-09-08: echter 121-s-Lauf ~11,7 GB RAM — RAM-Risiko des
Ganzsong-Modus, dokumentiert in docs/TIEFENANALYSE_RESTAURIERUNGSABLAUF.md).

Behauptungen:
1. whole_song=True → Chunked-Pfad wird NICHT aufgerufen (Spy).
2. Die Pipeline erhält das GESAMTE 121-s-Audio (kein 30-s-Slice).
3. Zwei Läufe liefern identische Ausgabe (Gate-Pfad deterministisch; die
   volle ML-Bit-Deterministik verifiziert erst der Stufe-2-Referenzlauf).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from backend.core.performance_guard import QualityMode
from backend.core.unified_restorer_v3 import RestorationConfig, UnifiedRestorerV3

_SR = 48000
_SECONDS = 121


def _synthetic_song(sr: int, seconds: int) -> np.ndarray:
    t = np.arange(sr * seconds, dtype=np.float64) / sr
    rng = np.random.default_rng(7)
    sig = 0.35 * np.sin(2 * np.pi * 220.0 * t) + 0.15 * np.sin(2 * np.pi * 440.0 * t)
    sig += 0.02 * rng.standard_normal(sr * seconds)
    return sig.astype(np.float32)


def _bounded_patches(audio: np.ndarray, sr: int) -> list:
    """Patch-Set: schwere ML-Vor-Analyse überspringen, Pipeline als Spy ersetzen."""

    from backend.core.defect_scanner import DefectAnalysisResult, DefectScanner, MaterialType

    _era_stub = SimpleNamespace(
        decade=1990,
        confidence=0.8,
        era_label="digital",
        primary_era="digital",
        era_score=0.8,
        material_type="digital",
        material_prior="digital",
        material="digital",
        primary_material="digital",
    )
    _scan_stub = DefectAnalysisResult(
        material_type=MaterialType.UNKNOWN,
        scores={},
        analysis_time_seconds=0.0,
        sample_rate=sr,
        duration_seconds=float(audio.shape[0]) / sr,
    )
    _qa_stub = SimpleNamespace()
    _qa_stub.analyze_quality = lambda a, s: SimpleNamespace(snr_db=40.0, warmth=0.8, naturalness=0.8, overall=0.8)
    _ast_stub = SimpleNamespace()
    _ast_stub.is_loaded = lambda: False

    _pipeline_len: dict[str, int] = {}

    def _execute_spy(self, a, s, *args, **kwargs):
        _pipeline_len["n"] = int(a.shape[0])
        return a, [], [], []

    _goals_ok = SimpleNamespace(passed=True, violations=[], passed_count=15, total_count=15)
    _artifact_ok = SimpleNamespace(
        passed=True,
        artifact_types=[],
        confidence=0.95,
        thresholds={},
        resolved_types=[],
        residual_scores={},
    )

    return [
        patch("backend.core.era_classifier.classify_era", return_value=_era_stub),
        patch.object(DefectScanner, "scan", return_value=_scan_stub),
        patch("backend.core.quality_prediction.QualityAnalyzer", return_value=_qa_stub),
        patch("backend.core.ast_audio_set_classifier.get_ast_classifier", return_value=_ast_stub),
        patch("backend.core.unified_restorer_v3.UnifiedRestorerV3._execute_pipeline", new=_execute_spy),
        patch("backend.core.musical_goals.musical_goals_metrics.MusicalGoalsChecker.measure_all", return_value=_goals_ok),
        patch("backend.core.artifact_freedom_gate.ArtifactFreedomGate.evaluate", return_value=_artifact_ok),
        patch("backend.core.feedback_chain.FeedbackChain.run", return_value=None),
    ], _pipeline_len


@pytest.mark.integration
@pytest.mark.slow
def test_whole_song_mode_single_pass_no_chunking_and_deterministic() -> None:
    _audio = _synthetic_song(_SR, _SECONDS)
    _patches, _pipeline_len = _bounded_patches(_audio, _SR)

    with _patches[0], _patches[1], _patches[2], _patches[3], _patches[4], _patches[5], _patches[6], _patches[7]:
        restorer = UnifiedRestorerV3(RestorationConfig(mode=QualityMode.FAST))
        _chunked_calls: list[tuple] = []
        _orig_chunked = restorer._restore_chunked

        def _spy_chunked(*args, **kwargs):
            _chunked_calls.append(args)
            return _orig_chunked(*args, **kwargs)

        restorer._restore_chunked = _spy_chunked  # type: ignore[method-assign]
        try:
            out1 = restorer.restore(_audio, _SR, whole_song=True)
            out2 = restorer.restore(_audio, _SR, whole_song=True)
        finally:
            restorer._restore_chunked = _orig_chunked  # type: ignore[method-assign]

    assert not _chunked_calls, "Ganzsong-Modus hat den Chunked-Pfad aufgerufen"
    assert _pipeline_len.get("n") == _audio.shape[0], (
        f"Pipeline erhielt nicht das ganze Audio: {_pipeline_len.get('n')} vs {_audio.shape[0]}"
    )

    a1 = np.asarray(out1.audio, dtype=np.float32)
    a2 = np.asarray(out2.audio, dtype=np.float32)
    assert a1.ndim == 1, f"Unerwartete Audioform: {a1.shape}"
    assert a1.shape[0] == _audio.shape[0], f"Länge verändert: {a1.shape[0]} vs {_audio.shape[0]}"
    assert bool(np.isfinite(a1).all()), "Ausgabe enthält NaN/Inf"
    assert np.array_equal(a1, a2), "Gate-Pfad nicht deterministisch: zwei Ganzsong-Läufe weichen ab"


@pytest.mark.integration
def test_whole_song_env_flag_disables_chunking(monkeypatch) -> None:
    """Env-Flag AURIK_WHOLE_SONG=1 wirkt ohne Kwarg (Verdrahtung am Gate)."""
    _audio = _synthetic_song(_SR, _SECONDS)
    monkeypatch.setenv("AURIK_WHOLE_SONG", "1")
    _patches, _pipeline_len = _bounded_patches(_audio, _SR)

    with _patches[0], _patches[1], _patches[2], _patches[3], _patches[4], _patches[5], _patches[6], _patches[7]:
        restorer = UnifiedRestorerV3(RestorationConfig(mode=QualityMode.FAST))
        _chunked_calls: list[tuple] = []
        _orig = restorer._restore_chunked

        def _spy(*args, **kwargs):
            _chunked_calls.append(args)
            return _orig(*args, **kwargs)

        restorer._restore_chunked = _spy  # type: ignore[method-assign]
        try:
            restorer.restore(_audio, _SR)
        finally:
            restorer._restore_chunked = _orig  # type: ignore[method-assign]

    assert not _chunked_calls, "AURIK_WHOLE_SONG=1 muss den Chunked-Pfad abschalten"
    assert _pipeline_len.get("n") == _audio.shape[0], "Env-Flag muss das ganze Audio an die Pipeline geben"
