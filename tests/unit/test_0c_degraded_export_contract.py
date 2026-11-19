"""§0c (copilot-instructions.md) — Export-Vertrag bei fehlgeschlagenem Quality-Gate.

Schließt den P1-4-Befund (2026-09-14): Drei MUSHRA-Pilot-Szenarien erzeugten
50-Byte-WAVs (Header + 2 Samples) mit rc=0/Status „ok", als das Export-Gate
scheiterte (quality_estimate ~0,48 < 0,55). §0c verlangt stattdessen: das
BESTMÖGLICHE sichere Ergebnis wird mit Status „degraded" exportiert — ein
Hardstop ohne Ausgabedatei ODER eine leere Datei ist normativ unzulässig.

Regressionstest auf dem aktuellen Export-Pfad
(``backend/core/export_workflow``): Gate-Fail ⇒ volle Audio-Länge +
Strategy „degraded" im Sidecar.

Autor: Aurik Testing Team
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from backend.core.export_workflow import _resolve_export_strategy, export_audio


class TestResolveExportStrategy:
    def test_fail_without_recovery_flag_is_degraded(self):
        gate = {"passed": False, "recovery_attempted": False}
        assert _resolve_export_strategy(gate, False) == "degraded"

    def test_fail_with_recovery_best_possible_is_recovered(self):
        gate = {"passed": False, "recovery_attempted": True, "best_possible_reached": True}
        assert _resolve_export_strategy(gate, False) == "recovered"

    def test_fail_with_recovery_incomplete_is_degraded(self):
        gate = {"passed": False, "recovery_attempted": True, "best_possible_reached": False}
        assert _resolve_export_strategy(gate, False) == "degraded"

    def test_fqf_triggered_is_authoritative(self):
        gate = {
            "passed": True,
            "fallback_quality_floor": {"triggered": True, "status": "degraded"},
        }
        assert _resolve_export_strategy(gate, True) == "degraded"

    def test_no_gate_is_success_legacy(self):
        assert _resolve_export_strategy(None, None) == "success"


class TestDegradedExportContract:
    def _audio(self, n: int = 48000) -> np.ndarray:
        t = np.linspace(0, n / 48000, n, endpoint=False, dtype=np.float32)
        return (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    def test_gate_fail_exports_full_audio_degraded(self, tmp_path):
        """Kern-Regression: Gate-Fail ⇒ volle Länge, NIE leere Datei (§0c)."""
        audio = self._audio()
        gate = {
            "passed": False,
            "fail_reason": "KRITISCH: quality_estimate 0.476 < 0.55",
            "recovery_attempted": False,
        }
        path = export_audio(audio, 48000, "gate_fail_case", quality_gate=gate, output_dir=str(tmp_path))
        assert os.path.exists(path)
        assert os.path.getsize(path) > 1000, "Export darf bei Gate-Fail nicht leer sein (§0c)"

        data, sr = sf.read(path)
        assert sr == 48000
        assert len(data) == len(audio), "Export muss die VOLLE Audio-Länge enthalten (kein 2-Sample-Stub)"

        sidecar = str(Path(path).with_suffix(".json"))
        assert os.path.exists(sidecar)
        meta = json.load(open(sidecar, encoding="utf-8"))
        assert meta.get("export_strategy") == "degraded"
        assert meta.get("quality_gate_passed") is False

    def test_gate_pass_exports_success_strategy(self, tmp_path):
        audio = self._audio(24000)
        gate = {"passed": True}
        path = export_audio(audio, 48000, "gate_ok_case", quality_gate=gate, output_dir=str(tmp_path))
        meta = json.load(open(str(Path(path).with_suffix(".json")), encoding="utf-8"))
        assert meta.get("export_strategy") == "success"
        data, _ = sf.read(path)
        assert len(data) == len(audio)

    def test_empty_input_audio_is_not_the_gate_contract(self, tmp_path):
        """Abgrenzung: leere EINGABE wird durchgereicht — aber das Gate selbst
        darf Audio niemals leeren. Hier: normaler Input + Fail-Gate bleibt voll."""
        audio = self._audio(12000)
        gate = {"passed": False, "recovery_attempted": False}
        path = export_audio(audio, 48000, "abgrenzung", quality_gate=gate, output_dir=str(tmp_path))
        data, _ = sf.read(path)
        assert len(data) == len(audio)
