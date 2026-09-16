"""Unit-Tests für die R2-Produktionsverdrahtung des MuQ-MOS-Witnesses (§0c).

Der Witness (ExportQualityGate) war bereits implementiert und getestet —
offen war die Produktionsverdrahtung: ``OneTakeExport.prepare`` bekam nie
``reference_audio``. Diese Tests sichern die Durchreichung ab.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.core.export_quality_gate import ExportQualityGate, ExportQualityResult
from backend.core.one_take_export import OneTakeExport, one_take_prepare


def _audio() -> np.ndarray:
    rng = np.random.RandomState(0)
    return (rng.randn(48000, 2).astype(np.float32) * 0.02) * 0.1


def _ok_result(**overrides) -> ExportQualityResult:
    base: dict = {
        "true_peak_dbtp": -2.5,
        "integrated_lufs": -14.0,
        "fatigue_score": 0.1,
        "stereo_correlation": 0.85,
        "muq_mos_in": 3.6,
        "muq_mos_out": 3.55,
        "muq_mos_delta": -0.05,
        "muq_mos_degraded": False,
        "warnings": [],
        "errors": [],
        "passed": True,
    }
    base.update(overrides)
    return ExportQualityResult(**base)


@pytest.mark.unit
class TestR2MuQMosProductionWiring:
    def test_prepare_passes_reference_audio_to_check(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict = {}

        def _fake_check(audio, sr, is_studio_2026=False, reference_audio=None):
            captured["reference_audio"] = reference_audio
            captured["sr"] = sr
            return _ok_result()

        monkeypatch.setattr(ExportQualityGate, "check", staticmethod(_fake_check))
        src = _audio()
        ref = _audio()
        out = OneTakeExport.prepare(src, 48000, reference_audio=ref)
        assert captured["reference_audio"] is ref
        assert out.passed is True
        # MuQ-Felder landen im Quality-Report (§0c-Transparenz).
        assert out.quality_report.get("muq_mos_delta") == pytest.approx(-0.05, abs=0.01)
        assert out.quality_report.get("muq_mos_degraded") is False

    def test_prepare_without_reference_skips_witness(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict = {}

        def _fake_check(audio, sr, is_studio_2026=False, reference_audio=None):
            captured["reference_audio"] = reference_audio
            return _ok_result(muq_mos_in=None, muq_mos_out=None, muq_mos_delta=None)

        monkeypatch.setattr(ExportQualityGate, "check", staticmethod(_fake_check))
        out = OneTakeExport.prepare(_audio(), 48000)
        assert captured["reference_audio"] is None
        assert out.quality_report.get("muq_mos_delta") is None

    def test_iterative_second_check_receives_reference(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """§v10.0.5 2-Pass-Mode: auch der Verifikations-Check bekommt die Referenz."""
        captured: list = []
        _real_check = ExportQualityGate.check

        def _fake_check(audio, sr, is_studio_2026=False, reference_audio=None):
            captured.append(reference_audio)
            if len(captured) == 1:
                # Erster Pass: Grenzwert-Verletzung erzwingt Korrektur + 2. Pass.
                return _ok_result(true_peak_dbtp=-0.5, warnings=["true_peak"], passed=False)
            return _ok_result()

        monkeypatch.setattr(ExportQualityGate, "check", staticmethod(_fake_check))
        ref = _audio()
        out = OneTakeExport.prepare(_audio(), 48000, iterative=True, reference_audio=ref)
        assert out.passed is True
        # Attempt 0: Check + Verifikations-Check2; Attempt 1: Check → 3 Checks.
        assert len(captured) == 3
        assert all(c is ref for c in captured)

    def test_wrapper_forwards_reference(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict = {}

        def _fake_prepare(audio, sr, *, is_studio_2026=False, iterative=False, reference_audio=None):
            captured["reference_audio"] = reference_audio
            captured["is_studio_2026"] = is_studio_2026
            return _ok_result()

        monkeypatch.setattr(OneTakeExport, "prepare", staticmethod(_fake_prepare))
        ref = _audio()
        one_take_prepare(_audio(), 48000, is_studio_2026=True, reference_audio=ref)
        assert captured["reference_audio"] is ref
        assert captured["is_studio_2026"] is True


@pytest.mark.unit
class TestBridgeMuQWitnessPayload:
    """Bridge-Payload surfed den MuQ-Witness informativ — nie als Gate-Urteil (§0c)."""

    def _result(self, **meta) -> object:
        class _R:
            metadata: dict = dict(meta)
            quality_estimate: float = 0.85
            vqi: float = 0.9
            lufs_delta: float = 0.0
            chroma_correlation: float = 0.7

        return _R()

    def test_payload_carries_muq_fields(self) -> None:
        from backend.api.bridge import build_export_quality_gate_payload

        payload = build_export_quality_gate_payload(
            self._result(export_muq_mos_in=3.61, export_muq_mos_out=3.58, export_muq_mos_delta=-0.03)
        )
        witness = payload["muq_mos_witness"]
        assert witness["available"] is True
        assert witness["muq_mos_delta"] == pytest.approx(-0.03, abs=1e-3)
        assert witness["muq_mos_in"] == pytest.approx(3.61, abs=1e-3)
        assert witness["muq_mos_out"] == pytest.approx(3.58, abs=1e-3)
        # Witness ändert NIE das Gate-Urteil.
        assert "passed" in payload

    def test_payload_without_muq_is_unavailable(self) -> None:
        from backend.api.bridge import build_export_quality_gate_payload

        payload = build_export_quality_gate_payload(self._result())
        witness = payload["muq_mos_witness"]
        assert witness["available"] is False
        assert witness["muq_mos_delta"] == 0.0
