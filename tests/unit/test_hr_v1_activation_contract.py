"""§SOTA-HR-V1 (Q6/F3, 2026-09-15) — BigVGAN-Aktivierungsvertrag: Tests.

Deckt die CPU-Vorbereitung des GPU-Buildouts:
- ``bigvgan_v2_ready()`` ist fail-closed (False ohne F3-Validierungs-Flag)
- ``hr_v1_activation_status()`` meldet Grund + Checkpoint-Präsenz (ZEUGE)
- Flag ON + Checkpoint ⇒ ready() True (Schaltstelle funktioniert)
- phase_07 meldet ``hr_v1``-Metadatum mit attempted=False (Status quo unverändert)

Autor: Aurik Testing Team
"""

import numpy as np

import plugins.bigvgan_v2_plugin as bvg


class TestActivationContract:
    def test_ready_false_by_default(self):
        assert bvg.bigvgan_v2_ready() is False

    def test_status_reports_f3_pending(self):
        status = bvg.hr_v1_activation_status()
        assert status["activated"] is False
        assert status["reason"] == "f3_validation_pending"
        # bigvgan_v2.pth ist lokal vorhanden — der Checkpoint allein reicht NICHT.
        assert status["checkpoint_present"] is True

    def test_flag_is_single_switch(self):
        old = bvg.BIGVGAN_V2_HR_ACTIVATED
        try:
            bvg.BIGVGAN_V2_HR_ACTIVATED = True
            assert bvg.bigvgan_v2_ready() is True
            assert bvg.hr_v1_activation_status()["reason"] == "activated"
        finally:
            bvg.BIGVGAN_V2_HR_ACTIVATED = old


class TestPhase07HrV1Witness:
    def test_phase07_reports_hr_v1_not_attempted(self):
        from backend.core.defect_scanner import MaterialType
        from backend.core.phases.phase_07_harmonic_restoration import HarmonicRestorationPhase

        t = np.linspace(0, 1, 48000, endpoint=False, dtype=np.float32)
        x = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        res = HarmonicRestorationPhase().process(x, sample_rate=48000, material_type=MaterialType.VINYL)
        hr = res.metadata.get("hr_v1")
        assert hr is not None, "phase_07 meldet kein hr_v1-Metadatum"
        assert hr.get("attempted") is False  # Aktivierungsvertrag: Status quo
        assert np.isfinite(np.asarray(res.audio)).all()
