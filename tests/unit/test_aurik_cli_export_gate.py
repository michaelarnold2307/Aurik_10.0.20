from __future__ import annotations

import pytest

"""Unit-Tests für adaptive Export-Gate-Profile in cli/aurik_cli.py."""


import math
from types import SimpleNamespace

import numpy as np

from cli.aurik_cli import (
    _compute_export_gate_adjustment,
    _compute_export_signal_signature,
)

SR = 48_000


def _sine(duration_s: float = 1.0, freq: float = 440.0) -> np.ndarray:
    t = np.linspace(0.0, duration_s, int(SR * duration_s), endpoint=False, dtype=np.float32)
    return np.asarray(np.sin(2.0 * np.pi * freq * t), dtype=np.float32)


@pytest.mark.unit
def test_export_signal_signature_returns_finite_metrics() -> None:
    sig = _compute_export_signal_signature(_sine(1.0), SR)
    for key in ("crest_db", "hf_ratio", "transient_ratio", "micro_dynamic_db"):
        assert key in sig
        assert math.isfinite(float(sig[key]))


def test_export_gate_adjustment_relaxes_for_fragile_risk() -> None:
    result = SimpleNamespace(
        stage_notes={"exzellenz_recovery_profile": {"preserve_signal": 0.6}},
    )
    sig = {
        "crest_db": 22.0,
        "hf_ratio": 0.01,
        "transient_ratio": 0.02,
        "micro_dynamic_db": 18.0,
    }
    qe_delta, pegel_delta, reason = _compute_export_gate_adjustment("tape", sig, result)
    assert qe_delta < 0.0
    assert pegel_delta > 0.0
    assert "risk=" in reason


def test_export_gate_adjustment_tightens_for_modern_stable() -> None:
    result = SimpleNamespace(stage_notes={})
    sig = {
        "crest_db": 10.0,
        "hf_ratio": 0.08,
        "transient_ratio": 0.002,
        "micro_dynamic_db": 8.0,
    }
    qe_delta, pegel_delta, reason = _compute_export_gate_adjustment("cd_digital", sig, result)
    assert qe_delta > 0.0
    assert pegel_delta < 0.0
    assert "modern_stable" in reason


def test_export_gate_adjustment_neutral_for_mid_risk() -> None:
    result = SimpleNamespace(stage_notes={"exzellenz_recovery_profile": {"preserve_signal": 0.1}})
    sig = {
        "crest_db": 13.0,
        "hf_ratio": 0.05,
        "transient_ratio": 0.004,
        "micro_dynamic_db": 10.0,
    }
    qe_delta, pegel_delta, reason = _compute_export_gate_adjustment("vinyl", sig, result)
    assert qe_delta == 0.0
    assert pegel_delta == 0.0
    assert "neutral" in reason


def test_cli_verbose_does_not_change_audio_before_export(monkeypatch, tmp_path) -> None:
    from cli import aurik_cli

    input_path = tmp_path / "input.wav"
    input_path.touch()
    audio = np.full((256, 2), 0.01, dtype=np.float32)
    exported: list[np.ndarray] = []

    def make_result():
        return SimpleNamespace(
            audio=audio.copy(),
            material="vinyl",
            quality_estimate=1.0,
            rt_factor=1.0,
            warnings=[],
            processing_note="",
            goals_passed=14,
            phases_executed=[],
            musical_goals_scores={},
            stage_notes={},
            metadata={},
        )

    monkeypatch.setattr(aurik_cli, "_load_audio", lambda _path: (audio.copy(), SR))
    monkeypatch.setattr(aurik_cli, "run_pre_analysis", lambda **_kwargs: object())
    monkeypatch.setattr(
        aurik_cli, "get_aurik_denker_instance", lambda: SimpleNamespace(denke=lambda *_args, **_kwargs: make_result())
    )
    monkeypatch.setattr(aurik_cli._bridge, "inject_cd_noise_profile", lambda data, *_args, **_kwargs: data + 0.1)

    def capture_export(_result, _path, restored, _reference, _logger, **_kwargs):
        exported.append(restored.copy())
        return True, [], {}

    monkeypatch.setattr(aurik_cli, "_export_audio_frontend_parity", capture_export)
    for verbose in (False, True):
        aurik_cli.process_audio(str(input_path), str(tmp_path / "out.wav"), verbose=verbose)

    np.testing.assert_array_equal(exported[0], exported[1])
    np.testing.assert_array_equal(exported[0], audio)


@pytest.mark.parametrize("vqi,hotspots,expected_sr", [(0.8, 0, 48000), (1.0, 2, 48000), (1.0, 0, 44100)])
def test_cli_export_uses_gui_downsampling_guard_and_polyphase(
    monkeypatch, tmp_path, vqi, hotspots, expected_sr
) -> None:
    from scipy.signal import resample_poly

    from cli import aurik_cli

    audio = np.random.default_rng(42).uniform(-0.2, 0.2, (1024, 2)).astype(np.float32)
    captured = {}

    class FakeExporter:
        FORMATS = {".wav": {}}

        def export(self, data, sr, _path, **kwargs):
            captured.update(audio=data.copy(), sr=sr, reference=kwargs["reference_audio"])

    monkeypatch.setattr(aurik_cli, "get_audio_exporter_class", lambda: FakeExporter)
    monkeypatch.setattr(aurik_cli, "export_guard", lambda x: x.copy())
    monkeypatch.setattr(aurik_cli, "validate_export_quality", lambda _result: (True, []))
    monkeypatch.setattr(
        aurik_cli,
        "build_export_quality_gate_payload",
        lambda _result: {"musiclover": {"vocal_integrity": {"vqi": vqi}, "temporal_risk": {"hotspot_count": hotspots}}},
    )
    result = SimpleNamespace(audio=audio.copy(), metadata={})
    aurik_cli._export_audio_frontend_parity(
        result, str(tmp_path / "out.wav"), audio, audio, aurik_cli.logger, output_sr=44100
    )

    expected = audio if expected_sr == SR else resample_poly(audio, 147, 160, axis=0).astype(np.float32)
    assert captured["sr"] == expected_sr
    np.testing.assert_array_equal(captured["audio"], expected)
    np.testing.assert_array_equal(captured["reference"], expected)


def test_cli_export_resamples_before_stereo_softening(monkeypatch, tmp_path) -> None:
    from scipy.signal import resample_poly

    from cli import aurik_cli

    audio = np.random.default_rng(42).uniform(-0.2, 0.2, (1024, 2)).astype(np.float32)
    captured = {}

    class FakeExporter:
        FORMATS = {".wav": {}}

        def export(self, data, _sr, _path, **_kwargs):
            captured["audio"] = data.copy()

    monkeypatch.setattr(aurik_cli, "get_audio_exporter_class", lambda: FakeExporter)
    monkeypatch.setattr(aurik_cli, "export_guard", lambda x: x.copy())
    monkeypatch.setattr(aurik_cli, "validate_export_quality", lambda _result: (True, []))
    monkeypatch.setattr(
        aurik_cli,
        "build_export_quality_gate_payload",
        lambda _result: {"musiclover": {"stereo_integrity": {"mono_compatibility_warning": True}}},
    )
    aurik_cli._export_audio_frontend_parity(
        SimpleNamespace(audio=audio.copy(), metadata={}),
        str(tmp_path / "out.wav"),
        audio,
        audio,
        aurik_cli.logger,
        output_sr=44100,
    )
    resampled = resample_poly(audio, 147, 160, axis=0).astype(np.float32)
    mid = 0.5 * (resampled[:, 0] + resampled[:, 1])
    side = 0.5 * (resampled[:, 0] - resampled[:, 1]) * 0.92
    expected = np.column_stack((np.clip(mid + side, -1.0, 1.0), np.clip(mid - side, -1.0, 1.0)))
    np.testing.assert_array_equal(captured["audio"], expected)
