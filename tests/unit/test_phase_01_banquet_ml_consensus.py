"""§SOTA-CR-V1 — BANQUET-ML als Zusatz-Detektor im phase_01-Multi-Scale-Konsens.

Never-worsen: ML-Regionen werden nur VEREINIGT (Union), DSP-Regionen bleiben
erhalten; ohne Modell oder bei Fehler bleibt der Pfad unverändert
(§V6 (copilot-instructions.md)).
Determinismus (§G5 (copilot-instructions.md)): gleicher Input ⇒ identische Regionen.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.core.phases.phase_01_click_removal import ClickRemovalPhase


class _FakeBanquetPlugin:
    """Simuliert das BANQUET-Plugin: entfernt einen synthetischen Klick."""

    def __init__(self, model_loaded: bool = True, click_center: int = 48000, fail: bool = False):
        self._model_loaded = model_loaded
        self._click_center = click_center
        self._fail = fail

    def ensure_model_loaded(self) -> bool:
        # §PERF-R (2026-09-18): Der Phasen-Code prüft jetzt über die
        # öffentliche Methode statt über das im Plugin nie existierende
        # `_model_loaded`-Attribut (war immer False → CR-V1 stumm tot).
        return bool(self._model_loaded)

    def _maybe_resample(self, audio, src, tgt):
        if src == tgt:
            return audio, tgt
        n = int(len(audio) * tgt / src)
        return (
            np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio).astype(np.float32),
            tgt,
        )

    def _process_onnx(self, audio, strength=1.0):
        if self._fail:
            raise RuntimeError("simulated banquett failure")
        out = np.array(audio, dtype=np.float32, copy=True)
        if out.size > self._click_center + 64:
            if out.ndim == 2:  # §PERF-R (2026-09-18): Aufruf jetzt 2-D [channels, N]
                out[0, self._click_center - 8 : self._click_center + 8] *= 0.15  # Klick deutlich abgeschwächt
            else:
                out[self._click_center - 8 : self._click_center + 8] *= 0.15
        return out


def _make_audio_with_click(n: int = 96000, sr: int = 48000, click_center: int = 48000):
    rng = np.random.default_rng(7)
    x = 0.05 * rng.standard_normal(n).astype(np.float32)
    x[click_center - 4 : click_center + 4] = 0.95
    return x, sr


@pytest.fixture
def phase():
    return ClickRemovalPhase()


def test_model_missing_returns_empty_and_dsp_unchanged(phase, monkeypatch):
    monkeypatch.setattr(
        "plugins.banquet_vinyl_plugin.get_banquet_plugin",
        lambda: _FakeBanquetPlugin(model_loaded=False),
    )
    audio, sr = _make_audio_with_click()
    regions = phase._detect_clicks_banquet_ml(audio, sr)
    assert regions == []


def test_ml_detects_synthetic_click(phase, monkeypatch):
    click_center = 48000
    monkeypatch.setattr(
        "plugins.banquet_vinyl_plugin.get_banquet_plugin",
        lambda: _FakeBanquetPlugin(model_loaded=True, click_center=click_center),
    )
    audio, sr = _make_audio_with_click(click_center=click_center)
    regions = phase._detect_clicks_banquet_ml(audio, sr)
    assert regions, "ML-Detektor muss die synthetische Klick-Region liefern"
    assert any(s <= click_center <= e for s, e in regions)


def test_exception_falls_back_to_empty(phase, monkeypatch):
    monkeypatch.setattr(
        "plugins.banquet_vinyl_plugin.get_banquet_plugin",
        lambda: _FakeBanquetPlugin(model_loaded=True, fail=True),
    )
    audio, sr = _make_audio_with_click()
    regions = phase._detect_clicks_banquet_ml(audio, sr)
    assert regions == []


def test_determinism_same_input_same_regions(phase, monkeypatch):
    monkeypatch.setattr(
        "plugins.banquet_vinyl_plugin.get_banquet_plugin",
        lambda: _FakeBanquetPlugin(model_loaded=True),
    )
    audio, sr = _make_audio_with_click()
    r1 = phase._detect_clicks_banquet_ml(audio, sr)
    r2 = phase._detect_clicks_banquet_ml(audio, sr)
    assert r1 == r2


def test_resample_path_scales_regions_back(phase, monkeypatch):
    # 96 kHz-Input mit Klick bei Index 96000 → in 48-kHz-Domäne bei 48000;
    # Regionen werden mit Faktor 96/48=2 zurück skaliert (≈ 96000).
    monkeypatch.setattr(
        "plugins.banquet_vinyl_plugin.get_banquet_plugin",
        lambda: _FakeBanquetPlugin(model_loaded=True, click_center=48000),
    )
    n = 192000
    x = 0.05 * np.random.default_rng(11).standard_normal(n).astype(np.float32)
    x[96000 - 8 : 96000 + 8] = 0.95
    regions = phase._detect_clicks_banquet_ml(x, 96000)
    assert regions, "ML-Detektor muss auch bei Resampling Regionen liefern"
    assert all(0 <= s <= e < n for s, e in regions)
    # Region liegt um den zurück skalierten Klick-Zeitpunkt (Toleranz ±1 %)
    assert any(abs(s - 96000) <= 960 and abs(e - 96000) <= 960 for s, e in regions)


def test_merge_union_preserves_dsp_regions():
    dsp = [(100, 200), (1000, 1100)]
    ml = [(500, 600), (1050, 1150)]
    merged = ClickRemovalPhase._merge_click_regions(dsp, ml)
    # DSP-Regionen sind Teilmenge (Never-worsen)
    for s, e in dsp:
        assert any(ms <= s and e <= me for ms, me in merged)
    # ML-Regionen sind enthalten (Union)
    assert any(ms <= 500 <= me for ms, me in merged)
    # Überlappende Regionen verschmolzen: (1000,1100) und (1050,1150) → eine Region
    assert any(ms <= 1000 and me >= 1150 for ms, me in merged)


def test_merge_empty_and_gap_tolerance():
    assert ClickRemovalPhase._merge_click_regions([], []) == []
    merged = ClickRemovalPhase._merge_click_regions([(100, 200)], [(230, 300)], gap_tolerance=32)
    assert merged == [(100, 300)]
    merged_wide = ClickRemovalPhase._merge_click_regions([(100, 200)], [(400, 500)], gap_tolerance=32)
    assert merged_wide == [(100, 200), (400, 500)]


def test_full_plan_integration_never_worsen(phase, monkeypatch):
    """End-to-End: Der Reparaturplan MIT ML ist eine Obermenge des Plans OHNE ML."""
    audio, sr = _make_audio_with_click()
    thresholds = {"short": 0.02, "medium": 0.05, "long": 0.1, "transient_preserve": 0.4}

    monkeypatch.setattr(
        "plugins.banquet_vinyl_plugin.get_banquet_plugin",
        lambda: _FakeBanquetPlugin(model_loaded=False),
    )
    severe_0, normal_0, _stats_0 = phase._build_click_repair_plan(
        audio, sr, thresholds, preserve_transients=True, use_ml=True
    )
    plan_0 = [(c["start"], c["end"]) for c in severe_0 + normal_0]

    monkeypatch.setattr(
        "plugins.banquet_vinyl_plugin.get_banquet_plugin",
        lambda: _FakeBanquetPlugin(model_loaded=True),
    )
    severe_1, normal_1, _stats_1 = phase._build_click_repair_plan(
        audio, sr, thresholds, preserve_transients=True, use_ml=True
    )
    plan_1 = [(c["start"], c["end"]) for c in severe_1 + normal_1]

    # Never-worsen: jede Baseline-Reparaturregion bleibt erhalten (Obermenge)
    for s, e in plan_0:
        assert any(ps <= s and e <= pe for ps, pe in plan_1)
    # ML-Erweiterung: der Plan wächst um die ML-Klick-Region (≈ 48000)
    assert len(plan_1) >= len(plan_0)
    assert any(abs(ps - 48000) <= 2000 for ps, _pe in plan_1)
