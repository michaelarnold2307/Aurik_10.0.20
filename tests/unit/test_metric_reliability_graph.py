from __future__ import annotations

import pytest


@pytest.mark.unit
@pytest.mark.timeout(15)
def test_metric_reliability_graph_updates_and_returns_contextual_values(tmp_path, monkeypatch):
    monkeypatch.setenv("AURIK_METRIC_RELIABILITY_PATH", str(tmp_path / "mrg.json"))

    from backend.core.metric_reliability_graph import MetricReliabilityGraph

    graph = MetricReliabilityGraph()
    graph.update_from_phase_delta(
        phase_id="phase_24_dropout_repair",
        goal_deltas={"natuerlichkeit": 0.04, "authentizitaet": 0.02, "brillanz": -0.01},
        phase_metadata={
            "pmgg_team_net_delta": 0.015,
            "pmgg_reconstruction_localized": True,
            "pmgg_reconstruction_epistemic_confidence": 0.78,
        },
        material_type="vinyl",
        transfer_chain=["vinyl", "mp3_low"],
        is_studio_2026=False,
        era_decade=1970,
    )

    conf = graph.get_goal_reliability(
        goal_scores={"natuerlichkeit": 0.72, "authentizitaet": 0.70, "brillanz": 0.66},
        material_type="vinyl",
        transfer_chain=["vinyl", "mp3_low"],
        is_studio_2026=False,
        era_decade=1970,
    )

    assert conf
    assert 0.20 <= float(conf["natuerlichkeit"]) <= 0.98
    assert 0.20 <= float(conf["authentizitaet"]) <= 0.98
    assert 0.20 <= float(conf["brillanz"]) <= 0.98
    assert float(conf["natuerlichkeit"]) >= float(conf["brillanz"])


@pytest.mark.unit
@pytest.mark.timeout(15)
def test_metric_reliability_graph_persists_across_instances(tmp_path, monkeypatch):
    monkeypatch.setenv("AURIK_METRIC_RELIABILITY_PATH", str(tmp_path / "mrg_persist.json"))

    from backend.core.metric_reliability_graph import MetricReliabilityGraph

    g1 = MetricReliabilityGraph()
    g1.update_from_phase_delta(
        phase_id="phase_50_spectral_repair",
        goal_deltas={"transparenz": 0.03},
        phase_metadata={"pmgg_team_net_delta": 0.01},
        material_type="cd_digital",
        transfer_chain=["cd_digital"],
        is_studio_2026=False,
        era_decade=1990,
    )

    g2 = MetricReliabilityGraph()
    conf = g2.get_goal_reliability(
        goal_scores={"transparenz": 0.68},
        material_type="cd_digital",
        transfer_chain=["cd_digital"],
        is_studio_2026=False,
        era_decade=1990,
    )
    assert 0.20 <= float(conf["transparenz"]) <= 0.98


@pytest.mark.unit
@pytest.mark.timeout(15)
def test_metric_reliability_graph_blend_weights_shift_with_support(tmp_path, monkeypatch):
    monkeypatch.setenv("AURIK_METRIC_RELIABILITY_PATH", str(tmp_path / "mrg_blend.json"))

    from backend.core.metric_reliability_graph import MetricReliabilityGraph

    graph = MetricReliabilityGraph()
    base_w0, runtime_w0 = graph.get_blend_weights(
        material_type="vinyl",
        transfer_chain=["vinyl", "mp3_low"],
        is_studio_2026=False,
        era_decade=1970,
    )
    assert runtime_w0 <= 0.30

    for _ in range(40):
        graph.update_from_phase_delta(
            phase_id="phase_24_dropout_repair",
            goal_deltas={"natuerlichkeit": 0.05, "authentizitaet": 0.04, "transparenz": 0.03},
            phase_metadata={"pmgg_team_net_delta": 0.02},
            material_type="vinyl",
            transfer_chain=["vinyl", "mp3_low"],
            is_studio_2026=False,
            era_decade=1970,
        )

    base_w1, runtime_w1 = graph.get_blend_weights(
        material_type="vinyl",
        transfer_chain=["vinyl", "mp3_low"],
        is_studio_2026=False,
        era_decade=1970,
    )
    assert 0.40 <= base_w1 <= 0.75
    assert 0.25 <= runtime_w1 <= 0.60
    assert runtime_w1 > runtime_w0
    assert abs((base_w1 + runtime_w1) - 1.0) < 1e-6
