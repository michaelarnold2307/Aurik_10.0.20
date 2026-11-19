"""Unit-Tests für RecordingChainProfiler (§2.66, v10.0.0)."""

import pytest

from backend.core.recording_chain_profiler import (
    ChainProfile,
    RecordingChainProfiler,
    get_recording_chain_profiler,
)


@pytest.mark.unit
class TestRecordingChainProfilerBelowThreshold:
    def test_two_causes_returns_empty_profile(self):
        rcp = RecordingChainProfiler()
        profile = rcp.profile_chain(["tape_hiss", "wow_flutter"], material="tape", era=1970)
        assert profile.chain_hint is None
        assert profile.cluster_weight == pytest.approx(0.0)

    def test_zero_causes_returns_neutral(self):
        rcp = RecordingChainProfiler()
        profile = rcp.profile_chain([], material="vinyl", era=1965)
        assert isinstance(profile, ChainProfile)
        assert profile.chain_hint is None or profile.chain_hint == {}


class TestRecordingChainProfilerTapeCluster:
    def test_tape_causes_detected(self):
        rcp = RecordingChainProfiler()
        causes = ["tape_hiss", "wow_flutter", "tape_dropout"]
        profile = rcp.profile_chain(causes, material="tape", era=1970)
        assert isinstance(profile, ChainProfile)
        assert "tape" in profile.dominant_cluster.lower() or "speed" in profile.dominant_cluster.lower()

    def test_tape_material_boost_affects_weight(self):
        """Material-Boost: tape-Material mit tape-Causes → höheres Gewicht als neutral."""
        rcp = RecordingChainProfiler()
        tape_causes = ["tape_hiss", "wow_flutter", "tape_dropout"]
        profile_tape = rcp.profile_chain(tape_causes, material="tape", era=1970)
        profile_vinyl = rcp.profile_chain(tape_causes, material="vinyl", era=1970)
        # Mit passendem Material soll das Gewicht höher oder gleich sein
        if profile_tape.dominant_cluster and profile_vinyl.dominant_cluster:
            assert profile_tape.cluster_weight >= profile_vinyl.cluster_weight - 0.01


class TestRecordingChainProfilerVinylCluster:
    def test_vinyl_causes_detected(self):
        rcp = RecordingChainProfiler()
        causes = ["vinyl_crackle", "vinyl_warp", "riaa_curve_error"]
        profile = rcp.profile_chain(causes, material="vinyl", era=1965)
        assert isinstance(profile, ChainProfile)
        assert "vinyl" in profile.dominant_cluster.lower()


class TestRecordingChainProfilerDigitalCluster:
    def test_digital_causes_detected(self):
        rcp = RecordingChainProfiler()
        causes = ["pre_echo", "digital_clip", "compression_artifacts"]
        profile = rcp.profile_chain(causes, material="mp3", era=2005)
        assert isinstance(profile, ChainProfile)
        assert "digital" in profile.dominant_cluster.lower()


class TestRecordingChainProfilerChainHint:
    def test_chain_hint_is_dict_or_none(self):
        rcp = RecordingChainProfiler()
        causes = ["TAPE_HUM", "TAPE_SPEED_DRIFT", "TAPE_SATURATION", "WOW_FLUTTER"]
        profile = rcp.profile_chain(causes, material="tape", era=1970)
        assert profile.chain_hint is None or isinstance(profile.chain_hint, dict)
        if profile.chain_hint is not None:
            assert "strength_scale" not in profile.chain_hint

    def test_active_clusters_non_empty_for_3_causes(self):
        rcp = RecordingChainProfiler()
        causes = ["tape_hiss", "wow_flutter", "tape_dropout"]
        profile = rcp.profile_chain(causes, material="tape", era=1970)
        assert isinstance(profile.active_clusters, dict)
        assert len(profile.active_clusters) > 0

    def test_suppress_causes_is_list(self):
        rcp = RecordingChainProfiler()
        causes = ["tape_hiss", "wow_flutter", "tape_dropout", "head_wear", "vinyl_crackle"]
        profile = rcp.profile_chain(causes, material="tape", era=1970)
        assert isinstance(profile.suppress_causes, list)


class TestRecordingChainProfilerSingleton:
    def test_singleton_returns_same_instance(self):
        a = get_recording_chain_profiler()
        b = get_recording_chain_profiler()
        assert a is b

    def test_singleton_is_recording_chain_profiler(self):
        inst = get_recording_chain_profiler()
        assert isinstance(inst, RecordingChainProfiler)
