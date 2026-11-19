import pytest

"""Unit tests for adaptive_material_learner.py"""
import tempfile


@pytest.mark.unit
class TestAdaptiveLearner:
    def test_01_init(self):
        from backend.core.adaptive_material_learner import MaterialAdaptiveLearner

        d = tempfile.mkdtemp()
        l = MaterialAdaptiveLearner(persist_dir=d)
        assert l is not None

    def test_02_suggest_default(self):
        from backend.core.adaptive_material_learner import MaterialAdaptiveLearner

        d = tempfile.mkdtemp()
        l = MaterialAdaptiveLearner(persist_dir=d)
        s = l.suggest_strength("vinyl")
        assert 0.3 <= s <= 1.0

    def test_03_record_no_crash(self):
        from backend.core.adaptive_material_learner import MaterialAdaptiveLearner

        d = tempfile.mkdtemp()
        l = MaterialAdaptiveLearner(persist_dir=d)
        l.record("vinyl", "test_action", 0.05)

    def test_04_unknown_material(self):
        from backend.core.adaptive_material_learner import MaterialAdaptiveLearner

        d = tempfile.mkdtemp()
        l = MaterialAdaptiveLearner(persist_dir=d)
        s = l.suggest_strength("nonexistent")
        assert 0.3 <= s <= 1.0

    def test_05_multiple_materials(self):
        from backend.core.adaptive_material_learner import MaterialAdaptiveLearner

        d = tempfile.mkdtemp()
        l = MaterialAdaptiveLearner(persist_dir=d)
        for m in ["vinyl", "tape", "cd_digital", "unknown"]:
            l.record(m, "action", 0.03)
            s = l.suggest_strength(m)
            assert 0.3 <= s <= 1.0

    def test_06_stats_returns_dict(self):
        from backend.core.adaptive_material_learner import MaterialAdaptiveLearner

        d = tempfile.mkdtemp()
        l = MaterialAdaptiveLearner(persist_dir=d)
        s = l.get_stats("vinyl")
        assert isinstance(s, dict)

    def test_07_singleton(self):
        from backend.core.adaptive_material_learner import get_learner

        l1 = get_learner()
        l2 = get_learner()
        assert l1 is l2

    def test_08_negative_reward(self):
        from backend.core.adaptive_material_learner import MaterialAdaptiveLearner

        d = tempfile.mkdtemp()
        l = MaterialAdaptiveLearner(persist_dir=d)
        l.record("vinyl", "bad_action", -0.1)

    def test_09_zero_reward(self):
        from backend.core.adaptive_material_learner import MaterialAdaptiveLearner

        d = tempfile.mkdtemp()
        l = MaterialAdaptiveLearner(persist_dir=d)
        l.record("vinyl", "neutral", 0.0)

    def test_10_many_records(self):
        from backend.core.adaptive_material_learner import MaterialAdaptiveLearner

        d = tempfile.mkdtemp()
        l = MaterialAdaptiveLearner(persist_dir=d)
        for i in range(20):
            l.record("vinyl", f"a{i}", 0.01 * i)
