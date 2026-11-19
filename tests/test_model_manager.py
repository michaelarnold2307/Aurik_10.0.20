import pytest

from backend.core.model_manager import ModelManager


class DummyModel:
    def process(self, audio, context):
        return audio


@pytest.mark.unit
def test_register_and_list_models():
    mm = ModelManager()
    mm.register_model("test_model", DummyModel(), {"type": "test", "quality": "high"})
    models = mm.list_models()
    assert "test_model" in models
    assert models["test_model"]["type"] == "test"


def test_select_model_and_fallback():
    mm = ModelManager()
    mm.register_model("m1", None, {"quality": "low"})
    mm.register_model("m2", DummyModel(), {"quality": "high"})
    selected = mm.select_model({})
    assert isinstance(selected, DummyModel)


def test_reload_model():
    mm = ModelManager()
    mm.register_model("m1", DummyModel(), {"quality": "high"})

    class NewModel:
        def process(self, audio, context):
            return audio

    mm.reload_model_api("m1", NewModel(), {"quality": "high"})
    assert isinstance(mm.models["m1"]["obj"], NewModel)


def test_audit_log():
    mm = ModelManager()
    mm.register_model("m1", DummyModel(), {"quality": "high"})
    mm.select_model({})
    log = mm.get_audit_log()
    assert len(log) > 0


def test_audit_log_export_json_csv():
    mm = ModelManager()
    mm.register_model("m1", DummyModel(), {"quality": "high"})
    mm.select_model({})
    json_log = mm.get_audit_log(as_json=True)
    csv_log = mm.get_audit_log(as_csv=True)
    assert json_log.startswith("[")
    assert "model" in json_log
    assert "model" in csv_log and "," in csv_log


def test_get_model_api_status():
    mm = ModelManager()
    mm.register_model("m1", DummyModel(), {"quality": "high"})
    mm.select_model({})
    status = mm.get_model_api_status()
    assert "m1" in status
    assert "status" in status["m1"]
    assert "last_selected" in status["m1"]


def test_add_user_feedback_and_voice_profile():
    mm = ModelManager()
    mm.register_model("m1", DummyModel(), {"quality": "high"})
    feedback = {"model": "m1", "score": 5, "comment": "gut"}
    mm.add_user_feedback(feedback)
    assert feedback in mm.user_feedback
    profile = {"age": 30, "gender": "female"}
    mm.set_voice_profile(profile)
    assert mm.voice_profile == profile
