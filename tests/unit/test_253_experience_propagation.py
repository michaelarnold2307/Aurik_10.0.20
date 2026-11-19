"""Tests für §2.53 Experience Closed Loop — Metadata-Propagation.

Verifiziert:
 1. RestaurierDenker._build_result() propagiert vollständiges UV3-Metadata
 2. joy_runtime_index, auto_improvement_recommendations, song_calibration landen in RestaurierErgebnis.metadata
 3. AurikErgebnis.metadata enthält §2.53 Pflichtfelder (song_calibration, joy_runtime_index, auto_improvement_recommendations)
 4. bridge.get_experience_insights() gibt gültige, NaN-freie Werte zurück
 5. RestaurierDenker._build_result() bricht nicht wenn raw.metadata None ist
"""

import math
import types

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------


def _make_mock_raw(metadata: dict | None = None, total_time: float = 1.23):
    """Erstellt ein minimales RestorationResult-Mock-Objekt."""
    raw = types.SimpleNamespace(
        audio=np.zeros(480, dtype=np.float32),
        phases_executed=["phase_01"],
        phases_skipped=[],
        musical_goals={"natuerlichkeit": 0.91},
        goals_passed=1,
        warnings=[],
        material_type=types.SimpleNamespace(value="vinyl"),
        quality_estimate=0.80,
        rt_factor=1.5,
        confidence=0.88,
        rollback_triggered=False,
        winning_variant=None,
        total_time_seconds=total_time,
        metadata=metadata,
    )
    return raw


def _expected_253_metadata() -> dict:
    """Minimale §2.53 konforme Metadata-Struktur (wie UV3 sie erstellt)."""
    return {
        "joy_runtime_index": {
            "joy_index": 0.72,
            "fatigue_index": 0.18,
            "components": {"natuerlichkeit": 0.91},
        },
        "auto_improvement_recommendations": {
            "count": 1,
            "recommendations": [
                {"focus": "noise_floor", "action": "reduce_denoise_strength", "reason": "over-processing"}
            ],
        },
        "song_calibration": {
            "cluster_key": "jazz:vinyl:pre-1980:good",
            "cluster_policy": {"cluster_key": "jazz:vinyl:pre-1980:good"},
        },
        "total_time_seconds": 7.5,
    }


# ---------------------------------------------------------------------------
# RestaurierDenker._build_result() Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRestaurierDenkerMetadataPropagation:
    """Stellt sicher dass RestaurierDenker das vollständige UV3-Metadata weitergibt."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from denker.restaurier_denker import RestaurierDenker

        # _konvertiere ist eine Instanzmethode — vereinfachte Instanz ohne UV3/ARE-Initialisierung
        self._denker = object.__new__(RestaurierDenker)
        self._denker._pipeline = None
        self._denker._lock = __import__("threading").Lock()

    def _konvertiere(self, raw, material="vinyl"):
        return self._denker._konvertiere(raw, material=material)

    def test_joy_runtime_index_propagated(self):
        """joy_runtime_index muss in RestaurierErgebnis.metadata vorhanden sein."""
        joy_data = {"joy_index": 0.75, "fatigue_index": 0.20, "components": {}}
        raw = _make_mock_raw(metadata={"joy_runtime_index": joy_data, "total_time_seconds": 1.0})
        result = self._konvertiere(raw, material="vinyl")
        assert "joy_runtime_index" in result.metadata, (
            "RestaurierDenker._build_result() hat joy_runtime_index verworfen"
        )
        assert result.metadata["joy_runtime_index"]["joy_index"] == pytest.approx(0.75, abs=0.01)

    def test_auto_improvement_propagated(self):
        """auto_improvement_recommendations muss in RestaurierErgebnis.metadata vorhanden sein."""
        auto = {"count": 2, "recommendations": [{"focus": "test", "action": "do_something", "reason": "x"}]}
        raw = _make_mock_raw(metadata={"auto_improvement_recommendations": auto, "total_time_seconds": 1.0})
        result = self._konvertiere(raw, material="vinyl")
        assert "auto_improvement_recommendations" in result.metadata, (
            "RestaurierDenker._build_result() hat auto_improvement_recommendations verworfen"
        )
        assert result.metadata["auto_improvement_recommendations"]["count"] == 2

    def test_song_calibration_propagated(self):
        """song_calibration mit cluster_key/cluster_policy muss propagiert werden."""
        song_cal = {
            "cluster_key": "rock:vinyl:pre-1980:fair",
            "cluster_policy": {"cluster_key": "rock:vinyl:pre-1980:fair"},
        }
        raw = _make_mock_raw(metadata={"song_calibration": song_cal, "total_time_seconds": 2.0})
        result = self._konvertiere(raw, material="vinyl")
        assert "song_calibration" in result.metadata, "RestaurierDenker._build_result() hat song_calibration verworfen"
        assert result.metadata["song_calibration"]["cluster_key"] == "rock:vinyl:pre-1980:fair"

    def test_total_time_seconds_always_present(self):
        """total_time_seconds muss auch ohne Metadata vorhanden sein."""
        raw = _make_mock_raw(metadata=None, total_time=3.14)
        result = self._konvertiere(raw, material="digital")
        assert "total_time_seconds" in result.metadata
        assert result.metadata["total_time_seconds"] == pytest.approx(3.14, abs=0.01)

    def test_total_time_overrides_metadata_value(self):
        """total_time_seconds aus raw.total_time_seconds hat Vorrang über Metadata-Eintrag."""
        raw = _make_mock_raw(
            metadata={"total_time_seconds": 999.0, "joy_runtime_index": {"joy_index": 0.5}},
            total_time=7.77,
        )
        result = self._konvertiere(raw, material="tape")
        # raw.total_time_seconds (7.77) sollte über den Dict-Wert (999.0) gewinnen
        assert result.metadata["total_time_seconds"] == pytest.approx(7.77, abs=0.01)

    def test_none_metadata_does_not_crash(self):
        """Wenn raw.metadata None ist, darf _konvertiere() nicht crashen."""
        raw = _make_mock_raw(metadata=None)
        result = self._konvertiere(raw, material="shellac")
        assert isinstance(result.metadata, dict)
        assert "total_time_seconds" in result.metadata

    def test_empty_metadata_dict_propagated(self):
        """Leeres Metadata-Dict liefert mindestens total_time_seconds."""
        raw = _make_mock_raw(metadata={})
        result = self._konvertiere(raw, material="digital")
        assert isinstance(result.metadata, dict)
        assert "total_time_seconds" in result.metadata

    def test_full_253_metadata_round_trip(self):
        """Vollständiger §2.53 Metadata-Satz überlebt den _konvertiere() Round-Trip."""
        full_meta = _expected_253_metadata()
        raw = _make_mock_raw(metadata=full_meta, total_time=7.5)
        result = self._konvertiere(raw, material="vinyl")
        for key in ("joy_runtime_index", "auto_improvement_recommendations", "song_calibration"):
            assert key in result.metadata, f"Pflichtfeld §2.53 '{key}' fehlt nach _build_result()"


# ---------------------------------------------------------------------------
# bridge.get_experience_insights() Tests
# ---------------------------------------------------------------------------


class TestBridgeExperienceInsights:
    """Verifiziert die Bridge-Funktion gemäß §2.53 Invariante #2."""

    def _make_ergebnis(self, metadata: dict | None = None):
        """Erstellt ein AurikErgebnis-ähnliches Objekt."""
        return types.SimpleNamespace(
            metadata=metadata or {},
        )

    def test_returns_dict_with_required_keys(self):
        """get_experience_insights() muss alle 6 Pflichtschlüssel zurückgeben."""
        from backend.api.bridge import get_experience_insights

        result = get_experience_insights(self._make_ergebnis(_expected_253_metadata()))
        required = {
            "joy_index",
            "fatigue_index",
            "frisson_index",
            "cluster_key",
            "cluster_policy",
            "recommendations",
            "recommendation_count",
        }
        assert required.issubset(result.keys()), f"Fehlende Schlüssel: {required - result.keys()}"

    def test_joy_index_between_0_and_1(self):
        """joy_index muss in [0, 1] liegen."""
        from backend.api.bridge import get_experience_insights

        result = get_experience_insights(self._make_ergebnis(_expected_253_metadata()))
        assert 0.0 <= result["joy_index"] <= 1.0

    def test_fatigue_index_between_0_and_1(self):
        """fatigue_index muss in [0, 1] liegen."""
        from backend.api.bridge import get_experience_insights

        result = get_experience_insights(self._make_ergebnis(_expected_253_metadata()))
        assert 0.0 <= result["fatigue_index"] <= 1.0

    def test_no_nan_inf_in_result(self):
        """Kein NaN oder Inf in numerischen Feldern."""
        from backend.api.bridge import get_experience_insights

        result = get_experience_insights(self._make_ergebnis(_expected_253_metadata()))
        for key in ("joy_index", "fatigue_index", "frisson_index"):
            assert math.isfinite(result[key]), f"{key} ist NaN/Inf"

    def test_correct_joy_value_extracted(self):
        """joy_index aus joy_runtime_index wird korrekt extrahiert."""
        from backend.api.bridge import get_experience_insights

        meta = {"joy_runtime_index": {"joy_index": 0.72, "fatigue_index": 0.18}}
        result = get_experience_insights(self._make_ergebnis(meta))
        assert result["joy_index"] == pytest.approx(0.72, abs=0.01)
        assert result["fatigue_index"] == pytest.approx(0.18, abs=0.01)

    def test_cluster_key_extracted(self):
        """cluster_key aus song_calibration wird korrekt extrahiert."""
        from backend.api.bridge import get_experience_insights

        meta = _expected_253_metadata()
        result = get_experience_insights(self._make_ergebnis(meta))
        assert result["cluster_key"] == "jazz:vinyl:pre-1980:good"

    def test_recommendations_list(self):
        """recommendations ist eine Liste."""
        from backend.api.bridge import get_experience_insights

        result = get_experience_insights(self._make_ergebnis(_expected_253_metadata()))
        assert isinstance(result["recommendations"], list)
        assert result["recommendation_count"] >= 1

    def test_empty_metadata_returns_safe_defaults(self):
        """Bei leerer Metadata keine Exception — sichere Defaults."""
        from backend.api.bridge import get_experience_insights

        result = get_experience_insights(self._make_ergebnis({}))
        assert result["joy_index"] == pytest.approx(0.0, abs=0.01)
        assert result["recommendation_count"] == 0

    def test_none_result_object_returns_safe_defaults(self):
        """Bei None-Objekt keine Exception."""
        from backend.api.bridge import get_experience_insights

        result = get_experience_insights(None)
        assert isinstance(result, dict)
        assert "joy_index" in result

    def test_nan_in_joy_index_sanitized(self):
        """NaN in joy_index wird auf 0.0 sanitisiert."""
        from backend.api.bridge import get_experience_insights

        meta = {"joy_runtime_index": {"joy_index": float("nan"), "fatigue_index": 0.5}}
        result = get_experience_insights(self._make_ergebnis(meta))
        assert isinstance(result["joy_index"], float)
        assert math.isfinite(result["joy_index"])
        assert result["joy_index"] == pytest.approx(0.0, abs=0.01)

    def test_nan_in_frisson_index_sanitized(self):
        """NaN in frisson_index wird auf 0.0 sanitisiert."""
        from backend.api.bridge import get_experience_insights

        meta = {"joy_runtime_index": {"joy_index": 0.4, "fatigue_index": 0.2, "frisson_index": float("nan")}}
        result = get_experience_insights(self._make_ergebnis(meta))
        assert isinstance(result["frisson_index"], float)
        assert math.isfinite(result["frisson_index"])
        assert result["frisson_index"] == pytest.approx(0.0, abs=0.01)
