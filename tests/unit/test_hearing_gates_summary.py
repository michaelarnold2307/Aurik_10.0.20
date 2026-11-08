"""Tests für Aurik10/ui/hearing_gates_summary.py (T1, headless)."""

from __future__ import annotations

from Aurik10.ui.hearing_gates_summary import (
    apply_resolved_defects,
    apply_resolved_event_counts,
    hearing_gate_status,
    hearing_gates_details,
    hearing_gates_line,
)


def _meta(**kw) -> dict:
    base: dict = {}
    base.update(kw)
    return base


class TestStatus:
    def test_green_empty(self) -> None:
        assert hearing_gate_status(_meta()) == "green"

    def test_red_audibility(self) -> None:
        m = _meta(audibility_gate={"gate_passed": False, "n_audible_unmasked": 3, "improvable_types": ["hum"]})
        assert hearing_gate_status(m) == "red"

    def test_yellow_corrected(self) -> None:
        m = _meta(einladungs_gate_passed=False, einladungs_gate_corrected=True)
        assert hearing_gate_status(m) == "yellow"

    def test_yellow_queue(self) -> None:
        m = _meta(audibility_gate={"gate_passed": True, "improvable_types": ["clicks"]})
        assert hearing_gate_status(m) == "yellow"

    def test_green_m1b_applied(self) -> None:
        # §P1-7: Backend hat die Stufe-2 intern ausgeführt → nicht mehr „gequeued“
        m = _meta(audibility_gate={"gate_passed": True, "improvable_types": ["clicks"]}, m1b_retry_applied=True)
        assert hearing_gate_status(m) == "green"

    def test_red_vocal_revert(self) -> None:
        m = _meta(vocal_drive_hard_revert=True)
        assert hearing_gate_status(m) == "red"

    def test_red_wohlklang_ordnung_violation(self) -> None:
        m = _meta(wohlklang_ordnung={"status": "VIOLATION", "violated_goals": ["waerme"]})
        assert hearing_gate_status(m) == "red"

    def test_green_wohlklang_ordnung_pass(self) -> None:
        m = _meta(wohlklang_ordnung={"status": "PASS"})
        assert hearing_gate_status(m) == "green"


class TestLines:
    def test_line_has_icon(self) -> None:
        assert "Hör-Gates" in hearing_gates_line(_meta())

    def test_details_audibility(self) -> None:
        m = _meta(
            audibility_gate={"gate_passed": False, "n_audible_unmasked": 2, "improvable_types": ["hum", "clicks"]}
        )
        det = hearing_gates_details(m)
        assert any("Restdefekte" in d for d in det)
        assert any("Stufe-2-Queue" in d for d in det)

    def test_details_m1b_applied(self) -> None:
        m = _meta(
            audibility_gate={"gate_passed": False, "n_audible_unmasked": 2, "improvable_types": ["hum", "clicks"]},
            m1b_retry_applied=True,
        )
        det = hearing_gates_details(m)
        assert any("Stufe-2: ausgeführt" in d for d in det)
        assert not any("Stufe-2-Queue" in d for d in det)

    def test_details_wohlklang(self) -> None:
        det = hearing_gates_details(_meta(einladungs_gate_passed=True))
        assert any("Wohlklang: erfüllt" in d for d in det)

    def test_details_wohlklang_ordnung_violation(self) -> None:
        m = _meta(wohlklang_ordnung={"status": "VIOLATION", "violated_goals": ["waerme"]})
        det = hearing_gates_details(m)
        assert any("Wohlklang-Ordnung: verletzt" in d and "waerme" in d for d in det)

    def test_details_wohlklang_ordnung_pass(self) -> None:
        m = _meta(wohlklang_ordnung={"status": "PASS"})
        det = hearing_gates_details(m)
        assert any("Wohlklang-Ordnung: erfüllt" in d for d in det)

    def test_details_empty(self) -> None:
        det = hearing_gates_details(_meta())
        assert any("n/a" in d for d in det)

    def test_blend_telemetry(self) -> None:
        det = hearing_gates_details(_meta(vocal_drive_blend=0.85))
        assert any("Vocal-Drive" in d for d in det)


class TestApplyResolvedDefects:
    def test_subtract_and_total(self) -> None:
        counts = {"clicks": 5, "hum": 2}
        new, total, done = apply_resolved_defects(counts, ["clicks", "hum", "hum"])
        assert new == {"clicks": 4, "hum": 0}
        assert total == 4
        assert done == ["hum"]

    def test_unknown_ignored_and_no_negative(self) -> None:
        new, total, done = apply_resolved_defects({"clicks": 0}, ["clicks", "fremd"])
        assert new == {"clicks": 0}
        assert total == 0
        assert done == []

    def test_none_safe(self) -> None:
        new, total, done = apply_resolved_defects({}, None)
        assert new == {} and total == 0 and done == []

    def test_original_untouched(self) -> None:
        counts = {"hum": 3}
        apply_resolved_defects(counts, ["hum"])
        assert counts == {"hum": 3}


class TestApplyResolvedEventCounts:
    """§v10.802 GUI-Sync: Echte Instanzen-Subtraktion für den Defektzähler."""

    def test_exact_subtraction(self) -> None:
        init = {"CLICKS": 26, "CRACKLE": 287, "HUM": 4}
        resolved = {"CLICKS": 10, "CRACKLE": 100}
        rem = apply_resolved_event_counts(init, resolved)
        assert rem == {"CLICKS": 16, "CRACKLE": 187, "HUM": 4}

    def test_case_insensitive_keys(self) -> None:
        init = {"clicks": 26}
        rem = apply_resolved_event_counts(init, {"CLICKS": 26})
        assert rem == {"clicks": 0}

    def test_never_negative(self) -> None:
        init = {"CLICKS": 5}
        rem = apply_resolved_event_counts(init, {"CLICKS": 99})
        assert rem == {"CLICKS": 0}

    def test_empty_resolved_keeps_init(self) -> None:
        init = {"CLICKS": 5}
        rem = apply_resolved_event_counts(init, None)
        assert rem == {"CLICKS": 5}
        rem2 = apply_resolved_event_counts(init, {})
        assert rem2 == {"CLICKS": 5}

    def test_unknown_types_ignored(self) -> None:
        init = {"CLICKS": 5}
        rem = apply_resolved_event_counts(init, {"WOW": 3})
        assert rem == {"CLICKS": 5}


class TestInvitingGateEbene4:
    """Hörordnungs-Einladungs-Gate (Ebene 4) in der Ampel — _meta["inviting_gate_song_level"]
    (2026-11-08): delta-basierte Regressionen (Pumpen/Lautheits-Sprung) machen die
    Ampel GELB (advisory), nie ROT; erfülltes Gate bleibt grün.
    """

    def test_pumpen_regression_macht_gelb_mit_detail(self) -> None:
        meta = {
            "inviting_gate_song_level": {
                "passed": False,
                "max_vacil": 0.55,
                "max_vacil_input": 0.10,
                "fluctuation_regression": True,
                "loudness_jump_regression": False,
            }
        }
        assert hearing_gate_status(meta) == "yellow"
        details = hearing_gates_details(meta)
        assert any("Pumpen-Regression" in d and "0.55 vacil" in d for d in details)

    def test_loudness_sprung_regression_macht_gelb(self) -> None:
        meta = {
            "inviting_gate_song_level": {
                "passed": False,
                "loudness_jump_max": 4.2,
                "loudness_jump_max_input": 1.1,
                "fluctuation_regression": False,
                "loudness_jump_regression": True,
            }
        }
        assert hearing_gate_status(meta) == "yellow"
        details = hearing_gates_details(meta)
        assert any("Lautheits-Sprung-Regression" in d and "4.20 phon" in d for d in details)

    def test_erfuelltes_gate_bleibt_gruen(self) -> None:
        meta = {
            "inviting_gate_song_level": {
                "passed": True,
                "max_vacil": 0.03,
                "max_vacil_input": 0.02,
                "fluctuation_regression": False,
                "loudness_jump_regression": False,
            }
        }
        assert hearing_gate_status(meta) == "green"
        details = hearing_gates_details(meta)
        assert any("Einladungs-Gate (Ebene 4): erfüllt" in d for d in details)

    def test_advisory_passed_false_ohne_regression_bleibt_gruen(self) -> None:
        # Sharpness/Asper-Absolutwerte sind advisory — KEIN Ampel-Befund
        # (Guard-Kalibrierungs-Grundsatz: absolute Normalwerte sind kein Hard-Fail).
        meta = {
            "inviting_gate_song_level": {
                "passed": False,
                "max_asper_in_voice": 0.9,
                "sharpness_jump_max": 0.85,
                "fluctuation_regression": False,
                "loudness_jump_regression": False,
            }
        }
        assert hearing_gate_status(meta) == "green"
        details = hearing_gates_details(meta)
        assert any("advisory" in d for d in details)

    def test_ohne_metadaten_unveraendert(self) -> None:
        assert hearing_gate_status({}) == "green"
        assert hearing_gates_details({}) == ["Hör-Gates: keine Gate-Metadaten (n/a)"]
