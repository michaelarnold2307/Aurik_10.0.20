import pytest

"""
Tests für PhaseDAG / validate_phase_order (§7.5a).
"""


@pytest.mark.unit
class TestPhaseDagImport:
    def test_import_ok(self):
        from backend.core.phase_dag import HARD_BEFORE_CONSTRAINTS, validate_phase_order

        assert callable(validate_phase_order)
        assert isinstance(HARD_BEFORE_CONSTRAINTS, list)
        assert len(HARD_BEFORE_CONSTRAINTS) > 0


class TestValidatePhaseOrder:
    def test_empty_list_no_violations(self):
        from backend.core.phase_dag import validate_phase_order

        violations = validate_phase_order([])
        assert violations == []

    def test_correct_order_no_violations(self):
        """Korrekte Reihenfolge: phase_03 vor phase_07 → keine Verletzung."""
        from backend.core.phase_dag import validate_phase_order

        phases = [
            "phase_01_click_removal",
            "phase_03_denoise",
            "phase_06_bw_extension",
            "phase_07_harmonic_restoration",
        ]
        violations = validate_phase_order(phases)
        assert violations == [], f"Unerwartete Verletzungen: {violations}"

    def test_wrong_order_detected(self):
        """phase_07 vor phase_03 muss als Verletzung erkannt werden."""
        from backend.core.phase_dag import validate_phase_order

        phases = [
            "phase_07_harmonic_restoration",
            "phase_03_denoise",
        ]
        violations = validate_phase_order(phases)
        assert len(violations) > 0, "Reihenfolge-Verletzung sollte erkannt werden"
        assert any("phase_07" in v or "phase_03" in v for v in violations)

    def test_phase_29_before_phase_07(self):
        """phase_29 (tape hiss) MUSS vor phase_07 sein (§7.5a kritische Kette)."""
        from backend.core.phase_dag import validate_phase_order

        # Korrekte Reihenfolge
        correct = ["phase_29_tape_hiss_reduction", "phase_07_harmonic_restoration"]
        assert validate_phase_order(correct) == []

        # Falsche Reihenfolge
        wrong = ["phase_07_harmonic_restoration", "phase_29_tape_hiss_reduction"]
        violations = validate_phase_order(wrong)
        assert len(violations) > 0

    def test_only_one_phase_no_violations(self):
        """Einzelne Phase → keine Constraint-Verletzung möglich."""
        from backend.core.phase_dag import validate_phase_order

        violations = validate_phase_order(["phase_07_harmonic_restoration"])
        assert violations == []

    def test_short_form_phase_ids(self):
        """Kurzform phase_03 wird korrekt normiert."""
        from backend.core.phase_dag import validate_phase_order

        phases = ["phase_03", "phase_07"]
        violations = validate_phase_order(phases)
        # Korrekte Reihenfolge → keine Verletzung
        assert violations == []

    def test_returns_list_of_strings(self):
        from backend.core.phase_dag import validate_phase_order

        result = validate_phase_order(["phase_07_harmonic_restoration", "phase_03_denoise"])
        assert isinstance(result, list)
        for v in result:
            assert isinstance(v, str)


class TestCheckConflict:
    def test_no_conflict_for_unrelated_phases(self):
        from backend.core.phase_dag import check_conflict

        result = check_conflict("phase_01_click_removal", "phase_09_crackle_removal")
        assert result is None

    def test_known_conflict_detected(self):
        """phase_06 und phase_23 sind im CONFLICT."""
        from backend.core.phase_dag import check_conflict

        result = check_conflict("phase_06_bw_extension", "phase_23_audio_sr_upsampling")
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0

    def test_conflict_bidirectional(self):
        """Konflikt gilt in beide Richtungen."""
        from backend.core.phase_dag import check_conflict

        a_b = check_conflict("phase_06_bw_extension", "phase_23_audio_sr_upsampling")
        b_a = check_conflict("phase_23_audio_sr_upsampling", "phase_06_bw_extension")
        assert a_b is not None
        assert b_a is not None


class TestParallelClass:
    def test_stereo_phases_class_a(self):
        from backend.core.phase_dag import get_parallel_class

        assert get_parallel_class("phase_14_stereo_width") == "A"
        assert get_parallel_class("phase_15_stereo_field_repair") == "A"
        assert get_parallel_class("phase_25_azimuth_correction") == "A"

    def test_local_defect_phases_class_b(self):
        from backend.core.phase_dag import get_parallel_class

        assert get_parallel_class("phase_09_crackle_removal") == "B"
        assert get_parallel_class("phase_24_dropout_repair") == "B"

    def test_unknown_phase_none(self):
        from backend.core.phase_dag import get_parallel_class

        assert get_parallel_class("phase_99_unknown") is None


class TestSortPhasesByDag:
    """Tests für sort_phases_by_dag() — topologischer Sort nach HARD_BEFORE-Constraints."""

    def test_import_ok(self):
        from backend.core.phase_dag import sort_phases_by_dag

        assert callable(sort_phases_by_dag)

    def test_empty_list(self):
        from backend.core.phase_dag import sort_phases_by_dag

        assert sort_phases_by_dag([]) == []

    def test_single_phase(self):
        from backend.core.phase_dag import sort_phases_by_dag

        result = sort_phases_by_dag(["phase_07_harmonic_restoration"])
        assert result == ["phase_07_harmonic_restoration"]

    def test_correct_order_preserved(self):
        """Bereits korrekte Reihenfolge bleibt erhalten."""
        from backend.core.phase_dag import sort_phases_by_dag, validate_phase_order

        phases = [
            "phase_01_click_removal",
            "phase_03_denoise",
            "phase_06_frequency_restoration",
            "phase_07_harmonic_restoration",
        ]
        result = sort_phases_by_dag(phases)
        assert validate_phase_order(result) == [], f"Verletzungen nach Sort: {validate_phase_order(result)}"

    def test_wrong_order_fixed(self):
        """phase_07 vor phase_03 → Sort stellt wissenschaftliche Reihenfolge her."""
        from backend.core.phase_dag import sort_phases_by_dag, validate_phase_order

        wrong_order = [
            "phase_07_harmonic_restoration",
            "phase_03_denoise",
        ]
        result = sort_phases_by_dag(wrong_order)
        assert validate_phase_order(result) == [], f"Verletzungen nach Sort: {validate_phase_order(result)}"
        # phase_03 muss vor phase_07 sein
        idx_03 = next(i for i, p in enumerate(result) if "phase_03" in p)
        idx_07 = next(i for i, p in enumerate(result) if "phase_07" in p)
        assert idx_03 < idx_07, "phase_03 muss vor phase_07 kommen"

    def test_phase_24_before_phase_06_hard_before(self):
        """Kritische HARD_BEFORE-Verletzung: phase_24 muss VOR phase_06 laufen.
        Numerischer Sort würde 06 vor 24 setzen — DAG-Sort korrigiert das.
        """
        from backend.core.phase_dag import sort_phases_by_dag, validate_phase_order

        # Numerischer Sort würde phase_06 (6) vor phase_24 (24) setzen — FALSCH
        phases = [
            "phase_06_frequency_restoration",
            "phase_24_dropout_repair",
        ]
        result = sort_phases_by_dag(phases)
        violations = validate_phase_order(result)
        assert violations == [], f"HARD_BEFORE phase_24→phase_06 verletzt: {violations}"
        idx_24 = next(i for i, p in enumerate(result) if "phase_24" in p)
        idx_06 = next(i for i, p in enumerate(result) if "phase_06" in p)
        assert idx_24 < idx_06, "phase_24_dropout_repair muss VOR phase_06 laufen (Dropout-Reparatur vor BW-Extension)"

    def test_phase_12_before_phase_25_respected(self):
        """phase_12 (Wow/Flutter) muss vor phase_25 (Azimuth) laufen."""
        from backend.core.phase_dag import sort_phases_by_dag, validate_phase_order

        phases = ["phase_25_azimuth_correction", "phase_12_wow_flutter_fix"]
        result = sort_phases_by_dag(phases)
        assert validate_phase_order(result) == []
        idx_12 = next(i for i, p in enumerate(result) if "phase_12" in p)
        idx_25 = next(i for i, p in enumerate(result) if "phase_25" in p)
        assert idx_12 < idx_25

    def test_phase_40_amplitude_drift_stage_45_ordering(self):
        """Phase 40 läuft nach Carrier-NR und vor additiver BW-/Harmonik-Restauration."""
        from backend.core.phase_dag import sort_phases_by_dag, validate_phase_order

        phases = [
            "phase_07_harmonic_restoration",
            "phase_40_loudness_normalization",
            "phase_06_frequency_restoration",
            "phase_29_tape_hiss_reduction",
            "phase_03_denoise",
        ]
        result = sort_phases_by_dag(phases)
        assert validate_phase_order(result) == []
        idx_03 = result.index("phase_03_denoise")
        idx_29 = result.index("phase_29_tape_hiss_reduction")
        idx_40 = result.index("phase_40_loudness_normalization")
        idx_06 = result.index("phase_06_frequency_restoration")
        idx_07 = result.index("phase_07_harmonic_restoration")
        assert idx_03 < idx_40
        assert idx_29 < idx_40
        assert idx_40 < idx_06
        assert idx_40 < idx_07

    def test_finalizer_chain_lufs_truepeak_format_ordering(self):
        """Finaler Exportpfad: Polish vor LUFS/Gain, dann TruePeak, dann Format/Dither."""
        from backend.core.phase_dag import sort_phases_by_dag, validate_phase_order

        phases = [
            "phase_41_output_format_optimization",
            "phase_47_truepeak_limiter",
            "phase_40_loudness_normalization",
            "phase_17_mastering_polish",
        ]
        result = sort_phases_by_dag(phases)
        assert validate_phase_order(result) == []
        idx_17 = result.index("phase_17_mastering_polish")
        idx_40 = result.index("phase_40_loudness_normalization")
        idx_47 = result.index("phase_47_truepeak_limiter")
        idx_41 = result.index("phase_41_output_format_optimization")
        assert idx_17 < idx_40 < idx_47 < idx_41

    def test_full_chain_complex(self):
        """Komplexe Kette aus allen Stufen — muss alle HARD_BEFORE-Constraints erfüllen."""
        from backend.core.phase_dag import sort_phases_by_dag, validate_phase_order

        # Absichtlich in falscher Reihenfolge übergeben
        phases = [
            "phase_07_harmonic_restoration",
            "phase_29_tape_hiss_reduction",
            "phase_12_wow_flutter_fix",
            "phase_06_frequency_restoration",
            "phase_24_dropout_repair",
            "phase_09_crackle_removal",
            "phase_03_denoise",
            "phase_01_click_removal",
            "phase_25_azimuth_correction",
        ]
        result = sort_phases_by_dag(phases)
        violations = validate_phase_order(result)
        assert violations == [], f"Verletzungen nach Sort: {violations}"

    def test_preserves_all_phases(self):
        """Kein Phase darf verloren gehen."""
        from backend.core.phase_dag import sort_phases_by_dag

        phases = [
            "phase_07_harmonic_restoration",
            "phase_03_denoise",
            "phase_06_frequency_restoration",
            "phase_29_tape_hiss_reduction",
            "phase_01_click_removal",
        ]
        result = sort_phases_by_dag(phases)
        assert set(result) == set(phases)
        assert len(result) == len(phases)

    def test_numeric_tiebreaker_without_constraints(self):
        """Phasen ohne gegenseitige Constraints werden numerisch sortiert."""
        from backend.core.phase_dag import sort_phases_by_dag

        # phase_20, phase_16, phase_04 haben keine gegenseitigen HARD_BEFORE-Constraints
        phases = ["phase_20_reverb_reduction", "phase_04_eq_correction", "phase_16_final_eq"]
        result = sort_phases_by_dag(phases)
        nums = [int(p.split("_")[1]) for p in result]
        assert nums == sorted(nums), f"Numerischer Tiebreaker verletzt: {result}"
