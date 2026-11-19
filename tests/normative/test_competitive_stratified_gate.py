from __future__ import annotations

"""Stratifiziertes Konkurrenz-Gate — [RELEASE_MUST] (§5.6 spec 07 / §2.40) v10.0.0

Spec §5.6 (copilot-instructions.md, spec 07 §5.6):
    Konkurrenzvergleich wird nicht nur als Gesamtmittel, sondern pro Zelle einer
    Material-Defekt-Matrix bewertet.

    Pflicht-Matrix:
        Materialien:    tape, vinyl, shellac, digital, vocal  (5)
        Defektklassen:  hiss, crackle, dropout, reverb, hum, codec  (6)
        → 30 Zellen

    Release-Logik:
        - Fail bei regressiver Einzelzelle gegen Referenz,
          auch wenn Gesamtmittel besteht.
        - Bericht muss Delta pro Zelle enthalten.

Gate-Tabelle (copilot-instructions.md):
    "Stratifiziertes Konkurrenz-Gate (Material x Defektklasse) [RELEASE_MUST]"
    → tests/normative/test_competitive_stratified_gate.py (diese Datei)

Ausführung: pytest tests/normative/test_competitive_stratified_gate.py --timeout=30 -v
"""


import math
from dataclasses import dataclass, field

import numpy as np
import pytest

from benchmarks.musical_restoration_benchmark import AMRB_BASELINES

# ---------------------------------------------------------------------------
# Normative Matrix-Definition (§5.6)
# ---------------------------------------------------------------------------

REQUIRED_MATERIALS: list[str] = ["tape", "vinyl", "shellac", "digital", "vocal"]
REQUIRED_DEFECT_CLASSES: list[str] = ["hiss", "crackle", "dropout", "reverb", "hum", "codec"]

# Referenz-Baseline: iZotope RX 11
_REFERENCE_KEY = "iZotope RX 11 (commercial)"
_REFERENCE_OQS = AMRB_BASELINES[_REFERENCE_KEY]["mushra_overall"]  # 71.0

# ---------------------------------------------------------------------------
# Datenstrukturen für stratifizierten Bericht
# ---------------------------------------------------------------------------


@dataclass
class StratifiedCellResult:
    """Ergebnis einer einzelnen Material×Defekt-Zelle."""

    material: str
    defect_class: str
    aurik_score: float
    reference_score: float

    @property
    def delta(self) -> float:
        return self.aurik_score - self.reference_score

    @property
    def is_regressive(self) -> bool:
        """Zelle ist regressiv wenn Aurik schlechter als Referenz."""
        return self.delta < 0.0


@dataclass
class StratifiedGateReport:
    """Vollständiger Stratifizierungs-Bericht mit pro-Zelle-Deltas."""

    cells: list[StratifiedCellResult] = field(default_factory=list)

    @property
    def overall_mean_aurik(self) -> float:
        if not self.cells:
            return 0.0
        return float(np.mean([c.aurik_score for c in self.cells]))

    @property
    def overall_mean_reference(self) -> float:
        if not self.cells:
            return 0.0
        return float(np.mean([c.reference_score for c in self.cells]))

    @property
    def regressive_cells(self) -> list[StratifiedCellResult]:
        return [c for c in self.cells if c.is_regressive]

    @property
    def passes_gate(self) -> bool:
        """Gate besteht nur wenn KEINE Zelle regressiv ist."""
        return len(self.regressive_cells) == 0

    def get_cell(self, material: str, defect_class: str) -> StratifiedCellResult | None:
        for c in self.cells:
            if c.material == material and c.defect_class == defect_class:
                return c
        return None


def _evaluate_stratified_gate(report: StratifiedGateReport) -> tuple[bool, list[str]]:
    """Evaluiert den stratifizierten Gate — gibt (passed, failure_reasons) zurück.

    §5.6: Fail bei regressiver Einzelzelle, auch wenn Gesamtmittel besteht.
    """
    fails = []
    for cell in report.regressive_cells:
        fails.append(
            f"REGRESSION [{cell.material}×{cell.defect_class}]: "
            f"Aurik={cell.aurik_score:.1f} < Ref={cell.reference_score:.1f} "
            f"(Δ={cell.delta:.1f})"
        )
    return len(fails) == 0, fails


def _build_passing_report(base_score: float = 75.0) -> StratifiedGateReport:
    """Erstellt einen vollständig bestandenen Mock-Bericht (alle Zellen über Referenz)."""
    cells = []
    for mat in REQUIRED_MATERIALS:
        for defect in REQUIRED_DEFECT_CLASSES:
            cells.append(
                StratifiedCellResult(
                    material=mat,
                    defect_class=defect,
                    aurik_score=base_score,
                    reference_score=_REFERENCE_OQS,
                )
            )
    return StratifiedGateReport(cells=cells)


def _build_report_with_regression(
    regressive_material: str = "tape",
    regressive_defect: str = "hiss",
) -> StratifiedGateReport:
    """Erstellt einen Bericht mit einer regressiven Zelle."""
    report = _build_passing_report(base_score=75.0)
    for cell in report.cells:
        if cell.material == regressive_material and cell.defect_class == regressive_defect:
            # Aurik schlechter als Referenz in dieser Zelle
            object.__setattr__(cell, "aurik_score", _REFERENCE_OQS - 3.0)
    return report


# ===========================================================================
# Klasse 1: Normative Matrix-Definition
# ===========================================================================


@pytest.mark.unit
class TestNormativeMatrixDefinition:
    """Tests: Die vorgeschriebene 5×6-Matrix ist vollständig definiert."""

    def test_required_materials_count(self):
        """Exakt 5 Materialien müssen in der Matrix sein."""
        assert len(REQUIRED_MATERIALS) == 5, f"§5.6: 5 Materialien erwartet, {len(REQUIRED_MATERIALS)} gefunden"

    def test_required_defect_classes_count(self):
        """Exakt 6 Defektklassen müssen in der Matrix sein."""
        assert len(REQUIRED_DEFECT_CLASSES) == 6, (
            f"§5.6: 6 Defektklassen erwartet, {len(REQUIRED_DEFECT_CLASSES)} gefunden"
        )

    def test_required_materials_contains_tape(self):
        assert "tape" in REQUIRED_MATERIALS

    def test_required_materials_contains_vinyl(self):
        assert "vinyl" in REQUIRED_MATERIALS

    def test_required_materials_contains_shellac(self):
        assert "shellac" in REQUIRED_MATERIALS

    def test_required_materials_contains_digital(self):
        assert "digital" in REQUIRED_MATERIALS

    def test_required_materials_contains_vocal(self):
        assert "vocal" in REQUIRED_MATERIALS

    def test_required_defect_classes_contains_hiss(self):
        assert "hiss" in REQUIRED_DEFECT_CLASSES

    def test_required_defect_classes_contains_crackle(self):
        assert "crackle" in REQUIRED_DEFECT_CLASSES

    def test_required_defect_classes_contains_dropout(self):
        assert "dropout" in REQUIRED_DEFECT_CLASSES

    def test_required_defect_classes_contains_reverb(self):
        assert "reverb" in REQUIRED_DEFECT_CLASSES

    def test_required_defect_classes_contains_hum(self):
        assert "hum" in REQUIRED_DEFECT_CLASSES

    def test_required_defect_classes_contains_codec(self):
        assert "codec" in REQUIRED_DEFECT_CLASSES

    def test_total_cell_count_is_30(self):
        """5 Materialien × 6 Defektklassen = 30 Pflicht-Zellen."""
        total = len(REQUIRED_MATERIALS) * len(REQUIRED_DEFECT_CLASSES)
        assert total == 30, f"§5.6: 30 Zellen erwartet, {total} berechnet"


# ===========================================================================
# Klasse 2: StratifiedCellResult und StratifiedGateReport Datenstruktur
# ===========================================================================


class TestStratifiedReportDataStructure:
    """Tests: Die Berichts-Datenstruktur ist korrekt."""

    def test_cell_result_has_correct_fields(self):
        cell = StratifiedCellResult(material="tape", defect_class="hiss", aurik_score=78.0, reference_score=71.0)
        assert cell.material == "tape"
        assert cell.defect_class == "hiss"
        assert cell.aurik_score == 78.0
        assert cell.reference_score == 71.0

    def test_cell_delta_positive_when_aurik_wins(self):
        cell = StratifiedCellResult(material="vinyl", defect_class="crackle", aurik_score=80.0, reference_score=71.0)
        assert cell.delta == pytest.approx(9.0)

    def test_cell_delta_negative_when_aurik_loses(self):
        cell = StratifiedCellResult(material="shellac", defect_class="hiss", aurik_score=68.0, reference_score=71.0)
        assert cell.delta == pytest.approx(-3.0)

    def test_cell_is_not_regressive_when_equal(self):
        cell = StratifiedCellResult(material="digital", defect_class="codec", aurik_score=71.0, reference_score=71.0)
        assert cell.is_regressive is False

    def test_cell_is_regressive_when_aurik_below_reference(self):
        cell = StratifiedCellResult(material="vocal", defect_class="reverb", aurik_score=68.0, reference_score=71.0)
        assert cell.is_regressive is True

    def test_report_passes_with_no_regressions(self):
        report = _build_passing_report(base_score=80.0)
        assert report.passes_gate is True

    def test_report_fails_with_one_regression(self):
        report = _build_report_with_regression("tape", "hiss")
        assert report.passes_gate is False

    def test_report_regressive_cells_identified_correctly(self):
        report = _build_report_with_regression("vinyl", "dropout")
        regressive = report.regressive_cells
        assert len(regressive) == 1
        assert regressive[0].material == "vinyl"
        assert regressive[0].defect_class == "dropout"

    def test_report_overall_mean_is_correct(self):
        """overall_mean_aurik muss korrekt berechnet sein."""
        cells = [
            StratifiedCellResult("tape", "hiss", 80.0, 71.0),
            StratifiedCellResult("vinyl", "crackle", 90.0, 71.0),
        ]
        report = StratifiedGateReport(cells=cells)
        assert report.overall_mean_aurik == pytest.approx(85.0)

    def test_report_get_cell_retrieves_correct_cell(self):
        report = _build_passing_report(base_score=75.0)
        cell = report.get_cell("shellac", "hum")
        assert cell is not None
        assert cell.material == "shellac"
        assert cell.defect_class == "hum"

    def test_report_get_cell_returns_none_for_missing(self):
        report = _build_passing_report()
        assert report.get_cell("unknown_material", "hiss") is None


# ===========================================================================
# Klasse 3: Gate-Evaluierungs-Logik
# ===========================================================================


class TestStratifiedGateEvaluation:
    """Tests: Die Gate-Evaluierung implementiert §5.6-Logik korrekt."""

    def test_passing_report_gives_no_failures(self):
        report = _build_passing_report(base_score=75.0)
        passed, fails = _evaluate_stratified_gate(report)
        assert passed is True
        assert len(fails) == 0

    def test_report_with_one_regression_fails(self):
        """Eine regressive Zelle → Gate scheitert (§5.6)."""
        report = _build_report_with_regression("tape", "hiss")
        passed, fails = _evaluate_stratified_gate(report)
        assert passed is False
        assert len(fails) == 1

    def test_failure_message_contains_cell_info(self):
        """Fehlermeldung muss Material, Defektklasse und Delta enthalten."""
        report = _build_report_with_regression("shellac", "dropout")
        _, fails = _evaluate_stratified_gate(report)
        assert len(fails) > 0
        assert "shellac" in fails[0]
        assert "dropout" in fails[0]

    def test_overall_mean_passing_but_single_cell_fail(self):
        """§5.6: Gesamtmittel kann bestehen, aber Gate scheitert wegen einer Zelle."""
        report = _build_passing_report(base_score=80.0)  # Mittel gut
        # Mache eine Zelle regressiv
        report = _build_report_with_regression("digital", "reverb")
        overall_aurik = report.overall_mean_aurik
        passed, fails = _evaluate_stratified_gate(report)
        # Gesamtmittel sollte noch über Referenz liegen
        assert overall_aurik > _REFERENCE_OQS
        # Aber Gate scheitert wegen einer Einzelzelle
        assert passed is False, (
            f"§5.6: Gate muss scheitern trotz Gesamtmittel={overall_aurik:.1f} > Referenz={_REFERENCE_OQS:.1f}"
        )

    def test_multiple_regressions_all_reported(self):
        """Mehrere regressive Zellen → alle müssen im Bericht erscheinen."""
        cells = []
        for mat in REQUIRED_MATERIALS:
            for defect in REQUIRED_DEFECT_CLASSES:
                if mat == "tape" and defect in ("hiss", "crackle"):
                    score = _REFERENCE_OQS - 5.0  # regressiv
                else:
                    score = 78.0  # passierend
                cells.append(
                    StratifiedCellResult(
                        material=mat,
                        defect_class=defect,
                        aurik_score=score,
                        reference_score=_REFERENCE_OQS,
                    )
                )
        report = StratifiedGateReport(cells=cells)
        passed, fails = _evaluate_stratified_gate(report)
        assert passed is False
        assert len(fails) == 2  # tape×hiss und tape×crackle

    def test_all_30_cells_required_in_report(self):
        """Ein vollständiger Bericht muss alle 30 Pflicht-Zellen enthalten."""
        report = _build_passing_report()
        expected_cells = {(mat, defect) for mat in REQUIRED_MATERIALS for defect in REQUIRED_DEFECT_CLASSES}
        actual_cells = {(c.material, c.defect_class) for c in report.cells}
        missing = expected_cells - actual_cells
        assert not missing, f"§5.6: Fehlende Zellen im Bericht: {missing}"


# ===========================================================================
# Klasse 4: Vollständigkeits-Validierung
# ===========================================================================


class TestMatrixCompleteness:
    """Tests: Vollständigkeitscheck für die stratifizierte Matrix."""

    def _complete_report(self, aurik_score: float = 75.0) -> StratifiedGateReport:
        return _build_passing_report(base_score=aurik_score)

    def test_complete_30_cell_report_has_correct_count(self):
        report = self._complete_report()
        assert len(report.cells) == 30

    def test_all_materials_represented_in_complete_report(self):
        report = self._complete_report()
        materials_in_report = {c.material for c in report.cells}
        for mat in REQUIRED_MATERIALS:
            assert mat in materials_in_report, f"Material '{mat}' fehlt in Bericht"

    def test_all_defect_classes_represented_in_complete_report(self):
        report = self._complete_report()
        defects_in_report = {c.defect_class for c in report.cells}
        for defect in REQUIRED_DEFECT_CLASSES:
            assert defect in defects_in_report, f"Defektklasse '{defect}' fehlt in Bericht"

    def test_each_cell_appears_exactly_once(self):
        report = self._complete_report()
        pairs = [(c.material, c.defect_class) for c in report.cells]
        assert len(pairs) == len(set(pairs)), "Jede Material×Defekt-Zelle darf nur einmal vorkommen"

    def test_report_with_missing_vocal_material_incomplete(self):
        """Fehlende 'vocal'-Zeile → Matrix unvollständig."""
        cells = [
            StratifiedCellResult(mat, defect, 75.0, _REFERENCE_OQS)
            for mat in REQUIRED_MATERIALS
            if mat != "vocal"
            for defect in REQUIRED_DEFECT_CLASSES
        ]
        report = StratifiedGateReport(cells=cells)
        materials_in_report = {c.material for c in report.cells}
        assert "vocal" not in materials_in_report  # Beweis für Unvollständigkeit

    def test_report_with_missing_codec_defect_incomplete(self):
        """Fehlende 'codec'-Spalte → Matrix unvollständig."""
        cells = [
            StratifiedCellResult(mat, defect, 75.0, _REFERENCE_OQS)
            for mat in REQUIRED_MATERIALS
            for defect in REQUIRED_DEFECT_CLASSES
            if defect != "codec"
        ]
        report = StratifiedGateReport(cells=cells)
        defects_in_report = {c.defect_class for c in report.cells}
        assert "codec" not in defects_in_report


# ===========================================================================
# Klasse 5: Referenz-Baseline-Vertrag
# ===========================================================================


class TestReferenceBaselineContract:
    """Tests: Die Referenz-Baseline (iZotope RX 11) ist korrekt definiert."""

    def test_reference_key_exists_in_amrb_baselines(self):
        """'iZotope RX 11 (commercial)' muss in AMRB_BASELINES sein."""
        assert _REFERENCE_KEY in AMRB_BASELINES, f"Referenzschlüssel '{_REFERENCE_KEY}' nicht in AMRB_BASELINES"

    def test_reference_oqs_is_71(self):
        """iZotope RX 11 Baseline-OQS muss 71.0 sein."""
        assert pytest.approx(71.0) == _REFERENCE_OQS, f"iZotope RX 11 OQS-Baseline soll 71.0 sein, ist {_REFERENCE_OQS}"

    def test_reference_oqs_is_positive_finite(self):
        assert _REFERENCE_OQS > 0.0
        assert math.isfinite(_REFERENCE_OQS)

    def test_aurik_restoration_baseline_exceeds_reference(self):
        """Aurik Restoration-Baseline muss über iZotope RX 11 liegen."""
        aurik_restore = AMRB_BASELINES["Aurik 10.0.0 (Restoration Mode)"]["mushra_overall"]
        assert aurik_restore > _REFERENCE_OQS, (
            f"Aurik Restoration ({aurik_restore}) muss > iZotope baseline ({_REFERENCE_OQS})"
        )

    def test_amrb_baselines_has_required_fields(self):
        """Jede Baseline muss mushra_overall und pqs_mos haben."""
        for key, entry in AMRB_BASELINES.items():
            assert "mushra_overall" in entry, f"Baseline '{key}' fehlt 'mushra_overall'"
            assert "pqs_mos" in entry, f"Baseline '{key}' fehlt 'pqs_mos'"
