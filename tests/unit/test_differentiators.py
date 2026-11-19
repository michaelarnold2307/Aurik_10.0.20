from __future__ import annotations

"""
tests/unit/test_differentiators.py
===================================
Unit-Tests für die 7 Weltspitzen-Differenzierer von Aurik 10.0.0.

Testet:
  1. CausalDefectGraph  — Kausale Reparaturreihenfolge
  2. PhysicalMediumChainModel — Physikalische Ketteninversion
  3. DefectQualityReport / DefectQualityReporter — Defektprotokoll
  4. ProvenanceAudit — Archivtaugliches Audit
  5. ARE Integration — Alle Differenzierer im ARE-Workflow
"""


import json

import numpy as np

np.random.seed(42)  # §5.4 Reproduzierbarkeit
import pytest

from backend.core.causal_defect_graph import CausalDefectGraph, CausalNode
from backend.core.defect_quality_report import (
    DefectQualityReport,
    DefectQualityReporter,
    DefectRepairEntry,
)
from backend.core.defect_scanner import DefectScore, DefectType, MaterialType
from backend.core.medium_chain_model import ChainInversionResult, PhysicalMediumChainModel
from backend.core.provenance_audit import ProvenanceAudit, ProvenanceEntry, _audio_hash

# ===========================================================================
# Fixtures
# ===========================================================================


def _make_score(defect_type: DefectType, severity: float, confidence: float = 0.9) -> DefectScore:
    return DefectScore(
        defect_type=defect_type,
        severity=severity,
        confidence=confidence,
    )


_SR = 44100
_N = _SR // 4  # 11025 Samples (0.25s) — reicht für alle DSP-Operationen


@pytest.fixture(scope="module")
def mono_audio() -> np.ndarray:
    """Synthetisches Mono-Audio (0.25s, 44100 Hz). scope=module: 1× pro Worker."""
    rng = np.random.default_rng(42)
    return np.clip(rng.normal(0, 0.1, _N).astype(np.float32), -1, 1)


@pytest.fixture(scope="module")
def stereo_audio() -> np.ndarray:
    """Synthetisches Stereo-Audio (0.25s, 44100 Hz). scope=module: 1× pro Worker."""
    rng = np.random.default_rng(42)
    return np.clip(rng.normal(0, 0.1, (_N, 2)).astype(np.float32), -1, 1)


@pytest.fixture(scope="module")
def sample_rate() -> int:
    return _SR


@pytest.fixture(scope="module")
def simple_defects() -> list[DefectScore]:
    """Typische Vinyl-Defekte. scope=module: einmalig erstellt."""
    return [
        _make_score(DefectType.CLICKS, 0.7),
        _make_score(DefectType.CRACKLE, 0.5),
        _make_score(DefectType.HUM, 0.4),
        _make_score(DefectType.LOW_FREQ_RUMBLE, 0.3),
    ]


@pytest.fixture(scope="module")
def causal_defects() -> list[DefectScore]:
    """Defekte mit kausalen Abhängigkeiten. scope=module: einmalig erstellt."""
    return [
        _make_score(DefectType.DROPOUTS, 0.8),
        _make_score(DefectType.CLICKS, 0.6),
        _make_score(DefectType.HUM, 0.5),
        _make_score(DefectType.LOW_FREQ_RUMBLE, 0.3),
    ]


# ===========================================================================
# 1. CausalDefectGraph Tests
# ===========================================================================


class TestCausalDefectGraph:
    def test_init(self):
        graph = CausalDefectGraph()
        assert isinstance(graph.CAUSAL_EDGES, dict)
        assert len(graph.CAUSAL_EDGES) > 0

    def test_empty_input(self):
        graph = CausalDefectGraph()
        result = graph.resolve_causal_order([])
        assert result == []

    def test_single_defect(self):
        graph = CausalDefectGraph()
        defects = [_make_score(DefectType.CLICKS, 0.8)]
        result = graph.resolve_causal_order(defects)
        assert len(result) == 1
        assert result[0].defect_type == DefectType.CLICKS

    def test_causal_order_root_before_symptom(self, causal_defects):
        """DROPOUTS muss vor CLICKS kommen (DROPOUTS → CLICKS)."""
        graph = CausalDefectGraph()
        result = graph.resolve_causal_order(causal_defects)
        types = [d.defect_type for d in result]

        assert DefectType.DROPOUTS in types
        assert DefectType.CLICKS in types
        # Root cause DROPOUTS muss vor Symptom CLICKS kommen
        assert types.index(DefectType.DROPOUTS) < types.index(DefectType.CLICKS)

    def test_causal_order_hum_before_rumble(self, causal_defects):
        """HUM muss vor LOW_FREQ_RUMBLE kommen (HUM → LOW_FREQ_RUMBLE)."""
        graph = CausalDefectGraph()
        result = graph.resolve_causal_order(causal_defects)
        types = [d.defect_type for d in result]

        assert types.index(DefectType.HUM) < types.index(DefectType.LOW_FREQ_RUMBLE)

    def test_all_defects_preserved(self, causal_defects, simple_defects):
        """Alle erkannten Defekte müssen in der Ausgabe vorhanden sein."""
        graph = CausalDefectGraph()
        for defects in [causal_defects, simple_defects]:
            result = graph.resolve_causal_order(defects)
            assert len(result) == len(defects)
            in_types = {d.defect_type for d in result}
            exp_types = {d.defect_type for d in defects}
            assert in_types == exp_types

    def test_build_returns_nodes(self, causal_defects):
        graph = CausalDefectGraph()
        nodes = graph.build(causal_defects)
        assert len(nodes) == len(causal_defects)
        for node in nodes:
            assert isinstance(node, CausalNode)
            assert hasattr(node, "defect_score")
            assert hasattr(node, "is_phantom")
            assert hasattr(node, "causal_note")

    def test_phantom_detection(self, causal_defects):
        """CLICKS ist Phantom von DROPOUTS, LOW_FREQ_RUMBLE Phantom von HUM."""
        graph = CausalDefectGraph()
        phantoms = graph.get_phantom_defects(causal_defects)
        assert DefectType.CLICKS in phantoms
        assert DefectType.LOW_FREQ_RUMBLE in phantoms
        # Root causes sind keine Phantome
        assert DefectType.DROPOUTS not in phantoms
        assert DefectType.HUM not in phantoms

    def test_explain_returns_string(self, causal_defects):
        graph = CausalDefectGraph()
        explanation = graph.explain(causal_defects)
        assert isinstance(explanation, str)
        assert len(explanation) > 50
        assert "Kausale Defekt-Analyse" in explanation

    def test_explain_contains_all_defects(self, causal_defects):
        graph = CausalDefectGraph()
        explanation = graph.explain(causal_defects)
        for d in causal_defects:
            assert d.defect_type.value in explanation

    def test_wow_flutter_causes_bandwidth_loss(self):
        """WOW → BANDWIDTH_LOSS: WOW muss zuerst repariert werden (getrennte WOW/FLUTTER-Semantik)."""
        graph = CausalDefectGraph()
        defects = [
            _make_score(DefectType.WOW, 0.7),
            _make_score(DefectType.BANDWIDTH_LOSS, 0.5),
        ]
        result = graph.resolve_causal_order(defects)
        types = [d.defect_type for d in result]
        assert types.index(DefectType.WOW) < types.index(DefectType.BANDWIDTH_LOSS)

    def test_crackle_causes_clicks(self):
        """CRACKLE → CLICKS: Crackle-Bursts erzeugen Click-artige Transienten."""
        graph = CausalDefectGraph()
        defects = [
            _make_score(DefectType.CRACKLE, 0.8),
            _make_score(DefectType.CLICKS, 0.5),
        ]
        result = graph.resolve_causal_order(defects)
        types = [d.defect_type for d in result]
        assert types.index(DefectType.CRACKLE) < types.index(DefectType.CLICKS)

    def test_crackle_causes_high_freq_noise(self):
        """CRACKLE → HIGH_FREQ_NOISE: Vinyl-Oberfläche erhöht HF-Rauschboden."""
        graph = CausalDefectGraph()
        defects = [
            _make_score(DefectType.CRACKLE, 0.7),
            _make_score(DefectType.HIGH_FREQ_NOISE, 0.4),
        ]
        result = graph.resolve_causal_order(defects)
        types = [d.defect_type for d in result]
        assert types.index(DefectType.CRACKLE) < types.index(DefectType.HIGH_FREQ_NOISE)

    def test_crackle_edges_exist_in_graph(self):
        """CausalDefectGraph enthält CRACKLE-Kanten."""
        graph = CausalDefectGraph()
        assert DefectType.CRACKLE in graph.CAUSAL_EDGES
        crackle_effects = graph.CAUSAL_EDGES[DefectType.CRACKLE]
        assert DefectType.CLICKS in crackle_effects
        assert DefectType.HIGH_FREQ_NOISE in crackle_effects

    def test_crackle_is_phantom_root_not_symptom(self):
        """CRACKLE ist Root Cause wenn CLICKS und HF-NOISE erkannt werden."""
        graph = CausalDefectGraph()
        defects = [
            _make_score(DefectType.CRACKLE, 0.8),
            _make_score(DefectType.CLICKS, 0.5),
            _make_score(DefectType.HIGH_FREQ_NOISE, 0.3),
        ]
        phantoms = graph.get_phantom_defects(defects)
        # CRACKLE ist Root Cause → kein Phantom
        assert DefectType.CRACKLE not in phantoms
        # CLICKS und HIGH_FREQ_NOISE sind Symptome von CRACKLE → Phantome
        assert DefectType.CLICKS in phantoms
        assert DefectType.HIGH_FREQ_NOISE in phantoms

    def test_severity_tiebreaker(self):
        """Bei gleichem In-Degree: höherer Schweregrad zuerst."""
        graph = CausalDefectGraph()
        defects = [
            _make_score(DefectType.CLIPPING, 0.3),
            _make_score(DefectType.HUM, 0.9),
        ]
        result = graph.resolve_causal_order(defects)
        # Beide unabhängig → höherer Schweregrad zuerst
        assert result[0].severity >= result[1].severity


# ===========================================================================
# 2. PhysicalMediumChainModel Tests
# ===========================================================================


class TestPhysicalMediumChainModel:
    def test_init(self):
        model = PhysicalMediumChainModel()
        assert isinstance(model.SHELLAC_EQ_CURVES, dict)
        assert isinstance(model.TAPE_HF_ROLLOFF, dict)

    def test_shellac_inversion_returns_result(self, stereo_audio, sample_rate):
        model = PhysicalMediumChainModel()
        result = model.invert_chain(stereo_audio, sample_rate, MaterialType.SHELLAC)
        assert isinstance(result, ChainInversionResult)
        assert result.audio.shape == stereo_audio.shape
        assert result.audio.dtype == stereo_audio.dtype
        assert len(result.corrections_applied) > 0
        assert result.material == MaterialType.SHELLAC

    def test_vinyl_inversion_returns_result(self, stereo_audio, sample_rate):
        model = PhysicalMediumChainModel()
        result = model.invert_chain(stereo_audio, sample_rate, MaterialType.VINYL)
        assert isinstance(result, ChainInversionResult)
        assert result.audio.shape == stereo_audio.shape
        assert "subsonic_filter_20hz" in result.corrections_applied

    def test_tape_inversion_returns_result(self, stereo_audio, sample_rate):
        model = PhysicalMediumChainModel()
        result = model.invert_chain(stereo_audio, sample_rate, MaterialType.TAPE)
        assert isinstance(result, ChainInversionResult)
        assert any("bias_hf_rolloff" in c for c in result.corrections_applied)

    def test_reel_tape_inversion(self, mono_audio, sample_rate):
        model = PhysicalMediumChainModel()
        result = model.invert_chain(mono_audio, sample_rate, MaterialType.REEL_TAPE)
        assert result.audio.shape == mono_audio.shape

    def test_cd_inversion_returns_result(self, stereo_audio, sample_rate):
        model = PhysicalMediumChainModel()
        result = model.invert_chain(stereo_audio, sample_rate, MaterialType.CD_DIGITAL)
        assert isinstance(result, ChainInversionResult)
        assert "reconstruction_filter_preringing_reduction" in result.corrections_applied

    def test_no_clipping_after_inversion(self, stereo_audio, sample_rate):
        """Ketteninversion darf kein Clipping erzeugen."""
        model = PhysicalMediumChainModel()
        for material in [MaterialType.SHELLAC, MaterialType.VINYL, MaterialType.TAPE]:
            result = model.invert_chain(stereo_audio, sample_rate, material)
            peak = float(np.max(np.abs(result.audio)))
            assert peak <= 1.01, f"Clipping nach {material.value}: peak={peak:.3f}"

    def test_mono_audio_inversion(self, mono_audio, sample_rate):
        """Alle Materialtypen müssen mit Mono-Audio funktionieren."""
        model = PhysicalMediumChainModel()
        for material in MaterialType:
            result = model.invert_chain(mono_audio, sample_rate, material)
            assert result.audio.shape == mono_audio.shape

    def test_spectral_change_measured(self, stereo_audio, sample_rate):
        """Spektrale Änderung muss gemessen werden."""
        model = PhysicalMediumChainModel()
        result = model.invert_chain(stereo_audio, sample_rate, MaterialType.SHELLAC)
        assert isinstance(result.spectral_change_db, float)
        assert result.spectral_change_db >= 0.0

    def test_with_detected_defects(self, stereo_audio, sample_rate, simple_defects):
        """Ketteninversion funktioniert mit erkannten Defekten."""
        model = PhysicalMediumChainModel()
        result = model.invert_chain(stereo_audio, sample_rate, MaterialType.VINYL, simple_defects)
        assert isinstance(result, ChainInversionResult)

    def test_cd_with_jitter_defect(self, stereo_audio, sample_rate):
        """Jitter-Unterdrückung bei erkanntem JITTER_ARTIFACTS."""
        model = PhysicalMediumChainModel()
        jitter_defects = [_make_score(DefectType.JITTER_ARTIFACTS, 0.6)]
        result = model.invert_chain(stereo_audio, sample_rate, MaterialType.CD_DIGITAL, jitter_defects)
        # Jitter-Suppression soll in Korrekturen enthalten sein
        assert any("jitter" in c for c in result.corrections_applied)


# ===========================================================================
# 3. DefectQualityReport / DefectQualityReporter Tests
# ===========================================================================


class TestDefectQualityReport:
    def test_empty_report(self):
        report = DefectQualityReport()
        assert len(report.entries) == 0
        assert report.total_snr_improvement_db == 0.0
        assert report.musical_context_preservation_rate == 1.0
        assert report.mean_confidence == 0.0
        assert report.worst_entry is None
        assert report.best_entry is None

    def test_add_entry(self):
        report = DefectQualityReport(material_type="vinyl", mode="RESTORATION")
        entry = DefectRepairEntry(
            defect_type=DefectType.CLICKS,
            severity_before=0.7,
            severity_after=0.1,
            confidence=0.95,
            snr_before_db=25.0,
            snr_after_db=40.0,
            snr_improvement_db=15.0,
            musical_context_preserved=True,
            context_note="Korrelation=0.97",
            repair_method="phase_01_click_removal",
            phase_id=1,
            processing_time_ms=45.3,
            timestamp_seconds=0.0,
        )
        report.add_entry(entry)
        assert len(report.entries) == 1
        assert report.total_snr_improvement_db == 15.0
        assert report.musical_context_preservation_rate == 1.0

    def test_to_dict_structure(self):
        report = DefectQualityReport(material_type="shellac", mode="STUDIO_2026")
        d = report.to_dict()
        assert "meta" in d
        assert "summary" in d
        assert "defect_repairs" in d
        assert d["meta"]["material_type"] == "shellac"

    def test_to_text_report_with_entries(self):
        report = DefectQualityReport(material_type="tape", mode="RESTORATION")
        entry = DefectRepairEntry(
            defect_type=DefectType.HUM,
            severity_before=0.5,
            severity_after=0.05,
            confidence=0.88,
            snr_before_db=18.0,
            snr_after_db=35.0,
            snr_improvement_db=17.0,
            musical_context_preserved=True,
            context_note="OK",
            repair_method="phase_02_hum_removal",
            phase_id=2,
            processing_time_ms=12.0,
            timestamp_seconds=0.5,
        )
        report.add_entry(entry)
        text = report.to_text_report()
        assert "AURIK 9.0" in text
        assert "hum" in text.lower()
        assert "+17" in text or "17.0" in text

    def test_worst_and_best_entry(self):
        report = DefectQualityReport()
        entries = [
            DefectRepairEntry(DefectType.CLICKS, 0.7, 0.1, 0.9, 20.0, 35.0, 15.0, True, "", "method", 1, 10.0, 0.0),
            DefectRepairEntry(DefectType.HUM, 0.5, 0.05, 0.85, 22.0, 44.0, 22.0, True, "", "method", 2, 8.0, 0.0),
            DefectRepairEntry(
                DefectType.CRACKLE, 0.6, 0.2, 0.80, 15.0, 20.0, 5.0, False, "Kontextverlust", "method", 9, 50.0, 0.0
            ),
        ]
        for e in entries:
            report.add_entry(e)

        assert report.best_entry is not None
        assert report.worst_entry is not None
        assert report.best_entry.defect_type == DefectType.HUM
        assert report.worst_entry.defect_type == DefectType.CRACKLE
        assert report.musical_context_preservation_rate == pytest.approx(2 / 3, abs=0.01)


class TestDefectQualityReporter:
    def test_measure_repair_returns_entry(self, mono_audio, sample_rate):
        reporter = DefectQualityReporter()
        audio_after = mono_audio * 0.95  # Leicht verändert
        entry = reporter.measure_repair(
            audio_before=mono_audio,
            audio_after=audio_after,
            sample_rate=sample_rate,
            defect_type=DefectType.CLICKS,
            severity_before=0.7,
            confidence=0.92,
            phase_id=1,
            repair_method="click_removal_v2",
            processing_time_ms=35.0,
        )
        assert isinstance(entry, DefectRepairEntry)
        assert entry.defect_type == DefectType.CLICKS
        assert isinstance(entry.snr_improvement_db, float)
        assert isinstance(entry.musical_context_preserved, bool)
        assert entry.confidence == 0.92
        assert entry.phase_id == 1

    def test_snr_estimate_positive(self, mono_audio, sample_rate):
        reporter = DefectQualityReporter()
        snr = reporter._estimate_snr(mono_audio, sample_rate)
        assert isinstance(snr, float)
        assert snr > 0.0  # SNR muss positiv sein

    def test_musical_context_preserved_similar_audio(self, mono_audio, sample_rate):
        reporter = DefectQualityReporter()
        # Sehr ähnliches Audio → Kontext erhalten
        audio_similar = mono_audio + np.random.normal(0, 0.001, mono_audio.shape).astype(np.float32)
        ok, note = reporter._check_musical_context(mono_audio, audio_similar, sample_rate)
        assert ok is True

    def test_musical_context_not_preserved_random_audio(self, mono_audio, sample_rate):
        reporter = DefectQualityReporter()
        rng = np.random.default_rng(99)
        completely_different = rng.normal(0, 0.3, mono_audio.shape).astype(np.float32)
        ok, note = reporter._check_musical_context(mono_audio, completely_different, sample_rate)
        assert ok is False


# ===========================================================================
# 4. ProvenanceAudit Tests
# ===========================================================================


class TestProvenanceAudit:
    def test_init(self):
        audit = ProvenanceAudit(source_file="test.wav", material="vinyl", mode="RESTORATION")
        assert len(audit) == 0
        assert audit.material == "vinyl"
        assert audit.mode == "RESTORATION"

    def test_record_creates_entry(self, mono_audio, sample_rate):
        audit = ProvenanceAudit()
        entry = audit.record(
            step="test_step",
            audio_in=mono_audio,
            audio_out=mono_audio,
            sample_rate=sample_rate,
            rationale="Testschritt",
            confidence=0.9,
        )
        assert isinstance(entry, ProvenanceEntry)
        assert len(audit) == 1
        assert entry.step == "test_step"
        assert entry.confidence == 0.9

    def test_record_decision(self):
        audit = ProvenanceAudit()
        entry = audit.record_decision(
            step="material_detection",
            rationale="Shellac erkannt: Bass-Pickup-Ratio > 20 dB",
            confidence=0.87,
            parameters={"material": "shellac"},
        )
        assert isinstance(entry, ProvenanceEntry)
        assert len(audit) == 1
        assert "shellac" in entry.parameters.get("material", "")

    def test_to_jsonl_is_valid(self, mono_audio, sample_rate):
        audit = ProvenanceAudit(source_file="beethoven.wav", material="shellac")
        audit.record(
            step="step_1",
            audio_in=mono_audio,
            audio_out=mono_audio,
            sample_rate=sample_rate,
            rationale="Erster Schritt",
            confidence=0.95,
        )
        jsonl = audit.to_jsonl()
        lines = jsonl.strip().split("\n")
        assert len(lines) == 2  # Header + 1 entry
        for line in lines:
            obj = json.loads(line)  # Muss valides JSON sein
            assert isinstance(obj, dict)

    def test_to_jsonl_header_fields(self, mono_audio, sample_rate):
        audit = ProvenanceAudit(source_file="beethoven.wav", material="shellac")
        jsonl = audit.to_jsonl()
        header = json.loads(jsonl.split("\n")[0])
        assert header["aurik_provenance_audit"] is True
        assert header["material"] == "shellac"
        assert header["source_file"] == "beethoven.wav"

    def test_save_jsonl_creates_file(self, mono_audio, sample_rate, tmp_path):
        audit = ProvenanceAudit(source_file="test.wav", material="vinyl")
        audit.record(
            step="save_test",
            audio_in=mono_audio,
            audio_out=mono_audio,
            sample_rate=sample_rate,
            rationale="Speichertest",
            confidence=1.0,
        )
        out_file = tmp_path / "test_audit.jsonl"
        saved = audit.save_jsonl(out_file)
        assert saved.exists()
        content = saved.read_text(encoding="utf-8")
        lines = content.strip().split("\n")
        assert len(lines) == 2

    def test_integrity_check_valid(self, mono_audio, sample_rate):
        audit = ProvenanceAudit()
        audit.record(
            step="integrity_test",
            audio_in=mono_audio,
            audio_out=mono_audio,
            sample_rate=sample_rate,
            rationale="Integritätstest",
            confidence=1.0,
        )
        result = audit.integrity_check()
        assert result["valid"] is True
        assert result["total_entries"] == 1
        assert len(result["failed_entries"]) == 0

    def test_entry_hash_deterministic(self, mono_audio, sample_rate):
        """Gleicher Eintrag → gleicher Hash."""
        audit = ProvenanceAudit()
        e1 = audit.record(
            step="hash_test",
            audio_in=mono_audio,
            audio_out=mono_audio,
            sample_rate=sample_rate,
            rationale="Hash-Test",
            confidence=0.8,
        )
        expected_hash = e1._compute_entry_hash()
        assert e1.entry_hash == expected_hash

    def test_audio_hash_deterministic(self, mono_audio, sample_rate):
        """Gleiche Audio → gleicher Hash."""
        h1 = _audio_hash(mono_audio, sample_rate)
        h2 = _audio_hash(mono_audio, sample_rate)
        assert h1 == h2

    def test_audio_hash_different_audio(self, mono_audio, stereo_audio, sample_rate):
        """Verschiedene Audio → verschiedene Hashes."""
        h1 = _audio_hash(mono_audio, sample_rate)
        h2 = _audio_hash(stereo_audio, sample_rate)
        assert h1 != h2

    def test_to_dict_structure(self):
        audit = ProvenanceAudit(source_file="test.wav", material="tape", mode="RESTORATION")
        d = audit.to_dict()
        assert "meta" in d
        assert "entries" in d
        assert d["meta"]["source_file"] == "test.wav"

    def test_multiple_entries(self, mono_audio, sample_rate):
        audit = ProvenanceAudit()
        for i in range(5):
            audit.record(
                step=f"step_{i}",
                audio_in=mono_audio,
                audio_out=mono_audio,
                sample_rate=sample_rate,
                rationale=f"Schritt {i}",
                confidence=0.9,
            )
        assert len(audit) == 5
        assert audit.integrity_check()["valid"]

    def test_text_summary(self, mono_audio, sample_rate):
        audit = ProvenanceAudit(source_file="wagner.wav", material="shellac")
        audit.record(
            step="defect_scan",
            audio_in=mono_audio,
            audio_out=mono_audio,
            sample_rate=sample_rate,
            rationale="Defektscan abgeschlossen",
            confidence=0.95,
        )
        text = audit.to_text_summary()
        assert "AURIK 9.0" in text
        assert "shellac" in text.lower()
        assert "defect_scan" in text


# ===========================================================================
# 5. ARE Integration — Neue Felder im Ergebnis
# ===========================================================================

# Alle zu prüfenden ARE-Result-Felder in einer einzigen parametrisierten Testfunktion
_ARE_RESULT_FIELDS = [
    "causal_order",
    "causal_explanation",
    "chain_corrections",
    "chain_spectral_change_db",
    "defect_quality_report",
    "provenance",
    "gaps_found",
    "gaps_repaired",
    "gap_total_repaired_ms",
]

# ARE-Subsysteme: (Attributname, Modulpfad, Klassenname)
_ARE_SUBSYSTEMS = [
    ("_causal_graph", "backend.core.causal_defect_graph", "CausalDefectGraph"),
    ("_chain_model", "backend.core.medium_chain_model", "PhysicalMediumChainModel"),
    ("_quality_reporter", "backend.core.defect_quality_report", "DefectQualityReporter"),
    ("_gap_reconstructor", "backend.core.gap_reconstructor", "GapReconstructor"),
]


class TestAREDifferentiatorIntegration:
    """Prüft ARE-Struktur: Differenzierer-Felder ohne echtes Processing (schnell).

    Das eigentliche ARE.process() ist zu langsam für regulären CI.
    Diese Tests prüfen Dataclass-Struktur + Subsystem-Initialisierung.
    """

    @pytest.mark.parametrize("field", _ARE_RESULT_FIELDS)
    def test_result_dataclass_has_field(self, field):
        """Alle 9 Differenzierer-Felder in AutonomousRestorationResult vorhanden."""
        import dataclasses

        from backend.core.autonomous_restoration_engine import AutonomousRestorationResult

        field_names = [f.name for f in dataclasses.fields(AutonomousRestorationResult)]
        assert field in field_names, f"Feld '{field}' fehlt in AutonomousRestorationResult"

    @pytest.mark.parametrize("attr,mod,cls", _ARE_SUBSYSTEMS)
    def test_are_has_subsystem(self, attr, mod, cls):
        """ARE initialisiert alle 4 Differenzierer-Subsysteme korrekt."""
        import importlib

        from backend.core.autonomous_restoration_engine import AutonomousRestorationEngine
        from backend.core.processing_modes import ProcessingMode

        engine = AutonomousRestorationEngine(mode=ProcessingMode.RESTORATION, enable_self_learning=False)
        expected_cls = getattr(importlib.import_module(mod), cls)
        assert hasattr(engine, attr), f"ARE fehlt Attribut '{attr}'"
        assert isinstance(getattr(engine, attr), expected_cls)

    def test_result_defaults(self):
        """Default-Werte der neuen Felder in AutonomousRestorationResult korrekt."""
        import dataclasses

        from backend.core.autonomous_restoration_engine import AutonomousRestorationResult

        # Prüfe Defaults über Field-Definitionen (ohne aufwändige Instanziierung)
        field_map = {f.name: f for f in dataclasses.fields(AutonomousRestorationResult)}
        # causal_order: default_factory=list → leere Liste
        assert field_map["causal_order"].default_factory is list  # type: ignore[misc]
        # causal_explanation: default=""
        assert field_map["causal_explanation"].default == ""
        # chain_corrections: default_factory=list
        assert field_map["chain_corrections"].default_factory is list  # type: ignore[misc]
        # chain_spectral_change_db: default=0.0
        assert field_map["chain_spectral_change_db"].default == 0.0
        # defect_quality_report: default=None
        assert field_map["defect_quality_report"].default is None
        # provenance: default=None
        assert field_map["provenance"].default is None
