from __future__ import annotations

import pytest

"""[RELEASE_MUST] Automatischer Spec-Konsistenz-Validator (v10.0.0)

CI-Gate das DAUERHAFT und AUTOMATISCH folgende Invarianten prüft:

  1. V12 Bidirektionale Sync (Code):
     Jeder CAUSES-Eintrag MUSS in CAUSE_TO_PHASES vorhanden sein und umgekehrt.
     Orphaned Keys → V12-Fehler → CI-Fail.

  2. Spec-06 vs. Code-Sync:
     Jeder CAUSES-Key im Code MUSS auch in Spec 06 CAUSE_TO_PHASES stehen.
     Jeder Spec-06-CAUSE_TO_PHASES-Key MUSS in CAUSES existieren.
     Divergenz → CI-Fail (verhindert Spec-Code-Drift).

  3. Phase-Datei-Existenz:
     Jede phase_xx_name in CAUSE_TO_PHASES (Code + Spec 06) MUSS eine .py-Datei
     in backend/core/phases/ haben. Missing file → CI-Fail.

  4. §0a Spec-06-Reinheit (ergänzend zu test_section_0a_restoration_guard.py):
     Keine §0a-verbotene Phase darf in Spec 06 CAUSE_TO_PHASES stehen.

  5. VERBOTEN.md Linter-Code-Präsenz in copilot-instructions.md:
      Alle Linter-Codes V01–V33 aus VERBOTEN.md müssen in copilot-instructions.md
     erwähnt sein. Stellt sicher, dass die KI-Agenten-Richtlinie vollständig ist.

Laufzeit: < 5 s (kein Audio, kein ML — pure Text/AST-Analyse).
Aufruf:
    pytest tests/normative/test_spec_consistency.py -v

Spec-Referenzen:
    §2.59 CAUSE_TO_PHASES/CAUSES Bidirektional-Sync
    §0a Crossfire-Modus-Invariante
    V12 VERBOTEN.md Linter-Code
    .github/VERBOTEN.md (normative Quelle für V01–V33)
    .github/copilot-instructions.md (KI-Agenten-Richtlinie)
    .github/specs/06_phases_system.md (kanonische Phase-/Ursachen-Tabelle)
"""


import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Pfade
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).parents[2]
_SPEC_06_PATH = _PROJECT_ROOT / ".github/specs/06_phases_system.md"
_VERBOTEN_MD_PATH = _PROJECT_ROOT / ".github/VERBOTEN.md"
_COPILOT_INSTRUCTIONS_PATH = _PROJECT_ROOT / ".github/copilot-instructions.md"
_PIPELINE_INSTRUCTIONS_PATH = _PROJECT_ROOT / ".github/instructions/pipeline.instructions.md"
_PHASES_INSTRUCTIONS_PATH = _PROJECT_ROOT / ".github/instructions/phases.instructions.md"
_UV3_PATH = _PROJECT_ROOT / "backend/core/unified_restorer_v3.py"
_PHASES_DIR = _PROJECT_ROOT / "backend/core/phases"
_CDR_PATH = _PROJECT_ROOT / "backend/core/causal_defect_reasoner.py"

# §0a — absolut verbotene Phasen in Restoration CAUSE_TO_PHASES
_SECTION_0A_FORBIDDEN: frozenset[str] = frozenset(
    {
        "phase_21_exciter",
        "phase_35_multiband_compression",
        "phase_42_vocal_enhancement",
    }
)

# ---------------------------------------------------------------------------
# Hilfsfunktionen — Spec 06 parsen
# ---------------------------------------------------------------------------


def _get_spec06_cause_to_phases_block(spec_path: Path) -> str:
    """Gibt den CAUSE_TO_PHASES-Block aus Spec 06 als String zurück."""
    content = spec_path.read_text(encoding="utf-8")
    start = content.find("CAUSE_TO_PHASES = {")
    assert start != -1, f"CAUSE_TO_PHASES-Block nicht gefunden in {spec_path}"
    # Find closing ``` of the Python code fence (the next ``` after the block start)
    end = content.find("\n```", start)
    if end == -1:
        # Fallback: read up to 15000 chars
        end = start + 15000
    return content[start:end]


def _parse_spec06_cause_keys(spec_path: Path) -> set[str]:
    """Extrahiert alle String-Keys aus dem CAUSE_TO_PHASES-Block in Spec 06.

    Regex-basiert (kein eval) — robust gegenüber Formatierungsfehlern.
    """
    block = _get_spec06_cause_to_phases_block(spec_path)
    # Pattern: "cause_name":
    # Exclude "CAUSE_TO_PHASES" itself (not a key)
    raw = re.findall(r'"([a-z][a-z_0-9]+)"\s*:', block)
    return {k for k in raw if k != "cause_to_phases"}


def _parse_spec06_cause_to_phases_values(spec_path: Path) -> dict[str, list[str]]:
    """Extrahiert cause→phases Mapping aus dem CAUSE_TO_PHASES-Block in Spec 06.

    Verarbeitet mehrzeilige Listeneinträge. Ignoriert Kommentar-Zeilen.
    """
    block = _get_spec06_cause_to_phases_block(spec_path)
    result: dict[str, list[str]] = {}
    # Match "cause_key": ["phase1", "phase2", ...] — potentially multiline
    # Using a non-greedy match up to the closing ]
    for match in re.finditer(r'"([a-z][a-z_0-9]+)"\s*:\s*\[([^\]]*)\]', block, re.DOTALL):
        cause = match.group(1)
        phases_raw = match.group(2)
        phases = re.findall(r'"(phase_[a-z0-9_]+)"', phases_raw)
        result[cause] = phases
    return result


# ---------------------------------------------------------------------------
# Hilfsfunktion — Code importieren
# ---------------------------------------------------------------------------


def _get_code_causes_and_c2p() -> tuple[list[str], dict[str, list[str]]]:
    """Importiert CAUSES und CAUSE_TO_PHASES aus causal_defect_reasoner.py."""
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    # Import fresh — don't rely on already-imported cached module in test session
    from backend.core.causal_defect_reasoner import CAUSE_TO_PHASES, CAUSES

    return list(CAUSES), dict(CAUSE_TO_PHASES)


# ---------------------------------------------------------------------------
# Infrastruktur-Checks
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSpecInfrastructure:
    """Prüft, dass alle notwendigen Dateien existieren."""

    def test_spec06_exists(self) -> None:
        assert _SPEC_06_PATH.exists(), f"Spec 06 nicht gefunden: {_SPEC_06_PATH}"

    def test_verboten_md_exists(self) -> None:
        assert _VERBOTEN_MD_PATH.exists(), f"VERBOTEN.md nicht gefunden: {_VERBOTEN_MD_PATH}"

    def test_copilot_instructions_exists(self) -> None:
        assert _COPILOT_INSTRUCTIONS_PATH.exists(), (
            f"copilot-instructions.md nicht gefunden: {_COPILOT_INSTRUCTIONS_PATH}"
        )

    def test_causal_defect_reasoner_exists(self) -> None:
        assert _CDR_PATH.exists(), f"causal_defect_reasoner.py nicht gefunden: {_CDR_PATH}"

    def test_phases_dir_exists(self) -> None:
        assert _PHASES_DIR.exists(), f"Phases-Verzeichnis nicht gefunden: {_PHASES_DIR}"

    def test_spec06_contains_cause_to_phases_block(self) -> None:
        content = _SPEC_06_PATH.read_text(encoding="utf-8")
        assert "CAUSE_TO_PHASES = {" in content, (
            "Spec 06 enthält kein 'CAUSE_TO_PHASES = {'-Block — Spec-Format geändert?"
        )


# ---------------------------------------------------------------------------
# V12 Bidirektionale Sync — Code
# ---------------------------------------------------------------------------


class TestV12BidirectionalSyncCode:
    """§2.59/V12: CAUSES ↔ CAUSE_TO_PHASES bidirektional konsistent (Code)."""

    def test_every_cause_has_c2p_entry(self) -> None:
        """Jeder CAUSES-Eintrag muss ein CAUSE_TO_PHASES-Gegenstück haben."""
        causes, c2p = _get_code_causes_and_c2p()
        missing = [c for c in causes if c not in c2p]
        assert not missing, (
            "V12: CAUSES-Einträge ohne CAUSE_TO_PHASES-Gegenstück in Code:\n"
            + "\n".join(f"  - {c}" for c in sorted(missing))
            + "\nJede neue Ursache MUSS in beiden Dicts stehen (§2.59)."
        )

    def test_every_c2p_key_has_causes_entry(self) -> None:
        """Jeder CAUSE_TO_PHASES-Key muss in CAUSES stehen (kein Orphan)."""
        causes, c2p = _get_code_causes_and_c2p()
        causes_set = set(causes)
        orphaned = [k for k in c2p if k not in causes_set]
        assert not orphaned, (
            "V12: Orphaned CAUSE_TO_PHASES-Keys in Code (kein CAUSES-Gegenstück):\n"
            + "\n".join(f"  - {k}" for k in sorted(orphaned))
            + "\nOrphaned Keys können nicht durch CausalDefectReasoner gefunden werden (§2.59)."
        )


# ---------------------------------------------------------------------------
# Spec 06 vs. Code Sync
# ---------------------------------------------------------------------------


class TestSpec06VsCodeSync:
    """Spec 06 CAUSE_TO_PHASES muss mit Code CAUSES synchron sein."""

    def test_all_code_causes_in_spec06(self) -> None:
        """Alle Code-CAUSES müssen als Key in Spec 06 CAUSE_TO_PHASES stehen."""
        causes, _ = _get_code_causes_and_c2p()
        spec_keys = _parse_spec06_cause_keys(_SPEC_06_PATH)
        missing_in_spec = [c for c in causes if c not in spec_keys]
        assert not missing_in_spec, (
            "Spec 06 CAUSE_TO_PHASES fehlt Einträge die in Code CAUSES vorhanden sind:\n"
            + "\n".join(f"  - {c}" for c in sorted(missing_in_spec))
            + "\nSpec 06 muss mit causal_defect_reasoner.py synchron sein."
        )

    def test_all_spec06_keys_in_code_causes(self) -> None:
        """Alle Spec 06 CAUSE_TO_PHASES-Keys müssen in Code CAUSES vorhanden sein."""
        causes, _ = _get_code_causes_and_c2p()
        causes_set = set(causes)
        spec_keys = _parse_spec06_cause_keys(_SPEC_06_PATH)
        orphaned_in_spec = [k for k in spec_keys if k not in causes_set]
        assert not orphaned_in_spec, (
            "Spec 06 CAUSE_TO_PHASES hat Einträge die NICHT in Code CAUSES stehen (V12/Spec-Drift):\n"
            + "\n".join(f"  - {k}" for k in sorted(orphaned_in_spec))
            + "\nOrphaned Spec-Keys → CausalDefectReasoner ignoriert sie → Spec beschreibt "
            "nicht existierende Kausallogik."
        )


# ---------------------------------------------------------------------------
# §0a — Spec 06 Reinheit
# ---------------------------------------------------------------------------


class TestSection0aSpec06Purity:
    """§0a: Keine verbotenen Restoration-Phasen in Spec 06 CAUSE_TO_PHASES."""

    def test_no_forbidden_phase_in_spec06_cause_to_phases(self) -> None:
        """Kein §0a-verbotener Phase-ID darf in Spec 06 CAUSE_TO_PHASES stehen."""
        cause_phases = _parse_spec06_cause_to_phases_values(_SPEC_06_PATH)
        violations: list[str] = []
        for cause, phases in cause_phases.items():
            for phase in phases:
                if phase in _SECTION_0A_FORBIDDEN:
                    violations.append(f"  {cause!r} → {phase!r}")
        assert not violations, (
            "§0a Verletzung in Spec 06 CAUSE_TO_PHASES — verbotene Phasen gefunden:\n"
            + "\n".join(violations)
            + "\n\n§0a-verbotene Phasen (Stem-Enhancement/Harmonic Exciter) dürfen "
            "NIEMALS in CAUSE_TO_PHASES stehen (BUG-FIX v10.0.0 §0a)."
        )


# ---------------------------------------------------------------------------
# Phase-Datei-Existenz
# ---------------------------------------------------------------------------


class TestPhaseFileExistence:
    """Alle in CAUSE_TO_PHASES referenzierten Phasen müssen als .py-Dateien existieren."""

    def test_code_cause_to_phases_all_files_exist(self) -> None:
        """Alle Phase-IDs in Code CAUSE_TO_PHASES haben .py-Dateien."""
        _, c2p = _get_code_causes_and_c2p()
        missing: list[str] = []
        for cause, phases in c2p.items():
            for phase_id in phases:
                if not (_PHASES_DIR / f"{phase_id}.py").exists():
                    missing.append(f"  {cause} → {phase_id}.py")
        assert not missing, (
            "Phase-Dateien fehlen für Code-CAUSE_TO_PHASES-Einträge:\n"
            + "\n".join(missing[:30])
            + (f"\n  ... und {len(missing) - 30} weitere" if len(missing) > 30 else "")
            + "\nJede referenzierte Phase muss eine Implementierungsdatei haben."
        )

    def test_spec06_cause_to_phases_all_files_exist(self) -> None:
        """Alle Phase-IDs in Spec 06 CAUSE_TO_PHASES haben .py-Dateien."""
        cause_phases = _parse_spec06_cause_to_phases_values(_SPEC_06_PATH)
        missing: list[str] = []
        for cause, phases in cause_phases.items():
            for phase_id in phases:
                if not (_PHASES_DIR / f"{phase_id}.py").exists():
                    missing.append(f"  {cause} → {phase_id}.py")
        assert not missing, (
            "Phase-Dateien fehlen für Spec-06-CAUSE_TO_PHASES-Einträge:\n"
            + "\n".join(missing[:30])
            + (f"\n  ... und {len(missing) - 30} weitere" if len(missing) > 30 else "")
        )


# ---------------------------------------------------------------------------
# VERBOTEN.md Linter-Codes in copilot-instructions.md
# ---------------------------------------------------------------------------


class TestVerbotenLinterCodesInCopilotInstructions:
    """Alle VERBOTEN.md Linter-Codes V01–V33 müssen in copilot-instructions.md stehen."""

    def test_linter_codes_present_in_copilot_instructions(self) -> None:
        """Alle in VERBOTEN.md definierten Linter-Codes müssen in copilot-instructions.md stehen.

        Die Codes werden aus der Linter-Referenz-Tabelle in VERBOTEN.md geparst —
        nur definierte Codes werden geprüft (nicht alle V01–V33 sequenziell).
        """
        verboten_content = _VERBOTEN_MD_PATH.read_text(encoding="utf-8")
        # Parse Linter-Referenz table: lines matching "| Vxx |"
        defined_codes = re.findall(r"\|\s*(V\d{2})\s*\|", verboten_content)
        defined_codes_set = sorted(set(defined_codes))

        copilot_content = _COPILOT_INSTRUCTIONS_PATH.read_text(encoding="utf-8")
        missing = [code for code in defined_codes_set if code not in copilot_content]
        assert not missing, (
            f"Linter-Codes aus VERBOTEN.md fehlen in copilot-instructions.md: {missing}\n"
            f"In VERBOTEN.md definierte Codes: {defined_codes_set}\n"
            "Alle in VERBOTEN.md definierten Linter-Codes müssen als normative Referenz "
            "in den KI-Agenten-Richtlinien stehen — damit jede neue Session die Codes kennt."
        )

    def test_section_0a_forbidden_phases_mentioned_in_copilot_instructions(self) -> None:
        """§0a-verbotene Phasen müssen in copilot-instructions.md VERBOTEN-Tabelle stehen."""
        content = _COPILOT_INSTRUCTIONS_PATH.read_text(encoding="utf-8")
        missing = [p for p in sorted(_SECTION_0A_FORBIDDEN) if p not in content]
        assert not missing, (
            f"§0a-verbotene Phasen fehlen in copilot-instructions.md: {missing}\n"
            "KI-Agenten müssen diese Phasen als normativ verboten kennen."
        )

    def test_verboten_md_linter_reference_section_exists(self) -> None:
        """VERBOTEN.md muss einen Linter-Referenz-Abschnitt haben."""
        content = _VERBOTEN_MD_PATH.read_text(encoding="utf-8")
        assert "V01" in content and "V33" in content, "VERBOTEN.md fehlt Linter-Referenz-Abschnitt mit V01–V33."


# ---------------------------------------------------------------------------
# Spec-Integrität: Keine azimuth_error-Regression
# ---------------------------------------------------------------------------


class TestSpecOrphanRegression:
    """Regressions-Schutz für bekannte frühere Spec-Bugs."""

    def test_azimuth_error_not_in_spec06_cause_to_phases(self) -> None:
        """azimuth_error darf NICHT in Spec 06 CAUSE_TO_PHASES stehen (BUG-FIX v10.0.0).

        azimuth_error ist ein DefectScanner-Messwert, keine CAUSES-Ursache.
        Früher hatte Spec 06 einen orphaned azimuth_error-CAUSE_TO_PHASES-Eintrag (V12-Bug).
        """
        spec_keys = _parse_spec06_cause_keys(_SPEC_06_PATH)
        assert "azimuth_error" not in spec_keys, (
            "Regression: 'azimuth_error' ist wieder in Spec 06 CAUSE_TO_PHASES aufgetaucht.\n"
            "BUG-FIX v10.0.0: azimuth_error ist kein CAUSES-Eintrag im Code — "
            "der Spec-Key muss entfernt bleiben (V12-Invariante)."
        )

    def test_vinyl_warp_in_spec06_cause_to_phases(self) -> None:
        """vinyl_warp MUSS in Spec 06 CAUSE_TO_PHASES stehen (BUG-FIX v10.0.0).

        vinyl_warp war früher im Code-CAUSES aber fehlte in Spec 06 — Spec-Code-Drift.
        """
        spec_keys = _parse_spec06_cause_keys(_SPEC_06_PATH)
        assert "vinyl_warp" in spec_keys, (
            "Regression: 'vinyl_warp' fehlt in Spec 06 CAUSE_TO_PHASES.\n"
            "BUG-FIX v10.0.0: vinyl_warp ist in Code-CAUSES definiert — "
            "Spec muss synchron sein."
        )

    def test_no_phase_39_in_bandwidth_loss_cause(self) -> None:
        """phase_39 darf NICHT in bandwidth_loss CAUSE_TO_PHASES stehen (BUG-FIX v10.0.0 §6.2c).

        BW-Extension über Material-BW-Ceiling erzeugt Halluzinationen in Restoration.
        """
        _, c2p = _get_code_causes_and_c2p()
        bw_phases = c2p.get("bandwidth_loss", [])
        assert "phase_39_air_band_enhancement" not in bw_phases, (
            "Regression: 'phase_39_air_band_enhancement' ist wieder in bandwidth_loss "
            "CAUSE_TO_PHASES.\nBUG-FIX v10.0.0 §6.2c: phase_39 erzeugt Halluzinationen "
            "über BW-Ceiling analoger Materialien im Restoration-Modus."
        )


# ---------------------------------------------------------------------------
# Gate-Architektur Drift-Guard (A/B/C + Recovery-State-Machine)
# ---------------------------------------------------------------------------


class TestGateArchitectureSpecSync:
    """Sichert Konsistenz zwischen Instruktions-Text und UV3-Implementierung."""

    def test_instruction_files_exist_for_gate_architecture(self) -> None:
        assert _PIPELINE_INSTRUCTIONS_PATH.exists(), (
            f"pipeline.instructions.md nicht gefunden: {_PIPELINE_INSTRUCTIONS_PATH}"
        )
        assert _PHASES_INSTRUCTIONS_PATH.exists(), f"phases.instructions.md nicht gefunden: {_PHASES_INSTRUCTIONS_PATH}"
        assert _UV3_PATH.exists(), f"unified_restorer_v3.py nicht gefunden: {_UV3_PATH}"

    def test_pipeline_instructions_define_gate_classes_and_state_machine(self) -> None:
        content = _PIPELINE_INSTRUCTIONS_PATH.read_text(encoding="utf-8")
        required_markers = [
            "§2.49a Zentrale Gate-Klassifikation (A/B/C)",
            "§2.49b Recovery-Zustandsautomat",
            "§2.49c Semantik-Konfliktregel Hard vs Recovery",
            "quality_gate_registry",
            "recovery_state_machine",
            "semantic_conflicts",
        ]
        missing = [m for m in required_markers if m not in content]
        assert not missing, (
            "Pipeline-Instruktionen fehlen Gate-Architektur-Marker:\n"
            + "\n".join(f"  - {m}" for m in missing)
            + "\nSpezifikation muss A/B/C-Klassen + Recovery-State-Machine vollständig dokumentieren."
        )

    def test_phases_instructions_enforce_uv3_final_authority(self) -> None:
        content = _PHASES_INSTRUCTIONS_PATH.read_text(encoding="utf-8")
        required_markers = [
            'metadata["phase_decision_scope"] = "local_only"',
            'metadata["final_authority"] = "UV3"',
            "quality_gate_registry",
            "recovery_state_machine",
        ]
        missing = [m for m in required_markers if m not in content]
        assert not missing, (
            "Phasen-Instruktionen fehlen UV3-Authority-Marker:\n"
            + "\n".join(f"  - {m}" for m in missing)
            + "\nPhasen dürfen nur lokal entscheiden; finale Gate-Authority liegt bei UV3."
        )

    def test_uv3_implements_canonical_gate_methods_and_metadata(self) -> None:
        content = _UV3_PATH.read_text(encoding="utf-8")
        required_markers = [
            "def _classify_quality_gate_events(",
            "def _resolve_recovery_state_machine(",
            '"quality_gate_registry"',
            '"recovery_state_machine"',
            '"semantic_conflicts"',
        ]
        missing = [m for m in required_markers if m not in content]
        assert not missing, (
            "UV3-Code fehlt kanonische Gate-/Recovery-Marker:\n"
            + "\n".join(f"  - {m}" for m in missing)
            + "\nImplementierung muss den Spezifikationsvertrag vollständig abbilden."
        )
