"""D-K3-66: Final-Ausgabe-Gate — Delta-Schutz gilt für ALLE Referenzarten.

Befund (Läufe 10.12.17 + 10.12.12, je 1×): `§2.44/§2.49 Final-Ausgabe-Audio-
Gate: af=0.000 hpi=0.0000 — Rollback auf sicheren finalen Exportpuffer`.
Root-Cause: `_best_carrier_checkpoint` (Stand nach phase_49) existiert im
Chunked-Pfad → Referenz = Checkpoint, `_using_degraded_ref=False` → der
Delta-Schutz (`if _using_degraded_ref and _input_baseline_af is not None`)
wurde komplett übersprungen → AFG(finales Audio vs. 14-Phasen-älterer Stand)
= 0.000 → Gate-Fail → Rollback auf den Checkpoint (Wirkung der Phasen nach
dem letzten Carrier-Checkpoint verworfen).

Kontrakte (Source-Pins):
1. Pipeline-Baseline wird AUSSERHALB des `if _using_degraded_ref:`-Zweigs
   geladen (gleiche Ebene wie `try:`).
2. Gate-Check prüft nur noch `_input_baseline_af is not None` — NICHT mehr
   die degraded-Ref-Bindung.
3. §D-K3-66-Marker vorhanden (Nachvollziehbarkeit, §G8 (copilot-instructions.md)).
4. Regression-Log spricht von „Pipeline-Baseline".
5. False-Positive-Schutz (elif-Zweig) bleibt erhalten.
"""

from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "backend" / "core" / "unified_restorer_v3.py"


def _src() -> str:
    return SRC.read_text(encoding="utf-8")


def test_baseline_loaded_for_all_reference_kinds() -> None:
    """Kontrakt 1: Baseline-Ladung nicht mehr im degraded-Ref-Zweig."""
    src = _src()
    marker = "            _input_baseline_af = None\n            try:\n                if _using_degraded_ref:\n"
    assert marker in src, "Baseline-Messung muss im try, der degraded-Zweig nur für die Diagnose-Messung (D-K3-66)"
    baseline_line = (
        '                _input_baseline_af = float(getattr(self, "_artifact_freedom_score", 0.0) or 0.0)\n'
        "            except Exception as _ibaf_exc:"
    )
    assert baseline_line in src, "Pipeline-Baseline muss auf try-Ebene geladen werden (D-K3-66)"


def test_gate_check_without_degraded_ref_binding() -> None:
    """Kontrakt 2: Delta-Schutz gilt ohne degraded-Ref-Bindung."""
    src = _src()
    assert "if _input_baseline_af is not None:" in src
    assert "if _using_degraded_ref and _input_baseline_af is not None:" not in src, (
        "Die degraded-Ref-Bindung des Delta-Schutzes ist die D-K3-66-Ursache"
    )


def test_dk366_marker_present() -> None:
    """Kontrakt 3: Nachvollziehbarkeit (§G8 (copilot-instructions.md))."""
    src = _src()
    assert src.count("§D-K3-66") >= 2, "Fix-Kommentare fehlen"


def test_regression_log_uses_pipeline_baseline_wording() -> None:
    """Kontrakt 4: Log-Text entspricht der erweiterten Semantik."""
    src = _src()
    assert "Regression gegenüber Pipeline-Baseline" in src


def test_false_positive_guard_kept() -> None:
    """Kontrakt 5: FP-Schutz (Input gut, Output extrem schlecht) unverändert."""
    src = _src()
    assert "false-positive gegen degraded Eingabe, behalte Baseline" in src
    assert "_final_export_af = _input_baseline_af" in src
