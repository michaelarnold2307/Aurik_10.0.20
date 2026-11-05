"""Vertragstests der K0/K1-Restlücken-Buchung 2026-11-05 (D-K0-2, D-K0-8, D-K1-5, D-K1-3).

Pinnt die Register-Entscheidungen in `.github/SOTA_DEFICIT_REGISTER.md` samt Beleg-Kette
(§III.13 (copilot-instructions.md): Evidenz am realen Artefakt, keine erfundenen Belege):

- D-K0-2: K0-C Hör-Entscheidung 2026-10-30 (NO-GO) ist gebucht — Status „geschlossen"
  nur mit existierendem Beleg-Pfad (Register-Kriterium WP-0).
- D-K0-8: echter Daten-Blocker mit benanntem Beschaffungs-/Trainingspfad — kein
  Schein-Abschluss (§V7 (copilot-instructions.md): kein Rollout ohne A/B).
- D-K1-5: der zitierte A1-Gate-Beleg trägt `gate_passed: true`; die verbleibende
  Schranke ist menschlich (Hör-Sign-off), nicht technisch.
- D-K1-3: Verdrahtung hinter `use_material_crnn` dokumentiert (Commit 28f2947d),
  Aktivierung menschlich-gated über §4-Verfahren (`scripts/flag_freigabe.py`).

Zusätzlich: die Flag-Defaults `use_material_crnn`/`use_bw_v5` bleiben `False`
(§v10.802 (copilot-instructions.md)) — KEIN Flag-Flip in dieser Welle.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.sota_deficit_gate import REGISTER, audit, parse_register

_ROOT = Path(__file__).resolve().parents[2]
_K0_C_REPORT = _ROOT / "docs" / "reports" / "current" / "2026-10-30_k0_signoff_paket.md"
_KP7_GATE_JSON = _ROOT / "docs" / "reports" / "current" / "2026-10-27_kp7_hf_gate_v2.json"
_WOHLKLANG = _ROOT / ".github" / "WOHLKLANG_CLAIMS.md"


def _entries() -> dict:
    """Register-Einträge nach ID — über den realen Gate-Parser (kein Eigen-Parsing)."""
    entries, _declared = parse_register(REGISTER.read_text(encoding="utf-8"))
    return {entry.id: entry for entry in entries}


def test_register_geht_durch_das_fail_closed_gate() -> None:
    """Die gebuchte Fassung bleibt gate-konform (fail-closed, WP-0 der Register-Pflege)."""
    result = audit()
    assert result["violations"] == [], result["violations"]


def test_dk0_2_ist_ueber_k0_c_entschieden_und_mit_beleg_geschlossen() -> None:
    """D-K0-2: Status „geschlossen" nur mit existierendem Beleg-Pfad zur Hör-Entscheidung."""
    entry = _entries()["D-K0-2"]
    assert entry.status == "geschlossen"
    assert "K0-C" in entry.blocker and "NO-GO" in entry.blocker
    assert "2026-10-30_k0_signoff_paket.md" in entry.beleg
    assert _K0_C_REPORT.is_file()


def test_dk0_2_beleg_traegt_das_hoer_urteil() -> None:
    """Der zitierte D-K0-2-Beleg trägt das echte Hör-Urteil (keine Leerformel)."""
    text = _K0_C_REPORT.read_text(encoding="utf-8")
    assert "NO-GO" in text
    assert "Basis-BigVGAN bleibt aktiv" in text


def test_dk0_8_bleibt_datenblocker_mit_beschaffungs_und_trainingspfad() -> None:
    """D-K0-8: kein Schein-Abschluss — Daten-Blocker benannt, Pfade zeigen auf existierende Werkzeuge."""
    entry = _entries()["D-K0-8"]
    assert entry.status == "in-arbeit"
    assert "DATEN" in entry.blocker
    assert "generate_corpus_from_public_domain.py" in entry.blocker
    assert "train_medium_classifier.py" in entry.blocker
    assert (_ROOT / "scripts" / "generate_corpus_from_public_domain.py").is_file()
    assert (_ROOT / "scripts" / "train_medium_classifier.py").is_file()


def test_dk1_5_a1_gatebeleg_im_json_ist_gepint() -> None:
    """D-K1-5: der zitierte Gate-Beleg trägt gate_passed=true und die Kalibrierung (0,73-Reproduktion)."""
    entry = _entries()["D-K1-5"]
    assert entry.status == "in-arbeit"
    assert "2026-10-27_kp7_hf_gate_v2.json" in entry.beleg
    data = json.loads(_KP7_GATE_JSON.read_text(encoding="utf-8"))
    assert data["summary"]["gate_passed"] is True
    assert data["summary"]["gate_threshold_linear"] == 1.02
    assert data["summary"]["kalibrierung"]["reproduktion"] is True


def test_dk1_5_aktivierung_bleibt_hoer_signoff_und_flag_gesperrt() -> None:
    """D-K1-5: die verbleibende Schranke ist menschlich (Hör-Sign-off) + §4-Verfahren, nicht technisch."""
    entry = _entries()["D-K1-5"]
    assert "Hör-Sign-off" in entry.blocker
    assert "use_bw_v5" in entry.blocker
    assert "flag_freigabe.py" in entry.blocker
    assert "use_bw_v5" in entry.blocker and "bleibt `False`" in entry.blocker


def test_dk1_3_status_dokumentiert_verdrahtung_und_kein_flag_flip() -> None:
    """D-K1-3: verdrahtet hinter use_material_crnn, Aktivierung menschlich-gated, KEIN Flag-Flip."""
    entry = _entries()["D-K1-3"]
    assert entry.status == "in-arbeit"
    assert "28f2947d" in entry.blocker
    assert "use_material_crnn" in entry.blocker
    assert "flag_freigabe.py" in entry.blocker
    assert "KEIN Flag-Flip" in entry.blocker


def test_kein_flag_flip_use_material_crnn() -> None:
    """§v10.802 (copilot-instructions.md): Default AUS bleibt unangetastet (bit-identisch)."""
    from backend.core import music_model_flags as flags

    assert flags.use_material_crnn is False


def test_kein_flag_flip_use_bw_v5() -> None:
    """§v10.802 (copilot-instructions.md): use_bw_v5 bleibt gesperrt bis zum Hör-Sign-off."""
    from backend.core import music_model_flags as flags

    assert flags.use_bw_v5 is False


def test_wohlklang_w8_sperrt_mit_aktueller_begruendung() -> None:
    """W-8 (.github/WOHLKLANG_CLAIMS.md) bleibt gesperrt — mit dem aktuellen Sperrgrund (§G9 (copilot-instructions.md))."""
    line = next(ln for ln in _WOHLKLANG.read_text(encoding="utf-8").splitlines() if ln.startswith("| W-8 |"))
    assert "gesperrt" in line
    assert "Hör-Sign-off offen" in line
    assert "A1-Gate nicht bestanden" not in line
