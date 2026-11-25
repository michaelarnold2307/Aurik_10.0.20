"""Vertragstests für die research-only Phasen-Auslassung (D-K3-110 Modus B)."""
from __future__ import annotations

from backend.core.unified_restorer_v3 import _AUSLASSUNG_ENV, phase_auslassungen_anwenden

_PLAN = [
    "phase_02_hum_removal",
    "phase_12_wow_flutter_fix",
    "phase_30_dc_offset_removal",
    "phase_glue_stage",
]


def test_leer_oder_unset_laesst_plan_unveraendert() -> None:
    for roh in ("", "   ", " , , "):
        assert phase_auslassungen_anwenden(_PLAN, roh) == _PLAN


def test_entfernt_genannte_phase_und_haelt_reihenfolge() -> None:
    out = phase_auslassungen_anwenden(_PLAN, "phase_12_wow_flutter_fix")
    assert out == [
        "phase_02_hum_removal",
        "phase_30_dc_offset_removal",
        "phase_glue_stage",
    ]


def test_mehrere_phasen_kommagetrennt() -> None:
    out = phase_auslassungen_anwenden(_PLAN, "phase_02_hum_removal, phase_glue_stage")
    assert out == ["phase_12_wow_flutter_fix", "phase_30_dc_offset_removal"]


def test_unbekannte_ids_werden_ignoriert() -> None:
    out = phase_auslassungen_anwenden(_PLAN, "phase_02_hum_removal, phase_999_gibts_nicht")
    assert out == ["phase_12_wow_flutter_fix", "phase_30_dc_offset_removal", "phase_glue_stage"]


def test_determinismus_gleicher_input_gleiche_auswahl() -> None:
    roh = "phase_02_hum_removal,phase_30_dc_offset_removal"
    assert phase_auslassungen_anwenden(_PLAN, roh) == phase_auslassungen_anwenden(_PLAN, roh)


def test_umgebungsvariable_ist_dokumentiert() -> None:
    assert _AUSLASSUNG_ENV == "AURIK_PHASE_AUSLASSEN"
