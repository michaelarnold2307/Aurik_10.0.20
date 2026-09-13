"""§GGB-1: Tests für das Global Gain Budget — inkl. Material-KOMBINATIONEN.

§Mixtape-Realität (2026-09-13): Importsongs sind selten REINES Material —
Vinyl-Rips mit MP3-Vorlauf, Kassetten-Kopien mit digitalen Artefakten.
Kombinierte Material-Strings (\"vinyl+digital\", \"cassette/mp3\") zählen als
Tape-Familie, sobald EINE Komponente Tape ist — die empfindlichste
Komponente bestimmt die Budget-Regel.
"""

from __future__ import annotations

from backend.core.global_gain_budget import GlobalGainBudget, _is_tape_family
from backend.core.klang_guards import GuardWisdom


def test_tape_family_pure_and_combined() -> None:
    assert _is_tape_family("cassette") is True
    assert _is_tape_family("reel_tape") is True
    assert _is_tape_family("tape") is True
    assert _is_tape_family("vinyl") is False
    assert _is_tape_family("digital") is False
    assert _is_tape_family("unknown") is False
    assert _is_tape_family("") is False


def test_tape_family_combined_strings() -> None:
    assert _is_tape_family("vinyl+cassette") is True
    assert _is_tape_family("cassette/mp3") is True
    assert _is_tape_family("digital, reel_tape") is True
    assert _is_tape_family("vinyl+digital") is False
    assert _is_tape_family("CASSETTE + digital") is True


def test_configure_combined_material_applies_tape_factor() -> None:
    """SNR-Adaptive Skalierung: 'vinyl+cassette' bekommt den Tape-Aufschlag —
    'vinyl+digital' nicht (empfindlichste Komponente entscheidet)."""
    ggb_cassette = GlobalGainBudget()
    ggb_cassette.configure_for_chain_depth(2, snr_db=14.0, material="vinyl+cassette")
    ggb_digital = GlobalGainBudget()
    ggb_digital.configure_for_chain_depth(2, snr_db=14.0, material="vinyl+digital")
    assert ggb_cassette._total_budget_db > ggb_digital._total_budget_db


def test_configure_snr_combined_material() -> None:
    ggb = GlobalGainBudget()
    ggb.configure_for_chain_depth(1, snr_db=30.0, material="digital")
    base = ggb._total_budget_db
    ggb.configure_snr(14.0, material="reel_tape+digital")
    assert ggb._total_budget_db > base


def test_unknown_material_unchanged_behavior() -> None:
    """unknown bleibt der konservative Default (kein Tape-Aufschlag)."""
    ggb = GlobalGainBudget()
    ggb.configure_for_chain_depth(4, snr_db=30.0, material="unknown")
    assert ggb._total_budget_db == 12.0


def test_guard_wisdom_combined_material_thresholds() -> None:
    """GuardWisdom.adaptive_threshold: Kombinationen nutzen den Faktor der
    EMPFINDLICHSTEN Komponente (max) — ein Vinyl-Rip mit MP3-Artefakten
    bleibt primär Vinyl."""
    assert GuardWisdom(material="vinyl+digital").adaptive_threshold("x", 1.0) == 1.1
    assert GuardWisdom(material="wax_cylinder+cd_digital").adaptive_threshold("x", 1.0) == 1.5
    assert GuardWisdom(material="cassette/mp3").adaptive_threshold("x", 1.0) == 1.15
    assert GuardWisdom(material="unknown").adaptive_threshold("x", 1.0) == 1.0
    assert GuardWisdom(material="shellac").adaptive_threshold("x", 1.0) == 1.3
