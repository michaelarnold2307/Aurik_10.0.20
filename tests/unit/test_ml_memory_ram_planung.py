"""Vertragstests der last-unabhängigen Phasen-RAM-Planung (§2.24-Determinismus).

Befund 2026-11-25 (Drift-Bisektion): die Vor-Phasen-Notbremse las den
LIVE-Speicher (available < 4.0 GB) — die Phasen-Teilnahme wurde
lastabhängig, identische Läufe drifteten (~0.17 Δ-MOS). Fix:
`ram_deferral_projektion` entscheidet über die EIGENE Prozess-Projektion
(own RSS + statische Phasen-Schätzung) gegen ein maschinenfestes Budget.
Diese Tests pinnen die reine Funktion (kein psutil, kein time.time()).
"""
from __future__ import annotations

from backend.core.ml_memory_budget import (
    OWN_RAM_BUDGET_MIN_GB,
    OWN_RAM_BUDGET_TOTAL_FACTOR,
    PHASE_RAM_ESTIMATE_GB,
    ram_deferral_projektion,
)


def test_reine_dsp_phase_traegt_keine_schaetzung() -> None:
    assert ram_deferral_projektion("phase_11_limiting", 0.0, 32.0) == (False, 0.0)


def test_unbekannte_phase_traegt_keine_schaetzung() -> None:
    assert ram_deferral_projektion("phase_xyz_unknown", 9.0, 32.0) == (False, 9.0)


def test_32gb_ruhige_maschine_deferriert_nicht() -> None:
    # Eigener Fußabdruck ~10.5 GB + phase_09 (1.5) = 12.0 < Budget 16.0
    soll, proj = ram_deferral_projektion("phase_09_crackle_removal", 10.5, 32.0)
    assert soll is False
    assert proj == 12.0


def test_budget_floor_greift_auf_kleinen_maschinen() -> None:
    # 8 GB total ⇒ Budget = max(10.0, 4.0) = 10.0 (Floor)
    soll, _ = ram_deferral_projektion("phase_09_crackle_removal", 9.0, 8.0)
    assert soll is True  # 9.0 + 1.5 = 10.5 > 10.0


def test_boundary_genau_auf_budget_deferriert_nicht() -> None:
    # 16 GB total ⇒ Budget 10.0; 9.0 + 1.0 = 10.0 ⇒ NICHT deferrieren (> strikt)
    soll, proj = ram_deferral_projektion("phase_03_denoise", 9.0, 16.0)
    assert soll is False
    assert proj == 10.0


def test_ueberschreitung_deferriert() -> None:
    soll, _ = ram_deferral_projektion("phase_53_semantic_audio", 15.0, 32.0)
    assert soll is True  # 15.0 + 1.2 = 16.2 > 16.0


def test_determinismus_gleicher_input_gleiche_entscheidung() -> None:
    entscheidungen = {
        ram_deferral_projektion("phase_68_vinylnet_decrackle", 12.3, 32.0)
        for _ in range(50)
    }
    assert len(entscheidungen) == 1


def test_schaetzungssumme_unter_32gb_budget() -> None:
    # Verhaltens-Erhalt: auf ≥ 32-GB-Maschinen darf die Planung den vollen
    # Phasen-Satz nicht kappen — Summe der Inkremente < Budget (16 GB).
    gesamt = sum(PHASE_RAM_ESTIMATE_GB.values())
    budget = max(OWN_RAM_BUDGET_MIN_GB, 32.0 * OWN_RAM_BUDGET_TOTAL_FACTOR)
    assert gesamt <= budget


def test_alle_schaetzungen_positiv_und_endlich() -> None:
    for phase, gb in PHASE_RAM_ESTIMATE_GB.items():
        assert gb > 0.0, phase
        assert gb < 5.0, phase
