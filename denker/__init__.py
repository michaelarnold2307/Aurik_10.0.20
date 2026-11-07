"""
Aurik 10.0.0 — Denker-Architektur
============================

Sieben spezialisierte Denker + ein Orchestrator (AurikDenker) bilden das
Fundament der kognitiven Restaurierungs-Intelligenz von Aurik 10.0.0.

Jeder Denker kapselt eine Wissensdomäne und stellt eine klare,
NaN-sichere API bereit.  Der AurikDenker koordiniert alle Denker
entlang der kanonischen Pipeline (§2.2).

Domänen:
    TontraegerDenker        – Tonträger-/Material-Erkennung (§2.1)
    DefektDenker            – Defekt-Scan + Kausale Ursachenanalyse (§2.4)
    StrategieDenker         – 8×RT-Budgetplanung + Performance-Guard (§2.1)
    RestaurierDenker        – Vollständige Pipeline-Ausführung (§2.2)
    ReparaturDenker         – Gezielte DSP-Reparaturen (Click, Hum, Clip)
    RekonstruktionsDenker   – Lückenerkennung + Inpainting (§2.12)
    ExzellenzDenker         – Musical-Goals + GP-Exzellenz (§2.5 / §1.2)
    AurikDenker             – Orchestrator (Haupt-Entry-Point)
    PerceptualQualityCouncil – Zentrale Qualitätsbewertung (§v10.3)
"""

from .aurik_denker import AurikDenker, get_aurik_denker, restauriere
from .cross_phase_coordinator import CrossPhaseCoordinator
from .defekt_denker import DefektDenker, get_defekt_denker
from .exzellenz_denker import ExzellenzDenker, get_exzellenz_denker
from .rekonstruktions_denker import RekonstruktionsDenker, get_rekonstruktions_denker
from .reparatur_denker import ReparaturDenker, get_reparatur_denker
from .restaurier_denker import RestaurierDenker, get_restaurier_denker
from .strategie_denker import StrategieDenker, get_strategie_denker
from .tontraeger_denker import TontraegerDenker, get_tontraeger_denker
from .tontraegerkette_denker import KettenErgebnis, TontraegerketteDenker, get_tontraegerkette_denker

__all__ = [
    "AurikDenker",
    "CrossPhaseCoordinator",
    "DefektDenker",
    "ExzellenzDenker",
    "KettenErgebnis",
    "RekonstruktionsDenker",
    "ReparaturDenker",
    "RestaurierDenker",
    "StrategieDenker",
    "TontraegerDenker",
    "TontraegerketteDenker",
    "get_aurik_denker",
    "get_defekt_denker",
    "get_exzellenz_denker",
    "get_rekonstruktions_denker",
    "get_reparatur_denker",
    "get_restaurier_denker",
    "get_strategie_denker",
    # Convenience-Accessors (§3.2 Singleton-Pattern)
    "get_tontraeger_denker",
    "get_tontraegerkette_denker",
    "restauriere",
]
