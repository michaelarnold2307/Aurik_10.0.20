"""D-K3-42-Guard: Era-Material-Flips brauchen physikalische Plausibilität.

Pinnt die zwei Vetos von ``_era_material_flip_admissible``
(``backend/core/unified_restorer_v3.py``):

  - **Veto A — Träger-Unmöglichkeit:** Das Era-Jahrzehnt liegt vor dem
    Erfindungs-Floor des physikalisch gemessenen Primärträgers
    (``MEDIUM_DECADE_FLOOR``).
  - **Veto B — UNKNOWN-Physik + restriktives Altmaterial:** physikalisch
    'unknown' + Era-Material mit Conservativeness-Rang ≥ 10 (never-worsen-
    Asymmetrie: für echtes Altmaterial ist der Cap ein No-Op).

Die Fälle (A)/(B)/(C) sind die gemessenen Real-Audio-Konstellationen des
Nachweises 2026-10-07 (Elke Best 1977, Register D-K3-42).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from backend.core.defect_scanner import MaterialType
from backend.core.unified_restorer_v3 import _era_material_flip_admissible


@dataclass
class _MC:
    primary_material: str
    confidence: float
    transfer_chain: list[str]
    is_multi_generation: bool = False


@dataclass
class _Era:
    decade: int
    confidence: float
    material_prior: str


def test_veto_a_mp3_chain_vs_1890_era() -> None:
    """(A) Realbefund Fenster 80–90 s: MD mp3_high 0,62 + Era 1890/wax → Veto A."""
    ok, reason = _era_material_flip_admissible(
        MaterialType.UNKNOWN,
        MaterialType.WAX_CYLINDER,
        _MC("mp3_high", 0.62, ["mp3_high"]),
        _Era(1890, 0.90, "wax_cylinder"),
    )
    assert not ok, "Träger-Unmöglichkeit muss den Flip blockieren"
    assert "Träger-Unmöglichkeit" in reason


def test_veto_a_vinyl_chain_vs_1890_era() -> None:
    """Vinyl-Kette (Floor 1950) kann keine 1890er-Ära tragen."""
    ok, _ = _era_material_flip_admissible(
        MaterialType.UNKNOWN,
        MaterialType.WAX_CYLINDER,
        _MC("vinyl", 0.60, ["vinyl"]),
        _Era(1890, 0.90, "wax_cylinder"),
    )
    assert not ok


def test_veto_b_unknown_physics_plus_wax() -> None:
    """(B) Realbefund Fenster 0–10 s: MD unknown 0,60 + Era 1890/wax → Veto B."""
    ok, reason = _era_material_flip_admissible(
        MaterialType.UNKNOWN,
        MaterialType.WAX_CYLINDER,
        _MC("unknown", 0.60, ["unknown"]),
        _Era(1890, 0.90, "wax_cylinder"),
    )
    assert not ok, "UNKNOWN-Physik darf keine Wax-Restriktion aus reinem Prior erzeugen"
    assert "restriktives" in reason


def test_no_veto_measured_slice_case() -> None:
    """(C) Realbefund 15-s-Slice: MD unknown 0,60 + Era 1960/vinyl 0,50 → zulässig."""
    ok, _ = _era_material_flip_admissible(
        MaterialType.UNKNOWN,
        MaterialType.VINYL,
        _MC("unknown", 0.60, ["unknown"]),
        _Era(1960, 0.50, "vinyl"),
    )
    assert ok, "der gutartige Gleichstand-Fall darf unverändert bleiben"


def test_no_veto_plausible_analog_flip() -> None:
    """Kassette (Floor 1960) + Era 1970/reel_tape: plausibel → Flip weiter erlaubt."""
    ok, _ = _era_material_flip_admissible(
        MaterialType.CASSETTE,
        MaterialType.REEL_TAPE,
        _MC("cassette", 0.60, ["cassette"]),
        _Era(1970, 0.83, "reel_tape"),
    )
    assert ok


def test_no_veto_restrictive_era_with_real_physics() -> None:
    """Veto B greift NICHT, wenn die Physik ein reales Material behauptet.

    Physik = Shellac (Floor 1900, kein UNKNOWN) + Era 1900/wax_cylinder:
    kein Träger-Widerspruch, kein UNKNOWN-Veto → Flip bleibt erlaubt.
    """
    ok, _ = _era_material_flip_admissible(
        MaterialType.SHELLAC,
        MaterialType.WAX_CYLINDER,
        _MC("shellac", 0.60, ["shellac"]),
        _Era(1900, 0.85, "wax_cylinder"),
    )
    assert ok


def test_guard_is_wired_into_conflict_rule() -> None:
    """Verdrahtungs-Anker (Muster D-K3-35): der Guard läuft in BEIDEN Branches."""
    src = Path(__file__).resolve().parents[2] / "backend" / "core" / "unified_restorer_v3.py"
    text = src.read_text(encoding="utf-8")
    assert text.count("_era_material_flip_admissible(") >= 3, "Helper-Definition + zwei Aufrufe erwartet"
    assert "Era-Dominanz VETO" in text
    assert "Konservativwahl VETO" in text
