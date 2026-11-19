"""Maßnahme 1 (2026-10-07): Pfad-E2E-Register — Phantom-Verbot (normativ).

Der Befund der K2-Inventur: Pfade existierten, ohne erreicht zu werden
(vocoder_chain: null Aufrufer), oder wirkten unter falschem Etikett (Routing-
„NVSR" = SBR-Kern; echtes nvsr.ts verwaist). Das Register macht den Status
jedes auditierten Pfads explizit; dieser Test pinnt die Kern-Invarianten:

  - Der Register-Check läuft sauber (fail-closed).
  - Die auditierten Kernpfade sind enthalten (kein stilles Leeren).
  - Produktive/Challenger-Pfade haben E2E-Nachweis oder dokumentierte Lücke.
  - Phantom-Einträge sind nur als Übergang MIT removal_plan erlaubt
    (Endzustand-Pflicht: verdrahten, archivieren oder entfernen).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_REGISTRY = _ROOT / "config" / "path_e2e_registry.json"

_KERN_PFADE = (
    "vocos_48khz_studio_finisher",
    "nvsr_sbr_phase06",
    "aero_bwe_challenger",
    "nvsr_ts_k0",
    "vocoder_chain_spec04",
    "sota_speech_superres",
)


def _load() -> dict:
    assert _REGISTRY.exists(), f"Pfad-E2E-Register fehlt: {_REGISTRY}"
    return json.loads(_REGISTRY.read_text(encoding="utf-8"))


def test_registry_check_passes() -> None:
    proc = subprocess.run(
        [sys.executable, str(_ROOT / "scripts" / "path_e2e_check.py"), "--json"],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(_ROOT),
        check=False,
    )
    assert proc.returncode == 0, f"Pfad-E2E-Gate fail:\n{proc.stdout}\n{proc.stderr}"


def test_registry_covers_audited_critical_paths() -> None:
    ids = {e["id"] for e in _load()["paths"]}
    for pfad in _KERN_PFADE:
        assert pfad in ids, f"Pfad fehlt im E2E-Register: {pfad}"


def test_productive_and_challenger_have_e2e_or_documented_gap() -> None:
    for e in _load()["paths"]:
        if e["status"] in ("productive", "challenger"):
            ok = bool(e.get("e2e_test")) or (e.get("e2e_gap") and e.get("gap_reason"))
            assert ok, f"{e['id']}: produktiv/challenger ohne E2E-Nachweis oder dokumentierte Luecke"


def test_phantom_entries_have_removal_plan() -> None:
    """phantom_transition ist ein Uebergangszustand — der Plan muss existieren.

    Dauerhaftigkeit wird nicht per Test verboten (Backlog-Sichtbarkeit statt
    rotem Test, D-K3-13-Lehre), sondern durch das Register dokumentiert und in
    den Welle-1-Aufgaben weiter verfolgt.
    """
    for e in _load()["paths"]:
        if e["status"] == "phantom_transition":
            assert str(e.get("removal_plan") or "").strip(), f"{e['id']}: phantom_transition ohne removal_plan"
