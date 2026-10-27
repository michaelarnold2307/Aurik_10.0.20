"""Tests für den Release-Report (Pfeiler 8: keine Zeile ohne Beleg)."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.release_report import build_report

_QUELLEN_RE = re.compile(r"\(Quelle: .+\)$")


def test_report_grundstruktur_ohne_quellen_ist_luecke() -> None:
    md, payload = build_report("0.0.0-unbekannt")
    assert "# Release-Report" in md
    assert "## Provenienz-Lücken" in md
    assert any("Single Source verletzt" in g for g in payload["provenienz_luecken"])
    assert any("nicht belegt" in g for g in payload["provenienz_luecken"]), "fehlende Belege = Lücke"
    for line in md.splitlines():
        if not line.startswith("- "):
            continue
        if line.startswith("- keine") or line.startswith("- nicht belegt") or "(Quelle: " in line:
            continue
        assert any(
            line[2:].startswith(g[:40]) or g.startswith(line[2:40]) for g in payload["provenienz_luecken"]
        ) or line.startswith("- Version "), f"Aussagezeile ohne Quelle: {line!r}"


def test_report_mit_kandidaten_und_hoerverdikten(tmp_path: Path) -> None:
    kand = tmp_path / "kand.json"
    kand.write_text(
        json.dumps(
            {
                "kandidaten": [
                    {
                        "flag": "use_aspade_declipper",
                        "summary": {"fallback_count": 0, "flag_freigabe": "offen (Hör-Gate ausstehend)"},
                    },
                    {"flag": "use_tf_gridnet_quality", "summary": {"n_ok": 3}},
                ],
            }
        ),
        encoding="utf-8",
    )
    verd = tmp_path / "verd.json"
    verd.write_text(
        json.dumps(
            {
                "statistik": {
                    "n_entscheidend": 20,
                    "siegquote_a": 0.75,
                    "p_wert_vorzeichen_test": 0.041389,
                    "signifikant": True,
                },
            }
        ),
        encoding="utf-8",
    )
    md, payload = build_report("0.0.0-unbekannt", candidates_report=kand, hoerverdikte=verd)
    assert "use_aspade_declipper" in md and "fallback_count=0" in md
    assert "use_tf_gridnet_quality" in md and "nicht im Report" in md
    assert "0.041389" in md and "signifikant" in md
    assert any("fallback_count fehlt" in g for g in payload["provenienz_luecken"]), "Lücke wird benannt"
    for line in md.splitlines():
        if line.startswith("- ") and "(Quelle: " not in line and not line.startswith("- keine"):
            assert (
                line.startswith("- nicht belegt")
                or line.startswith("- Version ")
                or line[2:] in payload["provenienz_luecken"]
            ), f"Zeile ohne Quelle: {line!r}"
