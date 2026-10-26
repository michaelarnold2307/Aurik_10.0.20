"""KP-5-Vertragstests: BigVGAN-F3-Studio-Finisher-Flag bleibt gesperrt (W-13).

Pinned den Ist-Zustand „Kandidat vorbereitet, Aktivierung ausstehend“:
- Flag `use_bigvgan_f3_studio` Default AUS (§v10.802 (copilot-instructions.md)).
- WOHLKLANG_CLAIMS-W-13 referenziert exakt dieses Flag und trägt `gesperrt`.
- Der E5-Abgleich (scripts/wohlklang_gate.py) prüft die Code-Vertrags-Kohärenz.

Lehre D-K3-20: keine `try/except: pass`-Hüllen.
"""

from __future__ import annotations

from pathlib import Path

import backend.core.music_model_flags as mmf

_ROOT = Path(__file__).resolve().parent.parent.parent


def test_kp5_flag_default_aus() -> None:
    """Flag ist aus — der Kandidat-Pfad darf ohne Sign-off nie laufen."""
    assert mmf.use_bigvgan_f3_studio is False


def test_w13_vertrag_referenziert_flag() -> None:
    """W-13 in WOHLKLANG_CLAIMS trägt das Flag + gesperrt (§G9 (copilot-instructions.md))."""
    claims = (_ROOT / ".github" / "WOHLKLANG_CLAIMS.md").read_text(encoding="utf-8")
    w13_lines = [line for line in claims.splitlines() if line.strip().startswith("| W-13 ")]
    assert len(w13_lines) == 1, "genau eine W-13-Zeile erwartet"
    row = w13_lines[0]
    assert "use_bigvgan_f3_studio" in row
    assert "gesperrt" in row


def test_w13_flag_und_claims_kohaerent() -> None:
    """Code-Flag und Vertragszeile dürfen nicht auseinanderlaufen (E5-Muster)."""
    claims = (_ROOT / ".github" / "WOHLKLANG_CLAIMS.md").read_text(encoding="utf-8")
    w13_rows = [line for line in claims.splitlines() if line.strip().startswith("| W-13 ")]
    assert w13_rows, "W-13 fehlt"
    cells = [c.strip() for c in w13_rows[0].strip().strip("|").split("|")]
    # Spalten: ID | Name | Flag-Referenz | Status | ...
    assert "use_bigvgan_f3_studio" in cells[2]
    assert cells[3] == "AUS"
    assert mmf.use_bigvgan_f3_studio is False
