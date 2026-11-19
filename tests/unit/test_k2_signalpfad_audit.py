"""WP-1-K2-Vertragstests: Domänen-Transparenz der Sprach-Signalpfade (D-K2-1…D-K2-5).

Pinnt die §WP-1-Audit-Marker fail-closed — fehlt eine Deklaration/Warnung an
einem K2-Ladepfad, fällt der Test. Keine `try/except: pass`-Hüllen
(Lehre D-K3-20).
"""

from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent


def _text(rel: str) -> str:
    return (_ROOT / rel).read_text(encoding="utf-8", errors="replace")


def test_vocos_plugin_domaenen_warnung() -> None:
    text = _text("plugins/vocos_plugin.py")
    assert "§V6 (copilot-instructions.md) Vocos" in text
    assert "sprach-trainiert" in text


def test_hifigan_plugin_domaenen_warnung() -> None:
    text = _text("plugins/hifigan_plugin.py")
    assert "§V6 (copilot-instructions.md) HiFi-GAN" in text
    assert "unbelegte Trainings-Domäne" in text


def test_sota_speech_superres_deklaration() -> None:
    text = _text("backend/core/dsp/sota_speech_superres.py")
    assert "§WP-1 (2026-10-25)" in text
    assert "§V6 (copilot-instructions.md) HiFi-GAN" in text


def test_aero_plugin_domaenen_deklaration() -> None:
    text = _text("plugins/aero_plugin.py")
    assert "§WP-1-Domänen-Deklaration" in text
    assert "SPRACH-trainiert" in text


def test_audit_skript_gruen() -> None:
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-B", str(_ROOT / "scripts" / "audit_k2_signalpfad.py")],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "K2-Signalpfad-Audit OK" in result.stdout
