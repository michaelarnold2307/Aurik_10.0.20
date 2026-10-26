"""WP-1 · K2-Signalpfad-Audit — §III.11/§III.13-Domänen-Transparenz der Vocoder/SR-Pfade.

Prüft (fail-closed) für die fünf K2-Artefakte (hifi_gan, vocos, vocos_48khz,
nvsr, aero), dass jeder Lade-/Verwendungsort die Domänen-Warnung
(§V6 (copilot-instructions.md)) bzw. die §WP-1-Domänen-Deklaration trägt.
Grundlage: `.github/SOTA_DEFICIT_REGISTER.md` D-K2-1…D-K2-5 und
`.github/ML_MODEL_DOMAIN_REGISTRY.md`.

Exit-Codes: 0 = alle Marker vorhanden, 2 = Verstoß (mit Pfad + fehlendem Marker).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Pfad → Liste der Marker (alle müssen enthalten sein).
CHECKS: list[tuple[str, list[str]]] = [
    (
        "plugins/vocos_plugin.py",
        [
            "§V6 (copilot-instructions.md) Vocos",
            "sprach-trainiert",
            "§III.11 (copilot-instructions.md)",
        ],
    ),
    (
        "plugins/hifigan_plugin.py",
        [
            "§V6 (copilot-instructions.md) HiFi-GAN",
            "unbelegte Trainings-Domäne",
            "§III.13 (copilot-instructions.md)",
        ],
    ),
    (
        "backend/core/dsp/sota_speech_superres.py",
        [
            "§WP-1 (2026-10-25)",
            "unbelegte",
            "§III.13 (copilot-instructions.md)",
            "§V6 (copilot-instructions.md) HiFi-GAN",
        ],
    ),
    (
        "plugins/aero_plugin.py",
        [
            "§WP-1-Domänen-Deklaration",
            "SPRACH-trainiert",
            "§III.11 (copilot-instructions.md)",
        ],
    ),
    (
        "plugins/nvsr_plugin.py",
        [
            "§V6 (copilot-instructions.md)",
        ],
    ),
]


def main() -> int:
    violations: list[tuple[str, str]] = []
    for rel, markers in CHECKS:
        path = ROOT / rel
        if not path.is_file():
            violations.append((rel, "Datei fehlt"))
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for marker in markers:
            if marker not in text:
                violations.append((rel, f"Marker fehlt: {marker}"))
    if violations:
        for rel, reason in violations:
            print(f"VERSTOSS {rel}: {reason}")
        return 2
    print(f"K2-Signalpfad-Audit OK ({len(CHECKS)} Artefakt-Pfade, alle Domänen-Marker vorhanden)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
