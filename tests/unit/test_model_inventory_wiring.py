"""§G9 (copilot-instructions.md): Verdrahtungs-Audit der Modellartefakte (WP0 · Befund 2026-10-06).

Hintergrund: `models/` enthielt 61 Verzeichnisse, während das Inventar nur 30
kuratierte Einträge pflegte. Lokal vorhandenes Kapital ohne Konsumenten —
trainiert oder A/B-gewonnen, aber von keiner Produktionsstelle geladen — war
damit unsichtbar (`ddsp_predictor/c4_head.pth`, `scnet_4stems`).

Die Tests sind modell-unabhängig: geprüft wird nur der Quelltext-Abgleich,
nicht die Existenz von Gewichten.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]


def _load_inventory():
    spec = importlib.util.spec_from_file_location("model_inventory", _ROOT / "scripts" / "model_inventory.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # @dataclass löst über sys.modules[cls.__module__] auf — ohne Registrierung
    # scheitert exec_module mit AttributeError (NoneType.__dict__).
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def inventory():
    return _load_inventory()


def test_no_undocumented_orphan_model_dirs(inventory):
    """Jedes Modellverzeichnis ist verdrahtet oder begründet ausgenommen."""
    wiring = inventory.audit_wiring()
    if not wiring:
        pytest.skip("kein models/-Bestand vorhanden (z. B. CI ohne Modell-Bundle)")
    orphan = sorted(name for name, status in wiring.items() if status == "verwaist")
    assert orphan == [], (
        f"verwaiste Modellverzeichnisse ohne Begründung: {orphan} — verdrahten, entfernen "
        "oder in scripts/model_inventory._WIRING_ALLOWLIST begründet aufnehmen (§G8 copilot-instructions.md)"
    )


def test_wiring_allowlist_points_at_existing_dirs(inventory):
    """Die Ausnahmeliste darf nicht verwaisen, sonst kaschiert sie spätere Löschungen."""
    models_dir = _ROOT / "models"
    if not models_dir.is_dir():
        pytest.skip("kein models/-Bestand vorhanden")
    for name in inventory.wiring_allowlist():
        assert (models_dir / name).is_dir(), f"Ausnahme verweist auf fehlendes Verzeichnis: {name}"


def test_wiring_status_vocabulary_is_closed(inventory):
    """Nur zwei Status-Werte — kein stiller Wildwuchs im Audit."""
    for name, status in inventory.audit_wiring().items():
        assert status == "verdrahtet" or status.startswith("bewusst-verwaist: "), f"{name}: {status!r}"
