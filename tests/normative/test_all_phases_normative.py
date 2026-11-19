import glob
import importlib
import os

import pytest

PHASES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../backend/core/phases"))
PHASE_PATTERN = os.path.join(PHASES_DIR, "phase_*.py")

# Liste aller Phasenmodule (ohne __init__.py, Hilfsdateien und Interface)
phase_files = [
    f for f in glob.glob(PHASE_PATTERN) if not f.endswith("__init__.py") and not f.endswith("phase_interface.py")
]
phase_names = sorted([os.path.splitext(os.path.basename(f))[0] for f in phase_files])


@pytest.mark.parametrize("phase_name", phase_names)
def test_phase_import_and_metadata(phase_name):
    """
    Normativer Basistest: Jedes Phasenmodul lässt sich importieren und liefert Metadaten.
    """
    module = importlib.import_module(f"backend.core.phases.{phase_name}")
    # Suche nach einer PhaseInterface-Instanz oder Klasse
    phase_class = None
    for attr in dir(module):
        obj = getattr(module, attr)
        if hasattr(obj, "__bases__") and any("PhaseInterface" in str(base) for base in getattr(obj, "__bases__", [])):
            phase_class = obj
            break
    assert phase_class is not None, f"{phase_name}: Keine PhaseInterface-Klasse gefunden!"
    # Instanziierung (mit Defaults)
    try:
        phase = phase_class()
    except Exception as e:
        pytest.fail(f"{phase_name}: Instanziierung fehlgeschlagen: {e}")
    # Metadaten abrufen
    if hasattr(phase, "get_metadata"):
        meta = phase.get_metadata()
        assert hasattr(meta, "phase_id"), f"{phase_name}: Metadaten ohne phase_id!"
        assert hasattr(meta, "name"), f"{phase_name}: Metadaten ohne name!"
    else:
        pytest.skip(f"{phase_name}: Keine get_metadata()-Methode vorhanden.")
