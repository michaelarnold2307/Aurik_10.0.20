"""
conftest.py für tests/musical_goals/
=====================================
Verhindert Segfaults durch xdist-Parallelisierung:
Numba und bestimmte C-Extensions sind nicht fork-safe wenn 2 Worker gleichzeitig
denselben Python-Prozess forken und danach dieselben C-Extensions importieren.

Lösung: Alle musical_goals-Tests werden in dieselbe xdist-Gruppe ("musical_goals_group")
eingeordnet → laufen auf genau einem Worker → kein gleichzeitiger Import-Konflikt.
"""

import pytest


def pytest_collection_modifyitems(items, config):
    """Gruppiert alle musical_goals-Tests auf einen einzigen xdist-Worker.

    Verhindert Segfaults durch Fork-unsafe C-Extensions (numba, scipy-Internals)
    bei paralleler xdist-Ausführung mit -n 2.
    """
    for item in items:
        if item.fspath and "musical_goals" in str(item.fspath):
            item.add_marker(pytest.mark.xdist_group("musical_goals_group"))
