from __future__ import annotations

"""Regressionstest für importierbare SOTA-Model-Registry in benchmarks/sota_eval.py."""


import importlib.util

from benchmarks.sota_eval import SUPPORTED_MODELS


def test_supported_models_module_paths_resolvable() -> None:
    """Alle in SUPPORTED_MODELS referenzierten Module müssen per importlib auffindbar sein."""
    missing: list[tuple[str, str]] = []

    for model_name, module_path in SUPPORTED_MODELS.items():
        if importlib.util.find_spec(module_path) is None:
            missing.append((model_name, module_path))

    assert not missing, f"Nicht auflösbare SUPPORTED_MODELS-Einträge: {missing}"
