from __future__ import annotations

import builtins
import importlib


def test_ml_refinement_thread_uses_core_fallback_when_bridge_import_fails(monkeypatch):
    """Bei Bridge-Importfehler muss MLRefinementThread auf Core-DeferredRefinementJob fallen."""
    import Aurik10.ui.ml_refinement_thread as mrt

    original_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "backend.api.bridge":
            raise ImportError("simulated bridge import failure")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    reloaded = importlib.reload(mrt)
    try:
        assert reloaded.DeferredRefinementJob is not None
        assert hasattr(reloaded.DeferredRefinementJob, "__name__")
        assert reloaded.DeferredRefinementJob.__name__ == "DeferredRefinementJob"
    finally:
        importlib.reload(mrt)
