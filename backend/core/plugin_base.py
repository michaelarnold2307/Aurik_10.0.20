"""Base class for all Aurik ML plugins (§A2).

Every plugin that loads an ML model MUST inherit from MLPluginBase and
implement the ``_model_loaded`` property.  This contract is validated at
startup by ml_model_readiness._validate_all_checks() — plugins without
this property produce a CRITICAL log entry immediately, not silently
months later when a phase happens to probe readiness.

Usage::

    from backend.core.plugin_base import MLPluginBase

    class MyPlugin(MLPluginBase):
        def __init__(self) -> None:
            super().__init__()
            self._session: object | None = None
            self._load_model()

        @property
        def _model_loaded(self) -> bool:
            return self._session is not None
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class MLPluginBase(ABC):
    """Abstract base for all ML model plugins.

    Enforces:
    - ``_model_loaded`` property: ml_model_readiness probes this to
      determine if the model is ready for inference.
    - ``__init__`` must call ``super().__init__()``.

    Plugins that do NOT inherit from this base (legacy plugins) still
    work, but they risk the silent PANNs-bug class: missing
    ``_model_loaded`` → permanent readiness failure.
    """

    def __init__(self) -> None:
        """Initialise base state (called by subclasses via super().__init__())."""
        super().__init__()

    @property
    @abstractmethod
    def _model_loaded(self) -> bool:
        """Return True if the ML model is loaded and ready for inference.

        This is probed by ml_model_readiness.check_ml_model_ready() and
        MUST be implemented by every ML plugin subclass.
        """
        ...
