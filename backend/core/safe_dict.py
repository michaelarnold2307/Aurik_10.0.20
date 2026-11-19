"""
§2.59 SafeDict: .get()-Masking-Detektor (2026-07-09)

Wrapper um dict, der warnt, wenn .get() mit Default auf einen
nicht existierenden Key zugreift. Verhindert stille Bugs wie:
  defect_scores.get("hiss", 0.0)  # "hiss" existiert nicht → KeyError-würdig

Usage:
  from backend.core.safe_dict import SafeDict
  scores = SafeDict({"clicks": 0.8, "hum": 0.3},
                     name="defect_scores",
                     known_keys={"clicks", "hum", "wow", "flutter", ...})

  scores.get("hiss", 0.0)  # WARNING: SafeDict 'defect_scores':
                            # key 'hiss' not in known_keys, default 0.0 returned
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

logger = logging.getLogger(__name__)


class SafeDict(dict):
    """Dict-Wrapper mit Key-Validierung bei .get()-Aufrufen."""

    def __init__(
        self,
        *args: Any,
        name: str = "SafeDict",
        known_keys: set[str] | frozenset[str] | None = None,
        warn_on_missing: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._safe_name = name
        self._known_keys = known_keys
        self._warn = warn_on_missing

    def get(self, key: Any, default: Any = None) -> Any:
        """Wie dict.get(), aber warnt bei unbekannten Keys."""
        if self._warn and self._known_keys is not None:
            if isinstance(key, str) and key not in self._known_keys:
                if key not in self:  # Key wirklich nicht vorhanden
                    logger.debug(
                        "SafeDict '%s': key '%s' not in known_keys (%d keys), "
                        "default %r returned. Möglicher Defekt-Namen-Mismatch.",
                        self._safe_name,
                        key,
                        len(self._known_keys),
                        default,
                    )
        return super().get(key, default)

    def __getitem__(self, key: Any) -> Any:
        """Direkter Zugriff — kein Warnung (KeyError ist genug Feedback)."""
        return super().__getitem__(key)


def make_safe_dict(
    data: Mapping[str, Any] | None,
    name: str = "SafeDict",
    known_keys: set[str] | frozenset[str] | None = None,
) -> SafeDict:
    """Factory: erzeugt SafeDict aus Mapping oder None."""
    return SafeDict(data or {}, name=name, known_keys=known_keys)
