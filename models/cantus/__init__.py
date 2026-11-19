# models/cantus — Cantus Vocal Restorer (MERT + Flow-Matching DiT)
# Architektur: siehe models/cantus/README.md · Gewichte: Platzhalter bis Training
from __future__ import annotations

__all__ = ["CantusConfig", "CantusModel", "CantusExportWrapper", "create_cantus"]


def __getattr__(name: str):
    # Lazy import: torch erst bei Bedarf laden (kein Torch-Import zur Paket-Initialisierung)
    if name in __all__:
        from models.cantus.cantus_model import (  # pylint: disable=import-outside-toplevel
            CantusConfig,
            CantusExportWrapper,
            CantusModel,
            create_cantus,
        )

        _map = {
            "CantusConfig": CantusConfig,
            "CantusModel": CantusModel,
            "CantusExportWrapper": CantusExportWrapper,
            "create_cantus": create_cantus,
        }
        return _map[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
