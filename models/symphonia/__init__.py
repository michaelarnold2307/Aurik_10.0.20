"""Lazy Exports für Symphonia ohne Torch-Import beim Paketimport."""

from __future__ import annotations

__all__ = ["SymphoniaModel", "SymphoniaExportWrapper", "create_symphonia"]


def __getattr__(name: str):
    if name in __all__:
        from models.symphonia.symphonia_model import (  # pylint: disable=import-outside-toplevel
            SymphoniaExportWrapper,
            SymphoniaModel,
            create_symphonia,
        )

        return {
            "SymphoniaModel": SymphoniaModel,
            "SymphoniaExportWrapper": SymphoniaExportWrapper,
            "create_symphonia": create_symphonia,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
