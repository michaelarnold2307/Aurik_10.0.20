"""Gemeinsame Policy-Helfer fuer Stem-Routing (Demucs/MDX Priorisierung)."""

from __future__ import annotations

from typing import Any

DEMUX_LIVE_HINTS = frozenset({"live", "concert", "audience", "crowd", "bootleg", "stage"})


def is_live_like_value(value: object) -> bool:
    """True, wenn ein Material/Context-Wert auf live/crowd-nahe Quelle deutet."""
    value_str = str(getattr(value, "value", value)).strip().lower()
    if not value_str:
        return False
    return any(hint in value_str for hint in DEMUX_LIVE_HINTS)


def prefer_demucs_native_from_ctx(ctx: dict[str, Any] | None) -> bool:
    """Aktiviert nativen HTDemucs-Pfad fuer live/crowd-nahe Routing-Kontexte."""
    if not isinstance(ctx, dict) or not ctx:
        return False

    for key in ("material_type", "material", "source_type", "medium", "recording_type"):
        if is_live_like_value(ctx.get(key, "")):
            return True

    transfer_chain = ctx.get("transfer_chain")
    if isinstance(transfer_chain, (list, tuple)):
        return any(is_live_like_value(step) for step in transfer_chain)

    return False


def prefer_demucs_native_from_material(material: object) -> bool:
    """Aktiviert nativen HTDemucs-Pfad fuer live/crowd-nahes Material."""
    return is_live_like_value(material)
