"""Phase Ambience-Polish (Spec 25, Hörordnung Ebene 2).

Additive, unter der Maskierungsschwelle kalibrierte Raumhülle als Politur
unmittelbar vor der Glue Stage. Schaltbar über das Materialprofil:
Default AN für Vintage-Klassen (Schellack, Kassette, Rumpel-/Hiss-lastige
Bänder), AUS für bereits „lebendiges“ Material (CD, Streaming, Vinyl …).
Überschreibbar über kwargs (`ambience_polish_enabled`, `ambience_blend`,
`ambience_safety_margin_db`, `ambience_defect_bands`).

Pipeline-Position (Spec 25, §III (copilot-instructions.md)):
    phase_47_truepeak_limiter → phase_ambience_polish → phase_glue_stage

Invarianten H1–H4 (nie verletzbar, Tests: tests/unit/test_ambience_match_plugin.py):
    H1 Ambiente je Bark-Band ≤ Maskierungsschwelle − σ.
    H2 Signal wird nie verändert (rein additiv).
    H3 Defekt-Bänder zusätzlich um ≥ 12 dB abgesenkt.
    H4 blend=0 ⇒ bit-identischer Passthrough.

Spec: .github/specs/25_ambience_match_plugin.md
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from backend.core.phase_strength_contract import resolve_phase_strength_contract

from .phase_interface import PhaseCategory, PhaseInterface, PhaseMetadata, PhaseResult

logger = logging.getLogger(__name__)

# Spec 25: Default AN für Vintage-Klassen (Schellack, Kassette, Bänder).
# „Lebendiges“ Material (Vinyl, CD, DAT, Streaming, Codecs) bleibt aus.
_VINTAGE_MATERIAL_KEYS = frozenset(
    {
        "shellac",
        "cassette",
        "tape",
        "reel_tape",
        "wire_recording",
        "wax_cylinder",
        "lacquer_disc",
    }
)

_BASE_BLEND = 0.30  # Default-Blend; PMGG-Strength skaliert zentral (§V7 (copilot-instructions.md))
_SAFETY_MARGIN_DB = 6.0  # Spec 25: σ = 6 dB (einziger freier Nutzparameter)


def _material_key(material: Any) -> str:
    if material is None:
        return ""
    if hasattr(material, "value"):
        material = material.value
    return str(material).strip().lower()


def _is_vintage_material(material: Any) -> bool:
    key = _material_key(material)
    if key in _VINTAGE_MATERIAL_KEYS:
        return True
    # Legacy-Schreibweisen: "Schellack", "Kassette", "Band" …
    return any(alias in key for alias in ("schellack", "kassette", "band"))


class AmbiencePolishPhase(PhaseInterface):
    """Spec 25: additive, maskierte Raumhülle — Politur vor der Glue Stage."""

    def get_metadata(self) -> PhaseMetadata:
        return PhaseMetadata(
            phase_id="phase_ambience_polish",
            name="Ambience-Polish",
            category=PhaseCategory.ENHANCEMENT,
            priority=4,
            version="1.0.0",
            estimated_time_factor=0.03,
            memory_requirement_mb=32,
            is_cpu_intensive=False,
            description="Spec 25: additive, maskierte Raumhülle unmittelbar vor der Glue Stage",
        )

    def process(  # type: ignore[override]
        self,
        audio: np.ndarray,
        sample_rate: int = 48000,
        **kwargs: Any,
    ) -> PhaseResult:
        audio_in = np.asarray(audio)
        if audio_in.dtype != np.float32:
            audio_in = audio_in.astype(np.float32)

        enabled = kwargs.get("ambience_polish_enabled")
        if enabled is None:
            enabled = _is_vintage_material(kwargs.get("material_type", kwargs.get("material")))
        if not bool(enabled) or audio_in.size == 0:
            return PhaseResult(
                audio=audio_in,
                success=True,
                metadata={"ambience_polish": "disabled", "reason": "material_profile"},
            )

        # Stärke zentral über den Strength-Contract (§V7 (copilot-instructions.md)).
        contract = resolve_phase_strength_contract(kwargs)
        effective_strength = float(contract["effective_strength"])
        if effective_strength <= 0.0:
            return PhaseResult(
                audio=audio_in,
                success=True,
                metadata={"ambience_polish": "disabled", "reason": "strength_zero"},
            )

        blend = float(np.clip(float(kwargs.get("ambience_blend", _BASE_BLEND)) * effective_strength, 0.0, 1.0))
        sigma = float(kwargs.get("ambience_safety_margin_db", _SAFETY_MARGIN_DB))
        defect_bands = kwargs.get("ambience_defect_bands")

        from plugins.ambience_match_plugin import AmbienceMatchPlugin

        plugin = AmbienceMatchPlugin(safety_margin_db=sigma, blend=blend)
        out = plugin.process(audio_in, sample_rate, defect_bands=defect_bands)

        # §0a (copilot-instructions.md): NaN/Inf-Schutz in jeder Phase.
        output_audio = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        if not np.isfinite(output_audio).all():
            logger.warning("Verarbeitungsschritt_ambience_polish: NaN/Inf im Ausgabe — mit nan_to_num bereinigt")
            output_audio = np.nan_to_num(output_audio, nan=0.0, posinf=0.0, neginf=0.0)

        return PhaseResult(
            audio=output_audio,
            success=True,
            metadata={
                "ambience_polish": "applied",
                "blend": float(blend),
                "safety_margin_db": float(sigma),
                "strength": effective_strength,
            },
        )
