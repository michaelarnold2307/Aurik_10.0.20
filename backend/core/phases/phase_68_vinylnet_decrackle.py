"""Phase 68: VinylNet De-Crackle (ML, chirurgisch) — gelernter Klickdetektor + Hybrid-Reparatur + Transformer.

Werkzeug fuer DefectType.CLICKS und dichte DefectType.CRACKLE-Impulse: entzaeunt das
**chirurgisch zu entfernende** Vinyl-Knistern (einzelne Klicks, Pops/Thumps, Kratzer-Bursts)
per gelernter Pipeline. Abgrenzung (Either-Or, wie Spec 06):
  * spaerliche Einzelimpulse ohne Textur -> phase_09_crackle_removal / phase_27 (klassisch),
  * kontinuierliche Knistern-TEXTUR -> phase_67_crackle_texture_removal (Route A, ML-subtraktiv),
  * diese Phase: ML-Restaurierung mit lokaler Luecken-Rekonstruktion (InpaintNet) und
    Band-Split-Transformer — setzt an, wo die klassische AR-Interpolation schwach ist
    (grosse Luecken, tonales Material) und wo der AR-Detektor Fehlalarme produziert.

 Modell: models/de_crackle/ (VinylNet v4 + ClickNet v4, trainiert auf MUSDB18-HQ-Mixturen +
 synthetische Vinyl-Degradation; s. docs/reports/current/2026-10-12_vinylnet_decrackle.md).
 Laden ueber plugins/vinylnet_decrackle_plugin.py (Torch primär, Checkpoint-Pfad mit
 Preset-Priorität vinylnet_base_best.pt → vinylnet_small_best.pt → runs/vinylnet/best_infer.pt;
 Detektor: detector_v4.pt → runs/detector/detector.pt; ONNX-CPU-Fallback erst nach
 Paritätsnachweis — §III.9 copilot-instructions.md).

Pflicht-Invarianten:
- Deterministisch (§G5 copilot-instructions.md): eval() + inference_mode im Plugin, keine RNG;
  Resampling (nur bei SR != 44100) per deterministischem resample_poly.
- Stereo-Layout-Invariante: beide Layouts bedienen, intern channels-first (C, N).
- Passthrough-Semantik (§0a copilot-instructions.md): Modell nicht ladbar -> bit-identischer
  Passthrough mit Warning (§V6 copilot-instructions.md: kein stiller Fallback).
"""

from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np

from backend.core.phases.phase_interface import (
    PhaseCategory,
    PhaseInterface,
    PhaseMetadata,
    PhaseMode,
    PhaseResult,
)

logger = logging.getLogger(__name__)


class VinylNetDeCracklePhase(PhaseInterface):
    """Chirurgische ML-De-Crackle-Phase (VinylNet v4)."""

    PHASE_ID = "phase_68_vinylnet_decrackle"

    def __init__(self, strength: float = 0.9) -> None:
        self.strength = float(strength)
        self._plugin: Any = None

    def get_metadata(self) -> PhaseMetadata:
        return PhaseMetadata(
            phase_id=self.PHASE_ID,
            name="VinylNet De-Crackle (ML, chirurgisch)",
            category=PhaseCategory.DEFECT_REMOVAL,
            priority=7,
            version="1.0.0",
            is_cpu_intensive=True,
            memory_requirement_mb=512,
            quality_impact=0.9,
            description=(
                "Chirurgische ML-De-Crackle-Restaurierung (VinylNet v4): gelernter Klickdetektor "
                "(M/S), InpaintNet fuer grosse Luecken, Band-Split-Transformer mit RoPE + ThumpNet. "
                "Either-Or: Textur->phase_67, spaerlich-klassisch->phase_09/27 (Spec 03/06)."
            ),
            defect_types=["clicks", "crackle"],
            phase_mode=PhaseMode.RESTORATION_ONLY,
        )

    # ------------------------------------------------------------------ Laden
    def _get_plugin(self) -> Any:
        """Lazy-Singleton auf das kanonische Plugin (§G9 copilot-instructions.md)."""
        if self._plugin is None:
            try:
                from plugins.vinylnet_decrackle_plugin import get_vinylnet_decrackle_plugin

                self._plugin = get_vinylnet_decrackle_plugin()
            except Exception as exc:  # pragma: no cover - Fallback-Pfad
                logger.warning(
                    "VinylNetDeCrackle: Plugin nicht ladbar (%s) -> Passthrough (§V6, copilot-instructions.md)", exc
                )
                self._plugin = False  # type: ignore[assignment]  # Mark as unavailable
        return self._plugin if self._plugin is not False else None

    # ------------------------------------------------------------- Verarbeitung
    def process(
        self,
        audio: np.ndarray,
        sample_rate: int = 44100,
        material_type: Any = None,
        strength: float | None = None,
        **kwargs: Any,
    ) -> PhaseResult:
        arr = np.asarray(audio, dtype=np.float32)
        if arr.ndim == 1:
            mono = True
            x = np.stack([arr, arr], axis=0)
            out_orientation = "mono"
        else:
            mono = False
            if arr.shape[0] == 2 and arr.shape[1] > 2:
                x = arr
                out_orientation = "channels_first"
            elif arr.shape[1] == 2 and arr.shape[0] > 2:
                x = arr.T
                out_orientation = "samples_first"
            else:
                x = np.stack([arr[:, 0], arr[:, 0]], axis=0)
                out_orientation = "channels_first"

        plugin = self._get_plugin()
        if plugin is None or not getattr(plugin, "available", False):
            logger.warning(
                "VinylNetDeCrackle: Modell nicht ladbar -> bit-identischer Passthrough (§V6, copilot-instructions.md)"
            )
            return PhaseResult(
                audio=arr.astype(np.float32).copy(),
                warnings=["modell_nicht_ladbar"],
                success=True,
                _skip_soft_clip=True,
            )

        st = float(strength if strength is not None else self.strength)
        out, meta = plugin.process(x, sample_rate=int(sample_rate), strength=st)

        # §0a (copilot-instructions.md): NaN/Inf-Schutz (Plugin genullt bereits; hier zweiter Rand).
        est: np.ndarray = np.asarray(out, dtype=np.float32)
        if not np.all(np.isfinite(est)):
            logger.warning(
                "VinylNetDeCrackle: %d nicht-finite Schätzwerte -> genullt (§0a/§V6, copilot-instructions.md)",
                int((~np.isfinite(est)).sum()),
            )
            est = np.nan_to_num(est, nan=0.0, posinf=0.0, neginf=0.0)

        if mono or out_orientation == "mono":
            out_arr: np.ndarray = est[0]
        elif out_orientation == "samples_first":
            out_arr = est.T
        else:
            out_arr = est
        return PhaseResult(
            audio=np.clip(np.asarray(out_arr, dtype=np.float32), -1.0, 1.0).astype(np.float32),
            ml_used=True,
            metadata={
                "strength": st,
                "mask_samples": int(meta.get("mask_samples", 0)),
                "model_device": meta.get("device", "cpu"),
                "passthrough": bool(meta.get("passthrough", False)),
            },
        )


# ---------------------------------------------------------------------------
# Kanonischer Zugang (Muster wie andere Phasen)
# ---------------------------------------------------------------------------
def get_phase(**kwargs: Any) -> VinylNetDeCracklePhase:
    """Factory fuer die Pipeline-Registrierung."""
    return VinylNetDeCracklePhase(**kwargs)
