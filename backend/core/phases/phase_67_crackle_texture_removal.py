"""Phase 67: Crackle Texture Removal (ML, Route A) — Knistern-Textur-Entfernung.

Werkzeug fuer DefectType.CRACKLE (kontinuierliches Vinyl-Knistern, Bailey et al.
2019 AES 147th Conv.), Spec 03 (03_cognitive_modules.md) Denker-Entscheidungskette:
Der DefectPhaseMapper routet CRACKLE-Textur auf diese Phase, waehrend CLICKS auf
phase_27 und spaerliches Knistern auf phase_09 laeuft (Either-Or, Spec 06).

Modell: models/crackle_texture_removal/crackle_texture_removal_base.pt (Torch-ROCm
primaer, ONNX-CPU-Fallback crackle_texture_removal_base.onnx — Paritaet 1,67e-06
nachgewiesen, §III.9 (copilot-instructions.md)). Abgrenzung: kein Bezug zu
"harmonic_inpainting" (Oberton-Rekonstruktion) oder phase_55 (Diffusion-Inpainting);
diese Phase ENTFERNT Knistern-Textur, sie rekonstruiert nichts.

Pflicht-Invarianten (Nacht-Befund, Ohr-validiert):
- Kontext-Padding (Spiegelung, 1024 Samples) je Fenster: Der Decoder-Rand-Transient
  faellt in den verworfenen Rand — verhindert injizierte Artefakte am Fensteranfang
  (Ohr-Befund "kratzig", Roxy-Fall).
- Crossfade-Overlap-Add (0,25 s Raised-Cosine): keine periodischen Fenster-Thumps.
- Deterministisch (§G5, copilot-instructions.md): eval() + no_grad, keine Seeds noetig.
- Passthrough-Semantik (§0a): Modell nicht ladbar -> bit-identischer Passthrough
  mit Warning (§V6, copilot-instructions.md: kein stiller Fallback).
"""

from __future__ import annotations

import logging
import os
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

_SR = 48000
_WIN = 288000  # 6 s
_HOP = _WIN - 12000  # 0,25 s Crossfade-Overlap
_PAD = 1024  # Kontext-Padding (Spiegelung)
_BASE_PT = "models/crackle_texture_removal/crackle_texture_removal_base.pt"
_BASE_ONNX = "models/crackle_texture_removal/crackle_texture_removal_base.onnx"


class _CrackleNet:
    """Architektur identisch zum Trainingslauf (Route A, 300k Parameter).

    Muss bit-kompatibel zu den trainierten Gewichten bleiben: H=96, B=192, X=4.
    Wird als echtes torch.nn.Module aufgebaut (torch wird injiziert), damit
    load_state_dict(strict=True) und .to(device) funktionieren.
    """

    def __init__(self, torch: Any) -> None:
        nn = torch.nn
        self._torch = torch
        H, B, X = 96, 192, 4
        self.enc = nn.Conv1d(2, H, 16, stride=8)
        blocks = []
        d = 1
        for _ in range(X):
            blocks.append(
                nn.Sequential(
                    nn.Conv1d(H, B, 3, padding=d, dilation=d),
                    nn.GELU(),
                    nn.Conv1d(B, H, 1),
                )
            )
            d *= 2
        self.blocks = nn.ModuleList(blocks)
        self.dec = nn.ConvTranspose1d(H, 2, 16, stride=8)
        nn.init.zeros_(self.dec.weight)
        nn.init.zeros_(self.dec.bias)
        self._module = nn.Module()
        self._module.enc = self.enc
        self._module.blocks = self.blocks
        self._module.dec = self.dec

    def to(self, device: str) -> _CrackleNet:
        self._module = self._module.to(device)
        self.enc = self._module.enc
        self.blocks = self._module.blocks
        self.dec = self._module.dec
        return self

    def eval(self) -> None:
        self._module.eval()

    def load_state_dict(self, state: Any, strict: bool = True) -> None:
        self._module.load_state_dict(state, strict=strict)

    def __call__(self, x: Any) -> Any:
        torch = self._torch
        z = self.enc(x)
        for blk in self.blocks:
            z = z + blk(z)
        y = self.dec(z)
        if y.shape[-1] > x.shape[-1]:
            y = y[..., : x.shape[-1]]
        elif y.shape[-1] < x.shape[-1]:
            y = torch.nn.functional.pad(y, (0, x.shape[-1] - y.shape[-1]))
        return y


class CrackleTextureRemovalPhase(PhaseInterface):
    """Knistern-Textur-Entfernung nach Spec 03/06 — einziger Eingriffspunkt ist der Mapper."""

    PHASE_ID = "phase_67_crackle_texture_removal"

    def __init__(self, strength: float = 2.0) -> None:
        self.strength = float(strength)
        self._net: Any = None
        self._device = "cpu"
        self._onnx_sess: Any = None
        self._load_attempted = False

    def get_metadata(self) -> PhaseMetadata:
        return PhaseMetadata(
            phase_id=self.PHASE_ID,
            name="Crackle Texture Removal (ML, Route A)",
            category=PhaseCategory.DEFECT_REMOVAL,
            priority=6,
            version="1.0.0",
            is_cpu_intensive=True,
            memory_requirement_mb=128,
            quality_impact=0.9,
            description=(
                "Knistern-Textur-Entfernung fuer DefectType.CRACKLE via ML "
                "(Bailey et al. 2019-Klasse). Either-Or-Routing: CLICKS->phase_27, "
                "spaerlich->phase_09, Textur->diese Phase (Spec 03)."
            ),
            defect_types=["crackle"],
            phase_mode=PhaseMode.RESTORATION_ONLY,
        )

    # ------------------------------------------------------------------ Laden
    def _load_model(self) -> bool:
        """Laedt Torch-ROCm primaer, ONNX-CPU-Fallback (§III.9). Deterministisch."""
        if self._load_attempted:
            return self._net is not None or self._onnx_sess is not None
        self._load_attempted = True
        pt_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", _BASE_PT)
        onnx_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", _BASE_ONNX)
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            if os.path.exists(pt_path):
                net = _CrackleNet(torch)
                state = torch.load(pt_path, map_location=device, weights_only=True)
                net.load_state_dict(state, strict=True)
                net = net.to(device)
                net.eval()
                self._net = net
                self._device = device
                logger.info("CrackleTextureRemoval: Torch-Modell geladen (%s)", device)
                return True
        except Exception as exc:  # pragma: no cover - Fallback-Pfad
            logger.warning("CrackleTextureRemoval: Torch-Pfad nicht verfuegbar (%s)", exc)
        try:
            if os.path.exists(onnx_path):
                import onnxruntime as ort

                self._onnx_sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
                logger.info("CrackleTextureRemoval: ONNX-CPU-Ersatzpfad geladen")
                return True
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "CrackleTextureRemoval: Modell nicht ladbar (%s) -> Passthrough (§V6, copilot-instructions.md)", exc
            )
        return False

    # ------------------------------------------------------------- Verarbeitung
    def process(
        self,
        audio: np.ndarray,
        sample_rate: int = _SR,
        material_type: Any = None,
        **kwargs: Any,
    ) -> PhaseResult:
        if sample_rate != _SR:
            logger.warning(
                "CrackleTextureRemoval: %d Hz statt 48000 Hz -> Passthrough (§V6, copilot-instructions.md)", sample_rate
            )
            return PhaseResult(audio=audio.astype(np.float32), warnings=["sample_rate"], _skip_soft_clip=True)
        if not self._load_model():
            return PhaseResult(
                audio=audio.astype(np.float32),
                warnings=["modell_nicht_ladbar"],
                success=True,
                _skip_soft_clip=True,
            )
        # Stereo-Invariante: beide Layouts bedienen, intern channels-first (C, N).
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
        est = self._estimate(x)
        # §0a (copilot-instructions.md): NaN/Inf-Schutz — nicht-finite Schätzwerte
        # werden genullt (Passthrough für dieses Sample) und protokolliert (§V6 (copilot-instructions.md)).
        if not np.all(np.isfinite(est)):
            logger.warning(
                "CrackleTextureRemoval: %d nicht-finite Schätzwerte -> genullt (§0a/§V6, copilot-instructions.md)",
                int((~np.isfinite(est)).sum()),
            )
            est = np.nan_to_num(est, nan=0.0, posinf=0.0, neginf=0.0)
        out = np.clip(x - self.strength * est, -1.0, 1.0)
        removed_rms = float(np.sqrt(np.mean((x - out) ** 2)))
        if mono or out_orientation == "mono":
            out_arr = out[0]
        elif out_orientation == "samples_first":
            out_arr = out.T
        else:
            out_arr = out
        return PhaseResult(
            audio=np.clip(
                np.nan_to_num(np.asarray(out_arr, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0),
                -1.0,
                1.0,
            ),
            ml_used=True,
            metadata={
                "strength": self.strength,
                "removed_rms": removed_rms,
                "model_device": self._device,
            },
        )

    def _estimate(self, x: np.ndarray) -> np.ndarray:
        """Knistern-Schaetzung: Fensterweise mit Kontext-Padding + Crossfade."""
        n = x.shape[1]
        est = np.zeros_like(x)
        wsum = np.zeros((1, n), dtype=np.float32)
        ramp = np.ones(_WIN, dtype=np.float32)
        ov = _WIN - _HOP
        ramp[:ov] = np.linspace(0, 1, ov, endpoint=False).astype(np.float32)
        ramp[-ov:] = np.linspace(1, 0, ov, endpoint=False).astype(np.float32)
        for start in range(0, n, _HOP):
            seg = np.zeros((2, _WIN), dtype=np.float32)
            nn = min(_WIN, n - start)
            seg[:, :nn] = x[:, start : start + nn]
            c = self._window_estimate(seg)
            end = min(start + _WIN, n)
            est[:, start:end] += c[:, : end - start] * ramp[: end - start]
            wsum[:, start:end] += ramp[: end - start]
        _est_result: np.ndarray = np.asarray(est / np.maximum(wsum, 1e-9), dtype=np.float32)
        return _est_result

    def _window_estimate(self, seg: np.ndarray) -> np.ndarray:
        """Ein Fenster mit Spiegelungs-Padding: Rand-Transient faellt in den verworfenen Rand.

        Torch-Pfad: Modell laeuft auf dem gepolsterten Fenster (dynamische Laenge),
        Ausgang wird auf die zentrale Fensterregion beschnitten.
        ONNX-Pfad (fixe Eingangsform 288000): zentrale 288000-Samples des gepolsterten
        Fensters, Ausgangsrand (PAD//2) genullt — der Crossfade gewichtet diese
        Randbereiche ohnehin gegen null.
        """
        pad = np.zeros((2, _WIN + 2 * _PAD), dtype=np.float32)
        pad[:, _PAD : _PAD + _WIN] = seg
        pad[:, :_PAD] = seg[:, _PAD - 1 :: -1] if _PAD > 1 else seg[:, :1]
        pad[:, _PAD + _WIN :] = seg[:, -1 : -_PAD - 1 : -1] if _PAD > 1 else seg[:, -1:]
        rms = float(np.sqrt(np.mean(pad**2)))
        g = 0.16 / max(rms, 1e-6)
        if self._net is not None:
            import torch

            with torch.no_grad():
                t = torch.from_numpy((pad * g)[None]).to(self._device)
                c = cast(np.ndarray, self._net(t).cpu().numpy()[0])
            _c_out: np.ndarray = np.asarray((c / g)[:, _PAD : _PAD + _WIN], dtype=np.float32)
            return _c_out
        half = _PAD // 2
        inp = pad[:, half : half + _WIN][None].astype(np.float32) * g
        c = cast(np.ndarray, self._onnx_sess.run(None, {"audio": inp})[0][0])
        c = (c / g).astype(np.float32)
        c[:, :half] = 0.0
        c[:, -half:] = 0.0
        _c_out_onnx: np.ndarray = np.asarray(c, dtype=np.float32)
        return _c_out_onnx


# ---------------------------------------------------------------------------
# Kanonischer Zugang (Muster wie andere Phasen)
# ---------------------------------------------------------------------------
def get_phase(**kwargs: Any) -> CrackleTextureRemovalPhase:
    """Factory fuer die Pipeline-Registrierung."""
    return CrackleTextureRemovalPhase(**kwargs)
