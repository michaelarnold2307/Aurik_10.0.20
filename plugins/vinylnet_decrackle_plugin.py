"""VinylNetDeCracklePlugin — chirurgische ML-De-Crackle-Inferenz (VinylNet v4).

Lädt die trainierten Gewichte aus models/de_crackle/ mit **Preset-Priorität** für den
Checkpoint-Ladepfad: `vinylnet_base_best.pt` → `vinylnet_small_best.pt` →
`runs/vinylnet/best_infer.pt` (erster vorhandener Kandidat gewinnt; Architektur kommt aus dem
Checkpoint-Config, Preset-Namen sind nur Deploy-Konvention). Detektor: `detector_v4.pt` →
`runs/detector/detector.pt`. Torch primär; die ONNX-Präsenz wird gemeldet, der CPU-Fallback ist
erst nach Paritätsnachweis aktiv (§III.9 copilot-instructions.md). Danach führt das Plugin die
komplette Pipeline aus: Klickdetektion (ClickNet v4 mit M/S-Eingabe) -> Hybrid-Reparatur
(InpaintNet für grosse Lücken) -> VinylNet-Band-Split-Transformer (RoPE + Band-Attention,
ThumpNet, Refiner).

Abgrenzung zu phase_67 (crackle_texture_removal): Diese Plugin behandelt die
**chirurgische** Entfernung einzelner Klicks/Pops/Thumps und dichter Knistern-Impulse;
phase_67 entfernt kontinuierliche Knistern-Textur (Route A).

Invarianten:
    - Deterministisch (§G5 copilot-instructions.md): eval() + inference_mode, keine RNG im Forward
    - Stereo-Layout-Invariante: intern channels-first (C, N)
    - NaN/Inf-Schutz (§0a copilot-instructions.md) mit Warnung (§V6 copilot-instructions.md)
    - Modell nicht ladbar -> None (Phase liefert bit-identischen Passthrough + Warning)
"""

from __future__ import annotations

import logging
import math
import sys
import threading
from pathlib import Path
from typing import Any, cast

import numpy as np

logger = logging.getLogger(__name__)

_SR: int = 44_100
_WEIGHTS_DIR: str = "models/de_crackle"

# Checkpoint-Ladepfade mit Preset-Priorität (deployte Kanon-Namen zuerst, danach die
# Trainings-Artefakte). Der erste vorhandene Kandidat gewinnt; das im Checkpoint gespeicherte
# Config bestimmt die Architektur (§G9 copilot-instructions.md: eine Quelle = Artefakt selbst).
_VINYLNET_CANDIDATES: tuple[str, ...] = (
    "vinylnet_base_best.pt",
    "vinylnet_small_best.pt",
    "runs/vinylnet/best_infer.pt",
)
_DETECTOR_CANDIDATES: tuple[str, ...] = (
    "detector_v4.pt",
    "runs/detector/detector.pt",
)


def _dc_dir() -> Path:
    return Path(__file__).resolve().parent.parent / _WEIGHTS_DIR


def _first_existing(d: Path, candidates: tuple[str, ...]) -> Path | None:
    """Erster vorhandener Kandidat in der Prioritätsliste (None = alle fehlen)."""
    for c in candidates:
        p = d / c
        if p.exists():
            return p
    return None


def _preset_name(model_mod: Any, cfg: dict[str, Any]) -> str:
    """Preset-Name aus dem Checkpoint-Config (nur für Log/Transparenz)."""
    for name, ref in getattr(model_mod, "PRESETS", {}).items():
        if all(cfg.get(k) == v for k, v in ref.items()):
            return str(name)
    return "unbekannt"


class VinylNetDeCracklePlugin:
    """Lazy-Loader + Inferenzwrapper um models/de_crackle (VinylNet v4)."""

    def __init__(self) -> None:
        self._model: Any = None  # VinylNet (torch.nn.Module, eval())
        self._det: Any = None  # ClickNet (oder None)
        self._device: str = "cpu"
        self._load_attempted = False

    # ------------------------------------------------------------------ Laden
    def _import_dc(self, name: str) -> Any:
        """Lädt ein Modul aus models/de_crackle mit eindeutigem Namen (sys.path-Shim)."""
        import importlib.util

        qual = f"aurik_dc_{name}"
        if qual in sys.modules:  # Cache: keine doppelte Ausführung der Modul-Imports
            return sys.modules[qual]
        d = _dc_dir()
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))  # intern `from dsp/model/...` muss auflösbar bleiben
        spec = importlib.util.spec_from_file_location(qual, d / f"{name}.py")
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[qual] = mod
        spec.loader.exec_module(mod)
        return mod

    def _load(self) -> bool:
        if self._load_attempted:
            return self._model is not None
        self._load_attempted = True
        d = _dc_dir()
        try:
            import torch

            from backend.core.ml_device_manager import get_torch_device

            self._device = get_torch_device("VinylNet")
            model_mod = self._import_dc("model")
            # Bringt dsp/simulate/datenpfad nach sys.path; der Modulkopf wird in process()
            # über sys.modules["aurik_dc_infer"] referenziert (keine lokale Variable nötig).
            self._import_dc("infer")

            vinylnet_path = _first_existing(d, _VINYLNET_CANDIDATES)
            if vinylnet_path is None:
                logger.warning(
                    "VinylNetDeCrackle: kein Checkpoint gefunden (Preset-Priorität: %s)",
                    ", ".join(_VINYLNET_CANDIDATES),
                )
                return False
            ck = torch.load(vinylnet_path, map_location=self._device)
            preset = _preset_name(model_mod, dict(ck["config"]))
            self._model = model_mod.VinylNet(**ck["config"]).to(self._device)
            self._model.load_state_dict(ck.get("ema", ck.get("model")), strict=True)
            self._model.eval()

            det_path = _first_existing(d, _DETECTOR_CANDIDATES)
            if det_path is not None:
                dck = torch.load(det_path, map_location=self._device)
                cfg = dict(dck["config"])
                self._det = model_mod.ClickNet(**cfg).to(self._device)
                self._det.load_state_dict(dck["model"], strict=True)
                self._det.eval()
            else:  # §V6 (copilot-instructions.md): gelernter Detektor fehlt -> AR-Fallback in get_mask
                logger.warning(
                    "VinylNetDeCrackle: kein Detektor-Checkpoint (%s) -> klassischer AR-Detektor als Fallback (§V6, copilot-instructions.md)",
                    ", ".join(_DETECTOR_CANDIDATES),
                )

            # ONNX-Präsenz merken (nur Torch-Pfad aktiv; §III.9 Paritätsnachweis separat)
            self._onnx_present = bool(list(d.glob("*.onnx")))
            logger.info(
                "VinylNetDeCrackle: geladen (%s, Preset %s aus %s, Detektor %s, ONNX-Artefakt %s)",
                self._device,
                preset,
                vinylnet_path.name,
                det_path.name if det_path is not None else "nein",
                "ja" if getattr(self, "_onnx_present", False) else "nein",
            )
            return True
        except Exception as exc:  # pragma: no cover - Fallback-Pfad
            logger.warning("VinylNetDeCrackle: Torch-Ladefehler -> Passthrough (§V6, copilot-instructions.md): %s", exc)
            self._model = None
            return False

    @property
    def available(self) -> bool:
        return self._load() and self._model is not None

    # ------------------------------------------------------------- Verarbeitung
    def process(
        self,
        audio: np.ndarray,
        sample_rate: int = _SR,
        strength: float = 0.9,
        det_threshold: float = 0.5,
        link_channels: bool = False,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """audio: (C,T) channels-first bei arbitrary SR -> (out (C,T), meta).

        Bei SR != 44100 wird deterministisch hin/zurück resampelt (Modell ist auf 44.1 kHz trainiert)."""
        x = np.asarray(audio, dtype=np.float32)
        mono = x.ndim == 1
        if mono:
            x = np.stack([x, x], axis=0)
        meta: dict[str, Any] = {"strength": float(strength), "device": self._device}

        if not self.available:
            return (x[0] if mono else x).copy(), {**meta, "passthrough": True}

        infer_mod = sys.modules["aurik_dc_infer"]
        work = x
        resampled = False
        if sample_rate != _SR:  # deterministisches Polyphase-Resampling (§G5, copilot-instructions.md)
            g = math.gcd(_SR, int(sample_rate))
            from scipy.signal import resample_poly

            work = resample_poly(work, _SR // g, int(sample_rate) // g, axis=1).astype(np.float32)
            resampled = True

        mask = infer_mod.get_mask(
            work,
            self._det if self._det is not None else None,
            self._device,
            "auto",
            det_threshold,
            6.0,
            grow=2,
            link=link_channels,
        )
        meta["mask_samples"] = int(mask.sum())
        out = infer_mod.restore(work, cast(Any, self._model), self._device, float(strength), mask)

        if not np.all(np.isfinite(out)):  # §0a (copilot-instructions.md)
            n_bad = int((~np.isfinite(out)).sum())
            logger.warning(
                "VinylNetDeCrackle: %d nicht-finite Samples -> genullt (§0a/§V6, copilot-instructions.md)", n_bad
            )
            out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

        if resampled:
            g = math.gcd(_SR, int(sample_rate))
            from scipy.signal import resample_poly

            out = resample_poly(out, int(sample_rate) // g, _SR // g, axis=1).astype(np.float32)
        out = np.clip(out, -1.0, 1.0).astype(np.float32)
        return (out[0] if mono else out), meta


_instance: VinylNetDeCracklePlugin | None = None
_lock = threading.Lock()


def get_vinylnet_decrackle_plugin() -> VinylNetDeCracklePlugin:
    """Thread-sicherer Singleton (Double-Checked Locking)."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = VinylNetDeCracklePlugin()
    return _instance
