"""§SOTA-VOCAL-INPAINT-S3: DiffWave-Torch-Inpaint für Gesangslücken (Runtime).

Ergänzt den ONNX-Pfad des Plugins um einen reinen PyTorch-Pfad, der den
FINEGETUNTEN Vokal-Checkpoint lädt (models/diffwave/diffwave_vocal_ft.ckpt —
Aktivierung erst nach dem F1-Gate, siehe Roadmap S2). Muster: der
BSR-Torch-ROCm-Pfad (bsr317_torch_rocm.py) — Checkpoint direkt, kein ONNX.

Kette je Lücke:
  channel → Resample 22,05 kHz → Fenster (16368, zentriert, geclippt/gepadet)
  → Lücke nullen → Mel des GAPPTEN Fensters (Checkpoint-Konvention T=65)
  → DDIM 50 Schritte mit input-abgeleitetem Seed (blake2b — §G5 (GEBOTE.md):
  gleicher Input + Version ⇒ bit-identisch) → Gap-Region zurück in die
  Original-Sample-Rate.

Never-worsen/Absicherung:
  - `diffwave_vocal_ready()`: False solange der Finetune-Checkpoint fehlt —
    phase_55 behält dann die bisherige Drosselung (Status quo).
  - Jeder Fehler ⇒ None + warning (§V6 (copilot-instructions.md)) — die
    Kaskade fällt auf DSP/NMF zurück.
  - Die IN-V1/V2-Naht-Gates laufen generisch NACH der Kaskade (phase_55).
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import threading
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]  # backend/core/dsp/<file> → Repo-Root
_CKPT_FINETUNED = _PROJECT_ROOT / "models" / "diffwave" / "diffwave_vocal_ft.ckpt"
_CKPT_BASE = _PROJECT_ROOT / "models" / "diffwave" / "diffwave.ckpt"
_MODEL_SR = 22050
_WIN = 16368
_VAL_STEPS = 50

_lock = threading.Lock()
_model: object | None = None
_model_ready: bool | None = None
_ready_cache: bool | None = None


def diffwave_vocal_ready() -> bool:
    """True, sobald der Finetune-Checkpoint existiert (S3-Aktivierungsvertrag)."""
    global _ready_cache  # pylint: disable=global-statement
    if _ready_cache is None:
        _ready_cache = _CKPT_FINETUNED.is_file()
    return _ready_cache


def reset_ready_cache() -> None:
    """Test-Hook: Cache zurücksetzen (z. B. nach Checkpoint-Anlage)."""
    global _ready_cache  # pylint: disable=global-statement
    _ready_cache = None


def _device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _load_model() -> object | None:
    """Lazy-Singleton: Finetune-Checkpoint bevorzugt, sonst Basis-Checkpoint."""
    global _model, _model_ready  # pylint: disable=global-statement
    if _model is not None:
        return _model
    # §G174 (GEBOTE.md): Imports VOR dem Lock auflösen.
    import torch

    from backend.core.dsp.diffwave_model import DiffWave
    from backend.core.plugin_lifecycle_manager import register_plugin as _reg_plm

    with _lock:
        if _model is not None:
            return _model
        if _model_ready is True:  # bereits erfolglos versucht
            return None
        try:
            ckpt = _CKPT_FINETUNED if _CKPT_FINETUNED.is_file() else _CKPT_BASE
            if not ckpt.is_file():
                _model_ready = True
                logger.warning("DiffWave-Torch: kein Checkpoint (%s) — Kaskade nutzt den DSP-Ersatzpfad.", ckpt)
                return None
            state = torch.load(str(ckpt), map_location="cpu", weights_only=True)
            model = DiffWave()
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing or unexpected:
                _model_ready = True
                logger.warning(
                    "DiffWave-Torch: Checkpoint %s passt nicht (missing=%d, unexpected=%d) — DSP-Fallback.",
                    ckpt.name,
                    len(missing),
                    len(unexpected),
                )
                return None
            model.to(_device())
            model.eval()
            try:
                _reg_plm("DiffWaveTorch", size_gb=0.05, unload_fn=_unload)
            except Exception as _plm_exc:  # pragma: no cover
                logger.debug("DiffWave-Torch: PLM-Registrierung nicht möglich (unkritisch): %s", _plm_exc)
            _model = model
            logger.info("DiffWave-Torch geladen: %s (%s)", ckpt.name, _device())
            return _model
        except Exception as _exc:
            _model_ready = True
            logger.warning("DiffWave-Torch nicht ladbar (%s) — DSP-Ersatzpfad (§V6 (copilot-instructions.md)).", _exc)
            return None


def _unload() -> None:
    global _model  # pylint: disable=global-statement
    with _lock:
        _model = None


def _seed_for(channel: np.ndarray, gap_start: int, gap_end: int) -> int:
    """Input-abgeleiteter Seed (blake2b) — deterministisch je Input+Version (§G5 (GEBOTE.md))."""
    h = hashlib.blake2b(digest_size=8)
    h.update(np.ascontiguousarray(channel, dtype=np.float32).tobytes())
    h.update(int(gap_start).to_bytes(8, "little"))
    h.update(int(gap_end).to_bytes(8, "little"))
    return int.from_bytes(h.digest(), "little") % (2**31)


@np.errstate(all="ignore")
def diffwave_inpaint_gap(
    channel: np.ndarray,
    gap_start: int,
    gap_end: int,
    sr: int,
    n_steps: int = _VAL_STEPS,
) -> np.ndarray | None:
    """Füllt EINE Lücke (Gap-Segment, Länge gap_end−gap_start) oder None.

    Args:
        channel: Mono float32, Lücke unverändert enthalten.
        gap_start/gap_end: Grenzen in Samples bei `sr`.
        sr: Sample-Rate des Kanals (wird intern auf 22,05 kHz resampled).

    Returns:
        Fill-Segment (float32, gleiche Länge wie die Lücke) oder None
        (§V6 (copilot-instructions.md)-Fallback).
    """
    model = _load_model()
    if model is None:
        return None
    try:
        import torch
        from scipy.signal import resample_poly

        from backend.core.dsp.diffwave_model import _ALPHA_BAR, _N_TIMESTEPS, mel_torch

        gap_len = int(gap_end) - int(gap_start)
        if gap_len <= 8:
            return None
        mono = np.nan_to_num(np.asarray(channel, dtype=np.float32).ravel(), nan=0.0, posinf=0.0, neginf=0.0)
        if sr != _MODEL_SR:
            g = int(np.gcd(sr, _MODEL_SR))
            mono22 = resample_poly(mono, _MODEL_SR // g, sr // g).astype(np.float32)
            gap_start22 = int(round(gap_start * _MODEL_SR / sr))
            gap_end22 = int(round(gap_end * _MODEL_SR / sr))
        else:
            mono22 = mono
            gap_start22, gap_end22 = int(gap_start), int(gap_end)
        center = (gap_start22 + gap_end22) // 2
        w0 = int(np.clip(center - _WIN // 2, 0, max(0, len(mono22) - _WIN)))
        window = np.zeros(_WIN, np.float32)
        seg = mono22[w0 : w0 + _WIN]
        window[: len(seg)] = seg
        gap_local = center - w0
        if gap_local < 0 or gap_local + (gap_end22 - gap_start22) > _WIN:
            return None
        gapped = window.copy()
        gapped[gap_local : gap_local + (gap_end22 - gap_start22)] = 0.0
        mel = mel_torch(gapped)
        dev = _device()
        mel_t = mel.to(dev)
        gen = torch.Generator(device=dev).manual_seed(_seed_for(mono, gap_start, gap_end))
        x = torch.randn(1, _WIN, device=dev, dtype=torch.float32, generator=gen) * 0.1
        stride = max(1, _N_TIMESTEPS // max(1, n_steps))
        steps = list(range(_N_TIMESTEPS, 0, -stride))[:n_steps]
        with torch.no_grad():
            for n in steps:
                step_t = torch.tensor([n - 1], dtype=torch.int64, device=dev)
                pred = model(x, mel_t, step_t)  # type: ignore[operator]
                a_bar = float(_ALPHA_BAR[n - 1])
                x0 = (x - math.sqrt(1.0 - a_bar) * pred) / math.sqrt(a_bar)
                nxt = n - stride
                if nxt <= 0:
                    x = x0
                else:
                    a_bar_nxt = float(_ALPHA_BAR[nxt - 1])
                    x = math.sqrt(a_bar_nxt) * x0  # z=0: deterministisch
        filled22 = x[0].detach().cpu().numpy().astype(np.float32)
        gap_fill22 = filled22[gap_local : gap_local + (gap_end22 - gap_start22)]
        if sr != _MODEL_SR:
            g2 = int(np.gcd(_MODEL_SR, sr))
            gap_fill = resample_poly(gap_fill22, sr // g2, _MODEL_SR // g2).astype(np.float32)
        else:
            gap_fill = gap_fill22
        n = gap_len
        if len(gap_fill) > n:
            gap_fill = gap_fill[:n]
        elif len(gap_fill) < n:
            gap_fill = np.pad(gap_fill, (0, n - len(gap_fill)), mode="edge")
        _gap_out: np.ndarray = np.clip(np.nan_to_num(gap_fill, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0).astype(
            np.float32
        )
        return _gap_out
    except Exception as _exc:
        logger.warning(
            "DiffWave-Torch-Inpaint fehlgeschlagen (%s) — DSP-Ersatzpfad (§V6 (copilot-instructions.md)).", _exc
        )
        return None
