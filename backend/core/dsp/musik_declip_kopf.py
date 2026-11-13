"""Musik-Declip-Kopf — Challenger für Phase 07 (D-K4-1-Durchbruch-Kandidat).

Waveform→Waveform-Declipper, trainiert auf MUSDB18-HQ-Mischungen +
k398-Clean-Kulturen (Report `docs/reports/current/2026-11-13_musik_declip_
durchbruch.md`; leak-freier LOO-Beweis: 3/3 k398-Titel SDR-verbessert,
Boogie −9,99 → +5,60 dB).

Sicherheits-Vertrag (Musikschutz, AGENTS.md Guard-Kalibrierung):
1. **Clip-Masken-Blend** — die Kopf-Ausgabe ersetzt den Eingang NUR in
   geclippten Regionen (|x| ≥ Schwelle) mit kurzen Überblendungen; saubere
   Abschnitte bleiben bit-identisch. Ohne diese Maske würde der Kopf den
   Rezeptur-Gain rückgängig machen und sauberes Material dämpfen.
2. **Never-worsen-Witness** — der Kandidat wird nur zurückgegeben, wenn der
   Harmonik-Proxy strikt unter dem Eingang liegt; sonst None (Ersatzpfad).
3. Deterministisch (§G5 (copilot-instructions.md)), CPU-only, fail-closed
   (§V6 (copilot-instructions.md)).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

MODELL_PFAD = (
    Path(__file__).resolve().parent.parent.parent.parent / "models" / "klassen_koepfe" / "musik_declip" / "best.pt"
)
WINDOW = 8192
HOP = 4096
_TARGET_SR = 48000

_singleton: dict = {}


def _lade_modell() -> Any | None:
    """Lazy-Cache des Torch-Checkpoints (deterministisch, CPU)."""
    if "model" in _singleton:
        return _singleton["model"]
    if not MODELL_PFAD.exists():
        logger.warning("Musik-Declip-Kopf nicht gefunden: %s — Ersatzpfad.", MODELL_PFAD)
        _singleton["model"] = None
        return None
    try:
        import torch
        from torch import nn

        # Architektur identisch zu scripts/train_klassen_kopf.py::build_model
        # (identity-initialisierter Residual-Kopf, ~60k Parameter) — bewusst
        # inline, damit backend/ nichts aus scripts/ importiert.
        class Kopf(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.enc1 = nn.Conv1d(1, 16, 15, stride=2, padding=7)
                self.enc2 = nn.Conv1d(16, 32, 15, stride=2, padding=7)
                self.enc3 = nn.Conv1d(32, 64, 15, stride=2, padding=7)
                self.dec3 = nn.ConvTranspose1d(64, 32, 15, stride=2, padding=7, output_padding=1)
                self.dec2 = nn.ConvTranspose1d(64, 16, 15, stride=2, padding=7, output_padding=1)
                self.dec1 = nn.ConvTranspose1d(32, 1, 15, stride=2, padding=7, output_padding=1)
                self.act = nn.LeakyReLU(0.2)
                if self.dec1.weight is not None:
                    nn.init.zeros_(self.dec1.weight)
                if self.dec1.bias is not None:
                    nn.init.zeros_(self.dec1.bias)

            def forward(self, x):  # type: ignore[no-untyped-def]
                e1 = self.act(self.enc1(x))
                e2 = self.act(self.enc2(e1))
                e3 = self.act(self.enc3(e2))
                d3 = self.act(self.dec3(e3))
                d3 = torch.cat([d3, e2[:, :, : d3.shape[-1]]], dim=1)
                d2 = self.act(self.dec2(d3))
                d2 = torch.cat([d2, e1[:, :, : d2.shape[-1]]], dim=1)
                d1 = self.dec1(d2)
                return x + d1[:, :, : x.shape[-1]]

        model = Kopf()
        state = torch.load(MODELL_PFAD, map_location="cpu")["model"]
        model.load_state_dict(state)
        model.eval()
        _singleton["model"] = model
        return model
    except Exception as exc:  # §V6-Warnung, Ersatzpfad
        logger.warning("Musik-Declip-Kopf nicht ladbar (%s) — Ersatzpfad.", exc)
        _singleton["model"] = None
        return None


def _kopf_ola(model: Any, x: np.ndarray) -> np.ndarray:
    """Overlap-Add über 8192-Fenster mit Rand-Reflection-Padding."""
    import torch

    xf = np.asarray(x, dtype=np.float32)
    win = np.hanning(WINDOW).astype(np.float32)
    pad = np.pad(xf, (HOP, HOP), mode="reflect")
    out = np.zeros(len(pad), dtype=np.float64)
    norm = np.zeros(len(pad), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(pad) - WINDOW + 1, HOP):
            seg = pad[start : start + WINDOW]
            t = torch.from_numpy(seg).unsqueeze(0).unsqueeze(0)
            rec = model(t).cpu().numpy()[0, 0, :].astype(np.float64)
            out[start : start + WINDOW] += rec * win
            norm[start : start + WINDOW] += win * win
    out = out / np.maximum(norm, 1e-6)
    aus: np.ndarray = out[HOP : HOP + len(xf)]
    return aus


def declip_musik_kopf(audio: np.ndarray, sample_rate: int, clip_threshold: float) -> np.ndarray | None:
    """Neuronaler Declip mit Masken-Blend + Witness; None = Ersatzpfad.

    Args:
        audio: Mono-Waveform (N,) float32.
        sample_rate: Abtastrate (48 kHz erwartet; sonst Ersatzpfad).
        clip_threshold: Detektionsschwelle für den Masken-Blend.

    Returns:
        Repariertes Mono-Signal (N,) float32 — oder None.
    """
    from scipy.signal import resample_poly  # pylint: disable=import-outside-toplevel

    from backend.core.dsp.aspade_declipper import harmonic_distortion_witness  # pylint: disable=import-outside-toplevel

    model = _lade_modell()
    if model is None:
        return None
    if int(sample_rate) <= 0:
        logger.warning("Musik-Declip-Kopf: sr=%d ungültig — Ersatzpfad.", sample_rate)
        return None
    try:
        x = np.asarray(audio, dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        sr = int(sample_rate)
        if sr != _TARGET_SR:
            g = int(np.gcd(sr, _TARGET_SR))
            x = resample_poly(x.astype(np.float64), _TARGET_SR // g, sr // g).astype(np.float32)
        theta = float(np.clip(clip_threshold, 0.05, 0.999))
        kopf_out = _kopf_ola(model, x)
        # Masken-Blend: Kopf nur in geclippten Regionen + kurze Überblendung.
        maske = np.abs(x) >= theta
        fade = int(0.005 * _TARGET_SR)
        win = 0.5 * (1 - np.cos(np.pi * np.arange(fade) / max(fade, 1)))
        blend = np.zeros(len(x), dtype=np.float32)
        kanten = np.diff(maske.astype(np.int8))
        for st in np.where(kanten == 1)[0]:
            a0 = max(0, st - fade)
            blend[a0:st] = np.maximum(blend[a0:st], win[: st - a0])
        for en in np.where(kanten == -1)[0]:
            b1 = min(len(x), en + fade)
            blend[en:b1] = np.maximum(blend[en:b1], win[::-1][: b1 - en])
        blend[maske] = 1.0
        kandidat = blend * kopf_out + (1.0 - blend) * x
        # Witness: nur bei striktem Harmonik-Proxy-Gewinn akzeptieren.
        w_in = harmonic_distortion_witness(x, _TARGET_SR)
        w_out = harmonic_distortion_witness(kandidat, _TARGET_SR)
        if w_out >= w_in:
            logger.info(
                "Musik-Declip-Kopf: Kandidat abgelehnt (Witness %.6f ≥ %.6f) — Eingang bleibt",
                w_out,
                w_in,
            )
            return None
        if sr != _TARGET_SR:
            g = int(np.gcd(sr, _TARGET_SR))
            kandidat = resample_poly(kandidat.astype(np.float64), sr // g, _TARGET_SR // g)[: len(audio)]
        aus: np.ndarray = np.clip(kandidat, -1.0, 1.0).astype(np.float32)
        return aus
    except Exception as exc:
        logger.warning("Musik-Declip-Kopf fehlgeschlagen (%s) — Ersatzpfad.", exc)
        return None
