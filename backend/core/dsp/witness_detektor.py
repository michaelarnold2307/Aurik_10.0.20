"""backend/core/dsp/witness_detektor.py — gelerntes Schadens-Witness je Q≈0-Klasse.

SOTA-Antwort auf die Metrik-Blindheit der §G13-Q≈0-Klassen (Handover
Punkte 25/29): die handgefertigten gain-invarianten Features sehen den
Schaden nicht (Q0-Witness-Forschung 2026-11-14, Δ ≈ 0), ein kleiner
CRNN-Detektor auf der Produktions-Mel (``compute_mel_produktion``, D-K3-96)
separiert clean/schaden dagegen leak-frei (Holdout-AUC 0,90–0,99 je Klasse,
Beleg ``models/witness_<klasse>/report.json``).

Dieses Modul ist die **Report-only-Inferenz**: P(schaden) in 0..1, höher =
mehr Schaden. Kein Gate, kein Flag, kein Sample-Eingriff — Zeuge, kein
Richter (§G191 (GEBOTE.md)). CPU-only per Konstruktion (kein CUDA-Aufruf).

Kanonische Checkpoints: ``models/witness_<klasse>/witness.pt`` (state_dict +
n_mels + frames + seed), trainiert mit
``scripts/train_witness_detektor.py`` (leak-frei, Seed 42, §G5 (copilot-instructions.md)).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = _ROOT / "models"
FRAMES = 64

# Kanonische Klassen→Checkpoint-Zuordnung. Die LITERALEN Pfadnamen sind
# zugleich der Verdrahtungs-Nachweis des Modell-Audits (§G9 (copilot-instructions.md)):
# jedes models/witness_<klasse>-Verzeichnis hat hier seinen statischen Konsumenten.
WITNESS_CHECKPOINT_PFADE: dict[str, str] = {
    "groove_echo": "models/witness_groove_echo/witness.pt",
    "sticky_shed_residue": "models/witness_sticky_shed_residue/witness.pt",
    "tape_head_level_dip": "models/witness_tape_head_level_dip/witness.pt",
    "sibilance": "models/witness_sibilance/witness.pt",
    "compression_artifacts": "models/witness_compression_artifacts/witness.pt",
    "transport_bump": "models/witness_transport_bump/witness.pt",
}

_cache: dict[str, Any] = {}


def _lade_model(klasse: str) -> Any | None:
    """Checkpoint-Lader mit Negativ-Cache (§V6 (copilot-instructions.md), kein stiller Rückfall)."""
    if klasse in _cache:
        return _cache[klasse]
    if klasse not in WITNESS_CHECKPOINT_PFADE:
        logger.warning("Unbekannte Witness-Klasse: %s — Kanal nicht verfügbar (§V6 (copilot-instructions.md))", klasse)
        _cache[klasse] = None
        return None
    import torch

    checkpoint = _ROOT / WITNESS_CHECKPOINT_PFADE[klasse]
    if not checkpoint.is_file():
        logger.warning(
            "Witness-Checkpoint fehlt: %s — Kanal nicht verfügbar (§V6 (copilot-instructions.md))", checkpoint
        )
        _cache[klasse] = None
        return None
    from scripts.train_witness_detektor import WitnessCRNN

    daten = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    model = WitnessCRNN(int(daten["n_mels"]))
    model.load_state_dict(daten["state_dict"])
    model.eval()
    _cache[klasse] = model
    return model


def witness_schaden_kanal(audio: Any, sample_rate: int, klasse: str) -> float:
    """P(schaden) je Klasse in 0..1 (report-only, höher = mehr Schaden).

    Gibt 0.0 zurück, wenn das Checkpoint fehlt oder die Eingabe zu kurz ist
    (§V6 (copilot-instructions.md)-Warnung beim Fehlen des Checkpoints).
    """
    import torch

    model = _lade_model(klasse)
    if model is None:
        return 0.0
    x = np.asarray(audio)
    if x.ndim == 2:
        if x.shape[0] <= 8 and x.shape[-1] > 8:
            x = x.T
        x = x.mean(axis=1)
    mono = np.asarray(x, dtype=np.float32).reshape(-1)
    if mono.size < 512:
        return 0.0
    from plugins.bigvgan_v2_plugin import compute_mel_produktion

    m = compute_mel_produktion(mono, int(sample_rate))
    if m.shape[1] >= FRAMES:
        m = m[:, :FRAMES]
    else:
        pad = np.full((m.shape[0], FRAMES - m.shape[1]), m.min(), dtype=np.float32)
        m = np.concatenate([m, pad], axis=1)
    x_t = torch.from_numpy(m.astype(np.float32)[None, None, :, :])
    with torch.no_grad():
        logit = model(x_t)
        p = float(torch.sigmoid(logit).item())
    return round(p, 4)


def witness_kanal_verfuegbar(klasse: str) -> bool:
    """Fail-closed-Prüfung, ob das Checkpoint der Klasse lokal liegt."""
    if klasse not in WITNESS_CHECKPOINT_PFADE:
        return False
    return (_ROOT / WITNESS_CHECKPOINT_PFADE[klasse]).is_file()
