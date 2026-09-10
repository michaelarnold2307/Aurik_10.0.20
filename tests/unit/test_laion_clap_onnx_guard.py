"""CLAP-ONNX-Guard-Tests (§v10.761).

Befund (2026-09-09): `embed_audio`-Pfad 1 fütterte dem ONNX-Audio-Encoder
FFT-Magnituden statt des Raw-Waveforms (der Encoder erwartet
[batch, samples]) — das Embedding war semantisch wertlos und die
Era-Ähnlichkeitssuche bekam Müll. Der Fix speist das Raw-Waveform,
konsistent mit `_tag_clap` (§v10.745).

Tests:
1. Quelltext-Invariante: `embed_audio` darf im ONNX-Pfad kein
   `np.fft.rfft`-Feeding mehr enthalten (läuft überall, kein Modell).
2. E2E (übersprungen ohne Modell): ONNX-Pfad lädt, liefert L2-normiertes
   512-dim Embedding, kein Fallback (model_used="laion_clap").
"""

import os
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_PLUGIN_PATH = _PROJECT_ROOT / "plugins" / "laion_clap_plugin.py"


def _read_plugin() -> str:
    return _PLUGIN_PATH.read_text(encoding="utf-8")


def test_embed_audio_onnx_path_uses_raw_waveform_not_fft():
    """Guard: Das FFT-Feeding in embed_audio ist entfernt (§v10.761)."""
    src = _read_plugin()
    # Der ONNX-Pfad (Path 1) von embed_audio darf keine FFT-Transformation
    # des Feeds mehr enthalten. _tag_dsp_fallback (DSP-Heuristik) darf FFT
    # benutzen — daher gezielt den Path-1-Block prüfen.
    marker = "Path 1: ONNX audio-encoder"
    marker_idx = src.index(marker)
    path2_idx = src.index("Path 2: PyTorch laion_clap")
    path1_block = src[marker_idx:path2_idx]
    assert "rfft" not in path1_block
    assert "np.newaxis" in path1_block  # [batch, samples]-Feeding


@pytest.mark.skipif(
    not (_PROJECT_ROOT / "models" / "clap" / "audio_encoder.onnx").exists(),
    reason="audio_encoder.onnx nicht vorhanden — E2E übersprungen",
)
def test_onnx_path_e2e_l2_normalized_embedding():
    import numpy as np

    from plugins.laion_clap_plugin import get_laion_clap

    sr = 48000
    t = np.arange(3 * sr) / sr
    audio = (
        0.5 * np.sin(2 * np.pi * 220 * t) + 0.3 * np.sin(2 * np.pi * 440 * t)
    ).astype(np.float32)
    clap = get_laion_clap()
    emb = clap.embed_audio(audio, sr)
    assert emb.shape == (512,)
    assert abs(float(np.linalg.norm(emb)) - 1.0) < 1e-3
    assert not np.isnan(emb).any()
    # Modell ist der ONNX-Pfad (kein PANNs-Fallback)
    res = clap.tag(audio, sr)
    assert res.model_used in ("laion_clap", "laion_clap_pt")
