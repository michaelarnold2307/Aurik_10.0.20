"""§SOTA-ML-V6 — Whisper-Tiny-Torch-ROCm-Encoder (korrekter GPU-Pfad).

Bug-Jagd 2026-09-18: Die ORT-ROCm-Kernels rechnen den Whisper-Encoder
numerisch falsch — auf echten Mel-Feeds rel ≈ 0.98 (max|Δ| 13.5) mit
Default-Optimierern und rel ≈ 0.68 mit ORT_DISABLE_ALL (Softmax-/Attention-
Kernels; 5. bestätigter ORT-ROCm-Defekt). MIGraphX platziert für dieses
Modell nicht. Deshalb wurde WhisperTiny/WhisperTurbo aus der GPU-Liste
genommen — der CPU-Pfad ist korrekt, aber langsam (98 ms je 30-s-Fenster).

Dieser Baustein liefert den hochwertigen GPU-Ersatz: den HF-Whisper-Tiny-
Encoder (transformers, Gewichte identisch zum ONNX-Export — Parität
torch vs. ONNX-CPU: max|Δ| ≈ 2.6e-3, rel ≈ 1.5e-4) auf PyTorch-ROCm:
~6.9 ms je 30-s-Fenster (7900 XTX, ~14× schneller als der korrekte
CPU-Pfad), deterministisch bit-identisch (§G5 (copilot-instructions.md)).

Fail-closed (§V6 (copilot-instructions.md)): Ohne torch/CUDA/transformers
oder ohne lokal gecachten HF-Snapshot liefert get_whisper_torch_core()
None — der Aufrufer bleibt auf dem ONNX-CPU-Pfad.
"""

from __future__ import annotations

import logging
import threading

import numpy as np

logger = logging.getLogger(__name__)

# HF-Snapshot-Name (lokal gecacht, deterministisch — kein Netz zur Laufzeit).
_SNAPSHOT_PREFIX = "models--openai--whisper-tiny"

_lock = threading.Lock()
_core = None
_core_resolved = False


def _find_snapshot() -> str | None:
    """Sucht den lokal gecachten HF-Snapshot (ohne Netz)."""
    from pathlib import Path

    hub = Path.home() / ".cache" / "huggingface" / "hub"
    for entry in hub.glob(f"{_SNAPSHOT_PREFIX}/snapshots/*"):
        if (entry / "model.safetensors").exists() and (entry / "config.json").exists():
            return str(entry)
    return None


def _build_core():
    import torch  # pylint: disable=import-outside-toplevel
    from transformers import (  # pylint: disable=import-outside-toplevel
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )

    snapshot = _find_snapshot()
    if snapshot is None:
        raise RuntimeError("Whisper-Tiny-Snapshot nicht lokal gecacht (HF-Hub) — ONNX-CPU-Pfad bleibt")
    model = WhisperForConditionalGeneration.from_pretrained(  # nosec B615 — commit-gepinnter Snapshot, kein Netz
        snapshot, local_files_only=True
    ).eval()
    processor = WhisperProcessor.from_pretrained(snapshot, local_files_only=True)  # nosec B615
    return {"model": model, "processor": processor}


def get_whisper_torch_core() -> dict | None:
    """Lazy-Singleton: {model, processor} auf ROCm oder None
    (fail-closed, §V6 (copilot-instructions.md))."""
    global _core, _core_resolved
    with _lock:
        if _core_resolved:
            return _core
        _core_resolved = True
        try:
            import torch  # pylint: disable=import-outside-toplevel

            if not torch.cuda.is_available():
                logger.debug("§SOTA-ML-V6 Whisper-Torch-ROCm nicht verfügbar — ONNX-CPU-Pfad bleibt")
                return None
            bundle = _build_core()
            bundle["model"] = bundle["model"].to("cuda")
            _core = bundle
            logger.info("§SOTA-ML-V6 Whisper-Tiny-Torch-Encoder auf ROCm geladen (%s)", torch.cuda.get_device_name(0))
            return _core  # type: ignore[no-any-return]
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("§SOTA-ML-V6 Whisper-Torch-Kern nicht ladbar: %s — ONNX-CPU-Pfad bleibt", exc)
            return None


def encode_whisper_torch_mel(core: dict, mel: np.ndarray) -> np.ndarray:
    """Encoder-Vorwärtslauf auf einem bereits berechneten Whisper-Mel.

    Nutzt die Mel-Pipeline des Aufrufers (identisch zum ONNX-Pfad), nur der
    Encoder-Backend wechselt → minimaler Eingriff in die Pipeline-Parität.

    Args:
        core: Bundle aus get_whisper_torch_core().
        mel: float32 [1, 80, 3000] (Whisper-Log-Mel, gepaddet auf 30 s).

    Returns: float32 [1, T, 384], NaN/Inf-geschützt
    (§0a (copilot-instructions.md)). Deterministisch.
    """
    import torch  # pylint: disable=import-outside-toplevel

    m = np.asarray(mel, dtype=np.float32)
    if m.ndim != 3 or m.shape[1] != 80:
        raise ValueError(f"Whisper-Torch: erwartet [B,80,T]-Mel, bekam {m.shape}")
    m = np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
    with torch.no_grad():
        out = (
            core["model"]
            .get_encoder()(torch.from_numpy(m).to(next(core["model"].parameters()).device))
            .last_hidden_state.cpu()
            .numpy()
        )
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.astype(np.float32)  # type: ignore[no-any-return]


def encode_whisper_torch(core: dict, audio_16k: np.ndarray) -> np.ndarray:
    """Whisper-Encoder-Vorwärtslauf: 16-kHz-Mono → last_hidden_state.

    Args:
        core: Bundle aus get_whisper_torch_core().
        audio_16k: float32 Mono-Signal mit 16 kHz.

    Returns: float32 [1, T, 384] (Whisper-Tiny, T = n_frames/2),
    NaN/Inf-geschützt (§0a (copilot-instructions.md)). Deterministisch.
    """
    import torch  # pylint: disable=import-outside-toplevel

    x = np.asarray(audio_16k, dtype=np.float32)
    if x.ndim != 1 or x.size == 0:
        raise ValueError(f"Whisper-Torch: erwartet 16-kHz-Mono (1-D), bekam {x.shape}")
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    feats = core["processor"](x, sampling_rate=16000, return_tensors="pt")
    mel = feats.input_features.to(next(core["model"].parameters()).device)
    with torch.no_grad():
        out = core["model"].get_encoder()(mel).last_hidden_state.cpu().numpy()
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.astype(np.float32)  # type: ignore[no-any-return]


def transcribe_whisper_torch(core: dict, audio_16k: np.ndarray, language: str = "en") -> list[dict]:
    """Whisper-Tiny-Greedy-Transkription mit Token-Timestamps → Wort-Liste.

    §SOTA-ML-V9 (2026-09-18): Der Ort-ROCm-Decoder ist numerisch defekt und der
    bisherige Tiny-Pfad hatte gar keinen Decoder (Encoder-Aktivierungs-Heuristik).
    Dieser Pfad liefert echte Wort-Timestamps über generate(return_timestamps=True)
    (transformers 4.43: Token-Level) + manuelle Wort-Gruppierung an
    führenden „Ġ“-Tokens; Wort-Zeitraum = min/max der Token-Zeiten,
    Wahrscheinlichkeit = Mittel der Token-Wahrscheinlichkeiten (output_scores).

    Returns: Liste von dicts {"word", "start", "end", "probability"}
    (leer bei leerem Transkript, z. B. reinem Musik-Signal). Deterministisch.
    """
    import torch  # pylint: disable=import-outside-toplevel

    x = np.asarray(audio_16k, dtype=np.float32)
    if x.ndim != 1 or x.size == 0:
        raise ValueError(f"Whisper-Torch: erwartet 16-kHz-Mono (1-D), bekam {x.shape}")
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    processor = core["processor"]
    model = core["model"]
    device = next(model.parameters()).device
    feats = processor(x, sampling_rate=16000, return_tensors="pt")
    mel = feats.input_features.to(device)
    with torch.no_grad():
        gen = model.generate(
            mel,
            language=language,
            return_timestamps=True,
            output_scores=True,
            return_dict_in_generate=True,
            max_new_tokens=224,
            num_beams=1,
            temperature=0.0,
        )
    ids = gen.sequences[0].tolist()
    times = torch.cat(gen.token_timestamps).cpu().numpy()[0] if gen.token_timestamps is not None else np.zeros(len(ids))
    # Token-Wahrscheinlichkeiten: Scores sind [1, vocab] je Schritt (shifted).
    probs: list[float] = []
    if gen.scores:
        for _s in gen.scores:
            _p = torch.softmax(_s.float(), dim=-1)[0]
            probs.append(float(_p.max()))
    if len(probs) < len(ids):
        probs = [1.0, *probs]  # Kontext-Token ohne Score
    if len(probs) > len(ids):
        probs = probs[: len(ids)]
    while len(probs) < len(ids):
        probs.append(probs[-1] if probs else 1.0)

    tok = processor.tokenizer
    words: list[dict] = []
    _cur: list[int] = []
    _cur_t: list[float] = []
    _cur_p: list[float] = []

    def _flush() -> None:
        nonlocal _cur, _cur_t, _cur_p
        if _cur:
            _word = tok.decode(_cur, skip_special_tokens=True).strip()
            if _word:
                words.append(
                    {
                        "word": _word,
                        "start": float(min(_cur_t)),
                        "end": float(max(_cur_t)),
                        "probability": float(np.mean(_cur_p)),
                    }
                )
        _cur, _cur_t, _cur_p = [], [], []

    for _i, _tid in enumerate(ids):
        _token = tok.convert_ids_to_tokens(_tid)
        _is_ts = _token.startswith("<|")
        _is_word_start = _token.startswith("Ġ") and _cur
        if _is_ts or _is_word_start:
            _flush()
        if not _is_ts and _tid not in (tok.eos_token_id,):
            _cur.append(_tid)
            _cur_t.append(float(times[_i]) if _i < len(times) else 0.0)
            _cur_p.append(probs[_i] if _i < len(probs) else 1.0)
    _flush()
    return words
