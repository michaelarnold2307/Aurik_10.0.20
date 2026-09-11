"""MuQ-Plugin — Musik-SSL-Embeddings (SOTA 2025) für die Aurik-Voranalyse.

MuQ (Tencent AILab, arXiv:2501.01108) ist der derzeitige SOTA-Nachfolger der
wav2vec2/HuBERT-Klasse für Musik-Verständnis. Eingefrorene MuQ-Features tragen
nachweislich Qualitätsinformation (MuQ-Eval 2026: SRCC 0.957 gegen menschliche
MOS mit einem winzigen Head auf frozen Features).

Dieses Plugin liefert:

1. ``extract_embedding`` — deterministische 1024-d Song-Einbettung
   (24 kHz, zentriertes 20-s-Fenster, Zeit-Mittel über last_hidden_state).
2. ``estimate_quality_witness`` — deterministischer Qualitäts-Witness 0–100 via
   Kosinus-Ähnlichkeit zum Schwerpunkt der Clean-Referenzen
   (``corpus/vinyl/clean/*.wav``, einmalig berechnet und gecacht).
3. ``estimate_muq_mos`` — MOS 1–5 über den publizierten MuQ-Eval-A1-Head
   (Attention-Pooling + 2-Layer-MLP, MIT), sobald der extrahierte Head unter
   ``models/muq/muq_eval_a1_head.pt`` liegt (siehe
   ``scripts/extract_muq_eval_a1_head.py``).

Lade-Richtlinie (Determinismus §G5 (GEBOTE.md), Silent-Failure §V6 (copilot-instructions.md)):

- Checkpoint-Suche: lokaler HF-Cache (``OpenMuQ/MuQ-large-msd-iter``),
  ``models/muq/``, ``AURIK_MUQ_DIR`` — KEIN stiller Auto-Download.
- Gerät: GPU via ``ml_device_manager`` (fp32, ROCm/CUDA). Auf reinem CPU-Setup
  werden die ML-Schätzungen übersprungen (``logger.warning``) — der 5-s-Guard
  des RestorabilityEstimators (§2.26) bliebe sonst unhaltbar.
  Opt-in für überwachte CPU-Läufe: ``AURIK_MUQ_CPU=1``.
- Fehlende Gewichte/Head ⇒ ``None`` + Begründung im Log — kein Crash, kein
  stiller Fallback (§V6 (copilot-instructions.md)).

CLI/GUI-Funktionsgleichheit (AGENTS.md §3): Dieses Plugin wird ausschließlich
aus dem Backend (``backend/core/restorability_estimator.py``) aufgerufen — CLI
und GUI erhalten den MuQ-Qualitätsprior damit automatisch identisch.
"""

from __future__ import annotations

import gc
import hashlib
import logging
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

try:
    import torchaudio
except Exception:  # pragma: no cover
    torchaudio = None  # type: ignore[assignment]

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MODEL_ID = "OpenMuQ/MuQ-large-msd-iter"
_TARGET_SR = 24000
_MAX_ANALYSIS_S = 20.0
_EMBED_DIM = 1024
_REF_DIR = _PROJECT_ROOT / "corpus" / "vinyl" / "clean"
_REF_CACHE = _PROJECT_ROOT / "models" / "muq_mulan" / "muq_ref_embeddings.npz"
_A1_HEAD_PATH = _PROJECT_ROOT / "models" / "muq_mulan" / "muq_eval_a1_head.pt"
# Kanonischer Projekt-lokaler Modellpfad (Rev. 2026-09-11):
# models/muq_mulan/ enthält den MuQ-Encoder-Checkpoint
# (config.json + model.safetensors, OpenMuQ/MuQ-large-msd-iter).
_LOCAL_DIR = _PROJECT_ROOT / "models" / "muq_mulan"
_LOCAL_DIR_FALLBACK = _PROJECT_ROOT / "models" / "muq"
_MEMORY_GB = 1.4

_state_lock = threading.Lock()
_model: Any = None
_model_attempted: bool = False
_ref_embeddings: np.ndarray | None = None
_ref_attempted: bool = False


def _candidate_checkpoint_dirs() -> list[Path]:
    """Kandidaten-Verzeichnisse für den lokalen MuQ-Checkpoint (Reihenfolge = Priorität)."""
    candidates: list[Path] = []
    env_dir = os.environ.get("AURIK_MUQ_DIR", "").strip()
    if env_dir:
        candidates.append(Path(env_dir))
    if _LOCAL_DIR.exists():
        candidates.append(_LOCAL_DIR)
    if _LOCAL_DIR_FALLBACK.exists():
        candidates.append(_LOCAL_DIR_FALLBACK)
    # HF-Cache: Standard-HOME und projektlokale HF_HOME-Variante abdecken.
    hub_roots = [Path.home() / ".cache" / "huggingface" / "hub"]
    hf_home = os.environ.get("HF_HOME", "").strip()
    if hf_home:
        hub_roots.insert(0, Path(hf_home) / "hub")
    for root in hub_roots:
        snap_base = root / f"models--{_MODEL_ID.replace('/', '--')}" / "snapshots"
        if snap_base.is_dir():
            for snap in sorted(snap_base.iterdir(), reverse=True):
                if snap.is_dir():
                    candidates.append(snap)
    return candidates


def _find_checkpoint_dir() -> Path | None:
    for cand in _candidate_checkpoint_dirs():
        if not cand.is_dir():
            continue
        has_cfg = (cand / "config.json").is_file()
        has_w = (cand / "model.safetensors").is_file() or (cand / "pytorch_model.bin").is_file()
        if has_cfg and has_w:
            return cand
    return None


def is_available() -> bool:
    """True, wenn der lokale MuQ-Checkpoint gefunden wurde (Modell evtl. noch ungeladen)."""
    return _find_checkpoint_dir() is not None


def _unload_muq() -> None:
    global _model
    with _state_lock:
        _model = None
    gc.collect()
    try:
        from backend.core.ml_memory_budget import release as _budget_release

        _budget_release("MuQ-310M")
    except Exception as _exc:  # pragma: no cover
        logger.debug("MuQ Grenze-Release fehlgeschlagen (unkritisch): %s", _exc)


def get_muq_model() -> Any | None:
    """Lazy-Singleton: lädt den lokalen MuQ-Checkpoint (GPU-fp32, Budget + PLM-registriert)."""
    global _model, _model_attempted
    if _model is not None:
        return _model
    if torch is None or torchaudio is None:  # pragma: no cover
        logger.warning(
            "MuQ: torch/torchaudio nicht verfügbar — Embeddings übersprungen (§V6 (copilot-instructions.md))"
        )
        return None
    # §G174: Import NIEMALS innerhalb eines Locks — vor dem Lock auflösen.
    from plugins._vendor_muq import MuQ

    with _state_lock:
        if _model is not None:
            return _model
        if _model_attempted:
            return None
        _model_attempted = True
        _dir = _find_checkpoint_dir()
        if _dir is None:
            logger.warning(
                "MuQ-Checkpoint (%s) nicht lokal gefunden (models/muq/, HF-Zwischenspeicher, AURIK_MUQ_DIR) — "
                "Qualitäts-Witness übersprungen (§V6 (copilot-instructions.md))",
                _MODEL_ID,
            )
            return None
        try:
            from backend.core.ml_memory_budget import try_allocate as _budget_allocate
            from backend.core.plugin_lifecycle_manager import register_plugin as _reg_plm

            if not _budget_allocate("MuQ-310M", _MEMORY_GB):
                logger.warning(
                    "MuQ: ML-Speicherbudget verweigert — Embeddings übersprungen (§V6 (copilot-instructions.md))"
                )
                return None
            _reg_plm("MuQ", size_gb=_MEMORY_GB, unload_fn=_unload_muq)
            _model = MuQ.from_pretrained(str(_dir))
            _model.to(_resolve_device())
            _model.eval()
            logger.info("MuQ-Plugin: %s geladen (device=%s, fp32)", _MODEL_ID, str(_resolve_device()))
        except Exception as _exc:
            _model = None
            logger.warning(
                "MuQ-Checkpoint konnte nicht geladen werden (%s) — DSP-Pfad bleibt aktiv (§V6 (copilot-instructions.md))",
                _exc,
            )
        return _model


_device: Any = None


def _resolve_device() -> Any:
    """Device-Auflösung: Device-Manager zuerst, torch.cuda als Fallback.

    In nackten Skript-Kontexten (CLI-Tests, Voranalyse ohne GUI-Warmup) kann
    der Device-Manager noch CPU melden, obwohl ROCm-torch die GPU sieht —
    dann wird die GPU verwendet, damit der 5-s-Guard des Estimators (§2.26)
    eingehalten wird. Produktion (GUI) liefert der Manager ohnehin CUDA.
    """
    global _device
    if _device is not None:
        return _device
    try:
        from backend.core.ml_device_manager import get_torch_device

        _dev = get_torch_device("MuQ-310M")
    except Exception:
        _dev = None
    if _dev is None or "cpu" in str(_dev):
        if torch is not None and torch.cuda.is_available():
            _dev = torch.device("cuda")
            logger.info("MuQ: Device-Manager meldet CPU — torch.cuda verfügbar, GPU wird verwendet")
    if _dev is None and torch is not None:
        _dev = torch.device("cpu")
    _device = _dev
    return _device


def _ml_allowed_on_device(device: Any) -> bool:
    """GPU-Standard; CPU nur mit explizitem Opt-in (5-s-Guard des Estimators)."""
    dev_str = str(device)
    if "cpu" not in dev_str:
        return True
    if os.environ.get("AURIK_MUQ_CPU", "0") == "1":
        return True
    logger.warning("MuQ: CPU-Gerät ohne AURIK_MUQ_CPU=1 — ML-Schätzung übersprungen (5-s-Guard §2.26)")
    return False


def _center_window(mono: np.ndarray, sr: int) -> np.ndarray:
    """Deterministisches, zentriertes Fenster (max. _MAX_ANALYSIS_S) — Intro/Outro-robust."""
    n_max = int(_MAX_ANALYSIS_S * sr)
    if len(mono) <= n_max:
        _w1: np.ndarray = np.asarray(mono, dtype=np.float32)
        return _w1
    start = (len(mono) - n_max) // 2
    _w2: np.ndarray = np.asarray(mono[start : start + n_max], dtype=np.float32)
    return _w2


def extract_embedding(audio: Any, sr: int) -> np.ndarray | None:
    """Deterministische Song-Einbettung (1024-d float32) oder None (Modell/GPU fehlt)."""
    model = get_muq_model()
    if model is None or torch is None or torchaudio is None:
        return None
    try:
        _dev = _resolve_device()
        if not _ml_allowed_on_device(_dev):
            return None
        arr = np.asarray(audio, dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if arr.ndim == 2:
            # Layout-sicher: kanonisch (N, C) nach §11 Spec 02; (C, N) ebenfalls bedienen.
            if arr.shape[0] <= 2 and arr.shape[0] < arr.shape[1]:
                mono = arr.mean(axis=0)
            else:
                mono = arr.mean(axis=1)
        else:
            mono = arr
        mono = _center_window(mono, sr)
        if int(sr) != _TARGET_SR:
            wav = torch.from_numpy(mono).unsqueeze(0).to(_dev)
            wav = torchaudio.functional.resample(wav, int(sr), _TARGET_SR)
        else:
            wav = torch.from_numpy(mono).unsqueeze(0).to(_dev)
        with torch.no_grad():
            out = model(wav, output_hidden_states=True)
        hidden = out.last_hidden_state  # (1, T, 1024)
        emb = hidden.mean(dim=1)[0].detach().cpu().numpy().astype(np.float32)
        _emb_out: np.ndarray = emb
        return _emb_out
    except Exception as _exc:
        logger.warning(
            "MuQ-Extraktion fehlgeschlagen (%s) — DSP-Pfad bleibt aktiv (§V6 (copilot-instructions.md))", _exc
        )
        return None


def _ref_file_list_hash() -> str:
    if not _REF_DIR.is_dir():
        return "missing"
    files = sorted(p.name for p in _REF_DIR.glob("*.wav"))
    return hashlib.sha256(",".join(files).encode("utf-8")).hexdigest()[:16]


def _get_ref_embeddings() -> np.ndarray | None:
    """Clean-Referenz-Einbettungen (einmalig berechnet, deterministisch gecacht)."""
    global _ref_embeddings, _ref_attempted
    if _ref_embeddings is not None:
        return _ref_embeddings
    with _state_lock:
        if _ref_embeddings is not None:
            return _ref_embeddings
        if _ref_attempted:
            return None
        _ref_attempted = True
        tag = _ref_file_list_hash()
        if tag != "missing" and _REF_CACHE.is_file():
            try:
                cache = np.load(str(_REF_CACHE), allow_pickle=False)
                if str(cache["ref_hash"]) == tag:
                    _ref_embeddings = cache["embeddings"].astype(np.float32)
                    return _ref_embeddings
            except Exception as _exc:
                logger.warning("MuQ-Referenz-Zwischenspeicher unlesbar (%s) — wird neu berechnet", _exc)
        if tag == "missing":
            logger.warning(
                "MuQ-Referenzverzeichnis %s fehlt — Qualitäts-Witness übersprungen (§V6 (copilot-instructions.md))",
                _REF_DIR,
            )
            return None
        try:
            import soundfile as sf

            emb_list: list[np.ndarray] = []
            for path in sorted(_REF_DIR.glob("*.wav")):
                data, file_sr = sf.read(str(path), dtype="float32")
                emb = extract_embedding(data, int(file_sr))
                if emb is None:
                    logger.warning("MuQ-Referenz-Einbettung fehlgeschlagen für %s", path.name)
                    continue
                emb_list.append(emb)
            if not emb_list:
                return None
            _ref_embeddings = np.stack(emb_list, axis=0).astype(np.float32)
            _REF_CACHE.parent.mkdir(parents=True, exist_ok=True)
            np.savez(str(_REF_CACHE), ref_hash=tag, embeddings=_ref_embeddings)
            logger.info("MuQ-Referenz-Einbettungen berechnet: %d Clean-Referenzen", len(emb_list))
            return _ref_embeddings
        except Exception as _exc:
            logger.warning(
                "MuQ-Referenz-Einbettungen fehlgeschlagen (%s) — Witness übersprungen (§V6 (copilot-instructions.md))",
                _exc,
            )
            return None


def _sim_to_witness(sim: float) -> float:
    """Deterministische Abbildung Kosinus-Ähnlichkeit → Witness 0–100 (linear, saturierend)."""
    return float(np.clip(50.0 + 100.0 * (sim - 0.5), 0.0, 100.0))


def estimate_quality_witness(audio: Any, sr: int) -> float | None:
    """Qualitäts-Witness 0–100: Nähe zur Clean-Referenz (MuQ-Raum), sonst None."""
    emb = extract_embedding(audio, sr)
    refs = _get_ref_embeddings()
    if emb is None or refs is None:
        return None
    centroid = refs.mean(axis=0).astype(np.float64)
    denom = float(np.linalg.norm(centroid)) * float(np.linalg.norm(emb))
    if denom < 1e-12:
        return None
    sim = float(np.dot(centroid, emb.astype(np.float64)) / denom)
    return round(_sim_to_witness(sim), 1)


# ─── MuQ-Eval-A1-Head (MIT, dgtql/MuQ-Eval) — MOS 1–5 ────────────────────────
#
# Architektur exakt nach MuQ-Eval `src/model.py`/`src/encoders.py`:
#   hidden (B,T,1024) → AttentionPooling → PredictionHead(2×MLP) → MI-Score.
# Der Head wird aus dem publizierten A1-Checkpoint extrahiert
# (scripts/extract_muq_eval_a1_head.py) und liegt als models/muq/muq_eval_a1_head.pt.


class _A1AttentionPooling(torch.nn.Module if torch is not None else object):  # type: ignore[misc]
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.attention = torch.nn.Sequential(  # type: ignore[attr-defined]
            torch.nn.Linear(dim, 128),  # type: ignore[attr-defined]
            torch.nn.Tanh(),  # type: ignore[attr-defined]
            torch.nn.Linear(128, 1),  # type: ignore[attr-defined]
        )

    def forward(self, x: Any) -> Any:  # type: ignore[override]
        attn = self.attention(x).squeeze(-1)
        attn = torch.softmax(attn, dim=-1)  # type: ignore[attr-defined]
        return torch.bmm(attn.unsqueeze(1), x).squeeze(1)  # type: ignore[attr-defined]


class _A1PredictionHead(torch.nn.Module if torch is not None else object):  # type: ignore[misc]
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float = 0.1) -> None:
        super().__init__()
        layers: list[Any] = []
        in_dim = input_dim
        for _ in range(num_layers - 1):
            layers.append(torch.nn.Linear(in_dim, hidden_dim))  # type: ignore[attr-defined]
            layers.append(torch.nn.GELU())  # type: ignore[attr-defined]
            layers.append(torch.nn.Dropout(dropout))  # type: ignore[attr-defined]
            in_dim = hidden_dim
        layers.append(torch.nn.Linear(in_dim, 1))  # type: ignore[attr-defined]
        self.mlp = torch.nn.Sequential(*layers)  # type: ignore[attr-defined]

    def forward(self, x: Any) -> Any:  # type: ignore[override]
        return self.mlp(x).squeeze(-1)


_a1_head_state: dict[str, Any] = {}
_a1_head_loaded: bool = False
_a1_head_lock = threading.Lock()


def _get_a1_head_modules() -> tuple[Any, Any] | None:
    global _a1_head_loaded
    if _a1_head_loaded:
        pooling: Any = _a1_head_state.get("pooling")
        head: Any = _a1_head_state.get("head")
        return (pooling, head) if pooling is not None and head is not None else None
    with _a1_head_lock:
        if not _a1_head_loaded:
            _a1_head_loaded = True
            if not _A1_HEAD_PATH.is_file() or torch is None:
                return None
            try:
                state = torch.load(str(_A1_HEAD_PATH), map_location="cpu", weights_only=True)
                pooling = _A1AttentionPooling(_EMBED_DIM)
                pooling.load_state_dict(state["pooling"])
                head = _A1PredictionHead(_EMBED_DIM, 256, 2)
                head.load_state_dict(state["head"])
                pooling.eval()
                head.eval()
                _a1_head_state["pooling"] = pooling
                _a1_head_state["head"] = head
                logger.info("MuQ-Eval-A1-Head geladen: %s", _A1_HEAD_PATH.name)
            except Exception as _exc:
                logger.warning(
                    "MuQ-Eval-A1-Head konnte nicht geladen werden (%s) — MOS-Prior übersprungen (§V6 (copilot-instructions.md))",
                    _exc,
                )
                return None
    pooling = _a1_head_state.get("pooling")
    head = _a1_head_state.get("head")
    return (pooling, head) if pooling is not None and head is not None else None


def estimate_muq_mos(audio: Any, sr: int) -> float | None:
    """MOS 1–5 über den MuQ-Eval-A1-Head; None wenn Head/Modell/GPU fehlt."""
    _modules = _get_a1_head_modules()
    if _modules is None:
        return None
    model = get_muq_model()
    if model is None or torch is None or torchaudio is None:
        return None
    try:
        _dev = _resolve_device()
        if not _ml_allowed_on_device(_dev):
            return None
        arr = np.asarray(audio, dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if arr.ndim == 2:
            if arr.shape[0] <= 2 and arr.shape[0] < arr.shape[1]:
                mono = arr.mean(axis=0)
            else:
                mono = arr.mean(axis=1)
        else:
            mono = arr
        mono = _center_window(mono, sr)
        if int(sr) != _TARGET_SR:
            wav = torch.from_numpy(mono).unsqueeze(0).to(_dev)
            wav = torchaudio.functional.resample(wav, int(sr), _TARGET_SR)
        else:
            wav = torch.from_numpy(mono).unsqueeze(0).to(_dev)
        pooling, head = _modules
        with torch.no_grad():
            out = model(wav, output_hidden_states=True)
            pooled = pooling(out.last_hidden_state.to(_dev))
            mi = head(pooled)
        return round(float(np.clip(float(mi.item()), 1.0, 5.0)), 2)
    except Exception as _exc:
        logger.warning(
            "MuQ-MOS-Schätzung fehlgeschlagen (%s) — DSP-MOS bleibt aktiv (§V6 (copilot-instructions.md))", _exc
        )
        return None
