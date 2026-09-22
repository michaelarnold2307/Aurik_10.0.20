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
_MOS_CLIP_S = 10.0  # MuQ-Eval base.yaml: clip_duration_sec
_MOS_CLIP_SAMPLES = 240000  # 24000 × 10 — A1-Head wurde auf 10-s-Clips trainiert
_EMBED_DIM = 1024
_REF_DIR = _PROJECT_ROOT / "corpus" / "vinyl" / "clean"
_REF_CACHE = _PROJECT_ROOT / "models" / "muq_mulan" / "muq_ref_embeddings.npz"
_A1_HEAD_PATH = _PROJECT_ROOT / "models" / "muq_mulan" / "muq_eval_a1_head.pt"
# Kanonische Projekt-lokale Modellpfade:
# - models/muq_mulan/ = MuQ-large-msd-iter Backbone (config.json + model.safetensors,
#                       ~1,3 GB, byte-identisch zum HF-Snapshot) — Primärpfad, auch
#                       vom ROCm-Kern (muq_mulan_torch_rocm) geladen.
# - models/muq/       = optionaler Offline-Fallback (aktuell nicht belegt — keine
#                       redundante 1,3-GB-Kopie neben dem Primärpfad).
# - models/muq_mulan/mulan/ = MuQ-MuLan-Checkpoint (anderer Trainingsstand, nur MuLan-Turm).
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
    candidates = _candidate_checkpoint_dirs()
    # A1-Head wurde auf OpenMuQ/MuQ-large-msd-iter trainiert (base.yaml) —
    # der HF-Cache-Snapshot dieses Backbones MUSS vor dem MuQ-MULAN-
    # Backbone (models/muq_mulan/, anderer Trainingsstand) priorisiert werden,
    # sonst invertiert die MOS-Richtung (Befund 2026-09-13).
    msd_first = [c for c in candidates if "MuQ-large-msd-iter" in str(c)] + [
        c for c in candidates if "MuQ-large-msd-iter" not in str(c)
    ]
    for cand in msd_first:
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
            # 1:1-Pfad wie der verifiziert richtungs-korrekte MuQ-Eval-Lauf:
            # MuQ.from_pretrained mit der HF-ID lädt die Revision MIT korrekten
            # BatchNorm-running-Statistiken; der lokale Verzeichnis-Pfad zog
            # eine Revision ohne (18 running_mean/var abweichend → MOS-Inversion,
            # Befund 2026-09-13). Fallback: lokales Verzeichnis.
            try:
                _model = MuQ.from_pretrained(_MODEL_ID)
            except Exception as _id_exc:
                logger.debug("MuQ: HF-ID-Laden fehlgeschlagen (%s) — lokaler Pfad-Ersatz", _id_exc)
                _model = MuQ.from_pretrained(str(_dir))
            _model.to(_resolve_device())
            # BatchNorm-Statistiken aus dem A1-Checkpoint nachladen (Befund
            # 2026-09-13): Der HF-Hub-Export des MuQ-Backbones weicht in
            # 18 running_mean/running_var-Paaren vom A1-Trainingsstand ab;
            # ohne Korrektur invertiert die MOS-Richtung (ref ok, noise10 kippt).
            _bn_path = _LOCAL_DIR / "muq_eval_bn_stats.pt"
            if _bn_path.is_file():
                try:
                    _bn = torch.load(str(_bn_path), map_location="cpu", weights_only=True)
                    _missing, _unexp = _model.load_state_dict(_bn, strict=False)
                    if _missing or _unexp:
                        logger.debug("MuQ-BN-Korrektur: missing=%d unexpected=%d", len(_missing), len(_unexp))
                except Exception as _bn_exc:
                    logger.warning("MuQ-BN-Korrektur fehlgeschlagen: %s", _bn_exc)
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

    §G5 (GEBOTE.md)/Defizit-Fix 2026-09-16: `AURIK_MUQ_GPU=0` erzwingt CPU
    (bit-deterministisch) — ROCm-GPU-Inferenz war nach Messung von 2026-09-16
    NICHT bit-deterministisch (vgl. MuLan-Befund). **Nachmessung 2026-09-18:**
    auf dem aktuellen Torch-ROCm-Stack sind zwei aufeinanderfolgende GPU-Läufe
    bit-identisch (max|Δ| = 0.0, 7900 XTX) — der Befund ist nicht mehr
    reproduzierbar. Der Schalter bleibt als expliziter Opt-out für
    Determinismus-Kontexte erhalten (Tests, §G5-Zertifikate).
    """
    global _device
    # §G5 (GEBOTE.md): Der Determinismus-Opt-out wird VOR dem Cache geprüft —
    # sonst ignoriert ein bereits warmgeladener GPU-Singleton den Schalter.
    if os.environ.get("AURIK_MUQ_GPU", "").strip() == "0":
        _device = torch.device("cpu") if torch is not None else "cpu"
        return _device
    if _device is not None:
        return _device
    _dev: Any = None
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
    """GPU-Standard; CPU nur mit explizitem Opt-in (5-s-Guard des Estimators).

    AURIK_MUQ_GPU=0 (Determinismus-Opt-out) impliziert CPU-Freigabe.
    """
    dev_str = str(device)
    if "cpu" not in dev_str:
        return True
    if os.environ.get("AURIK_MUQ_CPU", "0") == "1":
        return True
    if os.environ.get("AURIK_MUQ_GPU", "").strip() == "0":
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


def _mos_eval_window(mono: np.ndarray, sr: int) -> np.ndarray:
    """10-s-Eval-Fenster exakt nach MuQ-Eval (base.yaml: 24 kHz, clip_samples=240000).

    Die richtungs-korrekte 1:1-Validierung (2026-09-13, MUSDB) nutzte die ERSTEN
    10 s mit librosa-Resample auf 24 kHz — das Plugin verwendete davor ein
    zentriertes 20-s-Fenster plus torchaudio-Resample und invertierte damit die
    MOS-Richtung (noise10 Δ−0.333 statt Δ+3.337). torchaudio-Resample bleibt
    dokumentierter Fallback, wenn librosa fehlt; kürzere Eingaben werden
    null-aufgefüllt (AudioProcessor-Pad-Konvention).
    """
    n_target = _MOS_CLIP_SAMPLES
    _librosa: Any | None = None
    try:
        import librosa as _imported_librosa

        _librosa = _imported_librosa
    except Exception:  # pragma: no cover
        _librosa = None
    if int(sr) != _TARGET_SR:
        # WIT-M1-Fix (2026-09-14): MuQ-Eval resampled exakt mit
        # torchaudio.functional.resample (Kaiser-Sinc) — librosa nutzt einen
        # anderen Filter; die abweichenden Embeddings können die invertierte
        # MOS-Richtung des gefrorenen A1-Heads erklären. Eval-exakt zuerst.
        if torchaudio is not None and torch is not None:
            _wav = torch.from_numpy(np.asarray(mono, dtype=np.float32)).float()
            mono = torchaudio.functional.resample(_wav, int(sr), _TARGET_SR).numpy().astype(np.float32)
        elif _librosa is not None:
            mono = np.asarray(_librosa.resample(mono, orig_sr=int(sr), target_sr=_TARGET_SR), dtype=np.float32)
    if mono.size >= n_target:
        _clip: np.ndarray = np.asarray(mono[:n_target], dtype=np.float32)
        return _clip
    _pad: np.ndarray = np.pad(np.asarray(mono, dtype=np.float32), (0, n_target - mono.size))
    return _pad


def extract_embedding(audio: Any, sr: int) -> np.ndarray | None:
    """Deterministische Song-Einbettung (1024-d float32) oder None (Modell/GPU fehlt)."""
    model = get_muq_model()
    if model is None or torch is None or torchaudio is None:
        return None
    try:
        _dev = _resolve_device()
        if not _ml_allowed_on_device(_dev):
            return None
        # Warm-GPU-Singleton + CPU-Anforderung (AURIK_MUQ_GPU=0): das Modell
        # auf das Zielgerät bewegen — sonst Device-Mismatch statt Fallback
        # (Defizit-Fix 2026-09-16; .to() ist idempotent/cheap bei Gleichheit).
        _first_param = next(model.parameters(), None)
        if _first_param is not None and str(_first_param.device) != str(_dev):
            model.to(_dev)
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
            from backend.file_import import load_audio_file

            emb_list: list[np.ndarray] = []
            for path in sorted(_REF_DIR.glob("*.wav")):
                _ld = load_audio_file(str(path))
                data = _ld.get("audio") if isinstance(_ld, dict) else None
                file_sr = _ld.get("sr") if isinstance(_ld, dict) else None
                if data is None or file_sr is None:
                    logger.warning("MuQ-Referenz-Einbettung fehlgeschlagen für %s (Datei nicht lesbar)", path.name)
                    continue
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


# MuQ-Eval-Originalklassen (models/muq_eval/src) für den 1:1-A1-Pfad:
# Der verifiziert richtungs-korrekte Lauf nutzt AttentionPooling/PredictionHead
# aus dem MuQ-Eval-Repo — die Plugin-Reimplementierungen sind ausgeschlossen,
# also werden hier die Originalklassen geladen (Fallback: alte Klassen).
try:
    import importlib.machinery as _ilm
    import importlib.util as _ilu
    import sys

    # §SOTA-Hygiene (2026-09-16, Defizit-Fix): models/muq_eval/src enthält
    # GENERISCHE Modulnamen (data.py, model.py, encoders.py). Ein dauerhaftes
    # sys.path-Insert an Position 0 kaperte jedes spätere `import data`/
    # `import model` im Prozess (u. a. brach es den Gacela-Upstream-Import
    # mit "'data' is not a package"). Der Import läuft jetzt in einem
    # scoped Kontext: sys.path wird nach dem Laden exakt wiederhergestellt;
    # die geladenen src.*-Module bleiben in sys.modules.
    _muq_eval_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "muq_eval")
    _saved_sys_path_muq = list(sys.path)
    try:
        if _muq_eval_root not in sys.path:
            sys.path.insert(0, _muq_eval_root)
            sys.path.insert(0, os.path.join(_muq_eval_root, "src"))
        if "peft" not in sys.modules:
            _peft_stub = _ilu.module_from_spec(_ilm.ModuleSpec("peft", loader=None))
            # Attribute via __dict__ (weder setattr noch direkte Zuweisung):
            # B010-sicher und mypy-sauber (kein attr-defined).
            _peft_stub.__dict__.update(
                {
                    "LoraConfig": lambda **kw: None,
                    "get_peft_model": lambda m, c: m,
                }
            )
            sys.modules["peft"] = _peft_stub
        import plugins._vendor_muq as _vendor_muq_alias

        sys.modules.setdefault("muq", _vendor_muq_alias)
        from src.encoders import AttentionPooling as _MuQEvalAttentionPooling
        from src.model import PredictionHead as _MuQEvalPredictionHead

        _MUQ_EVAL_CLASSES = True
    finally:
        # Exakte Wiederherstellung: kein globaler sys.path-Seiteneffekt
        # (§V7 (copilot-instructions.md)).
        sys.path[:] = _saved_sys_path_muq
except Exception as _muq_eval_exc:
    logger.debug("MuQ-Eval-Originalklassen nicht ladbar (%s) — Plugin-Klassen bleiben", _muq_eval_exc)
    _MUQ_EVAL_CLASSES = False


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
                pooling = _MuQEvalAttentionPooling(_EMBED_DIM) if _MUQ_EVAL_CLASSES else _A1AttentionPooling(_EMBED_DIM)
                pooling.load_state_dict(state["pooling"])
                head = (
                    _MuQEvalPredictionHead(_EMBED_DIM, 256, 2)
                    if _MUQ_EVAL_CLASSES
                    else _A1PredictionHead(_EMBED_DIM, 256, 2)
                )
                head.load_state_dict(state["head"])
                # Auf das aktive ML-Device bewegen — das MuQ-Backbone läuft auf
                # cuda; ein CPU-Head würde im Head-MLP zu Device-Mismatch führen.
                _dev = _resolve_device()
                pooling = pooling.to(_dev)
                head = head.to(_dev)
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
        # Warm-GPU-Singletons + CPU-Anforderung (AURIK_MUQ_GPU=0): Backbone
        # UND A1-Module auf das Zielgerät bewegen (Defizit-Fix 2026-09-16).
        _first_param_mos = next(model.parameters(), None)
        if _first_param_mos is not None and str(_first_param_mos.device) != str(_dev):
            model.to(_dev)
        pooling, head = _modules
        _first_param_pool = next(pooling.parameters(), None)
        if _first_param_pool is not None and str(_first_param_pool.device) != str(_dev):
            pooling.to(_dev)
            head.to(_dev)
        arr = np.asarray(audio, dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if arr.ndim == 2:
            if arr.shape[0] <= 2 and arr.shape[0] < arr.shape[1]:
                mono = arr.mean(axis=0)
            else:
                mono = arr.mean(axis=1)
        else:
            mono = arr
        mono = _mos_eval_window(mono, sr)
        wav = torch.from_numpy(mono).unsqueeze(0).to(_dev)
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
