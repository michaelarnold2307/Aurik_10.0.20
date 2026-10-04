"""Cantus Plugin — MERT-conditionierte Multi-Scale Flow-Matching Gesangsrestaurierung.

Cantus Vocal Restorer (Eigenentwicklung, models/cantus/README.md):
  - Feature Encoder: MERT-v1-330M (1024-d, music-aware SSL) + Pitch-Track
    (FCPE → CREPE) + Harmonic Context (MuQ-MuLan, 768-d)
  - Flow-Matching DiT Backbone (miipher_dit-Infrastruktur, Multi-Scale:
    Low-Freq harmonisch / High-Freq transient, Temporal Attention)
  - Inferenz: ŷ = x + (1−t)·v̂ bei t=0.5 (Flow-Matching-Formel,
    siehe scripts/train_cantus.py)

Aurik-Integration (nach MiipherDiTPlugin-Muster):
  - PLM-registriert (§4.6b-Praxis der Repo-Plugins): set_active("CANTUS", ...)
  - Hallucination-Guard: spectral_novelty > 0.35 → Rollback
  - SingMOS Never-worsen-Guard: delta-basiert (Guard-Kalibrierung: Hard-Fail nur
    bei Regression gegenüber dem Input)
  - DSP-Fallback (§V6 (copilot-instructions.md)): Wiener bei Modell-Fehlern,
    IMMER mit logger.warning + Begründung
  - M/S-Processing: Stereo→Mid→Restore→Stereo; Layout-Invariante: Mono (N,),
    (N, 2) und channels-first (C, N) werden an der Modul-Grenze normalisiert
  - Sample-Rate-Adapter: 44.1/48 kHz Input → 48 kHz Modell-SR
  - RAM-Budget: 3.0 GB (DiT + MERT + Pitch + MuQ-Sessions)
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np

logger = logging.getLogger(__name__)

# ── Lazy imports für optionale Abhängigkeiten ────────────────────────────────
try:
    import onnxruntime as ort

    _ONNX_AVAILABLE = True
except ImportError:
    _ONNX_AVAILABLE = False
    ort = None  # type: ignore[assignment]

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]

try:
    from backend.core.plugin_lifecycle_manager import (
        get_plugin_lifecycle_manager,
    )
    from backend.core.plugin_lifecycle_manager import (
        register_plugin as _plm_register,
    )

    _PLM_AVAILABLE = True
except ImportError:
    _PLM_AVAILABLE = False
    _plm_register = None  # type: ignore[assignment]

try:
    from backend.core.ml_memory_budget import release as _ml_budget_release
    from backend.core.ml_memory_budget import try_allocate as _ml_budget_try_allocate
except ImportError:
    _ml_budget_try_allocate = None  # type: ignore[assignment]
    _ml_budget_release = None  # type: ignore[assignment]

try:
    from backend.core.music_model_flags import resolve_model_path as _resolve_flag
except Exception:
    _resolve_flag = None  # type: ignore[assignment]


def _resolve_or(fallback: Path, model_key: str) -> Path:
    """§v10.19 Feature-Flag-Routing: aufgelöster Pfad, sonst Fallback."""
    if _resolve_flag is not None:
        try:
            resolved = _resolve_flag(model_key)
            if resolved is not None and Path(resolved).exists():
                return Path(resolved)
        except Exception:
            logger.debug("music_model_flags.resolve_model_path fehlgeschlagen", exc_info=True)
    return fallback


@dataclass
class CantusResult:
    """Ergebnis der Cantus-Gesangsrestaurierung."""

    audio: np.ndarray
    applied: bool
    model_used: str  # "cantus" | "dsp_fallback" | "none"
    novelty: float = 0.0
    singmos_before: float | None = None
    singmos_after: float | None = None
    processing_time_s: float = 0.0
    metadata: dict | None = field(default_factory=dict)


class CantusPlugin:
    """MERT-conditionierter Flow-Matching Vocal Restorer.

    Aktiviert für stark degradierten Gesang (Codec-Materialien oder
    restorability < 30) — analog zur MiipherDiT-Material-Gate-Logik.
    """

    _BUDGET_NAME: str = "CANTUS"
    _BUDGET_SIZE_GB: float = 3.0  # DiT + MERT + FCPE + MuQ-MuLan-Sessions
    _MODEL_SR: int = 48000
    _HALLUCINATION_THRESHOLD: float = 0.35
    _SINGMOS_MARGIN: float = 0.1  # Never-worsen: Toleranzband in MOS-Punkten
    _FLOW_TIME: float = 0.5  # Inferenz-Schnitt des OT-Pfads (trainingskonform)
    _TARGET_MATERIALS: frozenset[str] = frozenset({"mp3_low", "streaming", "aac", "minidisc", "karaoke_low"})

    def __init__(self, model_path: Path | None = None) -> None:
        self._model_loaded = False
        self._fallback_active = False
        self._fallback_reason = ""
        self._ort_session: Any | None = None
        self._torch_model: Any | None = None
        self._inference_backend = "none"
        self._warned: set[str] = set()
        self._last_use_cond: float = 0.0
        self._mert_extractor: Any | None = None
        self._mert_lock = threading.Lock()
        self._mert_init_failed = False

        model_dir = Path(__file__).parent.parent / "models" / "cantus"
        self._onnx_path: Path = Path(model_path) if model_path else _resolve_or(model_dir / "cantus_dit.onnx", "cantus")
        # Ein expliziter model_path isoliert Tests und alternative Modelle. Der
        # produktive Torch-ROCm-Kern gehört nur zum kanonischen Cantus-Modell.
        self._checkpoint_path: Path | None = model_dir / "checkpoint_best.pt" if model_path is None else None
        self._try_load_model()

    # ── Modell-Ladung + Budget ──────────────────────────────────────────

    def _production_qualified(self) -> bool:
        """Kanonische Gewichte erst nach expliziter Modell-Zoo-Freigabe laden."""
        if self._checkpoint_path is None:
            return True  # Expliziter Pfad: isolierte Tests/Alternative.
        try:
            from backend.core.model_zoo_registry import get_model

            entry = get_model("cantus")
            return entry is not None and entry.status == "active"
        except Exception as exc:
            logger.warning("§V6 (copilot-instructions.md) Cantus-Freigabestatus nicht lesbar: %s", exc)
            return False

    def _activate_fallback(self, reason: str) -> None:
        """§V6 (copilot-instructions.md): Ersatzpfad NUR mit Warnung + Begründung."""
        self._fallback_active = True
        self._fallback_reason = reason
        logger.warning(
            "§V6 (copilot-instructions.md) ML→DSP-Ersatzpfad: Cantus nicht verfügbar (%s) — Wiener-Fallback",
            reason,
        )

    def _try_load_model(self) -> None:
        if not self._production_qualified():
            self._activate_fallback("Training/Release-Evidenz fehlt; Model-Zoo-Status ist nicht active")
            return
        if _ml_budget_try_allocate is not None:
            if not _ml_budget_try_allocate(self._BUDGET_NAME, size_gb=self._BUDGET_SIZE_GB):
                self._activate_fallback("ML-Speicherbudget erschoepft")
                return
        if self._try_load_torch_rocm():
            self._model_loaded = True
            self._inference_backend = "torch_rocm"
            self._register_lifecycle()
            return
        if not _ONNX_AVAILABLE:
            self._activate_fallback("weder Torch-ROCm noch onnxruntime verfügbar")
            return
        if not self._onnx_path.exists():
            self._activate_fallback(f"ONNX-Gewichte fehlen: {self._onnx_path}")
            return
        try:
            # §III.9 (copilot-instructions.md): ONNX ist ausschließlich der
            # paritätsverifizierte CPU-Fallback; GPU-Inferenz läuft über Torch.
            self._ort_session = ort.InferenceSession(str(self._onnx_path), providers=["CPUExecutionProvider"])
            self._model_loaded = True
            self._inference_backend = "onnx_cpu"
            logger.info(
                "Cantus geladen: %s (%s, %.1f MB)",
                self._onnx_path.name,
                self._ort_session.get_providers()[0],
                self._onnx_path.stat().st_size / 1e6,
            )
            self._register_lifecycle()
        except Exception as exc:
            self._activate_fallback(f"Ladefehler: {exc}")

    def _register_lifecycle(self) -> None:
        """Registriert den aktiven ML-Kern beim zentralen Speicher-Lifecycle."""
        if _PLM_AVAILABLE and _plm_register is not None:
            try:
                _plm_register(self._BUDGET_NAME, size_gb=self._BUDGET_SIZE_GB, unload_fn=self.unload)
            except Exception:
                logger.debug("Cantus PLM-Registrierung fehlgeschlagen", exc_info=True)

    def _try_load_torch_rocm(self) -> bool:
        """Lädt den paritätsprüfbaren Cantus-Kern auf ROCm, sonst False.

        ONNX-ROCm bleibt wegen der projektnormativen EP-Politik ausgeschlossen;
        der identische Torch-Kern wird gegen ONNX-CPU geprüft und ist damit der
        deterministische GPU-Primärpfad (§III.9, §G5 copilot-instructions.md).
        """
        if not _TORCH_AVAILABLE or self._checkpoint_path is None or not self._checkpoint_path.is_file():
            return False
        try:
            if not torch.cuda.is_available():
                return False
            from models.cantus.cantus_model import create_cantus  # pylint: disable=import-outside-toplevel

            config_path = self._checkpoint_path.parent / "cantus_config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))["model"]["full"]
            model = create_cantus(**{key: value for key, value in config.items() if not key.startswith("_")})
            checkpoint = torch.load(self._checkpoint_path, map_location="cpu", weights_only=True)
            model.load_state_dict(checkpoint["model_state_dict"])
            self._torch_model = model.to("cuda").eval()
            logger.info(
                "Cantus Torch-ROCm geladen: %s (%s, %.1fM Parameter)",
                self._checkpoint_path.name,
                torch.cuda.get_device_name(0),
                sum(parameter.numel() for parameter in self._torch_model.parameters()) / 1e6,
            )
            return True
        except Exception as exc:  # pylint: disable=broad-except
            self._torch_model = None
            logger.warning(
                "§V6 (copilot-instructions.md) Cantus-Torch-ROCm nicht verfügbar (%s) — ONNX-CPU-Fallback",
                exc,
            )
            return False

    def unload(self) -> None:
        """PLM-Eviction-Callback: ONNX-Session aus dem RAM entfernen."""
        self._ort_session = None
        self._torch_model = None
        self._inference_backend = "none"
        if _TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._model_loaded = False
        if _ml_budget_release is not None:
            try:
                _ml_budget_release(self._BUDGET_NAME)
            except Exception:
                logger.debug("Budget-Release fehlgeschlagen (Bug 9/V74)", exc_info=True)
        logger.debug("Cantus entladen")

    # ── Preprocessing: Resample → 48 kHz, Normalize, DC-Remove ──────────

    @staticmethod
    def _normalize_layout(audio: np.ndarray) -> tuple[np.ndarray, str]:
        """Layout-Grenze (Stereo-Layout-Invariante): alles → channels-first (C, N)."""
        arr = np.asarray(audio, dtype=np.float32)
        if arr.ndim == 1:
            return arr[np.newaxis, :], "mono"
        if arr.ndim == 2:
            if arr.shape[0] <= 8 and arr.shape[1] > arr.shape[0]:
                return arr, "channels_first"  # (C, N) — Pipeline-internes Layout
            return arr.T, "samples_first"  # (N, C) → (C, N)
        raise ValueError(f"Unsupported audio shape: {arr.shape}")

    def _preprocess(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, dict]:
        """DC-Remove + Resample auf 48 kHz + Peak-Normalisierung (deterministisch)."""
        channels, layout = self._normalize_layout(audio)
        channels = channels - channels.mean(axis=1, keepdims=True)  # DC-Remove
        if sr != self._MODEL_SR:
            from scipy import signal  # lokaler Import: Resample-Only-Pfad

            g = np.gcd(sr, self._MODEL_SR)
            channels = np.stack(
                [signal.resample_poly(ch, self._MODEL_SR // g, sr // g).astype(np.float32) for ch in channels],
                axis=0,
            )
        peak = float(np.max(np.abs(channels))) + 1e-10
        return (channels / peak).astype(np.float32), {"layout": layout, "gain": peak, "sr_in": sr}

    @staticmethod
    def _recombine_vocal_delta(mix: np.ndarray, vocal_before: np.ndarray, vocal_after: np.ndarray) -> np.ndarray:
        """Führt ausschließlich die geprüfte Vocal-Änderung in den Mix zurück.

        Der Instrumentalanteil des Mixes bleibt bit-identisch. Der Stem darf
        mono oder stereo vorliegen; seine mittlere Änderung wird phasengleich
        auf beide Mix-Kanäle gelegt.
        """
        mix_channels, mix_layout = CantusPlugin._normalize_layout(mix)
        before_channels, _ = CantusPlugin._normalize_layout(vocal_before)
        after_channels, _ = CantusPlugin._normalize_layout(vocal_after)
        if before_channels.shape[1] != mix_channels.shape[1] or after_channels.shape[1] != mix_channels.shape[1]:
            raise ValueError("Vocal-Stem und Mix müssen dieselbe Sample-Länge haben")
        delta = after_channels.mean(axis=0) - before_channels.mean(axis=0)
        return CantusPlugin._restore_layout(mix_channels + delta[np.newaxis, :], mix_layout)

    # ── Conditions: MERT + Pitch (FCPE/CREPE) + MuQ-MuLan ───────────

    def _warn_once(self, key: str, message: str, *args: Any) -> None:
        if key in self._warned:
            return
        self._warned.add(key)
        logger.warning("§V6 (VERBOTEN.md) ML→Ersatzpfad: " + message, *args)

    def _extract_conditions(self, mono: np.ndarray) -> dict[str, np.ndarray]:
        """Musikalische Konditionierung auf dem degradierten Signal.

        MERT-v1-330M (1024-d frame-level) + Pitch-Track (FCPE → CREPE) +
        Harmonic Context (MuQ-MuLan). Fehlt MERT, nutzt das Modell die
        gelernten Null-Tokens (use_cond=0, trainiert über Bedingungs-Dropout).
        """
        mono = np.asarray(mono, dtype=np.float32).reshape(-1)
        mert = None
        if not self._mert_init_failed:
            try:
                if self._mert_extractor is None:
                    with self._mert_lock:
                        if self._mert_extractor is None and not self._mert_init_failed:
                            from backend.core.mert_feature_extractor import MERTFeatureExtractor

                            try:
                                self._mert_extractor = MERTFeatureExtractor()
                            except Exception:
                                self._mert_init_failed = True
                                raise
                if self._mert_extractor is not None:
                    mert = np.asarray(self._mert_extractor.extract(mono, self._MODEL_SR), dtype=np.float32)
            except Exception as exc:
                self._warn_once("MERT", "MERT-Feature-Encoder fehlt (%s) — Null-Tokens (use_cond=0)", exc)

        harm = None
        try:
            from plugins.muq_mulan_plugin import (
                extract_muq_mulan_embedding,
            )

            emb = extract_muq_mulan_embedding(mono, self._MODEL_SR)
            harm = None if emb is None else np.asarray(emb, dtype=np.float32).reshape(-1)
        except Exception as exc:
            self._warn_once("MuQ-MuLan", "Harmonic Context fehlt (%s) — Null-Vektor", exc)

        if mert is None:
            return {
                "mert": np.zeros((1, 1024), dtype=np.float32),
                "pitch": np.zeros((1, 2), dtype=np.float32),
                "harm": np.zeros((768,), dtype=np.float32),
                "use_cond": np.asarray(0.0, dtype=np.float32),
            }

        n_frames = int(mert.shape[0])
        pitch = np.zeros((n_frames, 2), dtype=np.float32)
        try:
            try:
                from plugins.fcpe_plugin import analyze_pitch
            except Exception:
                from plugins.crepe_plugin import analyze_pitch
            result = analyze_pitch(mono, self._MODEL_SR)
            f0 = np.asarray(result.f0_hz, dtype=np.float32)
            voiced = np.asarray(result.voiced_prob, dtype=np.float32)
            if f0.size > 0:
                src = np.linspace(0.0, 1.0, num=f0.size, dtype=np.float32)
                dst = np.linspace(0.0, 1.0, num=n_frames, dtype=np.float32)
                f0_i = np.interp(dst, src, f0).astype(np.float32)
                v_i = np.interp(dst, src, voiced).astype(np.float32)
                log2f0 = np.log2(np.maximum(f0_i, 1e-3))
                f0_norm = np.clip((log2f0 - np.log2(50.0)) / (np.log2(1200.0) - np.log2(50.0)), 0.0, 1.0)
                pitch = np.stack([np.where(v_i > 0.45, f0_norm, 0.0).astype(np.float32), v_i], axis=-1)
        except Exception as exc:
            self._warn_once("Pitch", "Pitch-Track fehlt (%s) — Null-Track", exc)

        return {
            "mert": np.ascontiguousarray(mert, dtype=np.float32),
            "pitch": np.ascontiguousarray(pitch, dtype=np.float32),
            "harm": np.ascontiguousarray(
                harm if harm is not None else np.zeros((768,), dtype=np.float32), dtype=np.float32
            ),
            "use_cond": np.asarray(1.0, dtype=np.float32),
        }

    # ── Inferenz ────────────────────────────────────────────────────────

    def _run_ort_with_cpu_fallback(self, feeds: dict[str, np.ndarray]) -> list[np.ndarray[Any, Any]]:
        """Führt den ONNX-CPU-Fallback aus und initialisiert ihn bei Bedarf (§V6 (VERBOTEN.md))."""
        try:
            if self._ort_session is None:
                self._ort_session = ort.InferenceSession(  # type: ignore[union-attr]
                    str(self._onnx_path), providers=["CPUExecutionProvider"]
                )
            outputs = self._ort_session.run(None, feeds)  # type: ignore[union-attr]
            return cast(list[np.ndarray[Any, Any]], outputs)
        except Exception as exc:
            logger.warning(
                "§V6 (VERBOTEN.md) Cantus-ONNX-CPU-Inferenz fehlgeschlagen (%s) — CPU-Session wird neu aufgebaut",
                exc,
            )
            self._ort_session = ort.InferenceSession(  # type: ignore[union-attr]
                str(self._onnx_path), providers=["CPUExecutionProvider"]
            )
            outputs = self._ort_session.run(None, feeds)
            return cast(list[np.ndarray[Any, Any]], outputs)

    def _run_torch_rocm(self, feeds: dict[str, np.ndarray]) -> np.ndarray:
        """Führt den Cantus-Primärpfad deterministisch auf Torch-ROCm aus."""
        if self._torch_model is None:
            raise RuntimeError("Cantus-Torch-ROCm-Kern ist nicht geladen")
        with torch.inference_mode():
            tensors = {name: torch.from_numpy(value).to("cuda") for name, value in feeds.items()}
            velocity = self._torch_model(
                tensors["x"],
                tensors["t"],
                tensors["mert"],
                tensors["pitch"],
                tensors["harm"],
                tensors["use_cond"],
            )
        return cast(
            np.ndarray[Any, Any],
            np.nan_to_num(velocity.detach().float().cpu().numpy(), nan=0.0, posinf=0.0, neginf=0.0),
        )

    def _restore_single(self, mono: np.ndarray) -> np.ndarray:
        """Flow-Matching-Schritt (trainingskonform): ŷ = x + (1−t)·v̂, t = 0.5."""
        x = mono.reshape(1, -1, 1).astype(np.float32)
        cond = self._extract_conditions(mono)
        self._last_use_cond = float(cond["use_cond"])
        feeds = {
            "x": x,
            "t": np.asarray([self._FLOW_TIME], dtype=np.float32),
            "mert": cond["mert"][np.newaxis, ...],
            "pitch": cond["pitch"][np.newaxis, ...],
            "harm": cond["harm"][np.newaxis, ...],
            "use_cond": cond["use_cond"][np.newaxis],
        }
        if self._torch_model is not None:
            try:
                velocity = self._run_torch_rocm(feeds)
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning(
                    "§V6 (copilot-instructions.md) Cantus-Torch-ROCm-Inferenz fehlgeschlagen (%s) — ONNX-CPU-Fallback",
                    exc,
                )
                self._torch_model = None
                self._inference_backend = "onnx_cpu"
                velocity = np.asarray(self._run_ort_with_cpu_fallback(feeds)[0], dtype=np.float32)
        else:
            velocity = np.asarray(self._run_ort_with_cpu_fallback(feeds)[0], dtype=np.float32)
        restored = x + (1.0 - self._FLOW_TIME) * velocity.reshape(1, -1, 1)
        result: np.ndarray[Any, Any] = np.asarray(restored.reshape(-1), dtype=np.float32)
        return result

    def _restore_chunked(self, mono: np.ndarray) -> np.ndarray:
        """Overlap-Add-Chunks via Aurik-ChunkedPipeline (lange Songs)."""
        try:
            from backend.core.chunked_streaming import (
                ChunkedPipeline,
                ChunkResult,
            )

            cp = ChunkedPipeline(chunk_duration_s=4.0, overlap_s=0.5, crossfade_s=0.05)
            results = []
            for i, (start, end) in enumerate(cp.compute_chunks(mono, self._MODEL_SR)):
                results.append(ChunkResult(self._restore_single(mono[start:end]), self._MODEL_SR, i, start, end))
            return cp.collect_results(results, self._MODEL_SR)
        except ImportError as exc:
            self._warn_once("Chunked", "ChunkedPipeline fehlt (%s) — Direktverarbeitung", exc)
            return self._restore_single(mono)

    # ── Guards ───────────────────────────────────────────────────────────

    @staticmethod
    def _spectral_novelty(before: np.ndarray, after: np.ndarray, sr: int) -> float:
        """Hallucination-Guard (§2.46e im Miipher-Präzedenzpfad): > 0.35 → Rollback."""
        try:
            n_fft = 2048
            spec_b = np.abs(np.fft.rfft(before[: min(len(before), sr * 5)], n=n_fft))
            spec_a = np.abs(np.fft.rfft(after[: min(len(after), sr * 5)], n=n_fft))
            spec_b = spec_b / (np.max(spec_b) + 1e-10)
            spec_a = spec_a / (np.max(spec_a) + 1e-10)
            return float(np.clip(np.mean(np.abs(spec_a - spec_b)), 0.0, 1.0))
        except Exception:
            logger.warning("_spectral_novelty fehlgeschlagen — neutraler Return 0.0")
            return 0.0

    def _singmos_scores(self, before: np.ndarray, after: np.ndarray) -> tuple[float | None, float | None]:
        """SingMOS-Messung (Quality-Predictors) fuer den Never-worsen-Delta-Guard."""
        try:
            from backend.core.dsp.quality_predictors import (
                get_singmos_predictor,
            )

            predictor = get_singmos_predictor()
            return (
                float(predictor.predict(before, self._MODEL_SR)),
                float(predictor.predict(after, self._MODEL_SR)),
            )
        except Exception as exc:
            self._warn_once("SingMOS", "SingMOS-Messung fehlt (%s) — Guard uebersprungen", exc)
            return None, None

    # ── Haupt-API ────────────────────────────────────────────────────────

    def should_apply(self, material: str, restorability_score: float = 50.0) -> bool:
        """Material-Gate: nur stark degradierte Quellen (Miipher-Präzedenz)."""
        return str(material).lower() in self._TARGET_MATERIALS or restorability_score < 30

    def _dsp_fallback_output(self, audio: np.ndarray, sr: int, vocal_stem: np.ndarray | None) -> np.ndarray:
        """Wendet den Ersatzpfad stem-first an und erhält den Instrumentalanteil."""
        restored = self._dsp_fallback(vocal_stem if vocal_stem is not None else audio, sr)
        if vocal_stem is not None:
            restored = self._recombine_vocal_delta(audio, vocal_stem, restored)
        return restored

    def enhance(
        self,
        audio: np.ndarray,
        sr: int,
        material: str = "unknown",
        *,
        restorability_score: float = 50.0,
        vocal_stem: np.ndarray | None = None,
    ) -> CantusResult:
        """Cantus-Gesangsrestaurierung mit allen Guards.

        Args:
            audio: mono (N,) oder Stereo — beide Layouts (N, 2)/(2, N) bedient
                   (§Stereo-Layout-Invariante (copilot-instructions.md)).
            sr: 44100 oder 48000 (intern immer 48 kHz).
        """
        t0 = __import__("time").perf_counter()
        if sr not in (44100, 48000):
            raise ValueError(f"Cantus erwartet 44.1/48 kHz Input, bekam {sr} Hz")
        if not self.should_apply(material, restorability_score):
            return CantusResult(
                audio=audio, applied=False, model_used="none", metadata={"reason": "material_not_target"}
            )
        if self._fallback_active or not self._model_loaded:
            restored = self._dsp_fallback_output(audio, sr, vocal_stem)
            return CantusResult(
                audio=restored,
                applied=True,
                model_used="dsp_fallback",
                processing_time_s=__import__("time").perf_counter() - t0,
                metadata={"fallback_reason": self._fallback_reason},
            )

        model_source = vocal_stem if vocal_stem is not None else audio
        channels, ctx = self._preprocess(model_source, sr)  # channels-first (C, N) @ 48 kHz, peak-norm.
        is_stereo = channels.shape[0] == 2
        mid = channels.mean(axis=0) if is_stereo else channels[0]
        side = (channels[0] - channels[1]) / 2.0 if is_stereo else None

        try:
            restored = self._restore_chunked(mid) if mid.size / self._MODEL_SR > 10.0 else self._restore_single(mid)
        except Exception as exc:
            self._activate_fallback(f"ORT-Inferenz auch auf CPU fehlgeschlagen: {exc}")
            return CantusResult(
                audio=self._dsp_fallback_output(audio, sr, vocal_stem),
                applied=True,
                model_used="dsp_fallback",
                processing_time_s=__import__("time").perf_counter() - t0,
                metadata={
                    "fallback_reason": self._fallback_reason,
                    "input_source": "vocal_stem" if vocal_stem is not None else "mix_mid",
                },
            )
        restored = restored[: mid.size]
        if restored.size < mid.size:  # §0a (copilot-instructions.md): Längen-Integrität
            restored = np.pad(restored, (0, mid.size - restored.size))

        novelty = self._spectral_novelty(mid, restored, self._MODEL_SR)
        mos_before, mos_after = self._singmos_scores(mid, restored)
        model_used = "cantus"
        if novelty > self._HALLUCINATION_THRESHOLD:
            self._warn_once(
                "Hallucination",
                "Hallucination-Guard novelty=%.3f > %.2f — Rollback",
                novelty,
                self._HALLUCINATION_THRESHOLD,
            )
            restored, model_used = mid, "none"
        elif mos_before is not None and mos_after is not None and mos_after < mos_before - self._SINGMOS_MARGIN:
            self._warn_once(
                "SingMOSRegression",
                "Never-worsen verletzt (SingMOS %.2f → %.2f) — Rollback (delta-basierter Guard)",
                mos_before,
                mos_after,
            )
            restored, model_used = mid, "none"

        restored = (restored * ctx["gain"]).astype(np.float32)  # Gain der Vorverarbeitung rückgängig
        if vocal_stem is not None:
            if model_used == "none":
                out = np.asarray(audio, dtype=np.float32).copy()
            else:
                out = self._recombine_vocal_delta(audio, vocal_stem, restored)
        elif is_stereo and side is not None:
            out = np.stack([restored + side[: restored.size], restored - side[: restored.size]], axis=0)
        else:
            out = restored[np.newaxis, :]
        if vocal_stem is None:
            out = self._restore_layout(out, ctx["layout"])
        return CantusResult(
            audio=out,
            applied=model_used != "none",
            model_used=model_used,
            novelty=novelty,
            singmos_before=mos_before,
            singmos_after=mos_after,
            processing_time_s=__import__("time").perf_counter() - t0,
            metadata={
                "use_cond": float(self._last_use_cond),
                "sr_in": sr,
                "input_source": "vocal_stem" if vocal_stem is not None else "mix_mid",
                "inference_backend": self._inference_backend,
            },
        )

    @staticmethod
    def _restore_layout(channels: np.ndarray, layout: str) -> np.ndarray:
        """Ausgabe im Eingangs-Layout (§Stereo-Layout-Invariante (copilot-instructions.md))."""
        if layout == "mono":
            return cast(np.ndarray[Any, Any], np.asarray(channels[0], dtype=np.float32))
        return channels.T if layout == "samples_first" else channels  # (N, 2) bzw. (C, N)

    # ── DSP-Ersatzpfad (§V6 (copilot-instructions.md)) ───────────────

    @staticmethod
    def _dsp_fallback(audio: np.ndarray, sr: int) -> np.ndarray:
        """Konservativer Wiener-Ersatzpfad — Primum non nocere, deterministisch."""
        try:
            from scipy import signal

            channels = CantusPlugin._normalize_layout(audio)[0]
            out = []
            for ch in channels:
                _f, _t, zxx = signal.stft(ch, fs=sr, nperseg=2048, noverlap=1024)
                noise_floor = np.mean(np.abs(zxx[:, :10]), axis=1, keepdims=True)
                gain = np.clip(np.maximum(0.0, 1.0 - noise_floor / (np.abs(zxx) + 1e-10)), 0.3, 1.0)
                _, rec = signal.istft(zxx * gain, fs=sr, nperseg=2048, noverlap=1024)
                out.append(rec[: len(ch)].astype(np.float32))
            return CantusPlugin._restore_layout(np.stack(out), CantusPlugin._normalize_layout(audio)[1])
        except Exception:
            logger.warning("DSP-Ersatzpfad fehlgeschlagen — Originalsignal (Primum non nocere)")
            return audio


# ── Singleton (PLM-kompatibel) ───────────────────────────────────────────

_instance: CantusPlugin | None = None


def get_cantus() -> CantusPlugin:
    """Process-weite Cantus-Singleton (Lazy-Load, PLM-kompatibel)."""
    global _instance
    if _instance is None:
        _instance = CantusPlugin()
    return _instance


def repair_vocal_cantus(
    audio: np.ndarray,
    sr: int,
    material: str = "unknown",
    *,
    restorability_score: float = 50.0,
) -> CantusResult:
    """Convenience-Wrapper für die Cantus-Gesangsrestaurierung."""
    return get_cantus().enhance(audio, sr, material, restorability_score=restorability_score)
