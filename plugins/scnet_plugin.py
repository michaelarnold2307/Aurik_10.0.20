"""SCNet-4-Stems-Separation (§P1-2 / TODO-P1-2) — Musik-Tier hinter Never-worsen.

**Kanonische SCNet-Implementierung** (§G9 copilot-instructions.md — *eine* Quelle
für Produktion und A/B-Harness `scripts/eval_scnet_vs_mdx23c.py`). Vorher trug das
Eval-Skript eine eigene Kopie von Laden und Inferenz; das ist der §G9-Verstoß,
den dieses Modul behebt.

Artefakt
--------
``models/scnet_4stems/huge_scnet_4stems_v1.2.ckpt`` — Aname-Tommy/Huge-SCNet-4stems
(Apache-2.0, SHA-256 ``807f470b…``, 65.292.464 Parameter, Strict-Load 0 missing /
0 unexpected). Architektur: vendored ZFTurbo-MSST ``models/scnet`` (MIT,
unverändert) unter ``models/scnet_4stems/zfturbo_scnet/``.

Kontrakt
--------
* **Modellrate 44.100 Hz** (``config.yaml`` → ``audio.sample_rate``); die Pipeline
  läuft mit 48 kHz ⇒ 48 k → 44,1 k → 48 k über den kanonischen
   ``plugins.htdemucs_plugin.resample_audio`` (§G9 (copilot-instructions.md) — kein zweiter Resampler).
* **Quellen-Reihenfolge** kommt aus ``config.yaml`` (``drums, bass, other, vocals``)
  und wird nie geraten.
* **Nur bei Freigabe**: ``music_model_flags.use_scnet_music`` (Default ``False``).
  Ohne Freigabe wird weder Checkpoint noch Torch-Zweig geladen (§III.11 +
  §V7 copilot-instructions.md: kein blindes Aktivieren) — `separate()` liefert
  dann ``None`` mit §V6-Begründung.
* **Deterministisch** (§G5 copilot-instructions.md): fester Seed beim
  Architektur-Aufbau, CPU, ``torch.no_grad()``.
* **Chunking**: Der Checkpoint ist auf 15-s-Fenster trainiert
  (``audio.chunk_size = 661500``). Die Produktion chunkt daher auf dieser Länge
  mit 0,5-s-Kreuzblende (≥ 200 ms Hanning-Regel, §G3/§V2 copilot-instructions.md);
  der A/B-Harness ruft ``chunked=False`` (ganzes Segment), weil die
  Referenzmessung so gemessen wurde. **Vor einer Aktivierung ist der RT-Beitrag
  gegen die Budget-Tabelle zu messen** (§9.5).
"""

from __future__ import annotations

import logging
import os
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_VENDOR_PARENT = _PROJECT_ROOT / "models" / "scnet_4stems"
_CONFIG_PATH = _VENDOR_PARENT / "config.yaml"

_MODEL_SR = 44100  # config.yaml → audio.sample_rate
_CHUNK_SAMPLES = 661500  # config.yaml → audio.chunk_size (15 s @ 44,1 kHz)
_CROSSFADE_SAMPLES = 22050  # 0,5 s @ 44,1 kHz (≥ 200 ms, §G3/§V2 (copilot-instructions.md))
_DEFAULT_SOURCES: tuple[str, ...] = ("drums", "bass", "other", "vocals")

# Der Checkpoint-Pickle referenziert bitsandbytes (Trainings-Umgebung); für reines
# Inferenz-Laden genügen Dummy-Module (Muster: scripts/_scnet_arch_probe.py).
_BITSANDBYTES_MODULES: tuple[str, ...] = (
    "bitsandbytes",
    "bitsandbytes.optim",
    "bitsandbytes.optim.adamw",
    "bitsandbytes.nn",
    "bitsandbytes.functional",
    "bitsandbytes.cextension",
    "bitsandbytes.triton",
)


def stub_bitsandbytes() -> None:
    """Stubbt ``bitsandbytes`` für ``torch.load`` (Trainings-Artefakt, kein Laufzeitpfad)."""
    if "bitsandbytes" in sys.modules:
        return

    class _Module(types.ModuleType):
        def __getattr__(self, name: str) -> Any:
            if name.startswith("__"):
                raise AttributeError(name)
            attr = type(name, (), {"__init__": lambda self, *a, **k: None})
            setattr(self, name, attr)
            return attr

    for name in _BITSANDBYTES_MODULES:
        sys.modules[name] = _Module(name)


def ensure_vendor_importable() -> None:
    """Legt den vendored ZFTurbo-SCNet-Code (MIT) auf ``sys.path``."""
    if str(_VENDOR_PARENT) not in sys.path:
        sys.path.insert(0, str(_VENDOR_PARENT))


def _as_stereo(audio: np.ndarray) -> np.ndarray:
    """Bringt Audio auf das SCNet-Eingangslayout ``(2, N)`` float32.

    Die Pipeline ist intern **channels-first (C, N)**; jede Modul-Grenze
    normalisiert explizit (Layout-Invariante aus ``AGENTS.md`` §3).
    """
    arr = np.asarray(audio, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.ndim == 1:
        stacked: np.ndarray = np.stack([arr, arr])
        return stacked
    if arr.shape[1] == 2 and arr.shape[0] != 2:
        transposed: np.ndarray = np.ascontiguousarray(arr.T)  # (N, 2) → (2, N)
        return transposed
    if arr.shape[0] == 2:
        channels_first: np.ndarray = np.ascontiguousarray(arr)
        return channels_first
    mono = arr.mean(axis=1) if arr.shape[1] > 1 else arr[:, 0]
    duplicated: np.ndarray = np.stack([mono, mono])
    return duplicated


class SCNetPlugin:
    """SCNet-4-Stems-Kandidat (Musik-Tier) — lazy geladen, CPU-deterministisch.

    ``separate()`` liefert ``None``, wenn das Flag gesperrt ist oder das Artefakt
    fehlt; jeder Ausfall wird nach §V6 (copilot-instructions.md) mit Begründung
    protokolliert (kein stilles Degradieren).
    """

    def __init__(self) -> None:
        self._model: Any | None = None
        self._sources: tuple[str, ...] = _DEFAULT_SOURCES
        self._resolved: Path | None = None

    # ── Auflösung / Status ────────────────────────────────────────────────
    def model_path(self) -> Path | None:
        """Aktiver Artefaktpfad über die kanonische Flag-Auflösung (§G9 (copilot-instructions.md))."""
        from backend.core.music_model_flags import resolve_model_path

        return resolve_model_path("scnet")

    def is_available(self) -> bool:
        """Wahr, wenn Flag gesetzt **und** Artefakt vorhanden ist (kein Laden)."""
        path = self.model_path()
        return path is not None and path.exists()

    @property
    def sources(self) -> tuple[str, ...]:
        return self._sources

    @property
    def resolved_path(self) -> Path | None:
        return self._resolved

    # ── Laden ─────────────────────────────────────────────────────────────
    def _ensure_loaded(self) -> bool:
        if self._model is not None:
            return True
        path = self.model_path()
        if path is None or not path.exists():
            logger.warning(
                "§V6 (copilot-instructions.md) SCNet nicht verfügbar (Flag aus oder Artefakt fehlt: %s) — Musik-Tier bleibt aus "
                "(§V7 copilot-instructions.md: kein Blind-Aktivieren).",
                path,
            )
            return False
        try:
            stub_bitsandbytes()
            ensure_vendor_importable()
            import torch  # lokaler Import: Torch ist optional (schwer)
            import yaml
            from zfturbo_scnet import SCNet  # type: ignore[import-not-found]

            with open(_CONFIG_PATH, encoding="utf-8") as fh:
                cfg_model = yaml.safe_load(fh)["model"]

            torch.manual_seed(0)  # §G5 (copilot-instructions.md): deterministischer Architektur-Aufbau
            model = SCNet(**cfg_model)
            checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
            state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing or unexpected:
                raise RuntimeError(
                    f"SCNet-Checkpoint passt nicht strikt zur Architektur: "
                    f"missing={len(missing)} unexpected={len(unexpected)}"
                )
            model.eval()
            torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
            self._model = model
            self._sources = tuple(cfg_model.get("sources", _DEFAULT_SOURCES))
            self._resolved = path
            logger.info(
                "SCNet geladen (§P1-2): %s Parameter, Quellen=%s, Rate=%d Hz",
                f"{sum(p.numel() for p in model.parameters()):,}",
                list(self._sources),
                _MODEL_SR,
            )
            return True
        except Exception as exc:  # pylint: disable=broad-except — §V6 (copilot-instructions.md): nie still degradieren
            logger.warning(
                "§V6 (copilot-instructions.md) SCNet-Laden fehlgeschlagen (%s) — Musik-Tier bleibt aus.", exc
            )
            self._model = None
            return False

    # ── Inferenz ──────────────────────────────────────────────────────────
    def _infer_once(self, chunk_2ch: np.ndarray) -> dict[str, np.ndarray]:
        """Ein Vorwärtslauf: ``(2, T)`` @ 44,1 kHz → ``{source: (2, T)}``."""
        import torch

        model = self._model
        if model is None:
            raise RuntimeError("SCNet-Modell nicht geladen (Aufruf ohne _ensure_loaded)")
        tensor = torch.from_numpy(np.ascontiguousarray(chunk_2ch, dtype=np.float32)).unsqueeze(0)
        with torch.no_grad():
            out = model(tensor)
        arr = np.asarray(out[0].detach().cpu().numpy(), dtype=np.float32)  # (S, 2, T)
        return {name: arr[idx] for idx, name in enumerate(self._sources)}

    def _infer_chunked(self, work_2ch: np.ndarray) -> dict[str, np.ndarray]:
        """Chunked Inferenz auf ``chunk_size`` mit Kreuzblende (§G3/§V2 (copilot-instructions.md))."""
        total = work_2ch.shape[1]
        if total <= _CHUNK_SAMPLES:
            return self._infer_once(work_2ch)

        overlap = min(_CROSSFADE_SAMPLES, _CHUNK_SAMPLES // 4)
        step = max(1, _CHUNK_SAMPLES - overlap)
        ramp = np.linspace(0.0, 1.0, overlap, endpoint=False, dtype=np.float32)
        num = {name: np.zeros((2, total), dtype=np.float64) for name in self._sources}
        den = np.zeros(total, dtype=np.float64)

        for start in range(0, total, step):
            end = min(start + _CHUNK_SAMPLES, total)
            seg_len = end - start
            chunk = np.zeros((2, _CHUNK_SAMPLES), dtype=np.float32)
            chunk[:, :seg_len] = work_2ch[:, start:end]
            stems = self._infer_once(chunk)
            weight = np.ones(seg_len, dtype=np.float32)
            if start > 0:
                weight[:overlap] = ramp
            if end < total:
                weight[-overlap:] = ramp[::-1]
            for name in self._sources:
                num[name][:, start:end] += stems[name][:, :seg_len] * weight
            den[start:end] += weight
            if end >= total:
                break

        den = np.maximum(den, 1e-8)
        return {name: (num[name] / den).astype(np.float32) for name in self._sources}

    def separate(
        self,
        audio: np.ndarray,
        sr: int,
        *,
        chunked: bool = True,
    ) -> dict[str, np.ndarray] | None:
        """Trennt ``audio`` in die 4 SCNet-Stems ``(N, 2)`` float32.

        Args:
            audio:   mono ``(N,)``, ``(N, 2)`` oder ``(2, N)``
            sr:      Abtastrate des Eingangs (Pipeline: 48 000)
            chunked: ``True`` = produktiver 15-s-Chunked-Pfad,
                     ``False`` = ganzes Segment (A/B-Referenzbedingung)

        Returns:
            ``{drums, bass, other, vocals}`` als ``(N, 2)`` float32 in Länge und
            Rate der Eingabe — oder ``None`` ohne Freigabe/bei Ausfall (§V6 (copilot-instructions.md)).
        """
        if not self.is_available():
            from backend.core import music_model_flags as _flags

            if getattr(_flags, "use_scnet_music", False):
                logger.warning(
                    "§V6 (copilot-instructions.md) SCNet-Kandidat nicht ausführbar: Artefakt nicht auflösbar (%s) — None, Bestand bleibt.",
                    self.model_path(),
                )
            else:
                logger.info(
                    "SCNet-Tier inaktiv (use_scnet_music=False) — None, Bestand bleibt (§V7 (copilot-instructions.md))."
                )
            return None
        if not self._ensure_loaded():
            return None

        src = _as_stereo(audio)
        length = int(src.shape[1])
        from plugins.htdemucs_plugin import resample_audio  # §G9 (copilot-instructions.md): EIN Resampler

        work = resample_audio(np.ascontiguousarray(src, dtype=np.float32), int(sr), _MODEL_SR)
        work = np.ascontiguousarray(np.asarray(work, dtype=np.float32))
        try:
            estimate = self._infer_chunked(work) if chunked else self._infer_once(work)
        except Exception as exc:  # pylint: disable=broad-except — §V6 (copilot-instructions.md): Grund nennen
            logger.warning(
                "§V6 (copilot-instructions.md) SCNet-Inferenz fehlgeschlagen (%s) — Musik-Tier bleibt aus.", exc
            )
            return None

        out: dict[str, np.ndarray] = {}
        for name in self._sources:
            back = np.asarray(resample_audio(estimate[name], _MODEL_SR, int(sr)), dtype=np.float32)
            arr = _fit_length(back, length)  # Resampling ändert die Länge rational
            out[name] = np.clip(np.nan_to_num(arr.T, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0).astype(np.float32)
        return out


def _fit_length(audio_2ch: np.ndarray, length: int) -> np.ndarray:
    """Trimmt/Polstert ``(2, T)`` exakt auf ``length`` (Resampling-Längen-Drift)."""
    total = audio_2ch.shape[1]
    if total == length:
        return audio_2ch
    if total > length:
        return audio_2ch[:, :length]
    pad = np.zeros((audio_2ch.shape[0], length - total), dtype=audio_2ch.dtype)
    padded: np.ndarray = np.concatenate([audio_2ch, pad], axis=1)
    return padded


_INSTANCE: SCNetPlugin | None = None


def get_scnet_plugin() -> SCNetPlugin:
    """Singleton-Zugriff (§G9 (copilot-instructions.md) — eine Instanz, ein Checkpoint im Speicher)."""
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = SCNetPlugin()
    return _INSTANCE


def reset_scnet_plugin() -> None:
    """Verwirft die Singleton-Instanz (Song-Isolation §V8/§G1 (copilot-instructions.md), Tests)."""
    global _INSTANCE
    _INSTANCE = None
