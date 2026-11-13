"""SOTA-M8 (D-K4-1): Neuronaler A-SPADE-Klassen-Declipper — CHALLENGER zur heuristischen Phase 07.

A-SPADE-Klasse (Gaultier et al., "A Sparsity-based Approach to Deep Declipping",
IEEE/ACM TASLP 2023): unrolled ADMM mit gelerntem 1-D-CNN-Proximaloperator
(deterministische Feed-Forward-Inferenz, kein Sampling) als neuronale Ergänzung
zur PCHIP-Heuristik (`backend/core/phases/phase_07_declipper.py`).

Vertrag:
- **Flag-Gate**: Die Freigabe liegt AUSSCHLIESSLICH beim Aufrufer
  (`use_aspade_declipper`, `backend/core/music_model_flags.py`, Default False).
  False ⇒ bit-identisches Bestandsverhalten (Phase 07 rief diesen Challenger
  nie auf). Dieses Modul prüft das Flag bewusst nicht selbst (§G9 (copilot-instructions.md): eine Gate-Quelle).
- **Determinismus (§G5 (copilot-instructions.md))**: gleicher Input + gleicher
  Seed ⇒ bit-identischer Output. Gewichte deterministisch initialisiert
  (Seeds pro Session, kein `time.time()` in Entscheidungslogik), Inferenz ist
  reines Feed-Forward in fester Op-Reihenfolge.
- **Stereo-Layout-Invariante (AGENTS.md §3)**: kanonisch channels-first (C, N);
  (N,), (C, N) und (N, C) mit C ≤ 2 werden bedient und im Eingangs-Layout
  zurückgegeben. Kanalprozessierung strikt unabhängig ⇒ (2, N) und (N, 2)
  liefern bit-identische Kanalergebnisse.
- **Kandidatenfilter**: Der Kandidat wird pro Kanal nur übernommen, wenn der
  Harmonik-Witness sinkt; sonst bleibt der Eingang bit-identisch. Der Witness
  beweist weder Never-worsen noch Hörgewinn (SOTA-M8-Holdoutbefund).
- **§V6 (copilot-instructions.md) — Silent-Failure-Verbot**: jeder Ersatzpfad
  (fehlendes Artefakt, NaN/Inf, nicht-finites Modell-Ergebnis) wird mit
  `logger.warning` + Begründung protokolliert — kein stiller Fallback.
  Ungültige Argumente (Layout, dtype, Schwelle) werfen `ValueError` (laut).
- **NaN/Inf-Schutz (§0a (copilot-instructions.md))**: Eingänge werden vor der
  Inferenz sanitisiert (mit Warnung).

Modell-Domäne: **Musik** (§III.11 (copilot-instructions.md)) — Deklippen ist
Signalpfad-Kern; das Trainingsartefakt wird auf MUSDB18-HQ-Vocals trainiert
(`scripts/train_aspade_declipper.py`). ONNX-Export + Paritäts-Gate liegen dort
(§III.9 (copilot-instructions.md)).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from backend.core import audio_layout

logger = logging.getLogger(__name__)

# ── Konstanten ───────────────────────────────────────────────────────────────

TARGET_SR = 48_000
CHUNK_SEC = 10.0
DEFAULT_SEED = 42
DEFAULT_STAGES = 4
DEFAULT_CHANNELS = 24
DEFAULT_KERNEL_SIZE = 7
DEFAULT_DILATIONS: tuple[int, ...] = (1, 2, 4, 8)

# Produktionsartefakt des Trainings (scripts/train_aspade_declipper.py).
DEFAULT_CHECKPOINT_SUBPATH = "models/aspade/aspade_class_v1.pt"
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT = _PROJECT_ROOT / DEFAULT_CHECKPOINT_SUBPATH


# ── Hilfsfunktionen (deterministisch) ───────────────────────────────────────


def _estimate_clip_threshold(work: np.ndarray) -> float:
    """Deterministische Clip-Schätzung, wenn kein Aufrufer-Wert vorliegt.

    P99.5 der Absolut-Amplituden über alle Kanäle, begrenzt auf [0.05, 0.999].
    Die Produktionskalibrierung liegt in Phase 07
    (`_histogram_clip_threshold`) — hier nur der Standalone-Fallback.
    """
    p995 = float(np.percentile(np.abs(work), 99.5))
    return float(np.clip(p995, 0.05, 0.999))


def harmonic_distortion_witness(x: np.ndarray, sample_rate: int) -> float:
    """THD-ähnlicher Harmonik-Witness (Grundton 60 Hz–2 kHz, Harmonische 2–10).

    Standalone-Filter für dieses Modul, kein Never-worsen-Nachweis; die
    Router-Entscheidung in Phase 07 nutzt weiterhin die Phase-lokale
    `_harmonic_distortion_proxy` (§G9 (copilot-instructions.md): eine Quelle je
    Entscheidung). Deterministisch, O(N log N).
    """
    x64 = np.asarray(x, dtype=np.float64)
    if x64.size == 0:
        return 0.0
    spec = np.abs(np.fft.rfft(x64))
    freqs = np.fft.rfftfreq(x64.size, 1.0 / max(int(sample_rate), 1))
    band = (freqs >= 60.0) & (freqs <= 2000.0)
    if not bool(band.any()):
        return 0.0
    f0_i = int(np.argmax(spec * band))
    f0 = float(freqs[f0_i])
    fund = float(spec[f0_i]) + 1e-12
    harm = 0.0
    for h in range(2, 11):
        hi = int(np.argmin(np.abs(freqs - f0 * h)))
        if hi != f0_i:
            harm += float(spec[hi] ** 2)
    return float(np.sqrt(harm) / fund)


# ── A-SPADE-Architektur (unrolled ADMM + 1-D-CNN-Proximal) ──────────────────


class _ResConv1d(nn.Module):
    """Residualer dilatierter 1-D-Conv-Block (Längentreue, ONNX-fähig)."""

    def __init__(self, channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        pad = (kernel_size // 2) * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = nn.functional.gelu(self.conv1(x))
        _out: torch.Tensor = x + self.conv2(h)
        return _out


class AspadeProximalNet(nn.Module):
    """T_θ (A-SPADE-Proximaloperator): 1-D-CNN-Stack, Residual-Parameterisierung.

    Eingabe/Ausgabe (B, 1, T) — zeitinvariant, längentreu, deterministisch.
    """

    def __init__(
        self,
        channels: int = DEFAULT_CHANNELS,
        kernel_size: int = DEFAULT_KERNEL_SIZE,
        dilations: tuple[int, ...] = DEFAULT_DILATIONS,
    ) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.inp = nn.Conv1d(1, channels, kernel_size, padding=pad)
        self.blocks = nn.ModuleList([_ResConv1d(channels, kernel_size, d) for d in dilations])
        self.out = nn.Conv1d(channels, 1, kernel_size, padding=pad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = nn.functional.gelu(self.inp(x))
        for block in self.blocks:
            h = block(h)
        _out2: torch.Tensor = x + self.out(h)
        return _out2


class AspadeClassNet(nn.Module):
    """Unrolled-ADMM-Declipper der A-SPADE-Klasse (K Stufen, gelernte Proximale).

    ADMM-Split (PnP/APPLADE-Schema, Zeitbereich):
      x ← Π_Γ(v − u)  (Datenkonsistenz-Projektion, nicht gelernt)
      v ← T_θ,k(x + u) (gelernter 1-D-CNN-Proximaloperator je Stufe k)
      u ← u + x − v
    Ausgabe = finale Projektion Π_Γ(v_K − u_K) — ungeclippte Samples bleiben
    exakt auf dem Eingang (Datenkonsistenz).

    Forward-Vertrag: y (B, 1, T), theta (1, 1, 1) ⇒ x (B, 1, T).
    Deterministisch (§G5 (copilot-instructions.md)), ONNX-exportierbar.
    """

    def __init__(
        self,
        stages: int = DEFAULT_STAGES,
        channels: int = DEFAULT_CHANNELS,
        kernel_size: int = DEFAULT_KERNEL_SIZE,
        dilations: tuple[int, ...] = DEFAULT_DILATIONS,
    ) -> None:
        super().__init__()
        if stages < 1:
            raise ValueError(f"stages muss ≥ 1 sein (got {stages})")
        self.stages = nn.ModuleList(
            [AspadeProximalNet(channels=channels, kernel_size=kernel_size, dilations=dilations) for _ in range(stages)]
        )

    @staticmethod
    def _project(z: torch.Tensor, y: torch.Tensor, mask_pos: torch.Tensor, mask_neg: torch.Tensor, theta: torch.Tensor):
        mask_obs = 1.0 - mask_pos - mask_neg
        return mask_obs * y + mask_pos * torch.maximum(z, theta) + mask_neg * torch.minimum(z, -theta)

    def forward(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        mask_pos = (y >= theta).to(dtype=y.dtype)
        mask_neg = (y <= -theta).to(dtype=y.dtype)
        x = y
        v = y
        u = torch.zeros_like(y)
        for proximal in self.stages:
            x = self._project(v - u, y, mask_pos, mask_neg, theta)
            v = proximal(x + u)
            u = u + x - v
        _out3: torch.Tensor = self._project(v - u, y, mask_pos, mask_neg, theta)
        return _out3


def build_model(
    *,
    stages: int = DEFAULT_STAGES,
    channels: int = DEFAULT_CHANNELS,
    kernel_size: int = DEFAULT_KERNEL_SIZE,
    dilations: tuple[int, ...] = DEFAULT_DILATIONS,
    seed: int = DEFAULT_SEED,
) -> AspadeClassNet:
    """Deterministische Gewichts-Initialisierung (§G5 (copilot-instructions.md)).

    Gleicher Seed ⇒ bit-identische Gewichte; der globale RNG-Zustand bleibt
    unangetastet (`torch.random.fork_rng`).
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = AspadeClassNet(
            stages=stages,
            channels=channels,
            kernel_size=kernel_size,
            dilations=dilations,
        )
    model.eval()
    return model


# ── Inferenz (Layout-sicher, chunked, Kandidatenfilter) ───────────────────────


def _to_channels_first(arr: np.ndarray) -> tuple[np.ndarray, str]:
    """Normalisiert auf kanonisches (C, N) float32 und merkt das Eingangs-Layout.

    Layout-Entscheidung ausschließlich über `backend/core/audio_layout.py`
    (§G9 (copilot-instructions.md): Layout-EINE-Quelle, Muster D-K3-6):
    `to_channels_first` transponiert bit-identisch nur wenn nötig — dieselbe
    Rückgabe (Identität) markiert das Nicht-Transponieren.
    """
    x = np.asarray(arr)
    if x.ndim == 1:
        return x.reshape(1, -1).astype(np.float32, copy=False), "mono"
    channels = audio_layout.to_channels_first(x)
    layout = "samples_first" if channels is not x else "channels_first"
    return np.ascontiguousarray(channels).astype(np.float32, copy=False), layout


class AspadeClassDeclipper:
    """Inferenz-Hülle um `AspadeClassNet` — Layout (C, N), Chunking, Kandidatenfilter.

    Ohne `checkpoint` werden deterministische Seed-Gewichte geladen (Test-/
    Challenger-Betrieb, §V6 (copilot-instructions.md)-Warnung); produktiv
    genutzt wird ausschließlich das trainierte Artefakt über
    `get_aspade_class_declipper()`.
    """

    def __init__(
        self,
        checkpoint: Path | str | None = None,
        *,
        stages: int = DEFAULT_STAGES,
        channels: int = DEFAULT_CHANNELS,
        kernel_size: int = DEFAULT_KERNEL_SIZE,
        dilations: tuple[int, ...] = DEFAULT_DILATIONS,
        seed: int = DEFAULT_SEED,
        never_worsen: bool = True,
        chunk_sec: float = CHUNK_SEC,
    ) -> None:
        self._never_worsen = bool(never_worsen)
        self._chunk_samples = max(1, int(float(chunk_sec) * TARGET_SR))
        if checkpoint is not None:
            payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
            cfg = dict(payload.get("aspade_config", {}))
            self._net: AspadeClassNet = AspadeClassNet(
                stages=int(cfg.get("stages", stages)),
                channels=int(cfg.get("channels", channels)),
                kernel_size=int(cfg.get("kernel_size", kernel_size)),
                dilations=tuple(cfg.get("dilations", dilations)),
            )
            self._net.load_state_dict(payload["model_state_dict"], strict=True)
            self._net.eval()
            logger.info("A-SPADE-Klassen-Declipper: Artefakt geladen: %s", checkpoint)
        else:
            self._net = build_model(
                stages=stages,
                channels=channels,
                kernel_size=kernel_size,
                dilations=dilations,
                seed=seed,
            )
            logger.warning(
                "A-SPADE-Klassen-Declipper ohne trainiertes Artefakt — deterministische Seed-Gewichte "
                "(seed=%d) nur für Test-/Challenger-Betrieb (§V6 (copilot-instructions.md))",
                seed,
            )
        self._context_samples = sum(
            (layer.kernel_size[0] - 1) * layer.dilation[0] // 2
            for layer in self._net.modules()
            if isinstance(layer, nn.Conv1d)
        )

    # ------------------------------------------------------------------

    def declip(
        self,
        audio: np.ndarray,
        sample_rate: int = TARGET_SR,
        clip_threshold: float | None = None,
    ) -> np.ndarray:
        """Declippt ein Signal und gibt es im Eingangs-Layout/-dtype zurück.

        Args:
            audio: (N,), (C, N) oder (N, C) mit C ≤ 2, float32/float64.
            sample_rate: Abtastrate (> 0) — nur für den Harmonik-Witness.
            clip_threshold: Clip-Schwelle θ > 0; None ⇒ deterministische
                P99.5-Schätzung (`_estimate_clip_threshold`).

        Returns:
            Array in Eingangs-Shape und -dtype. Bei Abwahl durch den Witness
            oder Fehlerpfaden (§V6 (copilot-instructions.md)) wird der Eingang
            unverändert zurückgegeben; ungültige Argumente werfen `ValueError`.
        """
        arr = np.asarray(audio)
        if arr.ndim not in (1, 2):
            raise ValueError(
                f"Layout nicht unterstützt: ndim={arr.ndim} — erwartet (N,), (C, N) oder (N, C) "
                "(Stereo-Layout-Invariante, AGENTS.md §3)"
            )
        if arr.dtype not in (np.float32, np.float64):
            raise ValueError(f"dtype {arr.dtype} nicht unterstützt — float32/float64 erwartet")
        if int(sample_rate) <= 0:
            raise ValueError(f"sample_rate muss > 0 sein (got {sample_rate})")
        if arr.size == 0:
            logger.warning("A-SPADE-Klassen-Declipper: leeres Audio — Passthrough (§V6 (copilot-instructions.md))")
            res_empty: np.ndarray = arr.copy()
            return res_empty

        work, layout = _to_channels_first(arr)
        if not bool(np.isfinite(work).all()):
            logger.warning(
                "A-SPADE-Klassen-Declipper: NaN/Inf im Eingang — sanitisiert (§0a/§V6 (copilot-instructions.md))"
            )
            work = np.nan_to_num(work, nan=0.0, posinf=0.0, neginf=0.0)

        if clip_threshold is None:
            theta = _estimate_clip_threshold(work)
        else:
            theta = float(clip_threshold)
            if not (theta > 0.0):
                raise ValueError(f"clip_threshold muss > 0 sein (got {clip_threshold})")

        out = np.empty_like(work)
        for ch in range(work.shape[0]):
            out[ch] = self._declip_channel(work[ch], theta, int(sample_rate))

        if layout == "samples_first":
            out = np.ascontiguousarray(out.T)
        elif layout == "mono":
            out = out.reshape(-1)
        res_out: np.ndarray = out.astype(arr.dtype, copy=False)
        return res_out

    # ------------------------------------------------------------------

    def _declip_channel(self, x32: np.ndarray, theta: float, sample_rate: int) -> np.ndarray:
        if not bool(np.any(np.abs(x32) >= theta)):
            logger.info(
                "A-SPADE-Klassen-Declipper: keine Samples ≥ θ=%.4f — Kanal passthrough",
                theta,
            )
            return x32.copy()

        cand = self._infer_channel(x32, theta)
        if not bool(np.isfinite(cand).all()):
            logger.warning(
                "A-SPADE-Klassen-Declipper: nicht-finites Modell-Ergebnis — Eingang bleibt "
                "(§V6 (copilot-instructions.md))"
            )
            return x32.copy()
        if not self._never_worsen:
            return cand

        w_in = harmonic_distortion_witness(x32, sample_rate)
        w_out = harmonic_distortion_witness(cand, sample_rate)
        if w_out < w_in:
            return cand
        logger.info(
            "A-SPADE-Klassen-Declipper: Kandidat abgelehnt (Witness %.6f ≥ %.6f) — Eingang bleibt",
            w_out,
            w_in,
        )
        return x32.copy()

    def _infer_channel(self, x32: np.ndarray, theta: float) -> np.ndarray:
        n = int(x32.size)
        chunk = self._chunk_samples
        if n <= chunk:
            return self._infer_block(x32, theta)

        # Der Kontextrand entspricht dem rezeptiven Feld aller Conv-Schichten.
        # Songgrenzen bleiben echte Netzgrenzen; künstliches Padding dort würde
        # wegen der Conv-Biases eine andere Vollsignal-Ausgabe erzeugen.
        out: np.ndarray = np.empty_like(x32)
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            left = max(0, start - self._context_samples)
            right = min(n, end + self._context_samples)
            rec = self._infer_block(x32[left:right], theta)
            out[start:end] = rec[start - left : end - left]
        return out

    def _infer_block(self, seg32: np.ndarray, theta: float) -> np.ndarray:
        y_t = torch.from_numpy(np.ascontiguousarray(seg32, dtype=np.float32)).reshape(1, 1, -1)
        theta_t = torch.tensor([[[theta]]], dtype=torch.float32)
        with torch.inference_mode():
            rec = self._net(y_t, theta_t)
        arr = np.asarray(rec, dtype=np.float32).reshape(-1)
        if arr.size != seg32.size:
            raise ValueError(f"Modell-Ausgabe {arr.size} ≠ Eingang {seg32.size}")
        res_block: np.ndarray = arr
        return res_block


# ── Fassade für Phase 07 (lazy Singleton, nur MIT trainiertem Artefakt) ─────

_declipper_singleton: AspadeClassDeclipper | None = None


def get_aspade_class_declipper() -> AspadeClassDeclipper | None:
    """Prozess-weite Singleton-Instanz (lazy) — nur mit trainiertem Artefakt.

    Ohne `models/aspade/aspade_class_v1.pt`: None + §V6 (copilot-instructions.md)-Warnung;
    der Aufrufer fällt auf die PCHIP-Kaskade zurück. Die Instanz hält nur unveränderliche Gewichte
    (kein Song-State, §V8/§G1 (copilot-instructions.md)).
    """
    global _declipper_singleton
    if _declipper_singleton is not None:
        return _declipper_singleton
    if not DEFAULT_CHECKPOINT.is_file():
        logger.warning(
            "A-SPADE-Klassen-Declipper: kein trainiertes Artefakt unter %s — Challenger nicht verfügbar, "
            "Ersatzpfad PCHIP-Kaskade (§V6 (copilot-instructions.md))",
            DEFAULT_CHECKPOINT,
        )
        return None
    _declipper_singleton = AspadeClassDeclipper(checkpoint=DEFAULT_CHECKPOINT)
    return _declipper_singleton


def declip_aspade_class(audio: np.ndarray, sample_rate: int, clip_threshold: float | None = None) -> np.ndarray | None:
    """Challenger-Fassade für Phase 07 — None ⇒ nicht verfügbar (§V6-Warnung erfolgte).

    Fehler laufen als Exception zum Aufrufer (§V6 (copilot-instructions.md):
    dort logger.warning + PCHIP-Kaskade) — kein stiller Fallback.
    """
    declipper = get_aspade_class_declipper()
    if declipper is None:
        return None
    return declipper.declip(audio, sample_rate=sample_rate, clip_threshold=clip_threshold)
