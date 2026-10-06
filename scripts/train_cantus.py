#!/usr/bin/env python3
"""Cantus-Training — MERT-conditioned Multi-Scale Flow-Matching DiT (3 Phasen).

Phasen (Cantus-Design, Trainingsstrategie):
  Phase 1  --phase=pretrain      Flow-Matching-Head + Conditions-Adapter auf
                                 gefrorenen MERT-Features (--freeze-encoder)
  Phase 2  --phase=fine_tune     Gesamtes trainierbares Netz + SingMOS-Learned-
                                 Loss (--singmos-loss)
  Phase 3  --phase=domain_adapt  Domänen-Adaptation auf echte degradierte
                                 Aufnahmen (--data=<dir/manifest>)

Multi-Objective-Loss (Cantus-Design):
    total = λ1·L_flow_matching      (Primär: MSE auf OT-Geschwindigkeitsfeld)
          + λ2·L_mel_spectral       (Mel-Spektrum, multi-resolution)
          + λ3·L_stft_phase         (phasengewichteter Komplex-STFT-Loss)
          + λ4·L_pitch_preservation (differenzierbarer f0-Verlauf, Soft-Argmax-
                                     Autokorrelation — erhält den Tonhöhenverlauf)
          + λ5·L_perceptual(SingMOS) (SingMOS als Learned Loss: Proxy-Netz wird
                                     gegen Teacher models/singmos/singmos_pro.onnx
                                     kalibriert; Gradienten laufen durch den
                                     Proxy in die Restaurierung)
          + λ6·L_temporal           (zeitliche Konsistenz)

Datensatz: Manifest-JSONL aus scripts/generate_synthetic_degraded_vocals.py
(synthetische Degradation auf MUSDB18-HQ-Vocals) plus echte Paare
(<stem>_clean.wav + <stem>_degraded.wav) in Ordnern.

Determinismus (§G5 (GEBOTE.md)): Session-Master-Seed → alle Ableitungen
(Chunk-Offsets, Flow-Zeiten, Initialisierung) deterministisch; im Checkpoint
und Report mitgeführt. NaN/Inf-Guard je Step (§0a (copilot-instructions.md)).

Usage:
    python3 -B scripts/train_cantus.py --phase=pretrain   --freeze-encoder=true  --epochs=50
    python3 -B scripts/train_cantus.py --phase=fine_tune  --singmos-loss=true    --epochs=30
    python3 -B scripts/train_cantus.py --phase=domain_adapt --data=data/real_degraded_vocals --epochs=10
    python3 -B scripts/train_cantus.py --smoke            # CPU-End-to-End-Smoke (2 Steps)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.core.training_artifacts import save_guarded, write_text_guarded
from models.cantus.cantus_model import CantusModel, create_cantus

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "models" / "cantus" / "cantus_config.json"
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "cantus"
DEFAULT_SEED = 20261004
SR = 48000  # Modell-Sample-Rate (48 kHz, siehe models/cantus/README.md)


def _load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


# ═══════════════════════════════════════════════════════════════════════════
# Loss-Bausteine (alle differenzierbar, ONNX-frei — Training only)
# ═══════════════════════════════════════════════════════════════════════════


class MelSpectralLoss(nn.Module):
    """λ2: L1 auf log-Mel-Spektrogramm über mehrere FFT-Größen."""

    def __init__(self, sr: int = 48000, n_ffts: tuple[int, ...] = (512, 1024, 2048), n_mels: int = 80):
        super().__init__()
        import librosa  # lokaler Import: Training-only-Abhängigkeit

        self.n_ffts = n_ffts
        for n_fft in n_ffts:
            fb = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=40.0, fmax=sr / 2.0)
            self.register_buffer(f"mel_fb_{n_fft}", torch.from_numpy(fb).float())

    def _mel_fb(self, n_fft: int) -> torch.Tensor:
        return getattr(self, f"mel_fb_{n_fft}")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.squeeze(1) if pred.dim() == 3 else pred
        target = target.squeeze(1) if target.dim() == 3 else target
        loss = pred.new_zeros(())
        for n_fft in self.n_ffts:
            window = torch.hann_window(n_fft, device=pred.device)
            hop = n_fft // 4
            sp = torch.stft(pred, n_fft=n_fft, hop_length=hop, window=window, return_complex=True)
            st = torch.stft(target, n_fft=n_fft, hop_length=hop, window=window, return_complex=True)
            mp = torch.matmul(self._mel_fb(n_fft), sp.abs().clamp_min(1e-7))
            mt = torch.matmul(self._mel_fb(n_fft), st.abs().clamp_min(1e-7))
            loss = loss + F.l1_loss(torch.log(mp + 1e-6), torch.log(mt + 1e-6))
        return loss / max(len(self.n_ffts), 1)


class StftPhaseLoss(nn.Module):
    """λ3: Phasen-Erhaltung — magnitudengewichteter Kosinus der Phasendifferenz.

    Zusätzlich kleiner Real/Imag-L1-Anteil für spektrale Konvergenz
    (phasenkorrekte Rekonstruktion ist das Ziel des Cantus-Ausgangs).
    """

    def __init__(self, n_ffts: tuple[int, ...] = (512, 1024, 2048)):
        super().__init__()
        self.n_ffts = n_ffts

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.squeeze(1) if pred.dim() == 3 else pred
        target = target.squeeze(1) if target.dim() == 3 else target
        loss = pred.new_zeros(())
        for n_fft in self.n_ffts:
            window = torch.hann_window(n_fft, device=pred.device)
            hop = n_fft // 4
            sp = torch.stft(pred, n_fft=n_fft, hop_length=hop, window=window, return_complex=True)
            st = torch.stft(target, n_fft=n_fft, hop_length=hop, window=window, return_complex=True)
            cos_phase = (sp.real * st.real + sp.imag * st.imag) / (sp.abs() * st.abs() + 1e-8)
            weight = st.abs() / (st.abs().sum() + 1e-8)
            phase_term = ((1.0 - cos_phase) * weight).sum()
            ri_term = 0.1 * (F.l1_loss(sp.real, st.real) + F.l1_loss(sp.imag, st.imag))
            loss = loss + phase_term + ri_term
        return loss / max(len(self.n_ffts), 1)


class PitchPreservationLoss(nn.Module):
    """λ4: Tonhöhen-Verlauf-Erhaltung via differenzierbarer f0-Schätzung.

    Autokorrelation je Frame (rFFT-Leistungsspektrum → irFFT), Soft-Argmax über
    den Lag-Bereich 50–1200 Hz → f0. Vergleich: Smooth-L1 auf log2(f0),
    gewichtet über Voicing-Stärke des Targets (detached). Dadurch fließen
    Gradienten durch die Wellenform in die f0 — die Restaurierung respektiert
    den Tonhöhenverlauf (Kernprinzip 4 des Cantus-Designs).
    """

    def __init__(
        self,
        sr: int = 48000,
        frame: int = 2048,
        hop: int = 512,
        fmin: float = 50.0,
        fmax: float = 1200.0,
        sharpness: float = 20.0,
    ):
        super().__init__()
        self.sr = sr
        self.frame = frame
        self.hop = hop
        self.min_lag = max(int(sr / fmax), 1)
        self.max_lag = int(sr / fmin)
        self.sharpness = sharpness

    def _f0(self, wave: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """wave [B, T] → (log2_f0 [B, N], voiced_strength [B, N])."""
        frames = wave.unfold(-1, self.frame, self.hop)  # [B, N, frame]
        spec = torch.fft.rfft(frames, n=self.frame * 2)
        ac = torch.fft.irfft(spec.abs() ** 2, n=self.frame * 2)[..., : self.max_lag + 1]
        ac = ac / (ac[..., :1] + 1e-8)
        region = ac[..., self.min_lag : self.max_lag]
        weights = torch.softmax(region * self.sharpness, dim=-1)
        lags = torch.arange(self.min_lag, self.max_lag, device=wave.device, dtype=wave.dtype)
        lag = (weights * lags).sum(-1) + 1e-8
        f0 = self.sr / lag
        voiced = region.max(-1).values.clamp(0.0, 1.0).detach()
        return torch.log2(f0.clamp_min(1.0)), voiced

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.squeeze(1) if pred.dim() == 3 else pred
        target = target.squeeze(1) if target.dim() == 3 else target
        logf0_p, _ = self._f0(pred)
        logf0_t, voiced = self._f0(target)
        mask = (voiced > 0.3).float()
        n_voiced = mask.sum()
        if float(n_voiced) < 1.0:
            return pred.new_zeros(())  # stille/unvoiced Abschnitte: neutral (§0a)
        err = F.smooth_l1_loss(logf0_p, logf0_t, reduction="none") * mask
        return err.sum() / n_voiced


class TemporalConsistencyLoss(nn.Module):
    """λ6: Zeitliche Konsistenz — Ableitungsdifferenzen (Sample- und Mel-Ebene)."""

    def __init__(self, sr: int = 48000, n_fft: int = 1024, n_mels: int = 64):
        super().__init__()
        import librosa

        fb = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
        self.register_buffer("mel_fb", torch.from_numpy(fb).float())
        self.n_fft = n_fft

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.squeeze(1) if pred.dim() == 3 else pred
        target = target.squeeze(1) if target.dim() == 3 else target
        wave_term = F.l1_loss(torch.diff(pred, dim=-1), torch.diff(target, dim=-1))
        window = torch.hann_window(self.n_fft, device=pred.device)
        mp = torch.matmul(
            self.mel_fb, torch.stft(pred, self.n_fft, self.n_fft // 4, window=window, return_complex=True).abs()
        )
        mt = torch.matmul(
            self.mel_fb, torch.stft(target, self.n_fft, self.n_fft // 4, window=window, return_complex=True).abs()
        )
        mel_term = F.l1_loss(torch.diff(mp, dim=-1), torch.diff(mt, dim=-1))
        return wave_term + 0.5 * mel_term


class SingMOSProxy(nn.Module):
    """Kleines differenzierbares Proxy-Netz auf Log-Mel → MOS-Schätzung ∈ [1, 5].

    Es wird kontinuierlich gegen den SingMOS-Teacher (ONNX, nicht
    differenzierbar) kalibriert und dann als Learned Loss genutzt: die
    Gradienten der Restaurierung fließen durch dieses Netz (Cantus-Design:
    „SingMOS als Learned Loss").
    """

    def __init__(self, sr: int = 48000, n_fft: int = 1024, n_mels: int = 64):
        super().__init__()
        import librosa

        fb = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
        self.register_buffer("mel_fb", torch.from_numpy(fb).float())
        self.n_fft = n_fft
        self.hop = n_fft // 4
        self.body = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.head = nn.Sequential(nn.Linear(32 * 4 * 4, 64), nn.GELU(), nn.Linear(64, 1))

    def _logmel(self, wave: torch.Tensor) -> torch.Tensor:
        wave = wave.squeeze(1) if wave.dim() == 3 else wave
        window = torch.hann_window(self.n_fft, device=wave.device)
        mag = torch.stft(wave, self.n_fft, self.hop, window=window, return_complex=True).abs()
        mel = torch.matmul(self.mel_fb, mag)
        return torch.log(mel + 1e-6).unsqueeze(1)  # [B, 1, M, F]

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        h = self.body(self._logmel(wave)).flatten(1)
        return 1.0 + 4.0 * torch.sigmoid(self.head(h)).squeeze(-1)  # [B] ∈ (1, 5)


# ── SingMOS Learned Loss (λ5) ───────────────────────────────────────


class SingMOSLearnedLoss(nn.Module):
    """SingMOS als differenzierbarer Learned-Loss (Cantus-Design-Innovation).

    Zwei Rollen:
      * Teacher  : models/singmos/singmos_pro.onnx (nicht differenzierbar) liefert
                   MOS-Zielscores auf detached Audio (Backend:
                   backend.core.dsp.quality_predictors).
      * Proxy    : kleines faltendes Netz (SingMOSProxy) — differenzierbar in der
                   Wellenform und wird periodisch gegen den Teacher kalibriert.

    Generator-Loss: hinge(target_mos − proxy(ŷ)) — treibt die Restaurierung
    direkt in Richtung wahrgenommene Qualität.

    §V6 (VERBOTEN.md): kein Teacher ⇒ dokumentierter Ersatzpfad (kalibrierte
    Proxy-los bleibt Regularisierer) mit logger.warning — kein Stillfehler.
    """

    def __init__(self, target_mos: float = 4.5, calibrate_every_steps: int = 100, sr: int = SR) -> None:
        super().__init__()
        self.target_mos = float(target_mos)
        self.calibrate_every_steps = max(int(calibrate_every_steps), 1)
        self.sr = int(sr)
        self.proxy = SingMOSProxy()
        self._teacher: Any | None = None
        self._teacher_failed = False
        self._calib_loss = nn.MSELoss()

    # Teacher (ONNX, numpy) — lazy, nie im Autograd-Pfad
    def _get_teacher(self) -> Any | None:
        if self._teacher is not None:
            return self._teacher
        if self._teacher_failed:
            return None
        try:
            from backend.core.dsp.quality_predictors import (
                get_singmos_predictor,
            )

            self._teacher = get_singmos_predictor()
            return self._teacher
        except Exception as exc:
            self._teacher_failed = True
            logger.warning(
                "§V6 (VERBOTEN.md) ML→Ersatzpfad: SingMOS-Teacher nicht verfügbar (%s) — "
                "Proxy-Loss läuft unkalibriert als Regularisierer weiter",
                exc,
            )
            return None

    def teacher_scores(self, wave: torch.Tensor) -> torch.Tensor | None:
        """MOS-Scores [B] auf detached Audio; None wenn kein Teacher."""
        teacher = self._get_teacher()
        if teacher is None:
            return None
        scores: list[float] = []
        with torch.no_grad():
            for i in range(wave.shape[0]):
                arr = wave[i, 0].detach().cpu().numpy().astype(np.float32)
                try:
                    scores.append(float(teacher.predict(arr, self.sr)))
                except Exception as exc:
                    logger.warning("SingMOS-Teacher-Inferenz fehlgeschlagen (%s) — neutral 3.0", exc)
                    scores.append(3.0)
        return torch.tensor(scores, dtype=torch.float32, device=wave.device)

    def generator_loss(self, wave_pred: torch.Tensor) -> torch.Tensor:
        """Hinge-Loss Richtung target_mos (differenzierbar durch den Proxy)."""
        mos = self.proxy(wave_pred)
        return F.relu(self.target_mos - mos).mean()

    def calibration_loss(self, wave_pred: torch.Tensor) -> torch.Tensor | None:
        """Proxy-Kalibrier-Schritt gegen Teacher-Scores; None wenn ohne Teacher."""
        targets = self.teacher_scores(wave_pred)
        if targets is None:
            return None
        mos = self.proxy(wave_pred.detach())
        return self._calib_loss(mos, targets)


# ── Multi-Objective-Loss (λ1…λ6) ───────────────────────────────────────


class MultiObjectiveLoss(nn.Module):
    """total = λ1·L_flow + λ2·L_mel + λ3·L_phase + λ4·L_pitch + λ5·L_SingMOS + λ6·L_temporal.

    forward() rekonstruiert ŷ = x_t + (1−t)·v̂ (Inferenzformel, siehe
    models/cantus/README.md) und wertet die Audio-Losses auf ŷ vs. clean aus.
    """

    COMPONENTS = (
        "flow_matching",
        "mel_spectral",
        "stft_phase",
        "pitch_preservation",
        "singmos_perceptual",
        "temporal",
    )

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        *,
        sr: int = SR,
        use_singmos: bool = False,
        singmos_target_mos: float = 4.5,
        singmos_calibrate_every: int = 100,
    ) -> None:
        super().__init__()
        cfg = json.loads((CONFIG_PATH).read_text(encoding="utf-8"))["training"]["loss_weights"]
        self.weights = dict(cfg)
        if weights:
            self.weights.update(weights)
        self.use_singmos = bool(use_singmos)
        self.mel = MelSpectralLoss(sr=sr)
        self.phase = StftPhaseLoss()
        self.pitch = PitchPreservationLoss(sr=sr)
        self.temporal = TemporalConsistencyLoss(sr=sr)
        self.singmos = SingMOSLearnedLoss(
            target_mos=singmos_target_mos, calibrate_every_steps=singmos_calibrate_every, sr=sr
        )

    def forward(
        self,
        v_pred: torch.Tensor,
        v_target: torch.Tensor,
        x_t: torch.Tensor,
        y_clean: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        """Liefert (total, Einzel-Losses, y_hat) — y_hat für Metriken/Export."""
        w = self.weights
        # Rekonstruktion wie bei der Inferenz (t als [B] → [B,1,1])
        t_col = t.float().reshape(-1, 1, 1)
        y_hat = x_t + (1.0 - t_col) * v_pred

        l_flow = F.mse_loss(v_pred, v_target)
        l_mel = self.mel(y_hat, y_clean)
        l_phase = self.phase(y_hat, y_clean)
        l_pitch = self.pitch(y_hat, y_clean)
        l_temp = self.temporal(y_hat, y_clean)
        l_sing = self.singmos.generator_loss(y_hat) if self.use_singmos else y_hat.new_zeros(())

        components = {
            "flow_matching": l_flow,
            "mel_spectral": l_mel,
            "stft_phase": l_phase,
            "pitch_preservation": l_pitch,
            "singmos_perceptual": l_sing,
            "temporal": l_temp,
        }
        total = (
            w["flow_matching"] * l_flow
            + w["mel_spectral"] * l_mel
            + w["stft_phase"] * l_phase
            + w["pitch_preservation"] * l_pitch
            + w["singmos_perceptual"] * l_sing
            + w["temporal"] * l_temp
        )
        return total, components, y_hat


# ── Conditions-Extraktion (MERT + Pitch + MuQ-MuLan) ──────────────────────


class ConditionExtractor:
    """Extrahiert die musikalische Konditionierung auf dem degradierten Chunk.

    * MERT-v1-330M  : backend.core.mert_feature_extractor (1024-d, frame-level)
    * Pitch (FCPE → CREPE): plugins.fcpe_plugin.analyze_pitch (f0 + Voicing)
    * Harmonic Context: plugins.muq_mulan_plugin.extract_muq_mulan_embedding (768-d)

    Alle Extraktoren laufen no-grad/außerhalb des Autograd-Pfads (MERT bleibt
    gefroren, §III.9 (copilot-instructions.md)). Fehlt ein Extraktor, wird
    use_cond=0 geliefert (Null-Tokens im Modell) und §V6 (VERBOTEN.md)
    konform geloggt — keine Zufalls-Features als Stillfehler.
    """

    def __init__(self, sr: int = SR) -> None:
        self.sr = int(sr)
        self._mert: Any | None = None
        self._mert_failed = False
        self._warned: set[str] = set()

    def _warn_once(self, key: str, exc: object) -> None:
        if key in self._warned:
            return
        self._warned.add(key)
        logger.warning(
            "§V6 (VERBOTEN.md) ML→Ersatzpfad: %s nicht verfügbar (%s) — Konditionierung use_cond=0 (Null-Tokens)",
            key,
            exc,
        )

    def _extract_mert(self, mono: np.ndarray) -> np.ndarray | None:
        if self._mert_failed:
            return None
        try:
            if self._mert is None:
                from backend.core.mert_feature_extractor import (
                    MERTFeatureExtractor,
                )

                self._mert = MERTFeatureExtractor()
            feats = self._mert.extract(mono.astype(np.float32), self.sr)
            return np.asarray(feats, dtype=np.float32)
        except Exception as exc:
            self._mert_failed = True
            self._warn_once("MERT", exc)
            return None

    def _extract_pitch(self, mono: np.ndarray, n_frames: int) -> np.ndarray | None:
        try:
            from plugins.fcpe_plugin import analyze_pitch
        except Exception:
            try:
                from plugins.crepe_plugin import analyze_pitch
            except Exception as exc:
                self._warn_once("Pitch", exc)
                return None
        try:
            result = analyze_pitch(mono.astype(np.float32), self.sr)
            f0 = np.asarray(result.f0_hz, dtype=np.float32)
            voiced = np.asarray(result.voiced_prob, dtype=np.float32)
            if f0.size == 0:
                return np.zeros((n_frames, 2), dtype=np.float32)
            # Frames an die MERT-Frame-Zahl angleichen (np.interp, deterministisch)
            src = np.linspace(0.0, 1.0, num=f0.size, dtype=np.float32)
            dst = np.linspace(0.0, 1.0, num=n_frames, dtype=np.float32)
            f0_i = np.interp(dst, src, f0).astype(np.float32)
            v_i = np.interp(dst, src, voiced).astype(np.float32)
            # log2-f0 normiert auf [0,1] (50–1200 Hz Spanne), gestimmt = 0
            log2f0 = np.log2(np.maximum(f0_i, 1e-3))
            f0_norm = np.clip((log2f0 - np.log2(50.0)) / (np.log2(1200.0) - np.log2(50.0)), 0.0, 1.0)
            f0_norm = np.where(v_i > 0.45, f0_norm, 0.0).astype(np.float32)
            return np.stack([f0_norm, v_i], axis=-1)
        except Exception as exc:
            self._warn_once("Pitch", exc)
            return None

    def _extract_harmonic(self, mono: np.ndarray) -> np.ndarray | None:
        try:
            from plugins.muq_mulan_plugin import (
                extract_muq_mulan_embedding,
            )

            emb = extract_muq_mulan_embedding(mono.astype(np.float32), self.sr)
            if emb is None:
                raise RuntimeError("MuQ-MuLan-Modell fehlt")
            return np.asarray(emb, dtype=np.float32).reshape(-1)
        except Exception as exc:
            self._warn_once("MuQ-MuLan", exc)
            return None

    def extract(self, mono: np.ndarray) -> dict[str, np.ndarray]:
        """mono [T] float32 → dict(mert [F,768], pitch [F,2], harm [768], use_cond scalar)."""
        mono = np.asarray(mono, dtype=np.float32).reshape(-1)
        mert = self._extract_mert(mono)
        harm = self._extract_harmonic(mono)
        if mert is None:
            n_frames = 1
            pitch = np.zeros((1, 2), dtype=np.float32)
            mert = np.zeros((1, 1024), dtype=np.float32)
            harm = np.zeros((768,), dtype=np.float32) if harm is None else harm
            use_cond = 0.0
        else:
            n_frames = int(mert.shape[0])
            pitch = self._extract_pitch(mono, n_frames)
            if pitch is None:
                pitch = np.zeros((n_frames, 2), dtype=np.float32)
            if harm is None:
                harm = np.zeros((768,), dtype=np.float32)
            use_cond = 1.0
        return {
            "mert": np.ascontiguousarray(mert, dtype=np.float32),
            "pitch": np.ascontiguousarray(pitch, dtype=np.float32),
            "harm": np.ascontiguousarray(harm, dtype=np.float32),
            "use_cond": np.asarray(use_cond, dtype=np.float32),
        }


# ── Datasets ───────────────────────────────────────────────────────


def _load_mono(path: Path) -> np.ndarray:
    """Laedt mono @SR float32 [T] (librosa, deterministisch)."""
    import librosa

    y, _ = librosa.load(str(path), sr=SR, mono=True)
    return np.asarray(y, dtype=np.float32)


def _set_seeds(seed: int) -> None:
    """Session-Master-Seed (§G5 (GEBOTE.md)) — np/torch deterministisch."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _chunk_seed(master: int, index: int, epoch: int) -> int:
    """Deterministische Chunk-Saat pro (Index, Epoch) statt time/urandom."""
    return (master * 1_000_003 + index * 7_919 + epoch * 1_047_29) % (2**31 - 1)


class PairDataset(Dataset):
    """(degraded, clean)-Paare aus Manifest-JSONL oder Ordnern.

    * Manifest: von scripts/generate_synthetic_degraded_vocals.py (Phasen 1–2)
    * Ordner  : `<stem>_degraded.wav` + `<stem>_clean.wav` (Phase 3, real)
    Chunks deterministisch pro (Index, Epoch) (§G5 (GEBOTE.md)).
    """

    def __init__(
        self,
        sources: list[Path],
        *,
        chunk_samples: int,
        extractor: ConditionExtractor,
        seed: int,
        with_cond: bool = True,
    ) -> None:
        self.chunk_samples = int(chunk_samples)
        self.extractor = extractor
        self.seed = int(seed)
        self.with_cond = bool(with_cond)
        self.epoch = 0
        self.pairs: list[dict[str, Path]] = self._collect(sources)
        if not self.pairs:
            raise FileNotFoundError(f"Keine Paare gefunden in: {[str(s) for s in sources]}")

    @staticmethod
    def _collect(sources: list[Path]) -> list[dict[str, Path]]:
        pairs: list[dict[str, Path]] = []
        for src in sources:
            src = Path(src)
            if src.suffix == ".jsonl":
                root = src.parent
                for line in src.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    pairs.append({"clean": root / row["clean"], "degraded": root / row["degraded"]})
            elif src.is_dir():
                for deg in sorted(src.rglob("*_degraded.wav")):
                    clean = deg.with_name(deg.name.replace("_degraded.wav", "_clean.wav"))
                    if clean.is_file():
                        pairs.append({"clean": clean, "degraded": deg})
        return pairs

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        pair = self.pairs[idx]
        clean = _load_mono(pair["clean"])
        degraded = _load_mono(pair["degraded"])
        n = min(clean.size, degraded.size)
        clean, degraded = clean[:n], degraded[:n]
        if n < self.chunk_samples:  # Zero-Pad, §0a (copilot-instructions.md) NaN/Inf-sicher
            pad = self.chunk_samples - n
            clean = np.pad(clean, (0, pad))
            degraded = np.pad(degraded, (0, pad))
            start = 0
        else:
            rng = np.random.default_rng(_chunk_seed(self.seed, idx, self.epoch))
            start = int(rng.integers(0, n - self.chunk_samples + 1))
        clean_c = clean[start : start + self.chunk_samples]
        deg_c = degraded[start : start + self.chunk_samples]
        clean_c = np.nan_to_num(clean_c, nan=0.0, posinf=0.0, neginf=0.0)
        deg_c = np.nan_to_num(deg_c, nan=0.0, posinf=0.0, neginf=0.0)

        cond = (
            self.extractor.extract(deg_c)
            if self.with_cond
            else {
                "mert": np.zeros((1, 1024), dtype=np.float32),
                "pitch": np.zeros((1, 2), dtype=np.float32),
                "harm": np.zeros((768,), dtype=np.float32),
                "use_cond": np.asarray(0.0, dtype=np.float32),
            }
        )
        return {
            "clean": torch.from_numpy(clean_c.copy()).view(1, -1),
            "degraded": torch.from_numpy(deg_c.copy()).view(1, -1),
            "mert": torch.from_numpy(cond["mert"]),
            "pitch": torch.from_numpy(cond["pitch"]),
            "harm": torch.from_numpy(cond["harm"]),
            "use_cond": torch.tensor(float(cond["use_cond"])),
        }


class SmokeDataset(Dataset):
    """Synthetische Sample-Vocal-Stems (Vibrato + Harmonische + Hüllkurve)

    für CPU-Smoke-Ende-zu-Ende ohne MUSDB18. Deterministisch pro Index/Epoch."""

    def __init__(self, size: int, *, chunk_samples: int, seed: int, snr_db: float = 8.0) -> None:
        self.size = int(size)
        self.chunk_samples = int(chunk_samples)
        self.seed = int(seed)
        self.snr_db = float(snr_db)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng(_chunk_seed(self.seed, idx, self.epoch))
        t = np.arange(self.chunk_samples, dtype=np.float32) / SR
        f0 = 220.0 + 30.0 * np.sin(2.0 * np.pi * 5.5 * t)  # Vibrato
        phase = 2.0 * np.pi * np.cumsum(f0) / SR
        clean = 0.6 * np.sin(phase) + 0.25 * np.sin(2.0 * phase) + 0.1 * np.sin(3.0 * phase)
        env = np.clip(np.sin(np.pi * t / max(t[-1], 1e-6)) * 1.5, 0.0, 1.0).astype(np.float32)
        clean = (clean * env).astype(np.float32)
        noise = rng.standard_normal(self.chunk_samples).astype(np.float32)
        noise *= float(np.std(clean) + 1e-10) / (10.0 ** (self.snr_db / 20.0) * (float(np.std(noise)) + 1e-10))
        degraded = clean + noise
        zero_cond = {
            "mert": np.zeros((1, 1024), dtype=np.float32),
            "pitch": np.zeros((1, 2), dtype=np.float32),
            "harm": np.zeros((768,), dtype=np.float32),
        }
        return {
            "clean": torch.from_numpy(clean.copy()).view(1, -1),
            "degraded": torch.from_numpy(degraded.astype(np.float32).copy()).view(1, -1),
            "mert": torch.from_numpy(zero_cond["mert"]),
            "pitch": torch.from_numpy(zero_cond["pitch"]),
            "harm": torch.from_numpy(zero_cond["harm"]),
            "use_cond": torch.tensor(0.0),
        }


# ── Training ─────────────────────────────────────────────────────


def _model_kwargs(model_cfg: dict) -> dict:
    return {k: v for k, v in model_cfg.items() if not k.startswith("_")}


def _flow_pair(clean: torch.Tensor, degraded: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """OT-Pfad: x_t = (1−t)·x_deg + t·y, v = y − x_deg (Flow-Matching)."""
    t_col = t.reshape(-1, 1, 1)
    x_t = (1.0 - t_col) * degraded + t_col * clean
    return x_t, clean - degraded


def _checkpoint_payload(
    model, loss_fn, optimizer, *, epoch: int, step: int, val_loss: float, phase: str, seed: int, cfg: dict
) -> dict:
    """Checkpoint inkl. Seeds/Phase für Reproduzierbarkeit (§G5 (GEBOTE.md))."""
    return {
        "model_state_dict": model.state_dict(),
        "singmos_proxy_state_dict": loss_fn.singmos.proxy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "val_loss": val_loss,
        "phase": phase,
        "seed": seed,
        "config": cfg,
    }


def _resolve_checkpoint_dir(smoke: bool) -> Path:
    """Checkpoint-/Report-Verzeichnis — Smoke isoliert, Produktion wird nie überschrieben.

    Befund 2026-10-06: `--smoke` schrieb nach `models/cantus/` und überschrieb
    `checkpoint_latest.pt`/`checkpoint_best.pt` (2,45 GB → 4,7 MB) sowie
    `train_report_pretrain.json`. Smoke-Läufe sind Validierung, keine Evidenz,
    und dürfen Produktionsartefakte nicht anfassen.
    """
    if smoke:
        return Path(tempfile.mkdtemp(prefix="cantus_smoke_"))
    return CHECKPOINT_DIR


@torch.no_grad()
def _evaluate(model, loss_fn, loader, device, max_batches: int = 8) -> float:
    model.eval()
    losses: list[float] = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        clean = batch["clean"].to(device)
        degraded = batch["degraded"].to(device)
        t = torch.rand(clean.shape[0], device=device)
        x_t, v_target = _flow_pair(clean, degraded, t)
        # Modellgrenze: Losses nutzen (B, 1, T), CantusModel erwartet (B, T, 1)
        v_pred = model(
            x_t.transpose(1, 2),
            t,
            batch["mert"].to(device),
            batch["pitch"].to(device),
            batch["harm"].to(device),
            batch["use_cond"].to(device),
        ).transpose(1, 2)
        total, _, _ = loss_fn(v_pred, v_target, x_t, clean, t)
        if torch.isfinite(total):
            losses.append(float(total))
    model.train()
    return float(np.mean(losses)) if losses else float("inf")


def train(args: argparse.Namespace) -> int:
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    tcfg = cfg["training"]
    model_cfg = cfg["model"][args.preset]
    seed = int(args.seed) if args.seed is not None else int(tcfg["seed"])
    _set_seeds(seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    chunk_samples = int(float(tcfg["chunk_sec"]) * SR)
    cond_dropout = float(model_cfg.get("cond_dropout", 0.0))

    extractor = ConditionExtractor()
    if args.smoke:
        train_ds = SmokeDataset(16, chunk_samples=chunk_samples, seed=seed)
        val_ds = SmokeDataset(4, chunk_samples=chunk_samples, seed=seed + 1)
    else:
        sources = [Path(s.strip()) for s in args.data.split(",") if s.strip()]
        if not sources:
            logger.error("--data fehlt (Manifest-JSONL/Ordner) — oder --smoke fuer den CPU-Smoke")
            return 2
        full = PairDataset(sources, chunk_samples=chunk_samples, extractor=extractor, seed=seed)
        n_train = max(1, int(0.8 * len(full)))
        train_ds = full
        val_ds = PairDataset(sources, chunk_samples=chunk_samples, extractor=extractor, seed=seed)
        val_ds.pairs = full.pairs[n_train:] or full.pairs[-1:]
        train_ds.pairs = full.pairs[:n_train]

    batch_size = 2 if args.smoke else int(args.batch_size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

    model = create_cantus(**_model_kwargs(model_cfg)).to(device)
    if args.freeze_encoder:
        for name, p in model.named_parameters():
            if name.startswith(("condition_encoder", "pitch_encoder", "harm_encoder")):
                p.requires_grad_(False)
        logger.info("Trainingsabschnitt %s: Conditions-Adapter eingefroren (--freeze-encoder)", args.phase)

    use_singmos = bool(args.singmos_loss) or args.phase == "fine_tune"
    loss_fn = MultiObjectiveLoss(
        use_singmos=use_singmos,
        singmos_target_mos=float(tcfg["singmos"]["target_mos"]),
        singmos_calibrate_every=int(tcfg["singmos"]["proxy_calibrate_every_steps"]),
    ).to(device)

    lr_default = {
        "pretrain": float(tcfg["lr_pretrain"]),
        "fine_tune": float(tcfg["lr_finetune"]),
        "domain_adapt": float(tcfg["lr_domain_adapt"]),
    }[args.phase]
    lr = float(args.lr) if args.lr else lr_default
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=float(tcfg["weight_decay"]),
        betas=(0.9, 0.999),
    )
    opt_proxy = torch.optim.AdamW(loss_fn.singmos.proxy.parameters(), lr=1e-3) if use_singmos else None

    start_epoch, global_step, best_val = 0, 0, float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        if ckpt.get("singmos_proxy_state_dict"):
            loss_fn.singmos.proxy.load_state_dict(ckpt["singmos_proxy_state_dict"])
        try:
            opt.load_state_dict(ckpt["optimizer_state_dict"])
        except Exception as exc:
            logger.warning("Optimizer-State nicht ladbar (%s) — frischer Optimizer", exc)
        start_epoch = int(ckpt.get("epoch", 0))
        global_step = int(ckpt.get("step", 0))
        best_val = float(ckpt.get("val_loss", float("inf")))
        logger.info(
            "Resume: epoch=%d step=%d best_val=%.4f (seed=%s)", start_epoch, global_step, best_val, ckpt.get("seed")
        )

    # Smoke-/Validierungsläufe dürfen NIE die Produktionsartefakte überschreiben
    # (Befund 2026-10-06: 2,45-GB-Pretrain-Checkpoints wurden durch den tiny-
    # Smoke überschrieben). Siehe `_resolve_checkpoint_dir`.
    ckpt_dir = _resolve_checkpoint_dir(args.smoke)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cal_every = int(tcfg["singmos"]["proxy_calibrate_every_steps"])
    report: dict[str, Any] = {
        "phase": args.phase,
        "seed": seed,
        "preset": args.preset,
        "freeze_encoder": bool(args.freeze_encoder),
        "singmos_loss": use_singmos,
        "device": str(device),
        "checkpoints": [],
        "note": "Kein automatischer Flag-Flip — Aktivierung nur nach A/B-Abnahme",
    }

    logger.info(
        "Training: phase=%s epochs=%d lr=%.1e device=%s model=%.1fM params",
        args.phase,
        args.epochs,
        lr,
        device,
        model.num_params,
    )
    for epoch in range(start_epoch, int(args.epochs)):
        train_ds.set_epoch(epoch)
        model.train()
        for step_i, batch in enumerate(train_loader):
            if step_i >= int(args.steps_per_epoch):
                break
            clean = batch["clean"].to(device)
            degraded = batch["degraded"].to(device)
            mert = batch["mert"].to(device)
            pitch = batch["pitch"].to(device)
            harm = batch["harm"].to(device)
            use_cond = batch["use_cond"].to(device)
            if cond_dropout > 0.0:  # Classifier-Free-Guidance-Training (Null-Tokens)
                keep = (torch.rand(clean.shape[0], device=device) >= cond_dropout).float()
                use_cond = use_cond * keep

            t = torch.rand(clean.shape[0], device=device)
            x_t, v_target = _flow_pair(clean, degraded, t)
            v_pred = model(x_t.transpose(1, 2), t, mert, pitch, harm, use_cond).transpose(1, 2)
            total, components, y_hat = loss_fn(v_pred, v_target, x_t, clean, t)

            if not torch.isfinite(total):  # §0a (copilot-instructions.md) NaN/Inf-Guard
                logger.warning(
                    "Nicht-finiter Loss (epoch=%d step=%d) — Step uebersprungen (§0a (copilot-instructions.md))",
                    epoch,
                    step_i,
                )
                opt.zero_grad(set_to_none=True)
                continue

            opt.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
            opt.step()

            if opt_proxy is not None and global_step % cal_every == 0:
                cal = loss_fn.singmos.calibration_loss(y_hat.detach())
                if cal is not None and torch.isfinite(cal):
                    opt_proxy.zero_grad(set_to_none=True)
                    cal.backward()
                    opt_proxy.step()

            global_step += 1
            if global_step % max(int(args.checkpoint_every), 1) == 0:
                val_loss = _evaluate(model, loss_fn, val_loader, device)
                mos_scores = loss_fn.singmos.teacher_scores(y_hat)
                mos_mean = float(mos_scores.mean()) if mos_scores is not None else None
                logger.info(
                    "step=%d val=%.4f singmos=%s flow=%.4f mel=%.4f phase=%.4f pitch=%.4f temp=%.4f",
                    global_step,
                    val_loss,
                    "n/a" if mos_mean is None else f"{mos_mean:.2f}",
                    float(components["flow_matching"].detach()),
                    float(components["mel_spectral"].detach()),
                    float(components["stft_phase"].detach()),
                    float(components["pitch_preservation"].detach()),
                    float(components["temporal"].detach()),
                )
                payload = _checkpoint_payload(
                    model,
                    loss_fn,
                    opt,
                    epoch=epoch,
                    step=global_step,
                    val_loss=val_loss,
                    phase=args.phase,
                    seed=seed,
                    cfg=cfg,
                )
                save_guarded(ckpt_dir / "checkpoint_latest.pt", payload)
                if val_loss < best_val:
                    best_val = val_loss
                    save_guarded(ckpt_dir / "checkpoint_best.pt", payload)
                report["checkpoints"].append({"step": global_step, "val_loss": val_loss, "singmos": mos_mean})

        val_loss = _evaluate(model, loss_fn, val_loader, device)
        logger.info("epoch=%d/%d val=%.4f (best=%.4f)", epoch + 1, int(args.epochs), val_loss, best_val)
        payload = _checkpoint_payload(
            model,
            loss_fn,
            opt,
            epoch=epoch + 1,
            step=global_step,
            val_loss=val_loss,
            phase=args.phase,
            seed=seed,
            cfg=cfg,
        )
        save_guarded(ckpt_dir / "checkpoint_latest.pt", payload)
        if val_loss < best_val:
            best_val = val_loss
            save_guarded(ckpt_dir / "checkpoint_best.pt", payload)

    report["final_val_loss"] = best_val
    report_path = ckpt_dir / f"train_report_{args.phase}.json"
    write_text_guarded(report_path, json.dumps(report, indent=2, ensure_ascii=False))
    logger.info("Fertig: best_val=%.4f — Report: %s", best_val, report_path)
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase", choices=("pretrain", "fine_tune", "domain_adapt"), default="pretrain")
    p.add_argument("--encoder", default="mert-v1-330m", help="MERT bleibt gefroren (§III.9 (copilot-instructions.md))")
    p.add_argument("--freeze-encoder", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--singmos-loss", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--data", default="", help="Komma-getrennte Manifest-JSONL/Ordner (synthetisch+real)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--preset", choices=("full", "tiny"), default=None)
    p.add_argument("--checkpoint-every", type=int, default=10000)
    p.add_argument("--steps-per-epoch", type=int, default=200)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--cpu", action="store_true", help="Erzwingt CPU-Inferenz/-Training")
    p.add_argument("--smoke", action="store_true", help="CPU-Ende-zu-Ende-Smoke auf synthetischen Sample-Vocals")
    args = p.parse_args(argv)
    if args.preset is None:
        args.preset = "tiny" if args.smoke else "full"
    return train(args)


if __name__ == "__main__":
    sys.exit(main())
