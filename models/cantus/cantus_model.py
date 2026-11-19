"""
Cantus — MERT-conditioned Multi-Scale Flow-Matching DiT für Gesangsrestaurierung.

Architektur (Kernprinzipien des Cantus-Designs):
  1. Music-aware Feature Extraction: MERT-v1-330M-Features (1024-d, frame-level,
     gemessen an models/mert/mert.onnx) als Konditionierung (trainingsseitig und
     Graph-Rand — MERT selbst bleibt gefroren bzw. ONNX-außerhalb, §III.9).
  2. Flow-Matching Generierung: wiederverwendet die miipher_dit-Infrastruktur
     (models/miipher_dit/dit_model.py — DiTBlock/AdaLNZero/SinusoidalTimeEmbedding).
  3. Multi-Scale Processing: komplementäre FIR-Bandaufteilung (Cutoff ≈ 2 kHz)
     in Low-Freq-(harmonisch) und High-Freq-(Transienten/Rauschen)-Pfad.
     Low: Transformer-Tokens mit Temporal Attention (langes musikalisches
     Fenster). High: lokale Convolutional-Blöcke (feine zeitliche Details).
  4. Musikalische Kontext-Konditionierung: Pitch-Track (f0 + Voicing) frameweise
     auf Tokens, MERT-Sequence als Condition-Tokens (Self-Attention über
     Audio+Condition = Temporal-Attention auf musikalischen Kontext),
     Harmonic Context (MuQ-MuLan, 768-d) global auf das Time-Embedding.

Training (Flow-Matching, konstruktive Reproduzierbarkeit §G5 (GEBOTE.md)):
    x_t = (1-t) * x_degraded + t * x_clean
    v   = x_clean - x_degraded            (OT-Geschwindigkeitsfeld)
    model(x_t, t, cond) → v̂ ≈ v
Inferenz (t=0.5):  ŷ = x + (1-t) · v̂

ONNX-Schnittstelle (CantusExportWrapper, OpSet 14+, dynamische Achsen):
    x        : [B, T, 1] float32 — degradierter Vocal (48 kHz)
    t        : [B]       float32 — Flow-Zeit (0..1)
    mert     : [B, F, 1024] float32 — MERT-v1-330M Features
    pitch    : [B, F, 2]  float32 — (log2-f0-normiert, voiced_prob)
    harm     : [B, 768]  float32 — MuQ-MuLan Embedding (L2-normiert)
    use_cond : [B]       float32 — 1.0 = Konditionierung, 0.0 = Null-Tokens
    Output   : [B, T, 1] float32 — v̂ (Velocity)

Bedingungs-Dropout (p ≈ 0.1 im Training) trainiert den unbedingten Pfad mit —
damit kann die Inferenz deterministisch zwischen bedingt/unbedingt wählen
(classifier-free-guidance-fähig), wenn Feature-Extractor fehlen (§V6
(VERBOTEN.md): dokumentierter Ersatzpfad statt Stillfehler).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Wiederverwendung der miipher_dit-Infrastruktur (keine Symbol-Duplikate §G…)
from models.miipher_dit.dit_model import AdaLNZero, DiTBlock, SinusoidalTimeEmbedding

logger = logging.getLogger(__name__)


# ── Konfiguration ────────────────────────────────────────────────────────────


@dataclass
class CantusConfig:
    """Hyperparameter des Cantus-Modells (Spiegel von cantus_config.json)."""

    dim: int = 768
    depth: int = 18
    heads: int = 12
    patch_size: int = 256  # Low-Freq-Pfad (Tempo-Auflösung 256 Samples ≈ 5.3 ms)
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    # Multi-Scale Bandaufteilung
    sample_rate: int = 48000
    band_cutoff_hz: float = 2000.0
    band_kernel_size: int = 513  # ungerade FIR-Länge (Hamming-windowed sinc)
    high_hidden: int = 128  # Conv-Pfad Breite
    high_depth: int = 6  # Conv-Pfad Tiefe (lokale Convolutional-Blöcke)
    high_kernel: int = 7
    # Konditionierung
    mert_dim: int = 1024  # MERT-v1-330M (models/mert/mert.onnx, gemessene Breite)
    pitch_dim: int = 2  # (log2-f0 normiert, voicing)
    harmonic_dim: int = 768  # MuQ-MuLan
    cond_dropout: float = 0.1  # Bedingungs-Dropout (Null-Tokens) im Training
    max_tokens: int = 4096

    @classmethod
    def from_dict(cls, data: dict) -> "CantusConfig":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_dict(self) -> dict:
        return {
            "dim": self.dim,
            "depth": self.depth,
            "heads": self.heads,
            "patch_size": self.patch_size,
            "mlp_ratio": self.mlp_ratio,
            "dropout": self.dropout,
            "sample_rate": self.sample_rate,
            "band_cutoff_hz": self.band_cutoff_hz,
            "band_kernel_size": self.band_kernel_size,
            "high_hidden": self.high_hidden,
            "high_depth": self.high_depth,
            "high_kernel": self.high_kernel,
            "mert_dim": self.mert_dim,
            "pitch_dim": self.pitch_dim,
            "harmonic_dim": self.harmonic_dim,
            "cond_dropout": self.cond_dropout,
            "max_tokens": self.max_tokens,
        }


# ── Multi-Scale: komplementäre FIR-Bandaufteilung ────────────────────────────


class FIRBandSplit(nn.Module):
    """Komplementäre Bandaufteilung: low = FIR-Tiefpass(x), high = x − low.

    Die komplementäre Konstruktion garantiert perfekte Rekonstruktion
    (low + high ≡ x) — phasenkorrekt und ONNX-sicher (faltungsfixe Buffer).
    Low = Grundton/Harmonische, High = Sibilanten/Transienten/Rauschen.
    """

    def __init__(self, kernel_size: int = 513, cutoff_hz: float = 2000.0, sample_rate: int = 48000):
        super().__init__()
        assert kernel_size % 2 == 1, "band_kernel_size muss ungerade sein"
        self.pad = kernel_size // 2
        # Hamming-windowed-sinc-Tiefpass, deterministisch konstruiert (§G5 (GEBOTE.md))
        n = torch.arange(kernel_size, dtype=torch.float64) - self.pad
        fc = cutoff_hz / sample_rate  # normierte Cutoff-Frequenz
        h = 2.0 * fc * torch.sinc(2.0 * fc * n)
        window = torch.hamming_window(kernel_size, periodic=False, dtype=torch.float64)
        h = h * window
        h = (h / h.sum()).float()
        self.register_buffer("lowpass", h.view(1, 1, kernel_size))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: [B, T, 1] → (low [B, T, 1], high [B, T, 1])"""
        xs = x.transpose(1, 2)  # [B, 1, T]
        low = F.conv1d(F.pad(xs, (self.pad, self.pad), mode="reflect"), self.lowpass)
        low = low.transpose(1, 2)
        return low, x - low


# ── High-Freq-Pfad: lokale Convolutional-Blöcke ──────────────────────────────


class LocalConvBlock(nn.Module):
    """Depthwise-separabler Conv-Block mit FiLM-Time-Konditionierung.

    Zero-init auf der Ausgangsprojektion → Residualpfad startet als Identität
    (AdaLN-Zero-Prinzip, Peebles & Xie 2023).
    """

    def __init__(self, dim: int, kernel: int = 7, cond_dim: int = 768):
        super().__init__()
        self.norm = nn.GroupNorm(1, dim)  # über Zeit (layout-stabil [B, C, T])
        self.depthwise = nn.Conv1d(dim, dim, kernel, padding=kernel // 2, groups=dim)
        self.pointwise = nn.Conv1d(dim, dim, 1)
        self.film = nn.Linear(cond_dim, dim * 2)  # scale, shift
        self.out_proj = nn.Conv1d(dim, dim, 1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """h: [B, C, T], cond: [B, cond_dim]"""
        residual = h
        h = self.depthwise(self.norm(h))
        h = F.gelu(h, approximate="tanh")
        h = self.pointwise(h)
        scale, shift = self.film(cond).chunk(2, dim=-1)  # je [B, C]
        h = h * (1 + scale.unsqueeze(-1)) + shift.unsqueeze(-1)
        return residual + self.out_proj(h)


class LocalConvPath(nn.Module):
    """High-Freq-Pfad: faltungslokale Verarbeitung für Transienten/Rauschen."""

    def __init__(self, hidden: int = 128, depth: int = 6, kernel: int = 7, cond_dim: int = 768):
        super().__init__()
        self.in_proj = nn.Conv1d(1, hidden, 15, padding=7)
        self.blocks = nn.ModuleList(LocalConvBlock(hidden, kernel, cond_dim) for _ in range(depth))
        self.out_norm = nn.GroupNorm(1, hidden)
        self.out_proj = nn.Conv1d(hidden, 1, 15, padding=7)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, high: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """high: [B, T, 1], cond: [B, cond_dim] → [B, T, 1]"""
        h = F.gelu(self.in_proj(high.transpose(1, 2)), approximate="tanh")
        for block in self.blocks:
            h = block(h, cond)
        return self.out_proj(F.gelu(self.out_norm(h), approximate="tanh")).transpose(1, 2)


# ── Cantus Hauptmodell ───────────────────────────────────────────────────────


class CantusModel(nn.Module):
    """Multi-Scale Flow-Matching DiT mit musikalischer Kontext-Konditionierung.

    Low-Pfad:  Patch-Tokens (patch_size=256) + MERT-Condition-Tokens entlang der
               Sequenz konkatiniert → DiT-Self-Attention wickelt Audio↔Condition
               und langen musikalischen Kontext gemeinsam ab (Temporal Attention).
    High-Pfad: lokale Convolutional-Blöcke auf dem Hochfrequenzband.
    Fusion:    v̂ = v_low + v_high (komplementäre Bänder).
    """

    def __init__(self, config: Optional[CantusConfig] = None, **overrides):
        super().__init__()
        cfg = config or CantusConfig()
        for key, value in overrides.items():
            if not hasattr(cfg, key):
                raise TypeError(f"Unbekannte Konfig-Option: {key}")
            setattr(cfg, key, value)
        self.config = cfg
        dim = cfg.dim
        assert dim % cfg.heads == 0, "dim muss durch heads teilbar sein"

        # Zeit-Konditionierung (miipher_dit-Baustein)
        self.time_embed = SinusoidalTimeEmbedding(dim)

        # Konditionierungs-Encoder (trainierbare Adapter auf gefrorene Features)
        self.mert_proj = nn.Linear(cfg.mert_dim, dim, bias=False)
        self.mert_norm = nn.LayerNorm(dim)
        self.pitch_proj = nn.Linear(cfg.pitch_dim, dim, bias=False)
        self.harm_proj = nn.Linear(cfg.harmonic_dim, dim, bias=False)

        # Gelernte Null-Tokens für unbedingten Betrieb (§V6 (VERBOTEN.md))
        self.null_mert = nn.Parameter(torch.zeros(1, 1, dim))
        self.null_harm = nn.Parameter(torch.zeros(1, dim))
        self.type_audio = nn.Parameter(torch.zeros(1, 1, dim))
        self.type_mert = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.null_mert, std=0.02)
        nn.init.trunc_normal_(self.null_harm, std=0.02)
        nn.init.trunc_normal_(self.type_audio, std=0.02)
        nn.init.trunc_normal_(self.type_mert, std=0.02)

        # Multi-Scale Bandaufteilung (feste FIR-Buffer)
        self.band_split = FIRBandSplit(cfg.band_kernel_size, cfg.band_cutoff_hz, cfg.sample_rate)

        # High-Pfad (lokal konvolutorial)
        self.high_path = LocalConvPath(cfg.high_hidden, cfg.high_depth, cfg.high_kernel, dim)

        # Low-Pfad: Patch-Embedding (wie miipher_dit, kernel=2·stride → 50 % Überlapp)
        self.patch_embed = nn.Conv1d(
            1, dim, kernel_size=cfg.patch_size * 2, stride=cfg.patch_size,
            padding=cfg.patch_size // 2, bias=False,
        )
        self.pos_embed = nn.Parameter(torch.randn(1, cfg.max_tokens, dim) * 0.02)

        # Temporal-Attention-Blöcke (wiederverwendet miipher_dit.DiTBlock)
        self.blocks = nn.ModuleList(
            DiTBlock(dim, cfg.heads, cfg.mlp_ratio, cfg.dropout) for _ in range(cfg.depth)
        )
        self.final_ada = AdaLNZero(dim)
        self.output_proj = nn.Linear(dim, cfg.patch_size)
        nn.init.trunc_normal_(self.output_proj.weight, std=0.02)
        nn.init.zeros_(self.output_proj.bias)

        self._init_trainable_weights()

    def _init_trainable_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if module.weight.requires_grad and torch.count_nonzero(module.weight.detach()).item() == 0 and module.bias is not None:
                    continue  # Zero-init-Pfade bewusst unangetastet lassen
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.Conv1d,)):
                if module.weight.requires_grad and torch.count_nonzero(module.weight.detach()).item() == 0:
                    continue  # Zero-init-Pfade (out_proj) bleiben Nullen
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                if module.elementwise_affine:
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)

    # ── Hilfsfunktionen ──────────────────────────────────────────────────

    def _apply_cond_mask(self, cond: torch.Tensor, null: torch.Tensor, use_cond: torch.Tensor) -> torch.Tensor:
        """Mischt Condition und Null-Token über use_cond ∈ {0,1}^B (ONNX-sicher).

        Im Training wird use_cond zusätzlich über cond_dropout zufällig gesetzt
        (Aufrufer-Verantwortung) — hier bleibt der Graph deterministisch.
        """
        eff = use_cond.view(-1, 1, 1) if cond.dim() == 3 else use_cond.view(-1, 1)
        return cond * eff + null * (1.0 - eff)

    def _patchify(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        """[B, T, 1] → Tokens [B, N, dim]; Länge auf Vielfaches von patch_size paden."""
        _b, t_orig, _c = x.shape
        pad = (self.config.patch_size - t_orig % self.config.patch_size) % self.config.patch_size
        x = F.pad(x, (0, 0, 0, pad))
        tokens = self.patch_embed(x.transpose(1, 2)).transpose(1, 2)
        return tokens, t_orig, tokens.shape[1]

    def _unpatchify(self, tokens: torch.Tensor, original_length: int) -> torch.Tensor:
        """Tokens [B, N, dim] → Wellenform [B, T, 1] (auf Original-Länge getrimmt)."""
        b, n, _d = tokens.shape
        x = self.output_proj(tokens).reshape(b, n * self.config.patch_size)
        return x[:, :original_length].unsqueeze(-1)

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        mert: torch.Tensor,
        pitch: torch.Tensor,
        harm: torch.Tensor,
        use_cond: torch.Tensor,
    ) -> torch.Tensor:
        """Velocity-Schätzung v̂ [B, T, 1].

        Args:
            x:     [B, T, 1] degradiertes Audio bei Flow-Zeit t
            t:     [B] Flow-Zeit ∈ [0, 1]
            mert:  [B, F, cfg.mert_dim] MERT-Features (gefüllt ignoriert bei use_cond=0)
            pitch: [B, F, cfg.pitch_dim] Pitch-Track (log2-f0-normiert, voicing)
            harm:  [B, cfg.harmonic_dim] MuQ-MuLan Embedding
            use_cond: [B] 1.0 = Konditionierung nutzen, 0.0 = Null-Tokens
        """
        cfg = self.config
        if t.dim() == 2:
            t = t.squeeze(-1)

        # Bedingungs-Encoder mit Null-Token-Mischung
        mert_tokens = self.mert_norm(self.mert_proj(mert)) + self.type_mert
        mert_tokens = self._apply_cond_mask(mert_tokens, self.null_mert + self.type_mert, use_cond)
        harm_ctx = self.harm_proj(harm) + self.null_harm
        harm_ctx = self._apply_cond_mask(harm_ctx, self.null_harm, use_cond)
        pitch_tok = self.pitch_proj(pitch)  # [B, F, dim] (Nullen bleiben Nullen)

        t_emb = self.time_embed(t) + harm_ctx  # [B, dim] — globaler harmonischer Kontext

        # Multi-Scale Bandaufteilung (komplementär: low + high ≡ x)
        low, high = self.band_split(x)

        # High-Pfad: lokale Convolutional-Blöcke
        v_high = self.high_path(high, t_emb)

        # Low-Pfad: Temporal-Attention über Audio- + MERT-Condition-Tokens
        tokens, t_orig, n_tokens = self._patchify(low)
        pos = F.interpolate(
            self.pos_embed.transpose(1, 2), size=n_tokens, mode="linear", align_corners=False
        ).transpose(1, 2)
        # Pitch frame-weise auf Audio-Tokens projizieren (Zeit-Interpolation)
        pitch_on_tokens = (
            F.interpolate(pitch_tok.transpose(1, 2), size=n_tokens, mode="linear", align_corners=False)
            .transpose(1, 2)
            * use_cond.view(-1, 1, 1)
        )
        audio_tokens = tokens + pos + self.type_audio + pitch_on_tokens

        seq = torch.cat([audio_tokens, mert_tokens], dim=1)
        for block in self.blocks:
            seq = block(seq, t_emb)

        seq = seq[:, :n_tokens]  # Condition-Tokens abschneiden
        seq, gate = self.final_ada(seq, t_emb)
        seq = gate.unsqueeze(1) * seq
        v_low = self._unpatchify(seq, t_orig)

        return v_low + v_high

    @property
    def num_params(self) -> float:
        return sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6


# ── ONNX-Export-Wrapper ──────────────────────────────────────────────────────


class CantusExportWrapper(nn.Module):
    """Normalisiert die ONNX-Schnittstelle des Cantus-Modells.

    Sichert float32-Tensoren, 1-D t [B] und konsistente Bedingungs-Dummies
    (use_cond=0 → Null-Tokens, unbedingter Pfad).
    """

    def __init__(self, model: CantusModel):
        super().__init__()
        self.model = model

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, mert: torch.Tensor, pitch: torch.Tensor,
        harm: torch.Tensor, use_cond: torch.Tensor,
    ) -> torch.Tensor:
        if t.dim() > 1:
            t = t.squeeze(-1)
        return self.model(x, t, mert, pitch, harm, use_cond)


# ── Factory ──────────────────────────────────────────────────────────────────


def create_cantus(config: Optional[CantusConfig] = None, **kwargs) -> CantusModel:
    """Erzeugt ein CantusModel (Default: cantus_config „full“, 18L/768-d/12H)."""
    return CantusModel(config=config, **kwargs)
