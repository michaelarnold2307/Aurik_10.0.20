"""F12 · BANQUET Real-Vinyl-Finetune — Trainingscode (SOTA4-6).

Schließt die Roadmap-Lücke ``F12 NEU — Real-Vinyl-Paare (Trainingscode fehlt
noch, SOTA4-6)`` (docs/TODOS_SOTA_ROADMAP.md): Finetune des 24-Zellen-BSRNN-
Kerns (``BanquetVinylCore``, §SOTA-ML-V5) auf Paaren echter Vinyl-Mitschnitte
↔ Referenz (Digital-Reissue). Ziel F12: realistisches Crackle/Rumble-Profil
entfernen, ohne die Musik zu färben (song-agnostisch, alle Import-Songs).

Architektur-Quelle (SOTA4-6 „Architektur-Re-Engineering aus dem Checkpoint"):
``backend/core/dsp/banquet_torch_rocm.py`` rekonstruiert den Kern 1:1 aus den
ONNX-Gewichten (Parität max|Δ| ≈ 1,9e-6 vs. ONNX). Der Kern ist ein normales
``torch.nn.Module`` und damit trainierbar — dieser Umstand macht F12 jetzt
überhaupt möglich.

Feature-Pipeline (Produktions-Parität): verbatim-Replica von
``plugins/banquet_vinyl_plugin.py::_prepare_input`` / ``_extract_output``.
Gemessene Skalen-Relation scipy↔torch bei nperseg=512 (parity-locked, Tests in
``tests/unit/test_train_banquet_vinyl_finetune.py``):
    scipy.signal.stft  = torch.stft  × 1/256
    scipy.signal.istft = torch.istft × 256
In der Rekonstruktionskette ``istft(mask × stft(x))`` heben sich beide Faktoren
auf; für die Feature-Parität wird der Faktor explizit mitgeführt.

Trainingsvertrag (ML-TRAININGS-SOTA-ROADMAP, docs/TODOS_SOTA_ROADMAP.md):
``ONNX-Export → Torch-ROCm-Paritäts-Gate (rel ≤ 1e-3) → A/B-Gate →
Witness-Belege → Hörordnungs-Abnahme → erst dann Feature-Flag-Flip``.
Dieses Skript endet mit einem Evidenz-Report — es flippt KEIN Flag und
tauscht KEIN Modell ungeprüft aus (§V7 (copilot-instructions.md)).

Loss: A1-Hör-Loss (``models/ear_vae_upstream/masking_loss.py``,
Muster ``scripts/train_diffwave_vocal_inpaint.py``) auf der Wellenform plus
Aux-L1 im Maskenraum gegen das Wiener-Zielmask (``_extract_output`` deutet den
Kern-Ausgang als „Wiener-like spectral mask" — dieselbe Semantik).

A/B-Gates (GPU-Finetune-Plan, docs/TODOS_SOTA_ROADMAP.md):
  - ΔSDR ≥ +2 dB vs. Zero-Shot-Baseline auf den destruktiven Fällen
  - Never-worsen auf sauberen Referenzen (ΔSDR ≥ −0,1 dB vs. Baseline)
Determinismus: Seeds je Session (§G5 (copilot-instructions.md)); alle
Entscheidungen aus Werten, nie aus ``time.time()``.

Usage (GPU 7900 XTX/ROCm, MUSDB/Vinyl-Korpus lokal):
    python scripts/train_banquet_vinyl_finetune.py --epochs 30 --seed 42
    python scripts/train_banquet_vinyl_finetune.py --pairs-dir /pfad/zu/realvinyl --epochs 50
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend.core.training_artifacts import save_guarded

logger = logging.getLogger(__name__)

# --- Signalvertrag (plugins/banquet_vinyl_plugin.py::_prepare_input) --------
TARGET_SR = 48000
N_FFT = 512
HOP = 375
N_BANDS = 128
N_FRAMES = 128
FEAT_DIM = 128
SEG_N = TARGET_SR  # 1-s-Segmente = 128 Frames bei hop 375
_SCIPY_STFT_SCALE = 1.0 / 256.0  # gemessene scipy↔torch-Relation (s. Docstring)
_SCIPY_ISTFT_GAIN = 256.0

# A/B-Gate-Schwellen (GPU-Finetune-Plan: ΔSDR ≥ +2 dB; Never-worsen-Floor)
GATE_DELTA_SDR_DB = 2.0
GATE_NEVER_WORSEN_DB = -0.1
GATE_PARITY_REL = 1e-3  # §III.9 (copilot-instructions.md) ONNX-Parität


# ---------------------------------------------------------------------------
# Daten: Paare echter Vinyl-Mitschnitte ↔ Referenz
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VinylPair:
    """(Vinyl-Rip, saubere Referenz) — Paarschlüssel ``vinyl_<genre>_<decade>``."""

    key: str
    damaged: Path
    clean: Path


@dataclass
class PairAudio:
    """Ausgerichtetes Paar, channels-first (C, N) — Stereo-Layout-Invariante."""

    key: str
    damaged: np.ndarray
    clean: np.ndarray
    lag: int = 0
    segments: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)


def pair_key(path: Path) -> str:
    """``vinyl_blues_1950s_crackle_hiss.wav`` → ``vinyl_blues_1950s``.

    Die ersten drei Tokens (``vinyl_<genre>_<decade>``) identifizieren den Song;
    die restlichen Tokens benennen die Schadenskette (crackle, hum_crackle,
    …_chain_vinyl_cassette, …).
    """
    parts = path.stem.split("_")
    return "_".join(parts[:3]) if len(parts) >= 3 else path.stem


def discover_pairs(corpus_dir: Path) -> list[VinylPair]:
    """Findet Paare in ``<corpus>/damaged`` ↔ ``<corpus>/clean/<key>_clean.wav``.

    Deterministische Sortierung; Paare ohne Referenz werden mit Warnung
    übersprungen (§V6 (copilot-instructions.md) — kein stilles Verschwinden).
    """
    damaged_dir = corpus_dir / "damaged"
    clean_dir = corpus_dir / "clean"
    if not damaged_dir.is_dir() or not clean_dir.is_dir():
        raise FileNotFoundError(f"Korpus-Layout erwartet: {damaged_dir} + {clean_dir}")

    pairs: list[VinylPair] = []
    skipped: list[str] = []
    for dmg in sorted(damaged_dir.glob("*.wav")):
        key = pair_key(dmg)
        ref = clean_dir / f"{key}_clean.wav"
        if ref.is_file():
            pairs.append(VinylPair(key=key, damaged=dmg, clean=ref))
        else:
            skipped.append(dmg.name)
    if skipped:
        logger.warning(
            "F12-Daten: %d Vinyl-Rips ohne Referenz übersprungen (z.B. %s)",
            len(skipped),
            skipped[0],
        )
    return pairs


def load_channels_first(path: Path) -> np.ndarray:
    """Lädt WAV → float32 (C, N) channels-first (Stereo-Layout-Invariante)."""
    try:
        import soundfile as sf

        data, sr = sf.read(str(path), always_2d=True, dtype="float32")
        audio = data.T.copy()
    except ImportError:
        from scipy.io import wavfile

        _wf = wavfile.read(str(path))
        if not isinstance(_wf, tuple) or len(_wf) < 2:  # Bug 12 (§V36): index-basiert statt Unpacking
            raise ValueError(f"Unerwartetes wavfile.read-Resultat fuer {path}") from None
        sr = int(_wf[0])
        data = np.asarray(_wf[1], dtype=np.float32) / 32768.0
        audio = data.T.copy() if data.ndim > 1 else data[np.newaxis, :].copy()
    if sr != TARGET_SR:
        from scipy.signal import resample_poly

        logger.info("F12: Resample %s (%d Hz → %d Hz, deterministisch)", path.name, sr, TARGET_SR)
        audio = resample_poly(audio, TARGET_SR, sr, axis=-1).astype(np.float32)
    return np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def align_pair(damaged: np.ndarray, clean: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Integer-Lag-Ausrichtung via FFT-Kreuzkorrelation (deterministisch).

    Echte Rip↔Reissue-Paare haben Aufnahme-Offsets; ohne Ausrichtung würde der
    Trainings-Loss Phasenversatz statt Knistern bestrafen. Rückgabe: beide
    Signale auf die Überlappung beschnitten plus der angewandte Lag.
    """
    a = damaged.mean(axis=0).astype(np.float64)
    b = clean.mean(axis=0).astype(np.float64)
    n = len(a) + len(b) + 1
    fa = np.fft.rfft(a, n)
    fb = np.fft.rfft(b, n)
    xcorr = np.fft.irfft(fa * np.conj(fb), n)
    peak = int(np.argmax(xcorr))
    # r_ab[l] = Sum_t a[t+l]·b[t]; negative Lags liegen zirkulaer bei n-|l|.
    lag = peak if peak <= n // 2 else peak - n
    if lag >= 0:
        d = damaged[:, lag:]
        c = clean[:, : damaged.shape[1] - lag]
    else:
        d = damaged[:, : damaged.shape[1] + lag]
        c = clean[:, -lag:]
    length = min(d.shape[1], c.shape[1])
    return d[:, :length], c[:, :length], lag


def make_segments(
    damaged: np.ndarray,
    clean: np.ndarray,
    rng: np.random.Generator,
    n_segments: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Zieht deterministisch 1-s-Segmente (C, SEG_N) aus dem ausgerichteten Paar."""
    length = damaged.shape[1]
    if length < SEG_N:
        return []
    segs: list[tuple[np.ndarray, np.ndarray]] = []
    for _ in range(n_segments):
        off = int(rng.integers(0, length - SEG_N + 1))
        segs.append((damaged[:, off : off + SEG_N], clean[:, off : off + SEG_N]))
    return segs


# ---------------------------------------------------------------------------
# Feature-Pipeline — Replica des Produktions-Pfads
# ---------------------------------------------------------------------------


def prepare_features_np(chunk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Verbatim-Replica ``BanquetVinylPlugin._prepare_input`` (Paritäts-Anker).

    chunk (C, N) → feat [1, 128, 128, 128] + stft_ctx [128, 128] (scipy-Skala).
    Nur zum Paritäts-Test; im Training läuft ``prepare_features_torch``.
    """
    from scipy.signal import stft as sci_stft

    mono = chunk.mean(axis=0).astype(np.float32)
    _, _, zxx = sci_stft(mono, nperseg=N_FFT, noverlap=N_FFT - HOP, boundary="zeros")
    n_frames = min(zxx.shape[1], N_FRAMES)
    stft_ctx = np.zeros((N_FRAMES, N_FRAMES), dtype=np.complex64)
    stft_ctx[:, :n_frames] = zxx[:N_BANDS, :n_frames].astype(np.complex64)
    band_real = stft_ctx.real.astype(np.float32)
    band_imag = stft_ctx.imag.astype(np.float32)
    nb_idx = np.minimum(2 * np.arange(N_BANDS, dtype=np.int64) + 1, zxx.shape[0] - 1)
    nb = zxx[nb_idx, :].astype(np.complex64)
    nb_real = np.zeros((N_BANDS, N_FRAMES), dtype=np.float32)
    nb_imag = np.zeros((N_BANDS, N_FRAMES), dtype=np.float32)
    nb_real[:, :n_frames] = nb[:, :n_frames].real.astype(np.float32)
    nb_imag[:, :n_frames] = nb[:, :n_frames].imag.astype(np.float32)
    band_feats = np.stack([band_real, band_imag, nb_real, nb_imag], axis=1)  # [128, 4, 128]
    feat = np.tile(band_feats, (1, 32, 1))[:, :FEAT_DIM, :]  # [128, 128, 128]
    feat = feat[np.newaxis, :, :, :]  # [1, 128, 128, 128]
    std = feat.std()
    if std > 1e-8:
        feat /= std
    return feat, stft_ctx


def stft_ctx_torch(chunk: np.ndarray, device, dtype) -> object:
    """scipy-kompatibler STFT-Kontext [128, 128] als Torch-Tensor (scipy-Skala)."""
    import torch

    mono = torch.from_numpy(chunk.mean(axis=0).astype(np.float32)).to(device)
    window = torch.hann_window(N_FFT, device=device)
    zxx = torch.stft(
        mono,
        n_fft=N_FFT,
        hop_length=HOP,
        win_length=N_FFT,
        window=window,
        center=True,
        pad_mode="constant",
        return_complex=True,
        normalized=False,
        onesided=True,
    )
    zxx = zxx * _SCIPY_STFT_SCALE  # scipy.signal.stft(scaling='spectrum')-Parität
    n_frames = min(zxx.shape[1], N_FRAMES)
    ctx = torch.zeros((N_BANDS, N_FRAMES), dtype=torch.complex64, device=device)
    ctx[:, :n_frames] = zxx[:N_BANDS, :n_frames]
    return ctx.to(dtype)


def prepare_features_torch(chunk: np.ndarray, device, dtype) -> object:
    """Differenzierbare Replica der Feature-Konstruktion (Parität zu np-Pfad)."""
    import torch

    ctx = stft_ctx_torch(chunk, device, torch.complex64)
    mono = torch.from_numpy(chunk.mean(axis=0).astype(np.float32)).to(device)
    window = torch.hann_window(N_FFT, device=device)
    zxx = torch.stft(
        mono,
        n_fft=N_FFT,
        hop_length=HOP,
        win_length=N_FFT,
        window=window,
        center=True,
        pad_mode="constant",
        return_complex=True,
        normalized=False,
        onesided=True,
    )
    zxx = zxx * _SCIPY_STFT_SCALE
    band_real = ctx.real.to(dtype)
    band_imag = ctx.imag.to(dtype)
    nb_idx = torch.minimum(
        2 * torch.arange(N_BANDS, device=device) + 1,
        torch.tensor(zxx.shape[0] - 1, device=device),
    )
    nb = zxx[nb_idx, :]
    n_frames = min(zxx.shape[1], N_FRAMES)
    nb_real = torch.zeros((N_BANDS, N_FRAMES), dtype=dtype, device=device)
    nb_imag = torch.zeros((N_BANDS, N_FRAMES), dtype=dtype, device=device)
    nb_real[:, :n_frames] = nb[:, :n_frames].real.to(dtype)
    nb_imag[:, :n_frames] = nb[:, :n_frames].imag.to(dtype)
    band_feats = torch.stack([band_real, band_imag, nb_real, nb_imag], dim=1)  # [128, 4, 128]
    feat = band_feats.tile(1, 32, 1)[:, :FEAT_DIM, :]  # [128, 128, 128]
    feat = feat.unsqueeze(0)  # [1, 128, 128, 128]
    std = feat.std()
    if float(std) > 1e-8:
        feat = feat / std
    return feat, ctx


def core_output_to_mask(raw: object) -> object:
    """Kern-Ausgang [B, 128, 128, 128] → Wiener-Maske [B, 128, 128] in (0, 1).

    Semantik identisch zu ``BanquetVinylPlugin._extract_output``:
    mean über hidden_dim (128) → sigmoid.
    """
    import torch

    return torch.sigmoid(raw.mean(dim=3))


def reconstruct_torch(mask: object, ctx: object, length: int = SEG_N) -> object:
    """Mask × stft_ctx → Wellenform (scipy-iSTFT-Parität, s. Docstring)."""
    import torch

    masked = ctx * mask.to(ctx.dtype)  # [B, 128, 128]
    full = torch.zeros((masked.shape[0], N_FFT // 2 + 1, N_FRAMES), dtype=torch.complex64, device=ctx.device)
    full[:, :N_BANDS, :] = masked
    window = torch.hann_window(N_FFT, device=ctx.device)
    # Natuerliche Ausgabelaenge ((N_FRAMES-1)*hop) verwenden: torch.istft mit
    # groesserem `length` teilt den Zero-Pad-Tail durch die Window-Envelope und
    # erzeugt dort Ausbrueche (gemessen max 11.6 vs. 0.12). Pad/trim wie das
    # Plugin (`_extract_output`) von Hand — identische Werte (Paritaetstest).
    audio = torch.istft(
        full,
        n_fft=N_FFT,
        hop_length=HOP,
        win_length=N_FFT,
        window=window,
        center=True,
    )
    audio = audio * _SCIPY_ISTFT_GAIN
    if audio.shape[-1] >= length:
        return audio[..., :length]
    return torch.nn.functional.pad(audio, (0, length - audio.shape[-1]))


def wiener_target_mask(damaged_ctx: object, clean_ctx: object) -> object:
    """Zielmask g = |C|² / (|C|² + |D−C|²), gedeckelt — Wiener-Semantik.

    D = Vinyl-Rip, C = Referenz; |D−C|² ist der Defekt-Anteil (Crackle,
    Rumble, …). Gegen dieses Ziel läuft der Aux-L1 im Maskenraum.
    """
    import torch

    c_pow = clean_ctx.abs() ** 2
    d_pow = (damaged_ctx - clean_ctx).abs() ** 2
    g = c_pow / (c_pow + d_pow + 1e-8)
    return torch.clamp(g, 1e-2, 1.0 - 1e-2)


def load_a1_masking_loss():
    """A1-Hör-Loss als uniforme ``(pred, target) -> loss``-Callable oder None.

    Reihenfolge (§V7 (copilot-instructions.md): keine Zweit-Implementierung):
    1. EAR-VAE-Rezept ``models/ear_vae_upstream/masking_loss.py`` (gitignored,
       Muster ``scripts/train_diffwave_vocal_inpaint.py``) — falls vorhanden.
    2. ``PsychoacousticMaskingLoss`` (ITU-R BS.1387, Bark + Spreading) aus dem
       Produktcode ``backend/core/optimization/perceptual_loss.py`` —
       maskierungs-informiert, torch/differenzierbar (SOTA-A1-Zielrichtung).
    Beides nicht verfügbar: None + Warnung (§V6 (copilot-instructions.md)).
    """
    import importlib.util

    loss_path = _ROOT / "models" / "ear_vae_upstream" / "masking_loss.py"
    if loss_path.is_file():
        try:
            spec = importlib.util.spec_from_file_location("masking_loss", loss_path)
            mod = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(mod)
            _recipe = mod.masking_loss

            def _ear_adapter(pred, target):
                return _recipe(pred, target, TARGET_SR)

            logger.info("A1-Hör-Loss: EAR-VAE-Rezept (%s)", loss_path)
            return _ear_adapter
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("A1-Hör-Loss nicht ladbar (%s) — weiche auf Produktcode aus", exc)
    try:
        from backend.core.optimization.perceptual_loss import PsychoacousticMaskingLoss

        _psy = PsychoacousticMaskingLoss(sr=TARGET_SR)

        def _psy_adapter(pred, target):
            loss, _details = _psy(pred, target)
            return loss

        logger.info("A1-Hör-Loss: PsychoacousticMaskingLoss (ITU-R BS.1387, Produktcode)")
        return _psy_adapter
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("A1-Hör-Loss fehlt (%s) — Aux-L1 im Maskenraum trägt den Loss", exc)
        return None


# ---------------------------------------------------------------------------
# Modell, Training, Gates
# ---------------------------------------------------------------------------


def load_core(device) -> object:
    """BANQUET-Kern (24 Zellen) mit Zero-Shot-ONNX-Gewichten, trainierbar.

    Nutzt die SOTA-ML-V5-Rekonstruktion aus ``banquet_torch_rocm.py``
    (Architektur-Re-Engineering aus dem Checkpoint, SOTA4-6). Die
    Lazy-Singleton-API dort ist GPU-only — hier wird der Builder direkt auf
    dem Ziel-Device benutzt (Training braucht vollen Parameter-Zugriff).
    """
    import torch

    from backend.core.dsp import banquet_torch_rocm as btr

    core = btr._build_core()
    core = core.to(device)
    core.train()
    for param in core.parameters():
        param.requires_grad_(True)
    logger.info("F12: BANQUET-Kern geladen (%d Parameter-Tensoren)", sum(1 for _ in core.parameters()))
    return core


def snr_db(est: np.ndarray, ref: np.ndarray) -> float:
    """Einfache SNR in dB (eps-sicher, §0a (copilot-instructions.md))."""
    est = np.nan_to_num(np.asarray(est, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    ref = np.nan_to_num(np.asarray(ref, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    noise = est - ref
    p_ref = float(np.mean(ref**2)) + 1e-12
    p_noise = float(np.mean(noise**2)) + 1e-12
    return 10.0 * float(np.log10(p_ref / p_noise))


@dataclass
class EpochStats:
    epoch: int
    train_loss: float
    val_loss: float


def _forward_loss(
    core,
    feat: object,
    ctx: object,
    clean_wave: object,
    target_mask: object,
    a1_loss_fn,
    masking_beta: float,
    device,
    dtype,
) -> object:
    import torch

    raw = core(feat)
    mask = core_output_to_mask(raw)
    aux = torch.mean(torch.abs(mask - target_mask))
    loss = 0.25 * aux
    if a1_loss_fn is not None:
        recon = reconstruct_torch(mask, ctx, SEG_N)
        loss = loss + masking_beta * a1_loss_fn(recon.unsqueeze(1), clean_wave.unsqueeze(1))
    return torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0), mask


def train(
    core,
    train_segments: list[tuple[str, np.ndarray, np.ndarray]],
    val_segments: list[tuple[str, np.ndarray, np.ndarray]],
    args: argparse.Namespace,
    device,
    dtype,
) -> list[EpochStats]:
    """AdamW + Early-Stop auf val_loss; Seeds je Session (§G5 (copilot-instructions.md))."""
    import torch

    a1_loss_fn = None if args.skip_a1 else load_a1_masking_loss()
    optimizer = torch.optim.AdamW(core.parameters(), lr=args.lr, weight_decay=1e-4)
    best_val = float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in core.state_dict().items()}
    bad_epochs = 0
    history: list[EpochStats] = []
    step_rng = np.random.default_rng(args.seed + 1)

    for epoch in range(args.epochs):
        core.train()
        order = step_rng.permutation(len(train_segments))
        losses = []
        for idx in order:
            _, dmg, cln = train_segments[int(idx)]
            feat_np, _ = prepare_features_np(dmg)
            _, dmg_ctx = prepare_features_torch(dmg, device, dtype)
            _, cln_ctx = prepare_features_torch(cln, device, dtype)
            feat = torch.from_numpy(feat_np).to(device=device, dtype=dtype)
            feat.requires_grad_(False)
            target_mask = wiener_target_mask(dmg_ctx, cln_ctx)
            clean_wave = torch.from_numpy(cln.mean(axis=0).astype(np.float32)).to(device)
            loss, _mask = _forward_loss(
                core, feat, dmg_ctx, clean_wave, target_mask, a1_loss_fn, args.masking_beta, device, dtype
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(core.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        train_loss = float(np.mean(losses)) if losses else float("nan")

        core.eval()
        val_losses = []
        with torch.no_grad():
            for _, dmg, cln in val_segments:
                feat_np, _ = prepare_features_np(dmg)
                _, dmg_ctx = prepare_features_torch(dmg, device, dtype)
                _, cln_ctx = prepare_features_torch(cln, device, dtype)
                feat = torch.from_numpy(feat_np).to(device=device, dtype=dtype)
                target_mask = wiener_target_mask(dmg_ctx, cln_ctx)
                clean_wave = torch.from_numpy(cln.mean(axis=0).astype(np.float32)).to(device)
                loss, _mask = _forward_loss(
                    core, feat, dmg_ctx, clean_wave, target_mask, a1_loss_fn, args.masking_beta, device, dtype
                )
                val_losses.append(float(loss.detach().cpu()))
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        history.append(EpochStats(epoch=epoch, train_loss=train_loss, val_loss=val_loss))
        logger.info("F12 Epoch %d: train=%.5f val=%.5f", epoch, train_loss, val_loss)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in core.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                logger.info("F12 Early-Stop nach %d Epochen ohne Verbesserung", args.patience)
                break

    core.load_state_dict(best_state)
    return history


def run_model_on_segment(core, seg: np.ndarray, device, dtype) -> np.ndarray:
    """Ein Segment (C, N) durch Kern + Maske + Rekonstruktion (mono, numpy)."""
    import torch

    feat_np, _ = prepare_features_np(seg)
    with torch.no_grad():
        feat = torch.from_numpy(feat_np).to(device=device, dtype=dtype)
        _, ctx = prepare_features_torch(seg, device, dtype)
        mask = core_output_to_mask(core(feat))
        recon = reconstruct_torch(mask, ctx, SEG_N)
    return np.nan_to_num(recon[0].cpu().numpy(), nan=0.0, posinf=0.0, neginf=0.0)


def evaluate(
    core,
    baseline,
    eval_pairs: list[PairAudio],
    device,
    dtype,
    rng: np.random.Generator,
) -> dict[str, object]:
    """A/B-Gates: ΔSDR vs. Zero-Shot-Baseline + Never-worsen auf sauber.

    Zero-Shot-Baseline = dasselbe Modul mit den unveränderten ONNX-Gewichten
    (Kopie vor dem Training — nie eine zweite Modell-Datei).
    """
    destructive: list[dict[str, float]] = []
    clean_cases: list[dict[str, float]] = []
    for pair in eval_pairs:
        for dmg, cln in pair.segments:
            ref = cln.mean(axis=0)
            if np.allclose(dmg, cln):
                est_base = run_model_on_segment(baseline, dmg, device, dtype)
                est_fine = run_model_on_segment(core, dmg, device, dtype)
                clean_cases.append(
                    {
                        "sdr_base": snr_db(est_base, ref),
                        "sdr_finetuned": snr_db(est_fine, ref),
                    }
                )
            else:
                est_in = dmg.mean(axis=0)
                est_base = run_model_on_segment(baseline, dmg, device, dtype)
                est_fine = run_model_on_segment(core, dmg, device, dtype)
                destructive.append(
                    {
                        "sdr_input": snr_db(est_in, ref),
                        "sdr_base": snr_db(est_base, ref),
                        "sdr_finetuned": snr_db(est_fine, ref),
                    }
                )
        rng.bit_generator.random_raw(1)  # RNG-Fortschritt deterministisch je Paar

    delta_destructive = [c["sdr_finetuned"] - c["sdr_base"] for c in destructive]
    delta_clean = [c["sdr_finetuned"] - c["sdr_base"] for c in clean_cases]
    mean_delta = float(np.mean(delta_destructive)) if delta_destructive else float("nan")
    min_delta_clean = float(np.min(delta_clean)) if delta_clean else float("nan")
    return {
        "destructive": destructive,
        "clean": clean_cases,
        "mean_delta_sdr_db": mean_delta,
        "min_delta_clean_db": min_delta_clean,
        "gate_delta_sdr": bool(delta_destructive and mean_delta >= GATE_DELTA_SDR_DB),
        "gate_never_worsen": bool(delta_clean and min_delta_clean >= GATE_NEVER_WORSEN_DB),
    }


def export_core_onnx(core, out_path: Path, device) -> tuple[Path, float]:
    """ONNX-Export des Kerns + Torch↔ONNX-CPU-Paritäts-Gate (rel ≤ 1e-3).

    §III.9 (copilot-instructions.md): ONNX-EPs außer CPU nur mit
    Paritäts-Nachweis; hier ist ONNX-CPU die Referenz, der Torch-Kern der
    Ziel-Pfad (wie im Produktions-Pattern SOTA-ML-V5).
    """
    import torch

    out_path.parent.mkdir(parents=True, exist_ok=True)
    core = core.to(device).eval()
    dummy = torch.zeros((1, N_BANDS, N_FRAMES, FEAT_DIM), device=device)
    torch.onnx.export(
        core,
        dummy,
        str(out_path),
        input_names=["feat"],
        output_names=["raw"],
        dynamic_axes={"feat": {0: "batch"}, "raw": {0: "batch"}},
        opset_version=17,
    )
    rel = float("inf")
    try:
        import onnxruntime as ort

        rng = np.random.default_rng(7)
        feed = rng.standard_normal((2, N_BANDS, N_FRAMES, FEAT_DIM)).astype(np.float32)
        with torch.no_grad():
            ref = core(torch.from_numpy(feed).to(device)).cpu().numpy()
        sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
        out = sess.run(["raw"], {"feat": feed})[0]
        denom = np.maximum(np.abs(ref).max(), 1e-8)
        rel = float(np.abs(ref - out).max() / denom)
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("F12-Parität ONNX-CPU nicht messbar: %s", exc)
    logger.info("F12 ONNX-Parität rel=%.3e (Gate ≤ %.0e)", rel, GATE_PARITY_REL)
    return out_path, rel


def main() -> int:
    ap = argparse.ArgumentParser(description="F12 BANQUET Real-Vinyl-Finetune (SOTA4-6)")
    ap.add_argument("--corpus", default="corpus/vinyl", help="Vinyl-Korpus (damaged/ + clean/)")
    ap.add_argument("--pairs-dir", default=None, help="Optionale Real-Paare (damaged/ + clean/)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-segments", type=int, default=16, help="Segmente je Paar und Epoche")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42, help="Session-Seed (§G5 (copilot-instructions.md))")
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--masking-beta", type=float, default=0.5)
    ap.add_argument("--skip-a1", action="store_true", help="A1-Hör-Loss deaktivieren")
    ap.add_argument("--val-frac", type=float, default=0.2, help="Val-Anteil auf Paar-Ebene")
    ap.add_argument("--no-align", action="store_true", help="Paar-Ausrichtung überspringen")
    ap.add_argument("--out-dir", default="output/banquet_f12")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Seeds je Session (§G5 (copilot-instructions.md))
    random.seed(args.seed)
    np.random.seed(args.seed)
    try:
        import torch

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Torch-Seed-Setup unvollständig: %s", exc)

    try:
        import torch
    except ImportError:
        logger.error("PyTorch fehlt — F12-Training braucht torch (GPU-Finetune).")
        return 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        logger.warning("F12: CPU-Modus ist nur Smoke/Test — das Finetune ist GPU-gebunden (7900 XTX/ROCm).")
    dtype = torch.float32

    corpus_dir = Path(args.corpus)
    pairs = discover_pairs(corpus_dir)
    if args.pairs_dir:
        pairs += discover_pairs(Path(args.pairs_dir))
    if not pairs:
        logger.error("Keine Vinyl-Paare gefunden — abbruch.")
        return 2

    # Paar-Ebene splitten (kein Segment-Leakage), deterministisch.
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(pairs))
    n_val = max(1, int(round(len(pairs) * args.val_frac))) if len(pairs) > 1 else 0
    val_idx = {int(i) for i in order[:n_val]}

    train_segments: list[tuple[str, np.ndarray, np.ndarray]] = []
    eval_pairs: list[PairAudio] = []
    for i, pair in enumerate(pairs):
        dmg = load_channels_first(pair.damaged)
        cln = load_channels_first(pair.clean)
        if not args.no_align:
            dmg, cln, lag = align_pair(dmg, cln)
        else:
            lag = 0
        length = min(dmg.shape[1], cln.shape[1])
        dmg, cln = dmg[:, :length], cln[:, :length]
        audio_pair = PairAudio(key=pair.key, damaged=dmg, clean=cln, lag=lag)
        segs = make_segments(dmg, cln, rng, args.batch_segments)
        audio_pair.segments = segs
        eval_pairs.append(audio_pair)
        if i not in val_idx:
            train_segments.extend((pair.key, s_d, s_c) for s_d, s_c in segs)
        logger.info("F12 Paar %s: %d Segmente, lag=%d", pair.key, len(segs), lag)

    val_segments: list[tuple[str, np.ndarray, np.ndarray]] = [
        (p.key, s_d, s_c) for i, p in enumerate(eval_pairs) if i in val_idx for s_d, s_c in p.segments
    ]
    if not train_segments:
        logger.error("Keine Trainings-Segmente — Korpus zu kurz?")
        return 2

    core = load_core(device)
    baseline = load_core(device)  # Zero-Shot-Kopie (unveränderte Gewichte)
    baseline.eval()
    for param in baseline.parameters():
        param.requires_grad_(False)

    history = train(core, train_segments, val_segments, args, device, dtype)

    gates = evaluate(core, baseline, eval_pairs, device, dtype, rng)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "banquet_core_f12.pt"
    save_guarded(str(ckpt_path), core.state_dict())
    onnx_path, rel = export_core_onnx(core, out_dir / "banquet_core_f12.onnx", device)
    gates["gate_parity"] = bool(rel <= GATE_PARITY_REL)
    gates["parity_rel"] = rel

    report = {
        "task": "F12",
        "source": "docs/TODOS_SOTA_ROADMAP.md (ML-TRAININGS-SOTA-ROADMAP, SOTA4-6)",
        "seed": args.seed,
        "pairs": [p.key for p in eval_pairs],
        "n_train_segments": len(train_segments),
        "n_val_segments": len(val_segments),
        "history": [vars(h) for h in history],
        "gates": gates,
        "artifacts": {"state_dict": str(ckpt_path), "onnx": str(onnx_path)},
        "next_steps_contract": [
            "A/B-Gate auf echten Songs (scripts/ab_test_exports.py, Hörordnungs-Abnahme)",
            "Witness-Belege (Resemblyzer cos >= 0.92 für Vokal-Identität, MuQ-MOS nicht schlechter als Baseline)",
            "Erst DANACH manueller Feature-Flag-/Modell-Rollout — dieses Skript flippt nichts",
        ],
    }
    report_path = out_dir / f"report_f12_seed{args.seed}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("F12 Report: %s — Gates: %s", report_path, json.dumps(gates.get("gate_delta_sdr"), default=str))

    if gates["gate_delta_sdr"] and gates["gate_never_worsen"] and gates["gate_parity"]:
        logger.info("F12 Trainings-Gates BESTANDEN — A/B-Abnahme und Witness folgen (Vertrag).")
        return 0
    logger.warning("F12 Trainings-Gates NICHT vollständig bestanden — siehe Report, kein Modell-Wechsel.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
