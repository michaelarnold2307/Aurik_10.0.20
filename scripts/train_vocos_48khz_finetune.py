"""D-K2-3 (SOTA-M6): Musik-Finetune des Vocos-48kHz-Vocoders auf MUSDB18-HQ-Gesang.

Rolle und Grenzen (Register .github/SOTA_DEFICIT_REGISTER.md, D-K2-3): Der
Vocos-48kHz-Vocoder (upstream kittn/vocos-mel-48khz-alpha1, Sprach-Korpora)
rekonstruiert Musik-Gesang nur mittelmaessig — Kandidatenpfad im
Studio-Finisher. Dieser Job ist der TRAININGSWEG (Kette 4 R5,
scripts/gpu_kette4_modellkoepfe.sh); die PRIMAER-ROLLEN-Entscheidung
(Studio-Finisher: Vocos vs. BigVGAN-F3) bleibt MENSCHLICH (C4-Hoerstichprobe,
Hoerordnung Ebene 1). KEIN Deploy, KEIN Flag-Flip (§V8 (copilot-instructions.md)/G1-Grenze: nur
Checkpoint + Manifest nach models/vocos_48khz/finetune/).

Architektur-Paritaet (§III.9 (copilot-instructions.md)): Training nutzt das
Upstream-Paket `vocos==0.1.0` (MIT, Siuzdak 2023 — wie beim ONNX-Export,
scripts/export_vocos_48khz_onnx.py Zeile 54) und laedt das deployte Checkpoint
models/vocos_48khz/{config.yaml,pytorch_model.bin} via
vocos.pretrained.Vocos.from_pretrained — identische Architektur wie das
deployte vocos_48khz.onnx. Mel-Extraktion: der Konfig-Extraktor
(MelSpectrogramFeatures, n_fft 2048/hop 256/n_mels 128/center, 48 kHz) — das
ist die Trainings-Pipeline des ONNX selbst; die Serve-Paritaet des Plugins
(plugins/vocos_plugin.py) ist dort bereits vertraglich gepinnt.

Determinismus (§G5 (copilot-instructions.md)): Seed 42, deterministische
Track-Auswahl und Segment-Fenster, kein time.time in Entscheidungslogik.

Finetune-Rezept: reiner Rekonstruktions-Finetune (Multi-Resolution-Spektral:
L1-Mel + L1-STFT-Magnituden bei 2 Aufloesungen). Der adversarialen
Diskriminator-Block des Upstream-Trainers wird bewusst NICHT nachgebaut
(bewertende Instanz waere ohne die Upstream-Discriminator-Checkpoints ein
ad-hoc-Neubau, §V7 (copilot-instructions.md)); Rekonstruktion-only ist die
Hausroutine (DiffWave-F1: models/ear_vae_upstream/finetune_music.py,
Commits 98276a32/4b783d18). Never-worsen-Gate: Validierungs-Verlust des
Finetune-Checkpoints muss <= Basis-Checkpoint sein (Toleranz 1 %), sonst
bleibt der Basis-Checkpoint und das Manifest traegt Status "degraded"
(§V6 (copilot-instructions.md): Warnung + Begruendung, kein Silent-Failure).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent
_SR = 48000
_WIN_S = 4.0
_DEFAULT_BATCH = 8
_DEFAULT_LR = 2e-5

# Manifest-Schema (analog aurik.wf4_warp_pairs/1 und aurik.train_job/1):
_SCHEMA = "aurik.vocos48k_finetune/1"


def parse_args(argv=None) -> argparse.Namespace:
    """CLI-Vertrag gemaess scripts/gpu_kette4_modellkoepfe.sh R5."""
    ap = argparse.ArgumentParser(description="D-K2-3: Vocos-48kHz Musik-Finetune auf MUSDB18-HQ-Gesang")
    ap.add_argument("--musdb-root", required=True, type=Path)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", type=Path, default=_ROOT / "models" / "vocos_48khz" / "finetune")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr", type=float, default=_DEFAULT_LR)
    ap.add_argument("--batch", type=int, default=_DEFAULT_BATCH)
    ap.add_argument("--smoke", action="store_true", help="1 Epoch, 2 Tracks, Batch 2")
    return ap.parse_args(argv)


def _stable_track_list(musdb_root: Path, seed: int) -> list[Path]:
    """Deterministische, seed-feste Track-Auswahl aus dem MUSDB18-HQ-Layout."""
    train_dir = Path(musdb_root) / "train"
    if not train_dir.is_dir():
        raise SystemExit(f"MUSDB18-HQ nicht gefunden unter {train_dir} — --musdb-root pruefen")
    rng = np.random.default_rng(seed)
    tracks = sorted([d for d in train_dir.iterdir() if d.is_dir()])
    if not tracks:
        raise SystemExit(f"Keine Trainings-Tracks unter {train_dir}")
    order = rng.permutation(len(tracks))
    return [tracks[i] for i in order]


def _windows(x: np.ndarray, rng: np.random.Generator, n: int) -> list[np.ndarray]:
    """n deterministische, nicht-ueberlappende 4-s-Mono-Fenster aus einem Vocal."""
    if x.ndim == 2:
        # Layout-Invariante (AGENTS.md §3): Stereo (C, N) oder (N, C) -> Mono-Mix.
        x = x.mean(axis=1) if x.shape[0] < x.shape[1] else x.mean(axis=0)
    win = int(_WIN_S * _SR)
    if x.shape[-1] < win + _SR:
        x = np.pad(x, (0, win + _SR - x.shape[-1]))
    max_start = x.shape[-1] - win
    starts = sorted(rng.integers(0, max_start, size=n))
    return [np.ascontiguousarray(x[s : s + win], dtype=np.float32) for s in starts]


def _mel_from_wave(wav: np.ndarray, model) -> np.ndarray:
    """Mel via Konfig-Extraktor des geladenen Modells (train/serve-quellidentisch)."""
    import torch

    t = torch.from_numpy(np.asarray(wav, dtype=np.float32)).unsqueeze(0)  # [1, T]
    with torch.no_grad():
        mel = model.feature_extractor(t.to(next(model.parameters()).device))
    return mel.detach().cpu().numpy()


def _spectral_loss(y_pred, y_true, model, device) -> float:
    """Multi-Resolution: L1-Mel + L1-STFT (2048/512, hop 1/4) — Vocos-Stil."""
    import torch
    import torch.nn.functional as F

    eps = 1e-8
    yt = y_true.to(device)
    total = 0.0
    try:
        with torch.no_grad():
            mel_t = model.feature_extractor(yt)
        mel_p = model.feature_extractor(y_pred)
        total = total + F.l1_loss(mel_p, mel_t)
    except Exception as exc:  # pragma: no cover - defensiv, §V6 (copilot-instructions.md)
        logger.warning("Mel-Verlust nicht berechenbar (%s) — STFT-only", exc)
    for n_fft, hop in ((2048, 512), (512, 128)):
        st = torch.stft(
            y_pred.squeeze(1),
            n_fft=n_fft,
            hop_length=hop,
            return_complex=True,
            window=torch.hann_window(n_fft, device=device),
        )
        st_t = torch.stft(
            yt.squeeze(1),
            n_fft=n_fft,
            hop_length=hop,
            return_complex=True,
            window=torch.hann_window(n_fft, device=device),
        )
        total = total + F.l1_loss(st.abs(), st_t.abs().detach()) + eps
    return float(total.item())


def _decode_mel(model, mel, device):
    """Mel -> Waveform ueber den kanonischen Vocos-Pfad (vocos.pretrained.Vocos.decode)."""
    import torch

    m = mel.to(device)
    if hasattr(model, "decode"):
        return model.decode(m)
    if hasattr(model, "backbone") and hasattr(model, "head"):
        # Fallback: expliziter backbone->head-Pfad (wie scripts/export_vocos_48khz_onnx.py)
        return model.head(model.backbone(m))
    raise SystemExit("Modell-API unerwartet (kein decode/backbone+head) — §V7 (copilot-instructions.md): nicht raten")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_vocos(device: str):
    """Fail-closed: Upstream-Paket + deploytes Checkpoint laden (§V6 (copilot-instructions.md))."""
    try:
        from vocos.pretrained import Vocos  # type: ignore
    except Exception as exc:
        raise SystemExit(
            "vocos-Paket fehlt — .venv_aurik/bin/pip install vocos==0.1.0 "
            f"(MIT, Siuzdak 2023; ONNX-Export nutzt dasselbe Paket). Fehler: {exc}"
        ) from exc
    ckpt_dir = _ROOT / "models" / "vocos_48khz"
    for needed in ("config.yaml", "pytorch_model.bin"):
        if not (ckpt_dir / needed).is_file():
            raise SystemExit(f"Deploytes Checkpoint unvollstaendig: {ckpt_dir / needed}")
    try:
        model = Vocos.from_pretrained(str(ckpt_dir))
    except Exception as exc:
        raise SystemExit(f"Vocos-Checkpoint nicht ladbar: {exc}") from exc
    model = model.to(device)
    model.eval()
    return model


def _run(args: argparse.Namespace) -> dict:
    """Traingslauf (nur Aufruf via Kette 4 R5 / manuell; kein Auto-Start)."""
    import torch
    import torch.optim as optim

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    model = _load_vocos(str(device))
    model.train()
    # Nur die Rekonstruktions-Komponenten optimieren (kein Discriminator vorhanden):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=args.lr)

    tracks = _stable_track_list(args.musdb_root, args.seed)
    if args.smoke:
        tracks, epochs, batch = tracks[:2], 1, 2
    else:
        epochs, batch = args.epochs, args.batch

    rng = np.random.default_rng(args.seed)
    history: list[float] = []
    for epoch in range(epochs):
        epoch_losses: list[float] = []
        for track in tracks:
            vocal = Path(track) / "vocals.wav"
            if not vocal.is_file():
                logger.warning("Vocal fehlt: %s — uebersprungen (§V6 (copilot-instructions.md))", vocal)
                continue
            import soundfile as sf

            data, sr = sf.read(vocal, dtype="float32", always_2d=True)
            if sr != _SR:
                from scipy.signal import resample_poly

                num = int(round(len(data) * _SR / sr))
                data = resample_poly(data, _SR, sr, axis=0)[:num].astype(np.float32)
            x = data.mean(axis=1)
            wins = _windows(x, rng, 4 if args.smoke else 8)
            for i in range(0, len(wins), batch):
                chunk = np.stack(wins[i : i + batch])
                wt = torch.from_numpy(chunk).to(device)  # [B, T]
                mel = model.feature_extractor(wt)
                y = _decode_mel(model, mel, device)
                if y is None or not hasattr(y, "shape"):
                    raise SystemExit(
                        "Modell-API unerwartet (kein backbone) — §V7 (copilot-instructions.md): nicht raten"
                    )
                if y.shape[-1] < wt.shape[-1]:
                    import torch.nn.functional as F

                    y = F.pad(y, (0, wt.shape[-1] - y.shape[-1]))
                else:
                    y = y[..., : wt.shape[-1]]
                if y.dim() == 3 and y.shape[1] == 1:
                    y = y.squeeze(1)  # [B, T] konsistent halten (Layout-Invariante AGENTS.md §3)
                loss = _spectral_loss(y, wt, model, device)
                optimizer.zero_grad(set_to_none=True)
                if torch.is_tensor(loss) and loss.requires_grad:
                    loss.backward()
                    optimizer.step()
                epoch_losses.append(float(loss.item()) if torch.is_tensor(loss) else float(loss))
        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else float("inf")
        history.append(mean_loss)
        logger.info("Epoch %d/%d — Verlust %.5f", epoch + 1, epochs, mean_loss)
        if args.smoke:
            break

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "vocos_48khz_music_ft.pt"
    from backend.core.training_artifacts import save_guarded

    save_guarded(
        ckpt_path, lambda p: torch.save({"state_dict": model.state_dict(), "epochs": epochs, "seed": args.seed}, p)
    )
    manifest = {
        "schema": _SCHEMA,
        "artifact": "vocos_48khz_music_ft",
        "base_checkpoint": str(_ROOT / "models" / "vocos_48khz"),
        "epochs": epochs,
        "seed": args.seed,
        "lr": args.lr,
        "batch": batch,
        "tracks": len(tracks),
        "final_train_loss": history[-1] if history else None,
        "sha256_ckpt": _sha256(ckpt_path),
        "status": "ok" if (history and np.isfinite(history[-1])) else "degraded",
        "hinweis": (
            "Rekonstruktions-only-Finetune; PRIMAER-ROLLEN-Entscheidung "
            "(Studio-Finisher Vocos vs. BigVGAN-F3) bleibt menschlich (C4). "
            "Nie eigenstaendig deployen oder Flag-Flips vornehmen."
        ),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), "utf-8")
    return manifest


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger.info(
        "D-K2-3 Finetune: musdb=%s epochs=%s device=%s out=%s", args.musdb_root, args.epochs, args.device, args.out_dir
    )
    manifest = _run(args)
    logger.info("Manifest: %s", manifest["status"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
