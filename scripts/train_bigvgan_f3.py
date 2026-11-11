"""scripts/train_bigvgan_f3.py — F3: BigVGAN-v2 Musik-Finetune (HR-V1/03/23/50/07).

§SOTA-F3 (Roadmap TODOS_SOTA_ROADMAP.md): Der Basischeckpoint
`models/bigvgan/bigvgan_v2.pth` ist BigVGAN v2 in der **44,1-kHz-Konfiguration**
(`bigvgan_v2_44khz_128band_512x`: 128 Mel-Bänder, 512×, 1536 Kanäle, SnakeBeta).
**KORREKTUR 2026-10-06 (§III.13 (copilot-instructions.md)):** Die frühere Angabe
„BigVGAN-v2 ist sprach-trainiert (LibriTTS/LJSpeech)" war eine unbelegte
Doku-Übernahme und ist architektur-**widerlegt** — die LibriTTS-Konfiguration
derselben Familie ist 24 kHz/100 Bänder/256×/512 Kanäle. Der Trainingskorpus der
Basis ist lokal **nicht belegbar** (Checkpoint enthält nur `{"generator": …}`
ohne Hparams, keine Modell-Karte, kein Upstream-SHA) und gilt daher nach §III.13
als `unbekannt` — nicht als „Sprache". Der Musik-Finetune adaptiert den Generator
an MUSDB18-HQ als Rekonstruktions-Vocoder — Ziel: der additive HR-V1-Synthesepfad
(phase_07 + 23/50/03) rekonstruiert Musik-Harmonik korrekt statt Basis-Artefakte.
A/B-Teilvalidierung liegt vor (af +0,0073, HNR +4,42 dB); dieses Skript liefert
den MUSDB-Finetune. **Gate (korrigiert 2026-11-11, D-K3-96):** gemessen wird
NACH dem Lauf mit ``scripts/verify_bigvgan_d3_gate.py`` — af/HNR/musik_mos/
defektlast gegen die **Basis** (Never-worsen + Vorteil auf ≥ 2/3 Tracks). Ein
zeitlicher ΔSDR ist für einen Mel-Vocoder unzulässig: der Generator erzeugt eine
neue Phase, der SDR liegt für Basis und Finetune alike bei ≈ 0 dB (gemessen).

Rezept (EAR-VAE-Muster, deterministisch §G5 (GEBOTE.md)):
  - Ziel: Mel → Wellenform-Rekonstruktion (self-supervised, kein Paar-Datensatz)
  - Loss: A1 (Waveform-L1) + Multi-Resolution-STFT (Spectral Convergence +
    Log-Magnitude-L1 bei n_fft ∈ {512, 1024, 2048})
  - Optimizer: AdamW lr=1e-4, Gradient-Clipping 1.0, fp16 (GradScaler)
  - Validierung: 2 feste Test-Tracks (deterministische Fenster), Early-Stop
    auf bestem Val-A1 (Patience 5)
  - Daten: MUSDB18-HQ train/ (44,1 kHz), 2-s-Zufalls-Crops, num_workers=0
    (Fork-Deadlock-Befund F2 2026-09-14), Seed 42

Aufruf:
  python scripts/train_bigvgan_f3.py --smoke           # 3 Schritte, 1 Track
  python scripts/train_bigvgan_f3.py --epochs 30       # voller Lauf (GPU)
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.core.training_artifacts import (
    historie_zusammenfuehren,
    report_historie_laden,
    save_guarded,
    write_text_guarded,
    zeitbudget_erreicht,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_bigvgan_f3")

# Checkpoint-Konfiguration (bigvgan_v2_44khz_128band_512x, aus config.json)
# `dict[str, Any]`: Die Werte sind heterogen (int, str, list, None) — ohne
# Parameterisierung sind die Zugriffe `object` und jede Arithmetik darauf ist
# ein Typfehler (mypy `operator` bei `CROP_S * SR`).
CFG: dict[str, Any] = {
    "resblock": "1",
    "upsample_rates": [8, 4, 2, 2, 2, 2],
    "upsample_initial_channel": 512,
    "upsample_kernel_sizes": [16, 8, 4, 4, 4, 4],
    "resblock_kernel_sizes": [3, 7, 11],
    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    "activation": "snakebeta",
    "snake_logscale": True,
    "use_tanh_at_final": False,
    "use_bias_at_final": False,
    "num_mels": 128,
    "sampling_rate": 44100,
    "n_fft": 2048,
    "hop_size": 512,
    "win_size": 2048,
    "fmin": 0,
    "fmax": None,
}
SR = int(CFG["sampling_rate"])
CROP_S = 0.5  # 1536-Kanal-Modell: 2-s-Crops × Batch 4 sprengten 24 GB VRAM (OOM-Befund);
# 0,5 s halbiert die Schritt-Zeit (~3 s → ~1,5 s bei Batch 4) und bleibt lokal genug
CROP_N = int(CROP_S * SR)
STFT_SIZES = [512, 1024, 2048]


def build_model(device: torch.device, checkpoint: Path) -> torch.nn.Module:
    from bigvgan import AttrDict, BigVGAN

    raw = torch.load(str(checkpoint), map_location="cpu", weights_only=True)  # nosec B614
    if isinstance(raw, dict) and "generator" in raw:
        gen_sd = raw["generator"]
    elif hasattr(raw, "state_dict"):
        gen_sd = raw.state_dict()
    else:
        gen_sd = raw
    # Kanal-Zahl aus dem Checkpoint ableiten (wie plugins/bigvgan_v2_plugin.py):
    # ups.0.0.weight_v → [out_ch, in_ch, ksize]; der Checkpoint hat 24, nicht 512.
    _up0 = gen_sd.get("ups.0.0.weight_v") if isinstance(gen_sd, dict) else None
    if _up0 is not None:
        CFG["upsample_initial_channel"] = int(_up0.shape[0])
    h = AttrDict(dict(CFG))
    model: torch.nn.Module = BigVGAN(h, use_cuda_kernel=False)
    model.load_state_dict(gen_sd)
    model = model.to(device)
    return model


def extract_mel(wave: torch.Tensor) -> torch.Tensor:
    from bigvgan.meldataset import mel_spectrogram

    # Wave [B, T] auf CPU/GPU; mel_spectrogram erwartet [1, T]
    mels = []
    for i in range(wave.shape[0]):
        m = mel_spectrogram(
            wave[i : i + 1],
            n_fft=CFG["n_fft"],
            num_mels=CFG["num_mels"],
            sampling_rate=SR,
            hop_size=CFG["hop_size"],
            win_size=CFG["win_size"],
            fmin=CFG["fmin"],
            fmax=CFG["fmax"],
            center=False,
        )
        mels.append(m)
    return torch.cat(mels, dim=0)  # [B, n_mels, T]


def stft_mag(x: torch.Tensor, n_fft: int, hop: int, win: int) -> torch.Tensor:
    x = torch.nn.functional.pad(x, (n_fft // 2, n_fft // 2), mode="reflect")
    X = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop,
        win_length=win,
        window=torch.hann_window(win, device=x.device),
        return_complex=True,
    )
    return X.abs()


def mr_stft_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Multi-Resolution-STFT-Loss (Spectral Convergence + Log-Mag-L1)."""
    sc = torch.zeros((), device=x.device)
    mag = torch.zeros((), device=x.device)
    for n_fft in STFT_SIZES:
        hop = n_fft // 4
        s_x = stft_mag(x, n_fft, hop, n_fft)
        s_y = stft_mag(y, n_fft, hop, n_fft)
        sc = sc + (torch.norm(s_y - s_x, p="fro") / (torch.norm(s_x, p="fro") + 1e-8))
        mag = mag + torch.mean(torch.abs(torch.log(s_y + 1e-4) - torch.log(s_x + 1e-4)))
    return sc / len(STFT_SIZES) + mag / len(STFT_SIZES)


def load_track(path: Path) -> np.ndarray:
    import scipy.io.wavfile as wav

    sr, data = wav.read(str(path))
    if data.ndim == 2:
        data = data.mean(axis=1)
    data = data.astype(np.float32) / max(float(np.abs(data).max()), 1e-9)
    if sr != SR:
        import librosa

        data = librosa.resample(data, orig_sr=sr, target_sr=SR)
    # `np.asarray` ohne dtype ist ein No-Op für ein ndarray — es fixiert nur den
    # Typ (data kommt aus scipy/librosa und ist für mypy `Any`).
    out: np.ndarray = np.asarray(data)
    return out


def random_crops(track: np.ndarray, batch: int, rng: np.random.Generator) -> torch.Tensor:
    n = len(track)
    starts = rng.integers(0, max(1, n - CROP_N), size=batch)
    out = np.stack([track[s : s + CROP_N] for s in starts])
    return torch.from_numpy(out.astype(np.float32))


def val_loss(model: torch.nn.Module, device: torch.device, tracks: list[np.ndarray], rng: np.random.Generator) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for tr in tracks:
            for _ in range(4):
                x = random_crops(tr, 2, rng).to(device)
                mel = extract_mel(x)
                y = model(mel)
                y = y.squeeze(1)  # [B, 1, T] → [B, T]
                _n_min = min(x.shape[-1], y.shape[-1])
                x_a = x[..., :_n_min]
                y_a = y[..., :_n_min]
                loss = torch.nn.functional.l1_loss(y_a, x_a) + 0.5 * mr_stft_loss(y_a.float(), x_a.float())
                total += float(loss.item())
                count += 1
    model.train()
    return total / max(1, count)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="3 Schritte auf 1 Track")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--no-amp", action="store_true", help="FP32 statt fp16 (ROCm-Kernel-Diagnose)")
    ap.add_argument("--steps-per-epoch", type=int, default=2000)
    ap.add_argument("--checkpoint", type=str, default="models/bigvgan/bigvgan_v2.pth")
    ap.add_argument("--data-dir", type=str, default="data/musdb18hq")
    ap.add_argument("--out-dir", type=str, default="output/f3_bigvgan")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", type=str, default="")
    ap.add_argument(
        "--max-seconds",
        type=float,
        default=0.0,
        help="Zeitbudget je Prozess; stoppt an der Epochen-Grenze (Exit 75), 0 = unbegrenzt",
    )
    args = ap.parse_args()

    t_prozessstart = time.perf_counter()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  # Kernel-Autotuning (ROCm/MIOpen-Fallback-Fix)
    logger.info("F3 BigVGAN-Finetune — Gerät: %s", device)

    data_root = Path(args.data_dir)
    train_root = data_root / "train"
    test_root = data_root / "test"
    train_files = sorted(train_root.glob("*/mixture.wav"))
    test_files = sorted(test_root.glob("*/mixture.wav"))
    if args.smoke:
        train_files = train_files[:1]
    logger.info("Trainings-Tracks: %d, Test-Tracks: %d", len(train_files), len(test_files))
    if not train_files:
        logger.error("Keine MUSDB-Tracks unter %s gefunden", data_root)
        return 2

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        logger.error("Checkpoint fehlt: %s", checkpoint)
        return 2
    model = build_model(device, checkpoint)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Modell geladen (%d Parameter) aus %s", n_params, checkpoint)

    # §F3-Perf (2026-09-17): untere, sprach-generische Stufen einfrieren
    # (EAR-VAE-Rezept: „Encoder-Frozen“) — die Musik-Adaption findet in den
    # oberen Stufen statt; spart ~40 % Backward-Zeit und verhindert Drift.
    _frozen = 0
    for _name, _p in model.named_parameters():
        if _name.startswith(("conv_pre.", "ups.0.", "resblocks.0.", "resblocks.1.", "resblocks.2.")):
            _p.requires_grad = False
            _frozen += 1
    if _frozen:
        logger.info("Untere Layer eingefroren (%d Parameter-Gruppen)", _frozen)

    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.8, 0.99), weight_decay=1e-2
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    rng = np.random.default_rng(args.seed)
    train_tracks = [load_track(p) for p in train_files]
    val_tracks = [load_track(p) for p in test_files[:2]]

    start_epoch = 0
    best_val = float("inf")
    patience_left = 5
    history: list[dict] = []

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.resume:
        if not Path(args.resume).is_file():
            logger.error(
                "Resume-Checkpoint fehlt: %s — Lauf NICHT gestartet. Ein Resume ohne "
                "Datei verbrennt GPU-Zeit (D-K3-72-Klasse). Ohne --resume neu starten.",
                args.resume,
            )
            return 2
        res = torch.load(args.resume, map_location="cpu", weights_only=True)  # nosec B614
        model.load_state_dict(res["generator"])
        opt.load_state_dict(res.get("optimizer", opt.state_dict()))
        start_epoch = int(res.get("epoch", 0)) + 1
        best_val = float(res.get("best_val", res.get("val_a1", best_val)))
        # §G5 (copilot-instructions.md): RNG-Stand fortsetzen, damit ein Zeitbudget-Stopp
        # die Crop-Folge nicht neu beginnt — der unterbrochene Lauf folgt derselben
        # Zufallsfolge wie ein durchgehender Lauf (Seed 42). Bit-Gleichheit garantiert
        # das nicht (Backend-Kernel können zwischen Prozessen abweichen).
        if "rng_state" in res:
            rng.bit_generator.state = res["rng_state"]
        if "torch_rng_state" in res:
            torch.set_rng_state(res["torch_rng_state"])
        history = report_historie_laden(out_dir, seed=args.seed)
        logger.info(
            "Resume ab Epoche %d (best_val=%.4f, Historie=%d Epochen, RNG=%s)",
            start_epoch,
            best_val,
            len(history),
            "fortgesetzt" if "rng_state" in res else "neu",
        )

    epochen_dieser_lauf = 0
    letzte_epoche_s = 0.0
    early_stopped = False

    def _report(*, completed: bool, stop_reason: str) -> None:
        """Schreibt den Beleg-Report atomar (Historie über Resume-Grenzen)."""
        zusammengefuehrt = historie_zusammenfuehren(history)
        report = {
            "device": str(device),
            "seed": args.seed,
            "history": zusammengefuehrt,
            "best_val_a1": round(best_val, 6),
            "completed": completed,
            "stop_reason": stop_reason,
            "target_epochs": args.epochs,
            "epochs_done": start_epoch + epochen_dieser_lauf,
            "gate_hint": (
                "D3-2-Gate via scripts/verify_bigvgan_d3_gate.py NACH dem Lauf messen "
                "(af/HNR/musik_mos/defektlast gegen die Basis; SDR ist fuer phasen-unabhaengige "
                "Mel-Vocoder unzulaessig)"
            ),
        }
        write_text_guarded(out_dir / "report.json", json.dumps(report, indent=2, ensure_ascii=False))

    if start_epoch > args.epochs:
        logger.error(
            "Resume-Anker liegt hinter dem Ziel: start_epoch=%d > epochs=%d — inkonsistent "
            "(fremder/zu hoher Anker), Lauf NICHT gestartet.",
            start_epoch,
            args.epochs,
        )
        return 3
    if start_epoch == args.epochs:
        # Kill zwischen letztem Checkpoint und Report ist kein No-Op-Fehler:
        # der Anker belegt, dass alle Ziel-Epochen trainiert sind (D-K3-87).
        _report(completed=True, stop_reason="ziel_epochen_bereits_erreicht")
        logger.info(
            "Ziel bereits erreicht: Anker steht auf Epoche %d/%d — kein GPU-Lauf nötig.",
            start_epoch,
            args.epochs,
        )
        return 0

    steps_per_epoch = 3 if args.smoke else args.steps_per_epoch
    model.train()
    for epoch in range(start_epoch, args.epochs):
        if zeitbudget_erreicht(
            verstrichen_s=time.perf_counter() - t_prozessstart,
            letzte_epoche_s=letzte_epoche_s,
            max_seconds=args.max_seconds,
        ):
            _report(completed=False, stop_reason="zeitbudget")
            logger.info(
                "Zeitbudget %.0f s erreicht — Stopp an der Epochen-Grenze (Epochen in diesem "
                "Versuch: %d). Resume: --resume %s",
                args.max_seconds,
                epochen_dieser_lauf,
                out_dir / "checkpoint_latest.pt",
            )
            return 75
        t0 = time.perf_counter()
        run_a1 = 0.0
        run_stft = 0.0
        for step in range(steps_per_epoch):
            tr = train_tracks[int(rng.integers(0, len(train_tracks)))]
            x = random_crops(tr, args.batch_size, rng).to(device)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and not args.no_amp)):
                mel = extract_mel(x)
                y = model(mel)
                y = y.squeeze(1)  # [B, 1, T] → [B, T]
                _n_min = min(x.shape[-1], y.shape[-1])
                x_a = x[..., :_n_min]
                y_a = y[..., :_n_min]
                a1 = torch.nn.functional.l1_loss(y_a.float(), x_a.float())
                st = mr_stft_loss(y_a.float(), x_a.float())
                loss = a1 + 0.5 * st
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            run_a1 += float(a1.item())
            run_stft += float(st.item())
            if device.type == "cuda" and (step + 1) % 50 == 0:
                torch.cuda.empty_cache()
            if step % 200 == 0 and step > 0:
                logger.info(
                    "Epoche %d Schritt %d/%d: A1=%.4f STFT=%.4f",
                    epoch,
                    step,
                    steps_per_epoch,
                    run_a1 / (step + 1),
                    run_stft / (step + 1),
                )
        run_a1 /= max(1, steps_per_epoch)
        run_stft /= max(1, steps_per_epoch)
        v_a1 = val_loss(model, device, val_tracks, rng)
        dt = time.perf_counter() - t0
        letzte_epoche_s = dt
        epochen_dieser_lauf += 1
        logger.info(
            "Epoche %d fertig (%.1f s): A1=%.4f STFT=%.4f Val=%.4f",
            epoch,
            dt,
            run_a1,
            run_stft,
            v_a1,
        )
        history.append(
            {
                "epoch": epoch,
                "train_a1": round(run_a1, 6),
                "train_stft": round(run_stft, 6),
                "val_a1": round(v_a1, 6),
                "seconds": round(dt, 1),
            }
        )
        ckpt = {
            "generator": model.state_dict(),
            "optimizer": opt.state_dict(),
            "epoch": epoch,
            "val_a1": v_a1,
            "best_val": best_val,
            "rng_state": rng.bit_generator.state,
            "torch_rng_state": torch.get_rng_state(),
            "cfg": CFG,
        }
        save_guarded(out_dir / f"checkpoint_epoch{epoch}.pt", ckpt)
        # Stabiler Resume-Anker (§G5 (copilot-instructions.md)): checkpoint_latest.pt — damit ein
        # Timeout-/Retry-Weiterlauf (D-K3-22 Budget vs. Epoch-Zeit) ab der
        # letzten Epoche fortsetzt statt bei Epoche 0 neu zu starten.
        save_guarded(out_dir / "checkpoint_latest.pt", ckpt)
        if v_a1 < best_val:
            best_val = v_a1
            patience_left = 5
            save_guarded(out_dir / "best.pt", ckpt)
            logger.info("Neues bestes Modell (Val=%.4f) gespeichert", v_a1)
        else:
            patience_left -= 1
            if patience_left <= 0 and not args.smoke:
                logger.info("Early-Stop nach Epoche %d (Val stagniert)", epoch)
                early_stopped = True
                break
        _report(
            completed=(epoch + 1) >= args.epochs,
            stop_reason="ziel_epochen" if (epoch + 1) >= args.epochs else "laeuft",
        )

    ziel_erreicht = (not early_stopped) and (start_epoch + epochen_dieser_lauf) >= args.epochs
    _report(completed=ziel_erreicht, stop_reason="early_stop" if early_stopped else "ziel_epochen")
    logger.info(
        "F3-Finetune beendet — %d/%d Epochen, bestes Val-A1: %.4f (Checkpoint: %s)",
        start_epoch + epochen_dieser_lauf,
        args.epochs,
        best_val,
        out_dir / "best.pt",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
