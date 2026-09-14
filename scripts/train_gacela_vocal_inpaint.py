#!/usr/bin/env python3
"""SOTA-VOCAL-INPAINT-F2-Vorbereitung: GaCELA-Vokal-Finetune (Pfad B).

Rezept nach dem EAR-VAE-Finetune (models/ear_vae_upstream/finetune_music.py)
und dem GaCELA-Upstream-Training (models/gacela_upstream/train.py, MIT):
  - Daten: MUSDB18HQ-Vocals (train) als 22,05-kHz-WAV-Ordner — das Upstream-
    TrainDataset berechnet die STFT on-the-fly (1024/256, Fenster 1024 Bins);
    der GAN inpaintet den Split [480, gap_bins, 480] (Default gap_bins=64 ≈
    0,74 s; 375–1500 ms = 33–131 Bins — variable Lücken als Verfeinerung).
  - Optimierung: TTUR (Gen 1e-4, Disc 1e-4, Adam β1 0.5/β2 0.9), Hinge-/
    Wasserstein-Loss — unverändert vom Upstream übernommen.
  - Seed 42, Early-Stop auf Val-SDR (F2-Session, nach F1), Resemblyzer-
    Witness-Gate (cos ≥ 0.92) als Identitäts-Absicherung der Fills.

Modi:
  --data-check   (Default): MUSDB-Vocals → temp-WAV-Ordner → TrainDataset-Batch
                 ziehen und Form/Endlichkeit prüfen (CPU, ohne Modell).
  --train        Upstream-Trainingsspiegel (GPU; Smoke in der F2-Session,
                 wenn die 7900 XTX nach F1 frei ist).
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
_MUSDB = _ROOT / "data" / "musdb18hq"
_UPSTREAM = _ROOT / "models" / "gacela_upstream"

_SR = 22050
_GAP_BINS = 64  # 0,74 s — 375–1500 ms = 33–131 Bins


def _write_vocals_wavs(tracks: list[Path], out_dir: Path) -> int:
    """MUSDB-Vocals → 22,05-kHz-Mono-WAVs in den Train-Ordner. Returns Anzahl."""
    import soundfile as sf
    from scipy.io import wavfile
    from scipy.signal import resample_poly

    written = 0
    for track in tracks:
        _wf = wavfile.read(str(track / "vocals.wav"))
        if not isinstance(_wf, tuple) or len(_wf) < 2:  # Bug 12: Index-basiert statt Unpacking
            raise ValueError("wavfile.read() ohne (sr, data)-Tupel")
        sr = int(_wf[0])
        wav = np.asarray(_wf[1])
        if wav.dtype == np.int16:
            wav = wav.astype(np.float32) / 32768.0
        else:
            wav = wav.astype(np.float32)
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        if sr != _SR:
            g = int(np.gcd(sr, _SR))
            wav = resample_poly(wav, _SR // g, sr // g).astype(np.float32)
        peak = float(np.max(np.abs(wav))) + 1e-12
        wav = (wav / peak * 0.9).astype(np.float32)
        sf.write(str(out_dir / f"{track.name}.wav"), wav, _SR)
        written += 1
    return written


def _data_check(args: argparse.Namespace) -> int:
    tracks = sorted(p for p in (_MUSDB / "test").glob("*") if p.is_dir())[: args.tracks]
    if not tracks:
        print("MUSDB18HQ fehlt")
        return 2
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp)
        n = _write_vocals_wavs(tracks, data_dir)
        print(f"WAVs geschrieben: {n}")
        sys.path.insert(0, str(_UPSTREAM))
        # F2-Entblockung 2026-09-14: eigener NumPy-Gabor-Shim ersetzt tifresi
        # (ltfatpy-Blocker) — installieren VOR dem Upstream-Import.
        from gacela_gabor_shim import install_tifresi_shim

        install_tifresi_shim()
        try:
            from data.audioLoader import AudioLoader
            from data.trainDataset import TrainDataset
        except ModuleNotFoundError as exc:
            print(f"F2-Voraussetzung fehlt: {exc}")
            print(
                "ltfatpy-Blocker (C-Build, kein Binary-Wheel): Pfade sind "
                "(1) ltfatpy mit Build-Toolchain installieren oder (2) den "
                "GaussTruncTF-STFT portieren (Roadmap SOTA-GACELA, KERN-Hinweis)."
            )
            return 3

        loader = AudioLoader(_SR, 1024, 256, 50)
        ds = TrainDataset(str(data_dir), window_size=1024, audio_loader=loader, examples_per_file=4, file_usages=2)
        batch = ds[0]
        arr = np.asarray(batch)
        print(f"Batch-Shape: {arr.shape} (erwartet [4, 512, 1024])")
        assert arr.shape[0] == 4 and arr.shape[1] == 512 and arr.shape[2] == 1024, "Unerwartete Batch-Shape"
        assert np.all(np.isfinite(arr)), "Nicht-finite Spektrogramm-Werte"
        assert float(np.max(arr)) > 0.0, "Spektrogramm leer"
        print("Data-Check OK: Upstream-TrainDataset liefert MUSDB-Vokal-Spektrogramme korrekt.")
    return 0


def _train(args: argparse.Namespace) -> int:
    if not Path(args.data_folder).is_dir():
        print("--data-folder fehlt: erst mit MUSDB-Vocals als WAV-Ordner befüllen.")
        return 2
    sys.path.insert(0, str(_UPSTREAM))
    from gacela_gabor_shim import install_tifresi_shim

    install_tifresi_shim()
    import torch
    from data.audioLoader import AudioLoader
    from data.trainDataset import TrainDataset
    from ganSystem import GANSystem

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    md = args.md
    signal_split = [480, _GAP_BINS, 480]

    params_generator = {
        "stride": [2, 2, 2, 2, 2],
        "nfilter": [8 * md, 4 * md, 2 * md, md, 1],
        "shape": [[4, 4], [4, 4], [8, 8], [8, 8], [8, 8]],
        "padding": [[1, 1], [1, 1], [3, 3], [3, 3], [3, 3]],
        "residual_blocks": 2,
        "full": 256 * md,
        "summary": True,
        "data_size": 2,
        "in_conv_shape": [16, 2],
        "borders": {
            "nfilter": [md, 2 * md, md, md / 2],
            "shape": [[5, 5], [5, 5], [5, 5], [5, 5]],
            "stride": [2, 2, 2, 2],
            "data_size": 2,
            "border_scale": 1,
            "width_full": None,
        },
    }
    params_stft_discriminator = {
        "stride": [2, 2, 2, 2, 2],
        "nfilter": [md, 2 * md, 4 * md, 8 * md, 16 * md],
        "shape": [[5, 5], [5, 5], [5, 5], [5, 5], [5, 5]],
        "data_size": 2,
    }
    params_mel_discriminator = {
        "stride": [2, 2, 2, 2, 2],
        "nfilter": [md // 4, md // 2, md, 2 * md, 4 * md],
        "shape": [[5, 5], [5, 5], [5, 5], [5, 5], [5, 5]],
        "data_size": 2,
    }
    params_optimization = {
        "batch_size": args.batch,
        "n_critic": 1,
        "generator": {"optimizer": "adam", "kwargs": [0.5, 0.9], "learning_rate": 1e-4},
        "discriminator": {"optimizer": "adam", "kwargs": [0.5, 0.9], "learning_rate": 1e-4},
    }
    params = {
        "net": {
            "generator": params_generator,
            "stft_discriminator": params_stft_discriminator,
            "mel_discriminator": params_mel_discriminator,
            "prior_distribution": "gaussian",
            "shape": [1, 512, 1024],
            "inpainting": {"split": signal_split},
            "gamma_gp": 10,
            "loss_type": "wasserstein",
        },
        "optimization": params_optimization,
    }
    gan_args = {
        "generator": params_generator,
        "stft_discriminator_count": 2,
        "mel_discriminator_count": 3,
        "stft_discriminator": params_stft_discriminator,
        "mel_discriminator": params_mel_discriminator,
        "borderEncoder": params_generator["borders"],
        "stft_discriminator_in_shape": [1, 512, 64],
        "mel_discriminator_in_shape": [1, 80, 64],
        "mel_discriminator_start_powscale": 2,
        "generator_input": 1440,
        "optimizer": params_optimization,
        "split": signal_split,
        "log_interval": 100,
        "spectrogram_shape": params["net"]["shape"],
        "gamma_gp": params["net"]["gamma_gp"],
        "save_path": args.save_path,
        "experiment_name": args.experiment_name,
        "save_interval": 10000,
        "fft_length": 1024,
        "fft_hop_size": 256,
        "sampling_rate": _SR,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    loader = AudioLoader(_SR, 1024, 256, 50)
    ds = TrainDataset(
        str(args.data_folder), window_size=1024, audio_loader=loader, examples_per_file=32, file_usages=30
    )
    train_loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch // 32, shuffle=True, num_workers=4, drop_last=True
    )
    gan = GANSystem(gan_args)
    for epoch in range(args.epochs):
        _step, can_restart = gan.train(train_loader, epoch, 0)
        if not can_restart:
            break
        # F2-Validierung (F2-Session): Val-SDR auf gehaltenen 375–1500-ms-Lücken
        # + Resemblyzer-Witness-Gate (cos ≥ 0.92) — nach dem Muster
        # scripts/train_diffwave_vocal_inpaint.py / benchmark_vocal_inpaint_baseline.py.
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-check", action="store_true", help="Upstream-Datenpfad mit MUSDB-Vocals prüfen (Default)")
    ap.add_argument("--train", action="store_true", help="Upstream-Trainingsspiegel (GPU, F2-Session)")
    ap.add_argument("--tracks", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--md", type=int, default=32)
    ap.add_argument("--data-folder", type=str, default="")
    ap.add_argument("--save-path", type=str, default="saved_results/")
    ap.add_argument("--experiment-name", type=str, default="gacela_vocal_ft")
    args = ap.parse_args()
    if args.train:
        return _train(args)
    return _data_check(args)


if __name__ == "__main__":
    sys.exit(main())
