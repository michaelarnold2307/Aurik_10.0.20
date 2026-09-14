#!/usr/bin/env python3
"""F2-Validierung: GaCELA-Vokal-Checkpoint auf 743-ms-Lücken (SDR + Witness).

Hintergrund: Der Upstream-Generator erzeugt FEST 64 Zeit-Bins (= 743 ms bei
hop 256/22,05 kHz) — der 375–1500-ms-Bereich der Roadmap wird wie in phase_55
durch Zuschneiden/Positionieren bedient; die Validierung misst hier die native
Länge direkt gegen die Ground-Truth-Lücke.

Ablauf je Lücke (Seed 42, deterministisch):
  MUSDB-Test-Vocal → 22,05-kHz-WAV → Spektrogramm (Shim-DGT + Log-Konvention)
  → Position → Borders [480, 64, 480] → generateGap → Inversion (Shim-GL)
  → SDR vs. Ground-Truth-Gap; zusätzlich Resemblyzer-Identitäts-Witness
  (cos vorher/nachher, best-effort).

Report: docs/reports/current/2026-09-14_gacela_vocal_val.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
_UPSTREAM = _ROOT / "models" / "gacela_upstream"
_SR = 22050
_GAP_BINS = 64
_BORDER_BINS = 480
_REPORT_DIR = _ROOT / "docs" / "reports" / "current"


def _install_shim() -> None:
    spec = importlib.util.spec_from_file_location("gacela_gabor_shim", str(_ROOT / "scripts" / "gacela_gabor_shim.py"))
    assert spec and spec.loader
    shim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shim)
    shim.install_tifresi_shim()
    globals()["_shim"] = shim


def _build_gan(ckpt_path: Path) -> object:
    sys.path.insert(0, str(_UPSTREAM))
    from ganSystem import GANSystem

    md = 32
    signal_split = [_BORDER_BINS, _GAP_BINS, _BORDER_BINS]
    params_generator = {
        "stride": [2, 2, 2, 2, 2],
        "nfilter": [8 * md, 4 * md, 2 * md, md, 1],
        "shape": [[4, 4], [4, 4], [8, 8], [8, 8], [8, 8]],
        "padding": [[1, 1], [1, 1], [3, 3], [3, 3], [3, 3]],
        "residual_blocks": 2,
        "full": 256 * md,
        "summary": False,
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
        "batch_size": 64,
        "stride": [2, 2, 2, 2, 2],
        "nfilter": [md, 2 * md, 4 * md, 8 * md, 16 * md],
        "shape": [[5, 5], [5, 5], [5, 5], [5, 5], [5, 5]],
        "data_size": 2,
    }
    params_mel_discriminator = {
        "batch_size": 64,
        "stride": [2, 2, 2, 2, 2],
        "nfilter": [md // 4, md // 2, md, 2 * md, 4 * md],
        "shape": [[5, 5], [5, 5], [5, 5], [5, 5], [5, 5]],
        "data_size": 2,
    }
    params_optimization = {
        "batch_size": 64,
        "n_critic": 1,
        "generator": {"optimizer": "adam", "kwargs": [0.5, 0.9], "learning_rate": 1e-4},
        "discriminator": {"optimizer": "adam", "kwargs": [0.5, 0.9], "learning_rate": 1e-4},
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
        "tensorboard_interval": 10**9,
        "spectrogram_shape": [1, 512, 1024],
        "gamma_gp": 10,
        "save_path": str(_ROOT / "output" / "gacela_f2") + "/",
        "experiment_name": "gacela_vocal_ft",
        "save_interval": 1000,
        "fft_length": 1024,
        "fft_hop_size": 256,
        "sampling_rate": _SR,
    }
    gan = GANSystem(gan_args)
    import torch

    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    gan.generator.load_state_dict(state["generator"])
    for encoder, enc_state in zip(gan.border_encoders, state["encoders"]):
        encoder.load_state_dict(enc_state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gan.generator.to(device)
    for enc in gan.border_encoders:
        enc.to(device)
    gan.generator.eval()
    for enc in gan.border_encoders:
        enc.eval()
    return gan


def _sdr_db(reference: np.ndarray, estimate: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=np.float64).ravel()
    est = np.asarray(estimate, dtype=np.float64).ravel()
    n = min(len(ref), len(est))
    ref, est = ref[:n], est[:n]
    err = ref - est
    denom = float(np.dot(ref, ref)) + 1e-12
    num = float(np.dot(err, err)) + 1e-12
    return float(10.0 * np.log10(denom / num))


def _witness_cos(orig_seg: np.ndarray, gen_seg: np.ndarray) -> float | None:
    """Sänger-Identitäts-Witness über das Resemblyzer-Plugin (best-effort)."""
    try:
        from plugins.resemblyzer_plugin import get_resemblyzer_plugin

        plugin = get_resemblyzer_plugin()
        if not plugin.available():
            return None
        emb_a = plugin.embed(orig_seg, _SR)
        emb_b = plugin.embed(gen_seg, _SR)
        if emb_a is None or emb_b is None:
            return None
        return float(plugin.cosine_similarity(emb_a, emb_b))
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=str, default="output/gacela_f2/gacela_vocal_ft_checkpoints/09_0499.pt")
    ap.add_argument("--tracks", type=int, default=3)
    ap.add_argument("--gaps-per-track", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    _install_shim()
    ckpt = _ROOT / args.ckpt
    if not ckpt.is_file():
        print(f"Checkpoint fehlt: {ckpt}")
        return 2
    gan = _build_gan(ckpt)
    shim = globals()["_shim"]
    torch_mod = sys.modules.get("torch") or __import__("torch")
    import soundfile as sf

    # Test-Vocals → WAV (22,05 kHz) über die F2-Datenfunktion.
    f2_spec = importlib.util.spec_from_file_location("f2", str(_ROOT / "scripts" / "train_gacela_vocal_inpaint.py"))
    assert f2_spec and f2_spec.loader
    f2 = importlib.util.module_from_spec(f2_spec)
    f2_spec.loader.exec_module(f2)

    tmp = _ROOT / "output" / "gacela_f2" / "val_wavs"
    tmp.mkdir(parents=True, exist_ok=True)
    tracks = sorted(p for p in (_ROOT / "data" / "musdb18hq" / "test").glob("*") if p.is_dir())
    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.choice(len(tracks), size=min(args.tracks, len(tracks)), replace=False))
    tracks = [tracks[int(i)] for i in idx]
    f2._write_vocals_wavs(tracks, tmp)

    sys.path.insert(0, str(_UPSTREAM))
    from data.audioLoader import AudioLoader

    loader = AudioLoader(_SR, 1024, 256, 50)
    device = torch_mod.device("cuda" if torch_mod.cuda.is_available() else "cpu")
    results: list[dict[str, float | str]] = []
    for track in tracks:
        spec_log_full = loader.loadAsSpectrogram(str(tmp / f"{track.name}.wav"))
        # Trainings-Konvention: letzte Frequenz-Bin wird verworfen (TrainDataset
        # liefert 512 Bins; mel_spectrogram erwartet [1, 512, T]).
        spec_log = spec_log_full[:-1, :]
        n_frames = spec_log.shape[1]
        span = _BORDER_BINS * 2 + _GAP_BINS
        if n_frames <= span + 4:
            continue
        for g in range(args.gaps_per_track):
            center = int(rng.integers(_BORDER_BINS, n_frames - _BORDER_BINS - _GAP_BINS))
            left = spec_log[:, center - _BORDER_BINS : center]
            right = spec_log[:, center + _GAP_BINS : center + _GAP_BINS + _BORDER_BINS]
            gt = spec_log[:, center : center + _GAP_BINS]
            with torch_mod.no_grad():
                ctx = [
                    torch_mod.from_numpy(left[None, None].astype(np.float64)).float().to(device),
                    torch_mod.from_numpy(right[None, None].astype(np.float64)).float().to(device),
                ]
                gen = gan.generateGap(ctx).detach().cpu().numpy().squeeze()
            gen_mag = shim.inv_log_spectrogram(25.0 * (gen - 1.0))
            gt_mag = shim.inv_log_spectrogram(25.0 * (gt - 1.0))
            gen_audio = shim.GaussTruncTFShim(256, 1024).invert_spectrogram(gen_mag, iterations=16)
            gt_audio = shim.GaussTruncTFShim(256, 1024).invert_spectrogram(gt_mag, iterations=16)
            sdr = _sdr_db(gt_audio, gen_audio)
            cos = _witness_cos(gt_audio, gen_audio)
            results.append(
                {
                    "track": track.name,
                    "gap": g,
                    "sdr_db": round(sdr, 2),
                    "witness_cos": round(cos, 3) if cos is not None else None,
                }
            )
            print(f"{track.name} gap{g}: SDR={sdr:+.2f} dB witness_cos={cos}")

    if not results:
        print("Keine Lücken validiert.")
        return 3
    mean_sdr = float(np.mean([r["sdr_db"] for r in results]))
    report = {
        "date": f"{date.today().isoformat()}",
        "script": "scripts/validate_gacela_vocal_inpaint.py",
        "ckpt": args.ckpt,
        "n_gaps": len(results),
        "mean_sdr_db": round(mean_sdr, 2),
        "min_sdr_db": round(float(np.min([r["sdr_db"] for r in results])), 2),
        "gate_mean_sdr_ge_0": mean_sdr >= 0.0,
        "results": results,
    }
    out = _REPORT_DIR / f"{date.today().isoformat()}_gacela_vocal_val.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Report: {out}")
    print(f"mean SDR = {mean_sdr:+.2f} dB (Gate ≥ 0: {report['gate_mean_sdr_ge_0']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
