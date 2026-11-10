#!/usr/bin/env python3
"""erstelle_hoerproben — repräsentative 10-s-Hörproben für B3/C4 (2026-11-09).

Erzeugt JETZT (CPU-ONNX, stört die laufende GPU-Kette nicht) die Proben, die
für die menschlichen Entscheidungen gebraucht werden — damit beim freien
GPU-Fenster KEINE Wartezeit fürs Anhören entsteht:

* ``c4_vocos`` — Studio-Finisher-Teilentscheidung (C4, D-K2-3):
  A = vocos-48k-Base (Sprach-Korpus), B = Musik-Finetune
  (`vocos_48khz_music_ft.onnx`, Export 2026-11-09). Beide ONNX, CPU.
* ``m5_rekonstruktion`` — Lücken-Synthese (M5): destruierte Passage
  (künstliche 2-s-Lücke) — A = lineare Interpolation, B = Finetune-Resynthese.

Quelle: `--verzeichnis` (Default ~/Musik — IHRE repräsentative Musik);
deterministische Auswahl (Seed 42, §G5 (copilot-instructions.md)),
10-s-Mittel-Ausschnitt je Stück. Kein Flag-Flip, kein Deploy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASE_ONNX = ROOT / "models" / "vocos_48khz" / "vocos_48khz.onnx"
FT_ONNX = ROOT / "models" / "vocos_48khz" / "finetune" / "vocos_48khz_music_ft.onnx"
SR = 48000
N_FFT = 2048
HOP = 256
N_MELS = 128


def _audio_dateien(verzeichnis: Path) -> list[Path]:
    return sorted(
        [p for p in Path(verzeichnis).rglob("*") if p.suffix.lower() in (".wav", ".flac") and p.is_file()],
        key=lambda p: str(p),
    )


def _lade_48k(pfad: Path) -> np.ndarray:
    import soundfile as sf

    data, sr = sf.read(str(pfad), always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1)
    x = np.asarray(data, dtype=np.float64)
    if sr != SR:
        import math

        from scipy.signal import resample_poly

        gcd = math.gcd(sr, SR)
        x = resample_poly(x, SR // gcd, sr // gcd)
    return x.astype(np.float32)


def _ausschnitt(audio: np.ndarray, sekunden: float) -> np.ndarray:
    want = int(round(sekunden * SR))
    n = audio.shape[-1]
    if n <= want:
        return audio
    start = (n - want) // 2
    return audio[..., start : start + want]


def _mel_torch(audio: np.ndarray) -> np.ndarray:
    """Mel [1, 128, T] — train/serve-konsistent zur vocos-48k-Config."""
    import torch
    import torchaudio

    w = torch.from_numpy(np.asarray(audio, dtype=np.float32)).unsqueeze(0)
    mel = torchaudio.transforms.MelSpectrogram(sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, power=1.0)(w)
    return mel.numpy().astype(np.float32)


def vocode(onnx: Path, mel: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
    out = sess.run(None, {sess.get_inputs()[0].name: mel})[0]
    audio = np.asarray(out).flatten()
    return audio[: int(round(10.0 * SR))].astype(np.float32) if len(audio) > SR else audio.astype(np.float32)


def erzeuge_c4_proben(
    verzeichnis: Path, ausgang: Path, anzahl: int = 3, sekunden: float = 10.0, seed: int = 42
) -> list[dict]:
    import soundfile as sf

    if not BASE_ONNX.is_file() or not FT_ONNX.is_file():
        raise FileNotFoundError("vocos-ONNX fehlt — erst scripts/export_vocos_48khz_onnx.py ausführen")
    ziel = ausgang / "c4_vocos"
    ziel.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    dateien = _audio_dateien(verzeichnis)
    if not dateien:
        raise FileNotFoundError(f"Keine Audiodateien in {verzeichnis}")
    gewaehlt = [dateien[i] for i in rng.choice(len(dateien), size=min(anzahl, len(dateien)), replace=False)]
    proben: list[dict] = []
    for stueck in gewaehlt:
        stamm = stueck.stem
        audio = _ausschnitt(_lade_48k(stueck), sekunden)
        mel = _mel_torch(audio)
        a = vocode(BASE_ONNX, mel)  # Base (Sprach-Korpus)
        b = vocode(FT_ONNX, mel)  # Musik-Finetune
        pa = ziel / f"{stamm[:60]}__A_base.wav"
        pb = ziel / f"{stamm[:60]}__B_finetune.wav"
        sf.write(str(pa), a, SR)
        sf.write(str(pb), b, SR)
        proben.append({"stamm": stamm, "a": str(pa), "b": str(pb)})
    return proben


def erzeuge_m5_proben(
    verzeichnis: Path, ausgang: Path, anzahl: int = 3, sekunden: float = 10.0, seed: int = 42
) -> list[dict]:
    import soundfile as sf

    if not FT_ONNX.is_file():
        raise FileNotFoundError("Finetune-ONNX fehlt")
    ziel = ausgang / "m5_rekonstruktion"
    ziel.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    dateien = _audio_dateien(verzeichnis)
    if not dateien:
        raise FileNotFoundError(f"Keine Audiodateien in {verzeichnis}")
    gewaehlt = [dateien[i] for i in rng.choice(len(dateien), size=min(anzahl, len(dateien)), replace=False)]
    proben: list[dict] = []
    for stueck in gewaehlt:
        stamm = stueck.stem
        audio = _ausschnitt(_lade_48k(stueck), sekunden)
        n = len(audio)
        lo, hi = n // 2 - SR, n // 2 + SR  # 2-s-Lücke in der Mitte
        destruiert = audio.copy()
        destruiert[lo:hi] = 0.0
        # A: lineare Interpolation der Lücke
        linear = destruiert.copy()
        linear[lo:hi] = np.interp(np.arange(lo, hi), [lo - 1, hi], [destruiert[lo - 1], destruiert[min(hi, n - 1)]])
        # B: vocos-Finetune-Resynthese des destruierten Fensters
        synthese = vocode(FT_ONNX, _mel_torch(destruiert))
        synthese = synthese[:n]
        pa = ziel / f"{stamm[:60]}__A_linear.wav"
        pb = ziel / f"{stamm[:60]}__B_vocos_ft.wav"
        sf.write(str(pa), linear, SR)
        sf.write(str(pb), synthese, SR)
        proben.append({"stamm": stamm, "a": str(pa), "b": str(pb)})
    return proben


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="10-s-Hörproben für B3/C4 erzeugen (CPU, ohne GPU-Wartezeit)")
    ap.add_argument("--verzeichnis", default=str(Path.home() / "Musik"), help="Musik-Ordner (Default ~/Musik)")
    ap.add_argument("--ausgang", default=str(ROOT / "study_stimuli" / "hoerproben"))
    ap.add_argument("--anzahl", type=int, default=3)
    ap.add_argument("--sekunden", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--nur", default="c4_vocos,m5_rekonstruktion", help="Komma-getrennt: c4_vocos, m5_rekonstruktion")
    args = ap.parse_args(argv)

    ausgang = Path(args.ausgang)
    # Bestehendes Manifest mergen — ein Set-Neulauf darf die anderen nicht löschen.
    manifest_path = ausgang / "manifest.json"
    manifest: dict = {"schema": "aurik.hoerproben/1", "sekunden": args.sekunden, "seed": args.seed, "sets": {}}
    if manifest_path.is_file():
        alt = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["sets"] = dict(alt.get("sets", {}))
    for set_name in [s.strip() for s in args.nur.split(",") if s.strip()]:
        if set_name == "c4_vocos":
            proben = erzeuge_c4_proben(Path(args.verzeichnis), ausgang, args.anzahl, args.sekunden, args.seed)
            manifest["sets"]["c4_vocos"] = {
                "entscheidung": "C4 Studio-Finisher-Teilentscheidung: vocos-Base (Sprach-Korpus) vs. Musik-Finetune (D-K2-3)",
                "proben": proben,
            }
        elif set_name == "m5_rekonstruktion":
            proben = erzeuge_m5_proben(Path(args.verzeichnis), ausgang, args.anzahl, args.sekunden, args.seed)
            manifest["sets"]["m5_rekonstruktion"] = {
                "entscheidung": "M5 Lücken-Synthese: lineare Interpolation vs. vocos-Finetune-Resynthese",
                "proben": proben,
            }
        else:
            print(f"Unbekanntes Set: {set_name}")
            return 2
        print(f"{set_name}: {len(manifest['sets'][set_name]['proben'])} Proben-Paare erzeugt")

    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Manifest: {manifest_path}")
    print("Anhören: python3 -B scripts/hoer_entscheidung.py --hoeren c4_vocos")
    return 0


if __name__ == "__main__":
    sys.exit(main())
