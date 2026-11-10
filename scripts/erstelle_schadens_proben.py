#!/usr/bin/env python3
"""erstelle_schadens_proben — „Gegner-Material“ für den Sinn-Beweis (2026-11-10).

Hintergrund (gemessen 2026-11-10): Auf sauberen HD-Quellen (320 kbps) ist die
Aurik-Restauration praktisch transparent — das Original gewinnt (gesundes
Never-worsen, kein Loudness-War, RMS/Crest ≈ identisch). Auriks Zweck ist
Schadens-Reparatur. Dieses Skript beschafft den fehlenden Schaden autonom:
es beschädigt EIGENE, saubere Titel deterministisch und lässt die
Produktions-Pipeline (Pre-Analysis → Denker via `backend/api/bridge`)
gegen den Schaden laufen — reine Hörproben, kein Flag-Flip, kein Deploy,
keine W-Stufen-Buchung (§G8 (copilot-instructions.md): Proben werden hier
transparent erzeugt und im Manifest belegt).

Rezepte:
  vinyl     Bandlimit + Flächenrauschen + Nadel-Klicks (Alterung)
  kassette  Flächenrauschen + Wow/Flutter + Bandlimit (Band-Kopien)
  clipping  harte Sättigung bis ins Clipping (Übersteuerung)
  mp3_128   ECHTES 128-kbps-Codec-Artefakt (ffmpeg/libmp3lame-Roundtrip)

Ausgabe (vom Ohr-Panel automatisch als Restauriert-Layout erkannt):
  <stamm> (<rezept>).wav              — beschädigter 10-s-Mittelausschnitt
  <stamm> (<rezept>)_restauriert.wav  — Aurik-Produktionsergebnis

CPU-only (`AURIK_FORCE_CPU=1`, nice-Priorität): stört die laufende GPU-Kette
nicht. Deterministisch (§G5 (copilot-instructions.md)): gleiche Quellen +
gleicher Seed ⇒ identische Proben.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SR = 48000
REZEPTE = ("vinyl", "kassette", "clipping", "mp3_128")


def _quell_dateien(verzeichnis: Path) -> list[Path]:
    """Saubere Quellen: mp3/flac/wav, keine Restaurations-Render und keine
    eigenen Ausgabeordner (kein Selbstvergleich)."""
    aus: list[Path] = []
    for p in sorted(verzeichnis.rglob("*")):
        if p.suffix.lower() not in (".mp3", ".flac", ".wav") or not p.is_file():
            continue
        s = p.stem.lower()
        if "_restauriert" in s or s.endswith("_original"):
            continue
        if any(f"/{name}/" in str(p) for name in ("output", "schadens_proben", "stimuli")):
            continue
        aus.append(p)
    return aus


def _decode_48k(pfad: Path) -> np.ndarray:
    """Mono@48 kHz via ffmpeg (deckt mp3/flac/wav ab)."""
    import soundfile as sf

    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(pfad), "-ac", "1", "-ar", str(SR), "-f", "wav", "-"],
        capture_output=True,
        check=True,
    )
    data, _ = sf.read(io.BytesIO(r.stdout), always_2d=False)
    return np.asarray(data, dtype=np.float32)


def _mittel_ausschnitt(audio, sekunden: float) -> np.ndarray:
    """Deterministischer Mittel-Ausschnitt (wie das Ohr-Panel)."""
    arr = np.asarray(audio)
    want = int(round(sekunden * SR))
    if arr.shape[-1] <= want or sekunden <= 0:
        return arr
    start = (arr.shape[-1] - want) // 2
    return arr[..., start : start + want]


def klicker(x: np.ndarray, rng, dichte: float = 4.0) -> np.ndarray:
    """Sparse Nadel-Klicks (Vinyl-Knistern-Komponente), deterministisch über rng."""
    y = np.asarray(x, dtype=np.float64).copy()
    n_klicks = max(1, int(y.size / SR * dichte))
    for _ in range(n_klicks):
        pos = int(rng.integers(0, max(1, y.size - 20)))
        amp = float(rng.uniform(0.08, 0.35)) * (1.0 if rng.random() < 0.5 else -1.0)
        y[pos : pos + 20] += amp * np.hanning(20)
    return y.astype(np.float32)


def _mp3_128(x: np.ndarray) -> np.ndarray:
    """Echtes 128-kbps-Artefakt: WAV → MP3-128 (libmp3lame) → zurück.

    Deterministisch für eine gegebene Encoder-Version (§G5 (copilot-instructions.md)).
    """
    import soundfile as sf

    wav_bytes = io.BytesIO()
    sf.write(wav_bytes, np.asarray(x, dtype=np.float32), SR, format="WAV")
    with tempfile.NamedTemporaryFile(suffix=".mp3") as tmp:
        r1 = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                "pipe:0",
                "-codec:a",
                "libmp3lame",
                "-b:a",
                "128k",
                "-f",
                "mp3",
                "pipe:1",
            ],
            input=wav_bytes.getvalue(),
            capture_output=True,
            check=True,
        )
        tmp.write(r1.stdout)
        tmp.flush()
        r2 = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", tmp.name, "-ac", "1", "-ar", str(SR), "-f", "wav", "-"],
            capture_output=True,
            check=True,
        )
        data, _ = sf.read(io.BytesIO(r2.stdout), always_2d=False)
        return np.asarray(data, dtype=np.float32)[: x.shape[-1]]


def schaden_rezept(x: np.ndarray, rezept: str, rng) -> np.ndarray:
    """Komposit-Schaden — deterministisch über rng (§G5 (copilot-instructions.md))."""
    from scripts.ohr_panel import defekt_variante

    x = np.asarray(x, dtype=np.float32)
    if rezept == "vinyl":
        y = defekt_variante(x, "bandlimit", rng)
        y = defekt_variante(y, "rausch", rng)
        return klicker(y, rng, dichte=4.0)
    if rezept == "kassette":
        y = defekt_variante(x, "rausch", rng)
        y = defekt_variante(y, "wowflutter", rng)
        return defekt_variante(y, "bandlimit", rng)
    if rezept == "clipping":
        y = np.tanh(8.0 * x) / np.tanh(8.0)
        return (y / (float(np.abs(y).max()) + 1e-9) * 0.9).astype(np.float32)
    if rezept == "mp3_128":
        return _mp3_128(x)
    raise ValueError(f"Unbekanntes Rezept: {rezept}")


def restauriere_schnipsel(audio: np.ndarray, sr: int, input_path: str, mode: str = "balanced") -> np.ndarray:
    """Produktions-Pfad: Pre-Analysis → Denker (backend/api/bridge), CPU-only.

    WICHTIG: `AURIK_FORCE_CPU=1` muss VOR dem Import gesetzt sein (main tut das),
    damit die laufende GPU-Kette nicht angefasst wird (ml_device_manager-Vertrag).
    """
    from backend.api.bridge import get_aurik_denker_instance, normalize_user_mode, run_pre_analysis

    pre = run_pre_analysis(
        audio_native=audio,
        sr_native=sr,
        audio_48k=audio,
        file_path=input_path,
        store_in_bridge_cache=True,
    )
    denker = get_aurik_denker_instance()
    result = denker.denke(
        audio,
        sr=sr,
        mode=normalize_user_mode(mode),
        no_rt_limit=os.environ.get("AURIK_NO_RT_LIMIT", "0") == "1",
        input_path=input_path,
        output_path="",
        pre_analysis_result=pre,
    )
    return np.asarray(result.audio, dtype=np.float32)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Beschädigte 10-s-Hörproben + Aurik-Restauration (CPU)")
    ap.add_argument("--verzeichnis", default=str(Path.home() / "Musik"), help="Musik-Ordner mit sauberen Quellen")
    ap.add_argument(
        "--ausgang", default=str(Path.home() / "Musik" / "schadens_proben"), help="Paar-Ordner (Panel-Layout)"
    )
    ap.add_argument("--rezepte", default=",".join(REZEPTE), help=f"Komma-getrennt: {REZEPTE}")
    ap.add_argument("--songs", type=int, default=3, help="Anzahl Quell-Titel (deterministische Auswahl)")
    ap.add_argument("--sekunden", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", default="balanced", help="Denker-Modus (balanced/…), nur Restauration")
    ap.add_argument("--nur-schaden", action="store_true", help="Nur beschädigen, KEINE Restauration (sofort)")
    args = ap.parse_args(argv)

    # CPU-Vertrag VOR allen backend-Importen (GPU-Kette läuft parallel).
    os.environ.setdefault("AURIK_FORCE_CPU", "1")
    os.environ.setdefault("AURIK_NO_RT_LIMIT", "1")

    import soundfile as sf

    verzeichnis = Path(args.verzeichnis)
    ausgang = Path(args.ausgang)
    ausgang.mkdir(parents=True, exist_ok=True)
    quellen = _quell_dateien(verzeichnis)
    if not quellen:
        raise FileNotFoundError(f"Keine sauberen Quellen in {verzeichnis}")
    rng = np.random.default_rng(args.seed)
    gewaehlt = [quellen[i] for i in rng.choice(len(quellen), size=min(args.songs, len(quellen)), replace=False)]
    rezepte = [r.strip() for r in args.rezepte.split(",") if r.strip()]
    for r in rezepte:
        if r not in REZEPTE:
            print(f"Unbekanntes Rezept: {r} (erlaubt: {REZEPTE})")
            return 2

    manifest: dict = {
        "schema": "aurik.schadensproben/1",
        "sekunden": args.sekunden,
        "seed": args.seed,
        "cpu_only": True,
        "flag_flips": "keine",
        "songs": [],
    }
    for idx_song, stueck in enumerate(gewaehlt):
        audio = _decode_48k(stueck)
        snippet = _mittel_ausschnitt(audio, args.sekunden)
        if snippet.shape[-1] < SR:  # kürzer als 1 s — überspringen
            continue
        for idx_rezept, rezept in enumerate(rezepte):
            rng_item = np.random.default_rng([args.seed, idx_song, idx_rezept])
            schaden = schaden_rezept(snippet, rezept, rng_item)
            stamm = stueck.stem[:60]
            schaden_pfad = ausgang / f"{stamm} ({rezept}).wav"
            sf.write(str(schaden_pfad), schaden, SR)
            restauriert_pfad: Path | None = None
            if not args.nur_schaden:
                restored = restauriere_schnipsel(schaden, SR, str(schaden_pfad), args.mode)
                restauriert_pfad = ausgang / f"{stamm} ({rezept})_restauriert.wav"
                sf.write(str(restauriert_pfad), np.asarray(restored, dtype=np.float32), SR)
            manifest["songs"].append(
                {
                    "quelle": str(stueck),
                    "stamm": stamm,
                    "rezept": rezept,
                    "schaden": str(schaden_pfad),
                    "restauriert": str(restauriert_pfad) if restauriert_pfad else None,
                }
            )
            print(f"{stamm[:40]:<40} [{rezept:<8}] {'+ restauriert' if restauriert_pfad else 'nur Schaden'}")

    (ausgang / "schadens_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nManifest: {ausgang / 'schadens_manifest.json'}")
    print(f'Anhören : python3 -B {ROOT / "scripts" / "ohr_panel.py"} --verzeichnis "{ausgang}"')
    return 0


if __name__ == "__main__":
    sys.exit(main())
