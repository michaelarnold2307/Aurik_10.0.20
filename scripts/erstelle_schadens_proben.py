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
  vinyl        Bandlimit + Flächenrauschen + Nadel-Klicks (Alterung)
  kassette     Flächenrauschen + Wow/Flutter + Bandlimit (Band-Kopien)
  clipping     harte Sättigung bis ins Clipping (Übersteuerung)
  mp3_128      ECHTES 128-kbps-Codec-Artefakt (ffmpeg/libmp3lame-Roundtrip)
  hum          Netzbrummen (Brummen-Variante)
  wowflutter   Wow/Flutter ohne weitere Alterung (Bandlauf)
  kompression  Überkompression (Pumpen/Dynamik-Verlust)
  kammfilter   Kammfilter (Phasen-/Reflexions-Artefakt)
  verzerrung   Verzerrung unterhalb des Clippings (Übersteuerungs-Vorstufe)
  clicks       Nadel-Klicks ohne weitere Alterung (Declick-Disziplin)
  pitch_drift  Langsame Tonhöhen-/Geschwindigkeitsdrift (+1 % Resample)
  dropout      Kurze Signal-Aussetzer (Riss-/Dropout-Disziplin)
  raum         Synthetischer Nachhall (0,5 s, exponentiell — Dereverb-Disziplin)
  lautheit     Überlaute Quelle (Lautheits-/Master-Finish-Disziplin)
  amp_drift    Tremolo/Amplituden-Drift (Pegel-Wogen)
  stereo_imbalance  Inter-Kanal-Gain-Ungleichgewicht ±1,5–6 dB (rng) + optionaler
                    einseitiger Tiefpass-Tilt — bleibt Stereo (2, N) @48 kHz

Paritäts-Register „Stereo-Imaging/Azimuth“: NUR das Rezept `stereo_imbalance`
wird stereo dekodiert (`_decode_48k_stereo`, `-ac 2`) und stereo durch den
Produktionspfad gereicht (Pre-Analysis → Denker, Stereo-Layout-Invariante
§III (copilot-instructions.md): channels-first (C, N)). Die Bestands-Rezepte
bleiben mono (bit-identisch, Pin-Tests in tests/unit/).

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
import logging
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
# Reihenfolge ist Vertrag: die Rezept-Indizes bestimmen die rng_item-Seeds
# [seed, song, idx] in main() — stereo_imbalance hängt deshalb HINTEN an,
# damit die Bestands-Rezepte bit-identisch bleiben (Pin-Test).
REZEPTE = (
    "vinyl",
    "kassette",
    "clipping",
    "mp3_128",
    "hum",
    "wowflutter",
    "kompression",
    "kammfilter",
    "verzerrung",
    "clicks",
    "pitch_drift",
    "dropout",
    "raum",
    "lautheit",
    "amp_drift",
    "stereo_imbalance",
)


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


def _decode_48k_stereo(pfad: Path) -> np.ndarray:
    """Stereo@48 kHz channels-first (C, N) via ffmpeg — NUR für `stereo_imbalance`.

    Bestands-Rezepte nutzen unverändert `_decode_48k` (mono, `-ac 1`).
    `-ac 2` erzwingt exakt zwei Kanäle (ffmpeg dupliziert Mono-Quellen);
    §III (copilot-instructions.md) Stereo-Layout-Invariante: intern (2, N),
    kein Mono-Kollaps (Paritäts-Register „Stereo-Imaging/Azimuth“).
    """
    import soundfile as sf

    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(pfad), "-ac", "2", "-ar", str(SR), "-f", "wav", "-"],
        capture_output=True,
        check=True,
    )
    data, _ = sf.read(io.BytesIO(r.stdout), always_2d=True)
    arr = np.asarray(data, dtype=np.float32).T  # (Frames, Kanäle) → (C, N)
    if arr.ndim != 2 or arr.shape[0] != 2:
        raise ValueError(f"Stereo-Decodierung lieferte {arr.shape}, erwartet (2, N)")
    return arr


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


STEREO_IMBALANCE_MIN_DB = 1.5
STEREO_IMBALANCE_MAX_DB = 6.0


def _stereo_imbalance(x: np.ndarray, rng) -> np.ndarray:
    """Deterministische Stereo-Schädigung (Paritäts-Register „Stereo-Imaging/Azimuth“).

    Inter-Kanal-Gain-Ungleichgewicht ±1,5–6 dB (rng) + optionaler einseitiger
    Tiefpass-Tilt auf demselben Kanal. Ausgabe bleibt Stereo channels-first
    (2, N) @48 kHz — KEIN Mono-Kollaps (§III (copilot-instructions.md)
    Stereo-Layout-Invariante). Deterministisch über rng (§G5 (copilot-instructions.md)).

    rng-Zieh-Reihenfolge ist Vertrag: kanal → gain_db → vorzeichen → tilt? →
    tilt_db → tilt-kante. Der Tilt ist RMS-neutral (nur die Spektral-Balance
    des Kanals kippt), damit die Inter-Kanal-Pegeldifferenz exakt gain_db bleibt.
    """
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        arr = np.stack([arr, arr], axis=0)  # defensiv: Mono-Quelle → (2, N)
    if arr.ndim != 2 or arr.shape[0] != 2:
        raise ValueError(f"stereo_imbalance erwartet (2, N), bekam {arr.shape}")
    kanal = int(rng.integers(0, 2))
    gain_db = float(rng.uniform(STEREO_IMBALANCE_MIN_DB, STEREO_IMBALANCE_MAX_DB))
    vorzeichen = 1.0 if rng.random() < 0.5 else -1.0
    y = arr.astype(np.float64)
    if rng.random() < 0.5:  # optionaler einseitiger Tiefpass-Tilt
        tilt_db = float(rng.uniform(1.0, 3.0))
        kante = float(rng.uniform(2000.0, 4000.0))
        spek = np.fft.rfft(y[kanal])
        freqs = np.fft.rfftfreq(y.shape[-1], d=1.0 / SR)
        h = np.ones_like(freqs)
        h[freqs > kante] = 10.0 ** (-tilt_db / 20.0)
        y[kanal] = np.fft.irfft(spek * h, n=y.shape[-1])
        rms_vor = float(np.sqrt((y[kanal] ** 2).mean()))
        if rms_vor > 1e-12:
            y[kanal] *= float(np.sqrt((arr[kanal].astype(np.float64) ** 2).mean())) / rms_vor
    y[kanal] *= 10.0 ** (vorzeichen * gain_db / 20.0)
    return y.astype(np.float32)


def _schreib_layout(audio) -> np.ndarray:
    """Intern channels-first (C, N) (§III (copilot-instructions.md)
    Stereo-Layout-Invariante) → Soundfile-Layout (Frames, Kanäle) für sf.write.

    Mono (1-D) und bereits frames-first gelieferte Arrays werden unverändert
    durchgereicht — Bestands-Rezepte bleiben bit-identisch.
    """
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[0] == 2 and arr.shape[-1] > 2:
        arr = arr.T  # (C, N) → (N, C)
    return arr


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
    if rezept == "hum":
        return defekt_variante(x, "brummen", rng)
    if rezept == "wowflutter":
        return defekt_variante(x, "wowflutter", rng)
    if rezept == "kompression":
        return defekt_variante(x, "kompression", rng)
    if rezept == "kammfilter":
        return defekt_variante(x, "kammfilter", rng)
    if rezept == "verzerrung":
        return defekt_variante(x, "verzerrung", rng)
    if rezept == "clicks":
        return klicker(x, rng, dichte=6.0)
    if rezept == "pitch_drift":
        from scipy.signal import resample_poly  # pylint: disable=import-outside-toplevel

        n = x.shape[-1]
        y = resample_poly(x.astype(np.float64), 101, 100).astype(np.float32)
        return y[:n] if len(y) >= n else np.pad(y, (0, n - len(y)))
    if rezept == "dropout":
        y = x.copy()
        for _ in range(20):
            s = int(rng.integers(0, max(1, y.shape[-1] - 240)))
            ln = int(rng.integers(24, 240))
            y[s : s + ln] = 0.0
        return y
    if rezept == "raum":
        ir = rng.standard_normal(24000).astype(np.float64)
        ir *= np.exp(-np.arange(24000) / 4000.0)
        ir /= float(np.abs(ir).max()) + 1e-9
        return np.convolve(x, ir, mode="full")[: x.shape[-1]].astype(np.float32)
    if rezept == "lautheit":
        return defekt_variante(x, "lautheit", rng)
    if rezept == "amp_drift":
        return defekt_variante(x, "tremolo", rng)
    if rezept == "stereo_imbalance":
        return _stereo_imbalance(x, rng)
    raise ValueError(f"Unbekanntes Rezept: {rezept}")


def restauriere_schnipsel(audio: np.ndarray, sr: int, input_path: str, mode: str = "balanced") -> np.ndarray:
    """Produktions-Pfad: Pre-Analysis → Denker (backend/api/bridge), CPU-only.

    WICHTIG: `AURIK_FORCE_CPU=1` muss VOR dem Import gesetzt sein (main tut das),
    damit die laufende GPU-Kette nicht angefasst wird (ml_device_manager-Vertrag).
    """
    # Exit-139-Fix (2026-11-19): numba/librosa-JIT-Kompilierung VOR dem
    # Denker-Thread erzwingen — lazy Erst-Kompilierung im Worker-Thread
    # segfaultete den Prozess (Faulthandler: <no Python frame> + numba).
    try:
        from backend.core.librosa_bootstrap import ensure_librosa_ready

        ensure_librosa_ready()
    except Exception:  # pragma: no cover - degradiert, nie den Produktionspfad blockieren
        logging.getLogger(__name__).debug("librosa/numba-Bootstrap nicht verfügbar — Ersatzpfade aktiv")
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
    # Crash-Diagnose (2026-11-19): nativer Crash landet im faulthandler-Log.
    import faulthandler as _fh_mod

    _fh_log = ROOT / "logs" / "python_faulthandler_ketten.log"
    _fh_log.parent.mkdir(parents=True, exist_ok=True)
    _fh_handle = open(_fh_log, "a", encoding="utf-8")
    _fh_mod.enable(file=_fh_handle, all_threads=True)

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
        mono_snippet: np.ndarray | None = None
        stereo_snippet: np.ndarray | None = None
        for idx_rezept, rezept in enumerate(rezepte):
            # Paritäts-Register „Stereo-Imaging/Azimuth“: NUR stereo_imbalance
            # wird stereo dekodiert (2, N) und stereo durchgereicht — die
            # Bestands-Rezepte bleiben mono und damit bit-identisch (Pin-Tests).
            if rezept == "stereo_imbalance":
                if stereo_snippet is None:
                    stereo_snippet = _mittel_ausschnitt(_decode_48k_stereo(stueck), args.sekunden)
                snippet = stereo_snippet
            else:
                if mono_snippet is None:
                    mono_snippet = _mittel_ausschnitt(_decode_48k(stueck), args.sekunden)
                snippet = mono_snippet
            if snippet.shape[-1] < SR:  # kürzer als 1 s — überspringen
                continue
            rng_item = np.random.default_rng([args.seed, idx_song, idx_rezept])
            schaden = schaden_rezept(snippet, rezept, rng_item)
            stamm = stueck.stem[:60]
            schaden_pfad = ausgang / f"{stamm} ({rezept}).wav"
            # Resume-Sicherheit (2026-11-19, Crash-Härtung): ein bereits fertiges
            # Paar (Schaden + Restaurat vorhanden) wird nicht neu gerechnet —
            # ein abgebrochener Lauf setzt ohne Zeitverlust fort.
            _rest_check = ausgang / f"{stamm} ({rezept})_restauriert.wav"
            if schaden_pfad.is_file() and _rest_check.is_file() and schaden_pfad.stat().st_size > 0:
                continue
            if rezept == "stereo_imbalance":
                # (C, N) → (Frames, Kanäle) für sf.write — Stereo bleibt Stereo.
                sf.write(str(schaden_pfad), _schreib_layout(schaden), SR)
            else:
                sf.write(str(schaden_pfad), schaden, SR)
            restauriert_pfad: Path | None = None
            if not args.nur_schaden:
                restored = restauriere_schnipsel(schaden, SR, str(schaden_pfad), args.mode)
                restauriert_pfad = ausgang / f"{stamm} ({rezept})_restauriert.wav"
                if rezept == "stereo_imbalance":
                    sf.write(str(restauriert_pfad), _schreib_layout(restored), SR)
                else:
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
