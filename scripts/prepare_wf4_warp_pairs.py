#!/usr/bin/env python3
"""D-K4-7 (SOTA-M12): Echt-Trainingspaare für WF-V4-Warp-Schätzung aus MUSDB18-HQ.

Erzeugt **Musik**-Mischungen (``mixture.wav``) aus MUSDB18-HQ, die mit BEKANNTEN
Wow/Flutter-Warp-Kurven (``warp_log2``) degradiert werden. Die GT ist die
Warp-Kurve selbst (``warp_log2(t)``) — Muster ``scripts/train_wf4_warp_head.py``.
``scripts/train_wf4_warp_head.py`` konsumiert die Paare über ``--real-dir``
(bzw. das Manifest ``aurik.wf4_warp_pairs/1``) und trainiert den Warp-Schätzer
auf echtem Musik-Material statt auf synthetischen Sinus-Kompositionen (§III.11
(copilot-instructions.md): Signalpfad-Kern ist Musik — Trainingsdaten müssen
Musik sein).

**Modell-Eingang ``x_f0`` (Tracker-Beobachtung, NICHT aus der GT gerechnet):**
Der Datensatz schreibt je Paar ``f0/<name>.npy`` (``log2(f0_degradiert /
f0_clean)``) und ``conf/<name>.npy`` (Frame-Konfidenz). Beide entstehen mit dem
PRODUKTIONS-Tracker ``backend/core/dsp/pyin_viterbi_fast.pyin_fast``
(bit-identisch zu librosa 0.11.0, CPU, deterministisch §G5 (GEBOTE.md)) — cleanes Segment
gegen degradiertes Segment. Damit ist die Beobachtung ehrlich ``clean`` vs.
``degradiert`` gemessen (kein Zirkelschluss über die bekannte Warp-Kurve) und
zugleich train/serve-konsistent zum Inferenzpfad (§III.11). Frames ohne
beidseitige Stimmhaftigkeit oder mit |Abweichung| > ``F0_MAX_ABWEICHUNG_OCT``
gelten als Tracker-Ausreißer: ``x_f0 = 0`` und ``x_conf = 0`` (der Kopf sieht
sie als „keine Beobachtung“).

Determinismus (§G5 (copilot-instructions.md)): Seed 42, Teil-Seeds als reine
Funktion von (seed, idx, Bestandteil), kein ``time.time()`` in der Logik.
Manifest mit sha256 je Datei (Pfeiler 4, Provenienz). 32-bit float WAV ⇒
bit_depth = 32 ⇒ keine Dither-Pflicht (§V5 (copilot-instructions.md) greift
nur bei bit_depth < 32).

Aufruf (Daten-Synthese, KEIN Training, KEIN Flag-Flip):
    python3 -B scripts/prepare_wf4_warp_pairs.py \\
        --musdb-root data/musdb18hq --out-dir data/wf4_warp_pairs --anzahl 300

K1-Konformität: KEIN ``--synthetic``-Pfad auf Sinus+Rauschen (nur „zum Test“),
sondern echte MUSDB18-HQ-Mischungen. Liegen keine ``mixture.wav`` vor, bricht
das Skript mit FileNotFoundError ab (§V7 (copilot-instructions.md) — nicht
geraten, nicht synthetisch aufgefüllt).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np

# Repo-Wurzel in den Suchpfad — der Produktions-Tracker
# (backend.core.dsp.pyin_viterbi_fast) wird beim Aufruf aus scripts/ sonst nicht
# gefunden (Skript-Start legt nur scripts/ in sys.path).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Synchro zu scripts/train_wf4_warp_head.py (TARGET_SR = 48000); hier als
# Konstante, damit das Datensatz-Werkzeug keine Pfad-Abhängigkeit zur
# Trainingsseite zieht (§G9 (copilot-instructions.md) Ein Quelle-Wahrung —
# über das Manifest gekoppelt).
TARGET_SR = 48_000
TARGET_FPS = 200  # Hz — Frame-Rate der Warp-Kurve (Synchro wf4_warp_head.TARGET_FPS)
WARP_LIMIT_LOG2 = 0.05  # ≈ 50 Cents — Grenze der Warp-Kurve (Synchro wf4_warp_head)

# Tracker-Parameter der F0-Beobachtung (Produktions-Tracker pyin_fast, §III.11).
F0_FMIN = 55.0  # Hz — tiefste Musik-Grundfrequenz (Bass/Mann)
F0_FMAX = 1000.0  # Hz — oberste Gesangs-/Instrumentalgrenze
F0_FRAME_LENGTH = 2048  # Samples @ 48 kHz (≈ 43 ms Analysefenster)
F0_HOP = TARGET_SR // TARGET_FPS  # 240 Samples ⇒ exakt 200 fps wie die Warp-Kurve
F0_MAX_ABWEICHUNG_OCT = 0.5  # ±600 Cent — darüber gilt die Beobachtung als Ausreißer

SCHEMA = "aurik.wf4_warp_pairs/1"
DEFAULT_SEED = 42


def _pcm_sha256(audio: np.ndarray) -> str:
    """sha256 über die **PCM-Daten** (float32 LE) — stabiler Provenienzwert.

    Der WAV-Container ist byte-instabil: libsndfile schreibt einen ``PEAK``-Chunk,
    dessen Float-Wert sich zwischen Läufen im letzten Bit unterscheiden kann
    (gemessen 2026-11-06: 1 Byte Differenz bei bit-identischer Audiodaten-Folge).
    ``sha256_wav`` allein wäre damit nicht reproduzierbar (§G5 (GEBOTE.md)) — der Vertrag ist
    deshalb der PCM-Hash.
    """
    a = np.ascontiguousarray(np.asarray(audio, dtype=np.float32).ravel(), dtype="<f4")
    return hashlib.sha256(a.tobytes()).hexdigest()


def _sha256(pfad: Path) -> str:
    """sha256 eines Binärpfads (Provenienz, Pfeiler 4)."""
    h = hashlib.sha256()
    with pfad.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _seed_for(seed: int, *parts: object) -> int:
    """Deterministischer Teil-Seed (§G5 (copilot-instructions.md)) — MUSTER
    scripts/train_wf4_warp_head.py:_seed_for."""
    raw = "|".join([str(seed), *(str(p) for p in parts)])
    h = hashlib.md5(raw.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def _lade_wav_mono(pfad: Path, target_sr: int) -> np.ndarray:
    """Lädt eine WAV mono auf ``target_sr`` (deterministische Resynthese)."""
    import librosa  # erst bei Bedarf — Hauptschnittstelle bleibt import-leicht

    mono, _ = librosa.load(str(pfad), sr=target_sr, mono=True)
    return mono.astype(np.float32)


def make_warp_curve(rng: np.random.Generator, n_frames: int) -> np.ndarray:
    """Bekannte Wow/Flutter-Warp-Kurve in log2-Stretch-Einheiten (GT).

    MUSTER scripts/train_wf4_warp_head.py:make_warp_curve — Wow 0,5–6 Hz +
    Flutter 8–60 Hz + langsamer Drift (Summe ≤ WARP_LIMIT_LOG2). Hier als
    Datensatz-Generator mit eigenem RNG (Seed 42, deterministisch).
    """
    t = np.arange(n_frames, dtype=np.float64) / TARGET_FPS

    def _komponente(freq_hz: float, amplitude: float, phase: float) -> np.ndarray:
        return amplitude * np.sin(2 * np.pi * freq_hz * t + phase)

    wow_freq = 0.5 + rng.uniform(0.0, 5.5)  # 0.5–6 Hz
    wow_amp = rng.uniform(0.01, 0.05) * WARP_LIMIT_LOG2
    flutter_freq = 8.0 + rng.uniform(0.0, 52.0)  # 8–60 Hz
    flutter_amp = rng.uniform(0.02, 0.08) * WARP_LIMIT_LOG2
    drift_amp = rng.uniform(0.0, 0.30) * WARP_LIMIT_LOG2

    warp = (
        _komponente(wow_freq, wow_amp, rng.uniform(0, 2 * np.pi))
        + _komponente(flutter_freq, flutter_amp, rng.uniform(0, 2 * np.pi))
        + drift_amp * np.linspace(-1.0, 1.0, n_frames)
    )
    # Strenge Einhaltung der Grenze (§0a (copilot-instructions.md) NaN/Inf-Schutz
    # + Warp-Grenzen wie in tests/unit/test_wf4_warp_head.py).
    peak = float(np.max(np.abs(warp))) if warp.size else 0.0
    if peak > WARP_LIMIT_LOG2:
        warp = warp * (WARP_LIMIT_LOG2 / peak)
    return warp.astype(np.float32)


def _wende_warp_an(audio: np.ndarray, warp_log2: np.ndarray) -> np.ndarray:
    """Degradiert ``audio`` durch Warp-Kurve (Capstan-Prinzip, y(t)=x(φ(t))).

    MUSTER backend/core/phases/phase_12_wow_flutter_fix.py: inverse-
    Geschwindigkeits-Warp; deterministische lineare Interpolation.
    """
    if audio.size == 0 or warp_log2.size == 0:
        return audio.astype(np.float32)
    # Warp-Kurve auf Sample-Rate hochinterpolieren (Frame-Rate TARGET_FPS).
    n = audio.size
    frame_idx = np.linspace(0.0, warp_log2.size - 1, n, dtype=np.float64)
    warp_per_sample = np.interp(frame_idx, np.arange(warp_log2.size, dtype=np.float64), warp_log2.astype(np.float64))
    # φ' = 2^warp ⇒ Geschwindigkeitsfaktor; kumulierte Verschiebung (Raten-Integration).
    rate = 2.0**warp_per_sample
    pos = np.cumsum(rate) - rate[0]  # monoton steigende Zielpositionen
    pos = np.clip(pos, 0.0, float(n - 1))
    # Normierung auf die Länge der Eingabe (kein Clipping am Ende).
    if pos[-1] > 0:
        pos = pos * ((n - 1) / pos[-1])
    out = np.interp(pos, np.arange(n, dtype=np.float64), audio.astype(np.float64))
    return out.astype(np.float32)


def beobachte_f0(
    clean: np.ndarray,
    degraded: np.ndarray,
    *,
    n_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Tracker-Beobachtung (``x_f0``, ``x_conf``) aus clean vs. degradiert.

    Nutzt den Produktions-Tracker ``pyin_fast`` (librosa-0.11.0-identisch, CPU,
    deterministisch) auf BEIDEN Signalen und bildet die log2-Abweichung der
    Grundfrequenz. Die GT-Warp-Kurve wird dabei bewusst NICHT verwendet — sonst
    wäre der Trainings-Eingang ein Zirkelschluss (§V6 copilot-instructions.md).

    Rückgabe: ``x_f0`` (``n_frames``, log2-Stretch) und ``x_conf`` (``n_frames``,
    0..1). Ausreißer/Nicht-Stimmhaftigkeit ⇒ beide 0 (keine Beobachtung).
    """
    x_f0 = np.zeros(int(n_frames), dtype=np.float32)
    x_conf = np.zeros(int(n_frames), dtype=np.float32)
    if clean.size < F0_FRAME_LENGTH or degraded.size < F0_FRAME_LENGTH:
        logger.warning("F0-Beobachtung übersprungen (Signal kürzer als frame_length=%d)", F0_FRAME_LENGTH)
        return x_f0, x_conf

    from backend.core.dsp.pyin_viterbi_fast import pyin_fast

    def _track(sig: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return pyin_fast(
            np.ascontiguousarray(sig, dtype=np.float32),
            fmin=F0_FMIN,
            fmax=F0_FMAX,
            sr=TARGET_SR,
            frame_length=F0_FRAME_LENGTH,
            hop_length=F0_HOP,
        )

    try:
        f0_c, v_c, p_c = _track(clean)
        f0_d, v_d, p_d = _track(degraded)
    except Exception as exc:  # §V6 (copilot-instructions.md): laut, nicht still
        logger.warning("F0-Beobachtung fehlgeschlagen (%s) — Paar ohne Beobachtung", exc)
        return x_f0, x_conf

    n = min(int(n_frames), f0_c.size, f0_d.size)
    if n <= 0:
        logger.warning("F0-Beobachtung leer (n=%d) — Paar ohne Beobachtung", n)
        return x_f0, x_conf

    with np.errstate(divide="ignore", invalid="ignore"):
        dev = np.log2(np.maximum(f0_d[:n], 1e-9) / np.maximum(f0_c[:n], 1e-9))
    gueltig = v_c[:n].astype(bool) & v_d[:n].astype(bool) & np.isfinite(dev) & (np.abs(dev) <= F0_MAX_ABWEICHUNG_OCT)
    x_f0[:n] = np.where(gueltig, dev, 0.0).astype(np.float32)
    x_conf[:n] = np.where(gueltig, np.minimum(p_c[:n], p_d[:n]), 0.0).astype(np.float32)
    # §0a: NaN/Inf-Schutz an der Datensatz-Grenze.
    return (
        np.nan_to_num(x_f0, nan=0.0, posinf=0.0, neginf=0.0),
        np.nan_to_num(x_conf, nan=0.0, posinf=0.0, neginf=0.0),
    )


def erzeuge_paare(
    musdb_root: Path,
    out_dir: Path,
    *,
    anzahl: int = 300,
    seg_s: float = 2.5,
    seed: int = DEFAULT_SEED,
    mit_f0: bool = True,
) -> dict:
    """Erzeugt ``anzahl`` Warp-Paare aus MUSDB18-HQ-Mischungen.

    Ausgabe: ``out_dir/audio/pair_XXXX.wav`` (degradiert, gewarpt) +
    ``out_dir/warp/pair_XXXX.npy`` (GT-Warp-Kurve, ``n_frames``) +
    ``out_dir/f0/pair_XXXX.npy``/``conf/pair_XXXX.npy`` (Tracker-Beobachtung,
    siehe :func:`beobachte_f0`) + Manifest.
    """
    if anzahl < 1 or seg_s <= 0:
        raise ValueError("anzahl >= 1 und seg_s > 0 erforderlich")

    mix_pfade = sorted(musdb_root.glob("train/*/mixture.wav")) + sorted(musdb_root.glob("test/*/mixture.wav"))
    if not mix_pfade:
        raise FileNotFoundError(f"keine mixture.wav unter {musdb_root} (train/test)")

    n_frames = int(round(seg_s * TARGET_FPS))
    seg_n = int(round(seg_s * TARGET_SR))

    audio_dir = out_dir / "audio"
    warp_dir = out_dir / "warp"
    f0_dir = out_dir / "f0"
    conf_dir = out_dir / "conf"
    audio_dir.mkdir(parents=True, exist_ok=True)
    warp_dir.mkdir(parents=True, exist_ok=True)
    if mit_f0:
        f0_dir.mkdir(parents=True, exist_ok=True)
        conf_dir.mkdir(parents=True, exist_ok=True)

    eintraege: list[dict] = []
    for i in range(anzahl):
        rng = np.random.default_rng(_seed_for(seed, "pair", i))
        src = mix_pfade[i % len(mix_pfade)]
        audio = _lade_wav_mono(src, TARGET_SR)
        n = audio.size
        if n < seg_n:
            logger.warning("Segment zu kurz (%d < %d) — %s übersprungen", n, seg_n, src)
            continue
        start = int(rng.integers(0, n - seg_n + 1))
        seg = audio[start : start + seg_n]

        warp = make_warp_curve(rng, n_frames)
        degraded = _wende_warp_an(seg, warp)

        name = f"pair_{i:04d}"
        wav_pfad = audio_dir / f"{name}.wav"
        npy_pfad = warp_dir / f"{name}.npy"
        # 32-bit float ⇒ bit_depth = 32 ⇒ keine Dither-Pflicht (§V5 (copilot-instructions.md)
        # (copilot-instructions.md) greift nur bei bit_depth < 32).
        import soundfile as sf

        sf.write(str(wav_pfad), degraded, TARGET_SR, subtype="FLOAT")
        np.save(str(npy_pfad), warp.astype(np.float32))

        eintrag = {
            "datei": name,
            "quelle": str(src),
            "start_sample": int(start),
            "seg_s": seg_s,
            "n_frames": n_frames,
            "warp_limit_log2": WARP_LIMIT_LOG2,
            "sha256_wav": _sha256(wav_pfad),
            "sha256_pcm": _pcm_sha256(degraded),
            "sha256_npy": _sha256(npy_pfad),
        }
        if mit_f0:
            x_f0, x_conf = beobachte_f0(seg, degraded, n_frames=n_frames)
            f0_pfad = f0_dir / f"{name}.npy"
            conf_pfad = conf_dir / f"{name}.npy"
            np.save(str(f0_pfad), x_f0)
            np.save(str(conf_pfad), x_conf)
            eintrag["sha256_f0"] = _sha256(f0_pfad)
            eintrag["sha256_conf"] = _sha256(conf_pfad)
            eintrag["f0_beobachtet_anteil"] = float(np.mean(x_conf > 0.0))
        eintraege.append(eintrag)
        if (i + 1) % 50 == 0:
            logger.info("%d/%d Paare erzeugt", i + 1, anzahl)

    manifest = {
        "schema": SCHEMA,
        "seed": seed,
        "target_sr": TARGET_SR,
        "target_fps": TARGET_FPS,
        "warp_limit_log2": WARP_LIMIT_LOG2,
        "seg_s": seg_s,
        "anzahl": len(eintraege),
        "musdb_root": str(musdb_root),
        "quelle": "scripts/prepare_wf4_warp_pairs.py",
        "provenienz": (
            "sha256_pcm = Hash der float32-PCM-Daten (verbindlich, §G5 (GEBOTE.md)); "
            "sha256_wav = Container-Bytes (informativ — libsndfile-PEAK-Chunk kann byte-abweichen)"
        ),
        "f0_beobachtung": (
            {
                "tracker": "backend.core.dsp.pyin_viterbi_fast.pyin_fast",
                "fmin_hz": F0_FMIN,
                "fmax_hz": F0_FMAX,
                "frame_length": F0_FRAME_LENGTH,
                "hop_length": F0_HOP,
                "max_abweichung_oct": F0_MAX_ABWEICHUNG_OCT,
                "felder": ["f0", "conf"],
            }
            if mit_f0
            else None
        ),
        "paare": eintraege,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("fertig: %d Paare nach %s (Manifest: manifest.json)", len(eintraege), out_dir)
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(
        description="D-K4-7 (SOTA-M12): WF-V4-Warp-Paare aus MUSDB18-HQ (Seed 42, Manifest+sha256)"
    )
    p.add_argument("--musdb-root", type=Path, default=Path("data/musdb18hq"))
    p.add_argument("--out-dir", type=Path, default=Path("data/wf4_warp_pairs"))
    p.add_argument("--anzahl", type=int, default=300)
    p.add_argument("--seg-s", type=float, default=2.5)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="§G5 (copilot-instructions.md)")
    p.add_argument(
        "--ohne-f0",
        action="store_true",
        help="keine Tracker-Beobachtung schreiben (nur GT-Warp; NICHT für Echt-Training)",
    )
    args = p.parse_args()
    erzeuge_paare(
        args.musdb_root,
        args.out_dir,
        anzahl=args.anzahl,
        seg_s=args.seg_s,
        seed=args.seed,
        mit_f0=not args.ohne_f0,
    )


if __name__ == "__main__":
    main()
