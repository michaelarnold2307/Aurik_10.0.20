"""KP-4 · VAD-A/B-Harness — PANNs-Singing-Offset vs. neutralem Gate-Pfad (report-only-Evidenz).

Vergleicht den **musikadaptierten** PANNs-Singing-Gate-Offset
(`backend/core/dsp/music_vad_oracle.compute_music_vad_offset_from_audio`)
gegen den heutigen neutralen Pfad (Silero-Flag `use_silero_vad_music=False`
⇒ Offset 0 dB, §V7 (copilot-instructions.md)). Determinismus §G5
(copilot-instructions.md): Seed 42, deterministische synthetische Signale.
Kein Training, keine Gewichtsänderung, kein Flag-Flip — der produktive
Gate bleibt report-only unverändert.

Signale (synthetisch, Seed 42):
- ``vocal_*``: harmonischer Gesangs-Ersatz (Grundfrequenz + Obertöne + Vibrato) ⇒ erwartet
  PANNs-Singing hoch ⇒ Offset < 0 (Gate hält offen).
- ``instrumental_*``: perkussives/noise-artiges Instrumental ⇒ erwartet
  PANNs-Singing niedrig ⇒ Offset == 0 (konservativ, identische Gate-Entscheidung
  wie heute — Never-worsen).

Metriken:
- ``direction_correct``: vocal ⇒ offset < 0, instrumental ⇒ offset == 0.
- ``never_worsen_agreement``: Anteil der Fälle, in denen der PANNs-Offset die
  Gate-Entscheidung gegenüber dem neutralen Pfad NICHT verschlechtert
  (instrumental mit Offset 0 ⇒ identische Entscheidung; vocal mit Offset < 0
  ⇒ dokumentierte Abweichung, die per KP-4-Vertrag erst nach
  Never-worsen-Beweis + Hörordnungs-Sign-off aktiviert werden dürfte).
- Ehrliche Lücke (§G8 (copilot-instructions.md)): Wenn PANNs nicht ladbar ist,
  wird das im Report vermerkt statt Werte zu erfinden.

Ausgabe: docs/reports/current/<datum>_kp4_musik_vad.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SR = 32000  # PANNs-AudioSet-Arbeitsrate
MUSDB_TEST = ROOT / "data" / "musdb18hq" / "test"


def _harmonic_vocal(n_samples: int, f0: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples, dtype=np.float64) / SR
    # Vibrato (5–6 Hz, ±0.5 Halbton) + Obertöne mit abfallender Amplitude.
    vib = 1.0 + 0.03 * np.sin(2 * np.pi * 5.5 * t + rng.uniform(0, 2 * np.pi))
    y = np.zeros(n_samples, dtype=np.float64)
    for h in range(1, 9):
        y += (0.5 / h**1.3) * np.sin(2 * np.pi * f0 * h * vib * t + rng.uniform(0, 2 * np.pi))
    y /= np.abs(y).max() + 1e-8
    return np.asarray(y, dtype=np.float32)


def _instrumental(n_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Perkussive Impulse + bandbegrenztes Rauschen — kein Gesangscharakter.
    y = np.zeros(n_samples, dtype=np.float64)
    for pos in np.arange(n_samples // 8, n_samples, n_samples // 8):
        i0 = int(pos) + rng.integers(-n_samples // 40, n_samples // 40)
        i0 = int(np.clip(i0, 0, n_samples - 1))
        length = min(n_samples - i0, int(0.05 * SR))
        if length <= 0:
            continue
        decay = np.exp(-np.arange(length) / (0.008 * SR))
        y[i0 : i0 + length] += 0.9 * decay * np.sin(2 * np.pi * rng.uniform(80, 400) * np.arange(length) / SR)
    noise = rng.normal(0, 0.05, n_samples)
    noise = np.convolve(noise, np.ones(64) / 64.0, mode="same")
    y += noise
    y /= np.abs(y).max() + 1e-8
    return np.asarray(y, dtype=np.float32)


def _load_oracle():
    from backend.core.dsp.music_vad_oracle import compute_music_vad_offset_from_audio

    return compute_music_vad_offset_from_audio


def _load_musdb_classes(n_samples: int, max_tracks: int = 3) -> list[dict]:
    """Echte MUSDB18-Teststems: vocals = 'vocal', drums/other = 'instrumental'.

    Resampling auf SR via scipy.signal.resample_poly (deterministisch, §G5 (copilot-instructions.md)).
    Fehlt das Korpus: leere Liste (der Harness
    fällt auf die synthetischen Klassen zurück und dokumentiert das).
    """
    import wave

    from scipy.signal import resample_poly

    classes: list[dict] = []
    if not MUSDB_TEST.is_dir():
        return classes
    for track_dir in sorted(MUSDB_TEST.iterdir())[:max_tracks]:
        if not track_dir.is_dir():
            continue
        for stem_name, kind in (("vocals", "vocal"), ("drums", "instrumental"), ("other", "instrumental")):
            stem_path = track_dir / (stem_name + ".wav")
            if not stem_path.is_file():
                continue
            try:
                with wave.open(str(stem_path), "rb") as wf:
                    sr_in = wf.getframerate()
                    n_ch = wf.getnchannels()
                    need = int(n_samples * sr_in / SR) + 10
                    raw = wf.readframes(need)
                audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                audio = audio.reshape(-1, n_ch).T if n_ch > 1 else audio.reshape(1, -1)
                audio = audio.mean(axis=0)
                if sr_in != SR:
                    from math import gcd

                    g = gcd(sr_in, SR)
                    audio = resample_poly(audio, SR // g, sr_in // g)
                if len(audio) > n_samples:
                    audio = audio[:n_samples]
                else:
                    audio = np.pad(audio, (0, n_samples - len(audio)))
                classes.append(
                    {"name": f"musdb:{track_dir.name}:{stem_name}", "kind": kind, "audio": audio.astype(np.float32)}
                )
            except Exception as _exc:  # pylint: disable=broad-except
                print(f"[musdb] {stem_path.name} nicht lesbar ({_exc})", file=sys.stderr)
    return classes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42, help="Seed (Determinismus §G5 (copilot-instructions.md))")
    ap.add_argument("--seconds", type=float, default=4.0, help="Signaldauer je Klasse")
    ap.add_argument("--musdb-tracks", type=int, default=3, help="Anzahl MUSDB18-Test-Tracks (0 = aus)")
    ap.add_argument(
        "--out",
        type=str,
        default=f"docs/reports/current/{date.today().isoformat()}_kp4_musik_vad.json",
    )
    args = ap.parse_args()

    n_samples = int(args.seconds * SR)
    compute_offset = _load_oracle()

    classes: list[dict] = []
    # Echte MUSDB-Stems zuerst (musiknativ, §III.11 (copilot-instructions.md)).
    musdb_classes = _load_musdb_classes(n_samples, max_tracks=args.musdb_tracks)
    classes.extend(musdb_classes)
    for i, f0 in enumerate((196.0, 261.63, 329.63)):
        classes.append(
            {
                "name": f"vocal_{int(f0)}hz",
                "kind": "vocal",
                "audio": _harmonic_vocal(n_samples, f0, args.seed * 1000 + i),
            }
        )
    for i in range(3):
        classes.append(
            {
                "name": f"instrumental_{i}",
                "kind": "instrumental",
                "audio": _instrumental(n_samples, args.seed * 1000 + 100 + i),
            }
        )

    results = []
    pan_ns_missing = 0
    for cls in classes:
        t0 = time.time()
        witness = compute_offset(cls["audio"], SR)
        elapsed = time.time() - t0
        results.append(
            {
                "name": cls["name"],
                "kind": cls["kind"],
                "panns_singing": witness.panns_singing,
                "offset_db": witness.offset_db,
                "reasons": witness.reasons,
                "seconds": round(elapsed, 2),
            }
        )
        if witness.panns_singing is None:
            pan_ns_missing += 1

    # Bewertung
    direction_correct = 0
    direction_total = 0
    never_worsen_agreement = 0
    never_worsen_total = 0
    for r in results:
        if r["kind"] == "vocal":
            direction_total += 1
            if r["panns_singing"] is not None and r["offset_db"] < 0.0:
                direction_correct += 1
        else:
            direction_total += 1
            never_worsen_total += 1
            if r["offset_db"] == 0.0:
                direction_correct += 1
                never_worsen_agreement += 1

    report = {
        "date": date.today().isoformat(),
        "script": "scripts/benchmark_vad_oracles.py",
        "seed": args.seed,
        "signal_seconds": args.seconds,
        "sr": SR,
        "musdb_tracks": len({c["name"] for c in musdb_classes}),
        "results": results,
        "summary": {
            "n": len(results),
            "direction_correct": direction_correct,
            "direction_total": direction_total,
            "direction_rate": round(direction_correct / max(1, direction_total), 4),
            "never_worsen_agreement": never_worsen_agreement,
            "never_worsen_total": never_worsen_total,
            "panns_missing": pan_ns_missing,
            "note": (
                "Instrumental mit Offset 0 dB ⇒ identische Gate-Entscheidung wie der "
                "neutrale Pfad (Never-worsen); vocal mit Offset < 0 ist die dokumentierte "
                "prospektive Abweichung, deren Aktivierung erst nach Never-worsen-Beweis + "
                "Hörordnungs-Sign-off zulässig ist (KP-4-Vertrag)."
            ),
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report: {out_path}")
    print(
        f"Richtung: {direction_correct}/{direction_total} (rate {report['summary']['direction_rate']}), "
        f"Never-worsen (instrumental): {never_worsen_agreement}/{never_worsen_total}, "
        f"PANNs fehlend: {pan_ns_missing}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
