"""SGMSE+ 3-Wege-Hörabnahme-Paket bauen (D-K0-3 / KP-1 des SOTA-Trainings-Konzepts).

Erzeugt aus den drei Metrik-A/B-Songs das fehlende Hörabnahme-Paket für die
Entscheidung ``use_sgmse_musik``: je Song × Fenster werden die drei Wege über
die **kanonischen Plugin-Pfade** generiert (WSJ0-Sprach-Core, Musik-Finetune-
Core, Nara-WPE-DSP) und als doppelblinde 2AFC-Präferenzstudie aufgebaut —
exakt das Format von ``scripts/mushra_harness.py thresholds-build`` +
``scripts/hoerpanel_player.py`` (task=preference).

**Evidenz-Kopplung:** Arm-Generierung ist identisch zum Metrik-Lauf 2026-10-08
(``scripts/benchmark_sgmse_ab.py``, Reports ``docs/reports/current/2026-10-08_
sgmse_ab_metrik*.json``): gleiche Lade-/Resample-Funktionen (importiert, nicht
kopiert — §G9 (copilot-instructions.md) eine Quelle), gleiche Degradation
(synthetischer Nachhall + Rauschen @ 10 dB SNR). Fenster ``w04`` mit Seed 42 ist
damit **bit-exakt** der Metrik-Evidenz; ``w20`` (Seed 43) erweitert die
Stichprobe („mehr Stichproben“ laut D-K0-3-Blocker).

**Triangelschaltung** (Drei-Wege-Vergleich über drei Paare, Kandidat = erste Seite):

1. ``sgmse_musik_vs_wsj0``  — Musik-Core schlägt den Sprach-Core?
2. ``sgmse_musik_vs_wpe``   — ML-Weg schlägt den DSP-Dereverb (Flag-Entscheid)?
3. ``sgmse_wsj0_vs_wpe``    — Transitivitäts-/Konsistenzprüfung der Urteile

+ 2 Fangfragen (ITU-R BS.1534-Anker, 3,5-kHz-Tiefpass; die ungedeckelte Seite
wird erwartet). Metriken bleiben Zeugen — das Ohr entscheidet (§III.11
copilot-instructions.md); das Skript aktiviert nichts selbst (§V7 copilot-
instructions.md).

**Lautheits-Fairness:** SGMSE+-Plugin-Ausgaben sind Peak-normalisiert (~0 dBFS),
WPE trägt Input-Energie — die verarbeiteten Arme werden daher deterministisch auf
die RMS-Lautheit des degradierten Inputs geanchert, bevor der Harness den
Präferenz-Level-Match vornimmt (sonst bliebe bis zu ~3.5 dB Restbias als
Hinweisreiz). SI-SDR ist skaleninvariant — die Metrik-Evidenz bleibt unberührt.

Determinismus: §G5 (copilot-instructions.md) — Seeds fix, A/B-Zuordnung aus
sha256(seed:trial_id), Lauf unter ``AURIK_FORCE_CPU=1`` (CPU = Referenz-EP,
§III.9 copilot-instructions.md). Fehlgeschlagene Arme brechen laut ab (§V6
copilot-instructions.md).

Nutzung::

    AURIK_FORCE_CPU=1 .venv_aurik/bin/python scripts/build_sgmse_hoerprobe_3weg.py
    # Player danach:  python scripts/hoerpanel_player.py --study output_audio/mushra/thresholds_sgmse_3weg --port 8765
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_ROOT / "scripts"))

# Identische Konstanten/Funktionen wie der Metrik-Lauf (keine Kopie):
from benchmark_sgmse_ab import _MUSIC_PATH, _SPEECH_PATH, _SR, _degrade, _run_sgmse

SONGS = (
    ("amcontra", "AM_Contra_-_Heart_Peripheral__gt_vocals.wav"),
    ("aljames", "Al_James_-_Schoolboy_Facination__gt_vocals.wav"),
    ("motortapes", "Motor_Tapes_-_Shore__gt_vocals.wav"),
)
_SRC_DIR = _ROOT / "output" / "scnet_ab_2026-10-06_fair"  # GT-Vocal-Stems (30 s, 44,1 kHz)

# (Fenstername, Start in s, Degradations-Seed): w04/Seed 42 = bit-exakt der
# Metrik-Evidenz 2026-10-08 (--start-s 4.0 --seconds 8.0 --snr-db 10 --seed 42).
WINDOWS = (
    ("w04", 4.0, 42),
    ("w20", 20.0, 43),
)
SECONDS = 8.0
SNR_DB = 10.0
SIGMA = 0.5
MAX_RUNTIME_S = 300.0
STUDY_SEED = 20261022  # A/B-Zuordnung + Hörer-Reihenfolge (doppelblind)
BUILD_NAME = "sgmse_3weg"

# Paare je Song × Fenster: (Kandidat, Referenz, Klasse). Der Kandidat ist die
# Seite, deren Präferenz thresholds-fit als Treffer zählt.
PAIRS = (
    ("musik", "wsj0", "sgmse_musik_vs_wsj0"),
    ("musik", "wpe", "sgmse_musik_vs_wpe"),
    ("wsj0", "wpe", "sgmse_wsj0_vs_wpe"),
)

QUESTION = (
    "Welches der beiden Gesangs-Ergebnisse klingt klarer und natuerlicher — "
    "Hall/Rauschen besser entfernt, Stimme ohne Verfärbung?"
)


def _load_clean(filename: str) -> np.ndarray:
    """GT-Vocal-Stem laden + auf 48 kHz bringen (EXAKT wie benchmark_sgmse_ab.main())."""
    import soundfile as sf
    from scipy.signal import resample_poly

    src, sr0 = sf.read(str(_SRC_DIR / filename), always_2d=True)
    return np.ascontiguousarray(resample_poly(src, _SR, sr0).T, dtype=np.float32)


def _clip(audio: np.ndarray, start_s: float) -> np.ndarray:
    n0 = int(start_s * _SR)
    n1 = n0 + int(SECONDS * _SR)
    clip = np.ascontiguousarray(audio[:, n0:n1], dtype=np.float32)
    if clip.shape[1] < int(SECONDS * _SR):
        raise SystemExit(f"Ausschnitt zu kurz: {clip.shape[1] / _SR:.2f} s (start_s={start_s})")
    return clip


def _write_wav(path: Path, audio_cn: np.ndarray) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    # FLOAT-WAV = exakter float32-Roundtrip; die Studie selbst quantisiert PCM_24.
    sf.write(str(path), audio_cn.T, _SR, format="WAV", subtype="FLOAT")


def _rms_db(x: np.ndarray) -> float:
    v = np.asarray(x, dtype=np.float64)
    return float(20.0 * np.log10(float(np.sqrt(np.mean(np.square(v)))) + 1e-12))


def _anchor_to_rms(audio_cn: np.ndarray, target_rms_db: float, label: str) -> np.ndarray:
    """Lautheits-Anker für Präferenzvergleich (deterministisch).

    Die SGMSE+-Plugin-Ausgaben sind Peak-normalisiert (~0 dBFS), die WPE-Ausgabe
    trägt Input-Energie — ohne Anker wären die Arme bis zu ~29 dB lautstärk-verschieden
    und der Harness-Level-Match (Peak-Cap 0.999) ließe >3 dB Restbias als Hinweisreiz.
    SI-SDR ist skaleninvariant, die Metrik-Evidenz bleibt damit unberührt."""
    g = 10.0 ** ((target_rms_db - _rms_db(audio_cn)) / 20.0)
    scaled = np.ascontiguousarray(audio_cn * g, dtype=np.float32)
    pk = float(np.max(np.abs(scaled))) if scaled.size else 0.0
    if pk > 0.999:  # fail-closed (§V6 copilot-instructions.md): Anker darf nie clippen
        raise SystemExit(f"RMS-Anker {label}: Peak {pk:.3f} > 0.999 nach Skalierung")
    return scaled


def build_arms(work_dir: Path) -> dict[str, dict]:
    """Generiert alle Arme (degraded/wsj0/musik/wpe) je Song × Fenster; gibt Pfade zurück."""
    from plugins.wpe_plugin import get_wpe_plugin

    wpe = get_wpe_plugin()
    paths: dict[str, dict] = {}
    for short, filename in SONGS:
        audio = _load_clean(filename)
        for win, start_s, seed in WINDOWS:
            ddir = work_dir / short / win
            clean = _clip(audio, start_s)
            degraded = _degrade(clean, seed=seed, snr_db=SNR_DB)
            target_rms = _rms_db(degraded)

            generated: dict[str, tuple[np.ndarray, str, float]] = {}
            for label, core in (("wsj0", _SPEECH_PATH), ("musik", _MUSIC_PATH)):
                if not core.exists():
                    raise SystemExit(f"Core fehlt: {core}")
                t0 = time.monotonic()
                out, used = _run_sgmse(
                    core, degraded, SIGMA, MAX_RUNTIME_S
                )  # §V6 (copilot-instructions.md): Fehler brechen laut ab
                generated[label] = (np.asarray(out, dtype=np.float32), used, round(time.monotonic() - t0, 1))
                logger.info("%s/%s %s: %.1f s (%s)", short, win, label, generated[label][2], used)

            t0 = time.monotonic()
            generated["wpe"] = (
                np.asarray(wpe.enhance(degraded, _SR), dtype=np.float32),
                "nara-wpe",
                round(time.monotonic() - t0, 1),
            )
            logger.info("%s/%s wpe: %.1f s", short, win, generated["wpe"][2])

            # Lautheits-Anker: alle verarbeiteten Arme auf Input-RMS (Präferenz-Fairness).
            _write_wav(ddir / "degraded.wav", degraded)
            arms: dict[str, tuple[Path, str, float]] = {"degraded": (ddir / "degraded.wav", "keine (Baseline)", 0.0)}
            for label in ("wsj0", "musik", "wpe"):
                arr, used, dt = generated[label]
                p = ddir / f"{label}.wav"
                _write_wav(p, _anchor_to_rms(arr, target_rms, f"{short}/{win} {label}"))
                arms[label] = (p, used, dt)

            paths[f"{short}:{win}"] = {
                label: {"file": str(p.relative_to(_ROOT)), "model_used": m, "seconds": d}
                for label, (p, m, d) in arms.items()
            }
    return paths


def build_manifest(arm_paths: dict[str, dict]) -> tuple[dict, list[dict]]:
    """Manifest + Trial-Liste für thresholds-build (semantische Rollen)."""
    trials: list[dict] = []
    for short, _ in SONGS:
        for win, _, _ in WINDOWS:
            arms = arm_paths[f"{short}:{win}"]
            for cand_arm, ref_arm, cls in PAIRS:
                trials.append(
                    {
                        "trial_id": f"sgmse_{short}_{win}_{cand_arm}_vs_{ref_arm}",
                        "class": cls,
                        "reference": arms[ref_arm]["file"],
                        "candidate": arms[cand_arm]["file"],
                        "start_s": 0.0,
                    }
                )
    # Fangfragen: Basis = degradierte Clips (2 verschiedene Songs/Fenster); der
    # 3,5-kHz-Tiefpass-Anker wird vom Harness auf die Kandidaten-Seite gelegt.
    for i, (short, win) in enumerate((("amcontra", "w04"), ("motortapes", "w20")), start=1):
        trials.append(
            {
                "trial_id": f"fangfrage_sgmse_{i}",
                "class": "catch",
                "reference": arm_paths[f"{short}:{win}"]["degraded"]["file"],
                "start_s": 0.0,
                "catch": True,
            }
        )

    n_real = len(SONGS) * len(WINDOWS) * len(PAIRS)
    manifest = {
        "build": BUILD_NAME,
        "task": "preference",
        "seed": STUDY_SEED,
        "question": QUESTION,
        "note": (
            f"SGMSE+ 3-Wege-Hörabnahme (D-K0-3): {n_real} Präferenz-Trials "
            f"(3 Songs × {len(WINDOWS)} Fenster × 3 Paare: musik_vs_wsj0, musik_vs_wpe, wsj0_vs_wpe) "
            "+ 2 Fangfragen. Arme über kanonische Plugin-Pfade generiert (identisch zum Metrik-Lauf "
            "2026-10-08; w04/Seed 42 bit-exakt der Metrik-Evidenz, w20/Seed 43 = erweiterte Stichprobe). "
            "Degradation: synthetischer Nachhall (45 %) + Rauschen @ 10 dB SNR, σ=0.5, AURIK_FORCE_CPU=1; "
            "verarbeitete Arme RMS-anchored an die Input-Lautheit (SGMSE-Plugin-Ausgaben sind Peak-normalisiert), "
            "danach Harness-Level-Match für den Präferenz-Typ. "
            "Entscheidung: use_sgmse_musik-Flag-Flip nur nach Hörordnungs-Sign-off (§III.11 "
            "copilot-instructions.md); Metrik-Berichte docs/reports/current/2026-10-08_sgmse_ab_metrik*.json."
        ),
        "trials": trials,
    }
    return manifest, trials


def main() -> int:
    ap = argparse.ArgumentParser(description="SGMSE+ 3-Wege-Hörabnahme-Paket (D-K0-3)")
    ap.add_argument("--work-dir", default=str(_ROOT / "output" / "sgmse_hoerprobe_3weg"))
    ap.add_argument(
        "--manifest",
        default=str(_ROOT / "docs" / "reports" / "current" / "2026-10-23_hoerprobe_sgmse_3weg.json"),
    )
    args = ap.parse_args()

    work_dir = Path(args.work_dir)
    manifest, trials = build_manifest(arm_paths=build_arms(work_dir))

    man_path = Path(args.manifest)
    man_path.parent.mkdir(parents=True, exist_ok=True)
    man_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Manifest: {man_path} ({len(trials)} Trials)")

    # Studie bauen (kanonischer Harness: A/B-Zuordnung, Level-Match für Präferenz, doppelblind).
    cmd = [
        sys.executable,
        str(_ROOT / "scripts" / "mushra_harness.py"),
        "thresholds-build",
        "--build",
        BUILD_NAME,
        "--manifest",
        str(man_path),
        "--level-match",
        "--seed",
        str(STUDY_SEED),
    ]
    print(" ".join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        return rc
    study_dir = _ROOT / "output_audio" / "mushra" / f"thresholds_{BUILD_NAME}"
    print(f"\nStudie: {study_dir}")
    print(f"Hörer starten:  python scripts/hoerpanel_player.py --study {study_dir} --port 8765")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    sys.exit(main())
