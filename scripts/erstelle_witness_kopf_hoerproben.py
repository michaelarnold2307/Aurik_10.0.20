"""scripts/erstelle_witness_kopf_hoerproben.py — §4-Hörproben für die Witness-Kopf-Kanäle.

D-K3-95-Folge (2026-11-15). Erzeugt deterministische A/B-Proben-Paare für die
Hör-Instanz (§4-Verfahren): A = schaden-Region (k398, unverändert), B = dieselbe
Region durch den gate-bestandenen Witness-Kopf (Produktions-DSP-Modul,
Trigger-Maske + Never-worsen). Verblindet über manifest.jsonl (Zuordnung A/B →
Kondition, Seed 42) — die Hör-Instanz sieht nur die Nummern.

CPU-only (der Kopf läuft auf CPU), Seed 42 (§G5 (copilot-instructions.md)),
kein Flag-Flip, keine Verdrahtung berührt. Proben ersetzen nie ein Gate: sie sind
das Material für den §4-Sign-off DER GATE-BESTANDENEN Kanäle.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

MANIFEST = _ROOT / "data" / "klassen_paare" / "manifest.jsonl"
AUSGANG_BASE = _ROOT / "study_stimuli" / "hoerproben" / "witness_kopf"
SR = 44100
SEED = 42
SCHEMA = "aurik.hoerproben_witness_kopf/1"

KANAELE = {
    "groove_echo": ("witness_kopf_groove_echo", "witness_kopf_groove_echo"),
    "tape_head_level_dip": ("witness_kopf_tape_head", "witness_kopf_tape_head_level_dip"),
    "clicks": ("witness_kopf_clicks", "witness_kopf_clicks"),
    "compression_artifacts": ("witness_kopf_compression", "witness_kopf_compression_artifacts"),
    "tape_head_clog": ("witness_kopf_tape_head_clog", "witness_kopf_tape_head_clog"),
    "phase_rotation": ("witness_kopf_phase_rotation", "witness_kopf_phase_rotation"),
    "sibilance": ("witness_kopf_sibilance", "witness_kopf_sibilance"),
    "sticky_shed_residue": ("witness_kopf_sticky_shed", "witness_kopf_sticky_shed_residue"),
    "jitter_artifacts": ("witness_kopf_jitter", "witness_kopf_jitter_artifacts"),
}


def _lade_mono(pfad: str) -> np.ndarray:
    from scipy.io import wavfile

    gelesen = wavfile.read(pfad)
    sr = gelesen[0]
    d = gelesen[1]
    x = np.asarray(d, np.float32) / 32768.0
    mono = x.mean(axis=1) if x.ndim == 2 else x
    if sr != SR:
        import librosa

        mono = librosa.resample(mono, orig_sr=sr, target_sr=SR).astype(np.float32)
    return mono


def _schreibe_wav(pfad: Path, x: np.ndarray) -> None:
    from scipy.io import wavfile

    x = np.clip(x, -1.0, 1.0)
    wavfile.write(str(pfad), SR, (x * 32767.0).astype(np.int16))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--klassen", default="groove_echo,tape_head_level_dip", help="Komma-Liste")
    ap.add_argument("--paare", type=int, default=6, help="A/B-Paare je Klasse")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args(argv)

    eintraege: dict[str, list[dict]] = {}
    for line in MANIFEST.open(encoding="utf-8"):
        d = json.loads(line)
        eintraege.setdefault(d["klasse"], []).append(d)

    gesamt = {"schema": SCHEMA, "datum": date.today().isoformat(), "seed": args.seed, "klassen": {}}
    for klasse in args.klassen.split(","):
        modul, funktion = KANAELE[klasse]
        import importlib

        dsp = importlib.import_module(f"backend.core.dsp.{modul}")
        f = getattr(dsp, funktion)
        regionen = [e for e in eintraege.get(klasse, []) if Path(e["schaden"]).is_file()]
        rng = np.random.default_rng(args.seed)
        kandidaten = [regionen[i] for i in rng.permutation(len(regionen))]
        ziel = AUSGANG_BASE / klasse
        ziel.mkdir(parents=True, exist_ok=True)
        zeilen = []
        for e in kandidaten:
            if len(zeilen) >= args.paare:
                break
            x = _lade_mono(e["schaden"])
            # 1-s-Fenster aus der Regionsmitte (konsistent mit Training/Gate); die
            # k398-Regionen sind 0,1 s — das Modul pad't auf 1 s (Training-Parität).
            if len(x) >= SR:
                start = (len(x) - SR) // 2
                x = x[start : start + SR]
            elif len(x) < 512:
                continue
            repariert = f(x, SR)
            if repariert is None:
                # Trigger ohne Gewinn/Never-worsen ⇒ Kopf lehnt ab — für die
                # Hör-Instanz nur WIRKSAME Regionen aufnehmen (Seed-Permutation
                # weiterlaufen, bis args.paare wirksame gefunden sind).
                continue
            if np.array_equal(np.asarray(repariert, np.float32), x):
                continue
            n = len(zeilen) + 1
            a_pfad = ziel / f"probe{n:02d}_a.wav"
            b_pfad = ziel / f"probe{n:02d}_b.wav"
            _schreibe_wav(a_pfad, x)
            _schreibe_wav(b_pfad, np.asarray(repariert, np.float32))
            zeilen.append(
                {
                    "probe": n,
                    "a": str(a_pfad.relative_to(_ROOT)),
                    "b": str(b_pfad.relative_to(_ROOT)),
                    "titel": e["titel"],
                }
            )
        (ziel / "manifest.jsonl").write_text(
            "\n".join(json.dumps(z, ensure_ascii=False) for z in zeilen) + "\n", encoding="utf-8"
        )
        gesamt["klassen"][klasse] = {"paare": len(zeilen), "ziel": str(ziel.relative_to(_ROOT))}
        print(json.dumps({klasse: gesamt["klassen"][klasse]}, ensure_ascii=False))
    beleg = _ROOT / "docs" / "reports" / "current" / f"{date.today().isoformat()}_hoerproben_witness_kopf.json"
    beleg.write_text(json.dumps(gesamt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Beleg: {beleg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
