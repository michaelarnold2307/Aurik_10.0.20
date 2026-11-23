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
from collections.abc import Callable
from datetime import date
from pathlib import Path

import numpy as np

from backend.core.dsp.witness_kopf_clicks import witness_kopf_clicks
from backend.core.dsp.witness_kopf_compression import witness_kopf_compression_artifacts
from backend.core.dsp.witness_kopf_dropout_head import witness_kopf_dropout_head_contact
from backend.core.dsp.witness_kopf_dropout_oxide import witness_kopf_dropout_oxide
from backend.core.dsp.witness_kopf_dropouts import witness_kopf_dropouts
from backend.core.dsp.witness_kopf_dynamic_compression import witness_kopf_dynamic_compression_excess
from backend.core.dsp.witness_kopf_groove_echo import witness_kopf_groove_echo
from backend.core.dsp.witness_kopf_hum import witness_kopf_hum
from backend.core.dsp.witness_kopf_jitter import witness_kopf_jitter_artifacts
from backend.core.dsp.witness_kopf_phase_rotation import witness_kopf_phase_rotation
from backend.core.dsp.witness_kopf_sibilance import witness_kopf_sibilance
from backend.core.dsp.witness_kopf_sticky_shed import witness_kopf_sticky_shed_residue
from backend.core.dsp.witness_kopf_tape_head import witness_kopf_tape_head_level_dip
from backend.core.dsp.witness_kopf_tape_head_clog import witness_kopf_tape_head_clog
from backend.core.dsp.witness_kopf_tape_splice_artifact import witness_kopf_tape_splice_artifact
from backend.core.dsp.witness_kopf_transport_bump import witness_kopf_transport_bump
from backend.core.dsp.witness_kopf_wow import witness_kopf_wow

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

MANIFEST = _ROOT / "data" / "klassen_paare" / "manifest.jsonl"
AUSGANG_BASE = _ROOT / "study_stimuli" / "hoerproben" / "witness_kopf"
SR = 44100
SEED = 42
SCHEMA = "aurik.hoerproben_witness_kopf/1"

KANAELE = frozenset(
    {
        "groove_echo",
        "tape_head_level_dip",
        "clicks",
        "compression_artifacts",
        "tape_head_clog",
        "phase_rotation",
        "sibilance",
        "sticky_shed_residue",
        "jitter_artifacts",
        "transport_bump",
        "dynamic_compression_excess",
        "dropout_head_contact",
        "dropouts",
        "dropout_oxide",
        "tape_splice_artifact",
        "hum",
        "wow",
    }
)


def _parse_klassen(value: str) -> list[str]:
    classes = [name.strip() for name in value.split(",") if name.strip()]
    unknown = [name for name in classes if name not in KANAELE]
    if not classes or unknown or len(set(classes)) != len(classes):
        raise argparse.ArgumentTypeError(f"Ungueltige oder doppelte Klassen: {', '.join(unknown or classes)}")
    return classes


def _lade_funktion(klasse: str) -> Callable[..., np.ndarray | None]:
    # Statische Allowlist: CLI-Eingaben werden niemals als Modulnamen importiert.
    funktionen: dict[str, Callable[..., np.ndarray | None]] = {
        "groove_echo": witness_kopf_groove_echo,
        "tape_head_level_dip": witness_kopf_tape_head_level_dip,
        "clicks": witness_kopf_clicks,
        "compression_artifacts": witness_kopf_compression_artifacts,
        "tape_head_clog": witness_kopf_tape_head_clog,
        "phase_rotation": witness_kopf_phase_rotation,
        "sibilance": witness_kopf_sibilance,
        "sticky_shed_residue": witness_kopf_sticky_shed_residue,
        "jitter_artifacts": witness_kopf_jitter_artifacts,
        "transport_bump": witness_kopf_transport_bump,
        "dynamic_compression_excess": witness_kopf_dynamic_compression_excess,
        "dropout_head_contact": witness_kopf_dropout_head_contact,
        "dropouts": witness_kopf_dropouts,
        "dropout_oxide": witness_kopf_dropout_oxide,
        "tape_splice_artifact": witness_kopf_tape_splice_artifact,
        "hum": witness_kopf_hum,
        "wow": witness_kopf_wow,
    }
    # Resolve only through this fixed mapping; never import or look up an
    # attribute using the CLI value.
    if klasse not in funktionen:
        raise ValueError("Unbekannte Witness-Klasse")
    return funktionen[klasse]


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
    ap.add_argument(
        "--klassen",
        type=_parse_klassen,
        default=["groove_echo", "tape_head_level_dip"],
        help="Komma-Liste",
    )
    ap.add_argument("--paare", type=int, default=6, help="A/B-Paare je Klasse")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args(argv)

    eintraege: dict[str, list[dict]] = {}
    for line in MANIFEST.open(encoding="utf-8"):
        d = json.loads(line)
        eintraege.setdefault(d["klasse"], []).append(d)

    gesamt = {"schema": SCHEMA, "datum": date.today().isoformat(), "seed": args.seed, "klassen": {}}
    for klasse in args.klassen:
        f = _lade_funktion(klasse)
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
