#!/usr/bin/env python3
"""bewerte_klassen_kopf — §G13-Q-Erstbeleg eines Per-Klassen-Reparatur-Kopfes.

Trainingsstrategie Stufe 3 (Report
`docs/reports/current/2026-11-13_hoerzentrierte_trainingsstrategie.md`):
Der Erfolg eines Kopfes wird NIE am Trainings-Verlust, sondern an der
Reparaturquote Q je Defektklasse gemessen (§G13 (copilot-instructions.md),
§G191 (GEBOTE.md)): Q = Δ(schaden→repariert) / Δ(schaden→clean) je Region,
pegel-gematcht, JND 0,05 (eine Wahrheit `wohlklang_zeuge.JND_TOLERANZ`),
Ziel Q ≥ `REPARATURQUOTE_ZIEL` (0,5). Report-Only-Zeuge (Hörordnung §7),
kein Flag-Flip — Freigaben laufen ausschließlich über §4.

Aufruf:
    python3 -B scripts/bewerte_klassen_kopf.py \
        --klasse groove_echo --manifest data/klassen_paare/manifest.jsonl \
        --checkpoint models/klassen_koepfe/groove_echo/checkpoint_latest.pt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.core.dsp.defekt_delta import REPARATURQUOTE_ZIEL, defektlast_lang
from backend.core.dsp.wohlklang_zeuge import JND_TOLERANZ

_SPLIT_SCHEMA = "aurik.klassen_kopf_split/1"
_MIN_Q_PAARE = 6


def _wav_lesen(pfad: Path) -> tuple[int, np.ndarray]:
    from scipy.io import wavfile

    gelesen = wavfile.read(pfad)
    if not isinstance(gelesen, tuple) or len(gelesen) < 2:  # Bug 12: Index-basiert statt Unpacking
        raise ValueError("wavfile.read() ohne (sr, data)-Tupel")
    sr = int(gelesen[0])
    daten = np.asarray(gelesen[1])
    if np.issubdtype(daten.dtype, np.integer):
        daten = daten.astype(np.float32) / 32768.0
    else:
        daten = daten.astype(np.float32)
    return sr, daten


def _fenster(audio: np.ndarray, laenge: int) -> np.ndarray:
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim == 2:
        x = x.mean(axis=-1)
    if x.shape[0] >= laenge:
        return x[:laenge]
    out = np.zeros(laenge, dtype=np.float32)
    out[: x.shape[0]] = x
    return out


def _titel_hash(titel: str) -> str:
    return hashlib.sha256(titel.encode("utf-8")).hexdigest()


def _q_berechnen(schaden: float, repariert: float, clean: float) -> float | None:
    """§G13-Q nur bei Clean-Kontrast oberhalb des gemeinsamen JND."""
    delta_clean = schaden - clean
    if delta_clean <= JND_TOLERANZ:
        return None
    return (schaden - repariert) / delta_clean


def _evidence_status(
    *,
    q_met: bool,
    never_worsen: bool,
    control_preserved: bool,
    evaluation_complete: bool,
    holdout_title_count: int,
) -> str:
    if not (q_met and never_worsen and control_preserved and evaluation_complete):
        return "nicht_bestanden"
    return "vorlaeufig_bestanden" if holdout_title_count < 3 else "bestanden"


def _holdout_zeilen(eintraege: list[dict], provenance: dict) -> tuple[list[dict], int]:
    """Wählt nur den manifestierten Titel-Holdout und entfernt sample-identische Lecks."""
    if provenance.get("schema") != _SPLIT_SCHEMA:
        raise ValueError("Checkpoint enthält keine unterstützte Titel-Holdout-Provenienz")
    train_titles = set(provenance.get("train_titel_sha256", []))
    holdout_title = provenance.get("holdout_titel_sha256")
    if not train_titles or not holdout_title or holdout_title in train_titles:
        raise ValueError("Checkpoint-Titel-Split ist unvollständig oder überlappt")

    train_entries = [e for e in eintraege if _titel_hash(str(e.get("titel", "")).strip()) in train_titles]
    holdout_entries = [e for e in eintraege if _titel_hash(str(e.get("titel", "")).strip()) == holdout_title]
    if not train_entries or not holdout_entries:
        raise ValueError("Manifest passt nicht zum gespeicherten Titel-Holdout")

    train_hashes = {
        e.get(key) for e in train_entries for key in ("sha256_pcm_schaden", "sha256_pcm_clean") if e.get(key)
    }
    disjoint = [
        e
        for e in holdout_entries
        if not any(e.get(key) in train_hashes for key in ("sha256_pcm_schaden", "sha256_pcm_clean") if e.get(key))
    ]
    return disjoint, len(holdout_entries) - len(disjoint)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="§G13-Q-Erstbeleg eines Per-Klassen-Kopfes")
    ap.add_argument("--klasse", default="groove_echo")
    ap.add_argument("--manifest", default="data/klassen_paare/manifest.jsonl")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--max-regionen", type=int, default=0, help="optionale deterministische Obergrenze; 0 = alle")
    args = ap.parse_args(argv)

    from scripts.train_klassen_kopf import _MODEL_SCHEMA, WINDOW, build_model

    manifest_pfad = ROOT / args.manifest
    if not manifest_pfad.exists():
        print(f"FEHLER: Manifest fehlt: {manifest_pfad}", file=sys.stderr)
        return 2
    manifest_bytes = manifest_pfad.read_bytes()
    zeilen = [json.loads(l) for l in manifest_bytes.decode("utf-8").splitlines() if l.strip()]
    eintraege = [z for z in zeilen if z.get("klasse") == args.klasse and z.get("clean") and z.get("titel")]
    if not eintraege:
        print(f"FEHLER: keine clean-Einträge für {args.klasse}", file=sys.stderr)
        return 2

    ckpt_pfad = Path(args.checkpoint)
    if not ckpt_pfad.exists():
        print(f"FEHLER: Checkpoint fehlt: {ckpt_pfad}", file=sys.stderr)
        return 2
    device = torch.device("cpu")
    try:
        res = torch.load(ckpt_pfad, map_location=device, weights_only=False)
        if res.get("model_schema") != _MODEL_SCHEMA:
            raise ValueError("Checkpoint-Modellvertrag fehlt oder ist inkompatibel")
        provenance = res.get("split_provenance", {})
        if provenance.get("manifest_sha256") != hashlib.sha256(manifest_bytes).hexdigest():
            raise ValueError("Manifest-Hash stimmt nicht mit dem Checkpoint überein")
        holdout_entries, n_overlap = _holdout_zeilen(eintraege, provenance)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"FEHLER: kein unabhängiger Holdout-Nachweis: {exc}", file=sys.stderr)
        return 2
    holdout_entries.sort(key=lambda z: z.get("sha256_pcm_schaden", ""))
    n_holdout_disjunkt = len(holdout_entries)
    if args.max_regionen > 0:
        holdout_entries = holdout_entries[: args.max_regionen]
    if len(holdout_entries) < _MIN_Q_PAARE:
        print(
            f"FEHLER: Holdout enthält nur {len(holdout_entries)} disjunkte Paare; "
            f"mindestens {_MIN_Q_PAARE} erforderlich",
            file=sys.stderr,
        )
        return 2

    model = build_model(device)
    try:
        model.load_state_dict(res["modell"])
    except (KeyError, RuntimeError, TypeError) as exc:
        print(f"FEHLER: Checkpoint-Modellzustand ungültig: {exc}", file=sys.stderr)
        return 2
    model.eval()

    quoten: list[float] = []
    defekt_deltas: list[float] = []
    kontroll_deltas: list[float] = []
    rms_delta_controls: list[float] = []
    zeilen: list[dict] = []
    for z in holdout_entries:
        sr, schaden = _wav_lesen(ROOT / z["schaden"])
        sr_clean, clean = _wav_lesen(ROOT / z["clean"])
        if sr != sr_clean:
            print(f"FEHLER: Sample-Raten unterscheiden sich: {z['schaden']} / {z['clean']}", file=sys.stderr)
            return 2
        schaden_w = _fenster(schaden, WINDOW)
        clean_w = _fenster(clean, WINDOW)
        with torch.no_grad():
            repariert = model(torch.from_numpy(schaden_w).unsqueeze(0).unsqueeze(0)).squeeze().numpy()
            clean_out = model(torch.from_numpy(clean_w).unsqueeze(0).unsqueeze(0)).squeeze().numpy()
        last_s = float(defektlast_lang(schaden_w, sr))
        last_r = float(defektlast_lang(repariert, sr))
        last_c = float(defektlast_lang(clean_w, sr))
        delta_rep = last_s - last_r  # positiv = besser
        delta_clean = last_s - last_c
        q = _q_berechnen(last_s, last_r, last_c)
        if q is not None:
            quoten.append(q)
        defekt_deltas.append(delta_rep)
        control_delta = float(defektlast_lang(clean_out, sr)) - last_c
        kontroll_deltas.append(control_delta)
        rms_delta_controls.append(float(np.sqrt(np.mean(np.square(clean_out - clean_w)))))
        zeilen.append(
            {
                "sha256": z.get("sha256_pcm_schaden", "")[:8],
                "defektlast_schaden": round(last_s, 5),
                "defektlast_repariert": round(last_r, 5),
                "defektlast_clean": round(last_c, 5),
                "delta_clean": round(delta_clean, 5),
                "q": round(q, 4) if q is not None else None,
                "delta_kontroll_clean": round(control_delta, 5),
                "rms_delta_kontroll_clean": round(rms_delta_controls[-1], 6),
            }
        )

    q_median = float(np.median(quoten)) if quoten else None
    never_worsen = bool(defekt_deltas and all(delta >= -JND_TOLERANZ for delta in defekt_deltas))
    control_preserved = bool(kontroll_deltas and all(abs(delta) <= JND_TOLERANZ for delta in kontroll_deltas))
    q_met = bool(len(quoten) >= _MIN_Q_PAARE and q_median is not None and q_median >= REPARATURQUOTE_ZIEL)
    holdout_titles = sorted({str(z.get("titel", "")) for z in holdout_entries})
    evaluation_complete = len(holdout_entries) == n_holdout_disjunkt
    evidence_status = _evidence_status(
        q_met=q_met,
        never_worsen=never_worsen,
        control_preserved=control_preserved,
        evaluation_complete=evaluation_complete,
        holdout_title_count=len(holdout_titles),
    )
    befund = {
        "schema": "aurik.klassen_kopf_q/2",
        "klasse": args.klasse,
        "checkpoint": str(ckpt_pfad),
        "checkpoint_sha256": hashlib.sha256(ckpt_pfad.read_bytes()).hexdigest(),
        "checkpoint_epoch": int(res.get("epoch", -1)) + 1,
        "checkpoint_seed": res.get("seed"),
        "checkpoint_val_mae": res.get("val_mae"),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "split_provenance": provenance,
        "metrik": "defektlast_lang",
        "metrik_richtung": "niedriger_besser",
        "q_definition": (
            "(defektlast_schaden - defektlast_repariert) / "
            "(defektlast_schaden - defektlast_clean); nur delta_clean > JND"
        ),
        "seed": provenance["seed"],
        "holdout_titel_sha256": provenance["holdout_titel_sha256"],
        "holdout_titel_anzahl": len(holdout_titles),
        "evidence_status": evidence_status,
        "n_regionen": len(zeilen),
        "n_holdout_regionen_disjunkt": n_holdout_disjunkt,
        "holdout_ausgewertet_vollstaendig": evaluation_complete,
        "n_ausgeschlossen_hash_ueberlappung": n_overlap,
        "n_q": len(quoten),
        "n_q_unter_jnd_ausgeschlossen": len(zeilen) - len(quoten),
        "q_jnd": JND_TOLERANZ,
        "q_mindest_paare": _MIN_Q_PAARE,
        "q_median": round(q_median, 4) if q_median is not None else None,
        "q_p25": round(float(np.percentile(quoten, 25)), 4) if quoten else None,
        "q_p75": round(float(np.percentile(quoten, 75)), 4) if quoten else None,
        "q_ziel": REPARATURQUOTE_ZIEL,
        "q_ziel_erfuellt": q_met,
        "nie_schlechter_anteil": round(float(np.mean([d >= -JND_TOLERANZ for d in defekt_deltas])), 4),
        "nie_schlechter_erfuellt": never_worsen,
        "kontrollregionen_jnd": JND_TOLERANZ,
        "kontrollregionen_erfuellt": control_preserved,
        "kontrollregionen_rms_delta_median": round(float(np.median(rms_delta_controls)), 6),
        "audibility_objective": "nicht_gemessen",
        "q_messung_vorlaeufig": len(holdout_titles) < 3,
        "zeilen": zeilen[:50],
        "hinweis": (
            "Report-Only-Zeuge (Hörordnung §7), keine Hörbarkeits-/Maskierungsmessung; "
            "kein Flag-Flip, Freigaben nur §4."
        ),
    }
    out_pfad = (
        Path(args.out) if args.out else (ROOT / "docs" / "reports" / "current" / f"klassen_kopf_q_{args.klasse}.json")
    )
    out_pfad.parent.mkdir(parents=True, exist_ok=True)
    out_pfad.write_text(json.dumps(befund, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Beleg: {out_pfad}")
    print(
        f"Q median={befund['q_median']} (Ziel >= {REPARATURQUOTE_ZIEL}) -> "
        f"{'erfuellt' if befund['q_ziel_erfuellt'] else 'verfehlt'} | "
        f"Never-worsen={'OK' if never_worsen else 'FAIL'} | "
        f"Kontrollregionen={'OK' if control_preserved else 'FAIL'} | "
        f"Holdout n={len(zeilen)}, Titel={len(holdout_titles)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
