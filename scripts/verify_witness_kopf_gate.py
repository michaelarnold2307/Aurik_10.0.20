"""scripts/verify_witness_kopf_gate.py — Aktivierungs-Gate für den Witness-Kopf (D-K3-95, 2026-11-15).

Misst das, was der M9-Pfad nicht konnte: defekt-spezifische Wirkung + Never-worsen.
- Gewinn-Arm: auf schaden-Regionen des Holdout-Titels muss der Kopf den Schaden
  reduzieren — Witness-Q = (P_s − P_out)/(P_s − P_c), Median ≥ 0,5.
- Schutz-Arm: auf clean-Regionen darf der Kopf nichts verschlechtern —
  P(out(clean)) ≤ P(clean) + 0,02 in ≥ 95 % der Regionen.
- DRT-8/10 (SOTA-HANDOVER Punkt 34f, D-K3-95-Folge): --mos schaltet die
  Zwei-Metrik-Messung zu — der zertifizierte musik_mos-Teacher misst den
  Wohlklang-Delta des Kopf-Ausgangs gegen den Schaden-Input als
  Report-only-Zeuge; das Gate-Urteil bleibt defektlast + Never-worsen.
CPU-only, Seed 42 (§G5 (copilot-instructions.md)), fail-closed, Report-only:
kein Flag-Flip, kein Produktionskonsument — die Aktivierung bleibt §4 vorbehalten.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

from scripts.train_witness_detektor import WitnessCRNN, lade_witness, witness_rms_tensor
from scripts.train_witness_kopf import (
    FENSTER,
    MANIFEST,
    NW_TOL,
    RMS_ZIEL,
    WitnessKopf,
    _baue_regionen,
    _lade_mono,
    _mel_torch,
    _mos_teacher_werte,
    _witness_q,
)
from scripts.train_witness_kopf import (
    SR as _TARGET_SR,
)

SCHEMA = "aurik.witness_kopf_gate/2"
Q_MIN = 0.5
NW_ANTEIL_MIN = 0.95
MOS_JND = 0.05  # DRT-8/10: musik_mos-JND (defekt_delta; Ablations-Befund 2026-11-16)
MAX_REGIONEN = 200
PRODUKTIONS_SEMANTIK: dict[str, tuple[str, str]] = {
    "groove_echo": ("witness_kopf_groove_echo", "witness_kopf_groove_echo"),
    "tape_head_level_dip": ("witness_kopf_tape_head", "witness_kopf_tape_head_level_dip"),
    "clicks": ("witness_kopf_clicks", "witness_kopf_clicks"),
    "compression_artifacts": ("witness_kopf_compression", "witness_kopf_compression_artifacts"),
    "tape_head_clog": ("witness_kopf_tape_head_clog", "witness_kopf_tape_head_clog"),
    "phase_rotation": ("witness_kopf_phase_rotation", "witness_kopf_phase_rotation"),
    "sibilance": ("witness_kopf_sibilance", "witness_kopf_sibilance"),
    "sticky_shed_residue": ("witness_kopf_sticky_shed", "witness_kopf_sticky_shed_residue"),
    "jitter_artifacts": ("witness_kopf_jitter", "witness_kopf_jitter_artifacts"),
    "transport_bump": ("witness_kopf_transport_bump", "witness_kopf_transport_bump"),
    "dynamic_compression_excess": (
        "witness_kopf_dynamic_compression",
        "witness_kopf_dynamic_compression_excess",
    ),
    "dropout_head_contact": ("witness_kopf_dropout_head", "witness_kopf_dropout_head_contact"),
    "hum": ("witness_kopf_hum", "witness_kopf_hum"),
    "dropouts": ("witness_kopf_dropouts", "witness_kopf_dropouts"),
    "dropout_oxide": ("witness_kopf_dropout_oxide", "witness_kopf_dropout_oxide"),
    "tape_splice_artifact": (
        "witness_kopf_tape_splice_artifact",
        "witness_kopf_tape_splice_artifact",
    ),
    "wow": ("witness_kopf_wow", "witness_kopf_wow"),
}


def _lade_produktions_semantik(
    klasse: str,
    checkpoint: Path,
    witness_checkpoint: Path,
) -> tuple[str | None, Any, str]:
    eintrag = PRODUKTIONS_SEMANTIK.get(klasse)
    if eintrag is None:
        return None, None, "nicht_zugeordnet"

    import importlib

    modul_name, funktions_name = eintrag
    modul = importlib.import_module(f"backend.core.dsp.{modul_name}")
    if Path(modul._KOPF_CKPT).resolve() != checkpoint.resolve():
        return modul_name, None, "kopf_checkpoint_abweichend"
    if Path(modul._WITNESS_CKPT).resolve() != witness_checkpoint.resolve():
        return modul_name, None, "witness_checkpoint_abweichend"
    modul_fn = getattr(modul, funktions_name)
    if not callable(modul_fn):
        raise TypeError(f"{modul_name}.{funktions_name} ist nicht aufrufbar")
    return modul_name, modul_fn, "verifiziert"


def _belege_produktions_semantik(
    ergebnis: dict[str, Any],
    modul_name: str | None,
    modul_fn: Any,
    status: str,
) -> dict[str, Any]:
    qualitaets_gate_bestanden = bool(ergebnis["gate_bestanden"])
    ergebnis["gate_qualitaet_bestanden"] = qualitaets_gate_bestanden
    ergebnis["produktions_semantik_verifiziert"] = modul_fn is not None
    ergebnis["schutz_arm_modul"] = modul_name
    ergebnis["schutz_arm_semantik"] = "produktions_dsp" if modul_fn is not None else "roher_kopf_diagnostisch"
    ergebnis["schutz_arm_status"] = status
    ergebnis["gate_bestanden"] = qualitaets_gate_bestanden and modul_fn is not None
    return ergebnis


def _lade(checkpoint: Path, klasse: str) -> tuple[Any, Any]:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Kopf-Checkpoint fehlt: {checkpoint}")
    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    kopf = WitnessKopf(int(ckpt.get("hidden", 64)), int(ckpt.get("stufen", 4)))
    kopf.load_state_dict(ckpt["state_dict"])
    kopf.eval()
    w_ckpt = _ROOT / "models" / f"witness_{klasse}" / "witness.pt"
    if not w_ckpt.is_file():
        raise FileNotFoundError(
            f"Witness-Checkpoint fehlt: {w_ckpt} — erst scripts/train_witness_detektor.py ausführen"
        )
    wd = torch.load(str(w_ckpt), map_location="cpu", weights_only=False)
    witness = lade_witness(wd)
    witness.load_state_dict(wd["state_dict"])
    witness.eval()
    return kopf, witness


def _roh_db(x: np.ndarray) -> float:
    """Wahrer Pegel (dBFS) eines RAW-Fensters — rms_skip-Quelle der Wahrheit.

    Formel-identisch zu ``witness_rms_global_db`` (eine Quelle, §V25)."""
    return float(10.0 * np.log10(float(np.mean(x.astype(np.float64) ** 2)) + 1e-10))


def _p(witness: Any, x: torch.Tensor, raw_db: float | None = None) -> float:
    """Witness-P auf dem RMS-normalisierten Fenster (Produktions-Parität, DRT-4).

    raw_db: wahrer Pegel (dBFS) des zugehörigen RAW-Fensters — die Produktions-
    Inferenz ``_witness_p`` führt genau diesen Pegel dem rms_skip-Witness zu
    (dort: Mel auf dem RMS-normalisierten Fenster, Pegel vom Roh-Signal). Ohne
    rms_skip-Witness ungenutzt (bit-identisch zum Vorgänger, §G5 (GEBOTE.md)).
    """
    mel = _mel_torch(x.reshape(len(x), -1))[:, None, :, :]
    if getattr(witness, "rms_skip", False):
        rms = torch.full((len(x),), float(raw_db) if raw_db is not None else -14.0, dtype=torch.float32)
        return float(torch.sigmoid(witness(mel, rms_global=rms)).mean())
    return float(torch.sigmoid(witness(mel)).mean())


def _fenster(x: np.ndarray) -> np.ndarray:
    if len(x) >= FENSTER:
        start = (len(x) - FENSTER) // 2
        return x[start : start + FENSTER]
    return np.pad(x, (0, FENSTER - len(x)))


def messe(
    kopf: Any,
    witness: Any,
    regionen: list[tuple[str, Path, Path]],
    modul_fn: Any = None,
    mos_messen: bool = False,
) -> dict[str, Any]:
    quoten: list[float] = []
    never_worsen: list[bool] = []
    mos_deltas: list[float] = []
    uebersprungen = 0
    with torch.no_grad():
        for _titel, s_pfad, c_pfad in regionen:
            s = _lade_mono(str(s_pfad))
            c = _lade_mono(str(c_pfad))
            if len(s) < 512 or len(c) < 512:
                uebersprungen += 1
                continue
            s_w, c_w = _fenster(s), _fenster(c)
            rms = float(np.sqrt(np.mean(s_w**2)) + 1e-8)
            rms_skip = bool(getattr(witness, "rms_skip", False))
            db_s = _roh_db(s_w)
            db_c = _roh_db(c_w)
            s_n = (s_w / rms * RMS_ZIEL).astype(np.float32)
            s_t = torch.from_numpy(s_n[None, None])
            out_s = kopf(s_t)
            if rms_skip:
                p_s = _p(witness, s_t, db_s)
                # Q-Arm in Produktions-Konvention (DRT-4): die Referenz p_c misst
                # clean mit der EIGENEN RMS + wahren Pegel — exakt ``_witness_p(c_w)``
                # der Produktions-Inferenz. Der Kopf-Ausgang wird wie in der
                # Produktion denormalisiert (·rms/RMS_ZIEL) und mit eigener RMS
                # renormiert gemessen.
                rms_c = float(np.sqrt(np.mean(c_w**2)) + 1e-8)
                c_ref = (c_w / rms_c * RMS_ZIEL).astype(np.float32)
                p_c = _p(witness, torch.from_numpy(c_ref[None, None]), db_c)
                out_raw = (out_s.detach().cpu().numpy()[0, 0] * rms / RMS_ZIEL).astype(np.float32)
                rms_out = float(np.sqrt(np.mean(out_raw**2)) + 1e-8)
                out_ref = (out_raw / rms_out * RMS_ZIEL).astype(np.float32)
                p_out_s = _p(witness, torch.from_numpy(out_ref[None, None]), _roh_db(out_raw))
            else:
                p_s = _p(witness, s_t)
                c_n = (c_w / rms * RMS_ZIEL).astype(np.float32)
                c_t = torch.from_numpy(c_n[None, None])
                p_c = _p(witness, c_t)
                p_out_s = _p(witness, out_s)
            q = _witness_q(p_s, p_c, p_out_s)
            if q is not None:
                quoten.append(q)
            if mos_messen:
                # DRT-8/10-Zwei-Metrik-Zeuge (Report-only, nur mit --mos): der
                # zertifizierte musik_mos-Teacher misst den Wohlklang-Delta des
                # Kopf-Ausgangs gegen den Schaden-Input (höher = besser; JND 0,05).
                lehrer_s = _mos_teacher_werte(s_w.astype(np.float32)[None])
                lehrer_out = _mos_teacher_werte(out_s.detach().cpu().numpy())
                if lehrer_s and lehrer_out:
                    mos_deltas.append(lehrer_out[0] - lehrer_s[0])
            # Schutz-Arm misst die PRODUKTIONS-Semantik (DRT-4, Befund 2026-11-15):
            # der Kopf wird in der Produktion auf clean-Regionen nur über Trigger +
            # Never-worsen-Guard angewendet — der rohe Kopf-auf-clean bestrafte etwas,
            # das nie passiert (Trigger auf clean nur 2 %, Witness p_c 0,21).
            if modul_fn is not None:
                out_c_prod = modul_fn(c_w, _TARGET_SR)
                if out_c_prod is None:
                    out_c_prod = c_w
                out_c_prod = np.asarray(out_c_prod, np.float32)
                # Produktions-Konvention (DRT-4) für ALLE Witnesses: der Modul-Ausgang
                # wird mit seiner EIGENEN RMS renormiert gemessen — die Referenz ist
                # clean mit der EIGENEN RMS (c_own), exakt die Inferenz-Konvention
                # ``_witness_p``. Die frühere RAW-Messung driftete bei leisen
                # clean-Regionen über die 0,02-Toleranz (dce-Befund 2026-11-23:
                # p_raw vs p_norm bis 0,046 auf Jets-Regionen mit rms 0,07–0,12).
                rms_p = float(np.sqrt(np.mean(out_c_prod.astype(np.float64) ** 2)) + 1e-8)
                out_p_ref = (out_c_prod / rms_p * RMS_ZIEL).astype(np.float32)
                rms_c = float(np.sqrt(np.mean(c_w.astype(np.float64) ** 2)) + 1e-8)
                c_own = (c_w / rms_c * RMS_ZIEL).astype(np.float32)
                if rms_skip:
                    p_out_c = _p(witness, torch.from_numpy(out_p_ref[None, None]), _roh_db(out_c_prod))
                    p_c_ref = _p(witness, torch.from_numpy(c_own[None, None]), db_c)
                else:
                    p_out_c = _p(witness, torch.from_numpy(out_p_ref[None, None]))
                    p_c_ref = _p(witness, torch.from_numpy(c_own[None, None]))
            else:
                # DRT-4-Parität (Spec 26): ohne Produktions-DSP misst der
                # Schutz-Arm den rohen Kopf auf clean mit CLEAN-EIGENER RMS —
                # exakt die Konvention des Identitäts-Ankers im Trainer
                # (_lade_split: xc_id/hc_id mit clean-eigener RMS) und der
                # Produktions-Inferenz (_kopf_out: jedes Fenster eigene RMS).
                # Die Schaden-RMS-Normalisierung (c_n/p_c) bleibt nur die
                # Witness-Referenz des Q-Arms (Trainer-Parität des Gewinn-Arms).
                rms_c = float(np.sqrt(np.mean(c_w**2)) + 1e-8)
                c_own = (c_w / rms_c * RMS_ZIEL).astype(np.float32)
                c_own_t = torch.from_numpy(c_own[None, None])
                if rms_skip:
                    p_c_ref = _p(witness, c_own_t, db_c)
                    out_own = kopf(c_own_t)
                    out_own_raw = (out_own.detach().cpu().numpy()[0, 0] * rms_c / RMS_ZIEL).astype(np.float32)
                    rms_o = float(np.sqrt(np.mean(out_own_raw**2)) + 1e-8)
                    out_own_ref = (out_own_raw / rms_o * RMS_ZIEL).astype(np.float32)
                    p_out_c = _p(witness, torch.from_numpy(out_own_ref[None, None]), _roh_db(out_own_raw))
                else:
                    p_c_ref = _p(witness, c_own_t)
                    p_out_c = _p(witness, kopf(c_own_t))
            never_worsen.append(p_out_c <= p_c_ref + NW_TOL)
    ergebnis = {
        "n_regionen": len(regionen),
        "n_uebersprungen": uebersprungen,
        "n_q_bewertbar": len(quoten),
        "n_never_worsen_bewertbar": len(never_worsen),
        "q_values": [float(q) for q in quoten],
        "never_worsen_flags": [bool(value) for value in never_worsen],
        "q_median": round(float(np.median(quoten)), 4) if quoten else None,
        "q_anteil_ge_0_5": round(float(np.mean([q >= Q_MIN for q in quoten])), 4) if quoten else None,
        "never_worsen_anteil": round(float(np.mean(never_worsen)), 4) if never_worsen else None,
        "gate_bestanden": bool(
            quoten and float(np.median(quoten)) >= Q_MIN and float(np.mean(never_worsen)) >= NW_ANTEIL_MIN
        ),
    }
    if mos_messen:
        # Report-only-DRT-8/10-Felder — ohne --mos bleibt das Ergebnis
        # bit-identisch zum Vorgänger (§G5 (GEBOTE.md)).
        ergebnis["musik_mos_delta_mittel"] = round(float(np.mean(mos_deltas)), 4) if mos_deltas else None
        ergebnis["musik_mos_delta_anteil_ge_minus_jnd"] = (
            round(float(np.mean([d >= -MOS_JND for d in mos_deltas])), 4) if mos_deltas else None
        )
        ergebnis["drt_8_10"] = "Report-only Zwei-Metrik-Messung (defektlast UND musik_mos); Gate-Urteil unverändert"
    return ergebnis


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--klasse", default="groove_echo")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--max-regionen", type=int, default=MAX_REGIONEN)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--mos",
        action="store_true",
        help="DRT-8/10-Zwei-Metrik-Messung (musik_mos-Teacher, Report-only) — Default AUS",
    )
    args = ap.parse_args(argv)
    checkpoint = (
        Path(args.checkpoint)
        if args.checkpoint
        else _ROOT / "models" / "klassen_koepfe" / f"witness_{args.klasse}_best.pt"
    )
    try:
        kopf, witness = _lade(checkpoint, args.klasse)
    except FileNotFoundError as exc:
        print(f"ABBRUCH: {exc}")
        return 2
    alle, holdout_titel, _ = _baue_regionen(args.klasse)
    hold = [r for r in alle if r[0] == holdout_titel]
    rng = np.random.default_rng(args.seed)
    hold = [hold[i] for i in rng.permutation(len(hold))][: args.max_regionen]
    witness_checkpoint = _ROOT / "models" / f"witness_{args.klasse}" / "witness.pt"
    modul_name, modul_fn, semantik_status = _lade_produktions_semantik(
        args.klasse,
        checkpoint,
        witness_checkpoint,
    )
    if modul_fn is None:
        print(
            "WARNUNG: Schutz-Arm nicht produktionsparitätsverifiziert "
            f"({semantik_status}); Rohkopf-Messung ist diagnostisch und kann das Gate nicht bestehen."
        )
    ergebnis = messe(kopf, witness, hold, modul_fn, mos_messen=args.mos)
    ergebnis = _belege_produktions_semantik(ergebnis, modul_name, modul_fn, semantik_status)
    beleg = {
        "schema": SCHEMA,
        "datum": date.today().isoformat(),
        "seed": args.seed,
        "klasse": args.klasse,
        "holdout_titel": holdout_titel,
        "checkpoint": str(checkpoint.relative_to(_ROOT)),
        **ergebnis,
    }
    ziel = _ROOT / "docs" / "reports" / "current" / f"{date.today().isoformat()}_witness_kopf_gate_{args.klasse}.json"
    ziel.write_text(json.dumps(beleg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(beleg, indent=2, ensure_ascii=False))
    print(f"Beleg: {ziel}")
    return 0 if ergebnis["gate_bestanden"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
