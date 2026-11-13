#!/usr/bin/env python3
"""Trainings-Flugschein — Fail-fast-Gate vor jedem GPU-Trainingsjob.

Strategie (Report `docs/reports/current/2026-11-13_hoerzentrierte_trainingsstrategie.md`):
Tagelange Fehlversuche (D-K3-94 FlashSR, D-K3-96 BigVGAN) scheiterten an
Fehlern, die in Minuten erkennbar gewesen wären: Mel-Drift (Training
`log(abs-STFT)` vs. Produktion `power_to_db(ref=max)`), fehlendes
`gate_hint`, Daten ohne fail-closed-Manifest. Dieses Skript ist die
**Stufe A** der Strategie: ohne Flugschein kein GPU-Job.

Prüfungen (deterministisch, CPU-only, Sekunden):
  1. Mel-Parität — der Trainer konsumiert die Produktions-Mel
     (`compute_mel_produktion`, D-K3-96/§G9 (copilot-instructions.md)); eigenes log(abs-STFT)-Mel im
     Trainer ist das D-K3-96-Muster und wird abgelehnt.
  2. gate_hint — der Trainer dokumentiert die Mel-/Gate-Konvention.
  3. Daten fail-closed — Manifest existiert, hat Einträge, sha256_pcm;
     synthetische Fallback-Marker („synthetic") ohne echte Paare werden
     abgelehnt (§V6 (copilot-instructions.md)).

Aufruf:
    python3 -B scripts/trainings_flugschein.py --trainer scripts/train_bigvgan_f3.py [--data data/.../manifest.jsonl] [--out ...]
Exit 0 = Flugschein ausgestellt; 3 = abgelehnt (Befund auf stderr).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# D-K3-96-Muster: eigenes Mel aus log(abs-STFT) statt Produktions-Mel.
_MEL_DRIFT_MUSTER = ("log(abs", "log10(abs", "np.log(np.abs", "amplitude_to_db", "librosa.power_to_db")
_PRODUKTIONS_MEL = "compute_mel_produktion"
# Trainer-Familie, für die die Produktions-Mel-Parität der VERIFIZIERTE Vertrag ist
# (HR-V1/§1.4-Vocoder, D-K3-96). Andere Trainer (Cantus/WF4/TP2/Gender/Vocos) haben
# eigene Paritäts-Verträge und werden vom Flugschein (noch) nicht geprüft.
MEL_FAMILIE_TRAINER: tuple[str, ...] = ("train_bigvgan_f3.py", "train_flashsr_f4.py")


def _ohne_docstrings_und_kommentare(text: str) -> str:
    """D-K3-96-Drift nur in ausführbarem Code zählen, nicht in Doku/Kommentaren."""
    import re

    ohne_docstrings = re.sub(r'""".*?"""', "", text, flags=re.DOTALL)
    zeilen = [l for l in ohne_docstrings.splitlines() if l.strip() and not l.strip().startswith("#")]
    return "\n".join(zeilen)


def mel_paritaet_pruefen(trainer_pfad: Path) -> tuple[bool, str]:
    """Trainer-Quelltext muss die PRODUKTIONS-Mel nutzen (D-K3-96)."""
    text = _ohne_docstrings_und_kommentare(trainer_pfad.read_text(encoding="utf-8"))
    nutzt_produktion = _PRODUKTIONS_MEL in text
    drift = [m for m in _MEL_DRIFT_MUSTER if m in text]
    if not nutzt_produktion:
        return False, (
            f"Mel-Parität verletzt: {trainer_pfad.name} nutzt nicht {_PRODUKTIONS_MEL} — "
            "der Trainer trainiert nicht in der Produktions-Domäne (D-K3-96, §G9 (copilot-instructions.md))"
        )
    if drift:
        return False, (
            f"Mel-Parität verletzt: {trainer_pfad.name} enthält eigenes Mel-Muster {drift} "
            f"neben {_PRODUKTIONS_MEL} — EINE Mel-Wahrheit (D-K3-96)"
        )
    return True, f"Mel-Parität ok: {_PRODUKTIONS_MEL} wird konsumiert"


def gate_hint_pruefen(trainer_pfad: Path) -> tuple[bool, str]:
    """gate_hint muss im Trainer stehen (D-K3-96-Fix, Test-gepinnt)."""
    text = trainer_pfad.read_text(encoding="utf-8")
    if "gate_hint" not in text:
        return False, f"gate_hint fehlt in {trainer_pfad.name} — Gate-Konvention undokumentiert (D-K3-96)"
    return True, "gate_hint vorhanden"


def daten_pruefen(daten_pfad: Path) -> tuple[bool, str]:
    """Manifest fail-closed: existiert, Einträge, sha256_pcm, kein stiller Synthetik-Fallback."""
    if not daten_pfad.exists():
        return False, f"Daten-Manifest fehlt: {daten_pfad}"
    zeilen = [l for l in daten_pfad.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not zeilen:
        return False, f"Daten-Manifest leer: {daten_pfad}"
    eintraege = sum(1 for l in zeilen if l.lstrip().startswith("{"))
    if eintraege == 0:
        return False, f"Daten-Manifest ohne JSON-Einträge: {daten_pfad}"
    text = "\n".join(zeilen)
    if "synthetic" in text and "sha256_pcm" not in text:
        return False, (
            f"Daten-Manifest {daten_pfad} nennt synthetische Paare ohne sha256_pcm-Nachweis — "
            "K1-Domänen-Defizit-Muster (§III.11 (copilot-instructions.md))"
        )
    return True, f"Daten ok: {eintraege} Manifest-Einträge"


def flugschein_ausstellen(trainer_pfad: Path, daten_pfad: Path | None, out_pfad: Path) -> int:
    pruefungen: list[dict] = []
    ok, befund = mel_paritaet_pruefen(trainer_pfad)
    pruefungen.append({"pruefung": "mel_paritaet", "ok": ok, "befund": befund})
    ok2, befund2 = gate_hint_pruefen(trainer_pfad)
    pruefungen.append({"pruefung": "gate_hint", "ok": ok2, "befund": befund2})
    if daten_pfad is not None:
        ok3, befund3 = daten_pruefen(daten_pfad)
        pruefungen.append({"pruefung": "daten", "ok": ok3, "befund": befund3})
    ergebnis = {
        "schema": "aurik.trainingsflugschein/1",
        "trainer": str(trainer_pfad),
        "daten": str(daten_pfad) if daten_pfad else None,
        "pruefungen": pruefungen,
        "flugschein": all(p["ok"] for p in pruefungen),
    }
    out_pfad.parent.mkdir(parents=True, exist_ok=True)
    out_pfad.write_text(json.dumps(ergebnis, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for p in pruefungen:
        print(f"{p['pruefung']}: {'OK' if p['ok'] else 'ABGELEHNT'} — {p['befund']}")
    if ergebnis["flugschein"]:
        print(f"Flugschein ausgestellt: {out_pfad}")
        return 0
    print("Flugschein ABGELEHNT — kein GPU-Job (Fail-fast, Stufe A)", file=sys.stderr)
    return 3


def epoch_gate_pruefung(model_fn, basis_fn, track, sr: int, device, *, schwelle_db: float = 1.0) -> dict:
    """Stufe B der Trainingsstrategie: Produktions-Domänen-HNR je Epoche.

    `model_fn`/`basis_fn`: callable(mel_tensor [1, M, T]) -> audio [1, T] —
    der Trainer übergibt sein Live-Modell und ein Basis-Modell (Checkpoint der
    Produktions-Basis). Gemessen wird die SYNTHESE-Güte im Produktions-Mel
    (eine Wahrheit: `verify_bigvgan_d3_gate._mel_prod` + `validate_hr_v1._hnr_db`) —
    exakt das D-K3-96-Signal, aber je Epoche statt nach Tagen.

    Rückgabe: `abbruch=True` bei Degradation > schwelle_db gegen die Basis.
    """
    import numpy as np
    import torch

    from scripts.validate_hr_v1 import _hnr_db
    from scripts.verify_bigvgan_d3_gate import _mel_prod

    mono = np.asarray(track, dtype=np.float32)
    if mono.ndim == 2:
        mono = mono.mean(axis=-1)
    mel = np.asarray(_mel_prod(mono, int(sr)), dtype=np.float32)
    mel_t = torch.from_numpy(mel).unsqueeze(0).to(device)
    with torch.no_grad():
        out_ft = model_fn(mel_t).squeeze(1)
        out_basis = basis_fn(mel_t).squeeze(1)
    n = min(mono.shape[-1], int(out_ft.shape[-1]), int(out_basis.shape[-1]))
    ft = out_ft.detach().cpu().numpy().reshape(-1)[:n]
    basis = out_basis.detach().cpu().numpy().reshape(-1)[:n]
    hnr_ft = float(_hnr_db(ft))
    hnr_basis = float(_hnr_db(basis))
    degradation = hnr_ft - hnr_basis
    return {
        "hnr_finetune_db": round(hnr_ft, 4),
        "hnr_basis_db": round(hnr_basis, 4),
        "degradation_db": round(degradation, 4),
        "abbruch": bool(degradation < -float(schwelle_db)),
    }


def flugschein_pflicht(trainer_pfad: Path) -> bool:
    """True nur für die Trainer-Familie mit verifiziertem Mel-Paritäts-Vertrag."""
    return trainer_pfad.name in MEL_FAMILIE_TRAINER


def flugschein_trainer_pruefen(trainer_pfad: Path) -> tuple[bool, str]:
    """Kombinierte Trainer-Prüfung für den Runner (Mel-Parität + gate_hint)."""
    ok1, befund1 = mel_paritaet_pruefen(trainer_pfad)
    if not ok1:
        return False, befund1
    ok2, befund2 = gate_hint_pruefen(trainer_pfad)
    if not ok2:
        return False, befund2
    return True, "Flugschein ok (Mel-Parität + gate_hint)"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Trainings-Flugschein (Fail-fast vor GPU-Jobs)")
    ap.add_argument("--trainer", required=True)
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default=str(ROOT / "docs" / "reports" / "current" / "trainings_flugschein.json"))
    args = ap.parse_args(argv)
    trainer_pfad = (ROOT / args.trainer) if not Path(args.trainer).is_absolute() else Path(args.trainer)
    if not trainer_pfad.exists():
        print(f"FEHLER: Trainer fehlt: {trainer_pfad}", file=sys.stderr)
        return 2
    daten_pfad = Path(args.data) if args.data else None
    if daten_pfad is not None and not daten_pfad.is_absolute():
        daten_pfad = ROOT / daten_pfad
    return flugschein_ausstellen(trainer_pfad, daten_pfad, Path(args.out))


if __name__ == "__main__":
    raise SystemExit(main())
