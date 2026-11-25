#!/usr/bin/env python
"""Wohlklang-Modul-Audit — Einzel-Modul-Beitrag je Phase auf dem Korpus.

§G14 (copilot-instructions.md) verlangt das Wohlklang-Maximum aus ALLEN
Modulen — auch in Kombination. Dieses Werkzeug misst den **Einzel-Modul-
Beitrag**: je Lauf werden die per-Phasen-Goal-Deltas (`result.metadata.
phase_deltas`, §v10.14 P3) und die Ordnungs-Metadaten (Verletzungen/
Enforcement) gesammelt und über den Korpus aggregiert — welche Phase
bewegt welches Wohlklang-Ziel um wie viel, und welche Phasen laufen in
Läufen mit Ordnungs-Verletzungen. (Kombinations-Effekte = Modus B/Ablation,
geplant als D-K3-110 — hier nicht enthalten.)

WICHTIG: NICHT parallel zur gepaarten DRT-12-Kampagne laufen lassen
(Systemlast stört die Kampagnen-Kinder über die Live-Notbremse).

Verwendung:
  AURIK_FORCE_CPU=1 AURIK_NO_RT_LIMIT=1 .venv_aurik/bin/python -B \
      scripts/wohlklang_modul_audit.py --paare 10 --seed 42
Beleg: docs/reports/current/<datum>_wohlklang_modul_audit.jsonl (Resume-fähig).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.environ.setdefault("AURIK_FORCE_CPU", "1")
os.environ.setdefault("AURIK_NO_RT_LIMIT", "1")

import numpy as np  # noqa: E402
from scipy.io import wavfile  # noqa: E402

logger = logging.getLogger(__name__)


def _rezept_aus_namen(name: str) -> str:
    stem = name[: -len(".wav")] if name.lower().endswith(".wav") else name
    if "(" in stem and stem.rstrip().endswith(")"):
        return stem.rsplit("(", 1)[1].rstrip(")").strip()
    return "unbekannt"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paare", type=int, default=10, help="Zahl der Titel (deterministische Auswahl)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verzeichnis", default=str(_ROOT / "data" / "schadens_proben"))
    ap.add_argument("--ausgang", default=None, help="JSONL-Beleg (Default: docs/reports/current/<datum>_…jsonl)")
    ap.add_argument("--fenster", type=float, default=0.0, help=">0 = nur Fenster dieser Länge (schnelle Prüfung)")
    ap.add_argument("--rezepte", default="", help="Komma-Liste; leer = alle")
    args = ap.parse_args(argv)

    from datetime import date as _date

    if not args.ausgang:
        args.ausgang = str(
            _ROOT / "docs" / "reports" / "current" / f"{_date.today().isoformat()}_wohlklang_modul_audit.jsonl"
        )
    ausgang = Path(args.ausgang)

    wavs = sorted(Path(args.verzeichnis).glob("*.wav"))
    rezepte = {r.strip() for r in args.rezepte.split(",") if r.strip()}
    if rezepte:
        wavs = [w for w in wavs if _rezept_aus_namen(w.name) in rezepte]
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(wavs), size=min(args.paare, len(wavs)), replace=False)
    auswahl = [wavs[int(i)] for i in sorted(idx)]
    print(f"MODUL-AUDIT: {len(auswahl)} Titel ausgewählt (von {len(wavs)})", flush=True)

    fertig: set[str] = set()
    if ausgang.exists():
        for line in ausgang.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                fertig.add(json.loads(line)["stamm"])
            except Exception as exc:  # noqa: BLE001 - Resume-Zeilen sind Belegdaten
                logger.debug("Modul-Audit-Resume: unlesbare Zeile übersprungen: %s", exc)
    if fertig:
        print(f"MODUL-AUDIT-RESUME: {len(fertig)} Titel bereits fertig", flush=True)

    from backend.api.bridge import get_aurik_denker_instance, normalize_user_mode, run_pre_analysis
    from backend.core.librosa_bootstrap import ensure_librosa_ready

    try:
        ensure_librosa_ready()
    except Exception:  # noqa: BLE001 - degradiert, nie den Produktionspfad blockieren
        logger.debug("librosa/numba-Bootstrap nicht verfügbar — Ersatzpfade aktiv")

    denker = get_aurik_denker_instance()
    phasen_beitrag: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    verletzungs_phasen: dict[str, int] = defaultdict(int)
    verletzungen = 0
    rollbacks = 0
    for n, pfad in enumerate(auswahl):
        stamm = pfad.name
        if stamm in fertig:
            continue
        try:
            roh = wavfile.read(str(pfad))
            sr = int(roh[0])
            d = np.asarray(roh[1], np.float32)
            if d.dtype != np.float32:
                d = d / 32768.0
            if d.ndim == 2:
                d = d.mean(axis=1)
            if args.fenster > 0:
                start = int(len(d) // 2)
                d = d[start : start + int(args.fenster * sr)]
            pre = run_pre_analysis(
                audio_native=d, sr_native=sr, audio_48k=d, file_path=str(pfad), store_in_bridge_cache=True
            )
            result = denker.denke(
                d,
                sr=sr,
                mode=normalize_user_mode("balanced"),
                no_rt_limit=True,
                input_path=f"{pfad}#modul_audit",
                output_path="",
                pre_analysis_result=pre,
            )
            meta = dict(getattr(result, "metadata", None) or {})
            phasen_deltas = dict(meta.get("phase_deltas") or {})
            wo = dict(meta.get("wohlklang_ordnung") or {})
            woe = dict(meta.get("wohlklang_ordnung_enforcement") or {})
            violiert = str(wo.get("status", "")) == "VIOLATION"
            rollback = str(woe.get("action", "")) == "ROLLBACK" and float(woe.get("blend_factor", 1.0)) == 0.0
            zeile = {
                "stamm": stamm,
                "rezept": _rezept_aus_namen(stamm),
                "material": str(getattr(result, "material", "")),
                "phase_deltas": {k: {g: float(v) for g, v in (dvals.items() if isinstance(dvals, dict) else {})} for k, dvals in phasen_deltas.items()},
                "wohlklang_ordnung": wo,
                "wohlklang_ordnung_enforcement": woe,
                "violiert": violiert,
                "voll_rollback": rollback,
            }
            with ausgang.open("a", encoding="utf-8") as f:
                f.write(json.dumps(zeile, ensure_ascii=False) + "\n")
            for pid, dvals in phasen_deltas.items():
                if isinstance(dvals, dict):
                    for goal, val in dvals.items():
                        try:
                            phasen_beitrag[str(pid)][str(goal)].append(float(val))
                        except (TypeError, ValueError):
                            pass
            if violiert:
                verletzungen += 1
                for pid in phasen_deltas:
                    verletzungs_phasen[str(pid)] += 1
            if rollback:
                rollbacks += 1
            print(
                f"[{n + 1}/{len(auswahl)}] {stamm} — {len(phasen_deltas)} Phasen mit Deltas | "
                f"VIOLATION={violiert} ROLLBACK={rollback}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - Titel überspringen statt Audit abbrechen
            logger.warning("Modul-Audit: Titel %s fehlgeschlagen (%s) — übersprungen", stamm, exc)

    print("MODUL-AUDIT AGGREGAT (Mittel |Δ| je Phase über die wichtigsten Ziele):", flush=True)
    for pid in sorted(phasen_beitrag):
        row = phasen_beitrag[pid]
        top = sorted(row.items(), key=lambda kv: -np.mean(np.abs(kv[1])))[:4]
        if top:
            print(
                f"  {pid}: " + ", ".join(f"{g}={np.mean(np.abs(v)):+.4f}" for g, v in top)
                + f" | in Verletzungs-Läufen: {verletzungs_phasen.get(pid, 0)}",
                flush=True,
            )
    print(
        f"MODUL_AUDIT_DONE: {verletzungen} Verletzungen, {rollbacks} Voll-Rollbacks "
        f"auf {len(auswahl) - len(fertig)} gelaufenen Titeln",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
