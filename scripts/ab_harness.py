#!/usr/bin/env python3
"""A/B-Hör-Harness — Verblindung + Statistik für Hör-Entscheidungen (Pfeiler 7).

Normative Einordnung: Die **Hör-Instanz entscheidet** — Metriken sind Zeugen
(`.github/instructions/hoerordnung.instructions.md`). Dieses Werkzeug liefert
ausschließlich den verblindeten Versuchsaufbau und die Auswertungsstatistik.
Es erfindet **keine** Hör-Urteile (Beleg-Prinzip des Wohlklang-Vertrags,
`.github/WOHLKLANG_CLAIMS.md`).

Drei Subkommandos:

* ``prepare`` — erzeugt aus Versuchspunkten (je Item zwei Kandidaten-Audiodateien,
  optional Referenz) ein verblindetes Versuchsblatt: Präsentationsreihenfolge
  seed-deterministisch (§G5 (copilot-instructions.md)), Stimuli byte-identisch
  unter neutralen Namen kopiert, Schlüssel getrennt vom Blatt.
* ``collect`` — liest Verdikte ein (Urteil der Hör-Instanz) und berechnet
  exakte Binomial-Statistik: zweiseitiger Vorzeichen-Test (exakt) und
  Wilson-95 %-Konfidenzintervall der Siegrate. Gleichstände fliegen aus der
  Binomialmenge (dokumentiert, nicht still).
* ``report`` — verdichtet Blatt + Verdikte + Statistik zu einem JSON-Bericht;
  **jede Zeile trägt ihre Quelle**.

Aufruf: ``python3 -B scripts/ab_harness.py {prepare,collect,report} ...``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
from pathlib import Path
from typing import Any

LABELS = ("A", "B")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


# ── prepare ─────────────────────────────────────────────────────────────────


def prepare(
    items: list[dict[str, str]],
    out_dir: Path,
    *,
    seed: int,
    label_a: str = "Kandidat A",
    label_b: str = "Kandidat B",
    repeats: int = 1,
) -> dict[str, Any]:
    """Erzeugt Versuchsblatt + Schlüssel. Items: [{"id", "a", "b", "ref"?}]."""
    if repeats < 1:
        raise ValueError("repeats muss >= 1 sein")
    rng = random.Random(seed)
    stimuli_dir = out_dir / "stimuli"
    stimuli_dir.mkdir(parents=True, exist_ok=True)

    trials: list[dict[str, Any]] = []
    key: dict[str, Any] = {}
    sources: dict[str, Any] = {}
    for item in items:
        item_id = str(item["id"])
        for role in ("a", "b"):
            src = Path(item[role])
            if not src.is_file():
                raise FileNotFoundError(f"Stimulus fehlt (Item {item_id}, Rolle {role}): {src}")
            sources[str(src)] = _sha256(src)
        if item.get("ref"):
            ref = Path(item["ref"])
            if not ref.is_file():
                raise FileNotFoundError(f"Referenz fehlt (Item {item_id}): {ref}")
            sources[str(ref)] = _sha256(ref)

        for rep in range(repeats):
            trial_id = f"{item_id}__r{rep}"
            left_is_a = rng.random() < 0.5
            mapping = {"left": "a" if left_is_a else "b", "right": "b" if left_is_a else "a"}
            # Stimuli byte-identisch unter neutralen Namen (echte Verblindung).
            slots: dict[str, str] = {}
            for slot in ("left", "right"):
                src = Path(item[mapping[slot]])
                dst = stimuli_dir / f"{trial_id}_{slot}{src.suffix}"
                shutil.copyfile(src, dst)
                slots[slot] = str(dst.relative_to(out_dir))
            trials.append(
                {
                    "trial_id": trial_id,
                    "item_id": item_id,
                    "repeat": rep,
                    "stimuli": slots,
                    "quelle": f"prepare(seed={seed}) · {item_id}",
                }
            )
            key[trial_id] = {
                "left": mapping["left"].upper(),
                "right": mapping["right"].upper(),
                "quelle": f"Schlüssel · {trial_id}",
            }

    sheet = {
        "schema": "aurik.ab_trials/1",
        "seed": seed,
        "label_a": label_a,
        "label_b": label_b,
        "repeats": repeats,
        "trials": trials,
        "quelle": "scripts/ab_harness.py prepare",
    }
    key_doc = {
        "schema": "aurik.ab_key/1",
        "seed": seed,
        "mapping": key,
        "quelle": "scripts/ab_harness.py prepare (VERTRAULICH — nicht an Hör-Instanz)",
    }
    _write_json(out_dir / "trials.json", sheet)
    _write_json(out_dir / "key.json", key_doc)
    _write_json(
        out_dir / "sources.json",
        {"schema": "aurik.ab_sources/1", "sha256": sources, "quelle": "scripts/ab_harness.py prepare"},
    )
    return sheet


# ── collect / Statistik ─────────────────────────────────────────────────────


def sign_test_two_sided(wins: int, n: int) -> float:
    """Exakter zweiseitiger Vorzeichen-Test gegen p = 0.5 (H0: kein Vorteil)."""
    if n <= 0:
        return 1.0
    wins = max(0, min(wins, n))
    tail = sum(math.comb(n, k) for k in range(0, min(wins, n - wins) + 1))
    return min(1.0, 2.0 * tail / float(2**n))


def wilson_ci(wins: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson-Score-95 %-Intervall der Siegrate (ohne SciPy, deterministisch)."""
    if n <= 0:
        return (0.0, 1.0)
    p = wins / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def collect(
    verdicts: list[dict[str, str]],
    key: dict[str, Any],
    *,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Verdikte der Hör-Instanz → Statistik. Verdikt: {"trial_id", "preferred": "left|right|tie"}."""
    mapping: dict[str, Any] = key["mapping"]
    wins_a = wins_b = ties = 0
    counted: list[dict[str, Any]] = []
    for v in verdicts:
        tid = str(v["trial_id"])
        if tid not in mapping:
            raise KeyError(f"Unbekannte trial_id in Verdikten: {tid}")
        pref = str(v["preferred"]).lower()
        if pref not in ("left", "right", "tie"):
            raise ValueError(f"preferred muss left|right|tie sein: {v!r}")
        if pref == "tie":
            ties += 1
            counted.append({"trial_id": tid, "preferred": "tie", "quelle": f"Verdikt · {tid}"})
            continue
        side_label = str(mapping[tid][pref])  # "A" | "B"
        if side_label == "A":
            wins_a += 1
        else:
            wins_b += 1
        counted.append({"trial_id": tid, "preferred": pref, "gewinner": side_label, "quelle": f"Verdikt · {tid}"})

    n_decisive = wins_a + wins_b
    p_value = sign_test_two_sided(wins_a, n_decisive)
    ci_low, ci_high = wilson_ci(wins_a, n_decisive)
    signifikant = bool(n_decisive > 0 and p_value < alpha)
    return {
        "schema": "aurik.ab_stats/1",
        "n_verdikte": len(verdicts),
        "n_entscheidend": n_decisive,
        "gleichstaende": ties,
        "siege_a": wins_a,
        "siege_b": wins_b,
        "siegquote_a": round(wins_a / n_decisive, 4) if n_decisive else None,
        "p_wert_vorzeichen_test": round(p_value, 6),
        "ci95_wilson_a": [round(ci_low, 4), round(ci_high, 4)],
        "alpha": alpha,
        "signifikant": signifikant,
        "empfehlung": (
            f"{key.get('label_a', 'A')} vorzuziehen (signifikant, p={p_value:.4f})"
            if signifikant and wins_a > wins_b
            else f"{key.get('label_b', 'B')} vorzuziehen (signifikant, p={p_value:.4f})"
            if signifikant and wins_b > wins_a
            else "kein signifikanter Vorteil — Hör-Instanz entscheidet ohne Statistik-Vorgabe"
        ),
        "hinweis": ("Hör-Instanz entscheidet (hoerordnung.instructions.md); Statistik ist Zeuge, kein Urteil."),
        "verdikte": counted,
        "quelle": "scripts/ab_harness.py collect",
    }


# ── report ──────────────────────────────────────────────────────────────────


def report(sheet: dict[str, Any], stats: dict[str, Any], out_path: Path) -> dict[str, Any]:
    """JSON-Bericht: Blatt + Statistik, jede Zeile mit Quelle."""
    payload = {
        "schema": "aurik.ab_report/1",
        "label_a": sheet.get("label_a"),
        "label_b": sheet.get("label_b"),
        "seed": sheet.get("seed"),
        "n_trials": len(sheet.get("trials", [])),
        "statistik": stats,
        "zeilen": [
            {
                "inhalt": f"Versuchsblatt: {len(sheet.get('trials', []))} Trials, Seed {sheet.get('seed')}",
                "quelle": "trials.json",
            },
            {
                "inhalt": (
                    f"Siege A/B: {stats.get('siege_a')}/{stats.get('siege_b')} "
                    f"bei {stats.get('n_entscheidend')} entscheidenden Verdikten"
                ),
                "quelle": "collect",
            },
            {"inhalt": f"p-Wert (Vorzeichen-Test, exakt): {stats.get('p_wert_vorzeichen_test')}", "quelle": "collect"},
            {"inhalt": f"Empfehlung: {stats.get('empfehlung')}", "quelle": "collect"},
        ],
        "quelle": "scripts/ab_harness.py report",
    }
    _write_json(out_path, payload)
    return payload


# ── CLI ─────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="A/B-Hör-Harness (Verblindung + Statistik, Pfeiler 7)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_prep = sub.add_parser("prepare", help="Verblindetes Versuchsblatt erzeugen")
    p_prep.add_argument("--items", required=True, help="JSON-Datei: [{id, a, b, ref?}]")
    p_prep.add_argument("--out-dir", required=True)
    p_prep.add_argument("--seed", type=int, required=True)
    p_prep.add_argument("--label-a", default="Kandidat A")
    p_prep.add_argument("--label-b", default="Kandidat B")
    p_prep.add_argument("--repeats", type=int, default=1)

    p_coll = sub.add_parser("collect", help="Verdikte auswerten")
    p_coll.add_argument("--key", required=True, help="key.json aus prepare")
    p_coll.add_argument("--verdicts", required=True, help="JSON: [{trial_id, preferred}]")
    p_coll.add_argument("--out", required=True)
    p_coll.add_argument("--alpha", type=float, default=0.05)

    p_rep = sub.add_parser("report", help="Hör-Bericht schreiben")
    p_rep.add_argument("--trials", required=True, help="trials.json aus prepare")
    p_rep.add_argument("--stats", required=True, help="Statistik-JSON aus collect")
    p_rep.add_argument("--out", required=True)

    args = parser.parse_args(argv)
    if args.cmd == "prepare":
        items = json.loads(Path(args.items).read_text(encoding="utf-8"))
        sheet = prepare(
            items, Path(args.out_dir), seed=args.seed, label_a=args.label_a, label_b=args.label_b, repeats=args.repeats
        )
        print(f"Versuchsblatt: {len(sheet['trials'])} Trials → {args.out_dir}/trials.json")
    elif args.cmd == "collect":
        key = json.loads(Path(args.key).read_text(encoding="utf-8"))
        verdicts = json.loads(Path(args.verdicts).read_text(encoding="utf-8"))
        stats = collect(verdicts, key, alpha=args.alpha)
        _write_json(Path(args.out), stats)
        print(
            f"Statistik: Siege A/B {stats['siege_a']}/{stats['siege_b']}, "
            f"p={stats['p_wert_vorzeichen_test']} → {args.out}"
        )
    else:
        sheet = json.loads(Path(args.trials).read_text(encoding="utf-8"))
        stats = json.loads(Path(args.stats).read_text(encoding="utf-8"))
        report(sheet, stats, Path(args.out))
        print(f"Hör-Bericht → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
