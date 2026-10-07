#!/usr/bin/env python3
"""mushra_harness — Doppelblinde MUSHRA-Studie (ITU-R BS.1534) für Aurik.

Vier Schritte (CLI):
  1) prepare  — deterministische Restauration der Corpus-Paare (nur fehlende,
                hash-basiert inkrementell) → output_audio/mushra/<run>/
  2) sessions — baut N doppelblinde Hörer-Sessions: je Trial [clean (hidden
                ref), anchor (3.5-kHz-Lowpass), Aurik, ggf. kommerzielle
                Referenzen aus corpus/references/] in seed-deterministisch
                zufälliger Reihenfolge; exportiert JSON (Player kann beliebig
                gebaut werden) + Antwort-CSV-Vorlage.
  3) analyze  — wertet Antworten aus: MUSHRA-Scores (0-100), Mittelwert +
                95-%-CI je Stimulus/Trial, Hörer-Validität (Hidden-Ref < 60 →
                Ausschluss), GO/NO-GO gegen konfigurierbare Kriterien.

Hörordnung: Metriken sind Zeugen, die Hör-Instanz entscheidet — dieser
Harness ist die operative Hör-Instanz (menschlich, doppelblind).

Nutzung:
  python scripts/mushra_harness.py prepare  --modes balanced maximum --seconds 12
  python scripts/mushra_harness.py sessions --listeners 5 --out study_run1
  python scripts/mushra_harness.py analyze  --answers study_run1/answers.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT_ROOT = ROOT / "output_audio" / "mushra"
ANCHOR_LP_HZ = 3500.0  # ITU: Anchor = 3.5-kHz-Bandbreite


# --------------------------------------------------------------------------
# Paar-Erkennung (damaged → clean)
# --------------------------------------------------------------------------
def find_pairs(corpus: Path, explicit: list[str] | None = None) -> list[tuple[Path, Path, str]]:
    """Liefert (damaged, clean, label)-Triple.

    Explizite Angabe: "damaged.wav,clean.wav,label".  Sonst Heuristik:
    <base>_crackle[_chain…].wav bzw. <base>_noise/_degraded → <base>_clean.wav.
    """
    pairs: list[tuple[Path, Path, str]] = []
    for spec in explicit or []:
        parts = [p.strip() for p in spec.split(",")]
        if len(parts) >= 2:
            dmg = Path(parts[0])
            cln = Path(parts[1])
            if not dmg.is_absolute():
                dmg = ROOT / dmg
            if not cln.is_absolute():
                cln = ROOT / cln
            pairs.append((dmg, cln, parts[2] if len(parts) > 2 else dmg.stem))
    if pairs:
        return pairs
    for mat_dir in sorted(corpus.iterdir()) if corpus.exists() else []:
        dmg_dir = mat_dir / "damaged"
        cln_dir = mat_dir / "clean"
        if not dmg_dir.is_dir() or not cln_dir.is_dir():
            continue
        clean_by_base = {p.stem: p for p in cln_dir.glob("*.wav") if p.stem.endswith("_clean")}
        for d in sorted(dmg_dir.glob("*.wav")):
            m = re.match(r"^(.*?)(?:_crackle|_noise|_degraded|_chain)", d.stem)
            base = m.group(1) if m else d.stem
            cand = clean_by_base.get(base + "_clean")
            if cand is None:
                cand = cln_dir / f"{base}_clean.wav"
            if cand.exists():
                pairs.append((d, cand, base))
    return pairs


# --------------------------------------------------------------------------
# Deterministische Restauration (prepare)
# --------------------------------------------------------------------------
def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare_pair(
    dmg: Path, clean: Path, label: str, out_dir: Path, mode: str, seconds: float, sr_out: int = 48000
) -> dict:
    import soundfile as sf

    from backend.core.unified_restorer_v3 import QualityMode, RestorationConfig, UnifiedRestorerV3

    audio, sr = sf.read(str(dmg), dtype="float32", always_2d=False)
    n = int(seconds * sr)
    if len(audio) > n:
        audio = audio[:n]
    cfg = RestorationConfig(mode=QualityMode(mode))
    engine = UnifiedRestorerV3(cfg)
    t0 = time.perf_counter()
    result = engine.restore(audio, sample_rate=sr)
    wall = time.perf_counter() - t0
    restored = np.asarray(result.audio, dtype=np.float32)
    if restored.ndim == 2 and restored.shape[0] == 2 and restored.shape[1] != 2:
        restored = restored.T
    out_path = out_dir / f"{label}__{mode}.wav"
    sf.write(str(out_path), restored, sr_out, format="WAV", subtype="PCM_24")
    return {
        "label": label,
        "mode": mode,
        "damaged_sha": sha256_file(dmg)[:16],
        "clean_sha": sha256_file(clean)[:16],
        "wall_s": round(wall, 1),
        "quality": float(getattr(result, "quality_estimate", 0.0) or 0.0),
        "audibility_gate": (result.metadata or {}).get("audibility_gate"),
        "out": str(out_path),
    }


def anchor_lowpass(audio: np.ndarray, sr: int) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt

    sos = butter(4, ANCHOR_LP_HZ / (sr / 2.0), btype="lowpass", output="sos")
    _filtered: np.ndarray = np.asarray(sosfiltfilt(sos, audio), dtype=np.float32)
    return _filtered


def cmd_prepare(args: argparse.Namespace) -> int:
    pairs = find_pairs(ROOT / args.corpus, args.pair)
    if not pairs:
        print("Keine Paare gefunden. --corpus prüfen oder --pair explizit angeben.")
        return 2
    run_dir = OUT_ROOT / time.strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    meta = {"pairs": [], "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "seconds": args.seconds, "modes": args.modes}
    for dmg, clean, label in pairs[: args.max_pairs]:
        for mode in args.modes:
            row = prepare_pair(dmg, clean, label, run_dir, mode, args.seconds)
            meta["pairs"].append(row)
            print(f"  {label} [{mode}]: {row['wall_s']}s quality={row['quality']}")
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Run-Verzeichnis:", run_dir)
    return 0


# --------------------------------------------------------------------------
# Sessions (doppelblind, seed-deterministisch)
# --------------------------------------------------------------------------
def build_sessions(pairs_meta: dict, n_listeners: int, seed: int, ref_root: Path) -> list[dict]:
    rng = np.random.default_rng(seed)
    # Kommerzielle Referenzen: corpus/references/<tool>/<label__mode>.wav?
    refs: dict[str, list[Path]] = {}
    if ref_root.is_dir():
        for tool_dir in sorted(ref_root.iterdir()):
            if tool_dir.is_dir():
                for wav in sorted(tool_dir.glob("*.wav")):
                    refs.setdefault(wav.stem, []).append(wav)
    sessions = []
    for li in range(n_listeners):
        trials = []
        for pr in pairs_meta["pairs"]:
            stim = [
                {"kind": "aurik", "path": pr["out"]},
            ]
            for rpath in refs.get(pr["label"], []):
                stim.append({"kind": "reference", "path": str(rpath), "tool": rpath.parent.name})
            # Hidden Reference + Anchor werden zur Laufzeit aus clean bzw. Rest erzeugt;
            # hier nur Markierung, Player rendert sie deterministisch.
            stim.append({"kind": "hidden_ref"})
            stim.append({"kind": "anchor"})
            order = rng.permutation(len(stim)).tolist()
            trials.append(
                {
                    "label": pr["label"],
                    "mode": pr["mode"],
                    "clean_sha": pr["clean_sha"],
                    "order": [stim[i]["kind"] for i in order],
                    "n_stimuli": len(stim),
                }
            )
        sessions.append({"listener": f"L{li + 1:02d}", "seed": int(rng.integers(0, 2**31)), "trials": trials})
    return sessions


def cmd_sessions(args: argparse.Namespace) -> int:
    runs = sorted(OUT_ROOT.glob("run_*"))
    if not runs:
        print("Kein Run gefunden — zuerst: prepare")
        return 2
    run_dir = runs[-1]
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    out = OUT_ROOT / (args.out or f"study_{time.strftime('%Y%m%d_%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)
    sessions = build_sessions(meta, args.listeners, args.seed, ROOT / args.refs)
    (out / "sessions.json").write_text(json.dumps(sessions, indent=2), encoding="utf-8")
    # Antwort-CSV-Vorlage: listener,trial,stimulus_index,rating
    with open(out / "answers_template.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["listener", "trial_label", "stimulus_index", "kind", "rating"])
        for s in sessions:
            for t in s["trials"]:
                for i, kind in enumerate(t["order"]):
                    w.writerow([s["listener"], t["label"], i, kind, ""])
    (out / "meta.json").write_text(
        json.dumps({"seed": args.seed, "run": run_dir.name, "listeners": args.listeners}, indent=2), encoding="utf-8"
    )
    print("Studien-Ordner:", out)
    return 0


# --------------------------------------------------------------------------
# Analyse (GO/NO-GO nach Hörordnung-Kriterien)
# --------------------------------------------------------------------------
@dataclass
class StimulusScore:
    kind: str
    mean: float
    ci95: float
    n: int


def analyze_answers(answers_csv: Path, out_dir: Path, args: argparse.Namespace) -> int:
    rows = list(csv.DictReader(open(answers_csv, encoding="utf-8")))
    if not rows:
        print("Leere Antwortdatei.")
        return 2
    # Hörer-Validität: Hidden-Ref muss im Mittel >= 60 liegen, sonst Ausschluss
    by_listener: dict[str, list[float]] = {}
    for r in rows:
        if r.get("kind") == "hidden_ref" and r.get("rating", "").strip():
            by_listener.setdefault(r["listener"], []).append(float(r["rating"]))
    valid = [li for li, vals in by_listener.items() if np.mean(vals) >= 60.0]
    kept = [r for r in rows if r["listener"] in valid]
    scores: dict[tuple[str, str], list[float]] = {}
    for r in kept:
        if r.get("rating", "").strip():
            scores.setdefault((r["trial_label"], r["kind"]), []).append(float(r["rating"]))
    summary: dict = {}
    for (label, kind), vals in sorted(scores.items()):
        m = float(np.mean(vals))
        ci = float(1.96 * np.std(vals) / np.sqrt(max(len(vals), 1)))
        summary.setdefault(label, {})[kind] = {"mean": round(m, 1), "ci95": round(ci, 1), "n": len(vals)}
    # GO/NO-GO: Aurik >= hidden_ref - gap  UND Aurik >= anchor + margin
    go: bool = True
    reasons: list[str] = []
    gap = float(args.gap)
    margin = float(args.margin)
    for label, kinds in summary.items():
        aur = kinds.get("aurik", {}).get("mean")
        hr = kinds.get("hidden_ref", {}).get("mean")
        an = kinds.get("anchor", {}).get("mean")
        if aur is None:
            go, reasons = False, [*reasons, f"{label}: Aurik fehlt"]
            continue
        if hr is not None and aur < hr - gap:
            go, reasons = False, [*reasons, f"{label}: Aurik {aur:.0f} < HiddenRef {hr:.0f} - {gap}"]
        if an is not None and aur < an + margin:
            go, reasons = False, [*reasons, f"{label}: Aurik {aur:.0f} < Anchor {an:.0f} + {margin}"]
    result = {
        "listeners_total": len(by_listener),
        "listeners_valid": len(valid),
        "excluded": sorted(set(by_listener) - set(valid)),
        "scores": summary,
        "go": go,
        "reasons": reasons,
        "criteria": {"hidden_ref_gap": gap, "anchor_margin": margin, "validity_min": 60},
    }
    out_file = out_dir / f"analysis_{answers_csv.stem}.json"
    out_file.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("GO" if go else "NO-GO", "|", "; ".join(reasons) or "Kriterien erfüllt")
    print("Bericht:", out_file)
    return 0 if go else 1


def cmd_analyze(args: argparse.Namespace) -> int:
    answers = Path(args.answers)
    if not answers.is_absolute():
        answers = OUT_ROOT / answers
    out_dir = answers.parent
    return analyze_answers(answers, out_dir, args)


# --------------------------------------------------------------------------
# thresholds-build / thresholds-fit — Wohlklang-2AFC (die menschliche Hör-Instanz)
# --------------------------------------------------------------------------
# Wozu: Die Wohlklang-Vertrags-Belegpflicht „blindes A/B für einen Menschen"
# (Beleg 4, .github/WOHLKLANG_CLAIMS.md) braucht eine baubare, doppelblinde
# Hörprobe. Der Player (scripts/hoerpanel_player.py) konnte sie seit 2026-10-03
# abspielen — der Erzeuger (dieser Subcommand) fehlte. Ohne ihn ist Beleg 4
# nicht führbar; die Lücke ist in TASK_CHANGES.md als offen dokumentiert.
#
# Vertrag (vom Player eingelesen, exakt):
#   <out>/meta.json          {build, seed, task, question, n_trials_per_listener, …}
#   <out>/trial_key.json     [{trial_id, defect_class, defective_interval, …}]
#   <out>/trials/<id>__A.wav, __B.wav
#   <out>/answers_template.csv   (listener,trial_id,answer_interval)
#
# Doppelblind-Prinzip: Das Manifest benennt NUR semantische Rollen
# (``reference`` = Bezug, ``candidate`` = Prüfling). Welche Rolle in A und
# welche in B landet, entscheidet allein der Seed — deterministisch aus
# sha256("seed:trial_id"), damit die Zuordnung nicht von Datei-/Dict-Reihenfolge
# abhängt (§G5 (copilot-instructions.md)).
#
# Aufgabenarten:
#   defect     — „In welchem Intervall ist der Artefakt-Störenfried hörbar?"
#                (Klassen-Schwelle; Standard, abwärtskompatibel)
#   preference — „Welches Intervall klingt besser/authentischer?"
#                (Wohlklang-Vertrag, Beleg 4)
TASK_DEFECT = "defect"
TASK_PREFERENCE = "preference"
_TASK_CHOICES = (TASK_DEFECT, TASK_PREFERENCE)

CSV_HEADER_THRESHOLDS = ["listener", "trial_id", "answer_interval"]


def _rms_db(x: np.ndarray) -> float:
    r = float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))
    return float(20.0 * np.log10(max(r, 1e-12)))


def _peak_db(x: np.ndarray) -> float:
    p = float(np.max(np.abs(np.asarray(x, dtype=np.float64)))) if x.size else 0.0
    return float(20.0 * np.log10(max(p, 1e-12)))


def _ab_side(seed: int, trial_id: str) -> bool:
    """Deterministische A/B-Zuordnung (§G5 (copilot-instructions.md)).

    True  → der **Kandidat** liegt in A, die Referenz in B.
    False → umgekehrt.

    Quelle ist ausdrücklich sha256(seed:trial_id) und NICHT ``random``: das
    Ergebnis ist unabhängig von Prozesszustand, Dict-Reihenfolge und Aufrufzähler.
    """
    h = hashlib.sha256(f"{seed}:{trial_id}".encode()).digest()
    return bool(h[0] & 1)


def _slice_excerpt(audio: np.ndarray, sr: int, start_s: float, seconds: float | None) -> np.ndarray:
    n0 = max(0, int(round(start_s * sr)))
    arr = audio[n0:]
    if seconds is not None and seconds > 0:
        arr = arr[: int(round(seconds * sr))]
    return np.asarray(arr, dtype=np.float32)


def _load_trial_audio(path: Path) -> tuple[np.ndarray, int, int]:
    """Lädt eine Datei; Rückgabe (float32-Array, sr, ch)."""
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return np.asarray(data, dtype=np.float32), int(sr), int(data.shape[1])


def cmd_thresholds_build(args: argparse.Namespace) -> int:
    """Baut eine doppelblinde 2AFC-Studie aus einem Manifest (semantische Rollen)."""
    import soundfile as sf

    man_path = Path(args.manifest)
    if not man_path.is_absolute():
        man_path = ROOT / man_path
    if not man_path.is_file():
        print(f"Manifest fehlt: {man_path}")
        return 2

    manifest = json.loads(man_path.read_text(encoding="utf-8"))
    task = str(manifest.get("task", TASK_DEFECT))
    if task not in _TASK_CHOICES:
        print(f"Ungültige task '{task}' — erlaubt: {_TASK_CHOICES}")
        return 2
    seed = int(args.seed if args.seed is not None else manifest.get("seed", 20261003))
    trials = list(manifest.get("trials") or [])
    if not trials:
        print("Manifest enthält keine trials.")
        return 2

    out_dir = Path(args.out) if args.out else (OUT_ROOT / f"thresholds_{args.build}")
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    trials_dir = out_dir / "trials"
    trials_dir.mkdir(parents=True, exist_ok=True)

    key: list[dict] = []
    errors: list[str] = []

    for i, tr in enumerate(trials, 1):
        tid = str(tr.get("trial_id") or f"t{i:02d}")
        cls = str(tr.get("class") or "unbenannt")
        is_catch = bool(tr.get("catch", False))

        if is_catch:
            # ITU-R BS.1534-Anker (3,5-kHz-Tiefpass) als absichtlich HÖRBARE
            # Differenz — validiert, ob der Hörer überhaupt hinhört.
            ref_path = Path(str(tr.get("reference", "")))
            if not ref_path.is_absolute():
                ref_path = ROOT / ref_path
            if not ref_path.is_file():
                errors.append(f"{tid}: Referenz fehlt {ref_path}")
                continue
            ref, sr, ch = _load_trial_audio(ref_path)
            # Anker je Kanal — die Referenz bleibt, nur der Prüfling wird gefiltert.
            cand = np.stack([anchor_lowpass(ref[:, c], sr) for c in range(ch)], axis=1).astype(np.float32)
        else:
            ref_path = Path(str(tr.get("reference", "")))
            cand_path = Path(str(tr.get("candidate", "")))
            ref_path = ref_path if ref_path.is_absolute() else ROOT / ref_path
            cand_path = cand_path if cand_path.is_absolute() else ROOT / cand_path
            if not ref_path.is_file() or not cand_path.is_file():
                errors.append(f"{tid}: Datei fehlt (reference={ref_path.is_file()} candidate={cand_path.is_file()})")
                continue
            ref, sr, ch = _load_trial_audio(ref_path)
            cand, sr_c, ch_c = _load_trial_audio(cand_path)
            if sr_c != sr:
                errors.append(f"{tid}: Samplerate unterschiedlich ({sr} vs {sr_c})")
                continue
            if ch_c != ch:
                errors.append(f"{tid}: Kanalzahl unterschiedlich ({ch} vs {ch_c})")
                continue

        ref = _slice_excerpt(ref, sr, float(tr.get("start_s", 0.0)), tr.get("seconds"))
        cand = _slice_excerpt(cand, sr, float(tr.get("start_s", 0.0)), tr.get("seconds"))

        if ref.shape != cand.shape:
            errors.append(f"{tid}: Ausschnitt-Länge unterschiedlich {ref.shape} vs {cand.shape}")
            continue
        if ref.size == 0:
            errors.append(f"{tid}: leerer Ausschnitt (start_s/seconds prüfen)")
            continue

        level_note: dict[str, float] = {"rms_db_ref": round(_rms_db(ref), 3), "rms_db_cand": round(_rms_db(cand), 3)}

        if args.level_match:
            # Nur wenn ausdrücklich verlangt: entfernt die Lautheit als Hinweisreiz
            # (für Präferenz-Tests unerlässlich, für Defekt-Schwellen verfälschend).
            target = 0.5 * (_rms_db(ref) + _rms_db(cand))
            for name, arr in (("ref", ref), ("cand", cand)):
                g = 10.0 ** ((target - _rms_db(arr)) / 20.0)
                scaled = arr * g
                pk = float(np.max(np.abs(scaled))) if scaled.size else 0.0
                if pk > 0.999:
                    scaled = scaled * (0.999 / pk)
                if name == "ref":
                    ref = scaled
                else:
                    cand = scaled

        level_note["rms_db_ref_after"] = round(_rms_db(ref), 3)
        level_note["rms_db_cand_after"] = round(_rms_db(cand), 3)
        level_note["peak_db_a"] = round(_peak_db(ref if _ab_side(seed, tid) else cand), 3)

        cand_in_a = _ab_side(seed, tid)
        a_audio = cand if cand_in_a else ref
        b_audio = ref if cand_in_a else cand

        sf.write(str(trials_dir / f"{tid}__A.wav"), a_audio, sr, format="WAV", subtype="PCM_24")
        sf.write(str(trials_dir / f"{tid}__B.wav"), b_audio, sr, format="WAV", subtype="PCM_24")

        # Antwort-Semantik ehrlich trennen:
        #   ``expected_answer``   — die richtige Wahl (autoritativ für thresholds-fit).
        #   ``defective_interval``— NUR gesetzt, wenn tatsächlich ein Defekt/Artefakt
        #                           existiert (Defekt-Aufgabe) oder eine Fangfrage
        #                           absichtlich hörbar degradiert wurde.
        # Bei einer Präferenz-Aufgabe ohne Defekt ist das Feld ``None`` — eine
        # erfundene „Defekt-Seite" wäre eine Falschaussage im Schlüssel.
        cand_side = "A" if cand_in_a else "B"
        ref_side = "B" if cand_in_a else "A"
        if is_catch:
            # Der Anker (3,5-kHz-Tiefpass) liegt im Kandidaten-Intervall.
            anchor_in = cand_side
            expected = ref_side if task == TASK_PREFERENCE else anchor_in
            defective_interval: str | None = anchor_in
        elif task == TASK_PREFERENCE:
            expected = cand_side
            defective_interval = None
        else:
            expected = cand_side if bool(tr.get("expected_in_candidate", True)) else ref_side
            defective_interval = expected

        key.append(
            {
                "trial_id": tid,
                "defect_class": cls,
                "task": task,
                "expected_answer": expected,
                "defective_interval": defective_interval,
                "candidate_in": cand_side,
                "reference_in": ref_side,
                "is_catch": is_catch,
                "expected_in_candidate": bool(tr.get("expected_in_candidate", True)),
                "sr": sr,
                "samples": int(ref.shape[0]),
                "seconds": round(ref.shape[0] / sr, 3),
                "start_s": float(tr.get("start_s", 0.0)),
                "level_match": bool(args.level_match),
                **level_note,
            }
        )

    if not key:
        print("Keine gültigen Trials — Abbruch (fail-closed).")
        for e in errors:
            print(f"  ! {e}")
        return 2

    meta = {
        "build": args.build,
        "seed": seed,
        "task": task,
        "question": str(manifest.get("question") or ""),
        "classes": sorted({k["defect_class"] for k in key}),
        "n_trials_per_listener": len(key),
        "n_catch": sum(1 for k in key if k["is_catch"]),
        "level_match": bool(args.level_match),
        "manifest": str(man_path.relative_to(ROOT)) if str(man_path).startswith(str(ROOT)) else str(man_path),
        "created": "mushra_harness.thresholds-build",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "trial_key.json").write_text(json.dumps(key, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Antwort-Vorlage (der Player schreibt answers.csv selbst; die Vorlage dient
    # Papier-/Excel-Hörern und dokumentiert das erwartete Format).
    tmpl = out_dir / "answers_template.csv"
    with open(tmpl, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(CSV_HEADER_THRESHOLDS)
        for k in key:
            w.writerow(["L01", k["trial_id"], ""])

    print(f"Studie: {out_dir}")
    print(f"  build={args.build}  task={task}  seed={seed}  trials={len(key)}  catches={meta['n_catch']}")
    print(f"  Hörer starten:  python scripts/hoerpanel_player.py --study {out_dir} --port 8765")
    if errors:
        print(f"  Übersprungen (fail-closed): {len(errors)}")
        for e in errors:
            print(f"    ! {e}")
    return 0


def cmd_thresholds_fit(args: argparse.Namespace) -> int:
    """Wertet answers.csv gegen trial_key.json aus (Trefferquote + Wilson-CI)."""
    study_dir = Path(args.study)
    if not study_dir.is_absolute():
        study_dir = ROOT / study_dir
    key_path = study_dir / "trial_key.json"
    ans_path = study_dir / "answers.csv"
    if not key_path.is_file():
        print(f"trial_key.json fehlt: {key_path}")
        return 2
    if not ans_path.is_file():
        print(f"answers.csv fehlt: {ans_path} (noch nicht gehört?)")
        return 2

    key = {k["trial_id"]: k for k in json.loads(key_path.read_text(encoding="utf-8"))}
    rows: list[dict] = []
    with open(ans_path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("trial_id") in key and str(row.get("answer_interval", "")).upper() in ("A", "B"):
                rows.append(row)

    if not rows:
        print("Keine verwertbaren Antworten.")
        return 2

    def _wilson(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
        if n == 0:
            return (0.0, 1.0)
        p = hits / n
        d = 1.0 + z * z / n
        c = p + z * z / (2 * n)
        s = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
        return (max(0.0, (c - s) / d), min(1.0, (c + s) / d))

    def _expected(entry: dict) -> str | None:
        """Richtige Wahl: ``expected_answer`` ist autoritativ.

        ``defective_interval`` bleibt als Fallback für ältere Schlüssel lesbar,
        ist aber bei Präferenz-Paaren ohne Defekt bewusst ``None``.
        """
        ans = entry.get("expected_answer") or entry.get("defective_interval")
        return str(ans) if ans in ("A", "B") else None

    def _is_hit(row: dict) -> bool:
        want = _expected(key[row["trial_id"]])
        return want is not None and str(row["answer_interval"]).upper() == want

    listeners = sorted({r["listener"] for r in rows})
    report: dict = {"study": str(study_dir.name), "listeners": {}, "by_class": {}}

    for ln in listeners:
        sub = [r for r in rows if r["listener"] == ln]
        catch = [r for r in sub if key[r["trial_id"]].get("is_catch")]
        real = [r for r in sub if not key[r["trial_id"]].get("is_catch")]
        catch_hits = sum(1 for r in catch if _is_hit(r))
        hits = sum(1 for r in real if _is_hit(r))
        lo, hi = _wilson(hits, len(real))
        catch_ok = (catch_hits == len(catch)) if catch else True
        report["listeners"][ln] = {
            "trials": len(real),
            "hits": hits,
            "hit_rate": round(hits / len(real), 4) if real else None,
            "hit_rate_ci95": [round(lo, 4), round(hi, 4)],
            "catch": {"n": len(catch), "hits": catch_hits, "valid": catch_ok},
            "valid": bool(catch_ok),
        }

    task = (
        str(json.loads((study_dir / "meta.json").read_text(encoding="utf-8")).get("task", TASK_DEFECT))
        if (study_dir / "meta.json").is_file()
        else TASK_DEFECT
    )

    for cls in sorted({k["defect_class"] for k in key.values() if not k.get("is_catch")}):
        ids = [tid for tid, k in key.items() if k["defect_class"] == cls and not k.get("is_catch")]
        # Nur GÜLTIGE Hörer (Fangfragen bestanden) zählen — sonst verwässert ein
        # nachweislich nicht hinhörender Teilnehmer das Klassen-Ergebnis.
        sub = [r for r in rows if r["trial_id"] in ids and report["listeners"].get(r["listener"], {}).get("valid")]
        hits = sum(1 for r in sub if _is_hit(r))
        lo, hi = _wilson(hits, len(sub))
        report["by_class"][cls] = {
            "trials": len(sub),
            "hits": hits,
            "hit_rate": round(hits / len(sub), 4) if sub else None,
            "hit_rate_ci95": [round(lo, 4), round(hi, 4)],
            "above_chance": bool(lo > 0.5) if sub else None,
        }

    thr = float(args.min_hit_rate)
    n_min = int(args.min_trials)
    valid = [l for l, v in report["listeners"].items() if v["valid"]]
    # Der gepoolte Block wird IMMER geschrieben — auch ohne gültigen Hörer.
    # Sonst wäre im Report nicht nachvollziehbar, WESSHALB kein Beleg entstand
    # (Ausschluss durch Fangfragen ist eine Aussage, kein fehlender Eintrag).
    all_sub = [r for r in rows if r["listener"] in valid and not key[r["trial_id"]].get("is_catch")]
    hits = sum(1 for r in all_sub if _is_hit(r))
    lo, hi = _wilson(hits, len(all_sub))
    rate = hits / len(all_sub) if all_sub else 0.0
    report["pooled"] = {
        "listeners_valid": valid,
        "listeners_excluded_by_catch": [l for l in listeners if l not in valid],
        "task": task,
        "trials": len(all_sub),
        "hits": hits,
        "hit_rate": round(rate, 4) if all_sub else None,
        "hit_rate_ci95": [round(lo, 4), round(hi, 4)],
        "min_hit_rate": thr,
        "min_trials": n_min,
        "trials_sufficient": len(all_sub) >= n_min,
        # Ein Unterschied gilt nur als belegt, wenn das 95 %-Intervall den
        # Zufallspegel (0,5) sicher überschreitet — die reine Trefferquote
        # genügt NICHT (3/3 wäre p = 0,125).
        "above_chance_ci95": bool(all_sub and lo > 0.5),
    }
    if not valid:
        verdict = "UNENTSCHEIDEN"  # kein gültiger Hörer (alle per Fangfrage ausgeschlossen)
    elif task == TASK_PREFERENCE:
        # Präferenz-Test: die Frage lautet „welches klingt besser?" — es gibt
        # kein „richtig", nur eine Abweichung von der Raterate (2AFC = 0,5).
        if not report["pooled"]["trials_sufficient"]:
            verdict = "ZU_WENIGE_TRIALS"
        else:
            verdict = "PRAEFERENZ_BELEGT" if report["pooled"]["above_chance_ci95"] else "KEINE_PRAEFERENZ"
    else:
        if not report["pooled"]["trials_sufficient"]:
            verdict = "ZU_WENIGE_TRIALS"
        else:
            verdict = "HOERBAR_BELEGT" if (report["pooled"]["above_chance_ci95"] and rate >= thr) else "NICHT_BELEGT"
    report["verdict"] = verdict

    out_json = study_dir / "fit.json"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Studie {report['study']} — task={task} — Hörer: {', '.join(listeners)}")
    for ln, v in report["listeners"].items():
        flag = "gültig" if v["valid"] else "AUSGESCHLOSSEN (Fangfrage verfehlt)"
        print(f"  {ln}: {v['hits']}/{v['trials']} = {v['hit_rate']}  CI95={v['hit_rate_ci95']}  {flag}")
    for cls, v in report["by_class"].items():
        print(
            f"  [{cls}] {v['hits']}/{v['trials']} = {v['hit_rate']}  CI95={v['hit_rate_ci95']}  über Zufall (95 %): {v['above_chance']}"
        )
    if "pooled" in report:
        p = report["pooled"]
        print(f"  gepoolt (nur gültige Hörer): {p['hits']}/{p['trials']} = {p['hit_rate']}  CI95={p['hit_rate_ci95']}")
        if not p["trials_sufficient"]:
            print(f"  ! Nur {p['trials']} Trials — Mindestzahl {n_min} nicht erreicht (kein Beleg möglich).")
    print(f"  VERDIKT: {verdict}")
    print(f"  Report: {out_json}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="MUSHRA-Harness (ITU-R BS.1534)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("prepare")
    p1.add_argument("--corpus", default="corpus")
    p1.add_argument("--pair", action="append", default=None)
    p1.add_argument("--modes", nargs="+", default=["balanced"])
    p1.add_argument("--seconds", type=float, default=12.0)
    p1.add_argument("--max-pairs", type=int, default=3)
    p1.set_defaults(fn=cmd_prepare)
    p2 = sub.add_parser("sessions")
    p2.add_argument("--listeners", type=int, default=5)
    p2.add_argument("--seed", type=int, default=20260906)
    p2.add_argument("--refs", default="corpus/references")
    p2.add_argument("--out", default=None)
    p2.set_defaults(fn=cmd_sessions)
    p3 = sub.add_parser("analyze")
    p3.add_argument("--answers", required=True)
    p3.add_argument("--gap", type=float, default=8.0)
    p3.add_argument("--margin", type=float, default=15.0)
    p3.set_defaults(fn=cmd_analyze)
    p4 = sub.add_parser(
        "thresholds-build",
        help="Doppelblinde 2AFC-Studie für den Hör-Player bauen (Wohlklang Beleg 4).",
    )
    p4.add_argument("--build", required=True, help="Name der Studie (Ordnername thresholds_<build>).")
    p4.add_argument("--manifest", required=True, help="JSON-Manifest mit semantischen Rollen (reference/candidate).")
    p4.add_argument("--out", default=None, help="Zielverzeichnis (Default output_audio/mushra/thresholds_<build>).")
    p4.add_argument("--seed", type=int, default=None, help="A/B-Zuordnung (Default: seed aus dem Manifest).")
    p4.add_argument(
        "--level-match",
        action="store_true",
        help="Beide Intervalle auf gleichen RMS bringen (für Präferenz-Tests; bei Defekt-Schwellen NICHT verwenden).",
    )
    p4.set_defaults(fn=cmd_thresholds_build)
    p5 = sub.add_parser("thresholds-fit", help="Antworten auswerten (Trefferquote + Wilson-CI + Verdikt).")
    p5.add_argument("--study", required=True, help="Studienverzeichnis (mit trial_key.json und answers.csv).")
    p5.add_argument("--min-hit-rate", type=float, default=0.75, help="Schwelle für HOERBAR_BELEGT.")
    p5.add_argument(
        "--min-trials",
        type=int,
        default=12,
        help="Mindestzahl auswertbarer Trials; darunter gilt ZU_WENIGE_TRIALS (kein Beleg möglich).",
    )
    p5.set_defaults(fn=cmd_thresholds_fit)
    args = ap.parse_args()
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
