#!/usr/bin/env python3
"""run_t61_ab.py — TODO-T6-1 A/B-Abnahme: schritt-granulare Core-Guards vs. Status quo.

Führt die in der Roadmap (Abschnitt TODO-T6-1) geforderte Vorher/Nachher-Abnahme
über 3 Referenz-Songs aus. Beide Konfigurationen restaurieren denselben Song
über den UNVERÄNDERTEN Produktions-Pfad (``UnifiedRestorerV3.restore`` →
``optimize_for_excellence`` → ``ExcellenceOptimizer.optimize``) bei identischen
Seeds (§G5 (GEBOTE.md)) und identischen Parametern. Der EINZIGE Unterschied
ist die Core-Guard-Logik in ``_t61_guarded_transition``:

  VAR (neues Verhalten, Sign-off 2026-10-03): schritt-granulare Core-Guards —
      nur regressierende Optimizer-Schritte werden verworfen
      (``_t61_guarded_transition`` aus excellence_optimizer.py).
  REF (Status quo ante, kompletter Rollback): block-globaler Core-Guard —
      regressiert EIN Schritt, wird die GESAMTE Optimizer-Kette auf den Zustand
      vor ALLEN Schritten rückgängig gemacht (Verhalten VOR TODO-T6-1, verifiziert
      via ``git diff 2450f6fe^..2450f6fe``).

Umschaltmethode (keine Produktions-Code-Änderung): der Runner ersetzt per
Monkeypatch das Modul-Attribut ``_t61_guarded_transition`` des Moduls
``backend.core.excellence_optimizer``. Für REF wird eine Funktion gesetzt, die
das alte block-globale Verhalten exakt abbildet (alle verworfenen Schritte
werden zusammen rückgängig gemacht); für VAR wird die Original-Funktion
wiederhergestellt. Der Produktions-Code bleibt unangetastet.

Score-Definition (Begründung im Report): score_ref/score_var je Konfiguration
= Mittel über die Kernziele aus ``_CORE_GOALS`` (excellence_optimizer.py):
natuerlichkeit, authentizitaet, timbre_authentizitaet, tonal_center,
artikulation, transient_energie, spatial_depth. Diese 7 Ziele sind exakt die
vom Core-Guard geschützten _CORE_GOALS (``_core_regressions_between``) —
direkte Zielgröße der Guard-Entscheidung. ``harmonicity``/``harm``
(Roadmap-TODO-T6-1) ist kein Goal-Key in ``_CANONICAL_15_KEYS`` und daher
hier nicht messbar (siehe Report-Limitation). Zusätzlich wird JEDES Kernziel
einzeln als Never-worsen-Delta (Δ = VAR − REF) protokolliert (Delta-Log).

Referenz-Songs (dokumentierter Ersatz): die original "3 Referenz-Songs" sind im
Repo nicht namentlich auffindbar. Als deterministische, dokumentierte Stell-
vertreter werden 3 MUSDB18-HQ-Test-Songs genutzt (umkehrbare Entscheidung;
Nachlauf mit benannten Songs möglich):
  "AM Contra - Heart Peripheral", "Al James - Schoolboy Facination",
  "Motor Tapes - Shore".

Usage:
    python3 scripts/run_t61_ab.py                     # volle Matrix (3 Songs)
    python3 scripts/run_t61_ab.py --seconds 30        # Auszugslänge begrenzen
    python3 scripts/run_t61_ab.py --skip-validate      # nur Messung, kein Harness

Ausgabe:
    output/t61_ab_2026-10-04/t61_ab_matrix.csv        (config,score_ref,score_var)
    output/t61_ab_2026-10-04/t61_ab_report.json       (Messwerte, Deltas, Verdikt)

Determinismus (§G5 (GEBOTE.md) / §G5 (copilot-instructions.md)): gleicher Input
+ gleiche Version ⇒ bit-identischer Output; fester Master-Seed pro Lauf, kein
``time.time()`` in der Entscheidungslogik. CPU-ONLY (GPU reserviert).
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import logging
import os
import sys
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logger = logging.getLogger(__name__)

# ─── Determinismus (§G5 (GEBOTE.md)) ─────────────────────────────────────────────────────
_MASTER_SEED = 42  # AURIK_MASTER_SEED-Basis; identisch für REF und VAR je Song

# ─── Referenz-Songs (dokumentierter Ersatz — siehe Modul-Docstring) ───────────
_REFERENCE_SONGS: tuple[str, ...] = (
    "AM Contra - Heart Peripheral",
    "Al James - Schoolboy Facination",
    "Motor Tapes - Shore",
)
_MUSDB_TEST_ROOT = _ROOT / "data" / "musdb18hq" / "test"

# ─── Kernziele (exakt wie _CORE_GOALS in excellence_optimizer.py) ────────────
# Verifiziert via _CANONICAL_15_KEYS (musical_goals_metrics.py): diese 7 Ziele
# sind die vom Core-Guard in _core_regressions_between geschützten _CORE_GOALS
# und werden alle real von measure_all() im Rückgabedict geliefert.
# "harmonicity"/"harm" (Roadmap-TODO-T6-1) ist ein interner MERT-Analysewert,
# KEIN Goal-Key in _CANONICAL_15_KEYS — er wird hier bewusst NICHT mitgeführt,
# weil er sonst künstlich 0.0 in Score/Deltas einbringen würde. Im Report als
# Limitation dokumentiert (nicht über measure_all messbar).
_CORE_GOALS: tuple[str, ...] = (
    "artikulation",
    "authentizitaet",
    "natuerlichkeit",
    "spatial_depth",
    "timbre_authentizitaet",
    "tonal_center",
    "transient_energie",
)


def _load_song(song_name: str, seconds: int | None) -> tuple[np.ndarray, int]:
    """Lädt mixture.wav eines Referenz-Songs als float32 (channels-first (C, N))."""
    import soundfile as sf  # deferred: nicht immer vorhanden

    wav_path = _MUSDB_TEST_ROOT / song_name / "mixture.wav"
    if not wav_path.exists():
        raise FileNotFoundError(f"Referenz-Song nicht gefunden: {wav_path}")
    audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)  # (N, C)
    audio = audio.T  # → (C, N) channels-first (Stereo-Layout-Invariante)
    if seconds is not None and seconds > 0:
        audio = audio[:, : int(seconds * sr)]
    return np.ascontiguousarray(audio, dtype=np.float32), int(sr)


def _mean_core_goals(goals: dict[str, float]) -> float:
    """Primärscore = Mittel über die Kernziele (Begründung im Report)."""
    vals = [float(goals[g]) for g in _CORE_GOALS if g in goals and np.isfinite(goals[g])]
    return float(np.mean(vals)) if vals else float("nan")


def _core_goal_deltas(goals_ref: dict[str, float], goals_var: dict[str, float]) -> dict[str, float]:
    """Δ = VAR − REF je Kernziel (Never-worsen-Delta-Log).

    ``harm``/``harmonicity`` ist kein Goal-Key in ``_CANONICAL_15_KEYS``
    (musical_goals_metrics.py) und wird daher hier nicht mitgeführt — siehe
    Report-Limitation.
    """

    def _get(goals: dict[str, float], key: str) -> float:
        return float(goals.get(key, 0.0))

    return {g: _get(goals_var, g) - _get(goals_ref, g) for g in _CORE_GOALS}


# ─── REF: altes Verhalten (block-globaler Core-Guard, Status quo ante) ────────
def _make_ref_guarded_transition(excellence_module: Any) -> Callable[..., tuple[np.ndarray, Any]]:
    """Bildet das block-globale Verhalten VOR TODO-T6-1 exakt ab.

    Status quo ante (verifiziert via ``git diff 2450f6fe^..2450f6fe``): vor
    T6-1 gab es KEINEN schritt-granularen Hook — alle 4 Optimizer-Schritte
    liefen blind durch, und der block-globale Core-Guard (in ``optimize()``
    direkt) machte bei Regression den GESAMTEN Optimizer-Output auf den
    Zustand vor ALLEN Schritten rückgängig (``out = audio.copy()``).

    Diese Funktion ersetzt den Hook durch eine No-op-Übernahme (alle Schritte
    laufen blind wie vor T6-1), merkt sich beim ersten Aufruf den Block-Start-
    Zustand und rollt bei erster Regression auf diesen zurück. Damit ist jeder
    spätere Aufruf ein No-op auf dem bereits zurückgesetzten Zustand.
    """

    state: dict[str, Any] = {"block_start": None, "rolled_back": False}

    def _ref_guarded_transition(
        result: Any,
        step_name: str,
        pre_out: np.ndarray,
        out_new: np.ndarray,
        goals_pre: Any,
        checker: Any,
        sample_rate: int,
    ) -> tuple[np.ndarray, Any]:
        # Beim ersten Hook-Aufruf den Block-Start-Zustand merken (= Zustand
        # vor ALLEN Optimizer-Schritten, wie `out = audio.copy()` in optimize()).
        if state["block_start"] is None:
            state["block_start"] = pre_out.copy()
        # Bereits gerollt? Dann bleibt der Zustand unverändert (Status quo ante:
        # kein erneutes Anwenden der Schritte nach einem Rollback).
        if state["rolled_back"]:
            return pre_out, goals_pre
        # Blind übernehmen (kein schritt-granulares Verwerfen wie in VAR).
        if checker is None or goals_pre is None:
            return out_new, goals_pre
        try:
            goals_post = checker.measure_all(out_new.astype(pre_out.dtype), sample_rate)
            regs = excellence_module._core_regressions_between(goals_pre, goals_post)
            if regs:
                logger.warning(
                    "REF (block-globaler Core-Guard): Rollback der GESAMTEN Optimizer-Kette "
                    "auf den Block-Start-Zustand wegen Kernziel-Regressionen (%s) "
                    "— Schritt '%s'",
                    ", ".join(regs[:6]),
                    step_name,
                )
                result.core_guard_triggered = True
                result.core_guard_regressions = list(regs)
                result.applied_steps.append("core_guard_rollback")
                state["rolled_back"] = True
                # Status quo ante: GANZER Block-Rollback → Zustand vor ALLEN Schritten.
                return state["block_start"], goals_pre
            return out_new, goals_post
        except Exception as exc:  # §G23 (ML→DSP-Ersatzpfad) — nie blind verwerfen (§V7 (copilot-instructions.md))
            logger.warning("REF Core-Guard fehlgeschlagen (%s) — Schritt behalten", exc)
            return out_new, goals_pre

    return _ref_guarded_transition


def _configure_variant(variant: str) -> dict[str, Any]:
    """Schaltet per Monkeypatch zwischen REF und VAR um (kein Produktions-Code)."""
    import backend.core.excellence_optimizer as eo

    original = eo._t61_guarded_transition
    method: str
    if variant == "VAR":
        eo._t61_guarded_transition = original  # explizit: neues Verhalten aktiv
        method = "Original `_t61_guarded_transition` (schritt-granulare Guards)"
    elif variant == "REF":
        eo._t61_guarded_transition = _make_ref_guarded_transition(eo)  # type: ignore[assignment]
        method = (
            "Monkeypatch: `_t61_guarded_transition` → block-globaler Rollback-Äquivalent "
            "des Status quo ante (alle Schritte rückgängig bei Regression)"
        )
    else:
        raise ValueError(f"Unbekannte Variante: {variant}")
    return {"excellence_module": eo, "original_transition": original, "method": method}


def _restore_variant(
    song_name: str,
    variant: str,
    audio: np.ndarray,
    sr: int,
) -> dict[str, Any]:
    """Restauriert einen Song in der gewählten Guard-Variante über den Produktions-Pfad."""
    from backend.core.unified_restorer_v3 import RestorationConfig, UnifiedRestorerV3

    switch_info = _configure_variant(variant)
    eo: Any = switch_info["excellence_module"]

    # §G5 (GEBOTE.md): fester Master-Seed — identisch für REF und VAR je Song (fairer A/B-Vergleich).
    os.environ["AURIK_MASTER_SEED"] = str(_MASTER_SEED)
    try:
        from backend.core.seed_manager import get_seed_manager

        get_seed_manager().start_session(song_id=song_name, master_seed=_MASTER_SEED)
    except Exception as seed_exc:  # §V6 (copilot-instructions.md): Fallback mit Begründung
        logger.warning(
            "§V6 (copilot-instructions.md) Fallback: seed_manager nicht verfügbar (%s) — nutze AURIK_MASTER_SEED=%d",
            seed_exc,
            _MASTER_SEED,
        )

    cfg = RestorationConfig()
    engine = UnifiedRestorerV3(cfg)
    try:
        result = engine.restore(audio, sample_rate=sr)
    finally:
        # Monkeypatch zurücksetzen — Produktions-Zustand nie dauerhaft verändern.
        eo._t61_guarded_transition = switch_info["original_transition"]
        os.environ.pop("AURIK_MASTER_SEED", None)

    restored = np.asarray(getattr(result, "audio", audio), dtype=np.float32)

    # Kernziele auf dem restaurierten Audio messen (identische Messkette für REF/VAR).
    from backend.core.musical_goals.musical_goals_metrics import get_checker

    checker = get_checker()
    goals = checker.measure_all(restored, sr)

    return {
        "song": song_name,
        "variant": variant,
        "switch_method": switch_info["method"],
        "master_seed": _MASTER_SEED,
        "goals": {k: float(v) for k, v in goals.items() if isinstance(v, (int, float))},
        "core_score": _mean_core_goals(goals),
        "audio_duration_s": float(audio.shape[-1]) / float(sr),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="TODO-T6-1 A/B-Abnahme (Referenzmatrix)")
    parser.add_argument("--seconds", type=int, default=0, help="Auszugslänge je Song in Sekunden (0 = ganz)")
    parser.add_argument("--skip-validate", action="store_true", help="Harness-Verdikt überspringen")
    parser.add_argument("--out", type=Path, default=None, help="Ausgabe-Verzeichnis")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(name)s — %(message)s")

    out_dir = args.out or (_ROOT / "output" / "t61_ab_2026-10-04")
    out_dir.mkdir(parents=True, exist_ok=True)

    # CPU-ONLY (GPU reserviert) — §V6-konform dokumentiert.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    rows: list[dict[str, Any]] = []
    per_song: list[dict[str, Any]] = []
    never_worsen_ok = True

    for song in _REFERENCE_SONGS:
        try:
            audio, sr = _load_song(song, args.seconds or None)
        except Exception as load_exc:
            logger.warning("§V6 (VERBOTEN.md) Ersatzpfad: Song '%s' nicht ladbar (%s) — übersprungen", song, load_exc)
            continue

        ref = _restore_variant(song, "REF", audio, sr)
        var = _restore_variant(song, "VAR", audio, sr)

        score_ref = ref["core_score"]
        score_var = var["core_score"]
        rows.append({"config": song, "score_ref": score_ref, "score_var": score_var})

        deltas = _core_goal_deltas(ref["goals"], var["goals"])
        regressions = [g for g, d in deltas.items() if d < 0.0]
        if regressions:
            never_worsen_ok = False

        per_song.append(
            {
                "song": song,
                "score_ref": score_ref,
                "score_var": score_var,
                "delta_total": score_var - score_ref,
                "core_goal_deltas": deltas,
                "never_worsen_regressions": regressions,
                "switch_method": var["switch_method"],
                "master_seed": var["master_seed"],
                "audio_duration_s": var["audio_duration_s"],
            }
        )

    if not rows:
        logger.error("Keine Songs verarbeitbar — Abbruch.")
        return 2

    matrix_csv = out_dir / "t61_ab_matrix.csv"
    with matrix_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["config", "score_ref", "score_var"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    verdict: dict[str, Any] | None = None
    if not args.skip_validate:
        spec = importlib.util.spec_from_file_location("validate_t61_ab", _ROOT / "scripts" / "validate_t61_ab.py")
        if spec and spec.loader:
            harness = importlib.util.module_from_spec(spec)
            sys.modules["validate_t61_ab"] = harness
            spec.loader.exec_module(harness)
            ref_scores = [float(r["score_ref"]) for r in rows]
            var_scores = [float(r["score_var"]) for r in rows]
            decision = harness.decide_ab(ref_scores, var_scores)
            verdict = asdict(decision)
            logger.info("Harness-Verdikt: %s", json.dumps(verdict, ensure_ascii=False))
        else:
            logger.warning("§V6 (VERBOTEN.md) Ersatzpfad: A/B-Prüfmodul nicht ladbar — kein Harness-Verdikt")

    report = {
        "task": "TODO-T6-1 A/B-Abnahme",
        "reference_songs_substitute": {
            "note": (
                "Die original '3 Referenz-Songs' sind im Repo nicht namentlich auffindbar. "
                "Dokumentierter, deterministischer Ersatz: 3 MUSDB18-HQ-Test-Songs "
                "(umkehrbare Entscheidung; Nachlauf mit benannten Songs möglich)."
            ),
            "songs": list(_REFERENCE_SONGS),
        },
        "switch_method": (
            "Monkeypatch des Modul-Attributs `_t61_guarded_transition` in "
            "backend.core.excellence_optimizer — REF bildet block-globalen Rollback "
            "(Status quo ante) ab, VAR setzt die Original-Funktion (schritt-granular). "
            "Kein Produktions-Code geändert."
        ),
        "score_definition": (
            "score_ref/score_var = Mittel über die Kernziele (_CORE_GOALS aus "
            "excellence_optimizer.py): artikulation, authentizitaet, natuerlichkeit, "
            "spatial_depth, timbre_authentizitaet, tonal_center, transient_energie. "
            "Diese 7 Ziele sind exakt die vom Core-Guard geschützten _CORE_GOALS "
            "(_core_regressions_between) — direkte Zielgröße der Guard-Entscheidung. "
            "harmonicity/harm (Roadmap-TODO-T6-1) ist kein Goal-Key in "
            "_CANONICAL_15_KEYS und daher nicht über measure_all() messbar — "
            "als Limitation im Report dokumentiert."
        ),
        "master_seed": _MASTER_SEED,
        "never_worsen_ok": never_worsen_ok,
        "per_song": per_song,
        "matrix_csv": str(matrix_csv),
        "harness_verdict": verdict,
    }

    report_path = out_dir / "t61_ab_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Report geschrieben: %s", report_path)
    logger.info("Matrix geschrieben: %s", matrix_csv)

    print(
        json.dumps(
            {"matrix_csv": str(matrix_csv), "report": str(report_path), "never_worsen_ok": never_worsen_ok},
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
