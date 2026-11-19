"""AMRB Baseline Runner — Aurik 10.0.0.x

Führt den Musical Restoration Benchmark mit der echten AurikDenker-Pipeline
durch und speichert den JSON-Bericht unter benchmarks/amrb_baseline_<mode>.json.

Verwendung::

    .venv_aurik/bin/python benchmarks/run_amrb_baseline.py
    .venv_aurik/bin/python benchmarks/run_amrb_baseline.py --mode studio
    .venv_aurik/bin/python benchmarks/run_amrb_baseline.py --n-items 3 --scenarios tape vinyl dropout
    .venv_aurik/bin/python benchmarks/run_amrb_baseline.py --dry-run

Optionen:
    --mode          restoration | studio  (default: restoration)
    --n-items       Stimuli pro Szenario  (default: 5)
    --duration      Länge je Stimulus in Sekunden (default: 30.0, Minimum: 30.0)
    --scenarios     Teilmenge: tape vinyl shellac digital codec vocal reverb hum dropout composite
    --report-path   Expliziter Ausgabepfad für JSON-Bericht
    --dry-run       Nur DSP-Pass-Through — kein ML, schnell
    --pre-listening-gate / --no-pre-listening-gate
                    Hard-Fails für Hörtest-Readiness (Default: aktiv)
    --max-restoration-exceptions
                    Max. erlaubte Restore-Exceptions (Default: 0)
    --max-mushra-fallbacks
                    Max. erlaubte MUSHRA-Fallbacks (Default: 0)
    --max-runtime-seconds
                    Max. Laufzeit für Gate (Default: 300)
    --no-rt-limit   Deaktiviert Denker-Ressourcenlimits (kann RAM-Spitzen erhöhen)
    --chain-hint    Exakte Tonträgerkette (z. B. vinyl>tape>mp3_low)
    --verbose       Ausführliches Logging
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

# Keep CPU thread fan-out conservative to avoid RAM spikes on desktop systems.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

# ---------------------------------------------------------------------------
# AMRB Constraints
# ---------------------------------------------------------------------------
# Minimum fragment duration for statistically reliable OQS/MUSHRA scoring.
# Fragments shorter than 30 s produce per-item variance exceeding ±8 OQS points,
# making OQS ≥ 80 pass/fail gating unreliable. [RELEASE_MUST §8.1.2]
_MIN_AMRB_FRAGMENT_S: float = 30.0

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("amrb_runner")

# ---------------------------------------------------------------------------
# Scenario name → AMRB key mapping
# ---------------------------------------------------------------------------
_SCENARIO_KEYS = {
    "tape": "AMRB-01-TAPE",
    "vinyl": "AMRB-02-VINYL",
    "shellac": "AMRB-03-SHELLAC",
    "digital": "AMRB-04-DIGITAL",
    "codec": "AMRB-05-CODEC",
    "vocal": "AMRB-06-VOCAL",
    "reverb": "AMRB-07-REVERB",
    "hum": "AMRB-08-HUM",
    "dropout": "AMRB-09-DROPOUT",
    "composite": "AMRB-10-COMPOSITE",
}

# Scenario-aware defaults for synthetic AMRB material priors.
# Applied only when user does not pass explicit --material-hint / --chain-hint.
_SCENARIO_DEFAULT_HINTS: dict[str, tuple[str, str]] = {
    "AMRB-01-TAPE": ("reel_tape", "reel_tape"),
    "AMRB-02-VINYL": ("vinyl", "vinyl"),
    "AMRB-03-SHELLAC": ("shellac", "shellac"),
    "AMRB-04-DIGITAL": ("cd_digital", "cd_digital"),
    "AMRB-05-CODEC": ("mp3_low", "mp3_low"),
    "AMRB-06-VOCAL": ("cd_digital", "cd_digital"),
    "AMRB-07-REVERB": ("reel_tape", "reel_tape"),
    "AMRB-08-HUM": ("tape", "tape"),
    "AMRB-09-DROPOUT": ("tape", "tape"),
    "AMRB-10-COMPOSITE": ("tape", "vinyl>tape"),
}

_VALID_MEDIA_HINTS = {
    "tape",
    "reel_tape",
    "vinyl",
    "shellac",
    "wax_cylinder",
    "wire_recording",
    "lacquer_disc",
    "dat",
    "cd_digital",
    "mp3_low",
    "mp3_high",
    "aac",
    "minidisc",
    "streaming",
}

_ANALOG_MEDIA_HINTS = {
    "tape",
    "reel_tape",
    "vinyl",
    "shellac",
    "wax_cylinder",
    "wire_recording",
    "lacquer_disc",
}


def _parse_chain_hint(chain_hint: str | None) -> list[str]:
    """Parse carrier chain hint with support for '>', ',', ';', '/' separators."""
    if not chain_hint:
        return []
    raw = str(chain_hint).strip().lower()
    if not raw:
        return []
    for sep in (",", ";", "/"):
        raw = raw.replace(sep, ">")
    chain = [part.strip() for part in raw.split(">") if part.strip()]
    if not chain:
        return []
    invalid = [m for m in chain if m not in _VALID_MEDIA_HINTS]
    if invalid:
        raise ValueError(f"Ungültiger chain-hint: {invalid}. Gültig: {sorted(_VALID_MEDIA_HINTS)}")
    return chain


def _derive_primary_material_from_chain(chain: list[str]) -> str:
    """Return primary material per spec intent: last analog stage in chain."""
    if not chain:
        return "unknown"
    analogs = [m for m in chain if m in _ANALOG_MEDIA_HINTS]
    if analogs:
        return analogs[-1]
    return chain[-1]


def _infer_input_extension_for_chain(chain: list[str], fallback_material: str) -> str:
    """Infer file extension so UV3 file_ext-dependent guards follow chain terminal stage."""
    terminal = chain[-1] if chain else fallback_material
    if terminal in {"mp3_low", "mp3_high"}:
        return ".mp3"
    if terminal == "aac":
        return ".m4a"
    if terminal in {
        "vinyl",
        "shellac",
        "wax_cylinder",
        "wire_recording",
        "lacquer_disc",
        "tape",
        "reel_tape",
        "dat",
        "cd_digital",
        "streaming",
        "minidisc",
    }:
        return ".wav"
    return ".wav"


def _build_cached_medium_hint(material_hint: str | None, chain_hint: str | None):
    """Build a lightweight cached_medium_result object for AurikDenker.

    This is used for controlled benchmarks where the source material is known
    by construction (e.g. AMRB-01-TAPE) and should not be re-inferred from
    short synthetic snippets.
    """
    try:
        chain = _parse_chain_hint(chain_hint)
    except ValueError as exc:
        raise ValueError(str(exc)) from exc

    if material_hint is not None and str(material_hint).strip():
        m = str(material_hint).strip().lower()
        if m not in _VALID_MEDIA_HINTS:
            raise ValueError(f"Ungültiger material-hint: {m}")
    else:
        m = ""

    if not chain and not m:
        return None

    if not chain:
        chain = [m]
    primary = _derive_primary_material_from_chain(chain)
    material_type = m or primary
    return SimpleNamespace(
        material_type=material_type,
        confidence=0.99,
        transfer_chain=chain,
        medium_confidences=[0.99 for _ in chain],
        primary_material=primary,
    )


def _build_restoration_fn(
    mode: str,
    dry_run: bool,
    no_rt_limit: bool,
    material_hint: str | None,
    chain_hint: str | None,
):
    """Return the (audio, sr) → restored_audio callable for the benchmark.

    dry_run: returns input unchanged (sanity check, fast).
    """
    if dry_run:
        logger.info("DRY-RUN: restoration_fn is pass-through (no ML).")
        return lambda audio, sr: audio.copy()

    # Lazy-import so benchmark CLI works even without full backend warm-up
    try:
        from denker.aurik_denker import get_aurik_denker
    except Exception as exc:
        logger.error("AurikDenker import failed: %s", exc)
        raise

    denker = get_aurik_denker()
    cached_medium_hint = _build_cached_medium_hint(material_hint, chain_hint)
    chain = list(getattr(cached_medium_hint, "transfer_chain", []) or []) if cached_medium_hint is not None else []
    _hint_ext = _infer_input_extension_for_chain(chain, str(material_hint or "unknown").lower())
    logger.info(
        "AurikDenker loaded, mode=%s, no_rt_limit=%s, material_hint=%s, chain_hint=%s",
        mode,
        no_rt_limit,
        material_hint or "auto",
        chain if chain else "auto",
    )

    def restoration_fn(audio: np.ndarray, sr: int, sid: str | None = None) -> np.ndarray:
        try:
            _effective_cached_hint = cached_medium_hint
            _effective_chain = chain
            _effective_ext = _hint_ext

            if _effective_cached_hint is None and sid is not None:
                _scenario_hint = _SCENARIO_DEFAULT_HINTS.get(str(sid))
                if _scenario_hint is not None:
                    _scenario_material, _scenario_chain = _scenario_hint
                    _effective_cached_hint = _build_cached_medium_hint(_scenario_material, _scenario_chain)
                    _effective_chain = list(getattr(_effective_cached_hint, "transfer_chain", []) or [])
                    _effective_ext = _infer_input_extension_for_chain(_effective_chain, _scenario_material)

            if _effective_cached_hint is not None:
                _hint_chain_label = "_".join(_effective_chain) if _effective_chain else str(material_hint or "auto")
                _hint_input_path = f"amrb_input_{_hint_chain_label}{_effective_ext}"
                result = denker.denke(
                    audio,
                    sr,
                    mode=mode,
                    no_rt_limit=no_rt_limit,
                    cached_medium_result=_effective_cached_hint,
                    input_path=_hint_input_path,
                )
            else:
                result = denker.denke(audio, sr, mode=mode, no_rt_limit=no_rt_limit)
            return result.audio
        except Exception as exc:
            logger.warning("restoration_fn error: %s — returning input unchanged", exc)
            return audio.copy()

    return restoration_fn


def _run(args: argparse.Namespace) -> int:
    # Import here so errors are caught after arg parsing
    try:
        import sys as _sys

        _sys.path.insert(0, str(Path(__file__).parent.parent))
        from benchmarks.musical_restoration_benchmark import BenchmarkConfig, run_benchmark
    except Exception as exc:
        logger.error("AMRB import failed: %s", exc)
        return 1

    # §8.1.2 Minimum fragment duration guard — fragments < 30 s produce OQS variance
    # exceeding ±8 points, making pass/fail gating unreliable. [RELEASE_MUST]
    if args.duration < _MIN_AMRB_FRAGMENT_S:
        logger.warning(
            "§8.1.2 AMRB: --duration %.1f s liegt unter dem Minimum von %.0f s. "
            "OQS-Bewertung auf kurzen Fragmenten ist statistisch unzuverlässig (±8 OQS). "
            "Verwende %.0f s.",
            args.duration,
            _MIN_AMRB_FRAGMENT_S,
            _MIN_AMRB_FRAGMENT_S,
        )
        args.duration = _MIN_AMRB_FRAGMENT_S

    # Resolve scenario filter
    scenario_filter: list[str] | None = None
    if args.scenarios:
        scenario_filter = []
        for s in args.scenarios:
            key = _SCENARIO_KEYS.get(s.lower())
            if key is None:
                logger.error("Unknown scenario '%s'. Valid: %s", s, list(_SCENARIO_KEYS))
                return 1
            scenario_filter.append(key)

    # Default report path
    if args.report_path:
        report_path = Path(args.report_path)
    else:
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        report_path = Path("benchmarks") / f"amrb_baseline_{args.mode}_{ts}.json"

    restoration_fn = _build_restoration_fn(
        args.mode,
        dry_run=args.dry_run,
        no_rt_limit=args.no_rt_limit,
        material_hint=args.material_hint,
        chain_hint=args.chain_hint,
    )

    config = BenchmarkConfig(
        restoration_fn=restoration_fn,
        sample_rate=48_000,
        n_items_per_scenario=args.n_items,
        duration_s=args.duration,
        scenarios=scenario_filter,
        report_path=report_path,
        system_name=f"Aurik 10.0.0.x ({args.mode})",
        verbose=args.verbose,
    )

    logger.info(
        "Starting AMRB: mode=%s, n_items=%d, duration=%.1fs, scenarios=%s",
        args.mode,
        args.n_items,
        args.duration,
        scenario_filter or "all",
    )
    t0 = time.monotonic()
    report = run_benchmark(config)
    elapsed = time.monotonic() - t0

    total_items = 0
    restoration_exceptions = 0
    mushra_fallbacks = 0
    for scenario_result in report.scenario_results.values():
        for item in scenario_result.items:
            total_items += 1
            restoration_exceptions += int(bool(item.get("restoration_exception", False)))
            mushra_fallbacks += int(bool(item.get("mushra_fallback_used", False)))

    pre_listening_fail_reasons: list[str] = []
    if args.pre_listening_gate:
        if restoration_exceptions > args.max_restoration_exceptions:
            pre_listening_fail_reasons.append(
                f"restore_exceptions={restoration_exceptions} > {args.max_restoration_exceptions}"
            )
        if mushra_fallbacks > args.max_mushra_fallbacks:
            pre_listening_fail_reasons.append(f"mushra_fallbacks={mushra_fallbacks} > {args.max_mushra_fallbacks}")
        if elapsed > args.max_runtime_seconds:
            pre_listening_fail_reasons.append(f"runtime={elapsed:.1f}s > {args.max_runtime_seconds:.1f}s")

    # Print summary
    print("\n" + "=" * 60)
    print(f"  AMRB Baseline — {config.system_name}")
    print("=" * 60)
    print(f"  Overall Score : {report.overall_score:.1f} / 100  (target ≥ 80)")
    print(f"  Scenarios     : {len(report.scenario_results)}")
    print(f"  Runtime       : {elapsed:.0f} s")
    print(f"  Report saved  : {report_path}")
    print(f"  Items         : {total_items}")
    print(f"  Restore errs  : {restoration_exceptions}")
    print(f"  MUSHRA fb     : {mushra_fallbacks}")
    print()

    passed = 0
    for name, res in report.scenario_results.items():
        score = res.mushra_mean
        ok = "✅" if score >= 80 else "❌"
        print(f"  {ok}  {name:<30} OQS={score:.1f}")
        if score >= 80:
            passed += 1

    print()
    print(f"  Gate: {passed}/{len(report.scenario_results)} scenarios ≥ 80")
    if args.pre_listening_gate:
        if pre_listening_fail_reasons:
            print("  ❌ PRE-LISTENING gate FAILED")
            print(f"     Gründe: {'; '.join(pre_listening_fail_reasons)}")
        else:
            print("  ✅ PRE-LISTENING gate PASSED")
    if report.overall_score >= 80:
        print("  ✅ RELEASE_MUST gate PASSED")
    else:
        print("  ❌ RELEASE_MUST gate FAILED")
    print("=" * 60)

    # Also append a compact summary to a persistent baseline log
    log_path = Path("benchmarks") / "amrb_baseline_log.jsonl"
    entry = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "mode": args.mode,
        "material_hint": args.material_hint,
        "chain_hint": args.chain_hint,
        "system_name": config.system_name,
        "overall_score": report.overall_score,
        "scenarios_passed": passed,
        "scenarios_total": len(report.scenario_results),
        "n_items": args.n_items,
        "total_items": total_items,
        "elapsed_s": round(elapsed, 1),
        "restoration_exceptions": restoration_exceptions,
        "mushra_fallbacks": mushra_fallbacks,
        "pre_listening_gate_enabled": bool(args.pre_listening_gate),
        "pre_listening_gate_passed": not pre_listening_fail_reasons,
        "pre_listening_fail_reasons": pre_listening_fail_reasons,
        "report_path": str(report_path),
    }
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    logger.info("Baseline entry appended to %s", log_path)

    hard_gate_ok = report.overall_score >= 80
    pre_listening_ok = (not args.pre_listening_gate) or (not pre_listening_fail_reasons)
    return 0 if (hard_gate_ok and pre_listening_ok) else 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AMRB Baseline Runner — misst OQS der aktuellen Aurik-Pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["restoration", "studio"],
        default="restoration",
        help="Restaurierungsmodus (default: restoration)",
    )
    parser.add_argument(
        "--n-items",
        type=int,
        default=5,
        metavar="N",
        help="Stimuli pro Szenario (default: 5)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=_MIN_AMRB_FRAGMENT_S,
        metavar="S",
        help=f"Stimulus-Länge in Sekunden (default: {_MIN_AMRB_FRAGMENT_S:.0f}, Minimum: {_MIN_AMRB_FRAGMENT_S:.0f})",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        metavar="NAME",
        help="Teilmenge der Szenarien, z.B. --scenarios tape dropout",
    )
    parser.add_argument(
        "--report-path",
        metavar="PATH",
        help="Expliziter JSON-Ausgabepfad",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Pass-Through ohne ML — Sanity-Check",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Ausführliches Logging",
    )
    parser.add_argument(
        "--no-rt-limit",
        action="store_true",
        help="Deaktiviert Echtzeit-/Ressourcenlimits im Denker (nur für leistungsstarke Systeme)",
    )
    parser.add_argument(
        "--material-hint",
        choices=sorted(_VALID_MEDIA_HINTS),
        help="Fixiert den Materialtyp für den Benchmarklauf (optional, für kontrollierte Szenarien)",
    )
    parser.add_argument(
        "--chain-hint",
        help=(
            "Fixiert exakte Tonträgerkette (beliebige Länge), z.B. vinyl>tape>mp3_low oder wax_cylinder,tape,mp3_high"
        ),
    )
    parser.add_argument(
        "--pre-listening-gate",
        dest="pre_listening_gate",
        action="store_true",
        help="Aktiviert Hard-Fails vor Hörtests (Default: aktiv)",
    )
    parser.add_argument(
        "--no-pre-listening-gate",
        dest="pre_listening_gate",
        action="store_false",
        help="Deaktiviert Pre-Listening-Hard-Fails",
    )
    parser.add_argument(
        "--max-restoration-exceptions",
        type=int,
        default=0,
        metavar="N",
        help="Maximal erlaubte Restore-Exceptions für Pre-Listening-Gate (Default: 0)",
    )
    parser.add_argument(
        "--max-mushra-fallbacks",
        type=int,
        default=0,
        metavar="N",
        help="Maximal erlaubte MUSHRA-Fallbacks für Pre-Listening-Gate (Default: 0)",
    )
    parser.add_argument(
        "--max-runtime-seconds",
        type=float,
        default=300.0,
        metavar="S",
        help="Maximale Laufzeit für Pre-Listening-Gate in Sekunden (Default: 300)",
    )
    parser.set_defaults(pre_listening_gate=True)
    args = parser.parse_args()
    try:
        _parse_chain_hint(args.chain_hint)
    except ValueError as exc:
        parser.error(str(exc))
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    sys.exit(_run(args))


if __name__ == "__main__":
    main()
