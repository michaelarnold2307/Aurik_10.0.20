#!/usr/bin/env python3
"""
§v10.800: Echter-Corpus-Benchmark — die ehrliche Wahrheitsmessung.

Misst die Restaurations-Kette gegen beschädigte Aufnahmen aus corpus/.
Jede Datei hat ein clean-Pendant. Metriken:

  1. SNR-Verbesserung (dB, gain-angepasst)  — Kernmetrik, pegel-invariant
  2. SNR roh (dB) + Output-Pegel (dB)       — zerlegt Pegel- vs. Inhalts-Fehler
  3. MSE vor/nach                           — Abstand zur Clean-Referenz
  4. MOS (optional, Zeuge)                  — auf synthetischem Korpus trügerisch
  5. Verdict pro Datei                      — verbessert / verschlechtert / neutral

Pfade (--pipeline):
  canonical   = Produktionspfad Import → Pre-Analysis → AurikDenker.denke() (§G9 (copilot-instructions.md))
                — der Default; NUR dieser Pfad misst, was CLI/GUI hören.
  uv3         = Engine direkt (UnifiedRestorerV3.restore()) — DIAGNOSE, nicht Produktion
                (ohne GlobalPlan greift der §9.7.7-Era-Fallback: Befund 2026-10-07
                vinyl → decade=1890/wax_cylinder → §2.46c LPF@5 kHz).
  coordinated = SOTA-Repair-Kette (bisheriges Verhalten) — Legacy, erwartet Mono.

Ehrlichkeits-Regeln (Befunde 2026-10-07, §v10.800b–d):
  - Stereo bleibt Stereo — kein Kanal-Kollaps (der alte Mono-Kollaps verfälschte
    jedes Verdikt, Stereo-Layout-Invariante).
  - SNR wird NUR gain-angepasst verdiktet. Der rohe SNR bestrafte die
    Lautheits-Normalisierung (OneTakeExport → −16 LUFS) fälschlich mit −22 dB,
    obwohl der Pegel nichts über Restaurations-Qualität sagt.
  - MOS auf synthetischem Korpus hat KEINE Urteilskraft (clean=1,61 vs.
    zerstört=4,73 — invertiert). Er läuft optional und bleibt Zeuge.
  - Ausgaben können mit --save-wav gesichert werden (Spektral-Diagnose + Hörprobe).
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT))

SR = 48000
FILES_PER_MEDIUM = 2  # Stichprobe: 2 Dateien pro Medium (0 = alle)


@dataclass
class FileResult:
    medium: str
    file: str
    snr_before_db: float  # gain-angepasst: damaged vs. clean
    snr_after_db: float  # gain-angepasst: restored vs. clean
    snr_after_raw_db: float  # roh (pegel-sensitiv) — nur Dokumentation
    out_gain_db: float  # Output-Pegel vs. Referenz (negativ = leiser)
    snr_improvement_db: float  # gain-angepasst — verdikt-tragend
    mse_before: float
    mse_after: float
    mos_damaged: float
    mos_restored: float
    mos_clean: float
    verdict: str  # improved / degraded / neutral
    processing_time: float
    output_path: str = ""


@dataclass
class BenchmarkReport:
    results: list[FileResult] = field(default_factory=list)
    total_time: float = 0.0

    @property
    def improved(self) -> int:
        return sum(1 for r in self.results if r.verdict == "improved")

    @property
    def degraded(self) -> int:
        return sum(1 for r in self.results if r.verdict == "degraded")

    @property
    def avg_snr_improvement(self) -> float:
        if not self.results:
            return 0.0
        return float(np.mean([r.snr_improvement_db for r in self.results]))


def _pair_files(medium: str) -> list[tuple[Path, Path]]:
    """Paart damaged-Dateien mit ihren clean-Pendants.

    §v10.800a (2026-10-07): Längster-Präfix-Abgleich statt „letztes `_`-Segment
    entfernen" — das alte Verfahren verfehlte jede Datei mit MEHRFACH-Suffix
    (`…_hiss_wow`, `…_combined`) und lieferte dadurch nur 4 statt 30+ Paaren.
    """
    clean_dir = _PROJECT / "corpus" / medium / "clean"
    damaged_dir = _PROJECT / "corpus" / medium / "damaged"
    _cleans = [(c, c.stem[: -len("_clean")]) for c in sorted(clean_dir.glob("*_clean.wav"))]
    pairs = []
    for damaged in sorted(damaged_dir.glob("*.wav")):
        stem = damaged.stem
        # Längster passender Clean-Basisname gewinnt (eindeutig bei mehrfachen Suffixen).
        _treffer = [(c, base) for c, base in _cleans if stem == base or stem.startswith(base + "_")]
        if not _treffer:
            continue
        clean, _base = max(_treffer, key=lambda t: len(t[1]))
        pairs.append((damaged, clean))
    return pairs


def _load_audio(path: Path, max_seconds: float = 0.0) -> tuple[np.ndarray, int]:
    """Kanonischer Loader (Bridge-Kaskade) — Stereo bleibt Stereo!

    §v10.800b: Kanal-Kollaps (`.mean(axis=1)`) entfernt — die Produktion
    verarbeitet Stereo, der alte Mono-Kollaps verfälschte jedes Verdikt
    (Stereo-Layout-Invariante). Layout (N, C) wie `cli/aurik_cli._load_audio`.
    """
    from backend.api.bridge import get_load_audio_fn

    loaded = get_load_audio_fn()(str(path), target_sr=None, mono=False, do_carrier_analysis=False)
    audio = np.asarray(loaded["audio"], dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]
    elif audio.ndim == 2 and audio.shape[0] < audio.shape[1]:
        audio = audio.T
    sr = int(loaded["sr"])
    if max_seconds and max_seconds > 0:
        audio = audio[: min(int(max_seconds * sr), audio.shape[0])]
    return np.clip(np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0), sr


def _as_audio_2d(arr: np.ndarray) -> np.ndarray:
    """Bringt Pipeline-Ausgaben auf (N, C) — Stereo-Layout-Invariante, beide Layouts bedienen."""
    out = np.asarray(arr, dtype=np.float32)
    if out.ndim == 1:
        return out[:, np.newaxis]
    if out.ndim == 2 and out.shape[0] <= 8 and out.shape[0] < out.shape[1]:
        return out.T  # channels-first (C, N) → (N, C)
    return out


def _align_pair(reference: np.ndarray, signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Trimmt auf gemeinsame Länge; bei Kanal-Mismatch beide Seiten auf Mono-Mittel."""
    ref, sig = _as_audio_2d(reference), _as_audio_2d(signal)
    if ref.shape[1] != sig.shape[1]:
        ref = ref.mean(axis=1, keepdims=True)
        sig = sig.mean(axis=1, keepdims=True)
    n = min(len(ref), len(sig))
    return ref[:n], sig[:n]


def _snr_pair(reference: np.ndarray, signal: np.ndarray) -> tuple[float, float, float]:
    """(snr_raw_db, snr_gain_matched_db, out_gain_db) — pegel-ehrliche SNR.

    §v10.800c: Der rohe SNR war pegel-sensitiv — jede lautheitsnormalisierte
    Ausgabe (OneTakeExport → −16 LUFS) kollabierte ihn zu Unrecht
    (Produktionsbefund 2026-10-07: UV3-direkt −22,1 dB trotz intaktem Inhalt).
    Primär-Metrik ist die least-squares-gain-angepasste SNR:
    alpha = <ref, sig> / <sig, sig>;  SNR_gm = SNR(ref, alpha·sig).
    out_gain_db = −20·log10(alpha) — negativ = Output leiser als Referenz.
    """
    ref, sig = _align_pair(reference, signal)
    ref = ref.astype(np.float64)
    sig = sig.astype(np.float64)
    sig_power = float(np.sum(sig * sig)) + 1e-20
    alpha = float(np.sum(ref * sig)) / sig_power
    ref_power = float(np.sum(ref * ref)) + 1e-20
    _diff = ref - sig
    snr_raw = float(10.0 * np.log10(ref_power / (float(np.sum(_diff * _diff)) + 1e-20)))
    _diff_gm = ref - alpha * sig
    snr_gm = float(10.0 * np.log10(ref_power / (float(np.sum(_diff_gm * _diff_gm)) + 1e-20)))
    out_gain_db = float(-20.0 * np.log10(abs(alpha) + 1e-20)) if alpha != 0.0 else 0.0
    return snr_raw, snr_gm, out_gain_db


def _save_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    """Sichert eine Ausgabe für Spektral-Diagnose und Hörprobe (PCM_24, §G4-Reihe folgt im Export)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), _as_audio_2d(audio), sr, subtype="PCM_24")


def _restore_canonical(damaged: np.ndarray, sr: int, mode: str, input_path: Path) -> np.ndarray:
    """KANONISCHER Produktionspfad: Import → Pre-Analysis → AurikDenker.denke() — §G9 (copilot-instructions.md).

    Exakt die Sequenz aus `cli/aurik_cli.process_audio`:
    run_pre_analysis(...) → get_aurik_denker_instance().denke(...).
    Ergebnis: AurikErgebnis.audio.

    Warum Default: Der Engine-direkt-Pfad (UnifiedRestorerV3.restore ohne
    Pre-Analysis/GlobalPlan) griff am 2026-10-07 in den §9.7.7-Era-Fallback
    (decade=1890 material_prior=wax_cylinder conf 0.42) → §2.46c BW-Hard-Cap
    LPF@5 kHz auf Vinyl — ein Artefakt, das die Produktion (GlobalPlan-Prior)
    so nicht erzeugt. Der Harness MUSS den Weg der Nutzer messen.
    """
    from backend.api.bridge import get_aurik_denker_instance, normalize_user_mode, run_pre_analysis

    if sr != SR:
        raise ValueError(f"Kanonischer Harness erwartet 48 kHz (Korpus), erhielt {sr} Hz")
    pre = run_pre_analysis(
        audio_native=damaged,
        sr_native=sr,
        audio_48k=damaged,
        file_path=str(input_path),
        store_in_bridge_cache=True,
    )
    denker = get_aurik_denker_instance()
    result = denker.denke(
        damaged,
        sr=sr,
        mode=normalize_user_mode(mode),
        no_rt_limit=os.environ.get("AURIK_NO_RT_LIMIT", "0") == "1",
        input_path=str(input_path),
        output_path="",
        pre_analysis_result=pre,
    )
    return np.asarray(result.audio)


def _restore_uv3(damaged: np.ndarray, sr: int, mode: str = "balanced") -> np.ndarray:
    """DIAGNOSE-Pfad — Engine direkt OHNE Pre-Analysis/GlobalPlan.

    NICHT der Produktionspfad (§G9 (copilot-instructions.md))! Ohne GlobalPlan greift der
    §9.7.7-Era-Fallback (Befund 2026-10-07: vinyl → wax_cylinder → LPF@5 kHz).
    Nur für Vergleiche „Engine ohne Denker".
    """
    from backend.api.bridge import get_restorer_classes, normalize_user_mode

    _, UnifiedRestorerV3 = get_restorer_classes()
    result = UnifiedRestorerV3().restore(damaged, sample_rate=sr, mode=normalize_user_mode(mode))
    return np.asarray(getattr(result, "audio", result), dtype=np.float32)


def _restore_coordinated(damaged: np.ndarray, sr: int, medium: str) -> np.ndarray:
    """SOTA-Repair-Kette (bisheriges Benchmark-Verhalten) — Legacy-Diagnose-Pfad.

    Die Legacy-Kette erwartet MONO (der alte Harness kollabierte vorab);
    das Kanal-Mittel ist hier explizit und dokumentiert, die Metrik vergleicht
    dann automatisch Mono↔Mono (_align_pair).
    """
    from backend.core.coordinated_repair import CoordinatedRepair, RepairPlanner
    from backend.core.defect_consensus_pipeline import DefectConsensusPipeline

    _mono = _as_audio_2d(damaged).mean(axis=1)
    manifest = DefectConsensusPipeline().analyze(
        _mono, sr, metadata={"material": medium, "is_digital": medium == "digital"}
    )
    plan = RepairPlanner().plan(manifest, len(_mono))
    restored, _ = CoordinatedRepair().execute(_mono, plan, manifest, sr, material=medium)
    return np.asarray(restored)


def run_benchmark(
    pipeline: str = "canonical",
    files_per_medium: int | None = None,
    medium: str | None = None,
    max_seconds: float = 0.0,
    save_dir: Path | None = None,
    with_mos: bool = False,
) -> BenchmarkReport:
    loop = None
    if with_mos:
        from backend.core.perceptual_closed_loop import PerceptualClosedLoop

        loop = PerceptualClosedLoop()
    _n_files = int(files_per_medium if files_per_medium is not None else FILES_PER_MEDIUM)

    report = BenchmarkReport()
    t0 = time.time()

    _medien = [medium] if medium else ["cassette", "digital", "reel_tape", "shellac", "tape", "vinyl"]
    for medium in _medien:
        pairs = _pair_files(medium)
        if _n_files > 0:
            pairs = pairs[:_n_files]
        if not pairs:
            print(f"\n{medium}: keine Paare gefunden — übersprungen")
            continue
        print(f"\n{'=' * 60}")
        print(f"Medium: {medium} ({len(pairs)} Paare)")
        print(f"{'=' * 60}")

        for damaged_path, clean_path in pairs:
            damaged, sr = _load_audio(damaged_path, max_seconds)
            clean, sr_c = _load_audio(clean_path, max_seconds)
            if sr != sr_c:
                print(f"  ❌ {damaged_path.name}: SR-Mismatch ({sr} vs {sr_c})")
                continue
            damaged, clean = _align_pair(damaged, clean)

            # Metriken VOR der Restauration (gain-angepasst)
            _raw_before, snr_before, _gain_before = _snr_pair(clean, damaged)
            mse_before = float(np.mean((damaged - clean) ** 2))
            mos_damaged = loop.estimate_mos(damaged.mean(axis=1), sr) if loop is not None else 0.0
            mos_clean = loop.estimate_mos(clean.mean(axis=1), sr) if loop is not None else 0.0

            # Restauration über den gewählten Pfad
            t_step = time.time()
            try:
                if pipeline == "canonical":
                    restored = _restore_canonical(damaged, sr, "balanced", damaged_path)
                elif pipeline == "uv3":
                    restored = _restore_uv3(damaged, sr, mode="balanced")
                else:
                    restored = _restore_coordinated(damaged, sr, medium)
                _ref_a, restored = _align_pair(clean, restored)
            except Exception as exc:
                print(f"  ❌ {damaged_path.name}: Kette fehlgeschlagen ({exc})")
                continue
            proc_time = time.time() - t_step

            # Metriken NACH der Restauration (gain-angepasst + roh + Pegel)
            snr_after_raw, snr_after, out_gain = _snr_pair(_ref_a, restored)
            mse_after = float(np.mean((restored - _ref_a) ** 2))
            mos_restored = loop.estimate_mos(restored.mean(axis=1), sr) if loop is not None else 0.0

            snr_improvement = snr_after - snr_before

            # Verdict — nur die gain-angepasste SNR trägt das Urteil
            if snr_improvement > 0.5:
                verdict = "improved"
            elif snr_improvement < -0.5:
                verdict = "degraded"
            else:
                verdict = "neutral"

            out_path_str = ""
            if save_dir is not None:
                _out = Path(save_dir) / medium / f"{damaged_path.stem}_restored.wav"
                _save_wav(_out, restored, sr)
                out_path_str = str(_out)

            result = FileResult(
                medium=medium,
                file=damaged_path.name,
                snr_before_db=snr_before,
                snr_after_db=snr_after,
                snr_after_raw_db=snr_after_raw,
                out_gain_db=out_gain,
                snr_improvement_db=snr_improvement,
                mse_before=mse_before,
                mse_after=mse_after,
                mos_damaged=mos_damaged,
                mos_restored=mos_restored,
                mos_clean=mos_clean,
                verdict=verdict,
                processing_time=proc_time,
                output_path=out_path_str,
            )
            report.results.append(result)

            _mos_txt = (
                f" | MOS {mos_damaged:.2f}→{mos_restored:.2f} (clean {mos_clean:.2f})" if loop is not None else ""
            )
            print(
                f"  {result.file[:42]:42s} SNR(gm) {snr_before:+6.1f}→{snr_after:+6.1f} dB "
                f"({snr_improvement:+5.1f}){_mos_txt} | gain {out_gain:+5.1f} dB | "
                f"raw {snr_after_raw:+6.1f} | {verdict}"
            )

    report.total_time = time.time() - t0
    return report


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Echter-Corpus-Benchmark (clean ↔ damaged ↔ restored)")
    ap.add_argument(
        "--pipeline",
        choices=["canonical", "uv3", "coordinated"],
        default="canonical",
        help="canonical = Produktionspfad Pre-Analysis → AurikDenker.denke() (§G9 (copilot-instructions.md), Default); "
        "uv3 = Engine direkt ohne GlobalPlan (Diagnose, §9.7.7-Era-Fallback-Risiko); "
        "coordinated = SOTA-Repair-Kette (Legacy-Verhalten)",
    )
    ap.add_argument("--files", type=int, default=None, help=f"Dateien je Medium (Default {FILES_PER_MEDIUM}, 0 = alle)")
    ap.add_argument("--medium", default=None, help="Nur EIN Medium (z. B. cassette) — für schnelle Läufe")
    ap.add_argument("--seconds", type=float, default=0.0, help="Nur die ersten N Sekunden je Datei (0 = ganze Datei)")
    ap.add_argument("--save-wav", default=None, help="Verzeichnis: restaurierte WAVs speichern (Diagnose/Hörprobe)")
    ap.add_argument(
        "--mos", action="store_true", help="MOS als Zeugen messen (auf synthetischem Korpus trügerisch und teuer)"
    )
    args = ap.parse_args()

    # §VI (copilot-instructions.md): ML-Device-Manager VOR der ersten Inferenz initialisieren.
    try:
        from backend.api.bridge import get_ml_device_manager

        _mgr = get_ml_device_manager()
        print(f"ML-Device-Manager: {getattr(_mgr, 'device', _mgr)!r}")
    except Exception as exc:  # §V6 (copilot-instructions.md): Warnung statt stiller Degradierung
        print(f"⚠ ML-Device-Manager nicht initialisiert ({exc}) — Lauf mit Defaults")

    _sec_txt = f"je {args.seconds:.0f}s" if args.seconds else "volle Datei"
    print("§v10.800 Echter-Corpus-Benchmark")
    print(
        f"Pfad: {args.pipeline} | Stichprobe: {args.files if args.files is not None else FILES_PER_MEDIUM} "
        f"Datei(en) je Medium, {_sec_txt}"
    )
    report = run_benchmark(
        pipeline=args.pipeline,
        files_per_medium=args.files,
        medium=args.medium,
        max_seconds=args.seconds,
        save_dir=Path(args.save_wav) if args.save_wav else None,
        with_mos=args.mos,
    )

    print(f"\n{'=' * 60}")
    print(f"GESAMTBILANZ (Pfad: {args.pipeline})")
    print(f"{'=' * 60}")
    print(f"Verbessert:   {report.improved}/{len(report.results)}")
    print(f"Neutral:      {sum(1 for r in report.results if r.verdict == 'neutral')}/{len(report.results)}")
    print(f"Verschlechtert: {report.degraded}/{len(report.results)}")
    print(f"Ø SNR-Verbesserung (gain-angepasst): {report.avg_snr_improvement:+.2f} dB")
    print(f"Gesamtzeit: {report.total_time:.0f}s")

    # Per-Medium-Bilanz
    print("\nPer Medium:")
    by_medium: dict[str, list[FileResult]] = {}
    for r in report.results:
        by_medium.setdefault(r.medium, []).append(r)
    for medium, results in sorted(by_medium.items()):
        avg = float(np.mean([r.snr_improvement_db for r in results]))
        improved = sum(1 for r in results if r.verdict == "improved")
        print(f"  {medium:12s}: Ø {avg:+5.2f} dB, {improved}/{len(results)} verbessert")

    # Speichern als JSON
    out = _PROJECT / "benchmarks" / f"corpus_benchmark_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "v10.800",
        "pipeline": args.pipeline,
        "max_seconds": args.seconds,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "summary": {
            "improved": report.improved,
            "degraded": report.degraded,
            "avg_snr_improvement_db": round(report.avg_snr_improvement, 2),
            "total_files": len(report.results),
        },
        "results": [
            {
                "medium": r.medium,
                "file": r.file,
                "snr_before_db": round(r.snr_before_db, 2),
                "snr_after_db": round(r.snr_after_db, 2),
                "snr_after_raw_db": round(r.snr_after_raw_db, 2),
                "out_gain_db": round(r.out_gain_db, 2),
                "snr_improvement_db": round(r.snr_improvement_db, 2),
                "output_path": r.output_path,
                "verdict": r.verdict,
            }
            for r in report.results
        ],
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nBericht: {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
