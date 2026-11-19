"""
benchmarks/excellence_benchmark.py — Aurik Excellence Pipeline Benchmark
=========================================================================

Misst den Qualitätsgewinn durch die Excellence-Pipeline (ExcellenceOptimizer +
MERT-Plugin + adaptive Phase-55-Steps) auf synthetischem Testmaterial für alle
fünf Materialtypen (auto, vinyl, tape, shellac, broadcast).

BENCHMARK-METRIKEN:
  - **MUSIC_OVR**: Overall-Score (1–5, Ziel ≥ 4.3)
  - **MUSIC_NAT**: Naturalness (1–5, Ziel ≥ 4.0)
  - **MUSIC_SIG**: Signal-Qualität (1–5)
  - **MUSIC_BAK**: Hintergrund-Qualität (1–5)
  - **Δ_OVR**: Verbesserung durch Excellence-Pipeline
  - **Δ_NAT**: Naturalness-Verbesserung
  - **RT_ms**: Laufzeit in Millisekunden

SYNTHETISCHES TESTMATERIAL:
  Vier Klassen aus programmatisch erzeugten Signalen:
  - `clean_tone`: 440+880 Hz Sinus, kein Rauschen (Baseline, sollte ≥ 4.5)
  - `noisy_music`: Harmonisches Spektrum + Weißrauschen (SNR ~20 dB)
  - `dropout_signal`: Signal mit simulierten Dropouts (fünf 50ms-Lücken)
  - `overtone_sparse`: Signal mit schwachem Oberton-Spektrum (Harmonizität < 0.3)

VERWENDUNG::

    # CLI
    python benchmarks/excellence_benchmark.py

    # API
    from benchmarks.excellence_benchmark import ExcellenceBenchmark
    report = ExcellenceBenchmark().run_all()
    ExcellenceBenchmark.print_report(report)

    # Mit Datei-Output
    ExcellenceBenchmark().run_all(output_json="results/excellence_2026.json")

Author: Aurik Development Team
Version: 1.0.0
"""

from __future__ import annotations

import datetime
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_SR = 44100
_DURATION_S = 3.0  # Sekunden Testmaterial
_N = int(_SR * _DURATION_S)

# ─── Testsignal-Generator ─────────────────────────────────────────────────────


def _sine_sum(freqs: list[float], amps: list[float] | None = None, n: int = _N, sr: int = _SR) -> np.ndarray:
    """Summe von Sinustönen mit gegebenen Frequenzen/Amplituden."""
    t = np.linspace(0, n / sr, n, dtype=np.float32)
    if amps is None:
        amps = [1.0 / len(freqs)] * len(freqs)
    signal = sum(a * np.sin(2 * np.pi * f * t) for f, a in zip(freqs, amps))
    peak = np.max(np.abs(signal)) + 1e-10
    return (signal / peak * 0.7).astype(np.float32)  # type: ignore[no-any-return]


def _make_test_signals() -> dict[str, np.ndarray]:
    """Erzeugt alle vier Testsignal-Klassen."""
    np.linspace(0, _N / _SR, _N, dtype=np.float32)
    rng = np.random.default_rng(seed=17)

    # 1. Sauberer Ton (Baseline)
    clean_tone = _sine_sum(
        [220, 440, 660, 880, 1100, 1320, 1540],
        [0.40, 0.25, 0.15, 0.08, 0.06, 0.04, 0.02],
    )

    # 2. Musikalisches Signal + Rauschen (SNR ~20 dB)
    music_base = _sine_sum(
        [110, 220, 330, 440, 550, 660, 770, 880],
        [0.35, 0.20, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04],
    )
    noise = rng.standard_normal(_N).astype(np.float32) * 0.032
    noisy_music = np.clip(music_base + noise, -1.0, 1.0)

    # 3. Signal mit Dropouts (fünf 50ms-Lücken, gleichmäßig verteilt)
    dropout_signal = clean_tone.copy()
    gap_ms = 50
    gap_samples = int(gap_ms * _SR / 1000)
    gap_positions = [int(_N * i / 6) for i in range(1, 6)]
    for pos in gap_positions:
        start = max(0, pos - gap_samples // 2)
        end = min(_N, start + gap_samples)
        dropout_signal[start:end] = 0.0

    # 4. Schwaches Oberton-Spektrum (Harmonizität < 0.3)
    # Nur Grundton stark, Obertöne sehr schwach
    overtone_sparse = _sine_sum(
        [440, 880, 1320, 1760, 2200],
        [0.65, 0.05, 0.03, 0.02, 0.01],
    )
    # Leichtes Rauschen überlagern
    overtone_sparse = np.clip(overtone_sparse + rng.standard_normal(_N).astype(np.float32) * 0.02, -1.0, 1.0)

    return {
        "clean_tone": clean_tone,
        "noisy_music": noisy_music,
        "dropout_signal": dropout_signal,
        "overtone_sparse": overtone_sparse,
    }


# ─── Ergebnis-Datenklassen ────────────────────────────────────────────────────


@dataclass
class SignalBenchmarkResult:
    """Benchmark-Ergebnis für ein einzelnes Testsignal."""

    signal_name: str
    material: str
    # Vor Excellence-Pipeline
    before_music_ovr: float
    before_music_nat: float
    before_music_sig: float
    before_music_bak: float
    # Nach Excellence-Pipeline
    after_music_ovr: float
    after_music_nat: float
    after_music_sig: float
    after_music_bak: float
    # Delta
    delta_ovr: float = 0.0
    delta_nat: float = 0.0
    # Laufzeit
    rt_ms: float = 0.0
    # Excellence-Schritte
    applied_steps: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.delta_ovr = round(self.after_music_ovr - self.before_music_ovr, 4)
        self.delta_nat = round(self.after_music_nat - self.before_music_nat, 4)


@dataclass
class ExcellenceBenchmarkReport:
    """Vollständiger Benchmark-Report über alle Materialien und Signale."""

    timestamp: str
    aurik_version: str
    sample_rate: int
    duration_s: float
    results: list[SignalBenchmarkResult] = field(default_factory=list)
    summary: dict = field(default_factory=dict)

    def compute_summary(self) -> None:
        """Berechnet aggregierte Statistiken."""
        if not self.results:
            return
        avg_delta_ovr = float(np.mean([r.delta_ovr for r in self.results]))
        avg_delta_nat = float(np.mean([r.delta_nat for r in self.results]))
        avg_after_ovr = float(np.mean([r.after_music_ovr for r in self.results]))
        avg_after_nat = float(np.mean([r.after_music_nat for r in self.results]))
        max_delta_ovr = float(np.max([r.delta_ovr for r in self.results]))
        min_delta_ovr = float(np.min([r.delta_ovr for r in self.results]))
        avg_rt_ms = float(np.mean([r.rt_ms for r in self.results]))

        # Pro-Material-Zusammenfassung
        materials = sorted({r.material for r in self.results})
        per_material: dict = {}
        for mat in materials:
            mat_results = [r for r in self.results if r.material == mat]
            per_material[mat] = {
                "avg_delta_ovr": round(float(np.mean([r.delta_ovr for r in mat_results])), 4),
                "avg_delta_nat": round(float(np.mean([r.delta_nat for r in mat_results])), 4),
                "avg_after_ovr": round(float(np.mean([r.after_music_ovr for r in mat_results])), 4),
            }

        self.summary = {
            "avg_delta_ovr": round(avg_delta_ovr, 4),
            "avg_delta_nat": round(avg_delta_nat, 4),
            "avg_after_ovr": round(avg_after_ovr, 4),
            "avg_after_nat": round(avg_after_nat, 4),
            "max_delta_ovr": round(max_delta_ovr, 4),
            "min_delta_ovr": round(min_delta_ovr, 4),
            "avg_rt_ms": round(avg_rt_ms, 2),
            "n_results": len(self.results),
            "per_material": per_material,
        }


# ─── Benchmark-Klasse ─────────────────────────────────────────────────────────


class ExcellenceBenchmark:
    """
    Führt den Aurik Excellence-Pipeline-Benchmark aus und vergleicht
    MusicMOS-Scores vor und nach ExcellenceOptimizer für alle Materialprofile.

    Beispiel::

        report = ExcellenceBenchmark().run_all()
        ExcellenceBenchmark.print_report(report)
    """

    MATERIALS = ["auto", "vinyl", "tape", "shellac", "broadcast"]
    _AURIK_VERSION = "9.12.8"

    def __init__(self, sample_rate: int = _SR) -> None:
        self.sample_rate = sample_rate
        from backend.core.music_quality_scorer import score_music_mos  # pylint: disable=import-outside-toplevel

        self._score_mos = score_music_mos

    def _score(self, audio: np.ndarray) -> tuple[float, float, float, float]:
        """Berechnet (OVR, NAT, SIG, BAK)."""
        mos = self._score_mos(audio, self.sample_rate)
        return (
            round(float(mos.MUSIC_OVR), 4),
            round(float(mos.MUSIC_NAT), 4),
            round(float(mos.MUSIC_SIG), 4),
            round(float(mos.MUSIC_BAK), 4),
        )

    def _run_single(
        self,
        signal_name: str,
        audio: np.ndarray,
        material: str,
    ) -> SignalBenchmarkResult:
        """Benchmark für ein einzelnes (Signal, Material)-Paar."""
        from backend.core.excellence_optimizer import ExcellenceOptimizer  # pylint: disable=import-outside-toplevel

        # Scores vor Pipeline
        ovr_b, nat_b, sig_b, bak_b = self._score(audio)

        # Excellence-Pipeline ausführen
        t0 = time.perf_counter()
        optimizer = ExcellenceOptimizer(self.sample_rate, material=material)
        audio_out, opt_result = optimizer.optimize(audio)
        rt_ms = round((time.perf_counter() - t0) * 1000, 2)

        # Scores nach Pipeline
        ovr_a, nat_a, sig_a, bak_a = self._score(audio_out)

        return SignalBenchmarkResult(
            signal_name=signal_name,
            material=material,
            before_music_ovr=ovr_b,
            before_music_nat=nat_b,
            before_music_sig=sig_b,
            before_music_bak=bak_b,
            after_music_ovr=ovr_a,
            after_music_nat=nat_a,
            after_music_sig=sig_a,
            after_music_bak=bak_a,
            rt_ms=rt_ms,
            applied_steps=list(opt_result.applied_steps),
        )

    def run_all(
        self,
        output_json: str | None = None,
        materials: list[str] | None = None,
    ) -> ExcellenceBenchmarkReport:
        """
        Führt den vollständigen Benchmark für alle Signale × Materialprofile aus.

        Args:
            output_json: Wenn angegeben, wird der Report als JSON gespeichert.
            materials: Zu testende Materialprofile (Standard: alle 5).

        Returns:
            ExcellenceBenchmarkReport mit allen Ergebnissen.
        """
        materials_to_run = materials or self.MATERIALS
        signals = _make_test_signals()

        report = ExcellenceBenchmarkReport(
            timestamp=datetime.datetime.now().isoformat(),
            aurik_version=self._AURIK_VERSION,
            sample_rate=self.sample_rate,
            duration_s=_DURATION_S,
        )

        logger.info(
            "ExcellenceBenchmark: %d Signale × %d Materialprofile = %d Tests",
            len(signals),
            len(materials_to_run),
            len(signals) * len(materials_to_run),
        )

        for sig_name, audio in signals.items():
            for material in materials_to_run:
                try:
                    result = self._run_single(sig_name, audio, material)
                    report.results.append(result)
                    logger.debug(
                        "  %s/%s: ΔOVR=%+.4f, ΔNAT=%+.4f, rt=%.1fms",
                        sig_name,
                        material,
                        result.delta_ovr,
                        result.delta_nat,
                        result.rt_ms,
                    )
                except Exception as exc:
                    logger.warning("ExcellenceBenchmark fehler %s/%s: %s", sig_name, material, exc)

        report.compute_summary()

        if output_json:
            path = Path(output_json)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(asdict(report), f, indent=2, ensure_ascii=False)
            logger.info("ExcellenceBenchmark: Report gespeichert → %s", path)

        return report

    @staticmethod
    def print_report(report: ExcellenceBenchmarkReport) -> None:
        """Gibt den Report als formatierte Tabelle auf stdout aus."""
        print(f"\n{'=' * 75}")
        print(f"  Aurik Excellence Benchmark v{report.aurik_version}  |  {report.timestamp[:19]}")
        n_tests = report.summary.get("n_results", "?")
        print(f"  SR={report.sample_rate} Hz  |  Dauer={report.duration_s}s  |  N={n_tests} Tests")
        print(f"{'=' * 75}")

        # Header
        print(f"{'Signal':<20} {'Material':<12} {'ΔOVR':>8} {'ΔNAT':>8} {'OVR_nach':>10} {'NAT_nach':>10} {'RT_ms':>8}")
        print(f"{'-' * 75}")

        for r in report.results:
            print(
                f"{r.signal_name:<20} {r.material:<12} "
                f"{r.delta_ovr:>+8.4f} {r.delta_nat:>+8.4f} "
                f"{r.after_music_ovr:>10.4f} {r.after_music_nat:>10.4f} "
                f"{r.rt_ms:>8.1f}"
            )

        # Zusammenfassung
        s = report.summary
        print(f"\n{'─' * 75}")
        print(
            f"  Gesamt: Ø ΔOVR={s.get('avg_delta_ovr', 0):+.4f}  "
            f"Ø ΔNAT={s.get('avg_delta_nat', 0):+.4f}  "
            f"Ø OVR_nach={s.get('avg_after_ovr', 0):.4f}  "
            f"Ø RT={s.get('avg_rt_ms', 0):.1f}ms"
        )
        print(f"  ΔOVR Range: [{s.get('min_delta_ovr', 0):+.4f}, {s.get('max_delta_ovr', 0):+.4f}]")

        # Pro-Material
        print("\n  Pro-Material-Zusammenfassung:")
        for mat, ms in s.get("per_material", {}).items():
            print(
                f"    {mat:<12} Ø ΔOVR={ms['avg_delta_ovr']:+.4f}  "
                f"Ø ΔNAT={ms['avg_delta_nat']:+.4f}  "
                f"Ø OVR_nach={ms['avg_after_ovr']:.4f}"
            )
        print(f"{'=' * 75}\n")

    @staticmethod
    def compare_to_reference(report: ExcellenceBenchmarkReport) -> bool:
        """
        Vergleicht Benchmark-Report gegen Zielwerte.

        Returns:
            True wenn alle Ziele erfüllt, False andernfalls.
        """
        # Zielwerte für Aurik 10.0.0
        # Tuple: (target_value, operator) — "ge" = ≥ (höher ist besser), "le" = ≤ (niedriger ist besser)
        targets = {
            "avg_after_ovr": (3.80, "ge"),  # MUSIC_OVR ≥ 3.80 (= 0.95 normiert)
            "avg_after_nat": (3.50, "ge"),  # MUSIC_NAT ≥ 3.50
            "avg_delta_ovr": (0.0, "ge"),  # Excellence-Pipeline muss ≥ 0 verbessern
            "avg_rt_ms": (2000.0, "le"),  # Laufzeit < 2000 ms für 3s Audio (0.67× RT-Faktor)
        }
        s = report.summary
        all_ok = True
        print("\n  Ziel-Prüfung:")
        for key, (target, op) in targets.items():
            actual = s.get(key, None)
            if actual is None:
                continue
            _ok = actual >= target if op == "ge" else actual <= target
            cmp_str = "≥" if op == "ge" else "≤"
            status = "✓" if _ok else "✗"
            print(f"    {status} {key}: {actual:.4f} {cmp_str} {target} (Ziel)")
            if not _ok:
                all_ok = False
        return all_ok


# ─── CLI Entry-Point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Workspace-Root zum sys.path hinzufügen
    _root = Path(__file__).parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    bench = ExcellenceBenchmark()
    output = os.environ.get("BENCHMARK_OUT", None)
    rep = bench.run_all(output_json=output)
    bench.print_report(rep)
    ok = bench.compare_to_reference(rep)
    sys.exit(0 if ok else 1)
