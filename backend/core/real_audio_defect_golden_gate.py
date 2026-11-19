"""Real-audio Golden-Set gate for DefectScanner readiness.

The gate is manifest-driven: real audio files stay on disk, annotations live in
JSON, and the scanner output is evaluated by the same explicit recall,
precision, confidence, locality, and runtime gate used by synthetic fixtures.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from backend.core.defect_detection_quality_gate import (
    DefectBenchmarkCaseResult,
    DefectDetectionGateResult,
    DefectDetectionGateThresholds,
    DefectExpectation,
    evaluate_defect_detection_gate,
)
from backend.core.defect_scanner import DefectScanner, DefectType, MaterialType
from backend.file_import import load_audio_file


@dataclass(frozen=True)
class RealAudioGoldenGateReport:
    """Serialisierbares Ergebnis eines Real-Audio-Golden-Set-Gate-Laufs."""

    gate: DefectDetectionGateResult
    cases: list[DefectBenchmarkCaseResult]
    skipped_cases: list[dict[str, str]]
    manifest_path: str
    scanned_cases: int
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        """Gibt a JSON-serializable representation zurück."""
        return {
            "gate": asdict(self.gate),
            "cases": [asdict(case) for case in self.cases],
            "skipped_cases": self.skipped_cases,
            "manifest_path": self.manifest_path,
            "scanned_cases": self.scanned_cases,
            "elapsed_seconds": self.elapsed_seconds,
        }


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Real-audio manifest must be a JSON object: {path}")
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise ValueError("Real-audio manifest requires a 'cases' list")
    return payload


def _thresholds_from_manifest(payload: dict[str, Any]) -> DefectDetectionGateThresholds:
    raw = payload.get("thresholds")
    raw = raw if isinstance(raw, dict) else {}
    return DefectDetectionGateThresholds(
        min_recall=float(raw.get("min_recall", 0.95)),
        min_precision=float(raw.get("min_precision", 0.92)),
        min_mean_confidence=float(raw.get("min_mean_confidence", 0.62)),
        min_locality_recall=float(raw.get("min_locality_recall", 0.90)),
        max_forbidden_severity=float(raw.get("max_forbidden_severity", 0.15)),
        max_runtime_factor=float(raw.get("max_runtime_factor", 1.20)),
    )


def _expectations(raw_expectations: object) -> tuple[DefectExpectation, ...]:
    if not isinstance(raw_expectations, list):
        raise ValueError("Each real-audio case requires an 'expected_defects' list")
    expectations: list[DefectExpectation] = []
    for item in raw_expectations:
        if not isinstance(item, dict):
            raise ValueError("Each expected defect must be a JSON object")
        defect = str(item.get("defect", "") or "").strip()
        if not defect:
            raise ValueError("Expected defect entry is missing 'defect'")
        DefectType(defect)
        expectations.append(
            DefectExpectation(
                defect=defect,
                min_severity=float(item.get("min_severity", 0.10)),
                min_confidence=float(item.get("min_confidence", 0.50)),
                require_locations=bool(item.get("require_locations", False)),
                critical=bool(item.get("critical", True)),
            )
        )
    return tuple(expectations)


def _forbidden(raw_forbidden: object) -> tuple[str, ...]:
    if raw_forbidden is None:
        return ()
    if not isinstance(raw_forbidden, list):
        raise ValueError("'forbidden_defects' must be a list when present")
    defects: list[str] = []
    for name in raw_forbidden:
        defect = str(name or "").strip()
        if defect:
            DefectType(defect)
            defects.append(defect)
    return tuple(defects)


def _audio_from_import(path: Path, target_sr: int) -> tuple[np.ndarray, int, float]:
    loaded = load_audio_file(str(path), target_sr=target_sr, mono=True, do_carrier_analysis=False)
    if not loaded or loaded.get("audio") is None:
        error = loaded.get("error") if isinstance(loaded, dict) else "unknown import error"
        raise RuntimeError(f"Audio import failed for {path}: {error}")
    audio = np.asarray(loaded["audio"], dtype=np.float32)
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1).astype(np.float32)
    sr = int(loaded.get("sr") or target_sr)
    duration = float(loaded.get("duration") or (len(audio) / max(sr, 1)))
    return audio, sr, duration


def _scan_manifest_case(raw_case: dict[str, Any], repo_root: Path, target_sr: int) -> DefectBenchmarkCaseResult:
    case_id = str(raw_case.get("case_id", "") or "").strip()
    if not case_id:
        raise ValueError("Real-audio case is missing 'case_id'")
    rel_path = str(raw_case.get("path", "") or "").strip()
    if not rel_path:
        raise ValueError(f"Real-audio case {case_id} is missing 'path'")

    material = MaterialType(str(raw_case.get("material_type", "unknown") or "unknown"))
    path = repo_root / rel_path
    if not path.exists():
        raise FileNotFoundError(str(path))

    audio, sr, imported_duration = _audio_from_import(path, target_sr=target_sr)
    max_seconds = float(raw_case.get("max_seconds", 30.0))
    if max_seconds > 0.0:
        audio = audio[: min(len(audio), int(sr * max_seconds))]

    scanner = DefectScanner(sample_rate=sr)
    result = scanner.scan(audio, sr, material_type=material, file_ext=path.suffix)

    severities: dict[str, float] = {}
    confidences: dict[str, float] = {}
    locations: dict[str, list[tuple[float, float]]] = {}
    for defect_type, score in result.scores.items():
        key = defect_type.value if isinstance(defect_type, DefectType) else str(defect_type)
        severities[key] = float(score.severity)
        confidences[key] = float(score.confidence)
        if score.locations:
            locations[key] = [(float(start), float(end)) for start, end in score.locations]

    return DefectBenchmarkCaseResult(
        case_id=case_id,
        expected=_expectations(raw_case.get("expected_defects")),
        forbidden_defects=_forbidden(raw_case.get("forbidden_defects")),
        severities=severities,
        confidences=confidences,
        locations=locations,
        runtime_seconds=float(result.analysis_time_seconds),
        duration_seconds=float(result.duration_seconds),
        metadata={
            "path": rel_path,
            "material": material.value,
            "imported_duration_seconds": imported_duration,
            "scan_duration_seconds": float(result.duration_seconds),
            "sample_rate": int(result.sample_rate),
            "description": str(raw_case.get("description", "") or ""),
        },
    )


def run_real_audio_defect_golden_gate(
    *,
    manifest_path: Path,
    repo_root: Path | None = None,
    allow_missing: bool = False,
    allow_empty: bool = False,
) -> RealAudioGoldenGateReport:
    """Führt aus: the manifest-driven real-audio defect Golden-Set gate."""
    start_time = time.time()
    manifest_path = manifest_path.resolve()
    repo_root = (repo_root or manifest_path.parents[1]).resolve()
    payload = _load_manifest(manifest_path)
    thresholds = _thresholds_from_manifest(payload)
    target_sr = int(payload.get("target_sample_rate", 48_000))

    scanned_cases: list[DefectBenchmarkCaseResult] = []
    skipped_cases: list[dict[str, str]] = []
    for raw_case in payload["cases"]:
        if not isinstance(raw_case, dict):
            raise ValueError("Every real-audio manifest case must be an object")
        case_id = str(raw_case.get("case_id", "") or "unknown")
        if raw_case.get("active", True) is False:
            skipped_cases.append({"case_id": case_id, "reason": "inactive"})
            continue
        try:
            scanned_cases.append(_scan_manifest_case(raw_case, repo_root, target_sr))
        except FileNotFoundError as exc:
            if not allow_missing:
                raise
            skipped_cases.append({"case_id": case_id, "reason": f"missing_file:{exc}"})

    if not scanned_cases and not allow_empty:
        raise RuntimeError("Real-audio Golden-Set gate has no scanned active cases")

    gate = evaluate_defect_detection_gate(scanned_cases, thresholds)
    elapsed = float(time.time() - start_time)
    return RealAudioGoldenGateReport(
        gate=gate,
        cases=scanned_cases,
        skipped_cases=skipped_cases,
        manifest_path=str(manifest_path),
        scanned_cases=len(scanned_cases),
        elapsed_seconds=elapsed,
    )


__all__ = ["RealAudioGoldenGateReport", "run_real_audio_defect_golden_gate"]
