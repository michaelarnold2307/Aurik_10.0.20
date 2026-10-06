#!/usr/bin/env python3
"""§F14 — Evidenz-Kalibrierung der Gender-Detektion auf echtem Musik-Korpus.

Warum: Die Schwellen der kanonischen Gender-Fusion (Spec 19) stammen aus
Anatomie-Literatur und Modell-Konventionen (Titze 1994, formant_ranges).
Sie sollen am REALEN Material geprüft werden — ohne erfundene Labels.

Ehrlichkeits-Vertrag
--------------------
Das Korpus-Manifest (``corpus/MANIFEST_SCHEMA.yaml``) kennt **kein** Gender-Feld;
es existiert also keine Ground Truth. Dieses Skript kalibriert deshalb nichts
blind, sondern weist die ÜBEREINSTIMMUNG der unabhängigen Evidenzquellen aus:

1. **Anatomie** — Gender-Lesart allein aus F0 + Formanten (Spec 19).
2. **PANNs** — AudioSet „Male/Female singing" (musiktaugliche ML-Evidenz),
   sofern Tags vorliegen (``--panns-tags``).

Stimmen beide in einer ausreichend großen Stichprobe häufig überein, ist das ein
belastbares Maß für die Verlässlichkeit der Anatomie-Pfade; weichen sie
systematisch ab, sind die Schwellen nachweislich zu justieren. Eine
Schwellenempfehlung wird NUR bei ausreichendem Konsens UND ausreichender
Stichprobe ausgegeben und im Report als „empirisch, kein Ground Truth" markiert.

Determinismus (§G5 GEBOTE.md): feste Seeds, keine zeitabhängigen Entscheidungen.

Usage:
    python scripts/calibrate_gender_evidence.py \\
        --corpus corpus --out reports/gender_calibration --limit 40
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)

_AUDIO_EXT = {".wav", ".flac", ".mp3", ".aiff", ".aif", ".ogg", ".m4a"}
_CONSENSUS_MIN_SAMPLES = 12
_CONSENSUS_MIN_RATE = 0.80


def _iter_audio(corpus_dir: Path) -> list[Path]:
    """Sammelt Audiodateien rekursiv (defekte Originale UND restaurierte)."""
    if not corpus_dir.exists():
        return []
    return sorted(p for p in corpus_dir.rglob("*") if p.suffix.lower() in _AUDIO_EXT)


def _load_mono(path: Path, target_sr: int = 48000) -> tuple[np.ndarray, int] | None:
    """Lädt Audio als float32-Mono; None bei Fehlern (§V6 copilot-instructions.md-Log)."""
    try:
        import soundfile as sf

        data, sr = sf.read(str(path), always_2d=True, dtype="float32")
        _arr = np.asarray(data, dtype=np.float32)
        mono = np.mean(_arr, axis=1) if _arr.ndim == 2 else _arr

        if int(sr) != target_sr:
            from scipy.signal import resample_poly

            _g = np.gcd(int(sr), int(target_sr))
            mono = resample_poly(mono, target_sr // _g, int(sr) // _g).astype(np.float32)
            sr = target_sr
        return np.nan_to_num(mono, nan=0.0, posinf=0.0, neginf=0.0), int(sr)
    except Exception as _exc:  # pylint: disable=broad-except
        logger.warning("Datei nicht lesbar (%s) — übersprungen (%s)", path.name, _exc)
        return None


def _measure_one(mono: np.ndarray, sr: int, panns_tags: dict | None) -> dict:
    """Misst die Evidenzen EINER Datei über den kanonischen Kern (kein Parallelpfad)."""
    from backend.core.vocal_ai_enhancement import GenderDetector, VoiceGender

    _det = GenderDetector(sample_rate=sr)
    _chars = _det.detect(mono, panns_tags=panns_tags)
    # Anatomie-Lesart OHNE ML-Evidenz: nur F0 + Formanten (Spec 19 Wurzelpfad).
    _gender_anatomy, _conf_anatomy = _det._classify_gender(  # pylint: disable=protected-access
        float(_chars.fundamental_freq), list(_chars.formants)
    )
    _ape = float(_det._detect_breathiness(mono))  # pylint: disable=protected-access
    return {
        "gender_fusion": str(getattr(_chars.gender, "value", "unknown")),
        "confidence": float(getattr(_chars, "confidence", 0.0) or 0.0),
        "gender_anatomy": str(getattr(_gender_anatomy, "value", "unknown"))
        if _gender_anatomy != VoiceGender.UNKNOWN
        else "unknown",
        "confidence_anatomy": float(_conf_anatomy or 0.0),
        "f0_hz": float(getattr(_chars, "fundamental_freq", 0.0) or 0.0),
        "formant_count": len(list(getattr(_chars, "formants", []) or [])),
        "aperiodicity": _ape,
        "panns_available": panns_tags is not None,
    }


def _load_panns_tags(path: Path | None) -> dict[str, dict] | None:
    """Lädt optionale PANNs-Tags (JSON: {dateiname: {"Male singing": x, ...}})."""
    if path is None:
        return None
    if not path.exists():
        logger.warning("PANNs-Tags-Datei fehlt (%s) — Kalibrierung nur über Anatomie", path)
        return None
    try:
        _raw = json.loads(path.read_text(encoding="utf-8"))
        return _raw if isinstance(_raw, dict) else None
    except Exception as _exc:  # pylint: disable=broad-except
        logger.warning("PANNs-Tags nicht lesbar (%s) — nur Anatomie (§V6 copilot-instructions.md)", _exc)
        return None


def _panns_tags_for(mono: np.ndarray, sr: int) -> dict | None:
    """Erzeugt die PANNs-Singing-Tags über das Plugin (Array-API, kein Temp-WAV).

    Gibt ``None`` zurück, wenn das Modell nicht verfügbar ist — der Aufrufer misst
    dann ausschließlich die Anatomie. Das ist ausdrücklich KEIN stilles Degradieren:
    Der Report weist „ohne ML-Evidenz" aus, und die Schwellenempfehlung bleibt
    gesperrt (§V6 copilot-instructions.md, Konsens-Gate).
    """
    try:
        from plugins.panns_plugin import classify_audio  # pylint: disable=import-outside-toplevel

        _tags = classify_audio(mono, sr)
        if isinstance(_tags, dict) and ("Male singing" in _tags or "Female singing" in _tags):
            return _tags
        logger.debug("PANNs lieferte keine Singing-Klassen — Messung ohne ML-Evidenz")
        return None
    except Exception as _exc:  # pylint: disable=broad-except
        logger.warning("PANNs-Tags nicht erzeugbar (%s) — Messung ohne ML-Evidenz (§V6 copilot-instructions.md)", _exc)
        return None


def _build_report(records: list[dict], min_samples: int) -> dict:
    """Erzeugt Kennzahlen und eine ehrliche Kalibrier- Empfehlung."""
    _n = len(records)
    _f0 = np.array([r["f0_hz"] for r in records], dtype=np.float64) if _n else np.array([])
    _conf = np.array([r["confidence"] for r in records], dtype=np.float64) if _n else np.array([])
    _ape = np.array([r["aperiodicity"] for r in records], dtype=np.float64) if _n else np.array([])

    _with_panns = [r for r in records if r["panns_available"]]
    _agree = [r for r in _with_panns if r["gender_fusion"] == r["gender_anatomy"]]
    _consensus_rate = (len(_agree) / len(_with_panns)) if _with_panns else None

    _recommendable = bool(
        _consensus_rate is not None and len(_with_panns) >= min_samples and _consensus_rate >= _CONSENSUS_MIN_RATE
    )
    return {
        "samples": _n,
        "samples_with_panns": len(_with_panns),
        "consensus_rate": None if _consensus_rate is None else round(float(_consensus_rate), 4),
        "consensus_min_samples": min_samples,
        "consensus_min_rate": _CONSENSUS_MIN_RATE,
        "threshold_recommendation_allowed": _recommendable,
        "distribution": {
            "f0_hz": _quantiles(_f0),
            "confidence": _quantiles(_conf),
            "aperiodicity": _quantiles(_ape),
        },
        "gender_counts": {
            _g: sum(1 for r in records if r["gender_fusion"] == _g)
            for _g in sorted({str(r["gender_fusion"]) for r in records})
        },
        "caveat": (
            "KEIN Ground Truth: Das Korpus-Manifest kennt kein Gender-Feld. Die Zahlen sind "
            "empirische Evidenz-Verteilungen, keine Validierung gegen Labels."
        ),
    }


def _quantiles(values: np.ndarray) -> dict:
    """Robuste Quantile (nicht-endliche Werte werden verworfen)."""
    _v = values[np.isfinite(values)] if values.size else values
    if _v.size == 0:
        return {"n": 0}
    return {
        "n": int(_v.size),
        "p10": round(float(np.percentile(_v, 10)), 4),
        "p50": round(float(np.percentile(_v, 50)), 4),
        "p90": round(float(np.percentile(_v, 90)), 4),
        "std": round(float(np.std(_v)), 4),
    }


def _write_reports(out_dir: Path, payload: dict, records: list[dict]) -> None:
    """Schreibt JSON und Markdown (Verzeichnis wird angelegt)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "gender_evidence_calibration.json").write_text(
        json.dumps({"summary": payload, "records": records}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _lines = [
        "# Gender-Evidenz-Kalibrierung (F14)",
        "",
        f"- Stichprobe: **{payload['samples']}** Dateien (davon {payload['samples_with_panns']} mit PANNs-Evidenz)",
        f"- Konsens Anatomie ↔ PANNs: {'n/a' if payload['consensus_rate'] is None else payload['consensus_rate']}",
        f"- Schwellenempfehlung erlaubt: **{payload['threshold_recommendation_allowed']}**",
        f"  (Mindest-Stichprobe {payload['consensus_min_samples']}, Mindest-Konsens {payload['consensus_min_rate']})",
        "",
        f"> {payload['caveat']}",
        "",
        "## Verteilungen",
        "",
    ]
    for _key, _q in payload["distribution"].items():
        _lines.append(f"- `{_key}`: {_q}")
    (out_dir / "gender_evidence_calibration.md").write_text("\n".join(_lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Sammelt Evidenzen, berechnet Konsens und schreibt die Reports."""
    _ap = argparse.ArgumentParser(description="Gender-Evidenz-Kalibrierung (F14)")
    _ap.add_argument("--corpus", type=Path, default=Path("corpus"), help="Korpus-Wurzel")
    _ap.add_argument("--out", type=Path, default=Path("reports/gender_calibration"), help="Ausgabe-Verzeichnis")
    _ap.add_argument("--limit", type=int, default=0, help="Maximale Dateianzahl (0 = alle)")
    _ap.add_argument(
        "--panns",
        default="auto",
        help="PANNs-Singing-Evidenz: 'auto' (selbst erzeugen), 'off' oder Pfad zu einer Tags-JSON",
    )
    _ap.add_argument(
        "--panns-out",
        type=Path,
        default=None,
        help="Erzeugte PANNs-Tags als JSON sichern (Cache für Wiederholungsläufe)",
    )
    _ap.add_argument("--min-samples", type=int, default=_CONSENSUS_MIN_SAMPLES, help="Mindest-Stichprobe mit PANNs")
    _args = _ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    _files = _iter_audio(_args.corpus)
    if _args.limit > 0:
        _files = _files[: _args.limit]
    if not _files:
        logger.warning("Keine Audiodateien unter %s gefunden — nichts zu messen", _args.corpus)
        return 2

    _tags_map: dict[str, dict] | None = None
    _panns_mode = str(_args.panns).strip().lower()
    _panns_external = _panns_mode not in {"auto", "off"}
    if _panns_external:
        _tags_map = _load_panns_tags(Path(_args.panns))
    elif _panns_mode == "off":
        logger.info("PANNs-Evidenz abgeschaltet (--panns off) — nur Anatomie-Verteilung")
    _generated_tags: dict[str, dict] = {}
    _records: list[dict] = []
    for _path in _files:
        _loaded = _load_mono(_path)
        if _loaded is None:
            continue
        _mono, _sr = _loaded
        if _mono.size < _sr // 2:
            logger.debug("Datei zu kurz (%s) — übersprungen", _path.name)
            continue
        if _panns_external:
            _tags = _tags_map.get(_path.name) if _tags_map else None
        elif _panns_mode == "off":
            _tags = None
        else:
            _tags = _panns_tags_for(_mono, _sr)
            if _tags is not None:
                _generated_tags[_path.name] = _tags
        try:
            _rec = _measure_one(_mono, _sr, _tags)
        except Exception as _exc:  # pylint: disable=broad-except
            logger.warning(
                "Messung fehlgeschlagen (%s) — übersprungen (§V6 copilot-instructions.md): %s", _path.name, _exc
            )
            continue
        _rec["file"] = _path.name
        _rec["material"] = _path.parent.name
        _records.append(_rec)

    if not _records:
        logger.warning("Keine auswertbaren Messungen erzeugt")
        return 2

    if _args.panns_out is not None and _generated_tags:
        _args.panns_out.parent.mkdir(parents=True, exist_ok=True)
        _args.panns_out.write_text(json.dumps(_generated_tags, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        logger.info("PANNs-Tags gesichert: %s (%d Dateien)", _args.panns_out, len(_generated_tags))

    _payload = _build_report(_records, int(_args.min_samples))
    _payload["panns_source"] = "external" if _panns_external else ("off" if _panns_mode == "off" else "auto")
    _write_reports(_args.out, _payload, _records)
    logger.info(
        "F14 fertig: %d Dateien, Konsens=%s, Empfehlung erlaubt=%s → %s",
        _payload["samples"],
        _payload["consensus_rate"],
        _payload["threshold_recommendation_allowed"],
        _args.out,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
