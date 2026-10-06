#!/usr/bin/env python3
"""§F13 — Musik-trainiertes Gender-Head auf MERT-Features.

Warum: Die einzige musiktaugliche Gender-Evidenz der kanonischen Fusion (Spec 19)
ist heute die PANNs-Klassen-Abfrage (AudioSet „Male/Female singing"). Ein Head auf
MERT-v1-330M-Features (auf 160k+ Stunden Musik vortrainiert, ``mert_feature_extractor``)
liefert eine kontinuierliche, kalibrierbare Evidenz und kann später
`_panns_singing_scores` in der Fusion ergänzen.

Labels sind PFLICHT
-------------------
Ohne gelabelten Musik-Korpus wird NICHT trainiert — das Skript bricht dann ab,
statt zu raten (§V7 copilot-instructions.md). Es erfindet weder Labels noch
Schwellen.

Determinismus (§G5 GEBOTE.md)
-----------------------------
Fester Seed, kein zeitabhängiger Split, kein ``time.time()`` in Entscheidungen.

Checkpoint-Schutz
-----------------
``backend/core/training_artifacts.save_guarded`` — rotierende Backups und
atomares Schreiben. Ein abgebrochener oder degenerierter Lauf kann bestehende
Artefakte damit nicht still überschreiben (Befund 2026-10-06).

Usage:
    python scripts/train_gender_head.py \\
        --labels data/gender_labels.csv --audio-dir data/music \\
        --out models/gender_head --epochs 200

Labels-CSV: Kopfzeile ``file,gender`` mit ``gender`` ∈ {male, female, child}.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)

_LABELS: tuple[str, ...] = ("male", "female", "child")
_FEATURE_DIM = 2048  # MERT 1024 (mean) + 1024 (std)
_SEED = 42  # §G5 GEBOTE.md — deterministisch
_CACHE_SUBDIR = ".feature_cache"


def _read_labels(csv_path: Path) -> list[tuple[str, str]]:
    """Liest `file,gender`-Paare und validiert die Labels streng."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Label-CSV fehlt: {csv_path}")
    _pairs: list[tuple[str, str]] = []
    with csv_path.open(newline="", encoding="utf-8") as _fh:
        _reader = csv.DictReader(_fh)
        if "file" not in (_reader.fieldnames or []) or "gender" not in (_reader.fieldnames or []):
            raise ValueError(f"Label-CSV braucht die Spalten 'file,gender' — gefunden: {_reader.fieldnames}")
        for _row in _reader:
            _name = str(_row.get("file", "")).strip()
            _gender = str(_row.get("gender", "")).strip().lower()
            if not _name:
                continue
            if _gender not in _LABELS:
                raise ValueError(f"Ungültiges Label '{_gender}' für {_name} — erlaubt: {_LABELS}")
            _pairs.append((_name, _gender))
    if not _pairs:
        raise ValueError("Label-CSV enthält keine gültigen Zeilen")
    return _pairs


def _load_mono(path: Path, target_sr: int = 48000) -> np.ndarray | None:
    """Lädt Audio als float32-Mono (None bei Fehlern, §V6 copilot-instructions.md-Log)."""
    try:
        import soundfile as sf

        _data, _sr = sf.read(str(path), always_2d=True, dtype="float32")
        _arr = np.asarray(_data, dtype=np.float32)
        _mono = np.mean(_arr, axis=1) if _arr.ndim == 2 else _arr
        if int(_sr) != target_sr:
            from scipy.signal import resample_poly

            _g = np.gcd(int(_sr), int(target_sr))
            _mono = resample_poly(_mono, target_sr // _g, int(_sr) // _g).astype(np.float32)
        return np.nan_to_num(_mono, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception as _exc:  # pylint: disable=broad-except
        logger.warning("Audio nicht lesbar (%s) — Paar übersprungen (%s)", path.name, _exc)
        return None


def _pool_mert(features: np.ndarray) -> np.ndarray:
    """Poolt MERT-Frames [T, 1024] zu einem festen Vektor [2048] (Mittel + Streuung)."""
    _f = np.asarray(features, dtype=np.float32)
    if _f.ndim != 2 or _f.size == 0:
        return np.zeros(_FEATURE_DIM, dtype=np.float32)
    _mean = np.mean(_f, axis=0)
    _std = np.std(_f, axis=0)
    _vec = np.concatenate([_mean, _std]).astype(np.float32)
    return np.nan_to_num(_vec, nan=0.0, posinf=0.0, neginf=0.0)


def _extract_dataset(
    audio_dir: Path, pairs: list[tuple[str, str]], cache_dir: Path
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extrahiert MERT-Features je Paar; Cache pro Datei (deterministisch).

    Der Cache ist bewusst datei-basiert und wird nur bei fehlendem/kaputtem
    Inhalt neu berechnet — die Extraktion ist der teuerste Schritt.
    """
    from backend.core.mert_feature_extractor import MERTFeatureExtractor

    cache_dir.mkdir(parents=True, exist_ok=True)
    _extractor: MERTFeatureExtractor | None = None
    _labels_seen: list[str] = []

    _feats: list[np.ndarray] = []
    _targets: list[str] = []
    for _name, _gender in pairs:
        _path = audio_dir / _name
        if not _path.exists():
            logger.warning("Audiodatei fehlt (%s) — Paar übersprungen", _path)
            continue
        _cache_file = cache_dir / f"{_path.stem}.npy"
        _vec: np.ndarray | None = None
        if _cache_file.exists():
            try:
                _cached = np.load(_cache_file)
                if _cached.shape == (_FEATURE_DIM,):
                    _vec = np.asarray(_cached, dtype=np.float32)
            except Exception as _exc:  # pylint: disable=broad-except
                logger.debug("Feature-Cache unlesbar (%s) — wird neu berechnet: %s", _cache_file.name, _exc)
        if _vec is None:
            _mono = _load_mono(_path)
            if _mono is None or _mono.size < 24000:
                logger.warning("Audio zu kurz/unlesbar (%s) — Paar übersprungen", _path.name)
                continue
            if _extractor is None:
                _extractor = MERTFeatureExtractor()
            try:
                _vec = _pool_mert(_extractor.extract(_mono, 48000))
            except Exception as _exc:  # pylint: disable=broad-except
                logger.warning("MERT-Extraktion fehlgeschlagen (%s) — Paar übersprungen: %s", _path.name, _exc)
                continue
            np.save(_cache_file, _vec)
        _feats.append(_vec)
        _targets.append(_gender)
        _labels_seen.append(_gender)

    if not _feats:
        return np.zeros((0, _FEATURE_DIM), dtype=np.float32), np.zeros((0,), dtype=np.int64), _labels_seen
    return np.stack(_feats), np.asarray([_LABELS.index(t) for t in _targets], dtype=np.int64), _labels_seen


def _stratified_split(y: np.ndarray, val_ratio: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    """Deterministischer, stratifizierter Split (§G5 GEBOTE.md)."""
    _rng = np.random.RandomState(_SEED)
    _train_idx: list[int] = []
    _val_idx: list[int] = []
    for _cls in np.unique(y):
        _idx = np.where(y == _cls)[0]
        _rng.shuffle(_idx)
        _n_val = max(1, int(round(len(_idx) * val_ratio))) if len(_idx) > 1 else 0
        _val_idx.extend(_idx[:_n_val].tolist())
        _train_idx.extend(_idx[_n_val:].tolist())
    return np.asarray(sorted(_train_idx), dtype=np.int64), np.asarray(sorted(_val_idx), dtype=np.int64)


def _train_head(x: np.ndarray, y: np.ndarray, epochs: int) -> tuple[object, dict]:
    """Trainiert ein kleines MLP auf den MERT-Features (deterministisch)."""
    import torch
    from torch import nn

    torch.manual_seed(_SEED)
    np.random.seed(_SEED)

    _tr, _va = _stratified_split(y)
    if _tr.size == 0:
        raise ValueError("Kein Trainingssplit möglich — zu wenige Paare")
    _x_tr = torch.from_numpy(x[_tr])
    _y_tr = torch.from_numpy(y[_tr])
    _x_va = torch.from_numpy(x[_va]) if _va.size else None
    _y_va = torch.from_numpy(y[_va]) if _va.size else None

    _model = nn.Sequential(
        nn.Linear(_FEATURE_DIM, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, len(_LABELS)),
    )
    _opt = torch.optim.Adam(_model.parameters(), lr=1e-3)
    _loss_fn = nn.CrossEntropyLoss()

    _model.train()
    for _epoch in range(int(epochs)):
        _opt.zero_grad()
        _logits = _model(_x_tr)
        _loss = _loss_fn(_logits, _y_tr)
        _loss.backward()
        _opt.step()
        if (_epoch + 1) % max(int(epochs) // 5, 1) == 0:
            logger.info("Epoche %d/%d — Loss %.4f", _epoch + 1, int(epochs), float(_loss.item()))

    _metrics: dict = {"train_n": int(_tr.size), "val_n": int(_va.size), "epochs": int(epochs), "seed": _SEED}
    if _x_va is not None and _y_va is not None and _y_va.numel() > 0:
        _model.eval()
        with torch.no_grad():
            _pred = torch.argmax(_model(_x_va), dim=1)
            _acc = float((_pred == _y_va).float().mean().item())
            _f1 = _macro_f1(_y_va.numpy(), _pred.numpy(), len(_LABELS))
        _metrics["val_accuracy"] = round(_acc, 4)
        _metrics["val_macro_f1"] = round(_f1, 4)
    return _model, _metrics


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Macro-F1 ohne sklearn-Abhängigkeit."""
    _f1s: list[float] = []
    for _c in range(n_classes):
        _tp = float(np.sum((y_pred == _c) & (y_true == _c)))
        _fp = float(np.sum((y_pred == _c) & (y_true != _c)))
        _fn = float(np.sum((y_pred != _c) & (y_true == _c)))
        if _tp + _fp + _fn == 0:
            continue
        _prec = _tp / (_tp + _fp) if (_tp + _fp) else 0.0
        _rec = _tp / (_tp + _fn) if (_tp + _fn) else 0.0
        _f1s.append(0.0 if (_prec + _rec) == 0 else 2 * _prec * _rec / (_prec + _rec))
    return float(np.mean(_f1s)) if _f1s else 0.0


def main(argv: list[str] | None = None) -> int:
    """Extrahiert MERT-Features, trainiert das Head und speichert guarded."""
    from backend.core.training_artifacts import save_guarded, write_text_guarded

    _ap = argparse.ArgumentParser(description="Gender-Head-Finetune auf MERT-Features (F13)")
    _ap.add_argument("--labels", type=Path, required=True, help="CSV mit 'file,gender' (PFLICHT)")
    _ap.add_argument("--audio-dir", type=Path, required=True, help="Verzeichnis der Audiodateien")
    _ap.add_argument("--out", type=Path, default=Path("models/gender_head"), help="Ausgabe-Verzeichnis")
    _ap.add_argument("--epochs", type=int, default=200, help="Trainings-Epochen")
    _args = _ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        _pairs = _read_labels(_args.labels)
    except (FileNotFoundError, ValueError) as _exc:
        logger.error("Abbruch ohne Training: %s", _exc)
        return 2

    logger.info("F13: %d gelabelte Paare — Extraktion startet", len(_pairs))
    _x, _y, _seen = _extract_dataset(_args.audio_dir, _pairs, _args.out / _CACHE_SUBDIR)
    if _x.shape[0] < 4:
        logger.error("Zu wenige verwertbare Paare (%d) — kein Training", _x.shape[0])
        return 2
    _missing = sorted(set(_LABELS) - set(_seen))
    if len(set(_seen)) < 2:
        logger.error("Nur eine Klasse im Datensatz (%s) — Training sinnlos", sorted(set(_seen)))
        return 2
    if _missing:
        logger.warning("Klassen ohne Beispiele: %s — Head lernt sie nicht (Hinweis, kein Fehler)", _missing)

    _model, _metrics = _train_head(_x, _y, int(_args.epochs))
    _payload = {
        "state_dict": _model.state_dict(),
        "labels": list(_LABELS),
        "feature_dim": _FEATURE_DIM,
        "feature_source": "MERT-v1-330M (mean+std, 2x1024)",
        "metrics": _metrics,
    }
    _ckpt = _args.out / "gender_head.pt"
    save_guarded(_ckpt, _payload)
    write_text_guarded(
        _args.out / "train_report_gender_head.json",
        _json_dump({"checkpoint": str(_ckpt), "metrics": _metrics, "samples": int(_x.shape[0])}),
    )
    logger.info("F13 fertig: %s (Metriken: %s)", _ckpt, _metrics)
    return 0


def _json_dump(payload: dict) -> str:
    """JSON-Text mit deutschem Encoding."""
    import json

    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


if __name__ == "__main__":
    sys.exit(main())
