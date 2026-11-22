"""scripts/train_witness_detektor.py — supervidiertes Schadens-Witness je Q≈0-Klasse.

SOTA-Antwort auf die Metrik-Blindheit der §G13-Q≈0-Klassen (Handover Punkte
25/26/28/29): handgefertigte gain-invariante Features sahen die Schäden nicht
(Q0-Witness-Forschung 2026-11-14: groove_echo Δ ≈ 0 für Cepstral-Echo/Ripple
etc.). Ein GELERNTER Detektor (Produktions-Mel ``compute_mel_produktion`` +
CRNN im Muster ``backend/models/de_crackle/train_defect_crnn.py``) separiert
clean/schaden dagegen messbar (Pilot 2026-11-14: groove_echo Holdout-AUC
0,993, sticky_shed 0,979).

Vertrag:
  * Leak-frei: Training auf 2 k398-Titeln, Holdout auf dem dritten (dem mit
    den meisten Regionen) — das Holdout-Audio ist nie im Training.
  * Input: Produktions-Mel (D-K3-96-Mel-Wahrheit), fest auf 64 Frames
    gepolstert (min-Padding, Muster Defekt-CRNN).
  * CPU-only: ``torch.device("cpu")`` hart verdrahtet — die GPU gehört der
    Trainingskette (§5 Handover). Kein CUDA-Aufruf im Modul (Test-gepinnt).
  * Deterministisch: Seed 42 (§G5 (copilot-instructions.md)).
  * Gate je Klasse: Holdout-AUC ≥ 0,80 UND Richtung P(schaden) > P(clean).
  * Report-only-Witness: der Detektor ist Zeuge (§G191 (GEBOTE.md)), kein
    Gate, kein Flag-Flip, kein Audio-Eingriff.

Checkpoint je Klasse: ``models/witness_<klasse>/witness.pt`` (state_dict +
``n_mels``), Beleg ``models/witness_<klasse>/report.json``
(``aurik.witness_training/1``). Fingerprints über
``scripts/model_artifact_probe.py`` (§III.13 (copilot-instructions.md)).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import date
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from plugins.bigvgan_v2_plugin import compute_mel_produktion

MANIFEST = _ROOT / "data" / "klassen_paare" / "manifest.jsonl"
MODELS_DIR = _ROOT / "models"
SEED = 42
FRAMES = 64
BATCH = 64
EPOCHS = 20
LR = 1e-3
AUC_ZIEL = 0.80
SR = 44100
SCHEMA = "aurik.witness_training/1"


class _BestCheckpoint(TypedDict):
    auc: float
    epoch: int
    state: dict[str, torch.Tensor] | None


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _mel_fest(audio: np.ndarray, sr: int) -> np.ndarray:
    m = compute_mel_produktion(audio, sr)
    if m.shape[1] >= FRAMES:
        m = m[:, :FRAMES]
    else:
        pad = np.full((m.shape[0], FRAMES - m.shape[1]), float(m.min()), dtype=np.float32)
        m = np.concatenate([m, pad], axis=1)
    return m.astype(np.float32)


def _mel_fest_abs(audio: np.ndarray, sr: int) -> np.ndarray:
    """Absolute Mel (ref=1.0) — identische Formel zur Produktions-Inferenz `_mel_torch`.

    D-K3-95-Nacharbeit 2026-11-22, Befund: `compute_mel_produktion` normiert je
    Fenster auf das Maximum (`power_to_db(ref=np.max)`) und löscht damit die
    absolute Pegelinformation — genau die Dimension, in der dropout_oxide
    (+17,4 dB Kontrast), tape_splice (+4,6 dB) und dropouts (+13,5 dB) leben.
    Der Witness wurde dadurch auf einer pegel-blinden Repräsentation trainiert,
    während die Produktion (`_mel_torch` in train_witness_kopf/witness_kopf_*)
    ABSOLUT misst — dieselbe Mel-Drift-Klasse wie der BigVGAN-Befund D-K3-96.
    Diese Variante schließt die Drift: absolute 10·log10(|Mel|²), Slaney,
    2048/512, 128 Bänder, Floor −100 dB — Formel-identisch zu `_mel_torch`.
    """
    import librosa

    a = np.asarray(audio, dtype=np.float32)
    if sr != 44100:
        a = librosa.resample(a, orig_sr=sr, target_sr=44100).astype(np.float32)
        sr = 44100
    m = librosa.feature.melspectrogram(
        y=a, sr=sr, n_fft=2048, hop_length=512, n_mels=128, fmin=0.0, fmax=None, norm="slaney"
    )
    db = (10.0 * np.log10(np.clip(m, 1e-10, None))).astype(np.float32)
    if db.shape[1] >= FRAMES:
        db = db[:, :FRAMES]
    else:
        pad = np.full((db.shape[0], FRAMES - db.shape[1]), -100.0, dtype=np.float32)
        db = np.concatenate([db, pad], axis=1)
    return db


def witness_rms_global_db(audio: np.ndarray) -> float:
    """Globaler Pegel (dBFS) des Witness-Fensters — Skip-Feature für den Head.

    D-K3-95-Nacharbeit 2026-11-22: BatchNorm entfernt die gemeinsame
    Pegel-Komponente der Mel-Bänder — genau die Dimension, in der
    dropout_oxide/dropouts/tape_splice leben (Kontrast +17,4/+13,5/+4,6 dB).
    Der Pegel wird deshalb am BatchNorm VORBEI direkt an den Head geführt.
    Deterministisch, eine Quelle der Wahrheit (§G5 (GEBOTE.md)): Trainer,
    Gate und Produktions-DSP rufen exakt diese Funktion.
    """
    a = np.asarray(audio, dtype=np.float64)
    return float(10.0 * np.log10(float(np.mean(a**2)) + 1e-10))


def _rms_pad(audio: np.ndarray) -> float:
    """Pegel-Feature auf exakt dem 1-s-Fenster des Gates (44,1 kHz, Zentrum/Pad).

    Parität (DRT-4): der Trainer misst das Skip-Feature auf demselben Fenster,
    das das Gate der Inferenz gibt — sonst driftet die Pegel-Skala zwischen
    Training und Produktion (Padding-Verdünnung bei Kurz-Regionen).
    """
    import librosa

    a = np.asarray(audio, dtype=np.float32)
    a = librosa.resample(a, orig_sr=SR, target_sr=44100).astype(np.float32)
    if len(a) < 44100:
        a = np.pad(a, (0, 44100 - len(a)))
    elif len(a) > 44100:
        start = (len(a) - 44100) // 2
        a = a[start : start + 44100]
    return witness_rms_global_db(a)


def lade_witness(wd: dict) -> WitnessCRNN:
    """Witness aus Checkpoint-Daten laden — rms_skip-kompatibel (eine Quelle, §V25)."""
    return WitnessCRNN(int(wd["n_mels"]), rms_skip=bool(wd.get("rms_skip", False)))


def witness_rms_tensor(x: torch.Tensor) -> torch.Tensor:
    """Globaler Pegel (dBFS) je Sample — Skip-Feature für die Inferenz (Torch-Pendant).

    Formel-identisch zu `witness_rms_global_db` (numpy), Batch-tauglich.
    """
    power = torch.mean(x.reshape(len(x), -1) ** 2, dim=1)
    return 10.0 * torch.log10(power + 1e-10)


def _lade_region(pfad: Path, sr: int) -> np.ndarray | None:
    from scipy.io import wavfile

    roh = wavfile.read(str(pfad))
    if not isinstance(roh, tuple) or len(roh) != 2:
        return None
    r, d = roh
    x = np.asarray(d, np.float32) / 32768.0
    mono = x.mean(axis=1) if x.ndim == 2 else x
    if int(r) != sr:
        import librosa

        mono = librosa.resample(mono, orig_sr=int(r), target_sr=sr).astype(np.float32)
    return mono


class WitnessCRNN(nn.Module):
    """Binärer Schadens-Detektor auf Produktions-Mel (Muster DefectCRNN).

    `rms_skip=True`: der globale Pegel (dBFS) wird am BatchNorm vorbei als
    Skip-Feature an den Head geführt (Befund 2026-11-22: BatchNorm löscht die
    Pegel-Dimension, die pegellebende Defekte erst sichtbar macht).
    """

    def __init__(self, n_mels: int, rms_skip: bool = False):
        super().__init__()
        self.n_mels = n_mels
        self.rms_skip = bool(rms_skip)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.gru = nn.GRU(64 * (n_mels // 8), 96, batch_first=True, bidirectional=True)
        self.head = nn.Linear(192 + (1 if self.rms_skip else 0), 1)

    def forward(self, x, rms_global=None):
        b = x.shape[0]
        h = self.conv(x)
        h = h.permute(0, 3, 1, 2).contiguous().view(b, -1, 64 * (x.shape[2] // 8))
        o, _ = self.gru(h)
        letzter = o[:, -1]
        if self.rms_skip and rms_global is not None:
            letzter = torch.cat([letzter, rms_global.view(b, 1)], dim=-1)
        return self.head(letzter).squeeze(-1)


def _baue_dataset(
    klasse: str, manifest: Path | None = None, abs_mel: bool = False, rms_skip: bool = False
) -> tuple[list[tuple[np.ndarray, int, str, float]], list[str], int]:
    # None-Default statt `= MANIFEST`: Default-Argumente binden zur Definitionszeit,
    # sonst sieht ein Monkeypatch auf das Modul-MANIFEST (Tests) den alten Pfad.
    if manifest is None:
        manifest = MANIFEST
    eintraege = []
    for line in manifest.open(encoding="utf-8"):
        d = json.loads(line)
        if d["klasse"] == klasse:
            eintraege.append(d)
    titel = sorted({e["titel"] for e in eintraege})
    holdout_titel = max(titel, key=lambda t: sum(1 for e in eintraege if e["titel"] == t))
    daten: list[tuple[np.ndarray, int, str, float]] = []
    uebersprungen = 0
    for e in eintraege:
        c = _lade_region(Path(e["clean"]), SR)
        s = _lade_region(Path(e["schaden"]), SR)
        if c is None or s is None or len(c) < 512 or len(s) < 512:
            uebersprungen += 1
            continue
        daten.append((_mel_fest_abs(c, SR) if abs_mel else _mel_fest(c, SR), 0, e["titel"], _rms_pad(c)))
        daten.append((_mel_fest_abs(s, SR) if abs_mel else _mel_fest(s, SR), 1, e["titel"], _rms_pad(s)))
    return daten, [holdout_titel], uebersprungen


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    if len(np.unique(y)) < 2:
        return 0.5
    return float(roc_auc_score(y, p))


def _kontrast_hinge(p_schaden: torch.Tensor, p_clean: torch.Tensor, margin: float = 0.5) -> torch.Tensor:
    """Kontrast-Hinge für den Witness-Trainer (D-K3-95-Nacharbeit, 2026-11-22).

    Erzwingt die DRT-3-Voraussetzung (p_schaden − p_clean ≥ 0,5) als Loss-Ziel
    statt nur über BCE zu hoffen: `clamp(margin − (mean_p_schaden − mean_p_clean), 0)`.
    Default-AUS (lambda_kontrast=0.0) — ohne Flag bit-identisch (§G5 (GEBOTE.md)).
    """
    differenz = torch.sigmoid(p_schaden).mean() - torch.sigmoid(p_clean).mean()
    return torch.clamp(margin - differenz, min=0.0)


def trainiere(
    klasse: str,
    manifest: Path | None = None,
    lambda_kontrast: float = 0.0,
    clean_gewicht: float = 1.0,
    abs_mel: bool = False,
    rms_skip: bool = False,
) -> dict[str, Any]:
    daten, holdouts, uebersprungen = _baue_dataset(klasse, manifest, abs_mel, rms_skip)
    if not daten:
        return {"klasse": klasse, "fehler": "keine verwertbaren Regionen"}
    train = [(m, l, r) for m, l, _t, r in daten if _t not in holdouts]
    hold = [(m, l, r) for m, l, _t, r in daten if _t in holdouts]
    rng = np.random.default_rng(SEED)
    train = [train[i] for i in rng.permutation(len(train))]
    hold = [hold[i] for i in rng.permutation(len(hold))]
    X_tr = np.stack([m for m, _, _ in train])[:, None, :, :]
    y_tr = np.asarray([l for _, l, _ in train], dtype=np.float32)
    R_tr = np.asarray([r for _, _, r in train], dtype=np.float32)
    X_ho = np.stack([m for m, _, _ in hold])[:, None, :, :]
    y_ho = np.asarray([l for _, l, _ in hold], dtype=np.float32)
    R_ho = np.asarray([r for _, _, r in hold], dtype=np.float32)
    n_mels = X_tr.shape[2]

    _seed_all(SEED)
    model = WitnessCRNN(n_mels, rms_skip=rms_skip)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    # Clean-Gewichtung (Kalibrierungs-Hebel, D-K3-95-Nacharbeit 2026-11-22):
    # p_clean zu hoch ⇒ Trigger feuert auf sauberen Regionen (dropouts-Befund
    # p_clean 0,76). Gewicht > 1 bestraft Clean-Fehlklassifikation stärker.
    # Default 1.0 = bit-identisch zum Vorgänger (§G5 (GEBOTE.md)).
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    Xt = torch.from_numpy(X_tr)
    yt = torch.from_numpy(y_tr)
    Rt = torch.from_numpy(R_tr)
    Xho_t = torch.from_numpy(X_ho)
    Rho_t = torch.from_numpy(R_ho)
    best: _BestCheckpoint = {"auc": 0.0, "epoch": -1, "state": None}
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), BATCH):
            idx = perm[i : i + BATCH]
            opt.zero_grad()
            logits = model(Xt[idx], rms_global=Rt[idx] if rms_skip else None)
            yb = yt[idx]
            bce = loss_fn(logits, yb)
            gewichte = torch.where(yb > 0.5, torch.ones_like(bce), torch.full_like(bce, clean_gewicht))
            loss = (bce * gewichte).mean()
            if lambda_kontrast > 0.0:
                schaden_mask = yb > 0.5
                clean_mask = ~schaden_mask
                if bool(schaden_mask.any()) and bool(clean_mask.any()):
                    loss = loss + lambda_kontrast * _kontrast_hinge(logits[schaden_mask], logits[clean_mask])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            p_ho = torch.sigmoid(model(Xho_t, rms_global=Rho_t if rms_skip else None)).numpy()
        auc_ho = _auc(y_ho, p_ho)
        if auc_ho > best["auc"]:
            best = {"auc": auc_ho, "epoch": ep, "state": {k: v.cpu().clone() for k, v in model.state_dict().items()}}
        if ep in (0, 4, 9, 19):
            print(f"  {klasse}: ep {ep + 1}/{EPOCHS} hoAUC {auc_ho:.3f}", flush=True)

    best_state = best["state"]
    if best_state is None:
        raise RuntimeError(f"Kein gültiger Witness-Checkpoint für {klasse} ausgewählt")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        p_ho = torch.sigmoid(model(Xho_t, rms_global=Rho_t if rms_skip else None)).numpy()
        p_tr = torch.sigmoid(model(Xt, rms_global=Rt if rms_skip else None)).numpy()
    richtung_ok = bool(np.mean(p_ho[y_ho == 1]) > np.mean(p_ho[y_ho == 0]))
    gate = bool(best["auc"] >= AUC_ZIEL and richtung_ok)

    out_dir = MODELS_DIR / f"witness_{klasse}"
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / "witness.pt"
    torch.save(
        {"state_dict": best_state, "n_mels": n_mels, "frames": FRAMES, "seed": SEED, "rms_skip": bool(rms_skip)},
        checkpoint,
    )
    report = {
        "schema": SCHEMA,
        "datum": date.today().isoformat(),
        "klasse": klasse,
        "seed": SEED,
        "n_train": len(train),
        "n_holdout": len(hold),
        "uebersprungen": uebersprungen,
        "holdout_titel": holdouts,
        "train_auc": round(float(_auc(y_tr, p_tr)), 4),
        "holdout_auc": round(float(best["auc"]), 4),
        "beste_epoche": best["epoch"] + 1,
        "richtung_ok": richtung_ok,
        "gate_bestanden": gate,
        "mean_p_schaden_holdout": round(float(np.mean(p_ho[y_ho == 1])), 4),
        "mean_p_clean_holdout": round(float(np.mean(p_ho[y_ho == 0])), 4),
        "lambda_kontrast": lambda_kontrast,
        "clean_gewicht": clean_gewicht,
        "abs_mel": bool(abs_mel),
        "rms_skip": bool(rms_skip),
        "checkpoint": str(checkpoint.relative_to(_ROOT)),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="Per-Klassen-Schadens-Witness trainieren (report-only)")
    ap.add_argument("--klassen", default=None, help="Komma-Liste (Default: alle Erstbeleg-Klassen)")
    ap.add_argument(
        "--manifest", type=Path, default=MANIFEST, help="Manifest-JSONL (Default: data/klassen_paare/manifest.jsonl)"
    )
    ap.add_argument(
        "--abs-mel",
        action="store_true",
        help="Absolute Mel (ref=1.0) statt compute_mel_produktion — schließt die Mel-Drift zur Produktion (D-K3-95-Nacharbeit 2026-11-22)",
    )
    ap.add_argument(
        "--rms-skip",
        action="store_true",
        help="Globaler Pegel (dBFS) als Skip-Feature am BatchNorm vorbei in den Head (pegellebende Defekte, D-K3-95-Nacharbeit 2026-11-22)",
    )
    args = ap.parse_args()
    klassen = (
        (args.klassen or "").split(",")
        if args.klassen
        else [
            "groove_echo",
            "sticky_shed_residue",
            "tape_head_level_dip",
            "sibilance",
            "compression_artifacts",
            "transport_bump",
        ]
    )
    out = [
        trainiere(k, args.manifest, args.lambda_kontrast, args.clean_gewicht, args.abs_mel, args.rms_skip)
        for k in klassen
    ]
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
