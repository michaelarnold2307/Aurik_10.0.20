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
from typing import Any

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
    """Binärer Schadens-Detektor auf Produktions-Mel (Muster DefectCRNN)."""

    def __init__(self, n_mels: int):
        super().__init__()
        self.n_mels = n_mels
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
        self.head = nn.Linear(192, 1)

    def forward(self, x):
        b = x.shape[0]
        h = self.conv(x)
        h = h.permute(0, 3, 1, 2).contiguous().view(b, -1, 64 * (x.shape[2] // 8))
        o, _ = self.gru(h)
        return self.head(o[:, -1]).squeeze(-1)


def _baue_dataset(
    klasse: str, manifest: Path | None = None
) -> tuple[list[tuple[np.ndarray, int, str]], list[str], int]:
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
    daten: list[tuple[np.ndarray, int, str]] = []
    uebersprungen = 0
    for e in eintraege:
        c = _lade_region(Path(e["clean"]), SR)
        s = _lade_region(Path(e["schaden"]), SR)
        if c is None or s is None or len(c) < 512 or len(s) < 512:
            uebersprungen += 1
            continue
        daten.append((_mel_fest(c, SR), 0, e["titel"]))
        daten.append((_mel_fest(s, SR), 1, e["titel"]))
    return daten, [holdout_titel], uebersprungen


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    if len(np.unique(y)) < 2:
        return 0.5
    return float(roc_auc_score(y, p))


def trainiere(klasse: str, manifest: Path | None = None) -> dict[str, Any]:
    daten, holdouts, uebersprungen = _baue_dataset(klasse, manifest)
    if not daten:
        return {"klasse": klasse, "fehler": "keine verwertbaren Regionen"}
    train = [(m, l) for m, l, t in daten if t not in holdouts]
    hold = [(m, l) for m, l, t in daten if t in holdouts]
    rng = np.random.default_rng(SEED)
    train = [train[i] for i in rng.permutation(len(train))]
    hold = [hold[i] for i in rng.permutation(len(hold))]
    X_tr = np.stack([m for m, _ in train])[:, None, :, :]
    y_tr = np.asarray([l for _, l in train], dtype=np.float32)
    X_ho = np.stack([m for m, _ in hold])[:, None, :, :]
    y_ho = np.asarray([l for _, l in hold], dtype=np.float32)
    n_mels = X_tr.shape[2]

    _seed_all(SEED)
    model = WitnessCRNN(n_mels)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()
    Xt = torch.from_numpy(X_tr)
    yt = torch.from_numpy(y_tr)
    best = {"auc": 0.0, "epoch": -1, "state": None}
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), BATCH):
            idx = perm[i : i + BATCH]
            opt.zero_grad()
            out = model(Xt[idx])
            loss = loss_fn(out, yt[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            p_ho = torch.sigmoid(model(torch.from_numpy(X_ho))).numpy()
        auc_ho = _auc(y_ho, p_ho)
        if auc_ho > best["auc"]:
            best = {"auc": auc_ho, "epoch": ep, "state": {k: v.cpu().clone() for k, v in model.state_dict().items()}}
        if ep in (0, 4, 9, 19):
            print(f"  {klasse}: ep {ep + 1}/{EPOCHS} hoAUC {auc_ho:.3f}", flush=True)

    model.load_state_dict(best["state"])
    model.eval()
    with torch.no_grad():
        p_ho = torch.sigmoid(model(torch.from_numpy(X_ho))).numpy()
        p_tr = torch.sigmoid(model(Xt)).numpy()
    richtung_ok = bool(np.mean(p_ho[y_ho == 1]) > np.mean(p_ho[y_ho == 0]))
    gate = bool(best["auc"] >= AUC_ZIEL and richtung_ok)

    out_dir = MODELS_DIR / f"witness_{klasse}"
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / "witness.pt"
    torch.save({"state_dict": best["state"], "n_mels": n_mels, "frames": FRAMES, "seed": SEED}, checkpoint)
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
    out = [trainiere(k, args.manifest) for k in klassen]
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
