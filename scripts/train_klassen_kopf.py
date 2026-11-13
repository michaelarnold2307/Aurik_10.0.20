#!/usr/bin/env python3
"""train_klassen_kopf — kleiner Per-Klassen-Reparatur-Kopf (Trainingsstrategie Stufe 3).

Hör-zentrierte Trainingsstrategie (Report
`docs/reports/current/2026-11-13_hoerzentrierte_trainingsstrategie.md`):
Statt tagelanger Vollband-Finetunes trainiert jeder Kopf EINE Defektklasse
auf den Rezept-abgeleiteten Regionen (`scripts/erstelle_klassen_paare.py`):
schaden → clean, kleines Modell, Stunden statt Tage. Erster Kopf:
**groove_echo** (4.072 Regionen, größte Q≈0-Klasse der §G13-Erstbelege).

Konventionen: Seed 42 (§G5 (copilot-instructions.md)), Zeitbudget
(`--max-seconds`, Exit 75, D-K3-87), Anker `checkpoint_latest.pt`,
fail-closed Datenprüfung (§V6 (copilot-instructions.md)), kein Flag-Flip —
der Erfolg wird nach dem Lauf als §G13-Q-Erstbeleg gebucht
(`scripts/bewerte_defekt_delta.py`).

Aufruf:
    nice -n 15 python3 -B scripts/train_klassen_kopf.py \
        --klasse groove_echo --manifest data/klassen_paare/manifest.jsonl \
        --out models/klassen_koepfe/groove_echo --smoke
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

log = logging.getLogger(__name__)

WINDOW = 8192  # feste Fensterlänge in Samples (Regionen werden gepaddet/gecroppt)


def _wav_lesen(pfad: Path) -> tuple[int, np.ndarray]:
    from scipy.io import wavfile

    gelesen = wavfile.read(pfad)
    if not isinstance(gelesen, tuple) or len(gelesen) < 2:  # Bug 12: Index-basiert statt Unpacking
        raise ValueError("wavfile.read() ohne (sr, data)-Tupel")
    sr = int(gelesen[0])
    daten = np.asarray(gelesen[1])
    if np.issubdtype(daten.dtype, np.integer):
        daten = daten.astype(np.float32) / 32768.0
    else:
        daten = daten.astype(np.float32)
    return sr, daten


def _fenster(audio: np.ndarray, laenge: int = WINDOW) -> np.ndarray:
    """Pad/Trunc auf feste Fensterlänge (mono float32)."""
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim == 2:
        x = x.mean(axis=-1)
    if x.shape[0] >= laenge:
        return x[:laenge]
    out = np.zeros(laenge, dtype=np.float32)
    out[: x.shape[0]] = x
    return out


def _eintraege_laden(manifest_pfad: Path, klasse: str) -> list[dict]:
    """Fail-closed: Manifest muss existieren und Einträge für die Klasse haben."""
    if not manifest_pfad.exists():
        print(f"FEHLER: Manifest fehlt: {manifest_pfad}", file=sys.stderr)
        raise SystemExit(2)
    zeilen = [json.loads(l) for l in manifest_pfad.read_text(encoding="utf-8").splitlines() if l.strip()]
    eintraege = [z for z in zeilen if z.get("klasse") == klasse and z.get("clean")]
    if not eintraege:
        print(f"FEHLER: keine clean-Einträge für Klasse {klasse} in {manifest_pfad}", file=sys.stderr)
        raise SystemExit(2)
    return eintraege


def _mr_stft_loss(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Multi-Resolution-STFT-Verlust (Muster train_bigvgan_f3, klein gehalten)."""
    import torch

    def stft(x, n_fft, hop):
        return torch.stft(
            x, n_fft=n_fft, hop_length=hop, return_complex=True, window=torch.hann_window(n_fft, device=x.device)
        )

    verlust = torch.tensor(0.0, device=y.device)
    for n_fft, hop in ((512, 128), (1024, 256)):
        a, b = stft(y_hat.squeeze(1), n_fft, hop), stft(y.squeeze(1), n_fft, hop)
        verlust = verlust + torch.nn.functional.l1_loss(torch.abs(a), torch.abs(b))
    return verlust


def build_model(device: torch.Module) -> torch.nn.Module:
    """Kompakter Conv1d-Encoder-Decoder (~60k Parameter, Reparatur-Kopf)."""
    import torch
    from torch import nn

    class Kopf(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.enc1 = nn.Conv1d(1, 16, 15, stride=2, padding=7)
            self.enc2 = nn.Conv1d(16, 32, 15, stride=2, padding=7)
            self.enc3 = nn.Conv1d(32, 64, 15, stride=2, padding=7)
            self.dec3 = nn.ConvTranspose1d(64, 32, 15, stride=2, padding=7, output_padding=1)
            self.dec2 = nn.ConvTranspose1d(64, 16, 15, stride=2, padding=7, output_padding=1)
            self.dec1 = nn.ConvTranspose1d(32, 1, 15, stride=2, padding=7, output_padding=1)
            self.act = nn.LeakyReLU(0.2)

        def forward(self, x):
            e1 = self.act(self.enc1(x))
            e2 = self.act(self.enc2(e1))
            e3 = self.act(self.enc3(e2))
            d3 = self.act(self.dec3(e3))
            d3 = torch.cat([d3, e2[:, :, : d3.shape[-1]]], dim=1)
            d2 = self.act(self.dec2(d3))
            d2 = torch.cat([d2, e1[:, :, : d2.shape[-1]]], dim=1)
            d1 = self.dec1(d2)
            return d1[:, :, : x.shape[-1]]

    return Kopf().to(device)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Per-Klassen-Reparatur-Kopf (Trainingsstrategie Stufe 3)")
    ap.add_argument("--klasse", default="groove_echo")
    ap.add_argument("--manifest", default="data/klassen_paare/manifest.jsonl")
    ap.add_argument("--out", default="")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--steps-per-epoch", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-seconds", type=float, default=7200.0)
    ap.add_argument("--resume", type=str, default="")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)

    import torch

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    eintraege = _eintraege_laden(ROOT / args.manifest, args.klasse)
    # Deterministischer Val-Split: jede 5. Region (nach sha256-Sortierung stabil).
    eintraege.sort(key=lambda z: z.get("sha256_pcm_schaden", ""))
    val_eintraege = eintraege[::5]
    train_eintraege = [z for i, z in enumerate(eintraege) if i % 5 != 0]
    log.info("Klasse %s: %d Train- / %d Val-Regionen", args.klasse, len(train_eintraege), len(val_eintraege))

    out_dir = Path(args.out) if args.out else (ROOT / "models" / "klassen_koepfe" / args.klasse)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    start_epoch = 0
    if args.resume:
        anker = Path(args.resume)
        if not anker.exists():
            print(f"FEHLER: Resume-Anker fehlt: {anker}", file=sys.stderr)
            return 2
        res = torch.load(anker, map_location=device)
        model.load_state_dict(res["modell"])
        opt.load_state_dict(res["optimizer"])
        start_epoch = res["epoch"] + 1

    def daten_laden(liste: list[dict]) -> list[np.ndarray]:
        x_liste, y_liste = [], []
        for z in liste:
            schaden = _fenster(_wav_lesen(ROOT / z["schaden"])[1])
            clean = _fenster(_wav_lesen(ROOT / z["clean"])[1])
            x_liste.append(schaden)
            y_liste.append(clean)
        return x_liste, y_liste

    train_x, train_y = daten_laden(train_eintraege)
    val_x, val_y = daten_laden(val_eintraege)
    rng = np.random.default_rng(args.seed)
    t_start = time.perf_counter()
    best_val = float("inf")
    history: list[dict] = []

    steps_per_epoch = 3 if args.smoke else args.steps_per_epoch
    epochen_dieser_lauf = 0
    for epoch in range(start_epoch, args.epochs):
        if time.perf_counter() - t_start > args.max_seconds:
            report = {
                "schema": "aurik.klassen_kopf/1",
                "klasse": args.klasse,
                "ergebnis": "zeitbudget",
                "epoch": epoch,
                "seed": args.seed,
                "val_mae": best_val,
                "history": history,
                "anker": str(out_dir / "checkpoint_latest.pt"),
            }
            (out_dir / "report.json").write_text(
                json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            print(
                f"Zeitbudget {args.max_seconds} s erreicht (Exit 75) — Resume: --resume {out_dir / 'checkpoint_latest.pt'}"
            )
            return 75
        model.train()
        verluste = []
        for _ in range(steps_per_epoch):
            idx = rng.integers(0, len(train_x), size=args.batch_size)
            x = torch.from_numpy(np.stack([train_x[i] for i in idx])).unsqueeze(1).to(device)
            y = torch.from_numpy(np.stack([train_y[i] for i in idx])).unsqueeze(1).to(device)
            y_hat = model(x)
            l1 = torch.nn.functional.l1_loss(y_hat, y)
            st = _mr_stft_loss(y_hat, y)
            verlust = l1 + 0.5 * st
            opt.zero_grad(set_to_none=True)
            verlust.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            verluste.append(float(verlust.item()))
        # Val
        model.eval()
        val_verluste = []
        with torch.no_grad():
            for vx, vy in zip(val_x[:8], val_y[:8]):
                x = torch.from_numpy(vx).unsqueeze(0).unsqueeze(0).to(device)
                y = torch.from_numpy(vy).unsqueeze(0).unsqueeze(0).to(device)
                val_verluste.append(float(torch.nn.functional.l1_loss(model(x), y).item()))
        v_mae = float(np.mean(val_verluste))
        if v_mae < best_val:
            best_val = v_mae
        history.append({"epoch": epoch, "train_l1": round(float(np.mean(verluste)), 6), "val_mae": round(v_mae, 6)})
        log.info("Epoche %d fertig: train=%.5f val_mae=%.5f", epoch, np.mean(verluste), v_mae)
        epochen_dieser_lauf += 1
        ckpt = {
            "modell": model.state_dict(),
            "optimizer": opt.state_dict(),
            "epoch": epoch,
            "val_mae": v_mae,
            "best_val": best_val,
            "seed": args.seed,
        }
        torch.save(ckpt, out_dir / f"checkpoint_epoch{epoch}.pt")
        torch.save(ckpt, out_dir / "checkpoint_latest.pt")

    report = {
        "schema": "aurik.klassen_kopf/1",
        "klasse": args.klasse,
        "ergebnis": "erfolgreich",
        "epoch": epochen_dieser_lauf,
        "seed": args.seed,
        "val_mae": best_val,
        "history": history,
        "anker": str(out_dir / "checkpoint_latest.pt"),
        "hinweis": "§G13 (copilot-instructions.md): der Erfolg wird als Q-Erstbeleg je Defektklasse gebucht (scripts/bewerte_defekt_delta.py) — KEIN Flag-Flip, Freigabe nur §4.",
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Report: {out_dir / 'report.json'} (val_mae={best_val:.5f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
