#!/usr/bin/env python3
"""train_klassen_kopf — kleiner Per-Klassen-Reparatur-Kopf (Trainingsstrategie Stufe 3).

Hör-zentrierte Trainingsstrategie (Report
`docs/reports/current/2026-11-13_hoerzentrierte_trainingsstrategie.md`):
Statt tagelanger Vollband-Finetunes trainiert jeder Kopf EINE Defektklasse
auf den Rezept-abgeleiteten Regionen (`scripts/erstelle_klassen_paare.py`):
schaden → clean, kleines Modell, Stunden statt Tage. Erster Kopf:
**groove_echo** (4.072 Regionen, größte Q≈0-Klasse der §G13 (copilot-instructions.md)-Erstbelege).

Konventionen: Seed 42 (§G5 (copilot-instructions.md)), Zeitbudget
(`--max-seconds`, Exit 75, D-K3-87), Anker `checkpoint_latest.pt`,
fail-closed Datenprüfung (§V6 (copilot-instructions.md)), kein Flag-Flip —
der Erfolg wird nach dem Lauf als §G13 (copilot-instructions.md)-Q-Erstbeleg gebucht
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
_SPLIT_SCHEMA = "aurik.klassen_kopf_split/1"
_MODEL_SCHEMA = "aurik.klassen_kopf_model/residual_identity_v2"


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


def _titel_hash(titel: str) -> str:
    return hashlib.sha256(titel.encode("utf-8")).hexdigest()


def _split_nach_titel(eintraege: list[dict], seed: int) -> tuple[list[dict], list[dict], dict]:
    """Hält einen ganzen, deterministischen Titel für die unabhängige Prüfung zurück."""
    titel = sorted({str(e.get("titel", "")).strip() for e in eintraege})
    if len(titel) < 2 or any(not name for name in titel):
        raise ValueError("Titel-Holdout benötigt mindestens zwei benannte Titel")
    holdout_titel = min(
        titel,
        key=lambda name: hashlib.sha256(f"{seed}:{name}".encode()).hexdigest(),
    )
    train_titel = [name for name in titel if name != holdout_titel]
    train = [e for e in eintraege if str(e.get("titel", "")).strip() in train_titel]
    holdout = [e for e in eintraege if str(e.get("titel", "")).strip() == holdout_titel]
    provenance = {
        "schema": _SPLIT_SCHEMA,
        "strategie": "whole_title_sha256_seeded",
        "seed": int(seed),
        "train_titel_sha256": sorted(_titel_hash(name) for name in train_titel),
        "holdout_titel_sha256": _titel_hash(holdout_titel),
    }
    return train, holdout, provenance


def _resume_provenance_gueltig(checkpoint: object, expected: dict) -> bool:
    return (
        isinstance(checkpoint, dict)
        and checkpoint.get("split_provenance") == expected
        and checkpoint.get("model_schema") == _MODEL_SCHEMA
    )


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


def _defektlast_proxy(y_hat: torch.Tensor) -> torch.Tensor:
    """Differenzierbarer defektlast-Proxy (Fahrplan B-0(2), 2026-11-13).

    Spiegel der kanonischen Metrik `backend/core/dsp/wohlklang_zeuge.defektlast`
    (clip(≥0,98)·2 + flatness + hf_verlust·0,5), aber differenzierbar:
    Soft-Clip bei 0,90 (Rezeptur-Schwelle — die Metrik zählt erst ab 0,98,
    das Reparaturziel ist die 0,90-Plateau-Beseitigung), Flatness und
    HF-Anteil über die 2048er-STFT. Minimiert wird der Proxy des OUTPUTS.
    """
    import torch

    x = y_hat.squeeze(1)
    tau = 0.005
    soft_clip = torch.sigmoid((x.abs() - 0.90) / tau).mean()
    window = torch.hann_window(2048, device=x.device)
    spec = torch.stft(x, n_fft=2048, hop_length=1024, return_complex=True, window=window).abs() ** 2 + 1e-18
    flatness = (torch.exp(spec.log().mean(dim=0)) / (spec.mean(dim=0) + 1e-12)).mean()
    freqs = torch.fft.rfftfreq(2048, d=1.0 / 48000, device=x.device)
    hf_mask = (freqs > 8000).float().unsqueeze(1)
    hf = (spec * hf_mask).sum(dim=0) / (spec.sum(dim=0) + 1e-12)
    hf_verlust = torch.clamp(1.0 - torch.clamp(hf / 0.02, 0.0, 1.0), min=0.0).mean()
    return 2.0 * soft_clip + flatness + 0.5 * hf_verlust


DEFEKTLAST_PROXY_GEWICHT = 0.1


def _proxy_gewicht_lesen(wert: float | None) -> float:
    """CLI-Override des Proxy-Gewichts (Fahrplan B-0(2)-Hebel); Default bleibt 0,1."""
    return DEFEKTLAST_PROXY_GEWICHT if wert is None else float(wert)


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
            nn.init.zeros_(self.dec1.weight)
            nn.init.zeros_(self.dec1.bias)

        def forward(self, x):
            e1 = self.act(self.enc1(x))
            e2 = self.act(self.enc2(e1))
            e3 = self.act(self.enc3(e2))
            d3 = self.act(self.dec3(e3))
            d3 = torch.cat([d3, e2[:, :, : d3.shape[-1]]], dim=1)
            d2 = self.act(self.dec2(d3))
            d2 = torch.cat([d2, e1[:, :, : d2.shape[-1]]], dim=1)
            d1 = self.dec1(d2)
            return x + d1[:, :, : x.shape[-1]]

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
    ap.add_argument(
        "--proxy-gewicht", type=float, default=None, help="Override des defektlast-Proxy-Gewichts (Default 0.1)"
    )
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)

    import torch

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    manifest_pfad = ROOT / args.manifest
    eintraege = _eintraege_laden(manifest_pfad, args.klasse)
    try:
        train_titel_eintraege, holdout_eintraege, split_provenance = _split_nach_titel(eintraege, args.seed)
    except ValueError as exc:
        print(f"FEHLER: {exc}", file=sys.stderr)
        return 2
    manifest_sha256 = hashlib.sha256(manifest_pfad.read_bytes()).hexdigest()
    split_provenance["manifest_sha256"] = manifest_sha256
    # Interne Val-Regionen stammen ausschließlich aus den Trainingstiteln.
    train_titel_eintraege.sort(key=lambda z: z.get("sha256_pcm_schaden", ""))
    val_eintraege = train_titel_eintraege[::5]
    train_eintraege = [z for i, z in enumerate(train_titel_eintraege) if i % 5 != 0]
    log.info(
        "Klasse %s: %d Train- / %d interne Val-Regionen; %d unabhängige Holdout-Regionen",
        args.klasse,
        len(train_eintraege),
        len(val_eintraege),
        len(holdout_eintraege),
    )

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
        if not _resume_provenance_gueltig(res, split_provenance):
            print(
                f"FEHLER: Resume-Anker ohne passende Titel-Holdout-Provenienz: {anker}",
                file=sys.stderr,
            )
            return 2
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
            y_control_hat = model(y)
            l1 = torch.nn.functional.l1_loss(y_hat, y)
            st = _mr_stft_loss(y_hat, y)
            clean_identity_l1 = torch.nn.functional.l1_loss(y_control_hat, y)
            clean_identity_stft = _mr_stft_loss(y_control_hat, y)
            verlust = (
                l1
                + 0.5 * st
                + clean_identity_l1
                + 0.5 * clean_identity_stft
                + _proxy_gewicht_lesen(args.proxy_gewicht) * _defektlast_proxy(y_hat)
            )
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
            "model_schema": _MODEL_SCHEMA,
            "epoch": epoch,
            "val_mae": v_mae,
            "best_val": best_val,
            "seed": args.seed,
            "split_provenance": split_provenance,
        }
        from backend.core.training_artifacts import save_guarded

        save_guarded(out_dir / f"checkpoint_epoch{epoch}.pt", ckpt)
        save_guarded(out_dir / "checkpoint_latest.pt", ckpt)

    report = {
        "schema": "aurik.klassen_kopf/1",
        "klasse": args.klasse,
        "ergebnis": "erfolgreich",
        "epoch": epochen_dieser_lauf,
        "seed": args.seed,
        "val_mae": best_val,
        "history": history,
        "anker": str(out_dir / "checkpoint_latest.pt"),
        "split_provenance": split_provenance,
        "holdout_regionen": len(holdout_eintraege),
        "hinweis": "§G13 (copilot-instructions.md): der Erfolg wird als Q-Erstbeleg je Defektklasse gebucht (scripts/bewerte_defekt_delta.py) — KEIN Flag-Flip, Freigabe nur §4.",
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Report: {out_dir / 'report.json'} (val_mae={best_val:.5f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
