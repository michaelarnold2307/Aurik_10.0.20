"""scripts/train_witness_kopf.py — defekt-klassenspezifischer Reparatur-Kopf mit Witness-Ziel (D-K3-95).

Muster hinter der M9-Niederlage (Handover 2026-11-15): Der TF-GridNet-Denoiser
war ein Dauerläufer (schadet auf sauberem Material ⇒ Never-worsen unerfüllbar),
optimierte Mask-MSE statt des Gate-Ziels und trainierte auf synthetischem
Rauschen statt echten Defekten. Dieser Kopf dreht alle drei Punkte um:

  1. **Defekt-spezifisch:** trainiert auf k398-Regionen der Klasse
     (clean/schaden, data/klassen_paare/manifest.jsonl) — der Kopf lernt den
     Klassen-Defekt zu reparieren, sonst nichts. Leak-frei: Training auf allen
     Titeln außer dem Holdout-Titel (Titel mit den meisten Regionen, Muster
     Witness-Trainer).
  2. **Gate-Ziel im Loss:** L = L1(out, clean) + λ·BCE(Witness(out), 0) — der
     gelernte Schadens-Witness (models/witness_<klasse>/witness.pt, Holdout-AUC
     0,90–1,00, differenzierbar über eine torch-Mel-Transformation) zieht die
     Ausgabe in Richtung clean. Das ist die Antwort auf „L1 ist blind für das,
     was gemessen wird".
  3. **Modellauswahl nach dem Gate-Ziel:** der beste Checkpoint ist der mit dem
     kleinsten Holdout-Score (L1 + λ·Witness-Penalty auf dem Holdout-Titel) —
     NICHT Val-L1 und NICHT In-Sample (M9-V2 wählte nach Val-MSE und verfehlte
     das Gate). Zusätzlich wird die Witness-Q-Quote auf In-Sample UND Holdout
     im Report ausgewiesen (Pre-flight: In-Sample-Q ≥ 0,5).

Soft-Gating (Sanft-Blend Input↔Kopf über global_scalar, 6 dB/200 ms §III
(copilot-instructions.md)) erfolgt bei der Verdrahtung — dieser Trainer liefert
den Kopf; die Aktivierung bleibt dem Never-worsen-Gate + §4 vorbehalten.
Report-only bis dahin: kein Flag, kein Produktionskonsument
(Allowlist „klassen_koepfe" bleibt gültig).

Deterministisch (Seed 42, §G5 (copilot-instructions.md)); CPU-Smoke über
--max-regionen/--epochen/--hidden/--stufen klein halten.
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

from scripts.train_witness_detektor import WitnessCRNN

MANIFEST = _ROOT / "data" / "klassen_paare" / "manifest.jsonl"
OUT_BASE = _ROOT / "models" / "klassen_koepfe"
SEED = 42
SR = 44100
FENSTER = 44100  # 1 s
N_FFT = 2048
HOP = 512
N_MELS = 128
FRAMES = 64
RMS_ZIEL = 0.2
SCHEMA = "aurik.witness_kopf_training/1"


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _lade_mono(pfad: str) -> np.ndarray:
    from scipy.io import wavfile

    roh = wavfile.read(pfad)
    sr, d = int(roh[0]), roh[1]
    x = np.asarray(d, np.float32) / 32768.0
    mono = x.mean(axis=1) if x.ndim == 2 else x
    if sr != SR:
        import librosa

        mono = librosa.resample(mono, orig_sr=sr, target_sr=SR).astype(np.float32)
    return mono


def _fenster_mitte(x: np.ndarray) -> np.ndarray | None:
    if len(x) < 512:
        return None
    if len(x) >= FENSTER:
        start = (len(x) - FENSTER) // 2
        return x[start : start + FENSTER]
    return np.pad(x, (0, FENSTER - len(x)))


def _mel_torch(audio: torch.Tensor) -> torch.Tensor:
    """Differenzierbare Mel-Transformation (Parameter der Produktions-Mel,
    D-K3-96). Näherung an compute_mel_produktion — für den Witness-Penalty
    ausreichend (dokumentiert, kein Produktionskonsument). Schneidet auf die
    64 Witness-Frames (identisch zur Witness-Inferenz)."""
    import librosa

    window = torch.hann_window(N_FFT, dtype=audio.dtype, device=audio.device)
    stft = torch.stft(audio, n_fft=N_FFT, hop_length=HOP, window=window, center=True, return_complex=True)
    mag = stft.abs().pow(2.0)
    # norm='slaney' wie librosa-Default in compute_mel_produktion — ohne die Norm
    # sieht das Witness eine andere Bandskalierung (Befund 2026-11-15: Kontrast
    # kollabierte, q_witness=None).
    mel_basis = torch.from_numpy(
        librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=0.0, fmax=None, norm="slaney")
    )
    mel_basis = mel_basis.to(audio.dtype).to(audio.device)
    mel = torch.matmul(mel_basis, mag)
    # power_to_db-Äquivalent (10·log10 mit amin-Clipping) wie librosa in
    # compute_mel_produktion — ohne Faktor 10/amin sättigt das Witness auf der
    # flachen Skala (Befund 2026-11-15: p_s 0,94 ≈ p_c 0,94, Kontrast kollabiert;
    # mit Produktions-Mel Kontrast 0,66).
    mel = 10.0 * torch.log10(torch.clamp(mel, min=1e-10))
    if mel.shape[-1] > FRAMES:
        mel = mel[..., :FRAMES]
    elif mel.shape[-1] < FRAMES:
        mel = torch.nn.functional.pad(mel, (0, FRAMES - mel.shape[-1]))
    return mel


class WitnessKopf(nn.Module):
    """1D-Conv-Autoencoder (Bottleneck, ohne Skips — bewusst einfach: korrekt,
    deterministisch; Skips sind eine spätere Verbesserung)."""

    def __init__(self, hidden: int = 64, stufen: int = 4):
        super().__init__()
        self.stufen = stufen
        self.enc_kanal = [1]
        self.enc = nn.ModuleList()
        ch = 1
        for i in range(stufen):
            ch_out = hidden * (2 ** min(i, 2))
            self.enc.append(nn.Sequential(nn.Conv1d(ch, ch_out, 9, stride=4, padding=4), nn.LeakyReLU(0.2)))
            ch = ch_out
            self.enc_kanal.append(ch)
        self.dec = nn.ModuleList()
        for i in range(stufen - 1, -1, -1):
            ch_out = self.enc_kanal[i]  # Kanal am Anfang der Stufe i
            self.dec.append(
                nn.Sequential(
                    nn.ConvTranspose1d(ch, ch_out, 9, stride=4, padding=4, output_padding=3),
                    nn.LeakyReLU(0.2),
                )
            )
            ch = ch_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for blk in self.enc:
            h = blk(h)
        for blk in self.dec:
            h = blk(h)
        if h.shape[-1] > x.shape[-1]:
            h = h[..., : x.shape[-1]]
        elif h.shape[-1] < x.shape[-1]:
            h = torch.nn.functional.pad(h, (0, x.shape[-1] - h.shape[-1]))
        return h


def _witness_q(ps: float, pc: float, pout: float) -> float | None:
    kontrast = ps - pc
    if kontrast <= 0.05:
        return None
    return float(np.clip((ps - pout) / kontrast, -1.0, 2.0))


def _baue_regionen(klasse: str) -> tuple[list[tuple[str, Path, Path]], str, int]:
    eintraege = []
    for line in MANIFEST.open(encoding="utf-8"):
        d = json.loads(line)
        if d["klasse"] == klasse:
            eintraege.append(d)
    titel = sorted({e["titel"] for e in eintraege})
    holdout_titel = max(titel, key=lambda t: sum(1 for e in eintraege if e["titel"] == t))
    return [(e["titel"], Path(e["schaden"]), Path(e["clean"])) for e in eintraege], holdout_titel, len(eintraege)


def _lade_split(regionen: list[tuple[str, Path, Path]], holdout_titel: str, max_regionen: int):
    """Regionen → (train, hold) mit Titel-Leak-Schutz; deterministische Permutation (Seed 42)."""
    rng = np.random.default_rng(SEED)
    regionen = [regionen[i] for i in rng.permutation(len(regionen))]
    if max_regionen > 0:
        regionen = regionen[:max_regionen]
    train = [r for r in regionen if r[0] != holdout_titel]
    hold = [r for r in regionen if r[0] == holdout_titel]
    xs, xc, xc_id = [], [], []
    for _, s, c in train:
        a, b = _fenster_mitte(_lade_mono(str(s))), _fenster_mitte(_lade_mono(str(c)))
        if a is None or b is None:
            continue
        rms = float(np.sqrt(np.mean(a**2)) + 1e-8)
        xs.append((a / rms * RMS_ZIEL).astype(np.float32))
        # Gewinn-Arm-Ziel: clean mit SCHADEN-RMS (Ausgabe wird so denormalisiert).
        xc.append((b / rms * RMS_ZIEL).astype(np.float32))
        # DRT-4-Parität (Befund 2026-11-15, Runde 2): der Identitäts-Anker braucht clean mit
        # der CLEAN-EIGENEN RMS — das Gate normalisiert clean-Regionen mit ihrer eigenen RMS.
        rms_c = float(np.sqrt(np.mean(b**2)) + 1e-8)
        xc_id.append((b / rms_c * RMS_ZIEL).astype(np.float32))
    hs, hc, hc_id = [], [], []
    for _, s, c in hold:
        a, b = _fenster_mitte(_lade_mono(str(s))), _fenster_mitte(_lade_mono(str(c)))
        if a is None or b is None:
            continue
        rms = float(np.sqrt(np.mean(a**2)) + 1e-8)
        hs.append((a / rms * RMS_ZIEL).astype(np.float32))
        hc.append((b / rms * RMS_ZIEL).astype(np.float32))
        rms_c = float(np.sqrt(np.mean(b**2)) + 1e-8)
        hc_id.append((b / rms_c * RMS_ZIEL).astype(np.float32))
    return xs, xc, xc_id, hs, hc, hc_id


def _stack(x: list[np.ndarray]) -> torch.Tensor:
    return torch.from_numpy(np.stack(x)[:, None]) if x else torch.zeros(0, 1, FENSTER)


def _witness_p_mittel(witness: nn.Module, mel: torch.Tensor, device: str) -> float:
    with torch.no_grad():
        logit = witness(mel.to(device)[:, None, :, :])
        return float(torch.sigmoid(logit).mean())


def trainiere(
    klasse: str,
    epochen: int,
    hidden: int,
    stufen: int,
    lam: float,
    device: str,
    max_regionen: int = 0,
    lam_id: float = 0.5,
) -> dict[str, Any]:
    regionen, holdout_titel, n_gesamt = _baue_regionen(klasse)
    xs, xc, xc_id, hs, hc, hc_id = _lade_split(regionen, holdout_titel, max_regionen)
    if not xs:
        return {"klasse": klasse, "fehler": "keine verwertbaren Regionen"}
    Xs_t, Xc_t = _stack(xs), _stack(xc)
    Xc_id_t = _stack(xc_id)
    Hs_t, Hc_t = _stack(hs), _stack(hc)
    Hc_id_t = _stack(hc_id)

    w_ckpt = _ROOT / "models" / f"witness_{klasse}" / "witness.pt"
    if not w_ckpt.is_file():
        raise FileNotFoundError(
            f"Witness-Checkpoint fehlt: {w_ckpt} — erst scripts/train_witness_detektor.py ausführen"
        )
    wdaten = torch.load(str(w_ckpt), map_location="cpu", weights_only=False)
    witness = WitnessCRNN(int(wdaten["n_mels"]))
    witness.load_state_dict(wdaten["state_dict"])
    # MIOpen/CUDA: der GRU-Backward braucht den Training-Modus („miopen RNN backward
    # can only be called in training mode", Befund 2026-11-15, venv_rocm72). Deshalb
    # train() fürs RNN — die BatchNorm-Layer bleiben einzeln auf eval (running stats,
    # deterministisch), die Parameter bleiben eingefroren (der Witness wird nicht trainiert).
    witness.train()
    for m in witness.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
    for p in witness.parameters():
        p.requires_grad_(False)
    witness.to(device)

    _seed_all(SEED)
    model = WitnessKopf(hidden, stufen).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best = {"score": float("inf"), "ep": -1, "state": None}
    batch = 8
    for ep in range(epochen):
        model.train()
        perm = torch.randperm(len(Xs_t))
        total = 0.0
        for i in range(0, len(Xs_t), batch):
            idx = perm[i : i + batch]
            s_b, c_b = Xs_t[idx].to(device), Xc_t[idx].to(device)
            out = model(s_b)
            l1 = nn.functional.l1_loss(out, c_b)
            # Identitäts-Anker (D-K3-95-Nacharbeiten 2026-11-15): der Kopf muss auf
            # clean-Regionen identisch bleiben — das ist der Never-worsen-Schutz-Arm
            # als Loss-Ziel. Die gate-verfehlten Kanäle überrestaurierten auf clean,
            # weil das Training nie clean→clean gesehen hatte (§V7 (copilot-instructions.md):
            # Ursache statt Symptom — nicht die Gate-Schwelle, das Training war falsch).
            out_id = model(Xc_id_t[idx].to(device))
            l1_id = nn.functional.l1_loss(out_id, Xc_id_t[idx].to(device))
            mel_out = _mel_torch(out.reshape(len(out), -1))
            pen = torch.sigmoid(witness(mel_out[:, None, :, :])).mean()
            loss = l1 + lam * pen + lam_id * l1_id
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
        # Holdout-Score (Gate-Ziel) für die Modellauswahl — inklusive Identitäts-Anker:
        # die Auswahl bewertet Gewinn- UND Schutz-Arm zusammen wie das Gate.
        model.eval()
        with torch.no_grad():
            out_h = model(Hs_t.to(device))
            l1_h = nn.functional.l1_loss(out_h, Hc_t.to(device)).item()
            out_h_id = model(Hc_id_t.to(device))
            l1_id_h = nn.functional.l1_loss(out_h_id, Hc_id_t.to(device)).item()
            mel_h = _mel_torch(out_h.reshape(len(out_h), -1))
            pen_h = float(torch.sigmoid(witness(mel_h[:, None, :, :])).mean())
        score_h = l1_h + lam * pen_h + lam_id * l1_id_h
        if score_h < best["score"]:
            best = {"score": score_h, "ep": ep, "state": {k: v.cpu().clone() for k, v in model.state_dict().items()}}
        if (ep + 1) % max(1, epochen // 4) == 0 or ep == epochen - 1:
            print(
                f"  {klasse}: ep {ep + 1}/{epochen} loss {total / max(1, len(Xs_t) // batch):.4f} hold {score_h:.4f}",
                flush=True,
            )

    model.load_state_dict(best["state"])
    model.eval()
    with torch.no_grad():
        out_s = model(Xs_t.to(device))
        out_h = model(Hs_t.to(device)) if len(Hs_t) else out_s
    ps = _witness_p_mittel(witness, _mel_torch(Xs_t.reshape(len(Xs_t), -1)), device) if len(Xs_t) else 0.0
    pc = _witness_p_mittel(witness, _mel_torch(Xc_t.reshape(len(Xc_t), -1)), device) if len(Xc_t) else 0.0
    pout_s = _witness_p_mittel(witness, _mel_torch(out_s.reshape(len(out_s), -1)), device)
    q_in = _witness_q(ps, pc, pout_s)
    q_hold = None
    if len(Hs_t):
        ps_h = _witness_p_mittel(witness, _mel_torch(Hs_t.reshape(len(Hs_t), -1)), device)
        pc_h = _witness_p_mittel(witness, _mel_torch(Hc_t.reshape(len(Hc_t), -1)), device)
        pout_h = _witness_p_mittel(witness, _mel_torch(out_h.reshape(len(out_h), -1)), device)
        q_hold = _witness_q(ps_h, pc_h, pout_h)

    report = {
        "schema": SCHEMA,
        "datum": date.today().isoformat(),
        "klasse": klasse,
        "seed": SEED,
        "epochen": epochen,
        "hidden": hidden,
        "stufen": stufen,
        "lambda_witness": lam,
        "lambda_id": lam_id,
        "n_regionen_gesamt": n_gesamt,
        "n_train": len(xs),
        "n_holdout": len(hs),
        "holdout_titel": holdout_titel,
        "beste_epoche": best["ep"] + 1,
        "holdout_score": round(float(best["score"]), 4),
        "q_witness_in_sample": round(q_in, 4) if q_in is not None else None,
        "q_witness_holdout": round(q_hold, 4) if q_hold is not None else None,
        "preflight_q_ge_0_5": bool(q_in is not None and q_in >= 0.5),
        "hinweis": "Report-only: Aktivierung erst nach Never-worsen-Gate (verify_witness_kopf_gate.py) + §4",
    }
    out_dir = OUT_BASE
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / f"witness_{klasse}_best.pt"
    torch.save({"state_dict": best["state"], "hidden": hidden, "stufen": stufen, "klasse": klasse}, ckpt)
    report["checkpoint"] = str(ckpt.relative_to(_ROOT))
    (out_dir / f"witness_{klasse}_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--klasse", default="groove_echo")
    ap.add_argument("--epochen", type=int, default=30)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--stufen", type=int, default=4)
    ap.add_argument("--lambda-witness", type=float, default=0.1)
    ap.add_argument(
        "--lambda-id", type=float, default=0.5, help="Identitäts-Anker-Gewicht (Never-worsen-Schutz im Loss)"
    )
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-regionen", type=int, default=0, help=">0 = Smoke (begrenzte Regionen)")
    args = ap.parse_args(argv)
    report = trainiere(
        args.klasse,
        args.epochen,
        args.hidden,
        args.stufen,
        args.lambda_witness,
        args.device,
        args.max_regionen,
        args.lambda_id,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
