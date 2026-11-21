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
  4. **musik_mos/Wohlklang-Term im Loss + Zwei-Metrik-Modellauswahl (DRT-8/10,
     SOTA-HANDOVER Punkt 34f, D-K3-95-Folge):** schaltbarer musik_mos-Proxy-Term —
     ein kleines differenzierbares Proxy (Log-Mel → MOS ∈ [1, 5]) zieht die
     Ausgabe per Hinge Richtung ziel-mos (singmos-Proxy-Muster in
     scripts/train_cantus.py); der nicht-differenzierbare musik_mos-Teacher
     (plugins/musik_mos_plugin, zertifizierter Zeuge) kalibriert das Proxy
     epoch-weise und liefert die Holdout-mos für die Modellauswahl nach
     defektlast UND musik_mos (Mindest-Schwelle JND 0,05 statt nur val_loss).
     Default AUS: ohne --lambda-mos > 0 bleibt jeder Lauf bit-identisch zum
     Vorgänger (§G5 (GEBOTE.md) Rückwärtskompatibilität).

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
import logging
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
MOS_JND = 0.05  # DRT-8/10: musik_mos-JND (defekt_delta; Ablations-Befund 2026-11-16)
MOS_ZIEL = 4.0  # Hinge-Ziel-mos des musik_mos-Proxy-Terms (Regularisierer-Richtwert)
NW_TOL = 0.02

logger = logging.getLogger(__name__)


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


def _spektral_abstand(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Mehrskaliger Spektral-Abstand (Wohlklang-Proxy) — Ablations-/Dämpfungs-Befund
    2026-11-16: L1 + Witness-BCE ziehen in Richtung „kein Defekt", aber nie in
    Richtung „mos-gut". Der Spektral-Abstand zum Clean misst die Klangtreue
    (das mos-relevante Ziel) und macht die Modellauswahl Zwei-Metrik-fähig
    (defektlast UND Wohlklang, DRT-8/10). Deterministisch, CPU-fähig."""
    total = torch.tensor(0.0, dtype=a.dtype, device=a.device)
    for n_fft in (2048, 1024, 512):
        hop = max(64, n_fft // 4)
        win = torch.hann_window(n_fft, dtype=a.dtype, device=a.device)
        sa = torch.stft(a, n_fft=n_fft, hop_length=hop, window=win, center=True, return_complex=True)
        sb = torch.stft(b, n_fft=n_fft, hop_length=hop, window=win, center=True, return_complex=True)
        total = total + nn.functional.l1_loss(sa.abs(), sb.abs())
    return total / 3.0


class MusikMOSProxy(nn.Module):
    """Kleines differenzierbares Proxy-Netz Log-Mel → MOS-Schätzung ∈ [1, 5].

    Das singmos-Proxy-Muster (scripts/train_cantus.py) für musik_mos
    (SOTA-HANDOVER Punkt 34f, DRT-8/10): der zertifizierte musik_mos-Teacher
    (plugins/musik_mos_plugin, Spearman 0,95) ist nicht differenzierbar — dieses
    Proxy trägt den Wohlklang-Gradienten in den Loss und wird epoch-weise gegen
    den Teacher kalibriert (detached, nie im Autograd-Pfad des Kopfes). Nur aktiv
    bei --lambda-mos > 0 (Default AUS, §G5 (GEBOTE.md) Rückwärtskompatibilität).
    """

    def __init__(self, n_mels: int = N_MELS, frames: int = FRAMES):
        super().__init__()
        self.n_mels = n_mels
        self.frames = frames
        self.body = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.head = nn.Sequential(nn.Linear(16 * 4 * 4, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """mel [B, 1, N_MELS, FRAMES] (Witness-Eingabelayout) → mos [B] ∈ (1, 5)."""
        h = self.body(mel).flatten(1)
        return 1.0 + 4.0 * torch.sigmoid(self.head(h)).squeeze(-1)


def _mos_proxy_verlust(mos: torch.Tensor, ziel: float) -> torch.Tensor:
    """Hinge Richtung ziel-mos (höher = besser): relu(ziel − mos).mean().

    Differenzierbar durch das Proxy (DRT-8/10). Mit Gewicht 0 wird der Term gar
    nicht aufgerufen — der Loss bleibt bit-identisch zum Vorgänger
    (§G5 (GEBOTE.md))."""
    return torch.relu(ziel - mos).mean()


def _mos_teacher_werte(wellen: np.ndarray, sr: int = SR) -> list[float] | None:
    """Nicht-differenzierbarer musik_mos-Teacher (plugins/musik_mos_plugin — der
    zertifizierte Wohlklang-Zeuge, Spearman 0,95): je Fenster ein MOS-Wert [N].
    None bei Ladefehler oder leeren Ergebnissen (§V6 (VERBOTEN.md):
    logger.warning statt Stillfehler). Nie im Autograd-Pfad."""
    wellen = np.asarray(wellen, dtype=np.float32)
    if wellen.ndim == 3:
        wellen = wellen.reshape(wellen.shape[0], -1)
    try:
        from plugins.musik_mos_plugin import get_musik_mos
    except Exception as exc:  # §V6 (VERBOTEN.md): dokumentierter Ersatzpfad
        logger.warning(
            "§V6 (VERBOTEN.md) ML→Ersatzpfad: musik_mos-Teacher nicht ladbar (%s) — "
            "Zwei-Metrik-Auswahl läuft ohne mos (Fallback defektlast)",
            exc,
        )
        return None
    ergebnisse: list[float] = []
    for i in range(int(wellen.shape[0])):
        try:
            ergebnisse.append(float(get_musik_mos().estimate_mos(wellen[i], sr).mos))
        except Exception as exc:
            logger.warning(
                "§V6 (VERBOTEN.md): musik_mos-Teacher-Inferenz fehlgeschlagen (%s) — Fenster übersprungen",
                exc,
            )
    return ergebnisse or None


def _zwei_metrik_auswahl(
    kandidaten: list[dict[str, Any]], mos_jnd: float = MOS_JND, nw_min: float | None = None
) -> dict[str, Any] | None:
    """Checkpoint-Auswahl nach defektlast UND musik_mos statt nur val_loss (DRT-8/10).

    Mindest-Schwelle (Pareto-konform): gewinnen kann nur, wer mit dem Holdout-mos
    höchstens mos_jnd unter dem besten gesehenen mos liegt — die reine
    defektlast-Auswahl opferte den Wohlklang (Ablations-Befund 2026-11-16:
    mos-Verluste 0,06–0,10 > JND 0,05). Unter den Schwellen-Kandidaten gewinnt
    die kleinste defektlast. Ohne messbaren mos gilt der alte Weg (kleinster
    Score, §V6 (VERBOTEN.md) Fallback).

    nw_min (Never-worsen-Schutz-Arm, Exit-1-Fix 2026-11-19): mit `--nw-min`
    (Default None = altes Verhalten, bit-identisch) gewinnen NUR Epochen,
    deren Holdout-never-worsen-Anteil ≥ nw_min liegt. Gibt es keine solche
    Epoche, wird kein Checkpoint ausgewählt; ein Fallback würde die gesetzte
    Schutzbedingung umgehen.
    """
    if not kandidaten:
        return None
    nw_kandidaten = kandidaten
    if nw_min is not None:
        nw_kandidaten = [k for k in kandidaten if k.get("nw") is not None and float(k["nw"]) >= float(nw_min)]
        if not nw_kandidaten:
            return None
    messbar = [k for k in nw_kandidaten if k.get("mos") is not None]
    if not messbar:
        return min(nw_kandidaten, key=lambda k: float(k["score"]))
    best_mos = max(float(k["mos"]) for k in messbar)
    schwellen_ok = [k for k in messbar if float(k["mos"]) >= best_mos - float(mos_jnd)]
    return min(schwellen_ok, key=lambda k: float(k["score"]))


def _kombinierter_loss(
    l1: torch.Tensor,
    pen: torch.Tensor,
    l1_id: torch.Tensor,
    spec_train: torch.Tensor,
    mos_term: torch.Tensor | None,
    lam: float,
    lam_id: float,
    lam_spec: float,
    lam_mos: float,
) -> torch.Tensor:
    """Gesamt-Loss-Assemblage des Witness-Kopfes.

    mos_term None oder lam_mos == 0 ⇒ exakt der bisherige Loss
    (l1 + λ·pen + λ_id·l1_id + λ_spec·spec_train) ohne Zusatz-Operation —
    bestehende Läufe ohne --lambda-mos bleiben bit-identisch
    (§G5 (GEBOTE.md) Rückwärtskompatibilität).
    """
    loss = l1 + lam * pen + lam_id * l1_id + lam_spec * spec_train
    if mos_term is not None and lam_mos > 0.0:
        loss = loss + lam_mos * mos_term
    return loss


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


def _never_worsen_anteil(
    witness: nn.Module,
    clean_output: torch.Tensor,
    clean_reference: torch.Tensor,
    device: str,
) -> float | None:
    if len(clean_output) == 0 or len(clean_output) != len(clean_reference):
        return None
    with torch.no_grad():
        mel_out = _mel_torch(clean_output.reshape(len(clean_output), -1))
        mel_reference = _mel_torch(clean_reference.reshape(len(clean_reference), -1))
        p_out = torch.sigmoid(witness(mel_out.to(device)[:, None, :, :])).view(len(clean_output), -1).mean(dim=1)
        p_reference = (
            torch.sigmoid(witness(mel_reference.to(device)[:, None, :, :])).view(len(clean_reference), -1).mean(dim=1)
        )
    return float((p_out <= p_reference + NW_TOL).float().mean())


def trainiere(
    klasse: str,
    epochen: int,
    hidden: int,
    stufen: int,
    lam: float,
    device: str,
    max_regionen: int = 0,
    lam_id: float = 0.5,
    lam_lokal: float = 0.0,
    lam_spec: float = 0.3,
    lam_mos: float = 0.0,
    mos_ziel: float = MOS_ZIEL,
    nw_min: float | None = None,
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
    # musik_mos-Proxy-Term (DRT-8/10, SOTA-HANDOVER Punkt 34f): Default AUS —
    # ohne --lambda-mos > 0 entsteht kein Proxy, kein Teacher-Aufruf, der Loss
    # bleibt bit-identisch zum Vorgänger (§G5 (GEBOTE.md)). Mit Flag: kleines
    # differenzierbares Proxy (singmos-Proxy-Muster, scripts/train_cantus.py),
    # epoch-weise gegen den nicht-differenzierbaren musik_mos-Teacher kalibriert.
    mos_proxy: MusikMOSProxy | None = None
    mos_opt: torch.optim.Adam | None = None
    kandidaten: list[dict[str, Any]] = []
    if lam_mos > 0.0:
        _seed_all(SEED)  # deterministische Proxy-Initialisierung (§G5 (GEBOTE.md))
        mos_proxy = MusikMOSProxy().to(device)
        mos_opt = torch.optim.Adam(mos_proxy.parameters(), lr=1e-3)
    batch = 8
    for ep in range(epochen):
        model.train()
        perm = torch.randperm(len(Xs_t))
        total = 0.0
        for i in range(0, len(Xs_t), batch):
            idx = perm[i : i + batch]
            s_b, c_b = Xs_t[idx].to(device), Xc_t[idx].to(device)
            out = model(s_b)
            if lam_lokal > 0.0:
                # Lokalisierter Loss (transport_bump-Befund 2026-11-15, DRT-3-Mechanismus):
                # der globale L1 gibt kurzen transienten Impulsen kaum Gradient — die
                # |schaden−clean|-Maske (aus den Trainings-Paaren, KEIN Leak) gewichtet den
                # Defekt-Ort proportional. Maske gedeckelt, damit kein Einzel-Sample dominiert.
                maske = (s_b - c_b).abs()
                maske = torch.clamp(maske / (maske.mean(dim=2, keepdim=True) + 1e-6), 0.0, 8.0)
                l1 = (maske * (out - c_b).abs()).mean() / (maske.mean() + 1e-8)
            else:
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
            # Wohlklang-Term IM TRAININGS-LOSS (lambda-Sweep-Befund 2026-11-16): die
            # Auswahl-Gewichtung allein konvergiert bei mos-Verlust 0,057 (knapp über JND) —
            # der Gradient muss Wohlklang optimieren, nicht nur die Auswahl (§V7 (copilot-instructions.md)).
            spec_train = _spektral_abstand(out.reshape(len(out), -1), c_b.reshape(len(c_b), -1))
            # musik_mos-Proxy-Term im Loss (DRT-8/10): Hinge Richtung ziel-mos durch
            # das differenzierbare Proxy — der Wohlklang-Gradient zieht die Ausgabe
            # nicht nur weg vom Defekt, sondern Richtung „mos-gut" (Ablations-Befund
            # 2026-11-16: L1 + Witness-BCE allein opfern den musik_mos-Gewinn).
            mos_term = (
                _mos_proxy_verlust(mos_proxy(mel_out[:, None, :, :]), mos_ziel) if mos_proxy is not None else None
            )
            loss = _kombinierter_loss(l1, pen, l1_id, spec_train, mos_term, lam, lam_id, lam_spec, lam_mos)
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
            spec_h = float(
                _spektral_abstand(out_h.reshape(len(out_h), -1), Hc_t.to(device).reshape(len(Hc_t), -1)).item()
            )
        score_h = l1_h + lam * pen_h + lam_id * l1_id_h + lam_spec * spec_h
        mos_h: float | None = None
        if mos_proxy is None and nw_min is None:
            # Unveränderter Default-Auswahlpfad: kleinster Holdout-Score
            # (bit-identisch ohne Flag, §G5 (GEBOTE.md)).
            if score_h < best["score"]:
                best = {
                    "score": score_h,
                    "ep": ep,
                    "state": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                }
        else:
            # --nw-min erfasst Epochen auch dann als Kandidaten, wenn der
            # optionale musik_mos-Proxy ausgeschaltet ist.
            lehrer = _mos_teacher_werte(out_h.detach().cpu().numpy()) if mos_proxy is not None else None
            mos_h = float(np.mean(lehrer)) if lehrer else None
            # NW-Schutzarm (DRT-2/5): nur Clean→Clean am Holdout zählt; der
            # beschädigte Eingang gehört zum Gewinn-Arm und darf NW nicht vortäuschen.
            nw_h = _never_worsen_anteil(witness, out_h_id, Hc_id_t, device)
            kandidaten.append(
                {
                    "ep": ep,
                    "score": float(score_h),
                    "mos": mos_h,
                    "nw": nw_h,
                    "state": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                }
            )
            # Proxy-Kalibrierung gegen den Teacher (detached; singmos-Muster):
            # MSE(mos_proxy(mel_h), teacher(out_h)) hält das Proxy am zertifizierten
            # Zeugen — ohne Teacher bleibt es Regularisierer (§V6 (VERBOTEN.md):
            # Warnung in _mos_teacher_werte, kein Stillfehler).
            if lehrer and mos_opt is not None and mos_proxy is not None:
                ziel_mos_t = torch.tensor(lehrer, dtype=torch.float32, device=device)
                mos_pred_h = mos_proxy(mel_h[:, None, :, :])
                kal = nn.functional.mse_loss(mos_pred_h, ziel_mos_t)
                mos_opt.zero_grad()
                kal.backward()
                mos_opt.step()
        if (ep + 1) % max(1, epochen // 4) == 0 or ep == epochen - 1:
            print(
                f"  {klasse}: ep {ep + 1}/{epochen} loss {total / max(1, len(Xs_t) // batch):.4f} hold {score_h:.4f}",
                flush=True,
            )
            if mos_proxy is not None:
                print(f"    mos-hold {mos_h:.3f}" if mos_h is not None else "    mos-hold n/a", flush=True)

    if mos_proxy is not None or nw_min is not None:
        # Auswahl am Epochen-Ende: NW-Minimum zuerst, dann ggf. MOS-JND und Score.
        gewaehlt = _zwei_metrik_auswahl(kandidaten, nw_min=nw_min)
        if gewaehlt is not None:
            best = {
                "score": float(gewaehlt["score"]),
                "ep": int(gewaehlt["ep"]),
                "state": gewaehlt["state"],
                "mos": gewaehlt["mos"],
                "nw": gewaehlt["nw"],
            }
        elif nw_min is not None:
            nw_werte = [float(k["nw"]) for k in kandidaten if k.get("nw") is not None]
            nw_max = max(nw_werte) if nw_werte else None
            nw_max_text = f"{nw_max:.4f}" if nw_max is not None else "nicht messbar"
            raise RuntimeError(
                f"Kein Witness-Kopf-Checkpoint für {klasse} erfüllt --nw-min={nw_min}; "
                f"bestes nw={nw_max_text}, "
                f"geeignete Epochen={sum(value >= nw_min for value in nw_werte)}/{len(kandidaten)}; "
                "der vorhandene Checkpoint bleibt unverändert."
            )
        else:
            logger.warning("§V6 (VERBOTEN.md): Zwei-Metrik-Auswahl ohne Kandidaten — kein Checkpoint gewählt")
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
        "lambda_lokal": lam_lokal,
        "lambda_spektral": lam_spec,
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
    if mos_proxy is not None or nw_min is not None:
        if mos_proxy is not None:
            report["lambda_mos"] = lam_mos
            report["mos_ziel"] = mos_ziel
            report["mos_jnd"] = MOS_JND
            report["auswahl_modus"] = "zwei_metrik_drt_8_10"
            report["holdout_mos"] = round(float(best["mos"]), 4) if best.get("mos") is not None else None
            report["drt_8_10"] = (
                "Zwei-Metrik-Modellauswahl (defektlast UND musik_mos, Mindest-Schwelle JND) — "
                "D-K3-95-Folge; Aktivierung bleibt dem Never-worsen-Gate + §4 vorbehalten"
            )
        else:
            report["auswahl_modus"] = "defektlast_unter_nw_min"
        report["nw_min"] = nw_min
        report["holdout_never_worsen"] = round(float(best["nw"]), 4) if best.get("nw") is not None else None
        if nw_min is not None:
            report["nw_auswahl_erfuellt"] = bool(best.get("nw") is not None and float(best["nw"]) >= float(nw_min))
        if kandidaten:
            report["kandidaten_zwischenstaende"] = [
                {
                    "ep": int(k["ep"]) + 1,
                    "score": round(float(k["score"]), 4),
                    "mos": round(float(k["mos"]), 4) if k["mos"] is not None else None,
                    "nw": round(float(k["nw"]), 4) if k["nw"] is not None else None,
                }
                for k in kandidaten
            ]
    out_dir = OUT_BASE
    out_dir.mkdir(parents=True, exist_ok=True)
    # Smoke-Schutz (2026-11-16): --max-regionen > 0 schreibt einen separaten
    # Smoke-Pfad — der kanonische Checkpoint wurde 2026-11-16 von einem
    # 1-Epochen-Smoke überschrieben (groove_echo, 7,68 MB -> 42 KB).
    _ist_smoke = max_regionen > 0
    ckpt_name = f"smoke_witness_{klasse}_best.pt" if _ist_smoke else f"witness_{klasse}_best.pt"
    ckpt = out_dir / ckpt_name
    torch.save({"state_dict": best["state"], "hidden": hidden, "stufen": stufen, "klasse": klasse}, ckpt)
    report["checkpoint"] = str(ckpt.relative_to(_ROOT))
    if _ist_smoke:
        report["smoke"] = True
    (out_dir / f"{ckpt_name.replace('.pt', '_report')}.json").write_text(
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
    ap.add_argument(
        "--lambda-lokal",
        type=float,
        default=0.0,
        help=">0 = lokalisierter L1 über die |schaden-clean|-Maske (transiente Defekte)",
    )
    ap.add_argument(
        "--lambda-spektral",
        type=float,
        default=0.3,
        help="Wohlklang-Term (MR-Spektral-Abstand zum Clean) im Holdout-Score — Zwei-Metrik-Modellauswahl (DRT-8/10)",
    )
    ap.add_argument(
        "--lambda-mos",
        type=float,
        default=0.0,
        help="musik_mos-Proxy-Term im Trainings-Loss (DRT-8/10); Default AUS — ohne Flag bit-identisch (§G5 (GEBOTE.md))",
    )
    ap.add_argument(
        "--mos-ziel", type=float, default=MOS_ZIEL, help="Hinge-Ziel-mos des Proxy-Terms (nur mit --lambda-mos > 0)"
    )
    ap.add_argument(
        "--nw-min",
        type=float,
        default=None,
        help=(
            "Harte Never-worsen-Schranke der Modellauswahl: nur Epochen mit Holdout-nw ≥ Schwelle gewinnen; "
            "ohne passende Epoche Abbruch ohne Checkpoint-Überschreibung. Funktioniert auch ohne --lambda-mos."
        ),
    )
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-regionen", type=int, default=0, help=">0 = Smoke (begrenzte Regionen)")
    args = ap.parse_args(argv)
    if args.nw_min is not None and not 0.0 <= args.nw_min <= 1.0:
        ap.error("--nw-min muss zwischen 0 und 1 liegen")
    report = trainiere(
        args.klasse,
        args.epochen,
        args.hidden,
        args.stufen,
        args.lambda_witness,
        args.device,
        args.max_regionen,
        args.lambda_id,
        args.lambda_lokal,
        args.lambda_spektral,
        args.lambda_mos,
        args.mos_ziel,
        nw_min=args.nw_min,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
