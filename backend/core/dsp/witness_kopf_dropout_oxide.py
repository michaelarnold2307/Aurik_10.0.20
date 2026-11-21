"""Witness-Kopf dropout_oxide — Produktions-DSP für den Reparatur-Kopf (DRT-7-Verdrahtung).

D-K3-95-Folge (2026-11-21). Der Kopf `models/klassen_koepfe/witness_dropout_oxide_best.pt`
ist defekt-spezifisch auf k398-Regionen trainiert (leak-frei, ehrlich: Detektor-AUC
0,79 — die Witness-Qualität ist die dokumentierte Schwäche dieses Kanals). Die
Aktivierungs-Entscheidung läuft über das Produktions-Pfad-Gate (DRT-7, Spec 26):
`scripts/verify_witness_kopf_gate.py --klasse dropout_oxide --mos` misst Gewinn-Arm
(Witness-Q ≥ 0,5) und Schutz-Arm über DIESE Produktions-Semantik (Trigger + Guard).
Beleg wird nach Bestehen in `docs/reports/current/<datum>_witness_kopf_gate_dropout_oxide.json`
gebucht; bis dahin bleibt der Kanal hinter `use_witness_kopf_dropout_oxide`
(Default AUS = bit-identisch) verdrahtet, aber ohne §4-Freigabe.

Sicherheits-Vertrag (Muster `witness_kopf_groove_echo.py`, Musikschutz AGENTS.md):
- Fail-closed: fehlendes Checkpoint/Witness ⇒ None (Aufrufer = Pass-Through,
  §V6 (copilot-instructions.md) mit logger.warning).
- Trigger-Maske: der Kopf wirkt NUR dort, wo der gelernte Witness
  dropout_oxide-Schaden detektiert (P > threshold) — außerhalb bit-identischer
  Passthrough.
- Soft-Fades: 200 ms Hann-Kanten (§III (copilot-instructions.md): 6 dB/200 ms
  Soft-Knee) statt harter Masken-Kanten.
- Never-worsen-Guard: Kandidat wird nur übernommen, wenn der Witness strikt
  besser wird (P_kandidat ≤ P_input − margin); sonst None.
- NaN/Inf-Schutz, Layout: Mono (N,) float32; Stereo verarbeitet der Aufrufer
  kanalweise.
- Deterministisch: feste Seeds nirgends nötig (reine Inferenz, kein Zufall).
- Aktivierung NUR über `music_model_flags.use_witness_kopf_dropout_oxide`
  (Default AUS); der Flip ist klangwirksam und läuft über §4.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

_KOPF_CKPT = _ROOT / "models" / "klassen_koepfe" / "witness_dropout_oxide_best.pt"
_WITNESS_CKPT = _ROOT / "models" / "witness_dropout_oxide" / "witness.pt"
_TARGET_SR = 44100
_FENSTER = 44100  # 1 s
_HOP = 22050  # 0,5 s OLA-Schritt
# Trainings-Normalisierung (KEINE Produktions-Schwelle): der Wert lebt im Trainer
# (scripts/train_witness_kopf.py::RMS_ZIEL) und wird für die Training→Produktion-
# Parität IMPORTIERT statt dupliziert (§V25 (copilot-instructions.md): kein
# hartcodierter Schwellwert; §G5 (copilot-instructions.md)-Determinismus: eine Quelle der Wahrheit).
from scripts.train_witness_kopf import RMS_ZIEL as _RMS_ZIEL

_FADE = 8820  # 200 ms @ 44,1 kHz (§III 6 dB/200 ms)
_DEFAULT_THRESHOLD = 0.5
_NW_MARGIN = 0.02
_MAX_TRIGGER_ANTEIL = 0.5  # DRT-11: breiter Trigger = Material-Kontext
_MIN_FENSTER_KONTEXT = 3  # DRT-11: Sperre erst ab mehreren Fenstern


def _mel_torch(audio: torch.Tensor) -> torch.Tensor:
    """Produktions-konsistente differenzierbare Mel (slaney + power_to_db, 64 Frames).

    Identisch zum Trainer `scripts/train_witness_kopf.py` — die Parität der
    Witness-Eingabe zwischen Training, Gate und Produktion ist der Kern des
    D-K3-95-Kanals (Befund 2026-11-15: ohne slaney/10·log10 sättigt das Witness).
    """
    import librosa.filters  # pylint: disable=import-outside-toplevel

    window = torch.hann_window(2048, dtype=audio.dtype, device=audio.device)
    stft = torch.stft(audio, n_fft=2048, hop_length=512, window=window, center=True, return_complex=True)
    mag = stft.abs().pow(2.0)
    mel_basis = (
        torch.from_numpy(
            librosa.filters.mel(sr=_TARGET_SR, n_fft=2048, n_mels=128, fmin=0.0, fmax=None, norm="slaney")  # type: ignore[attr-defined]
        )
        .to(audio.dtype)
        .to(audio.device)
    )
    mel = 10.0 * torch.log10(torch.clamp(torch.matmul(mel_basis, mag), min=1e-10))
    mel = mel[..., :64] if mel.shape[-1] >= 64 else torch.nn.functional.pad(mel, (0, 64 - mel.shape[-1]))
    return mel


def _witness_p(witness: Any, x: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-8)
    t = torch.from_numpy((x / rms * _RMS_ZIEL).astype(np.float32))[None]
    with torch.no_grad():
        return float(torch.sigmoid(witness(_mel_torch(t)[:, None])).mean())


def _kopf_out(kopf: Any, x: np.ndarray) -> np.ndarray:
    """OLA-Inferenz des Kopfs (Muster _kopf_ola des Musik-Declip-Kopfs)."""
    win = np.hanning(_FENSTER).astype(np.float32)
    xf = np.asarray(x, dtype=np.float32)
    pad = np.pad(xf, (_HOP, _HOP), mode="reflect")
    out = np.zeros(len(pad), dtype=np.float64)
    norm = np.zeros(len(pad), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, len(pad) - _FENSTER + 1, _HOP):
            seg = pad[start : start + _FENSTER]
            rms = float(np.sqrt(np.mean(seg.astype(np.float64) ** 2)) + 1e-8)
            t = torch.from_numpy((seg / rms * _RMS_ZIEL).astype(np.float32))[None, None]
            rec = kopf(t).cpu().numpy()[0, 0, :].astype(np.float64) * rms / _RMS_ZIEL
            out[start : start + _FENSTER] += rec * win
            norm[start : start + _FENSTER] += win * win
    out = out / np.maximum(norm, 1e-6)
    aus: np.ndarray = np.asarray(out[_HOP : _HOP + len(xf)], dtype=np.float32)
    return aus


def witness_kopf_dropout_oxide(
    audio: np.ndarray,
    sample_rate: int,
    threshold: float = _DEFAULT_THRESHOLD,
) -> np.ndarray | None:
    """dropout_oxide-Reparatur via Witness-Kopf; None = Pass-Through (fail-closed).

    Args:
        audio: Mono-Waveform (N,) float32.
        sample_rate: Abtastrate (44,1 kHz erwartet; sonst Ersatzpfad).
        threshold: Witness-Detektionsschwelle (P > threshold ⇒ Kopf aktiv).

    Returns:
        Repariertes Mono-Signal (N,) float32 — oder None (Ersatzpfad).
    """
    from scipy.signal import resample_poly  # pylint: disable=import-outside-toplevel

    if not _KOPF_CKPT.is_file():
        logger.warning("Witness-Kopf dropout_oxide: Checkpoint fehlt (%s) — Ersatzpfad.", _KOPF_CKPT)
        return None
    if not _WITNESS_CKPT.is_file():
        logger.warning("Witness-Kopf dropout_oxide: Witness-Checkpoint fehlt (%s) — Ersatzpfad.", _WITNESS_CKPT)
        return None
    if int(sample_rate) <= 0:
        logger.warning("Witness-Kopf dropout_oxide: sr=%d ungültig — Ersatzpfad.", sample_rate)
        return None
    try:
        from scripts.train_witness_detektor import WitnessCRNN  # pylint: disable=import-outside-toplevel
        from scripts.train_witness_kopf import WitnessKopf  # pylint: disable=import-outside-toplevel

        x = np.asarray(audio, dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        sr = int(sample_rate)
        if sr != _TARGET_SR:
            g = int(np.gcd(sr, _TARGET_SR))
            x = resample_poly(x.astype(np.float64), _TARGET_SR // g, sr // g).astype(np.float32)
        laenge_in = len(x)
        # Training-Parität (§G5 (copilot-instructions.md)): Trainer/Gate padden kurze
        # k398-Regionen (0,1 s) mit Nullen auf 1 s — ohne dieses Pad sieht das Witness
        # in der Produktion eine andere Repräsentation (Befund 2026-11-15: P=0,0 statt 0,92).
        if len(x) < _FENSTER:
            x = np.pad(x, (0, _FENSTER - len(x)))

        ckpt = torch.load(str(_KOPF_CKPT), map_location="cpu", weights_only=False)
        kopf = WitnessKopf(int(ckpt.get("hidden", 64)), int(ckpt.get("stufen", 4)))
        kopf.load_state_dict(ckpt["state_dict"])
        kopf.eval()
        wd = torch.load(str(_WITNESS_CKPT), map_location="cpu", weights_only=False)
        witness = WitnessCRNN(int(wd["n_mels"]))
        witness.load_state_dict(wd["state_dict"])
        witness.eval()

        theta = float(np.clip(threshold, 0.05, 0.999))
        kopf_signal = _kopf_out(kopf, x)

        # Trigger-Maske: Witness-Detektion je 1-s-Fenster (0,5 s Hop) + 200 ms Fades.
        maske = np.zeros(len(x), dtype=np.float32)
        win = 0.5 * (1 - np.cos(np.pi * np.arange(_FADE) / _FADE)).astype(np.float32)
        p_werte: list[float] = []
        for start in range(0, max(1, len(x) - _FENSTER + 1), _HOP):
            seg = x[start : start + _FENSTER]
            if len(seg) < _FENSTER:
                break
            p = _witness_p(witness, seg)
            p_werte.append(p)
            if p > theta:
                lo = max(0, start)
                hi = min(len(x), start + _FENSTER)
                fade = _FADE if start > 0 else 0
                if fade:
                    maske[lo : lo + fade] = np.maximum(maske[lo : lo + fade], win[: min(fade, hi - lo)])
                    maske[lo + fade : hi] = 1.0
                else:
                    maske[lo:hi] = 1.0

        # DRT-11 (Spec 26, 2026-11-15): Kontext-Sperre — ein breiter, homogener Trigger
        # ist Material-Kontext, kein lokaler k398-Defekt. Sperre ⇒ Bypass.
        anteil_trigger = sum(1 for p in p_werte if p > theta) / max(1, len(p_werte))
        if len(p_werte) >= _MIN_FENSTER_KONTEXT and anteil_trigger > _MAX_TRIGGER_ANTEIL:
            logger.info(
                "Witness-Kopf dropout_oxide: Kontext-Sperre (Trigger-Anteil %.2f > %.2f) — Umgehung (DRT-11).",
                anteil_trigger,
                _MAX_TRIGGER_ANTEIL,
            )
            return None
        if not np.any(maske > 0):
            logger.info(
                "Witness-Kopf dropout_oxide: kein Schaden detektiert (max P=%.4f) — Durchreichung.",
                max(p_werte) if p_werte else 0.0,
            )
            return None

        kandidat = maske * kopf_signal + (1.0 - maske) * x

        # Never-worsen-Guard: nur bei striktem Witness-Gewinn übernehmen.
        p_in = float(np.mean(p_werte)) if p_werte else 0.0
        p_out = 0.0
        n = 0
        for start in range(0, max(1, len(kandidat) - _FENSTER + 1), _HOP):
            seg = kandidat[start : start + _FENSTER]
            if len(seg) < _FENSTER:
                break
            p_out += _witness_p(witness, seg)
            n += 1
        p_out = p_out / max(1, n)
        if p_out > p_in - _NW_MARGIN:
            logger.info(
                "Witness-Kopf dropout_oxide: Kandidat abgelehnt (Witness P_out=%.4f nicht strikt < P_in=%.4f − %.2f) — Eingang bleibt",
                p_out,
                p_in,
                _NW_MARGIN,
            )
            return None

        # Rückgabe auf die Eingangslänge schneiden — das 1-s-Pad ist nur
        # Inferenz-Parität, kein Ausgabe-Artefakt (Phase 07 erwartet dieselbe Shape).
        kandidat = kandidat[:laenge_in]
        if sr != _TARGET_SR:
            g = int(np.gcd(sr, _TARGET_SR))
            kandidat = resample_poly(kandidat.astype(np.float64), sr // g, _TARGET_SR // g)[: len(audio)]
        aus: np.ndarray = np.asarray(np.clip(kandidat, -1.0, 1.0), dtype=np.float32)
        return aus
    except Exception as exc:
        logger.warning("Witness-Kopf dropout_oxide fehlgeschlagen (%s) — Ersatzpfad.", exc)
        return None
