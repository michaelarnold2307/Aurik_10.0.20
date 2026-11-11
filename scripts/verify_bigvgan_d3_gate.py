"""scripts/verify_bigvgan_d3_gate.py — D3-2-Gate: BigVGAN-F3-Finetune gegen die Basis.

Der Trainer ``scripts/train_bigvgan_f3.py`` schreibt seit jeher die Auflage in
seinen Report (``gate_hint``): **„MUSDB-ΔSDR ≥ +2 dB via
scripts/validate_hr_v1.py NACH dem Lauf messen“**. Diese Auflage ist für einen
Mel-Vocoder **nicht messbar**: BigVGAN rekonstruiert die Wellenform aus Mel-
Bändern und erzeugt dabei eine neue Phase — der zeitliche SDR liegt für Basis
und Finetune alike bei ~0 dB (auch nach Peilungs-Suche). ``validate_hr_v1.py``
misst deshalb zu Recht **af** (Click-/Pre-Echo-Proxy) und **HNR**, nicht SDR.

Dieses Skript misst das, was der Produktionspfad tatsächlich tut:
``plugins.bigvgan_v2_plugin.apply_hr_v1_additive`` (5-%-Längen-Deckel,
``additive_synthesis_gate``, 200-ms-Rampe) — einmal mit der **Basis** und
einmal mit dem **F3-Finetune**. Verglichen werden die vertraglich vereinbarten
Zeugen (af, HNR, musik_mos, defektlast), nicht ein für Phasen-Rekonstruktion
unzulässiger SDR.

Zwei F3-Varianten werden gemessen:

``f3_prod``
    F3-Checkpoint mit der **Produktions-Mel** (``librosa.power_to_db(ref=max)``,
    Bereich −80…0 dB). Das ist der heute ohne Code-Änderung deploybare Pfad.
``f3_trainmel``
    Derselbe Checkpoint mit der **Trainings-Mel** (``log(abs(STFT))`` aus
    ``bigvgan.meldataset.mel_spectrogram``, Bereich ≈ −12…+2). Der Trainer
    optimiert auf diese Verteilung; die Produktion speist eine andere ein.
    Diese Variante trennt Modellgüte von Mel-Verteilungs-Drift (D-K3-85-Klasse:
    erst das richtige Artefakt messen, dann urteilen).

Gate (fail-closed, dokumentierte Toleranzen):
  * **Never-worsen je Track** gegen die Basis: af ≥ Basis − 0,02;
    HNR ≥ Basis − 0,5 dB; musik_mos ≥ Basis − 0,05; defektlast ≤ Basis + 0,05.
  * **Vorteil**: auf ≥ 2 von 3 Tracks muss F3 die Basis um mehr als die
    Toleranz schlagen (af +0,02, HNR +0,5 dB, musik_mos +0,05 oder
    defektlast −0,05).
  * Ein Kandidat, der auf einem Track **nicht angewandt** wurde
    (``applied=False``, z. B. Gate gibt kein Band frei), kann keinen Vorteil
    belegen und fällt durch.

Exit-Codes: 0 = mindestens eine F3-Variante besteht, 1 = Messung gültig,
Gate nicht bestanden, 2 = Messung nicht möglich (Setup-Fehler).

CPU-only (§5 Handover: keine GPU-Kette stören), deterministisch (§G5 (copilot-instructions.md)):
feste Track-Auswahl über Seed 42, keine Zeit-/Zufallsentscheidung in der Logik.

Kein Deploy, kein Flag-Flip: ein bestandenes Gate ist die Voraussetzung für
das §4-Verfahren, nicht dessen Ersatz.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.validate_hr_v1 import _af_proxy, _hnr_db

logger = logging.getLogger("verify_bigvgan_d3_gate")

SCHEMA = "aurik.bigvgan_d3_gate/1"
MUSDB_TEST = _ROOT / "data" / "musdb18hq" / "test"
REPORT_DIR = _ROOT / "docs" / "reports" / "current"
SR = 48000
SR_MODELL = 44100

AF_TOLERANZ = 0.02
HNR_TOLERANZ_DB = 0.5
JND = 0.05
BENEFIT_MIN_TRACKS = 2

BASIS_CHECKPOINT = "models/bigvgan/bigvgan_v2.pth"
F3_CHECKPOINT = "output/f3_bigvgan/best.pt"


def _cpu_guard() -> str | None:
    """Die Messung darf keine GPU-Kette belegen (Handover §5)."""
    try:
        import torch

        if torch.cuda.is_available():
            return "GPU sichtbar — D3-2-Gate läuft ausschließlich CPU-only (keine Trainings-Kette stören)."
    except Exception:  # pragma: no cover - torch fehlt → CPU ist ohnehin der einzige Pfad
        return None
    return None


def _sha256(pfad: Path) -> str:
    h = hashlib.sha256()
    with pfad.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _waehle_tracks(anzahl: int, seed: int) -> list[Path]:
    """Deterministische Track-Auswahl (sortiert, Seed 42, §G5 (copilot-instructions.md))."""
    if not MUSDB_TEST.is_dir():
        return []
    tracks = sorted(p for p in MUSDB_TEST.iterdir() if p.is_dir() and (p / "mixture.wav").is_file())
    if not tracks:
        return []
    rng = random.Random(seed)
    gewaehlt = rng.sample(tracks, min(anzahl, len(tracks)))
    return sorted(gewaehlt)


def _lade_segment(track: Path, seconds: int, *, suchfenster_s: float = 60.0) -> np.ndarray:
    """Lädt ein deterministisches, nicht-stilles Segment (48 kHz mono, normiert).

    Stille am Track-Anfang ist keine Modellgüte: das lauteste 1-s-Fenster in den
    ersten 60 s bestimmt den Start (Ties → erstes Fenster), danach werden
    ``seconds`` geschnitten. Kein RNG, kein time.time.
    """
    import librosa

    y, _ = librosa.load(str(track / "mixture.wav"), sr=SR, mono=True, duration=suchfenster_s)
    y = np.asarray(y, dtype=np.float32)
    seg_len = int(seconds) * SR
    if y.size < SR:
        raise ValueError(f"Track zu kurz: {track}")
    hop = SR // 2
    # Das Segment muss die volle Laenge haben — sonst faellt das erste
    # HR-V1-Fenster in den Rand-Pad (Stille) und die Rohsynthese misst Null.
    max_start = max(0, min(y.size, int(suchfenster_s * SR)) - seg_len)
    if max_start <= 0:
        start = 0
    else:
        starts = list(range(0, max_start + 1, hop))
        rms = np.array([float(np.mean(y[s : s + seg_len] ** 2)) for s in starts])
        start = starts[int(np.argmax(rms))]
    seg = y[start : start + seg_len]
    if seg.size < seg_len:
        seg = np.pad(seg, (0, seg_len - seg.size), mode="edge")
    peak = float(np.max(np.abs(seg)))
    if peak > 0:
        seg = seg / peak
    return np.asarray(seg, dtype=np.float32)


def _mel_prod(audio: np.ndarray, sr: int) -> np.ndarray:
    """Produktions-Mel (unverändert der Plugin-Pfad)."""
    from plugins.bigvgan_v2_plugin import BigVGANv2Plugin

    plugin = object.__new__(BigVGANv2Plugin)  # __init__ lädt das Basis-Modell — hier nicht nötig
    return np.asarray(plugin._compute_mel(audio, sr), dtype=np.float32)[None, :, :]


def _mel_train(audio: np.ndarray, sr: int) -> np.ndarray:
    """Trainings-Mel des F3-Checkpoints (``bigvgan.meldataset.mel_spectrogram``)."""
    import librosa
    import torch
    from bigvgan.meldataset import mel_spectrogram

    a44 = librosa.resample(audio, orig_sr=sr, target_sr=SR_MODELL)
    return (
        mel_spectrogram(
            torch.from_numpy(np.asarray(a44, dtype=np.float32)[None, :]),
            n_fft=2048,
            num_mels=128,
            sampling_rate=SR_MODELL,
            hop_size=512,
            win_size=2048,
            fmin=0,
            fmax=None,
            center=False,
        )
        .numpy()
        .astype(np.float32)
    )


@dataclass
class SynthStatistik:
    """Beobachtbare Rohdaten je Kandidat (Diagnose, keine Metrik)."""

    calls: int = 0
    rms_summe: float = 0.0
    peak: float = 0.0
    clip_samples: int = 0
    samples: int = 0
    mel_mittel: list[float] = field(default_factory=list)
    mel_std: list[float] = field(default_factory=list)

    def als_dict(self) -> dict[str, float | int]:
        calls = max(self.calls, 1)
        return {
            "aufrufe": self.calls,
            "rms_mittel": round(self.rms_summe / calls, 5),
            "peak": round(self.peak, 5),
            "clip_anteil": round(self.clip_samples / max(self.samples, 1), 6),
            "mel_mittel": round(float(np.mean(self.mel_mittel)) if self.mel_mittel else 0.0, 3),
            "mel_std": round(float(np.mean(self.mel_std)) if self.mel_std else 0.0, 3),
        }


class Kandidat:
    """Torch-Checkpoint + Mel-Vertrag als ``synthesize_audio``-Ersatz."""

    def __init__(self, name: str, checkpoint: Path, mel_modus: str) -> None:
        import torch

        from scripts.train_bigvgan_f3 import build_model

        self.name = name
        self.checkpoint = checkpoint
        self.mel_modus = mel_modus
        self.model = build_model(torch.device("cpu"), checkpoint)
        self.model.eval()
        self.statistik = SynthStatistik()
        # Rohsynthese je Aufruf: das Gate misst die Modellgüte VOR dem
        # additiven Schutz. Ohne diese Trennung misst man nur das unveränderte
        # Baseline-Signal (das Gate gibt oft 0 Bänder frei).
        self.letzte: list[tuple[np.ndarray, np.ndarray]] = []
        self.letzte_mel: list[tuple[float, float]] = []

    def __call__(self, audio: np.ndarray, sr: int, *, mode: str = "studio2026") -> Any:
        import librosa
        import torch

        from plugins.bigvgan_v2_plugin import BigVGANv2Plugin, VocoderResult

        if int(sr) != SR:
            raise ValueError(f"BigVGAN-D3-Gate erwartet 48 kHz, erhalten: {sr}")
        audio = np.asarray(audio, dtype=np.float32)
        mel = _mel_train(audio, sr) if self.mel_modus == "train" else _mel_prod(audio, sr)
        self.statistik.mel_mittel.append(float(np.mean(mel)))
        self.statistik.mel_std.append(float(np.std(mel)))
        self.letzte_mel.append((float(np.mean(mel)), float(np.std(mel))))
        with torch.no_grad():
            out = self.model(torch.from_numpy(mel)).squeeze().numpy()
        out = librosa.resample(np.asarray(out, dtype=np.float32), orig_sr=SR_MODELL, target_sr=SR)
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        if out.size < audio.size:
            out = np.pad(out, (0, audio.size - out.size), mode="edge")
        out = np.clip(out[: audio.size], -1.0, 1.0).astype(np.float32)
        self.statistik.calls += 1
        self.statistik.rms_summe += float(np.sqrt(np.mean(out**2)))
        self.statistik.peak = max(self.statistik.peak, float(np.max(np.abs(out))))
        self.statistik.clip_samples += int(np.sum(np.abs(out) >= 0.999))
        self.statistik.samples += int(out.size)
        self.letzte.append((np.asarray(audio, dtype=np.float32).copy(), out.copy()))
        return VocoderResult(
            audio=out,
            sr=SR,
            pqs_mos=BigVGANv2Plugin._estimate_pqs_mos(audio, out, SR),
            model_used="bigvgan_v2",
            confidence=0.95,
            mel_snr_db=BigVGANv2Plugin._mel_snr(audio, out, SR),
            metadata={"mel_modus": self.mel_modus, "checkpoint": self.checkpoint.name},
        )


def _fenster_grenzen(n: int, sr: int) -> tuple[int, int]:
    """Erstes HR-V1-Fenster inkl. 100-ms-Guard — dort wirkt das Modell."""
    return _fenster_fuer_span(n, sr, 0)


def _fenster_fuer_span(n: int, sr: int, index: int) -> tuple[int, int]:
    """HR-V1-Fenster ``index`` inkl. 100-ms-Guard."""
    from plugins.bigvgan_v2_plugin import hr_v1_budget_seconds, select_hr_v1_windows

    spans = select_hr_v1_windows(n, sr, hr_v1_budget_seconds(n / float(sr)))
    if not spans:
        return 0, min(n, int(0.25 * sr))
    s, e = spans[min(max(index, 0), len(spans) - 1)]
    pad = int(round(0.1 * sr))
    return max(0, s - pad), min(n, e + pad)


def _pegel_normiert(ausgang: np.ndarray, eingang: np.ndarray, *, max_peak: float = 0.95) -> np.ndarray:
    """Pegel-auf-Eingang (RMS) — trennt Modellgüte von reiner Ausgangs-Lautheit."""
    r_in = float(np.sqrt(np.mean(np.asarray(eingang, dtype=np.float64) ** 2)))
    r_out = float(np.sqrt(np.mean(np.asarray(ausgang, dtype=np.float64) ** 2)))
    out = np.asarray(ausgang, dtype=np.float32) * (r_in / max(r_out, 1e-9))
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > max_peak:
        out = out * (max_peak / peak)
    return np.asarray(out, dtype=np.float32)


def _messe_kandidat(
    eingang: np.ndarray,
    synth: Kandidat,
    *,
    sr: int = SR,
) -> dict[str, Any]:
    """Führt den echten HR-V1-Produktionsschritt mit dem Kandidaten aus.

    Gemessen werden zwei Ebenen:
      * ``roh_norm`` — die Modellgüte der Synthese (RMS-auf-Eingang), VOR dem
        additiven Gate. Das ist die Entscheidungsgrundlage: ein Kandidat, der
        schlechter synthetisiert, wird durch den Schutz höchstens unsichtbar,
        aber nie besser.
      * ``fenster_ausgang`` — der tatsächliche Produktionsmix (5-%-Deckel +
        ``additive_synthesis_gate``) als Sicherheits-Zeuge.
    """
    import plugins.bigvgan_v2_plugin as bvg
    from backend.core.dsp.wohlklang_zeuge import messe_wohlklang_zeuge

    synth.letzte.clear()
    synth.letzte_mel.clear()
    original = bvg.synthesize_audio
    bvg.synthesize_audio = synth  # type: ignore[assignment]
    try:
        ausgang, meta = bvg.apply_hr_v1_additive(eingang, sr)
    finally:
        bvg.synthesize_audio = original  # type: ignore[assignment]

    if not synth.letzte:
        raise RuntimeError(f"Kandidat {synth.name} hat nicht synthetisiert")
    # Nicht das erste, sondern das energiereichste Fenster messen: das erste
    # HR-V1-Fenster kann bei einem leisen Track-Anfang Stille sein.
    idx = int(np.argmax([float(np.mean(np.asarray(p[0], dtype=np.float64) ** 2)) for p in synth.letzte]))
    roh_in, roh_out = synth.letzte[idx]
    roh_out_norm = _pegel_normiert(roh_out, roh_in)
    roh_zeuge = messe_wohlklang_zeuge(roh_in, roh_out_norm, sr)
    mel_mittel, mel_std = synth.letzte_mel[idx]

    ws, we = _fenster_fuer_span(eingang.size, sr, idx)
    w_in = eingang[ws:we]
    w_out = np.asarray(ausgang[ws:we], dtype=np.float32)
    zeuge = messe_wohlklang_zeuge(w_in, w_out, sr)
    return {
        "applied": bool(meta.get("applied", False)),
        "attempted": bool(meta.get("attempted", False)),
        "bands_released": int(meta.get("bands_released", 0)),
        "processed_seconds": float(meta.get("processed_seconds", 0.0)),
        "pqs_mos": float(meta.get("pqs_mos", 0.0)),
        "fenster_s": round((we - ws) / float(sr), 3),
        "fenster": {
            "af": round(_af_proxy(w_in), 4),
            "hnr_db": round(_hnr_db(w_in), 4),
            "musik_mos": float(zeuge["eingang"]["musik_mos"]),
            "defektlast": float(zeuge["eingang"]["defektlast"]),
        },
        "fenster_ausgang": {
            "af": round(_af_proxy(w_out), 4),
            "hnr_db": round(_hnr_db(w_out), 4),
            "musik_mos": float(zeuge["ausgang"]["musik_mos"]),
            "defektlast": float(zeuge["ausgang"]["defektlast"]),
        },
        "roh_norm": {
            "af": round(_af_proxy(roh_out_norm), 4),
            "hnr_db": round(_hnr_db(roh_out_norm), 4),
            "musik_mos": float(roh_zeuge["ausgang"]["musik_mos"]),
            "defektlast": float(roh_zeuge["ausgang"]["defektlast"]),
            "rms": round(float(np.sqrt(np.mean(roh_out_norm**2))), 5),
            "peak": round(float(np.max(np.abs(roh_out_norm))), 5),
        },
        "roh": {
            "rms": round(float(np.sqrt(np.mean(roh_out**2))), 5),
            "peak": round(float(np.max(np.abs(roh_out))), 5),
            "clip_anteil": round(float(np.mean(np.abs(roh_out) >= 0.999)), 6),
            "mel_mittel": round(mel_mittel, 3),
            "mel_std": round(mel_std, 3),
        },
        "delta": {
            "af": round(_af_proxy(w_out) - _af_proxy(w_in), 4),
            "hnr_db": round(_hnr_db(w_out) - _hnr_db(w_in), 4),
            "musik_mos": float(zeuge["delta"]["musik_mos"]),
            "defektlast": float(zeuge["delta"]["defektlast"]),
        },
        "synth_statistik": synth.statistik.als_dict(),
    }


def _vergleich(kandidat: dict[str, Any], basis: dict[str, Any]) -> dict[str, Any]:
    """Never-worsen + Vorteil eines Kandidaten gegen die Basis (Fenster-Metriken)."""
    k = kandidat["roh_norm"]
    b = basis["roh_norm"]
    deltas = {
        "af": round(float(k["af"]) - float(b["af"]), 4),
        "hnr_db": round(float(k["hnr_db"]) - float(b["hnr_db"]), 4),
        "musik_mos": round(float(k["musik_mos"]) - float(b["musik_mos"]), 4),
        "defektlast": round(float(k["defektlast"]) - float(b["defektlast"]), 4),
    }
    never_worsen = {
        "af": bool(deltas["af"] >= -AF_TOLERANZ),
        "hnr_db": bool(deltas["hnr_db"] >= -HNR_TOLERANZ_DB),
        "musik_mos": bool(deltas["musik_mos"] >= -JND),
        "defektlast": bool(deltas["defektlast"] <= JND),
    }
    vorteil = bool(
        deltas["af"] >= AF_TOLERANZ
        or deltas["hnr_db"] >= HNR_TOLERANZ_DB
        or deltas["musik_mos"] >= JND
        or deltas["defektlast"] <= -JND
    )
    return {
        "delta_gegen_basis": deltas,
        "never_worsen": never_worsen,
        "vorteil": vorteil,
        "bestanden": bool(all(never_worsen.values()) and kandidat.get("applied", False)),
    }


def _gate_urteil(tracks: list[dict[str, Any]], variante: str) -> dict[str, Any]:
    """Aggregiert die Track-Vergleiche zu einem Gate-Urteil."""
    zeilen = [t["varianten"][variante] for t in tracks]
    alle_nie = all(z["bestanden"] for z in zeilen)
    vorteile = sum(1 for z in zeilen if z["vorteil"])
    angewendet = sum(1 for z in zeilen if z["applied"])
    bestanden = bool(alle_nie and vorteile >= BENEFIT_MIN_TRACKS and angewendet == len(zeilen))
    gruende: list[str] = []
    if angewendet < len(zeilen):
        gruende.append(f"nur {angewendet}/{len(zeilen)} Tracks angewandt")
    if not alle_nie:
        schlecht = [t["track"] for t, z in zip(tracks, zeilen) if not z["bestanden"]]
        gruende.append(f"Never-worsen verletzt: {', '.join(schlecht)}")
    if vorteile < BENEFIT_MIN_TRACKS:
        gruende.append(f"Vorteil nur {vorteile}/{len(zeilen)} (nötig ≥ {BENEFIT_MIN_TRACKS})")
    return {
        "bestanden": bestanden,
        "never_worsen_alle": bool(alle_nie),
        "vorteil_tracks": int(vorteile),
        "angewendet_tracks": int(angewendet),
        "grund": "bestanden" if bestanden else "; ".join(gruende),
    }


def _zielpfad(ziel: Path, inhalt: str) -> Path:
    """Nie einen vorhandenen Beleg überschreiben (Lehre D-K3-85)."""
    if not ziel.exists():
        return ziel
    if ziel.read_text(encoding="utf-8") == inhalt:
        return ziel
    for nummer in range(2, 100):
        kandidat = ziel.with_name(f"{ziel.stem}_{nummer}{ziel.suffix}")
        if not kandidat.exists():
            return kandidat
        if kandidat.read_text(encoding="utf-8") == inhalt:
            return kandidat
    raise RuntimeError(f"kein freier Beleg-Pfad neben {ziel}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="D3-2-Gate: BigVGAN-F3 gegen die Basis (CPU, §G5 (copilot-instructions.md))"
    )
    ap.add_argument("--basis-checkpoint", default=BASIS_CHECKPOINT, help="aktive Basis (Torch-Checkpoint)")
    ap.add_argument("--checkpoint", default=F3_CHECKPOINT, help="F3-Finetune (Torch-Checkpoint)")
    ap.add_argument("--tracks", type=int, default=3, help="MUSDB18HQ-Testtracks (≥ 3 für B3)")
    ap.add_argument("--seconds", type=int, default=10, help="Sekunden je Track")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out", default="", help="Report-Pfad (Default: docs/reports/current/<datum>_bigvgan_d3_gate.json)"
    )
    args = ap.parse_args(argv)

    grund = _cpu_guard()
    if grund:
        print(f"ABBRUCH: {grund}")
        return 2

    basis_pfad = (_ROOT / args.basis_checkpoint).resolve()
    f3_pfad = (_ROOT / args.checkpoint).resolve()
    for pfad in (basis_pfad, f3_pfad):
        if not pfad.is_file():
            print(f"ABBRUCH: fehlt: {pfad}")
            return 2

    tracks = _waehle_tracks(args.tracks, args.seed)
    if len(tracks) < 3:
        print(f"ABBRUCH: MUSDB18HQ-Testset unvollständig ({len(tracks)} Tracks): {MUSDB_TEST}")
        return 2

    import torch

    torch.set_num_threads(max(1, min(4, (torch.get_num_threads() or 4))))
    torch.set_grad_enabled(False)
    logger.info("D3-2-Gate — %d Track(s) × %d s, Seed %d, CPU", len(tracks), args.seconds, args.seed)

    basis = Kandidat("basis", basis_pfad, "prod")
    f3_prod = Kandidat("f3_prod", f3_pfad, "prod")
    f3_train = Kandidat("f3_trainmel", f3_pfad, "train")

    ergebnis: dict[str, Any] = {
        "schema": SCHEMA,
        "datum": date.today().isoformat(),
        "seed": args.seed,
        "sekunden": args.seconds,
        "toleranzen": {
            "af": AF_TOLERANZ,
            "hnr_db": HNR_TOLERANZ_DB,
            "jnd": JND,
            "vorteil_min_tracks": BENEFIT_MIN_TRACKS,
        },
        "checkpoint": {"pfad": str(f3_pfad.relative_to(_ROOT)), "sha256": _sha256(f3_pfad)},
        "basis_checkpoint": {"pfad": str(basis_pfad.relative_to(_ROOT)), "sha256": _sha256(basis_pfad)},
        "tracks": [],
    }

    for track in tracks:
        eingang = _lade_segment(track, args.seconds)
        zeile: dict[str, Any] = {"track": track.name, "varianten": {}}
        for kandidat in (basis, f3_prod, f3_train):
            messung = _messe_kandidat(eingang, kandidat)
            messung["mel_modus"] = kandidat.mel_modus
            zeile["varianten"][kandidat.name] = messung
        zeile["varianten"]["f3_prod"].update(_vergleich(zeile["varianten"]["f3_prod"], zeile["varianten"]["basis"]))
        zeile["varianten"]["f3_trainmel"].update(
            _vergleich(zeile["varianten"]["f3_trainmel"], zeile["varianten"]["basis"])
        )
        ergebnis["tracks"].append(zeile)
        print(
            f"  {track.name[:34]:34s} basis af={zeile['varianten']['basis']['roh_norm']['af']:.3f} "
            f"HNR={zeile['varianten']['basis']['roh_norm']['hnr_db']:+.2f} | "
            f"F3-prod af={zeile['varianten']['f3_prod']['roh_norm']['af']:.3f} "
            f"HNR={zeile['varianten']['f3_prod']['roh_norm']['hnr_db']:+.2f} | "
            f"F3-trainmel af={zeile['varianten']['f3_trainmel']['roh_norm']['af']:.3f} "
            f"HNR={zeile['varianten']['f3_trainmel']['roh_norm']['hnr_db']:+.2f} "
            f"(bänder {zeile['varianten']['f3_prod']['bands_released']}"
            f"/{zeile['varianten']['f3_trainmel']['bands_released']})"
        )

    ergebnis["gate"] = {
        "f3_prod": _gate_urteil(ergebnis["tracks"], "f3_prod"),
        "f3_trainmel": _gate_urteil(ergebnis["tracks"], "f3_trainmel"),
    }
    ergebnis["gate"]["gesamt"] = {
        "bestanden": bool(ergebnis["gate"]["f3_prod"]["bestanden"] or ergebnis["gate"]["f3_trainmel"]["bestanden"]),
        "deploybar_heute": bool(ergebnis["gate"]["f3_prod"]["bestanden"]),
        "kandidat_mit_mel_adapter": bool(ergebnis["gate"]["f3_trainmel"]["bestanden"]),
    }
    ergebnis["rohdaten"] = {
        "basis": basis.statistik.als_dict(),
        "f3_prod": f3_prod.statistik.als_dict(),
        "f3_trainmel": f3_train.statistik.als_dict(),
    }

    ziel = Path(args.out) if args.out else REPORT_DIR / f"{date.today().isoformat()}_bigvgan_d3_gate.json"
    if not ziel.is_absolute():
        ziel = _ROOT / ziel
    ziel.parent.mkdir(parents=True, exist_ok=True)
    inhalt = json.dumps(ergebnis, indent=2, ensure_ascii=False) + "\n"
    ziel = _zielpfad(ziel, inhalt)
    ziel.write_text(inhalt, encoding="utf-8")

    for variante, label in (
        ("f3_prod", "F3 + Produktions-Mel"),
        ("f3_trainmel", "F3 + Trainings-Mel "),
    ):
        g = ergebnis["gate"][variante]
        print(
            f"{label}: {'BESTANDEN' if g['bestanden'] else 'NICHT BESTANDEN'} "
            f"(Vorteil {g['vorteil_tracks']}/{len(ergebnis['tracks'])}, {g['grund']})"
        )
    print(f"Report: {ziel.relative_to(_ROOT)}")
    return 0 if ergebnis["gate"]["gesamt"]["bestanden"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
