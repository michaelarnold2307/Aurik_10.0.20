#!/usr/bin/env python3
"""eval_scnet_vs_mdx23c.py — TODO-P1-2: SCNet-4-Stems-Kandidat vs. Produktionsketten-Stand.

Bewertet den Kandidaten-Checkpoint ``models/scnet_4stems/huge_scnet_4stems_v1.2.ckpt``
(Aname-Tommy/Huge-SCNet-4stems, Apache-2.0) gegen die Demucs-v4-Stufe der
Produktionskette (``models/demucs/htdemucs_6s.onnx``, CPU) auf drei
MUSDB18-HQ-Test-Songs.

**Auftragstreue/Abweichungen (dokumentiert):** Der Auftrag M2 nennt als Baseline
„MDX23C-Stand". MDX23C-Gewichte wurden nach §v10.73 entfernt (keine ONNX-Datei
im Repo; der v5-Arm existiert nie — htdemucs_6s = v4). Baseline ist daher die
Demucs-v4-Stufe der Produktionskette (ONNX-CPU; im Eval verifiziert: Vocals
korrelieren, SI-SDR positiv). **Zusatzbefund 2026-10-05 (defekt, nicht als
Baseline verwendet):** der MelBandRoformer-ONNX-Pfad des
BS-RoFormer-Plugins liefert auf identischem Material duplizierte
drums/bass/other-Stems und einen entkoppelten Vocals-Stem (Korrelation ≈ 0) —
Belege: `output/scnet_ab_smoke_20261005/`, `output/scnet_ab_smoke2_20261005/`.
Kein Produktions-Flag wird angefasst; das Ergebnis ist eine Empfehlung — die
Integrationsentscheidung bleibt menschlich (C4-Sign-off).

Verträge:
- §V6 (VERBOTEN.md): keine stillen Fallbacks — Fehler werden je Song als
  ``status="error"`` protokolliert; Exit 1 wenn Fehlerfälle existieren, 2 bei
  Setup-Fehlern.
- §G5 (GEBOTE.md): deterministisch — Master-Seed 42, keine Zeit-/Zufallslogik
  in Entscheidungen; Resampling deterministisch (torchaudio sinc).
- §III.9/CPU-Vertrag: Läuft CPU-only (``AURIK_FORCE_CPU=1`` wird VOR den
  Torch/ONNX-Importen gesetzt, damit das parallel laufende F7-Training nicht
  gestört wird). Beide Systeme rechnen über ONNX-CPU bzw. Torch-CPU.

Metriken (je Song × System):
- ``si_sdr_db``: Scale-invariant SDR (Le Roux et al. 2019) der geschätzten
  Vocals gegen das Ground-Truth-Vocals-Stem (mono, 44,1 kHz).
- ``singer_identity_cosine``: Resemblyzer-d-vector-Cosinus (Plugin, §2.35c)
  zwischen geschätzten und Referenz-Vocals.
- ``separation_fidelity``: kanonische Formel aus
  ``musical_goals_metrics``: 1 − RMS(Mixtur − Stem-Summe) / RMS(Stem-Summe),
  geclippt [0, 1] (Rekonstruktions-Treue der Stem-Zerlegung).

Ausgaben:
- ``output/scnet_ab_2026-10-05/matrix.csv``  (song,system,si_sdr_db,singer_identity_cosine,separation_fidelity,status)
- ``output/scnet_ab_2026-10-05/report.json`` (Rohwerte, Seed, Modell-Pfade, SHA-256, Laufzeiten)
- Hör-Artefakte je Song: ``<song>__scnet_vocals.wav``, ``<song>__melband_vocals.wav``,
  ``<song>__gt_vocals.wav``, ``<song>__mix.wav`` (32-bit-Float, §V5-sicher)

Usage:
    AURIK_FORCE_CPU=1 .venv_aurik/bin/python scripts/eval_scnet_vs_mdx23c.py \
        [--seconds 30] [--songs 3] [--out output/scnet_ab_2026-10-05]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

# §CPU-Vertrag VOR allen Torch/ORT-Importen: F7-Training belegt die GPU.
os.environ.setdefault("AURIK_FORCE_CPU", "1")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCNET_VENDOR_PARENT = ROOT / "models" / "scnet_4stems"
if str(SCNET_VENDOR_PARENT) not in sys.path:
    sys.path.insert(0, str(SCNET_VENDOR_PARENT))

SEED = 42
SAMPLE_RATE = 44100
SCNET_CKPT = ROOT / "models" / "scnet_4stems" / "huge_scnet_4stems_v1.2.ckpt"
SCNET_CONFIG = ROOT / "models" / "scnet_4stems" / "config.yaml"
MUSDB_TEST = ROOT / "data" / "musdb18hq" / "test"
DEFAULT_SONGS = (
    "AM Contra - Heart Peripheral",
    "Al James - Schoolboy Facination",
    "Motor Tapes - Shore",
)

logger = logging.getLogger("eval_scnet_ab")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _stub_bitsandbytes() -> None:
    """Checkpoint-Pickle referenziert bitsandbytes (Training) — Dummy-Module
    genügen für reines Inferenz-Laden (Muster: scripts/_scnet_arch_probe.py)."""
    import types

    class _Mod(types.ModuleType):
        def __getattr__(self, n: str):
            if n.startswith("__"):
                raise AttributeError(n)
            val = type(n, (), {"__init__": lambda self, *a, **k: None})
            setattr(self, n, val)
            return val

    for name in (
        "bitsandbytes",
        "bitsandbytes.optim",
        "bitsandbytes.optim.adamw",
        "bitsandbytes.nn",
        "bitsandbytes.functional",
        "bitsandbytes.cextension",
        "bitsandbytes.triton",
    ):
        sys.modules[name] = _Mod(name)


def _si_sdr_db(estimate: np.ndarray, reference: np.ndarray) -> float:
    """Scale-invariant SDR (Le Roux et al. 2019), mono float64."""
    est = np.asarray(estimate, dtype=np.float64).ravel()
    ref = np.asarray(reference, dtype=np.float64).ravel()
    n = min(est.size, ref.size)
    est, ref = est[:n], ref[:n]
    est = est - est.mean()
    ref = ref - ref.mean()
    denom = float(np.dot(ref, ref)) + 1e-12
    alpha = float(np.dot(est, ref)) / denom
    target = alpha * ref
    err = est - target
    num = float(np.dot(target, target)) + 1e-12
    den = float(np.dot(err, err)) + 1e-12
    return float(10.0 * np.log10(num / den))


def _stem_matrix(stems: dict[str, np.ndarray], gt_monos: dict[str, np.ndarray]) -> dict[str, float | None]:
    """SI-SDR je Stem gegen den zugehörigen GT-Stem (nur gemeinsame Namen).

    Stiller GT-Stem (RMS < 1e-4) ⇒ None (§V6-ehrlich statt Rauschzahl).
    """
    out: dict[str, float | None] = {}
    for name, arr in stems.items():
        if name not in gt_monos:
            continue
        _g = gt_monos[name]
        _r = float(np.sqrt(np.mean(_g.astype(np.float64) ** 2)))
        out[name] = None if _r < 1e-4 else round(_si_sdr_db(_to_mono(arr), _g), 1)
    return out


def _separation_fidelity(mixture: np.ndarray, stems_sum: np.ndarray) -> float:
    """Kanonische Formel (musical_goals_metrics): 1 − RMS(Diff)/RMS(Summe)."""
    mix = np.asarray(mixture, dtype=np.float64)
    ssum = np.asarray(stems_sum, dtype=np.float64)
    n = min(mix.size, ssum.size)
    diff = mix.ravel()[:n] - ssum.ravel()[:n]
    rms_diff = float(np.sqrt(np.mean(diff**2)) + 1e-12)
    rms_sum = float(np.sqrt(np.mean(ssum.ravel()[:n] ** 2)) + 1e-12)
    return float(np.clip(1.0 - rms_diff / rms_sum, 0.0, 1.0))


def _to_mono(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 2:
        return arr.mean(axis=1).astype(np.float32)
    return arr.astype(np.float32)


def _resample_np(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    """Deterministisches sinc-Resampling via torchaudio (Float32, layout-tolerant)."""
    if sr_from == sr_to:
        return np.asarray(x, dtype=np.float32)
    import torch
    import torchaudio

    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        t = torch.from_numpy(arr)
        out = torchaudio.functional.resample(t, sr_from, sr_to)
        return out.numpy().astype(np.float32)
    # (N, C) → (C, N) fürs Resampling
    t = torch.from_numpy(arr.T.copy())
    out = torchaudio.functional.resample(t, sr_from, sr_to)
    return out.numpy().T.astype(np.float32)


def _choose_offset(voc: np.ndarray, sr: int, seconds: float) -> int:
    """Deterministischer Fenster-Start: Fenster mit maximaler Vocals-Energie.

    Ohne diesen Bezug misst der A/B auf Song-Intros mit (nahezu) stillem
    Gesangs-Stem — SI-SDR/Singer-Identity werden dann bedeutungslos (Befund
    2026-10-05: GT-Vocals-RMS 1,8e-5 im Intro von „AM Contra“). 1-s-Raster,
    argmax ist eindeutig (kein Zufall, §G5 (copilot-instructions.md)).
    """
    win = int(seconds * sr)
    mono = _to_mono(voc).astype(np.float64)
    if mono.size <= win:
        return 0
    step = max(1, sr)  # 1-s-Raster
    best_off, best_e = 0, -1.0
    for off in range(0, mono.size - win + 1, step):
        e = float(np.dot(mono[off : off + win], mono[off : off + win]))
        if e > best_e:
            best_off, best_e = off, e
    return int(best_off)


def _load_song(song: str, seconds: float, offset: float = -1.0) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Lädt Mixtur + Vocals-Stem (44,1 kHz) als ``seconds``-Segment.

    ``offset < 0`` ⇒ automatisch das vocals-energiereichste Fenster; sonst
    fester Start in Sekunden. Rückgabe: (mix, vocals, sr, offset_s).
    """
    import soundfile as sf

    base = MUSDB_TEST / song
    mix, sr_m = sf.read(str(base / "mixture.wav"), dtype="float32", always_2d=True)
    voc, sr_v = sf.read(str(base / "vocals.wav"), dtype="float32", always_2d=True)
    if sr_m != sr_v:
        raise ValueError(f"Sample-Rate-Mismatch Mixtur/Stem: {sr_m} vs {sr_v}")
    off = _choose_offset(voc, int(sr_m), seconds) if offset < 0 else int(offset * sr_m)
    off = max(0, min(off, min(mix.shape[0], voc.shape[0])))
    n_end = min(mix.shape[0], voc.shape[0], off + int(seconds * sr_m))
    if n_end - off < int(0.5 * sr_m):
        raise ValueError(f"Segment zu kurz nach Offset ({song}): {n_end - off} Samples")
    return (
        mix[off:n_end].astype(np.float32),
        voc[off:n_end].astype(np.float32),
        int(sr_m),
        off / float(sr_m),
    )


def _load_scnet() -> tuple[object, list[str]]:
    """Baut den SCNet-Kandidaten (vendored ZFTurbo-Code, MIT) und lädt den
    Checkpoint strikt. Gibt (Modell, source-reihenfolge) zurück."""
    _stub_bitsandbytes()
    import torch
    import yaml
    from zfturbo_scnet import SCNet  # type: ignore[import-not-found]

    with open(SCNET_CONFIG, encoding="utf-8") as fh:
        cfg_model = yaml.safe_load(fh)["model"]

    torch.manual_seed(SEED)
    model = SCNet(**cfg_model)
    ck = torch.load(str(SCNET_CKPT), map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"]
    sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"SCNet-Checkpoint passt nicht strikt zur Architektur: "
            f"missing={len(missing)} unexpected={len(unexpected)} "
            f"(Beispiele: {list(missing)[:3]} | {list(unexpected)[:3]})"
        )
    model.eval()
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
    logger.info(
        "SCNet geladen: %s Parameter | SHA256=%s",
        f"{sum(p.numel() for p in model.parameters()):,}",
        _sha256(SCNET_CKPT)[:16],
    )
    return model, list(cfg_model["sources"])


def _scnet_separate(model: object, sources: list[str], mix: np.ndarray) -> dict[str, np.ndarray]:
    """SCNet-Inferenz (whole-segment, CPU) — Ausgabe (B, S, C, L) → Stem-Dict (N, C)."""
    import torch

    t = torch.from_numpy(np.ascontiguousarray(mix.T)).unsqueeze(0)  # (1, C, N)
    with torch.no_grad():
        out = model(t)  # (1, S, C, N)
    out = out[0].cpu().numpy()  # (S, C, N)
    stems: dict[str, np.ndarray] = {}
    for idx, name in enumerate(sources):
        stems[name] = out[idx].T.astype(np.float32)  # (N, C)
    return stems


def _demucs4_baseline(mix44: np.ndarray, sr: int = SAMPLE_RATE) -> tuple[dict[str, np.ndarray], str]:
    """Produktions-Stand: Demucs v4 (htdemucs_6s, ONNX-CPU).

    Nutzt den **kanonischen** Aufrufvertrag aus ``plugins.htdemucs_plugin``
    (STFT-Eingang ``x``, Hybrid-Summe beider Zweige, Stem-Reihenfolge,
    Modellrate 44,1 kHz) — §G9 (copilot-instructions.md): **eine**
    Implementierung für Plugin und Eval.

    Befund 2026-10-06: Der frühere eigene Aufruf fütterte ``x`` mit Nullen und
    nutzte nur den Wellenform-Zweig (``add_67``); die Baseline verlor dadurch
    rund **13 dB** (Vocals-SI-SDR −1,80 dB statt +11,59 dB, „Motor Tapes“).
    Der Report ``2026-10-04_p1_2_scnet_vs_mdx23c_ab.md`` ist damit überholt,
    sofern er gegen diese handicapierte Baseline argumentiert.
    """
    import onnxruntime as ort

    from plugins.htdemucs_plugin import (
        _HTDEMUCS_SEGMENT,
        _HTDEMUCS_STEM_ORDER,
        htdemucs_onnx_stems,
    )

    onnx_path = ROOT / "models" / "demucs" / "htdemucs_6s.onnx"
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    win = _HTDEMUCS_SEGMENT
    n_total = mix44.shape[0]
    acc: dict[str, np.ndarray] = {name: np.zeros_like(mix44) for name in _HTDEMUCS_STEM_ORDER}
    for start in range(0, n_total, win):
        chunk = mix44[start : start + win]
        pad = np.zeros((win, mix44.shape[1]), dtype=np.float32)
        pad[: chunk.shape[0]] = chunk
        stems_6 = htdemucs_onnx_stems(sess, pad.T)  # (6, C, win)
        seg_len = chunk.shape[0]
        for pos, name in enumerate(_HTDEMUCS_STEM_ORDER):
            acc[name][start : start + seg_len] = stems_6[pos].T[:seg_len].astype(np.float32)
    return acc, "demucs_v4_htdemucs_6s_onnx"


def main() -> int:
    parser = argparse.ArgumentParser(description="TODO-P1-2 A/B: SCNet-Kandidat vs. Produktions-Stand")
    parser.add_argument("--seconds", type=float, default=30.0, help="Segmentlänge je Song in s")
    parser.add_argument(
        "--offset",
        type=float,
        default=-1.0,
        help="Segment-Start in s; -1 = automatisch (vocals-energiereichstes Fenster, §G5-deterministisch)",
    )
    parser.add_argument("--songs", type=int, default=3, help="Anzahl Songs (Index aus Fixliste)")
    parser.add_argument("--out", type=Path, default=ROOT / "output" / "scnet_ab_2026-10-05")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s — %(message)s")
    import torch

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(out_dir / "eval_scnet_ab.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s — %(message)s"))
    logging.getLogger().addHandler(fh)

    songs = list(DEFAULT_SONGS)[: max(1, int(args.songs))]

    # Setup (fail-fast, Exit 2)
    try:
        scnet_model, scnet_sources = _load_scnet()
    except Exception as exc:  # §V6 (copilot-instructions.md): Setup-Fehler laut
        logger.error("Setup-Fehler SCNet: %s", exc)
        return 2

    from plugins.resemblyzer_plugin import get_resemblyzer_plugin

    resemblyzer = get_resemblyzer_plugin()

    rows: list[dict[str, object]] = []
    per_song: list[dict[str, object]] = []
    error_cases = 0

    for song in songs:
        logger.info("=== Song: %s ===", song)
        try:
            t0 = time.perf_counter()
            mix, gt_voc, sr, offset_s = _load_song(song, args.seconds, offset=args.offset)
            gt_mono = _to_mono(gt_voc)
            gt_rms = float(np.sqrt(np.mean(gt_mono.astype(np.float64) ** 2)))
            logger.info("Segment: offset=%.3fs (%.2fs) | GT-Vocals-RMS=%.2e", offset_s, args.seconds, gt_rms)

            import soundfile as sf

            _off_n = int(round(offset_s * sr))
            _end_n = _off_n + int(args.seconds * sr)
            gt_stems_mono: dict[str, np.ndarray] = {}
            for _name in ("drums", "bass", "other", "vocals"):
                _g, _gsr = sf.read(str(MUSDB_TEST / song / f"{_name}.wav"), dtype="float32", always_2d=True)
                if int(_gsr) != int(sr):
                    raise ValueError(f"GT-Stem-SR-Mismatch {_name}: {_gsr} vs {sr}")
                gt_stems_mono[_name] = _to_mono(_g[_off_n:_end_n])

            # --- SCNet-Kandidat ---
            t_scnet = time.perf_counter()
            scnet_stems = _scnet_separate(scnet_model, scnet_sources, mix)
            scnet_secs = time.perf_counter() - t_scnet
            scnet_sum = np.sum(np.stack(list(scnet_stems.values()), axis=0), axis=0)
            scnet_voc_mono = _to_mono(scnet_stems.get("vocals", np.zeros_like(mix)))

            # --- Baseline (Produktions-Stand) ---
            t_base = time.perf_counter()
            base_stems, base_model = _demucs4_baseline(mix, sr)
            base_secs = time.perf_counter() - t_base
            base_sum = np.sum(np.stack(list(base_stems.values()), axis=0), axis=0)
            base_voc_mono = _to_mono(base_stems.get("vocals", np.zeros_like(mix)))

            # --- Metriken (Mono-Bezug für BEIDE Systeme; GT-Stillheits-Guard) ---
            mix_mono = _to_mono(mix)
            silent_gt = gt_rms < 1e-4  # GT-Vocals praktisch still ⇒ SI-SDR/Singer bedeutungslos
            if silent_gt:
                logger.warning("GT-Vocals quasi still (RMS=%.2e) — SI-SDR/Singer = null (§V6-ehrlich)", gt_rms)
            scnet_metrics = {
                "si_sdr_db": None if silent_gt else round(_si_sdr_db(scnet_voc_mono, gt_mono), 3),
                "separation_fidelity": round(_separation_fidelity(mix_mono, _to_mono(scnet_sum)), 4),
                "vocals_rms": round(float(np.sqrt(np.mean(np.asarray(scnet_voc_mono, dtype=np.float64) ** 2))), 8),
            }
            base_metrics = {
                "si_sdr_db": None if silent_gt else round(_si_sdr_db(base_voc_mono, gt_mono), 3),
                "separation_fidelity": round(_separation_fidelity(mix_mono, _to_mono(base_sum)), 4),
                "vocals_rms": round(float(np.sqrt(np.mean(np.asarray(base_voc_mono, dtype=np.float64) ** 2))), 8),
            }
            if not silent_gt:
                try:
                    emb_scnet = resemblyzer.embed(scnet_voc_mono, sr)
                    emb_gt = resemblyzer.embed(gt_mono, sr)
                    emb_base = resemblyzer.embed(base_voc_mono, sr)
                    scnet_metrics["singer_identity_cosine"] = (
                        round(float(resemblyzer.cosine_similarity(emb_scnet, emb_gt)), 4)
                        if emb_scnet is not None and emb_gt is not None
                        else None
                    )
                    base_metrics["singer_identity_cosine"] = (
                        round(float(resemblyzer.cosine_similarity(emb_base, emb_gt)), 4)
                        if emb_base is not None and emb_gt is not None
                        else None
                    )
                except Exception as exc:  # §V6 (copilot-instructions.md): Messfehler explizit
                    logger.warning("Resemblyzer-Witness fehlgeschlagen (%s) — Metrik bleibt None", exc)
                    scnet_metrics.setdefault("singer_identity_cosine", None)
                    base_metrics.setdefault("singer_identity_cosine", None)
            else:
                scnet_metrics["singer_identity_cosine"] = None
                base_metrics["singer_identity_cosine"] = None

            scnet_matrix = _stem_matrix(scnet_stems, gt_stems_mono)
            base_matrix = _stem_matrix(base_stems, gt_stems_mono)

            # Hör-Artefakte (für C4-Hörstichprobe)
            import soundfile as sf

            short = song.replace(" ", "_").replace("/", "_")
            sf.write(str(out_dir / f"{short}__mix.wav"), mix, sr, subtype="FLOAT")
            sf.write(str(out_dir / f"{short}__gt_vocals.wav"), gt_voc, sr, subtype="FLOAT")
            sf.write(
                str(out_dir / f"{short}__scnet_vocals.wav"),
                scnet_stems.get("vocals", np.zeros_like(mix)),
                sr,
                subtype="FLOAT",
            )
            sf.write(
                str(out_dir / f"{short}__baseline_vocals.wav"),
                base_stems.get("vocals", np.zeros_like(mix)),
                sr,
                subtype="FLOAT",
            )

            for system, metrics, secs, used in (
                ("scnet_candidate", scnet_metrics, scnet_secs, "scnet_4stems_v1.2"),
                ("production_stand", base_metrics, base_secs, base_model),
            ):
                rows.append(
                    {
                        "song": song,
                        "system": system,
                        "model_used": used,
                        "seconds": round(args.seconds, 2),
                        "offset_s": round(offset_s, 3),
                        "gt_vocals_rms": round(gt_rms, 8),
                        "si_sdr_db": metrics.get("si_sdr_db"),
                        "singer_identity_cosine": metrics.get("singer_identity_cosine"),
                        "separation_fidelity": metrics.get("separation_fidelity"),
                        "vocals_rms": metrics.get("vocals_rms"),
                        "wall_s": round(secs, 2),
                        "status": "ok",
                    }
                )
            per_song.append(
                {
                    "song": song,
                    "status": "ok",
                    "offset_s": round(offset_s, 3),
                    "gt_vocals_rms": round(gt_rms, 8),
                    "scnet": scnet_metrics,
                    "production_stand": base_metrics,
                    "stem_si_sdr_db": {"scnet": scnet_matrix, "production_stand": base_matrix},
                    "wall_s": round(time.perf_counter() - t0, 2),
                }
            )
            logger.info("Song ok: %s | scnet=%s | stand=%s", song, scnet_metrics, base_metrics)
        except Exception as exc:
            error_cases += 1
            logger.error(
                "FALL-FEHLER (%s): %s: %s — als error protokolliert (§V6 (copilot-instructions.md))",
                song,
                type(exc).__name__,
                exc,
            )
            rows.append(
                {
                    "song": song,
                    "system": "all",
                    "model_used": "",
                    "seconds": round(args.seconds, 2),
                    "si_sdr_db": None,
                    "singer_identity_cosine": None,
                    "separation_fidelity": None,
                    "wall_s": None,
                    "status": "error",
                }
            )
            per_song.append({"song": song, "status": "error", "error": f"{type(exc).__name__}: {exc}"})

    matrix = out_dir / "matrix.csv"
    with matrix.open("w", newline="", encoding="utf-8") as fhw:
        writer = csv.DictWriter(
            fhw,
            fieldnames=[
                "song",
                "system",
                "model_used",
                "seconds",
                "offset_s",
                "gt_vocals_rms",
                "si_sdr_db",
                "singer_identity_cosine",
                "separation_fidelity",
                "vocals_rms",
                "wall_s",
                "status",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    report = {
        "task": "TODO-P1-2 A/B — SCNet-Kandidat vs. Produktions-Stand",
        "generated_utc_note": "Kun-Ausführungssitzung 2026-10-05",
        "baseline_note": (
            "Auftrag nennt 'MDX23C-Stand'; MDX23C-Gewichte nach §v10.73 entfernt, "
            "Demucs-v5-Arm existiert nicht. Baseline = Demucs-v4-Ketten-Stufe "
            "(htdemucs_6s ONNX, CPU), im Eval verifiziert. Zusatzbefund: der "
            "MelBandRoformer-ONNX-Pfad des BS-RoFormer-Plugins liefert duplizierte "
            "Stems + entkoppelte Vocals (nicht als Baseline verwendet; Defekt-Report)."
        ),
        "cpu_contract": "AURIK_FORCE_CPU=1 (F7-Training belegt GPU); beide Systeme ONNX-CPU/Torch-CPU",
        "seed": SEED,
        "seconds": args.seconds,
        "songs": songs,
        "scnet_ckpt_sha256_prefix": _sha256(SCNET_CKPT)[:16],
        "rows": rows,
        "per_song": per_song,
        "error_cases": error_cases,
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("Matrix: %s", matrix)
    logger.info("Report: %s", out_dir / "report.json")
    if error_cases:
        logger.error("Fehlerfälle: %d — Exit 1 (§V6 (copilot-instructions.md))", error_cases)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
