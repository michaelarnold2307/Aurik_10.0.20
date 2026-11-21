"""Pfeiler 3 — Release-Kandidaten-Pipeline (SOTA-Roadmap 2026-10-27).

Fester Referenz-Korpus × Kandidaten-Flag-Varianten → Metrik-Report +
Hör-Pakete (2AFC-Never-worsen, `mushra_harness.py thresholds-build`).

Kandidaten in der Evidenz-Reihenfolge der Roadmap (Pfeiler 3):
  `use_scnet_music` → `use_sgmse_musik` → `use_bigvgan_f3_studio` →
  `use_bw_v5` → `use_wf4_warp_head` / `use_tp_v2_phase` /
  `use_rt60_regressor` / `use_aspade_declipper` / `use_tf_gridnet_quality`

Jede Variante läuft über den **Produktionspfad** (`UnifiedRestorerV3`) in
einem isolierten Kind-Prozess mit Timeout (Host-Schutz, Pfeiler 6 der
SOTA-Roadmap). Das Kandidaten-Flag wird im Kind **vor** dem Import des
Restorers gesetzt, damit auch modulweite Importe den Wert sehen.

Kein Flag-Flip durch dieses Skript (§v10.802 (copilot-instructions.md)):
der Report ist die Entscheidungsvorlage, das Hör-Paket die menschliche
Instanz (Hörordnung Ebene 4).

Nutzung:
  python scripts/release_candidate_pipeline.py --smoke
  python scripts/release_candidate_pipeline.py --candidates scnet,bw_v5 --seconds 10
  python scripts/release_candidate_pipeline.py --hearing

Ausgabe:
  docs/reports/current/<datum>_release_candidates.json  (Vergleichs-Report)
  output/release_candidates/<key>/  (WAVs + Hör-Manifest je Kandidat)
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)

SR_DEFAULT = 48_000

# Evidenz-Reihenfolge der Roadmap Pfeiler 3; jede Zeile trägt Flag und
# Register-ID (Zitierdisziplin AGENTS.md §5: IDs mit Quelle im Register).
CANDIDATES: tuple[dict[str, str], ...] = (
    {"key": "scnet", "flag": "use_scnet_music", "defizit": "D-K0-5", "stufe": "Separation (4Stems)"},
    {"key": "sgmse", "flag": "use_sgmse_musik", "defizit": "D-K0-3", "stufe": "Denoise/Dereverb"},
    {"key": "bigvgan_f3", "flag": "use_bigvgan_f3_studio", "defizit": "D-K0-1/D-K2-3", "stufe": "Studio-Finisher"},
    {"key": "bw_v5", "flag": "use_bw_v5", "defizit": "D-K1-5", "stufe": "HF-Rekonstruktion"},
    {"key": "wf4", "flag": "use_wf4_warp_head", "defizit": "D-K4-7", "stufe": "Warp-Schätzung"},
    {"key": "tp_v2", "flag": "use_tp_v2_phase", "defizit": "D-K4-8", "stufe": "Phasenschätzung"},
    {"key": "rt60", "flag": "use_rt60_regressor", "defizit": "D-K4-10", "stufe": "Dereverb-RT60"},
    {"key": "aspade", "flag": "use_aspade_declipper", "defizit": "D-K4-1", "stufe": "Declipper"},
    {"key": "tf_gridnet", "flag": "use_tf_gridnet_quality", "defizit": "D-K4-2", "stufe": "Vollband-Denoise"},
)

# Fester Referenz-Korpus (MUSTER tests/conftest.py::_real_audio_candidate_paths,
# materialbalanciert; existierende Dateien werden gefiltert).
CORPUS: tuple[dict[str, str], ...] = (
    {"path": "test_audio/tape/cassette_1980s_wow.wav", "material": "cassette"},
    {"path": "test_audio/tape/reel_1940s_dropout.wav", "material": "reel_tape"},
    {"path": "test_audio/vinyl/jazz_1950s_scratched.wav", "material": "vinyl"},
    {"path": "test_audio/vinyl/rock_1970s_worn.wav", "material": "vinyl"},
    {"path": "test_audio/digital/mp3_64kbps_artifacts.wav", "material": "mp3_low"},
    {"path": "test_audio/digital/cd_clipped_2000s.wav", "material": "cd"},
    {"path": "test_audio/vocals/opera_sibilance.wav", "material": "vocals"},
    {"path": "test_audio/vocals/choir_breaths.wav", "material": "vocals"},
)

HEARING_QUESTION = (
    "Welches Intervall klingt besser/authentischer — Referenz (unverarbeitet) oder Kandidat (Restaurierung)?"
)


def _existing_corpus() -> list[dict[str, str]]:
    return [c for c in CORPUS if (ROOT / c["path"]).is_file()]


def _load_clip(path: Path, *, start_s: float, seconds: float) -> tuple[np.ndarray, int]:
    """Ausschnitt laden (Produktions-Loader, Stereo (N, 2) float32)."""
    from backend.file_import import load_audio_file

    # Achtung API-Falle (Befund 2026-10-27): load_audio_file(Path…) liefert
    # sr=None — nur der str-Aufruf füllt die Samplerate. Deshalb str(path).
    data = load_audio_file(str(path))
    if isinstance(data, dict):  # Produktions-Loader liefert ein Dict (audio, sr, …)
        audio = np.asarray(data["audio"], dtype=np.float32)
        sr_val = data.get("sr") or data.get("sample_rate")
        sr = int(sr_val) if sr_val else int(SR_DEFAULT)
    else:
        audio = np.asarray(getattr(data, "audio", data), dtype=np.float32)
        sr = int(getattr(data, "sample_rate", getattr(data, "sr", SR_DEFAULT)))
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=1)
    elif audio.shape[0] <= 8 and audio.shape[-1] > 8:  # (C, N) → (N, C)
        audio = audio.T
    n0 = int(start_s * sr)
    n1 = n0 + int(seconds * sr)
    clip = np.ascontiguousarray(audio[max(0, n0) : n1])
    if clip.shape[0] < sr:  # < 1 s ist kein belastbares Fenster
        raise ValueError(f"Ausschnitt zu kurz: {path}")
    return clip, sr


def _restore_child(audio: np.ndarray, sr: int, flag_name: str, payload_path: str, error_path: str) -> None:
    """Produktionspfad im Kind-Prozess; Kandidaten-Flag VOR dem Restore-Import."""
    try:
        import os

        os.environ.setdefault("AURIK_SAFE_VALIDATION_PROFILE", "1")
        # Host-Schutz (Pfeiler 6) im Kind-Prozess: ML-Restores auf CPU dürfen
        # die Kun-Umgebung nicht ausbremsen (gemessen: ~96 s je Audio-Sekunde).
        _threads = max(1, min(4, (os.cpu_count() or 4) // 4))
        os.environ.setdefault("OMP_NUM_THREADS", str(_threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(_threads))
        try:
            os.nice(10)
        except OSError as exc:
            logger.debug("os.nice(10) im Kindprozess nicht anwendbar: %s", exc)
        import backend.core.music_model_flags as mmf

        setattr(mmf, flag_name, True)  # Kandidaten-Flag für den gesamten Import
        from backend.core.fallback_ledger import install_log_counter
        from backend.core.fallback_ledger import snapshot as _fb_snapshot

        install_log_counter()  # Pfeiler 5: §V6 (copilot-instructions.md)-Rückfälle zählen
        from backend.core.performance_guard import QualityMode
        from backend.core.unified_restorer_v3 import RestorationConfig, UnifiedRestorerV3

        cfg = RestorationConfig(
            mode=QualityMode.FAST,
            enable_performance_guard=True,
            enable_phase_gate=True,
            enable_phase_skipping=True,
        )
        restorer = UnifiedRestorerV3(config=cfg)
        result = restorer.restore(audio, sample_rate=sr, mode="fast", ml_runtime_budget_s=2.0)
        material = getattr(getattr(result, "material_type", "unknown"), "value", "unknown")
        np.savez(
            payload_path,
            audio=np.asarray(result.audio, dtype=np.float32),
            material_type=np.asarray([str(material)]),
            fallbacks=np.asarray([int(_fb_snapshot()["total"])]),
            fallbacks_defekt=np.asarray([int(_fb_snapshot()["by_art"].get("defekt", 0))]),
        )
    except Exception as exc:
        Path(error_path).write_text(f"{type(exc).__name__}: {exc}", encoding="utf-8")


def _run_restored(audio: np.ndarray, sr: int, flag_name: str, *, timeout_s: float) -> tuple[np.ndarray, str, int, int]:
    ctx = mp.get_context("spawn")
    tmp_dir = Path(tempfile.mkdtemp(prefix="aurik_rcp_"))
    payload_path = tmp_dir / "payload.npz"
    error_path = tmp_dir / "error.txt"
    proc = ctx.Process(
        target=_restore_child,
        args=(audio, sr, flag_name, str(payload_path), str(error_path)),
        daemon=True,
    )
    proc.start()
    proc.join(max(1.0, timeout_s))
    if proc.is_alive():
        proc.terminate()
        proc.join(10.0)
        raise RuntimeError(f"Restore-Timeout nach {timeout_s:.0f}s (flag={flag_name})")
    if error_path.exists():
        raise RuntimeError(error_path.read_text(encoding="utf-8"))
    if not payload_path.exists():
        raise RuntimeError(f"Kind ohne Ergebnis (exitcode={proc.exitcode}, flag={flag_name})")
    with np.load(payload_path, allow_pickle=True) as npz:
        out = np.asarray(npz["audio"], dtype=np.float32)
        # Stereo-Layout-Invariante (AGENTS.md §3): der Produktionspfad liefert
        # channels-first (C, N) — hier wird auf (N, C) normalisiert, sonst
        # schreibt sf.write ein N-Kanal-„WAV" und die Metriken rechnen falsch.
        if out.ndim == 2 and out.shape[0] <= 8 < out.shape[1]:
            out = out.T
        return (
            out,
            str(npz["material_type"][0]),
            int(np.asarray(npz.get("fallbacks", [0]))[0]),
            int(np.asarray(npz.get("fallbacks_defekt", [0]))[0]),
        )


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return 1.0
    return float(np.corrcoef(a, b)[0, 1])


def _noise_floor_dbfs(x: np.ndarray) -> float:
    mag = np.abs(x)
    if mag.size == 0:
        return -120.0
    floor = float(np.percentile(mag, 5.0))
    return float(20.0 * np.log10(floor + 1e-12))


def _metrics(original: np.ndarray, restored: np.ndarray) -> dict[str, Any]:
    """Kompakte Metrik-Schicht (Pfeiler 5: Fallback-Zählung erfolgt separat)."""
    n = min(original.shape[0], restored.shape[0])
    a, b = original[:n], restored[:n]
    rms_a = float(np.sqrt(np.mean(a**2)) + 1e-12)
    rms_b = float(np.sqrt(np.mean(b**2)) + 1e-12)
    return {
        "finite": bool(np.isfinite(b).all()),
        "peak_out": float(np.max(np.abs(b))) if b.size else 0.0,
        "rms_delta_db": float(20.0 * np.log10(rms_b / rms_a)),
        "noise_floor_delta_db": _noise_floor_dbfs(b) - _noise_floor_dbfs(a),
        "stereo_corr_before": _corr(a[:, 0], a[:, 1]) if a.ndim == 2 else 1.0,
        "stereo_corr_after": _corr(b[:, 0], b[:, 1]) if b.ndim == 2 else 1.0,
        "clip_ratio": float(np.mean(np.abs(b) >= 0.999)) if b.size else 0.0,
    }


def _write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    Path(path).unlink(missing_ok=True)  # 0-Byte-Reste dürfen das Write nicht blockieren
    sf.write(str(path), audio, sr)


def _manifest(trials: list[dict[str, Any]], seed: int, question: str = HEARING_QUESTION) -> dict[str, Any]:
    """Hör-Manifest im Schema von ``mushra_harness.py thresholds-build``."""
    return {
        "task": "preference",
        "question": question,
        "seed": seed,
        "trials": trials,
    }


def run_candidate(
    cand: dict[str, str],
    clips: list[dict[str, str]],
    out_dir: Path,
    *,
    seconds: float,
    start_s: float,
    seed: int,
    timeout_s: float,
) -> dict[str, Any]:
    key_dir = out_dir / cand["key"]
    trials: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for clip_info in clips:
        clip_path = ROOT / clip_info["path"]
        try:
            window, sr = _load_clip(clip_path, start_s=start_s, seconds=seconds)
            restored, material, fallbacks, fb_defekt = _run_restored(window, sr, cand["flag"], timeout_s=timeout_s)
        except Exception as exc:
            rows.append({"clip": clip_info["path"], "status": "fehler", "error": str(exc)[:300]})
            continue
        stem = clip_path.stem[:32]
        ref_wav = key_dir / f"{stem}__referenz.wav"
        cand_wav = key_dir / f"{stem}__kandidat.wav"
        _write_wav(ref_wav, window, sr)
        _write_wav(cand_wav, restored, sr)
        m = _metrics(window, restored)
        rows.append(
            {
                "clip": clip_info["path"],
                "material": material,
                "status": "ok",
                "fallbacks": fallbacks,
                "fallbacks_defekt": fb_defekt,
                **m,
            }
        )
        trials.append(
            {
                "trial_id": f"{cand['key']}_{stem}",
                "class": clip_info["material"],
                "catch": False,
                "reference": str(ref_wav),
                "candidate": str(cand_wav),
                "start": 0.0,
                "seconds": round(window.shape[0] / sr, 3),
                "expected_in_candidate": False,
            }
        )
    manifest_path = key_dir / "hoer_manifest.json"
    key_dir.mkdir(parents=True, exist_ok=True)
    manifest = _manifest(trials, seed=seed)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    summary = {
        "n_ok": len(ok_rows),
        "n_fehler": len(rows) - len(ok_rows),
        "mean_rms_delta_db": round(float(np.mean([r["rms_delta_db"] for r in ok_rows])), 3) if ok_rows else None,
        "max_clip_ratio": round(float(max((r["clip_ratio"] for r in ok_rows), default=0.0)), 5),
        "min_stereo_corr_after": round(float(min((r["stereo_corr_after"] for r in ok_rows), default=1.0)), 3),
        "fallback_count": int(sum(int(r.get("fallbacks", 0)) for r in ok_rows)),
        "fallbacks_defekt": int(sum(int(r.get("fallbacks_defekt", 0)) for r in ok_rows)),
    }
    summary["flag_freigabe"] = (
        "blockiert (ungeplanter Rueckfall > 0, B1)"
        if summary["fallbacks_defekt"] > 0
        else "offen (Hör-Gate ausstehend)"
    )
    return {
        "key": cand["key"],
        "flag": cand["flag"],
        "defizit": cand["defizit"],
        "stufe": cand["stufe"],
        "status": "gemessen",
        "summary": summary,
        "cases": rows,
        "hoer_manifest": str(manifest_path),
        "n_trials": len(trials),
    }


def main() -> int:
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates", type=str, default="", help="kommagetrennte Kandidaten-Keys (Default: alle)")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=8, help="max. Korpus-Dateien")
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--timeout-s", type=float, default=600.0)
    parser.add_argument("--out", type=Path, default=Path("output/release_candidates"))
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--hearing", action="store_true", help="Hör-Pakete via mushra_harness thresholds-build bauen")
    parser.add_argument("--smoke", action="store_true", help="2 Kandidaten, 1 Fenster, 6 s (Schnellverifikation)")
    args = parser.parse_args()

    if args.smoke:
        args.limit = 1
        args.seconds = 6.0
        wanted = [CANDIDATES[0]["key"], CANDIDATES[3]["key"]]
    else:
        wanted = [k.strip() for k in args.candidates.split(",") if k.strip()]
    selected = [c for c in CANDIDATES if not wanted or c["key"] in wanted]
    if not selected:
        print("FEHLER: keine Kandidaten selektiert", file=sys.stderr)
        return 2

    clips = _existing_corpus()[: max(1, args.limit)]
    if not clips:
        print("FEHLER: keine Korpus-Dateien gefunden (test_audio/…)", file=sys.stderr)
        return 2
    print(f"Kandidaten: {[c['key'] for c in selected]}")
    print(f"Korpus: {[c['path'] for c in clips]}")

    results = []
    for cand in selected:
        res = run_candidate(
            cand,
            clips,
            args.out,
            seconds=args.seconds,
            start_s=args.start,
            seed=args.seed,
            timeout_s=args.timeout_s,
        )
        results.append(res)
        s = res["summary"]
        print(
            f"[{res['key']:>10}] ok={s['n_ok']} fehler={s['n_fehler']} "
            f"rmsΔ={s['mean_rms_delta_db']} clip={s['max_clip_ratio']} "
            f"trials={res['n_trials']}"
        )
        if args.hearing and res["n_trials"] > 0:
            import subprocess

            build = f"rcp_{res['key']}"
            cmd = [
                sys.executable,
                str(ROOT / "scripts" / "mushra_harness.py"),
                "thresholds-build",
                "--build",
                build,
                "--manifest",
                res["hoer_manifest"],
                "--out",
                str(args.out / res["key"] / f"thresholds_{build}"),
                "--seed",
                str(args.seed),
            ]
            subprocess.run(cmd, check=False)
            res["hoer_study"] = str(args.out / res["key"] / f"thresholds_{build}")

    report_path = args.report or (
        ROOT
        / "docs"
        / "reports"
        / "current"
        / f"{__import__('datetime').date.today().isoformat()}_release_candidates.json"
    )
    report = {
        "schema": "release_candidates_v1",
        "seed": args.seed,
        "smoke": bool(args.smoke),
        "seconds": args.seconds,
        "corpus": [c["path"] for c in clips],
        "kandidaten": results,
        "hinweis": (
            "Kein Flag-Flip durch dieses Skript (§v10.802 (copilot-instructions.md)); "
            "Freigabe erst nach Hör-Verdikt + Wohlklang-Vertrag (Belege 1–6, "
            ".github/WOHLKLANG_CLAIMS.md)."
        ),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
