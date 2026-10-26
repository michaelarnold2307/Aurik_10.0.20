"""KP-3 · Vokal-Identitäts-Zeugen-A/B-Harness (DSP-Proxy vs. MuQ-MuLan).

Vergleicht zwei Identitäts-Witness-Verfahren nach **Kosinus-Invarianz**:

- heutige DSP-Proxymessung (`measure_singer_identity_cosine` —
  MFCC + spektraler Centroid-Korrelation, domänenneutral);
- MuQ-MuLan-Zeuge (`music_identity_cosine` aus
  `backend/core/dsp/music_identity_oracle.py`, 1024-d, musiktrainiert).

Meta-Hypothese (KP-3, `docs/SOTA_TRAINING_KONZEPT.md` §3): identischer Sänger
⇒ Kosinus HOCH trotz Verarbeitung (Hall/Sättigung/Bandlimit); verschiedener
Sänger ⇒ NIEDRIGER. Der MuQ-Zeuge soll diese Trennung (Separation identisch /
verschieden) mindestens so gut treffen wie der DSP-Proxy.

Determinismus (§G5 (copilot-instructions.md)): Seed 42, deterministische
Degradationsklassen. Kein Training, keine Gewichtsänderung, kein Flag-Flip.

Quellen: Golden Set `audit/golden_listening_set.json` (Paare aus dem Set, wenn
geeignet) plus deterministisch degradierte Kopien eines Referenzsignals als
„identisch"-Paare und unabhängige synthetische Sänger (Sinus/Ton-Klassen) als
„verschieden"-Paare — ehrlich dokumentiert (§G8 (copilot-instructions.md)),
wenn das Golden Set keine Sänger-Identitäts-Paare enthält.

Report: docs/reports/current/<datum>_kp3_vokal_identitaet.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import wavfile

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_GOLDEN = _ROOT / "audit" / "golden_listening_set.json"
_REPORT_DIR = _ROOT / "docs" / "reports" / "current"
_SR = 48000
_CLIP_S = 6


def _load_mono(path: Path, seconds: int, seed: int) -> tuple[np.ndarray, int] | None:
    """Lädt mono, float32; None wenn Datei/Codec nicht lesbar (ehrlich §G8, copilot-instructions.md)."""
    try:
        wf = wavfile.read(str(path))
    except Exception:
        return None
    sr = int(wf[0])
    wav = np.asarray(wf[1])
    if wav.dtype == np.int16:
        wav = wav.astype(np.float32) / 32768.0
    elif wav.dtype == np.int32:
        wav = wav.astype(np.float32) / 2147483648.0
    else:
        wav = wav.astype(np.float32)
    if wav.ndim == 2:
        if wav.shape[0] <= 2 and wav.shape[0] < wav.shape[1]:
            wav = wav.mean(axis=0)
        else:
            wav = wav.mean(axis=1)
    n_need = seconds * sr
    if wav.size < 256:
        return None
    if wav.size <= n_need:
        return np.ascontiguousarray(wav.astype(np.float32)), sr
    rng = np.random.RandomState(seed)
    start = rng.randint(0, wav.size - n_need)
    return np.ascontiguousarray(wav[start : start + n_need].astype(np.float32)), sr


def _peak_norm(x: np.ndarray, peak: float = 0.9) -> np.ndarray:
    m = float(np.max(np.abs(x))) + 1e-12
    return np.asarray(x * (peak / m), dtype=np.float32)


def _degrade_hall(x: np.ndarray, sr: int, seed: int) -> np.ndarray:
    # Deterministische Hall-Klasse (IR = exponentiell abklingendes Rauschen, Seed 42).
    rng = np.random.RandomState(seed)
    ir_len = int(0.15 * sr)
    ir = rng.randn(ir_len).astype(np.float32) * np.exp(-np.linspace(0, 8.0, ir_len))
    ir /= np.abs(ir).sum() + 1e-12
    return np.asarray(np.convolve(x, ir, mode="same"), dtype=np.float32)


def _degrade_saturation(x: np.ndarray) -> np.ndarray:
    # Deterministische Sättigung (tanh-Nichtlinearität) — „identischer" Sänger, verarbeitet.
    return np.asarray(np.tanh(2.0 * x) / np.tanh(2.0), dtype=np.float32)


def _degrade_bandlimit(x: np.ndarray, sr: int, cutoff_hz: float) -> np.ndarray:
    from scipy.signal import butter

    from backend.core.audio_utils import safe_filtfilt  # pylint: disable=import-outside-toplevel

    b, a = butter(5, cutoff_hz / (0.5 * sr), btype="low", output="ba")
    return np.asarray(safe_filtfilt(b, a, x), dtype=np.float32)


def _synthetic_singer(seed: int, seconds: int = _CLIP_S, sr: int = _SR, fund: float = 220.0) -> np.ndarray:
    """Deterministischer synthetischer „Sänger" (harmonisches Summensignal) für verschieden-Paare."""
    rng = np.random.RandomState(seed)
    t = np.linspace(0, seconds, seconds * sr, endpoint=False, dtype=np.float32)
    x = np.zeros_like(t)
    for h in range(1, 6):
        f = fund * h * (1.0 + 0.01 * np.sin(2 * np.pi * 5.0 * t))
        x = x + (1.0 / h) * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    return _peak_norm(x.astype(np.float32))


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.clip(np.dot(a, b) / (na * nb), 0.0, 1.0))


def _dsp_proxy(pre: np.ndarray, post: np.ndarray, sr: int) -> float | None:
    """Heutige DSP-Proxymessung (measure_singer_identity_cosine) — resemblyzer_music=False ⇒ Proxy."""
    try:
        from backend.core.dsp.level_1_invariants_guard import measure_singer_identity_cosine

        return measure_singer_identity_cosine(pre, post, sr)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[dsp] nicht verfügbar ({exc})", file=sys.stderr)
        return None


def _muq_witness(pre: np.ndarray, post: np.ndarray, sr: int) -> float | None:
    try:
        from backend.core.dsp.music_identity_oracle import music_identity_cosine

        cosine, _reasons = music_identity_cosine(pre, post, sr)
        return cosine
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[muq] nicht verfügbar ({exc})", file=sys.stderr)
        return None


def _degradations(x: np.ndarray, sr: int, seed: int) -> dict[str, np.ndarray]:
    return {
        "hall": _degrade_hall(x, sr, seed),
        "saturation": _degrade_saturation(x),
        "bandlimit_8k": _degrade_bandlimit(x, sr, 8000.0),
        "bandlimit_12k": _degrade_bandlimit(x, sr, 12000.0),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42, help="Seed (Determinismus §G5 (copilot-instructions.md))")
    ap.add_argument("--seconds", type=int, default=_CLIP_S)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    # ── Identisch-Paare: deterministisch degradierte Kopien einer Referenz ──
    identity_pairs: list[tuple[str, np.ndarray, np.ndarray, int]] = []
    # Referenzsignale: ein synthetischer Sänger (Seed 42) als „identischer Sänger".
    ref = _synthetic_singer(args.seed, args.seconds, _SR, 220.0)
    # Golden-Set-Quelle als zusätzliche Referenz (falls lesbar).
    golden_ref: tuple[np.ndarray, int] | None = None
    golden_path = ""
    if _GOLDEN.is_file():
        try:
            items = json.loads(_GOLDEN.read_text(encoding="utf-8")).get("items", [])
        except Exception:
            items = []
        for it in items:
            p = _ROOT / it.get("path", "")
            if p.is_file():
                loaded = _load_mono(p, args.seconds, args.seed)
                if loaded is not None:
                    golden_ref = loaded
                    golden_path = it.get("path", "")
                    break
    src_refs: list[tuple[str, np.ndarray, int]] = [("synthetic_singer", ref, _SR)]
    if golden_ref is not None:
        src_refs.append((f"golden:{Path(golden_path).stem}", _peak_norm(golden_ref[0]), golden_ref[1]))

    for name, r, sr in src_refs:
        degs = _degradations(r, sr, args.seed)
        for cls, d in degs.items():
            identity_pairs.append((f"{name}/{cls}", r, d, sr))

    # ── Verschieden-Paare: unabhängige synthetische Sänger (andere Grundfreq) ──
    different_pairs: list[tuple[str, np.ndarray, np.ndarray, int]] = []
    other_funds = [196.0, 261.6, 329.6]
    for i, f in enumerate(other_funds):
        other = _synthetic_singer(args.seed + 100 + i, args.seconds, _SR, f)
        different_pairs.append((f"other_singer_{i}", ref, other, _SR))

    report: dict[str, Any] = {
        "date": date.today().isoformat(),
        "script": "scripts/benchmark_identity_oracles.py",
        "seed": args.seed,
        "note": (
            "Golden Set audit/golden_listening_set.json enthält material/degradationsbezogene "
            "Metadaten, aber KEINE Sänger-Identitäts-Paare (gleicher Sänger in mehreren Aufnahmen) — "
            "für Identitäts-Invarianz werden deterministische synthetische Sänger + degradierte Kopien "
            "genutzt (§G8 (copilot-instructions.md), ehrlich dokumentiert)."
        ),
        "identity_pairs": [],
        "different_pairs": [],
        "summary": {},
    }

    def _record(pairs: list[tuple[str, np.ndarray, np.ndarray, int]], out_key: str) -> None:
        for name, pre, post, sr in pairs:
            dsp = _dsp_proxy(pre, post, sr)
            muq = _muq_witness(pre, post, sr)
            report[out_key].append(
                {
                    "name": name,
                    "dsp_cosine": round(dsp, 4) if dsp is not None else None,
                    "muq_cosine": round(muq, 4) if muq is not None else None,
                }
            )

    _record(identity_pairs, "identity_pairs")
    _record(different_pairs, "different_pairs")

    # ── Zusammenfassung: Trennung identisch vs. verschieden je Verfahren ──
    def _summary(key: str, want_high: bool) -> dict[str, Any]:
        rows = report[key]
        dsp = [r["dsp_cosine"] for r in rows if r["dsp_cosine"] is not None]
        muq = [r["muq_cosine"] for r in rows if r["muq_cosine"] is not None]
        return {
            "n": len(rows),
            "dsp_mean": round(float(np.mean(dsp)), 4) if dsp else None,
            "dsp_n_measurable": len(dsp),
            "muq_mean": round(float(np.mean(muq)), 4) if muq else None,
            "muq_n_measurable": len(muq),
            "want_high": want_high,
        }

    report["summary"]["identity"] = _summary("identity_pairs", True)
    report["summary"]["different"] = _summary("different_pairs", False)

    # Separation-Metrik: mean(cos_ident) − mean(cos_diff) je Verfahren (höher = besser getrennt).
    for proc in ("dsp", "muq"):
        i_mean = report["summary"]["identity"].get(f"{proc}_mean")
        d_mean = report["summary"]["different"].get(f"{proc}_mean")
        if i_mean is not None and d_mean is not None:
            report["summary"][f"{proc}_separation"] = round(i_mean - d_mean, 4)
        else:
            report["summary"][f"{proc}_separation"] = None

    out = Path(args.out) if args.out else _REPORT_DIR / f"{date.today().isoformat()}_kp3_vokal_identitaet.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Report: {out}")
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
