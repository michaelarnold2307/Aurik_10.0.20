"""WP-3 · Wohlklang-Ordnung-Forensik — verletzt der Studio-Vocoder die Hörordnung?

Hypothese (WP-3, `docs/TODOS_SOTA_ROADMAP.md`): der sprach-trainierte
Vocos-48k-Primärpfad des §1.4-Studio-Finishers (D-K2-3) könnte niederrangige
Ziele (z. B. Brillanz) verbessern, während höherrangige (Timbre-Authentizität,
Natürlichkeit) degradieren — ein Hörordnungs-Verstoß (§5
`hoerordnung.instructions.md`).

Messung (deterministisch, §G5 (copilot-instructions.md), Seed 42):
1. MUSDB18-Teststems (vocals + mixture), feste 8-s-Ausschnitte.
2. Vocoder-Pfad: `plugins/vocos_plugin.get_vocos_plugin().vocode(audio, sr,
   mode="studio2026")` — derselbe Aufruf wie im §1.4-Block von
   `unified_restorer_v3.py` (48-kHz-Stufe primär, §V6-Fallback intern).
3. Goal-Scores vor/nach über
   `backend.core.musical_goals.musical_goals_metrics.measure_all(audio, sr)`.
4. Ordnungs-Prüfung je Signal: für jedes Paar (verbessert, degradiert) mit
   |delta| ≥ 0.02 (Messrauschen-Schwelle, §v10.306 (copilot-instructions.md))
   prüft `GoalPriorityProtocol.would_violate_hearing_order(improving, at_cost_of)`.

Ausgabe: docs/reports/current/<datum>_wp3_wohlklang_forensik.json
"""

from __future__ import annotations

import argparse
import json
import sys
import wave
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MUSDB_TEST = ROOT / "data" / "musdb18hq" / "test"
DELTA_EPS = 0.02  # Messrauschen-Schwelle (§v10.306 (copilot-instructions.md))


def _read_excerpt(path: Path, n_samples: int, offset_s: float = 30.0) -> np.ndarray | None:
    try:
        with wave.open(str(path), "rb") as wf:
            sr = wf.getframerate()
            n_ch = wf.getnchannels()
            start = int(offset_s * sr)
            need = n_samples  # Aufrufer rechnet Sekunden bereits in Datei-Rate um
            wf.setpos(start)
            raw = wf.readframes(need)
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        audio = audio.reshape(-1, n_ch).T if n_ch > 1 else audio.reshape(1, -1)
        return audio.mean(axis=0)  # Mono für die Goal-Messung
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[wav] {path.name} nicht lesbar ({exc})", file=sys.stderr)
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42, help="Seed (§G5 (copilot-instructions.md))")
    ap.add_argument("--seconds", type=float, default=8.0, help="Ausschnittlänge je Signal")
    ap.add_argument("--offset", type=float, default=30.0, help="Start-Offset im Stem (s)")
    ap.add_argument("--tracks", type=int, default=2, help="Anzahl MUSDB-Test-Tracks")
    ap.add_argument(
        "--out", type=str, default=f"docs/reports/current/{date.today().isoformat()}_wp3_wohlklang_forensik.json"
    )
    args = ap.parse_args()

    np.random.seed(args.seed)

    from backend.core.goal_priority_protocol import GoalPriorityProtocol
    from backend.core.musical_goals.musical_goals_metrics import measure_all

    _proto = GoalPriorityProtocol()
    _tier = _proto.HEARING_TIER_MAP

    results: list[dict] = []
    tracks = sorted([p for p in MUSDB_TEST.iterdir() if p.is_dir()])[: args.tracks] if MUSDB_TEST.is_dir() else []

    for track_dir in tracks:
        for stem_name in ("vocals", "mixture"):
            stem_path = track_dir / (stem_name + ".wav")
            if not stem_path.is_file():
                continue
            with wave.open(str(stem_path), "rb") as wf:
                sr_in = wf.getframerate()
                n_samples = int(args.seconds * sr_in)
            audio = _read_excerpt(stem_path, n_samples, offset_s=args.offset)
            if audio is None or len(audio) < sr_in // 2:
                continue

            # Produktions-Äquivalenz: UV3 arbeitet intern mit 48 kHz — der
            # §1.4-Vocode-Aufruf erhält 48 kHz. Deterministisch resamplen (§G5, copilot-instructions.md).
            if sr_in != 48000:
                from math import gcd

                from scipy.signal import resample_poly

                _g = gcd(sr_in, 48000)
                audio = resample_poly(audio, 48000 // _g, sr_in // _g).astype(np.float32)
                sr_in = 48000

            goals_before = measure_all(audio, sr_in)

            vocoded = None
            vocode_reason = ""
            try:
                from plugins.vocos_plugin import get_vocos_plugin

                _plug = get_vocos_plugin()
                _res = _plug.vocode(audio, sr_in, mode="studio2026")
                vocoded = _res.audio
                if vocoded is not None and vocoded.ndim > 1:
                    vocoded = vocoded.mean(axis=0)
                if vocoded is not None and len(vocoded) > len(audio):
                    vocoded = vocoded[: len(audio)]
                if vocoded is not None and len(vocoded) < len(audio):
                    vocoded = np.pad(vocoded, (0, len(audio) - len(vocoded)))
                vocode_reason = str(getattr(_res, "model_used", "?"))
            except Exception as exc:  # pylint: disable=broad-except
                vocode_reason = f"nicht verfügbar ({exc}) — §V6 (copilot-instructions.md)"

            goals_after = measure_all(vocoded, sr_in) if vocoded is not None else {}
            goals_after = {k: v for k, v in goals_after.items() if isinstance(v, (int, float)) and np.isfinite(v)}
            goals_before = {k: v for k, v in goals_before.items() if isinstance(v, (int, float)) and np.isfinite(v)}

            # Per-Goal-Deltas über die gemeinsamen Ziele.
            common = sorted(set(goals_before) & set(goals_after))
            deltas = {g: float(goals_after[g]) - float(goals_before[g]) for g in common}
            improved = [g for g, d in deltas.items() if d >= DELTA_EPS]
            degraded = [g for g, d in deltas.items() if d <= -DELTA_EPS]

            violations: list[dict] = []
            for g_i in improved:
                for g_j in degraded:
                    if _proto.would_violate_hearing_order(g_i, g_j):
                        violations.append(
                            {
                                "improving_goal": g_i,
                                "improving_delta": round(deltas[g_i], 4),
                                "at_cost_of": g_j,
                                "cost_delta": round(deltas[g_j], 4),
                                "tier_improving": _tier.get(g_i, 3),
                                "tier_cost": _tier.get(g_j, 3),
                            }
                        )

            results.append(
                {
                    "track": track_dir.name,
                    "stem": stem_name,
                    "sr": sr_in,
                    "vocode_model": vocode_reason,
                    "goals_before": {k: round(v, 4) for k, v in sorted(goals_before.items())},
                    "goals_after": {k: round(v, 4) for k, v in sorted(goals_after.items())},
                    "hearing_order_violations": violations,
                }
            )

    report = {
        "date": date.today().isoformat(),
        "script": "scripts/forensik_vocoder_ordnung.py",
        "seed": args.seed,
        "excerpt_seconds": args.seconds,
        "offset_seconds": args.offset,
        "delta_eps": DELTA_EPS,
        "results": results,
        "summary": {
            "n_signals": len(results),
            "n_violations": sum(len(r["hearing_order_violations"]) for r in results),
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Report: {out_path}")
    for r in results:
        print(f"{r['track']}/{r['stem']}: model={r['vocode_model']} Verstöße={len(r['hearing_order_violations'])}")
    print(f"Gesamt-Verstöße: {report['summary']['n_violations']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
