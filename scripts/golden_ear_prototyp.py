"""Golden-Ear-Prototyp: Hörer-Präferenz-Modell aus echten Hör-Runden fitten.

Lernt `(perzeptive Feature-Differenz) -> deine Praeferenz` aus den gehörten
A/B-Trials (K0 + thresholds). Prototyp: logistische Regression auf den
MusikMOS-Subscore-Differenzen (Kandidat − Referenz). Zeigt, wie gut dein Ohr
schon abgebildet ist + welche Feature-Lücken die nächste Hör-Runde schließen muss.

(Artefakt-Prototyp, kein Produktionsmodul — Auswertung + Report.)
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")

REPO = Path(__file__).resolve().parent.parent
MUSHRA = REPO / "output_audio" / "mushra"


def load_study(study: str) -> list[dict]:
    """Lädt gehörte Trials: (candidate_clip, reference_clip, preferred_candidate)."""
    d = MUSHRA / study
    if not d.is_dir():
        return []
    key = {t["trial_id"]: t for t in json.loads((d / "trial_key.json").read_text())}
    rows = []
    with (d / "answers.csv").open() as f:
        for r in csv.DictReader(f):
            tid = r["trial_id"]
            t = key.get(tid)
            if not t or t.get("is_catch"):
                continue
            ans = r["answer_interval"].strip()
            # Kandidat/Referenz-Clip aus trials/ (X__A.wav / X__B.wav)
            cand = d / "trials" / f"{tid}__{t['candidate_in']}.wav"
            ref = d / "trials" / f"{tid}__{t['reference_in']}.wav"
            if not (cand.exists() and ref.exists()):
                continue
            rows.append(
                {
                    "study": study,
                    "trial_id": tid,
                    "cand": cand,
                    "ref": ref,
                    "pref_cand": 1 if ans == t["candidate_in"] else 0,
                }
            )
    return rows


def musikmos_vec(path: Path) -> np.ndarray | None:
    """MusikMOS-Subscore-Vektor (q_rausch, q_tonal, q_clip, q_hf, q_dyn, overall)."""
    try:
        import soundfile as sf

        from plugins.musik_mos_plugin import get_musik_mos

        audio, sr = sf.read(str(path), always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        res = get_musik_mos().estimate_mos(audio.astype(np.float32), sr)
        return np.array(
            [
                float(getattr(res, "q_rausch", 0.0)),
                float(getattr(res, "q_tonal", 0.0)),
                float(getattr(res, "q_clip", 0.0)),
                float(getattr(res, "q_hf", 0.0)),
                float(getattr(res, "q_dyn", 0.0)),
                float(getattr(res, "overall", getattr(res, "mos", 0.0))),
            ]
        )
    except Exception as e:  # §V6 (copilot-instructions.md)
        print(f"  [warn] MusikMOS fehlgeschlagen {path.name}: {e}")
        return None


def timbre_naturalness_vec(cand: Path, ref: Path) -> np.ndarray | None:
    """Timbre-/Natürlichkeits-Features (Kandidat − Referenz), die die MusikMOS
    verfehlt: Spektral-Huellkurven-Distanz (Timbre-Sprung/Dumpfheit), HF-Tilt,
    Transienten-Erhalt (Crest). §V6 (copilot-instructions.md): bei Fehlern None.
    """
    try:
        import soundfile as sf

        def _mono(p):
            a, sr = sf.read(str(p), always_2d=False)
            if a.ndim > 1:
                a = a.mean(axis=1)
            return a.astype(np.float64), sr

        ac, sr = _mono(cand)
        ar, _ = _mono(ref)
        n = min(len(ac), len(ar))
        if n < 1024:
            return None
        ac, ar = ac[:n], ar[:n]

        def _logmel_env(x):
            n_fft, hop = 2048, 512
            frames = 1 + max(0, (len(x) - n_fft) // hop)
            w = np.hanning(n_fft)
            S = np.zeros((frames, n_fft // 2 + 1))
            for i in range(frames):
                seg = x[i * hop : i * hop + n_fft] * w
                S[i] = np.abs(np.fft.rfft(seg))
            return np.log1p(S).mean(axis=0)

        timbre_dist = float(np.sqrt(np.mean((_logmel_env(ac) - _logmel_env(ar)) ** 2)))

        freqs = np.fft.rfftfreq(2048, 1.0 / sr)
        hi = freqs > 8000

        def _hf_tilt(x):
            S = np.abs(np.fft.rfft(x))
            return float(np.log(1e-10 + S[hi].mean())) - float(np.log(1e-10 + S.mean()))

        hf_tilt = _hf_tilt(ac) - _hf_tilt(ar)

        def _crest(x):
            return float(np.abs(x).max() / (np.sqrt(np.mean(x**2)) + 1e-9))

        transient_pres = _crest(ac) - _crest(ar)
        return np.array([timbre_dist, hf_tilt, transient_pres])
    except Exception as e:  # §V6 (copilot-instructions.md)
        print(f"  [warn] Timbre-Features fehlgeschlagen {cand.name}: {e}")
        return None


def fit_logistic(X: np.ndarray, y: np.ndarray, lr: float = 0.5, epochs: int = 2000):
    """Einfache logistische Regression (Gradientenabstieg, deterministisch §G5 (copilot-instructions.md))."""
    w = np.zeros(X.shape[1])
    b = 0.0
    for _ in range(epochs):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        err = p - y
        w -= lr * (X.T @ err) / len(y)
        b -= lr * err.mean()
    return w, b


def main() -> int:
    studies = ["thresholds_scnet_c4_fair_r2", "thresholds_sgmse_3weg"]
    rows = []
    for s in studies:
        rows.extend(load_study(s))
    print(f"Gehörte gültige Trials: {len(rows)}  (aus {len(studies)} Studien)")

    X, y = [], []
    for r in rows:
        cv = musikmos_vec(r["cand"])
        rv = musikmos_vec(r["ref"])
        if cv is None or rv is None:
            continue
        diff = list(cv - rv)  # MusikMOS-Differenz Kandidat − Referenz
        tn = timbre_naturalness_vec(r["cand"], r["ref"])
        if tn is not None:
            diff.extend(tn.tolist())
        X.append(diff)
        y.append(r["pref_cand"])
    X = np.array(X)
    y = np.array(y, dtype=float)
    print(f"Für Fit nutzbar: {len(y)} Trials  (Präferenz Kandidat: {int(y.sum())}/{len(y)})")

    if len(y) < 6:
        print("Zu wenig Daten für einen belastbaren Fit — mehr Hör-Runden nötig.")
        return 0

    # Holdout: letzte ~25% als Validierung (Determinismus §G5 (copilot-instructions.md), fixe Reihenfolge)
    n_tr = max(4, int(len(y) * 0.75))
    Xtr, ytr = X[:n_tr], y[:n_tr]
    Xte, yte = X[n_tr:], y[n_tr:]
    w, b = fit_logistic(Xtr, ytr)

    def acc(X_, y_):
        z = X_ @ w + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        pred = (p >= 0.5).astype(float)
        return float((pred == y_).mean()) if len(y_) else float("nan")

    names = [
        "q_rausch(Defektlast)",
        "q_tonal",
        "q_clip",
        "q_hf",
        "q_dyn",
        "overall",
        "timbre_dist",
        "hf_tilt",
        "transient_pres",
    ]
    print(
        f"\nTrain-Accuracy: {acc(Xtr, ytr):.2f}   Holdout-Accuracy: {acc(Xte, yte):.2f}  (n_train={len(ytr)}, n_hold={len(yte)})"
    )
    print("Feature-Gewichte (positiv = bevorzugt wenn Kandidat höher):")
    order = np.argsort(-np.abs(w))
    for i in order:
        print(f"  {names[i]:22s}  w={w[i]:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
