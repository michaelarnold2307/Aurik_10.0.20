#!/usr/bin/env python3
"""§SOTA-ML-V10 (2026-10-03) — Whisper large-v3-turbo A/B-Vertrag (GESANG).

Aktivierungsvertrag im validate_hr_v1-Muster: Turbo (large-v3-Encoder,
1500×1280) wird nur dort aktiviert, wo er den Tiny-Encoder (1500×384, aktiv)
NIE verschlechtert. Priorität (Vorgabe 2026-10-03): Aurik ist Musik-
restaurierung — GESANG zählt; Sprache ist nachrangig. ASR/WER-Gates sind
deshalb explizit AUSSER SCOPE (negatives Resultat-Design: nie gegen Sprach-
Metriken optimieren, wenn Gesang das Ziel ist).

Rollen werden UNABHÄNGIG bewertet („Modellwahl nach Kompetenz, nicht
Verfügbarkeit"):

  Rolle A „Embedding-Platz" — Hidden-Zustände für den Hidden-RMS-Verbraucher
    (lyrics_guided_enhancement._transcribe_onnx, phase_58):
    A1 Vocal-Activity-Never-worsen auf GESANG: IoU vs Ground-Truth ≥ Tiny − TOL
       (95 %-CI über Fenster, Seed-paritätisch)
    A2 Verbraucher-Parität: Pearson(frame-RMS_tiny, frame-RMS_turbo) ≥ 0.90 —
       die Verbraucher-Oberfläche ist die Frame-RMS-Folge, NICHT die Breite
       (384 vs 1280). Direkte 384-dim-Verbraucher (§v10.20 2M-Decoder) bleiben
       ohne Projektions-Adapter gesperrt.
    A3 Determinismus: zwei Läufe bit-identisch (max|Δ| == 0, §G5 (GEBOTE.md)).

  Rolle B „Gesangs-Guide" — Phonem-/Wortgrenzen für die gesangsgeführte
    Bearbeitung (phase_42 Formant-Gates, phase_58 Alignment, §SOTA-ML-V10:
    „bessere Wortgrenzen"):
    B1 Onset-Never-worsen auf GESANG: F1 der Silben-/Konsonant-Grenzen
       (± 2 Frames = 40 ms) ≥ Tiny − TOL (95 %-CI).

Stimuli: --mix/--vocals (echte geseedete Vocal-Stems, z. B. MUSDB18) ODER
deterministische Gesangs-Synthese (Vibrato 5–7 Hz, AM-Silben, Formant-
Harmonische, geseedet — §G5 (GEBOTE.md)).

Exit-Codes: 0 = alle geprüften Rollen bestanden (Rolle darf aktiviert werden),
1 = mindestens eine Rolle verschlechtert/nicht kompatibel (Ausschluss dieser
Rolle — negatives Resultat als Spec führen), 2 = Setup-Fehler (fail-closed:
es wird nichts aktiviert).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logger = logging.getLogger(__name__)

SR16 = 16_000
N_FFT = 400
HOP = 160
MAX_FRAMES = 3000  # 30 s bei 10 ms Hop
FRAME_SAMPLES_20MS = 320  # Encoder-Fenster: 1500 Frames à 20 ms über 30 s
IOU_TOL = 0.02  # Never-worsen-Toleranz Rolle A1
PARITY_R_MIN = 0.90  # Mindest-Korrelation Rolle A2
ONSET_F1_TOL = 0.05  # Never-worsen-Toleranz Rolle B1
ONSET_TOL_FRAMES = 2  # ± 40 ms Grenz-Toleranz
CI_Z = 1.96  # 95 %-CI (Normal-Approximation über Fenster)

TINY_ONNX = "models/whisper/whisper_tiny.onnx"
TURBO_ONNX = "models/whisper/whisper_large_v3_turbo_encoder_fp16.onnx"


def _mel_features(mono_16k: np.ndarray, n_mels: int) -> np.ndarray:
    """Whisper-Log-Mel (Radford et al. 2022) — 80 (Tiny) bzw. 128 (Turbo) Bins.

    Identische Normalisierung zu lyrics_guided_enhancement._compute_mel_features;
    Rückgabe (1, n_mels, 3000), deterministisch (§G5 (GEBOTE.md)), NaN-sicher
    (§0a (copilot-instructions.md)).
    """
    import librosa

    x = np.nan_to_num(np.asarray(mono_16k, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    max_samples = MAX_FRAMES * HOP
    if x.size < max_samples:
        x = np.pad(x, (0, max_samples - x.size))
    else:
        x = x[:max_samples]
    mel = librosa.feature.melspectrogram(
        y=x, sr=SR16, n_fft=N_FFT, hop_length=HOP, n_mels=n_mels, fmin=0.0, fmax=8_000.0
    )
    log_mel = np.log10(np.maximum(mel, 1e-10))
    log_mel = np.maximum(log_mel, log_mel.max() - 8.0)
    log_mel = (log_mel + 4.0) / 4.0
    if log_mel.shape[1] < MAX_FRAMES:
        log_mel = np.pad(log_mel, ((0, 0), (0, MAX_FRAMES - log_mel.shape[1])))
    else:
        log_mel = log_mel[:, :MAX_FRAMES]
    return np.asarray(log_mel, dtype=np.float32)[np.newaxis, ...]


def _load_encoder(path: str, disable_optimizers: bool = False):
    """ONNX-CPU-Encoder laden. Turbo-fp16-Export lädt nur mit ORT_DISABLE_ALL

    (Befund 2026-10-03: SimplifiedLayerNormFusion bricht den Graph —
    §V6 (VERBOTEN.md): dokumentierter Workaround statt stilles Scheitern).
    """
    import onnxruntime as ort

    so = ort.SessionOptions()
    if disable_optimizers:
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        logger.warning(
            "§V6 (VERBOTEN.md): %s lädt nur mit ORT_DISABLE_ALL (fp16-Export-Defekt "
            "SimplifiedLayerNormFusion) — dokumentierter Workaround",
            path,
        )
    return ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])


def _encode(session, mel: np.ndarray) -> np.ndarray:
    out = session.run(None, {"input_features": mel})[0]
    return np.nan_to_num(np.asarray(out, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def _consumer_activity(hidden: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """EXAKTE Verbraucher-Logik (lyrics_guided_enhancement._transcribe_onnx):

    frame_energy = RMS über Hidden-Dim → max-normalisiert → Aktivität über
    60. Perzentil. Rückgabe (frame_energy_norm, active_bool) à 1500 Frames.
    """
    frame_energy = np.sqrt(np.mean(hidden[0] ** 2, axis=-1))
    e_max = float(frame_energy.max()) or 1.0
    frame_energy = (frame_energy / e_max).astype(np.float32)
    active = frame_energy > float(np.percentile(frame_energy, 60.0))
    return frame_energy, active


def _frame_truth(gt_samples: np.ndarray, n_frames: int = 1500) -> np.ndarray:
    g = np.asarray(gt_samples, dtype=bool)
    if g.size < n_frames * FRAME_SAMPLES_20MS:
        g = np.pad(g, (0, n_frames * FRAME_SAMPLES_20MS - g.size))
    return g[: n_frames * FRAME_SAMPLES_20MS].reshape(n_frames, FRAME_SAMPLES_20MS).mean(axis=1) > 0.5


def _onsets_from_energy(energy: np.ndarray) -> list[int]:
    """Grenzen aus Frame-Energie-Anstiegen (Silben-/Konsonant-Onsets)."""
    d = np.diff(energy, prepend=energy[:1])
    thr = float(np.percentile(d, 92.0))
    cand = np.flatnonzero(d > max(thr, 1e-6))
    out: list[int] = []
    for i in cand:
        if not out or i - out[-1] > 5:  # min. 100 ms Abstand
            out.append(int(i))
    return out


def _onset_f1(pred: list[int], truth: list[int], tol: int = ONSET_TOL_FRAMES) -> float:
    if not truth and not pred:
        return 1.0
    if not truth or not pred:
        return 0.0
    hit_p = sum(1 for p in pred if any(abs(p - t) <= tol for t in truth))
    hit_t = sum(1 for t in truth if any(abs(p - t) <= tol for p in pred))
    prec, rec = hit_p / len(pred), hit_t / len(truth)
    return float(2 * prec * rec / (prec + rec)) if prec + rec > 0 else 0.0


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = int(np.sum(a | b))
    return float(np.sum(a & b) / union) if union else 1.0


def _mean_ci(values: list[float]) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean()) if arr.size else 0.0
    if arr.size < 2:
        return mean, mean, mean
    lo = mean - CI_Z * float(arr.std(ddof=1)) / np.sqrt(arr.size)
    hi = mean + CI_Z * float(arr.std(ddof=1)) / np.sqrt(arr.size)
    return mean, lo, hi


def _synth_singing_window(seed: int, seconds: float = 30.0) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Deterministische Gesangs-Synthese: Phrasen aus Vibrato-Silben mit
    Formant-Harmonischen über Rauschteppich + Ground-Truth-Aktivität/-Onsets."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR16)
    t = np.arange(n) / SR16
    audio = 0.005 * rng.standard_normal(n)
    gt = np.zeros(n, dtype=bool)
    onsets: list[int] = []
    pos = int(rng.uniform(0.2, 0.8) * SR16)
    while pos < n - SR16:
        end = min(n, pos + int(rng.uniform(4.0, 6.5) * SR16))
        syl = pos
        while syl < end - int(0.15 * SR16):
            seg_end = min(end, syl + int(rng.uniform(0.18, 0.38) * SR16))
            onsets.append(syl)
            seg_t = t[syl:seg_end] - t[syl]
            f0 = float(rng.uniform(180.0, 500.0))
            vib = 1.0 + 0.012 * np.sin(2 * np.pi * float(rng.uniform(5.0, 7.0)) * t[syl:seg_end])
            env = np.hanning(seg_end - syl) ** 0.5
            tone = np.zeros(seg_end - syl)
            for h in range(1, 14):
                formant = np.exp(-0.5 * ((h * f0 - float(rng.uniform(500.0, 2800.0))) / 450.0) ** 2)
                tone += (formant / h) * np.sin(
                    2 * np.pi * h * f0 * np.cumsum(vib) / SR16 + float(rng.uniform(0.0, 2 * np.pi))
                )
            audio[syl:seg_end] += 0.22 * env * tone / (float(np.max(np.abs(tone))) + 1e-9)
            gt[syl:seg_end] = True
            del seg_t
            syl = seg_end + int(rng.uniform(0.02, 0.12) * SR16)
        pos = end + int(rng.uniform(0.8, 2.0) * SR16)
    return audio.astype(np.float32), gt, onsets


def _load_real_stems(mix_path: str, vocals_path: str, seconds: float) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Echte Stems: Input = Mix (Verbraucher-Realität), GT-Aktivität/Onsets aus
    dem Vocal-Stem (Gesang vs Atem/Pausen)."""
    import librosa

    mix, _ = librosa.load(mix_path, sr=SR16, mono=True)
    voc, _ = librosa.load(vocals_path, sr=SR16, mono=True)
    n = int(seconds * SR16)
    mix = np.pad(np.asarray(mix[:n], dtype=np.float32), (0, max(0, n - len(mix))))
    voc = np.pad(np.asarray(voc[:n], dtype=np.float32), (0, max(0, n - len(voc))))
    env = np.sqrt(np.mean(voc.reshape(-1, FRAME_SAMPLES_20MS) ** 2, axis=1))
    gt_frames = env > max(0.25 * float(env.max()), 1e-9)
    gt = np.repeat(gt_frames, FRAME_SAMPLES_20MS)[: mix.size]
    onsets = [i * FRAME_SAMPLES_20MS for i in _onsets_from_energy(env / (float(env.max()) or 1.0))]
    return mix, gt, onsets


def run_ab(args: argparse.Namespace) -> int:
    try:
        tiny = _load_encoder(args.tiny_onnx)
        turbo = _load_encoder(args.turbo_onnx, disable_optimizers=True)
    except Exception as exc:  # fail-closed (§V6 (VERBOTEN.md))
        print(f"[Setup] Encoder nicht ladbar: {exc} → Exit 2 (fail-closed, keine Aktivierung)")
        return 2

    windows: list[tuple[np.ndarray, np.ndarray, list[int]]] = []
    if args.mix and args.vocals:
        try:
            windows.append(_load_real_stems(args.mix, args.vocals, args.seconds))
        except Exception as exc:
            print(f"[Setup] Stems nicht ladbar: {exc} → Exit 2 (fail-closed)")
            return 2
    for i in range(args.windows):
        windows.append(_synth_singing_window(args.seed + i, args.seconds))

    iou_t, iou_x, parity_r, f1_t, f1_x = [], [], [], [], []
    det_t = det_x = None
    try:
        for w_i, (audio, gt, gt_on) in enumerate(windows):
            mel80 = _mel_features(audio, 80)
            mel128 = _mel_features(audio, 128)
            h_t, h_x = _encode(tiny, mel80), _encode(turbo, mel128)
            if w_i == 0:
                det_t = float(np.max(np.abs(h_t - _encode(tiny, mel80))))
                det_x = float(np.max(np.abs(h_x - _encode(turbo, mel128))))
            e_t, a_t = _consumer_activity(h_t)
            e_x, a_x = _consumer_activity(h_x)
            gt_frames = _frame_truth(gt)
            iou_t.append(_iou(a_t, gt_frames))
            iou_x.append(_iou(a_x, gt_frames))
            parity_r.append(float(np.corrcoef(e_t, e_x)[0, 1]))
            f1_t.append(_onset_f1(_onsets_from_energy(e_t), [o // FRAME_SAMPLES_20MS for o in gt_on]))
            f1_x.append(_onset_f1(_onsets_from_energy(e_x), [o // FRAME_SAMPLES_20MS for o in gt_on]))
    except Exception as exc:
        print(f"[Setup] Messung fehlgeschlagen: {exc} → Exit 2 (fail-closed)")
        return 2

    d_iou = [x - t for x, t in zip(iou_x, iou_t)]
    d_f1 = [x - t for x, t in zip(f1_x, f1_t)]
    m_iou, lo_iou, hi_iou = _mean_ci(d_iou)
    m_f1, lo_f1, hi_f1 = _mean_ci(d_f1)
    r_mean = float(np.mean(parity_r))

    a1 = lo_iou >= -IOU_TOL
    a2 = r_mean >= PARITY_R_MIN
    a3 = det_t == 0.0 and det_x == 0.0
    b1 = lo_f1 >= -ONSET_F1_TOL

    print("=== §SOTA-ML-V10 Whisper-Turbo A/B (GESANG) ===")
    print(f"[A1 IoU Δ(turbo−tiny)]: {m_iou:+.3f} (95 %-CI {lo_iou:+.3f}…{hi_iou:+.3f} | Never-worsen ≥ −{IOU_TOL:.2f})")
    print(f"[A2 Parität frame-RMS]: r={r_mean:.3f} (min {PARITY_R_MIN:.2f})")
    print(f"[A3 Determinismus]: tiny max|Δ|={det_t:.1e} | turbo max|Δ|={det_x:.1e} (0.0 = bit-identisch)")
    print(f"[B1 Onset-F1 Δ]: {m_f1:+.3f} (95 %-CI {lo_f1:+.3f}…{hi_f1:+.3f} | Never-worsen ≥ −{ONSET_F1_TOL:.2f})")
    role_a = a1 and a2 and a3
    role_b = a3 and b1
    print(f"[Rolle A Embedding-Platz]: {'bestanden' if role_a else 'verschlechtert'}")
    print(f"[Rolle B Gesangs-Guide]:   {'bestanden' if role_b else 'verschlechtert'}")
    if role_a:
        print(
            "[Vertrag] Rolle A: Hidden-RMS-Verbraucher dürfen Turbo nutzen; 384-dim-Verbraucher bleiben ohne Adapter gesperrt."
        )
    else:
        print("[Vertrag] Rolle A: AUSGESCHLOSSEN — negatives Resultat als Spec führen (§SOTA-ML-V10).")
    if role_b:
        print("[Vertrag] Rolle B: Turbo für Phonem-/Wortgrenzen der gesangsgeführten Bearbeitung freigeben.")
    else:
        print("[Vertrag] Rolle B: AUSGESCHLOSSEN — negatives Resultat als Spec führen.")
    return 0 if (role_a and role_b) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Whisper large-v3-turbo A/B-Vertrag (Gesang)")
    parser.add_argument("--mix", type=str, default="", help="Mix-WAV (Verbraucher-Input)")
    parser.add_argument("--vocals", type=str, default="", help="Vocal-Stem-WAV (Ground-Truth Gesang)")
    parser.add_argument("--windows", type=int, default=5, help="synthetische Fenster (Seed-Serie)")
    parser.add_argument("--seconds", type=float, default=30.0, help="Fensterlänge (max. 30 s)")
    parser.add_argument("--seed", type=int, default=20261003, help="Basis-Seed (§G5 (GEBOTE.md))")
    parser.add_argument("--tiny-onnx", type=str, default=TINY_ONNX)
    parser.add_argument("--turbo-onnx", type=str, default=TURBO_ONNX)
    args = parser.parse_args()
    if not Path(args.tiny_onnx).exists() or not Path(args.turbo_onnx).exists():
        print("[Setup] Encoder-ONNX fehlt → Exit 2 (fail-closed, keine Aktivierung)")
        return 2
    return run_ab(args)


if __name__ == "__main__":
    sys.exit(main())
