# Whisper large-v3-turbo A/B — NEGATIVES RESULTAT (Rollen A/B ausgeschlossen)

**Datum:** 2026-10-03 · **Vertrag:** `scripts/validate_whisper_turbo_ab.py` (§SOTA-ML-V10)
**Grundsatz:** „Negative Resultate als Spec schützen vor Wiederholung" (Wohlklang-Roadmap, Punkt 5).
**Urteil:** **NO_GO** für beide geprüften Rollen — Turbo wird auf dem vorhandenen Verbraucher NICHT aktiviert.

## Gemessene Gates (Gesang, geseedet, Seed 20261003, 3 Fenster × 30 s)

| Gate | Messung | Vertrag | Urteil |
|---|---|---|---|
| A1 Vocal-Activity-IoU Δ(turbo−tiny) | **−0,199** (95 %-CI −0,231…−0,166) | Never-worsen ≥ −0,02 | FAIL |
| A2 Verbraucher-Parität frame-RMS | **r = 0,243** | ≥ 0,90 | FAIL |
| A3 Determinismus | tiny/turbo max&#124;Δ&#124; = 0,0 | bit-identisch (§G5) | PASS |
| B1 Grenzen-F1 Δ (Onsets) | **−0,118** (95 %-CI −0,159…−0,078) | ≥ −0,05 | FAIL |

## Interpretation (für künftige Sessions)

1. **Die Hidden-RMS-Verbraucher-Heuristik ist modellspezifisch** (r = 0,243): Der 60-Perzentil-Schwellwert in `lyrics_guided_enhancement._transcribe_onnx` ist auf Tiny-384-Statistik kalibriert. Ein Embedding-Platz-Tausch ohne **Adapter + Re-Kalibrierung + Vertrags-Neulauf** ist gesperrt.
2. **Kein direkter 384-dim-Verbraucher** (§v10.20 2M-Decoder) darf Turbo-Zustände sehen — Dimension 1280 statt 384, Aktivierung ohne Projektions-Adapter verboten.
3. **Messbedingung:** ONNX-CPU; Turbo-fp16-Export lädt nur mit `ORT_DISABLE_ALL` (Export-Defekt `SimplifiedLayerNormFusion`, ORT 1.27 — Engineering-Befund, Registry-Note).
4. **Stimuli waren synthetischer Gesang** — vor endgültiger Ablage des Ausschlusses mit echten Vokalstems wiederholen:
   `python scripts/validate_whisper_turbo_ab.py --mix <mix.wav> --vocals <vocals.wav>` (MUSDB18). Bis dahin gilt der Ausschluss fail-closed (§V6).

## Turbo bleibt hochwertig nutzbar — Kompetenz statt Verfügbarkeit

| Rolle | Nutzen für Gesang | Vertrag |
|---|---|---|
| Grenzen-Referenz (Decoder-Zeitmarken) | Wort-/Silben-/Konsonant-Onsets für phase_42 Formant-Gates, phase_58 Alignment, Konsonantenklarheit-Boosts | eigener Vertrag nötig |
| Semantischer Verständlichkeits-Witness | Onset-Schärfe original vs. restauriert als Never-worsen-Zweitstimme (HASPI/STI-Lücke, Punkt 6) | soft, Zeuge |
| Atem-/Phrasensegmentierung | Level-1-Invariante „Atemsegmente ±10 %" messbar machen (§v10.303.17) | soft |
| Teacher/Distillation | Offline-Verbesserung von Tiny-Student/wav2vec2-Aligner, null Laufzeit-Kosten | offline |
| Golden-Ear-Annotation | Erst-Annotation des Hörkorpus, menschlich korrigiert (Punkt 6) | assistiv |

ASR-*Text* auf Gesang ist bewusst außer Scope (Vorgabe 2026-10-03: Gesang priorisiert, Sprache nachrangig).

## Aktivierungs-Sperre (maschinenlesbar)

`AURIK_WHISPER_TURBO` (Opt-in in `whisper_torch_rocm`) darf erst nach dokumentiertem A/B-GO
einer Rolle gesetzt werden. Bis dahin: Tiny bleibt aktiv (deterministisch, ONNX-Parität belegt).
