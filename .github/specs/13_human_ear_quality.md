# Spec 13: Klangqualität für das menschliche Ohr | §v10 Pleasantness-First

> **Version:** Aurik 10.0.0 · **Scope:** Perceptual Audio Quality, ISO 226, Listening Modes, Fatigue Prevention

## §13.0 [RELEASE_MUST] Oberstes Prinzip: Natürlicher Wohlklang

Aurik optimiert nicht für technische Metriken (SNR, THD, PESQ), sondern für das
menschliche Gehör. Das bedeutet:

1. **Bewahren vor Verbessern** — Was nicht hörbar kaputt ist, wird nicht angefasst
2. **Physikalische Grenzen akzeptieren** — Fehlende Frequenzen können nicht erfunden werden
3. **Fließend, nicht segmentiert** — Keine hörbaren Übergänge zwischen Sektionen
4. **Das Ohr entscheidet, nicht die Metrik** — MUSHRA misst Ähnlichkeit, nicht Qualität

## §13.1 Das menschliche Ohr als ultimatives Messinstrument

### §13.1.1 Frequenzabhängige Wahrnehmung (ISO 226:2003)

| Bereich | Empfindlichkeit | Gewichtung |
|---|---|---|
| Bass (20–200 Hz) | −20 dB | 0.5× |
| Mitten (200–2000 Hz) | Referenz | 1.0× |
| Präsenz (2000–8000 Hz) | +10 dB | 1.5× |
| Luft (8000–20000 Hz) | Altersabhängig | 0.7× |

### §13.1.2 Psychoakustische Grenzen der Restaurierung

**§13.1.2.1 Air-Band auf analogem Material:** Unwiderruflich zerstört. Kein Exciter.

**§13.1.2.2 Transient Shaping ohne HF:** bw_loss > 0.8 → Phase 36 skipped.

**§13.1.2.3 Preservation Mode (§2.16):** bw_loss ≥ 0.90 ∧ SNR < 16 dB → global_scalar ≤ 0.70.

## §13.2 SectionStrengthEnvelope — Fließende per-Segment-Anpassung

**§13.2.1 Aktivierung (§2.17-ACTIVE):** Die Envelope-Infrastruktur existiert und wird
zentral in `_profiled_phase_call()` injiziert. Alle Phasen, die im Frequenzbereich
2–8 kHz (Präsenz, max. Ohrempfindlichkeit) arbeiten, MÜSSEN die Envelope lesen:

- Phase 19 (De-Esser) — `strength × envelope[frame]`
- Phase 38 (Presence Boost) — `strength × envelope[frame]`
- Phase 18 (Noise Gate) — `strength × envelope[frame]`

**§13.2.2 Cosine-Crossfade** 200ms zwischen allen Sektionen. Max. 1 dB/100ms.

**§13.2.3 Invarianten:** Räumlichkeit, Rauschflor, Loudness bleiben song-global.

## §13.3 Blind Reference-Free Quality (§13.8 ✅ Implementiert — mert_mushra_proxy.py, v10.0.8)

**Konzept:** „Wie gut KÖNNTE dieser Song klingen?" — absolute Qualitätsschätzung
ohne Vergleich zum degradierten Original. MERT-Embedding-basiert. Ermöglicht:

- Qualitätsprognose VOR der Restaurierung
- Abbruchkriterium: „Besser geht's nicht"
- Kein Over-Processing für bereits optimales Material

## §13.4 Human-Panel-kalibrierter MUSHRA (§13.9 ROADMAP)

**Konzept:** Ridge-Regression auf echten Hörtest-Daten kalibriert den MUSHRA-Proxy.

- Stufe 1 (heute): Literatur-Korrelationen
- Stufe 2 (implementiert, unkalibriert): Ridge-Regression via `calibrate_from_panel()`
- Stufe 3 (Ziel): Echte Panel-Daten → Gewichte rückprojiziert → CI-Proxy

## §13.5 Hörermüdungs-Prävention

### §13.5.1 Over-Processing vermeiden

Zu viele Phasen erzeugen ein „zu perfektes", künstlich klingendes Signal.

### §13.5.2 Fragile-Material-Guard (§2.15)

bw_loss ≥ 0.90 ∧ SNR < 16 dB → global_scalar ≤ 0.70.

### §13.5.3 GrooveMetric Onset-Guard (§2.14)

≥90% Onsets erhalten → Score ≥ 0.85 trotz DTW-Fehlschlag.

### §13.5.4 Cross-Phase Naturalness Consensus (§13.10 ✅ Implementiert — CrossPhaseCoordinator, Steering-Default seit 10.0.2)

Phasen im gleichen Frequenzbereich addieren ihre Effekte unabhängig.
→ **Naturalness-Guard** prüft kumulative Wirkung und reduziert bei Bedarf.
→ Musical-Noise-, Metallic-Ringing- und Roughness-Regression-Detektion.

## §13.6 Loudness für analoges Vokalmaterial

Phase 40: ±8 dB Cap, uniformer Gain, keine Gate-Sprünge.

## §13.7 Formant-Stabilität & Gender

- Vocal Analysis Shared Memory (§2.9): VFA → restoration_context → Phase 19 + SVM
- Contralto-Erkennung: F0 145–195 Hz + weibliche Formanten → FEMALE
- Register-adaptives De-Essing: Chest/Head → spezifische Parameter

## §13.8 Artist/Track-Fingerprint (§13.11 ✅ Implementiert — Spec 11 ROADMAP-3, BatchSessionLearner)

**Konzept:** Test-Track-Stimmenmodell persistieren, beim nächsten Song wiederverwenden.
BatchSessionLearner existiert bereits — Transfer-Learning für Künstler-Fingerprints:

- Stimm-Modell (Formanten, Vibrato-Rate, HNR, spektrale Hüllkurve)
- Track-Modell (Genre, Era, Aufnahmekette, typische Defekte)
- Wiederverwendung beschleunigt wiederholte Restaurierungen desselben Künstlers

## §13.9 Qualitäts-Schwellwerte

| Material | Min. MOS | Preservation-Trigger |
|---|---|---|
| Vinyl | 4.0 | bw_loss ≥ 0.90 |
| Kassette | 3.8 | bw_loss ≥ 0.90, SNR < 16dB |
| Tonband | 4.2 | — |
| CD/Digital | 4.5 | — |

---

> **v10.0.0:** SectionStrengthEnvelope, Preservation Mode, Fragile-Guard,
> Onset-Guard, Contralto-Erkennung, Uniformer Gain.
> **v10.0.0-Phantom:** ComfortGuard (§16.2), BreathPreserver (§16.3),
> VocalQualityGate 6-dim (§16.4), PhantomDetector Zero-Config (§16.1).
> **Implementiert:** Blind Reference-Free Quality (MERT-Proxy),
> Cross-Phase Consensus, Artist/Track-Fingerprint.
> **v10.0.0-Phantom:** ABX/MUSHRA Listener-Endpoints bereit (§16.9), Human-Panel MUSHRA-Protokoll erstellt, Studie ausstehend.
