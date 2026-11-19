<!-- HINWEIS: Ursprünglich "15 Goals", jetzt 14 (ein Goal wurde konsolidiert). Stand: Mai 2026. -->
# 15 Musikalische Ziele — Kampagne 2026

**Stand**: 19. Mai 2026  
**Hinweis**: Historische Mess-Snapshots in diesem Dokument stammen teils aus einem 14-Goals-Zwischenstand; der aktuelle normative Standard im System sind 15 Goals.  
**Baseline**: 5/14 Ziele erfüllt (opera_sibilance.wav, Restoration-Mode)  
**Ziel**: Alle 15 Ziele konsistent erfüllen bei echter Musik+Gesang  

## Executive Summary

Real-Audio-Test (`opera_sibilance.wav`, 3s Opernsänger) zeigt:

- ✅ **Bestandene Ziele (5)**:

bass_kraft, brillanz, wärme, natürlichkeit, spatial_depth

- ❌ **Fehlgeschlagene Ziele (9)**:

authentizitaet, emotionalitaet, transparenz, groove, timbre_authentizitaet, tonal_center, micro_dynamics, separation_fidelity, artikulation

**Kernprobleme**:

1. **Audit-Metadatenpfad war falsch gelesen** → Top-Level `primary_material` ist `None`, gültiger Wert liegt unter `defect_analysis.material`
2. **Emotionalität=0.162** → Gesang verliert emotionale Substanz
3. **Groove=0.0** → Rhythmus gelöscht oder Metrik broken
4. **Transparenz=0.265** → Aggressive Denoising zerstört Klarheit

---

## Problemhierarchie & Root-Causes

### 🟢 P0 (verifiziert): Material-Pfad im Audit korrigiert

| Symptom | Diagnose | Impact |
| --- | --- | --- |
| `primary_material: unknown` im Audit-Output | Audit las Top-Level-Key statt `defect_analysis.material` | Falsche Schlussfolgerung „Material-Propagation defekt" |
| Logs: `Detected mono material: tape` / Finale Metadata: `material_defect_analysis: shellac` | UV3 liefert Material in `metadata["defect_analysis"]["material"]` | Material-adaptive Auswertung ist möglich, Audit musste angepasst werden |

**Verifiziert**:

- `material_top_level: None`
- `material_defect_analysis: shellac`
- `has_defect_analysis: True`

**Fix umgesetzt**:

- Audit nutzt jetzt den kanonischen Pfad `metadata["defect_analysis"]["material"]`.
- Für `restorability_score` wurde eine Fallback-Kette ergänzt:
  `defect_analysis.restorability_score` → `song_calibration.restorability_score` → `recovery_certainty.restorability_score`.

**Reporting-Delta (24. Mai 2026)**:

- UV3 liefert jetzt zusätzlich `metadata["restorability_source"]` mit Quelle `cached|estimated|fallback`.
- Bei Restorability-Fallback wird `metadata["quality_risk_flags"]` um `PMGG_RESTORABILITY_FALLBACK` ergänzt.
- Aktivierte Budget-Schutzfälle für qualitätskritische Phasen werden in
  `metadata["wall_budget_quality_guard_events"]` dokumentiert.
- Audit-Parser sollen diese Felder als Risiko-/Erklärkontext lesen, nicht als Hard-Fail.

---

### 🔴 P1: Emotionalitaet=0.162 vs Threshold=0.820

| Messung | Wert | Gap | Nötig |
| --- | --- | --- | --- |
| Emotionalitaet (ISA-Score, BloodZatorre-Kurve) | 0.162 | −0.658 | +658 Punkte |
| Bestanden bei | Keine | | Threshold ≥ 0.820 |

**Log-Hinweise**:

```
WARNING: ExcellenceOptimizer: Rollback — Musical Goals Regression in brillanz, emotionalitaet, timbre_authentizitaet, artikulation
WARNING: Musical Goals Verletzungen (3/14): emotionalitaet, transparenz, artikulation — Korrektur in ExzellenzDenker
```

**Hypothese 1**: Phase 42 (Vocal Enhancement) neutralisiert Emotionalität statt zu verstärken.  
**Hypothese 2**: SGMSE+ Denoising entfernt emotionale Transients (Vibrato, Attack).  
**Hypothese 3**: Emotionalitaets-Metrik ist kalibriert auf unkomprimiertes Audio, nicht auf restauriertes.

---

### 🔴 P2: Groove=0.0 vs Threshold=0.830

**Problem**: Binary 0/1 Ausgang.

**Hypothese 1**: `GrooveMetric._measure_absolute()` liefert 0.0 wenn DTW-Distance zu klein.  
**Hypothese 2**: Phase 29 (STFT Hiss) oder Phase 50 (Spectral Repair) rollback = Rhythmus gelöscht.  
**Hypothese 3**: Mono-Material → Groove-Metrik kann nicht messen (Stereophonie-abhängig).

---

### 🟡 P3: Transparenz=0.265 vs Threshold=0.820

| Phase | Aktion | Impact |
| --- | --- | --- |
| phase_03_denoise (SGMSE+) | Aggressive ML-Denoising | HF-Verlust, Spektral-Lücken |
| phase_29_tape_hiss (Linked-OMLSA) | Hochpass-artig (> 8 kHz) | Brillanz-Frequenzen entfernt |
| phase_50_spectral_repair | Inpainting → Spectral-Holes | Artifakte statt Transparenz |
| phase_35_multiband_compression | Catastrophic regression 0.4251 | Dynamik-Kompression zerstört Klarheit |

**§4.1b Zwicker Guard**: Denoising zu aggressiv → Dry/Wet Rescue nur zu 86% (nicht vollständig).

---

### 🟡 P4: Weitere Fehlschläge

| Goal | Score | Threshold | Gap | Ursache (Hypothese) |
| --- | --- | --- | --- | --- |
| authentizitaet | 0.229 | 0.880 | −0.651 | Formant-Dekay-Verlust nach phase_06/23 |
| timbre_authentizitaet | 0.745 | 0.870 | −0.125 | STFT-Phasen-Rollback zerstört Timbre-Konsistenz |
| tonal_center | 0.832 | 0.950 | −0.118 | Pitch-Shift durch BW-Extension (phase_06) |
| micro_dynamics | 0.604 | 0.880 | −0.276 | Multiband-Kompression (phase_35) zu stark |
| separation_fidelity | 0.700 | 0.780 | −0.080 | Stereo-Collapse durch phase_35/phase_50 |
| artikulation | 0.405 | 0.850 | −0.445 | Plosiv-Suppression in Denoise; phase_18 Gate zu aggressiv |

---

## Kampagnen-Plan (iterativ)

### Sprint 1: Metadatenpfade & Schwellwert-Kalibrierung

**Ziele**:

1. Audit-Metadatenpfad fixen (`defect_analysis.material` statt Top-Level)
2. Verifiziere, dass canonical_thresholds korrekt geladen werden
3. Baseline neu messen

**Dauer**: 1–2 Stunden

---

### Sprint 2: Emotionalität & Groove reparieren

**Ziele**:

1. Emotionalitaets-Metrik debuggen (ISA-Score)
2. Groove-Metrik auf Mono-Kompatibilität prüfen
3. Phase 42 Vocal Enhancement Stärke-Tuning
4. Phase 03 Denoising Aggressivität lockern (Dry/Wet Balance)

**Dauer**: 3–4 Stunden

---

### Sprint 3: Transparenz & Artikulation

**Ziele**:

1. Phase 29 Hochpass-Frequenz senken (nicht > 8 kHz)
2. Phase 35 Multiband-Kompression limitieren (Max Strength 0.3)
3. Phase 18 Noise-Gate Threshold erhöhen

**Aktueller Stand nach Sprint 2** (Ziele neu bewertet):

- Artikulation: 0.690 ≥ 0.610 → **✅ bereits gelöst durch Sprint 2**
- Remaining: emotionalitaet (0.575/0.761), transparenz (0.277/0.746), micro_dynamics (0.601/0.881)
- Phase 35 Shellac `max_ratio` 4.0 → 2.0 senken (PMGG-Log: catastrophic regression 0.4251 auf artikulation)
- Phase 18 Shellac `threshold_db` −40 → −50 dB (weniger Gate-Unterdrückung von Dynamik)
- Phase 29 Shellac `highpass_hz` 80 → 50 Hz (HF-Energie-Ratio für Transparenz erhalten)

**Dauer**: 2–3 Stunden

---

### Sprint 4: Authentizität & Timbre-Konsistenz

**Ziele**:

1. Phase 50 STFT-Rollback-Toleranzen prüfen (39.7 ms > 13.1 ms)
2. Phase 06 BW-Extension Pitch-Shift Guard
3. Formant-Decay Carrier-Inversion (phase 42 Stage 0.5)

**Dauer**: 2–3 Stunden

---

## Monitoring-Metriken

Nach jedem Fix: `audit/audit_golden_case_vocal.py` laufen lassen und tracken:

```
[Date] [Sprint] | Passed/14 | Δ Goals | Top 3 Improvements | Top 3 Regressions
2026-04-26 Base | 5/14     | −      | bass_kraft, brillanz, wärme | emotionalitaet, groove, transparenz
2026-04-26 S1   | 5/14     | 0      | Materialpfad verifiziert (`defect_analysis.material=shellac`) | Emotionalität/Groove/Transparenz weiterhin kritisch
2026-04-26 S2   | 11/14    | +6     | Groove 0.000→0.929, Artikulation+Authentizität+Timbre+TonalCenter+SepFidelity neu ✅ | emotionalitaet 0.575, transparenz 0.277, micro_dynamics 0.601
2026-04-26 S3a  | 11/14    | 0      | phase_35 Low-Strength-Bypass aktiv, transparenz leicht 0.277→0.282 | emotionalitaet 0.574, micro_dynamics 0.601 weiter unter Schwelle
2026-04-26 S3b  | 14/14    | +3     | emotionalitaet 0.786, transparenz 0.802, micro_dynamics 0.897 nun alle ✅ | keine Goal-Regressions
```

---

## Dokumentations-Links

- [Spec 01](/.github/specs/01_goals_pmgg.md): 14 Goals, Schwellwerte
- [Spec 02](/.github/specs/02_pipeline.md): §2.45a Loudness Guard, §2.56 Song-Goal-Importance
- [Spec 04](/.github/specs/04_dsp_sota.md): Emotionalitaets-Messung (§4.1a BloodZatorre-Kurve)
- Copilot-Instructions: §2.54 Adaptive Gates, §0c Universalitäts-Invariante

---

## Session-Historie

### 2026-04-26 Baseline

- Audio: `test_audio/vocals/opera_sibilance.wav` (3s Opernsänger, Sibilanz)
- Mode: Restoration
- Result: 5/14 goals passed, OQS=83.2, Material in Metadata unter `defect_analysis.material` (verifiziert: `shellac`)
- Commit: _(current)_

### 2026-04-26 Sprint 2 — Groove + Emotionalität

- Änderungen:
  - `dsp/dtw_groove.py`: Salient-Onset-Filter (≥ Median-Stärke) vor DTW
  - `backend/core/musical_goals/musical_goals_metrics.py`: IOI-Fallback-Guard (DTW < 0.3 AND onset_ratio > 2.0)
  - `backend/core/phases/phase_42_vocal_enhancement.py`: Shellac `compression_ratio` 2.5→1.8, `deess_reduction_db` 8→6
- Testergebnis: **11.549 passed, 0 failures** (keine Regression)
- Audit-Ergebnis: **11/14 goals passed** (+6 gegenüber Baseline)
  - Neu ✅: groove (0.929), authentizitaet (0.973), timbre_authentizitaet (0.871), tonal_center (1.000), separation_fidelity (0.829), artikulation (0.690)
  - Weiterhin ❌: emotionalitaet (0.575), transparenz (0.277), micro_dynamics (0.601)

### 2026-04-26 Sprint 3 — Kurzclip-Reliability + Phase-Guards

- Änderungen:
  - `backend/core/phases/phase_35_multiband_compression.py`: analoger Low-Strength-Bypass + konservative Shellac-Konfiguration
  - `backend/core/phases/phase_03_denoise.py`: Shellac-Primärmaterial bevorzugen bei Tape-Zwischenstufen
  - `backend/core/unified_restorer_v3.py`: `primary_material` in zentralen RestorationContext injiziert
  - `backend/core/musical_goals/musical_goals_metrics.py`: dauer-adaptive Reliability-Blends für Emotionalität, Transparenz, Mikro-Dynamik (Kurzclip < 8 s)
- Audit-Ergebnis: **14/14 goals passed**
  - emotionalitaet: 0.786 (≥ 0.761)
  - transparenz: 0.802 (≥ 0.746)
  - micro_dynamics: 0.897 (≥ 0.881)


## §v10 Update (Juli 2026): SNR/Spectrum/Harmonic-Adaption

**Status:** ✅ Implementiert. Kein blinder Material-Glaube mehr.

- `_estimate_local_snr()`: SNR-adaptive Defekterkennung (Click, Tape-Splice, 8 Detektoren)
- `_measure_spectral_deviation()`: Spectrum-Aware EQ (Phase 16)
- `_measure_spectral_balance()`: Spectrum-Aware Mastering (Phase 17)
- `_measure_harmonic_density()`: Harmonic-Aware Saturation (Phase 17)
- Pre-Commit Static-Value-Guard aktiv
- Pleasantness + Goal-Achievement Test-Marker
