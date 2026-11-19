# §EVO [RELEASE_MUST]: Denker → UV3 Evolutionsstufen (§2.60–§3.0) | §v10 Pleasantness-First

Stand: 2026-07-09 | Status: ✅ Implementiert und aktiv

## Übersicht der aktiven Features

| Feature | Status | Aktivierung |
|---|---|---|
| §2.60 Fahrplan-Brücke | ✅ Code | `AURIK_EVOLUTION=1` |
| §2.61 SectionGoalAdapter | ✅ Code | via MusicalStructureAnalyzer |
| §2.62 Per-Segment-Executor | ✅ **Aktiv (Standard)** | Automatisch bei non-uniform Fahrplan |
| §2.63 Closed-Loop PID | ✅ Code | `AURIK_EVOLUTION=1` |
| §3.0 Source-Separation | ✅ Code | `AURIK_SOURCE_SEPARATION=1` |
| §CODEC Chain-Contamination | ✅ **Aktiv (Standard)** | Automatisch bei mp3_low/aac/streaming |
| §CODEC CausalReasoner | ✅ **Aktiv (Standard)** | Automatisch via codec_contamination |
| §CODEC Phase 03 Guard | ✅ **Aktiv (Standard)** | transfer_chain + panns_singing |
| §MP3 Click-Cap | ✅ **Aktiv (Standard)** | >5000 clicks + Codec → 3000 cap |
| §2.70 Joint-Calibrator | ✅ Code | Goal-gap-driven, keine hartcodierten Regeln |

## §CODEC: Vollständige Codec-Awareness-Kette

Problem: MP3/AAC-Kompressionsartefakte werden als analoge Defekte fehlklassifiziert.
→ BS-RoFormer + MIIPHER laufen auf sauberen Vocals → Verzerrung/Kratzen.
→ 15.031 Clicks auf 225s MP3 → ReparaturDenker zerstört Vocal-Transienten.

### Datenfluss

```
MediumDetector → transfer_chain = ['vinyl', 'cassette', 'mp3_low']
     │
     ▼
DefectScanner → _codec_disc = make_discriminator(chain)
     │  ├─ _detect_crackle: onset-Korrelation → codec discount
     │  ├─ _detect_clicks: 26ms-Gitter → severity ×0.45 + Cap 3000
     │  └─ _apply_chain_contamination_discount: 8 Typen ×0.45
     │
     ▼
CausalDefectReasoner → codec_contamination → analoge Priors ×0.45
     │
     ▼
PhaseInteractionDenker → terminal_codec → audio_ctx
     │
     ▼
PhaseEffectCatalog Rule 13 → codec-aware per-phase calibration
     │
     ▼
Phase 03 → kwargs['transfer_chain'] → mp3_low + panns>0.25 → use_lightweight=True
     │  → Kein BS-RoFormer, kein MIIPHER. Nur Spectral-Gate.
     │
     ▼
ReparaturDenker → MDCT-Guard: click_iqr 5.0→8.5 + _detect_clicks Cap 3000
```

### Dateien

| Datei | Änderung |
|---|---|
| `backend/core/defect_scanner.py` | `_apply_chain_contamination_discount()`, `_codec_disc` in `scan()`, Click-Cap |
| `backend/core/causal_defect_reasoner.py` | `codec_contamination` → Bayesian-Prior-Adjustment |
| `denker/phase_interaction_denker.py` | `terminal_codec`+`codec_avg_discount` → `audio_ctx` |
| `backend/core/phase_effect_catalog.py` | Rule 13: codec-aware calibration via risks |
| `backend/core/phases/phase_03_denoise.py` | `transfer_chain`-Guard: codec+voice → lightweight |
| `backend/core/dsp/codec_discriminator.py` | 7 Diskriminator-Methoden (NEU) |
| `backend/core/joint_calibrator.py` | Goal-Gap→Utility→Strength (NEU) |
| `plugins/flashsr_plugin.py` | ROCm-Fix v2: first_stage_model.cpu() |
| `denker/reparatur_denker.py` | Bestehender MDCT-Guard (Brandenburg 1999) |

### Regeln

1. **Keine hartcodierten Phasen-Namen**: Alle Entscheidungen aus Goal-Gaps + PhaseEffectCatalog ableitbar
2. **Eine Quelle**: `transfer_chain` fließt von MediumDetector → RestorationContext → Phase-Kwargs
3. **Opt-in für experimentelle Features**: Fahrplan-Kalibrierung + PID hinter `AURIK_EVOLUTION=1`
4. **Schützend, nicht amputierend**: Keine Pauschal-Suppression — Denker dämpft, würzt, erhält

---

## Evidenz-Referenz

Siehe `docs/reports/spec_evidence/2026-07-11_spec_evolution_v10_chain_phases_linter.md`
(Maintainer Sign-off: §2.46a Chain-Metadaten, Phasen 59–66 Export, VERBOTEN-Linter v4).
