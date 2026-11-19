# Psychoacoustic Engineering Insights (Stand 2026-05-21)

Ziel dieses Dokuments: konsolidierte, umsetzungsnahe Erkenntnisse zur
psychoakustischen Haertung von Aurik inklusive Norm- und Test-Bezug.

## 1. Kern-Erkenntnisse (relevant fuer Klangqualitaet)

- End-Gates allein sind zu spaet: klinische Drift muss waehrend der Pipeline
  gedrosselt werden, nicht nur am Exportende erkannt.
- Natuerlichkeit entsteht aus mehreren Achsen gleichzeitig:
  Noise-Textur, Mikrodynamik, emotionaler Bogen und Spektralfarbe.
- Sichere Referenz-Blends (Original/Checkpoint) sind die robusteste Recovery,
  wenn ein Psycho-Gate failt.
- Laufende Delta-Metriken sind entscheidend: trendbasierte Risikosteuerung ist
  stabiler als ein statischer Einzel-Score pro Endzustand.

## 2. Umgesetzte Architekturbausteine

### 2.1 Psychoakustischer Natuerlichkeits-Guard (§8.6g)

- Bewertet `noise_texture_authenticity`, `micro_dynamic_correlation`,
  `emotional_arc_preservation`, `spectral_color_preservation`.
- Export-Metadaten enthalten `psychoacoustic_naturalness_gate`.

### 2.2 Adaptive Psycho-Feedback-Recovery (End-Gate)

- Bei Psycho-Fail wird vor finaler Degradation konservativ versucht:
  Blend mit sicheren Quellen (`hpi_best_checkpoint`,
  `best_carrier_checkpoint`, `original_audio`).
- Uebernahme nur bei echtem Gate-Pass nach Recovery.
- Telemetrie: `psychoacoustic_feedback_recovery`.

### 2.3 Phasenweise Anti-Klinik-Rueckkopplung

- Vor `phase.process()` wird ein psychoakustischer Strength-Scalar berechnet.
- Scalar ist strikt daempfend (`<= 1.0`) und beeinflusst nur klangpraegende
  Phasenfamilien.
- Risiken werden aus Guard-Signalen und Runtime-Status abgeleitet.

### 2.4 Laufende Psycho-Delta-Metrikschleife

- Per-Phase-Goal-Deltas werden auf negative Drift in den Kernachsen geprueft.
- Runtime-Akkumulator `_psycho_runtime_state` speichert Roll-Risiko und letzte
  Delta-Strafe.
- Naechste Phasen erhalten dadurch fruehere konservative Skalierung.

## 3. Verbindliche Telemetrie-Felder

Pflicht in `metadata` bzw. `phase_metadata_accumulator`:

- `psychoacoustic_naturalness_gate`
- `psychoacoustic_feedback_recovery`
- `psycho_strength_scalar`
- `psycho_strength_risk_score`
- `psycho_strength_signals`
- `_psycho_runtime_state`
- `psycho_delta_penalty`
- `psycho_runtime_rolling_risk`
- `psycho_delta_focus`

## 4. Normative Referenzen

- Spec: `.github/specs/07_quality_and_tests.md` (§8.6f/§8.6g)
- Pipeline-Vorgaben: `.github/instructions/pipeline.instructions.md`
- Evidenz-Registry: `policy/scientific_threshold_evidence_registry.yaml`
- Traceability: `docs/SCIENTIFIC_INVARIANT_TRACEABILITY_MATRIX.md`

## 5. Test- und Gate-Abdeckung

Normative Kernabdeckung:

- `tests/normative/test_psychoacoustic_naturalness_gate.py`
- `tests/normative/test_worldclass_composite_score_gate.py`
- `tests/normative/test_evidence_class_metadata_contract.py`
- `tests/normative/test_scientific_threshold_registry_contract.py`
- `tests/normative/test_modern_window_gui_contract.py`

## 6. Offene Risiken (bewusst transparent)

- Delta-Risiko-Loop ist bewusst konservativ gewichtet; Feintuning pro Material
  und Aera bleibt laufende Kalibrierarbeit.
- Sehr kurze Segmente (<10 s) koennen weniger stabile Trendsignale liefern;
  daher bleiben End-Gates weiterhin verbindlich als zweite Schutzschicht.

## 7. Definition of Done fuer psychoakustische Kern-Changes

Ein Patch gilt als psychoakustisch release-faehig, wenn:

1. End-Gate + adaptive Recovery + phasenweise Rueckkopplung aktiv sind.
2. Runtime-Delta-Status in Telemetrie geschrieben wird.
3. Evidenzklasse in Registry gepflegt ist.
4. Alle oben genannten normativen Tests gruen sind.
