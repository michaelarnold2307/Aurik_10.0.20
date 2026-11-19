# Maßnahmen 1-7 (Operationalisierung)

Stand: 20.03.2026

## 1. RELEASE_MUST-Abdeckung

- Script: scripts/release_must_coverage_check.py
- Ergebnis: reports/release_must_coverage.json
- Zweck: Traceability von RELEASE_MUST-Anforderungen zu Normtests

## 2. Drift-Detektor

- Script: scripts/spec_drift_check.py
- Ergebnis: reports/spec_drift_report.json
- Baseline: reports/spec_drift_baseline.json
- Zweck: Änderungen an Spezifikation/Gates sichtbar machen

## 3. Qualitäts-Gates Exportpfad

- Modul: backend/core/export_workflow.py
- Verhalten: Export ist immer best-effort (nicht blockierend)
- Dokumentation: quality_gate_passed / quality_gate_fail_reason im Sidecar
- Test: tests/normative/test_export_quality_gate.py

## 4. AMRB + Nightly

- Workflow: .github/workflows/nightly-quality.yml
- Enthält: Coverage-Check, Drift-Check, Normtests, AMRB-Gate

## 5. 48kHz-Invariante

- Test: tests/normative/test_sample_rate_48k_gate.py
- Fokus: Kritische DSP-Test-Fixtures müssen 48 kHz als Default verwenden

## 6. Fault-Injection / Fallback

- Modul: backend/core/fallback_guard.py
- Test: tests/normative/test_fault_injection_fallbacks.py
- Release-Mode-Validierung: primary | fallback | blocked

## 7. Hörtest-Pipeline

- Script: scripts/generate_mushra_pack.py
- Ergebnis: CSV-Paket für subjektive Blind-Hörtests
- Hinweis: Für belastbare Ergebnisse min. 10 Hörer und randomisierte Reihenfolge
