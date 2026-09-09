# P2-1: Monolith-Refactor-Plan — `unified_restorer_v3.py`

> **Datum:** 2026-09-09 · **Status:** Plan (keine Code-Änderung)
> **Auslöser:** 46.000+ Zeilen in einer Datei; jede Qualitäts-Änderung kostet
> Gate-Zyklen und erhöht das Risiko unbeabsichtigter Seiteneffekte (§V7).

## 1. Ist-Zustand

- `backend/core/unified_restorer_v3.py`: ~46.000 Zeilen, 30+ Zuständigkeitsbereiche
  (RestorationConfig, Kalibrierung/CalibrationContext-Bau, Phase-Auswahl,
  Chunked-Streaming-Orchestrierung, Goal-Kaskade, PQS/§2.14-Gate,
  ExzellenzDenker-Anbindung, Export).
- Befunde dieser Session (§v10.730–§v10.741) lagen alle in genau dieser Datei —
  ein Beleg für die Dringlichkeit, aber auch für das Risiko großer Schnitte.

## 2. Zielbild (Fassaden-Split, kein Big-Bang)

1. **Phase 1 — Kontext-Modul** `backend/core/restoration_context.py`:
   CalibrationContext-Bau (§v10.737-Logik), Era/Material-Chunk-Cache (§v10.738),
   `_restoration_context`-Zugriffe hinter getter/setter-API.
2. **Phase 2 — Phase-Selektion** `backend/core/phase_selection.py`:
   `_select_phases`, Gender-Block (§v10.734), Verbotene-Phasen-Filter (§0a).
3. **Phase 3 — End-Gates** `backend/core/end_gate_cascade.py`:
   Goal-Kaskade (§v10.732), PQS/§2.14 (§v10.735), EmotionalArc, letzter-Chunk-Logik.
4. **Phase 4 — Chunked-Orchestrierung** `backend/core/chunked_orchestrator.py`:
   `_restore_chunked`, Chunk-Kwargs-Cache, Assembly (bleibt in `chunked_streaming.py`).

**Invarianten:** Bit-Determinismus (§G5) über einen Referenzlauf (AURIK_FORCE_CPU=1)
vor/nach jeder Phase; jede Phase einzeln gepusht + gategrün; keine Verhaltensänderung
in einem Refactor-Commit (getrennte fix-Commits).

## 3. Ablauf & Risiken

- Reihenfolge strikt 1→4 (Kontext zuerst — alle anderen hängen daran).
- Risiko: Import-Zyklen (bridge-Verbot §V4 beachten — die neuen Module liegen
  in backend/core, GUI/CLI dürfen sie nur via bridge.py erreichen).
- Risiko: mypy/ruff-Gates auf 46k-Zeilen-Datei — Split verkleinert die Blast-Radius
  je Commit erheblich.

## 4. Offene Voraussetzungen

- Referenzlauf (P1-3) muss vor Phase 1 stehen (Bit-Baseline für jeden Split).
- Hörstudie (P3-9) parallel: subjektive Validierung, dass Splits nichts verschlechtern.
