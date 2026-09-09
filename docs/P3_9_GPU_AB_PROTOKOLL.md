# P3-9: GPU-A/B-Protokoll + Hörstudien-Vorbereitung

> **Datum:** 2026-09-09 · **Status:** Protokoll (Durchführung braucht Nutzer-Audio + Hör-Instanz)

## 1. GPU-A/B-Protokoll (deterministisch, §G5-konform)

**Ziel:** Nachweis, dass der GPU-Pfad (ROCm 7.2.4) dem CPU-Pfad qualitativ
gleichwertig ist — Metriken als Zeugen, die Hör-Instanz entscheidet (Hörordnung §8a).

1. **CPU-Referenz:** Lauf mit `AURIK_FORCE_CPU=1` (224-s-Elke-Best) → Bit-Referenz
   + alle Song-Metriken (PQS-MOS, Goal-Vektor, n_audible, LUFS-I).
2. **GPU-Lauf:** gleicher Input ohne FORCE_CPU → gleiche Metriken.
3. **A/B-Regeln:**
   - Differenzen in PQS-MOS/Goal-Scores ≥ JND (6 dB Maskierungs-JND, §P1-3) → untersuchen.
   - Bit-Vergleich ist NICHT das Kriterium (GPU-fp32 ≠ bit-identisch, Spec §v10.40c).
   - n_audible (hörbare Restdefekte) muss in beiden Pfaden → 0 streben.
4. **Blind-Hörtest:** ABX-Format, 8× (CPU|GPU|Original), Reihenfolge gewürfelt,
   Seed dokumentiert; Ergebnis in den PR-Evidenzblock.

## 2. Hörstudien-Vorbereitung (GO/NO-GO, beraten docs/guides/GO_NO_GO_DECISION_PROTOCOL.md)

- **Panels:** (a) Restaurations-Referenz: Elke Best 224 s, (b) Shellac/78rpm-Sample,
  (c) Rausch-Sample (mp3_low 128 kbps).
- **Bewertungsskala:** 1–5 pro Hör-Invariante (Natürlichkeit, Artikulation,
  Wärme, Brillanz, Authentizität) + freie Kommentare; ≥ 2 Hörer empfohlen.
- **Gate:** GO nur wenn n_audible→0 UND keine Ebene-1-Hörinvariante regressiert.

## 3. Offene Abhängigkeiten

- P1-3-Referenzlauf (AURIK_FORCE_CPU=1) — Voraussetzung für 1.1.
- Nutzer-Audio für die drei Panels.
