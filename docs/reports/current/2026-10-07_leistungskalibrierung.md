# Leistungs-Kalibrierung 2026-10-07 — UV3-End-to-End, Zelle `balanced`

Beleg-Dokument zur Ist-Kalibrierung der Performance-Budget-Wahrheit
(`.github/copilot-instructions.md`, Abschnitt „Budget-Wahrheit“) und zu den
Register-Einträgen D-K3-36/D-K3-37/D-K3-38 in `.github/SOTA_DEFICIT_REGISTER.md`.

## Messung

- **Quelle:** `test_audio/Elke Best - Hey Kleiner, mit Dir spielt wohl keiner (1977) HD audio 320 kbps german (original).wav`
  (176,4 s, 48 kHz, Stereo, PCM_16) — davon **30 s** verarbeitet.
- **Kommando:**

  ```bash
  .venv_aurik/bin/python scripts/benchmark_effizienz_matrix.py \
    --input "test_audio/Elke Best - Hey Kleiner, mit Dir spielt wohl keiner (1977) HD audio 320 kbps german (original).wav" \
    --seconds 30 --cells balanced --no-wav --enforce-budget --profile-top-phases 5
  ```

- **Ergebnis-JSON (Arbeitskopie, per `.gitignore` nicht versioniert — `reports/*`):**
  `output_audio/benchmark_effizienz/results_20261007_133349.json`
  (das vorliegende Dokument ist der versionierte Beleg; das JSON ist mit dem
  Kommando oben vollständig reproduzierbar)
- **Lauf:** 2026-10-07, 13:33:49 – 14:10:45 (MATRIX-RC=0)

## Zahlen

| Größe | Wert |
| --- | --- |
| Wand-Zeit | 2200,4 s für 30 s Audio = **73,3× RT** |
| Engine-Zeit | 1976,1 s = **65,9× RT** |
| Phasen | 39 ausgeführt, 6 übersprungen |
| Quality-Estimate | 0,705 |
| PQS-MOS | 2,42 |
| Master-Seed | (siehe JSON, `master_seed`) |

## Verletzungen der Per-Operation-Budget-Tabelle

| Operation | Limit | gemessen | Faktor |
| --- | --- | --- | --- |
| `defect_scanner` | 4 s/min | 157,2 s/min | **39,3×** |
| `phase_pipeline_total` | 240 s/min | 1752,8 s/min | **7,3×** |
| `restorability_estimator` | 5 s/min | 13,7 s/min | **2,7×** |
| `feedback_chain` | 120 s/min | 167,4 s/min | **1,4×** |
| `excellence_optimizer` | 60 s/min | (unter Limit) | — |

`pipeline_budget_timings` der Pipeline: `defect_scanner_s` 78,6 s ·
`phase_pipeline_s` 876,4 s · `feedback_chain_s` 83,7 s ·
`excellence_optimizer_s` 21,2 s · `restorability_estimator_s` 6,9 s.
Nicht verfügbare Größen (Export außerhalb des Restorers) werden als `null`
geführt, nicht geschätzt (Warnung: `Grenze-Pruefung 'Ausgabe_flac' nicht
verfügbar`).

## Schwerste Einzelphasen (Wall-Zeit, `top_phases`)

| Phase / Label | Wall-Zeit |
| --- | --- |
| `phase_12_wow_flutter_fix` | 440,0 s (20 % der Wand-Zeit) |
| „Pipeline started…“ (Fortschrittslabel) | 249,3 s |
| „Qualitätsprüfung…“ (Fortschrittslabel) | 197,1 s |
| „Musical Goals geprüft…“ (Fortschrittslabel) | 144,6 s |

`phase_12` lief auf **Vinyl**-Material, dessen Wow-Anteil die Phase selbst mit
0,00 ausweist; sie reparierte 28/32 Transport-Bumps und 45 Pegel-Einbrüche.

## Guard-Status (Befund D-K3-37)

Das Log dokumentiert in Zeile 3:

```text
PerformanceGuard initialisiert: Betriebsart=balanced, Target=32.0× RT, Enforce=False, Adaptive=False
```

`RestorationConfig.enforce_3x_rt` steht per Default auf `False`
(`backend/core/unified_restorer_v3.py:113`, „opt-in only“); die einzige
`True`-Fundstelle liegt in einem `__main__`-Demo (Z. 47550). Die Messung
überschreitet das 32×-Ziel um **2,3×** und wurde ohne Abbruch als Erfolg
gewertet; verletzte Per-Operation-Budgets erscheinen ausschließlich als
`budget_violations` im JSON.

## Einordnung (ehrlich)

1. Die frühere normative Angabe „53× RT (Matrix-Endlauf 2026-09-07/08)“ hatte im
   Workspace keinen Beleg und ist durch diese Messung ersetzt — sie ist **nicht**
   besser, sondern schlechter ausgefallen.
2. Nach **TODO-P0-1** (song-globale Analytik/End-Gate nach der Chunk-Assembly)
   wurde der Lauf **nicht** schneller: die Zeit liegt in Defekt-Scanner,
   Phase-Pipeline und Einzelphasen, nicht in der Analytik. Die Verdrahtung von
   P0-1 ist seit 2026-10-07 per Test gepinnt
   (`tests/unit/test_p0_1_end_gate_chunking.py::test_restore_chunked_wires_the_deferral_flags`,
   `::test_song_level_tail_runs_in_the_chunked_assembly_path`).
3. Der Harness hat **keinen Warmup**; die Wand-Zahl enthält jede einmalige
   Initialisierung (Modell-Ladevorgänge), die Engine-Zahl nicht. Beide werden
   deshalb getrennt geführt.
4. Der Lauf ist langsam **und** die Qualitätsanzeige niedrig (Quality 0,705 /
   PQS-MOS 2,42) — als Beobachtung notiert, nicht bewertet; sie gehört zur
   Qualitätsspur, nicht zur Budget-Frage.
