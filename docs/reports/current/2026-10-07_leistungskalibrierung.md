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


## Nachher-Messung (2026-10-07, nach Teilbehebung D-K3-39)

Gleiches Kommando, gleiche Quelle, gleiche Zelle — nur mit der skalaren RTS-Rekursion
(`backend/core/dsp/warp_kalman.py`):

| Größe | vorher (13:33) | nachher (15:11) | Δ |
| --- | --- | --- | --- |
| Wand-Zeit | 2200 s (73.3× RT) | **1802 s (60.1× RT)** | +399 s (+18.1 %) |
| Engine-Zeit | 65.9× RT | **52.5× RT** | — |
| Verletzungen | 4 | 4 | — |
| Schwerste Einzelphase | `phase_12_wow_flutter_fix` 440 s | `phase_12_wow_flutter_fix` 250 s | — |

Verletzte Per-Operation-Budgets nachher: `phase_pipeline_total` 1334,9 s/min (Limit 240 s/min, 5,6× über); `defect_scanner` 140,4 s/min (Limit 4 s/min, 35,1× über); `feedback_chain` 165,0 s/min (Limit 120 s/min, 1,4× über); `restorability_estimator` 13,1 s/min (Limit 5 s/min, 2,6× über).

### Phasenebene (isolierter Nachweis)

| Größe | alt (Original-Fassung) | neu (skalar) |
| --- | --- | --- |
| Phase-12-Laufzeit | 193,1 s (zuletzt, warm) | **76,1 s** = 2,54× |
| Audio-Differenz | — | max 1,9e-6 / rms 8,5e-8 = **−111,6 dBFS** |
| Metadaten | — | 19 Schlüssel, nur `rms_drop_db` auf 1e-9-Ebene verschieden |

Die Audio-Differenz liegt 15 dB unter dem CD-Rauschboden (−96 dBFS) und damit
weit unter jeder Hörschwelle; die Phasen-Metriken (Cents-Spannen, Kohärenz) sind
unverändert.


## Scanner-Kostenanalyse (2026-10-07, D-K3-40)

`cProfile` eines `DefectScanner.scan` auf demselben 30-s-Material (58,7 s):

| Kostenstelle | Zeit | Kern |
| --- | --- | --- |
| `_auto_detect_material` → `_detect_stereo_material` | 22,2 s (38 %) | Feature-Extraktion über die vollen Detektoren |
| ↳ `_detect_flutter` → `_coherent_subband_fm` | 20,4 s | **424 Hilbert-Aufrufe = 15,1 s** |
| `perceptual_salience.annotate_defect_scores` | 10,0 s | 2077 Residuum-Maskierungen, 98 089 `np.median`-Aufrufe |
| Per-Kanal-Block (7 Detektoren × 2 Kanäle) | **3,06 s = 5,2 %** (direkt gemessen) | für `channel_locations` (GUI-sichtbar) |
| Stereo↔Mono-Differenz (offen) | 58,7 s vs. 28,0 s = **30,7 s unerklärt** | NICHT der Per-Kanal-Block — eigenes Kanal-Profil nötig |

**Differenz-Messung Stereo/Mono:** dasselbe Material stereo **58,7 s** vs. mono **28,0 s**.
Die frühere Zuordnung dieser Differenz zum Per-Kanal-Block ist **widerlegt** (direkte
Messung: der Block kostet 3,06 s = L 1,54 s + R 1,52 s). Wohin die **30,7 s** gehen, ist
offen und erfordert ein eigenes Kanal-Profil (Kandidaten: stereo-spezifische Detektoren
wie Crosstalk/Stereo-Imbalance, Detektoren, die Stereo intern mitteln, sowie Welch auf
beiden Kanälen). Unabhängig davon unterscheiden sich Mono und Stereo in **16 Defekttypen
um mehr als 0,05** (bis 1,0) — ein Umleiten des §SR-CG8-Scans auf das
Stereo-Post-Scan-Ergebnis wäre also eine Verhaltensänderung.

**Nutzer-sichtbar:** `channel_locations` wird von `Aurik10/ui/modern_window.py`
(Z. 892/1060/5187) gelesen und in der GUI angezeigt — ein Gate auf diesem Block würde
eine sichtbare Anzeige verschlechtern, nicht nur internes Metadata.

**Modulweiter Ergebnis-Cache:** `_scan_cache` (Zeile 80, inhaltsgehasht, größenbegrenzt)
liefert denselben Scan in **0,01 s** zurück, wenn dieselbe Instanz dasselbe Audio erneut
sieht. Der §SR-CG8-Pfad (frische Instanz, **Mono-Downmix**) trifft ihn daher nie — daher
die 78,7 s im Lauf vom 13:33 gegenüber 37,2 s für den Stereo-Post-Scan.

### Widerlegte Optimierung (Negativ-Befund, Änderung zurückgenommen)

`scipy.signal.hilbert` → numpy-`rfft`-basiertes analytisches Signal:

| Messung | Ergebnis |
| --- | --- |
| Mikro-Benchmark (isolierte Arrays, N = 4 096…480 000) | rfft **1,63–2,72× schneller** |
| In-situ-A/B im Scanner (verschränkt, Cache geleert) | rfft **1,7× langsamer**: scipy 58,63 / 59,53 s vs. rfft 99,62 / 99,13 s |

Ursache: vier große Arrays je Aufruf (spec, hil, imag, out ≈ 27 MB × 424 Aufrufe) statt
zwei, plus `scipy.fft` statt `numpy.fft`. Die In-situ-Messung ist speichergebunden; der
Mikro-Benchmark misst die falsche Größe. **Lehre für alle weiteren Optimierungen:**
A/B im selben Prozess mit geleertem Cache, niemals nur isolierte Mikro-Benchmarks.
