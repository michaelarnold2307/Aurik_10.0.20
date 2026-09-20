# Boundary-Maschinerie — Messstand & Design-Session-Entscheidungen

> **Stand:** 2026-09-20 · Quelle: Elke-Supervised-Lauf r10 (225,3 s, 8×30-s-Chunks),
> Mikro-Benchmarks auf 60-s-Produktionsgröße, cProfile-Zerlegung.
> Commits dieser Session: `f3e79dcf` (§PERF-R15 PGHI-Numba), `65a9ecec`
> (§PERF-R16 AURIK_CHUNK_S), `55f48dcc` (§PERF-R17 PGHI-Push-Dedup).

## 1. Ausgangslage (vor §PERF-R15/17)

Je Chunk-Boundary (Elke r10, Log-Zerlegung): ~78 s phase_28-Oszillation,
~89–106 s ExcellenceOptimizer, ~168 s Post-Chunk-Qualitätsblock,
~25 s Modell-Reloads, PGHI 121,6 s — zusammen ~552 s Fenster je Grenze,
8 Chunks ⇒ ~70 min Boundary-Kosten je Song.

## 2. Was diese Session bereits geliefert hat (alles bit-identisch)

| Baustein | vorher | nachher | Beleg |
|---|---|---|---|
| PGHI-Heap (dsp/pghi.py) | 121,6 s/Pass (heapq) | Kern 6,9 s → **2,6 s** (Push-Dedup) | 12 Fälle `np.array_equal`, 58 Tests |
| `_enhance_spectral_continuity` (60 s) | 7,67 s | **4,97 s** | Mikro-Benchmark |
| `_reinforce_harmonics` (60 s) | 8,86 s | **4,61 s** | Mikro-Benchmark |
| `optimize_for_excellence` (60 s) | 24,02 s | **20,86 s** (kalt) | Mikro-Benchmark |
| Chunk-Größe | hardcodiert 30/60 s | `AURIK_CHUNK_S` [10–600 s], Default unverändert | 6 Tests |

Evidenz-Lauf (Elke, Default-Chunking) läuft: Bit-Identität vs. r10-WAV +
Neumessung der Boundary-Kosten nach den Fixes (Ergebnis folgt).

## 3. Rest-Kosten im Optimize-Pass (60 s, nach R15+R17)

- `_enhance_spectral_continuity`: 4,97 s (STFT ~0,5 + Flux/Smooth ~0,5 +
  PGHI ~2,6 + ISTFT ~0,6 + Overhead)
- `_inject_micro_dynamics`: 0,23 s
- `_reinforce_harmonics`: 4,61 s (STFT ~0,5 + Frame-Loop ~0,2 + PGHI ~2,6 +
  ISTFT ~0,6 + Overhead)
- `_ola_crossfade_edges`: 0,03 s
- `measure_all` (2×: pre/post): 5,22 s kalt + 1,70 s warm
- Core-Guard-Zusatzprüfungen: ~4 s

## 4. Offene Hebel — alle bedürfen der Design-Session (nicht bit-identisch
oder Guard-Struktur betroffen)

1. **m1b-Retry/FeedbackChain** (im Evidenz-Lauf sichtbar: `retries=1,
   t=44.31s`): Die Retry-Schleife ist Guard-Struktur — ein Early-Out muss
   hörordnungs-konform entscheiden (Konfliktregel: Metriken sind Zeugen,
   Hör-Instanz entscheidet, nie gegen Ebene 1).
2. **Chunk-Vergrößerung 30→60/120 s**: Enabler liegt (`AURIK_CHUNK_S`).
   Struktur-Befund: per-Chunk-Fixkosten dominieren (LGE-Saliency ~212 s,
   carrier_chain ~48 s, FCPE-Reload ~38 s, Post-Chunk ~168 s).
   Erwartung: 120 s ⇒ 8→2 Chunks, weniger Crossfade-Seams = Qualitäts-Plus.
   ABER: Ausgabe ändert sich (neue Grenz-Positionen) ⇒ Supervised-Validierung
   Pflicht (Export-Quality-Gates + Hör-Check gegen Referenz).
3. **measure_all 24× im End-Gate-Loop** (t7): 8-Einträge-Content-Hash-Cache
   existiert bereits; die 24 Aufrufe messen verschiedene Varianten. Kopplung
   an t6: welche Messungen entfallen dürfen, entscheidet die Hörordnung.
4. **Core-Guard-Rollback-Verhalten** (beobachtet auf synthetischem Signal:
   `tonal_center 1.000→0.956, timbre_authentizitaet 0.896→0.859` → Rollback):
   korrektes Verhalten bei echten Regressionen, verhindert aber naive
   Early-Outs im Optimize-Pass.

## 5. Entscheidungsbedarf (Design-Session)

1. Chunk-Vergrößerung freigeben? (Evidenz-Lauf `AURIK_CHUNK_S=120` starten,
   Gate-Kriterien definieren.)
2. m1b-Retry: Welche Abbruch-Bedingung ist hörordnungs-konform?
3. measure_all im End-Gate-Loop: Welche der 24 Messungen sind redundant?
4. Priorisierung danach: P1-GPU-Ports (t9) vs. Rest-Boundary (t6/t7).

## 6. Anhang: P1-Voranalyse (Per-Phase-Dauern, r10-Log, alte Timing-Basis)

Akkumulierte Phase-Dauern über den ganzen Elke-Lauf (225 s, 8 Chunks) —
Grundlage für die GPU-Port-Priorisierung; wird nach dem Evidenz-Lauf
mit frischen Zahlen aktualisiert:

| Phase | r10 gesamt | Anmerkung |
|---|---|---|
| phase_31 speed/pitch | 309 s | §PERF-R14 (FCPE/pyin_compat) bereits gefixt |
| phase_12 wow/flutter | 228 s | FCPE-basiert → R14-Effekt im Evidenz-Lauf sichtbar |
| phase_13 stereo enhancement | 171 s | DSP |
| phase_54 transparent dynamics | 163 s | DSP |
| phase_27 click/pop | 155 s | DSP |
| phase_01 click removal | 141 s | DSP |
| phase_48 stereo width | 132 s | DSP |
| phase_18 noise gate | 109 s | DSP |
| phase_09 crackle | 88 s | DSP |
| phase_23 spectral repair | 82 s | STFT/ML-Kandidat für Torch-ROCm |
| phase_28 surface noise | 80 s | IMCRA/OMLSA (DSP, NumPy) |
| phase_50 spectral repair | 76 s | STFT/ML-Kandidat |

Kandidaten-Kriterium P1: ML-/STFT-dominiert UND ONNX-CPU-Fallback aktiv
(Paritäts-Gate rel ≤ 1e-3 gegen ONNX-CPU auf strukturierten Feeds,
Torch-ROCm-Kerne in `backend/core/dsp/*_torch_rocm.py`, Einhängung via
`gpu_model_registry`). DSP-Phasen (13/54/27/01/48/18/09/28) sind eher
Numba-Kandidaten als GPU-Kandidaten.
