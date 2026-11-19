# PERF-R20 (D-K3-38) — Salience-Kette: gecachte FFT-Vorbereitung + ein Permutations-Copy

**Datum:** 2026-11-07
**Bezug:** D-K3-38 (Register, `in-arbeit`), §2 Zeile 4 (Laufzeit) aus
`docs/SOTA_HANDOVER.md`; Vorgänger §PERF-R19 / §PERF-R19b (Scanner 62 → 16,0 s/min).
**Kernaussage:** Die Salience-Kette läuft **18,5 % schneller** (6,50 → 5,30 s/min
je 33-s-Fenster) — bei **bit-identischem** Ergebnis. Der End-to-End-Scan bleibt
in seiner Ausgabe **hash-gleich**; die Gesamt-Scan-Zeit sinkt nur um 2,3 %, weil
der Salience-Block ~25 % der Scanner-Zeit trägt.

## 1. Ausgangslage (gemessen, nicht geschätzt)

cProfile eines **echten** 33-s-Scans (`test_audio/Elke Best - Du wolltest nur
ein Abenteuer…mp3`, 30–63 s, 44,1 kHz, unter GPU-Ketten-Kontention, `nice 15`):

| Block | cum [s] | Anteil |
| --- | --- | --- |
| `_stft_magnitude_db` (Residuum-Kontext-STFT) | 1,52 | 16 % |
| `_to_bark_bands` (24× Band-Median) | 0,96 | 10 % |
| `compute_masking_threshold` (ERB) | 0,53 | 6 % |
| `estimate` (Broadband) | 0,91 | 10 % |
| **`annotate_defect_scores` gesamt** | **4,73** | **25 %** |

Der Salience-Block war damit der größte verbleibende Posten, den das Register
nach R19/R19b benannt hatte („größter verbleibender Block =
`perceptual_salience.annotate_defect_scores`").

## 2. Änderungen (alle bit-identisch, §G5 (copilot-instructions.md))

1. **`residuum_masking._stft_magnitude_db`** — `np.hanning(4096)` und
   `np.fft.rfftfreq(4096, 1/sr)` wurden bei **jedem** der ~2 200 Aufrufe neu
   gebaut (Fenster allein 0,14 s). Jetzt modulweite, schreibgeschützte Caches
   (`_hann_window`, `_rfftfreq`).
2. **`residuum_masking._to_bark_bands`** — statt 24 Fancy-Index-Copies pro
   Aufruf ein **einziges** Permutations-Copy (Spalten in Band-Reihenfolge,
   bin-major), danach zusammenhängende Slices. Der Median ist
   ordnungsunabhängig, die Element-Menge je Band identisch ⇒ bit-gleich.
   Zusätzlich Identitäts-Fast-Path für die Frequenzachse (spart 2 049-Float-
   `array_equal` je Aufruf).
3. **`perceptual_salience._estimate_event_spectral_content`** — Fenster,
   Frequenzachse und die 24 Bark-Bin-Indizes hängen nur von `(n_fft, sr)` ab,
   wurden aber je Event neu gebaut; jetzt `_event_fft_bins` (bounded, 16
   Einträge). Die Mittelung läuft über dieselben aufsteigenden Indizes in
   derselben Reihenfolge (pairwise-Summation unverändert) ⇒ bit-gleich.
4. **`defect_scanner._coherent_subband_fm`** — `np.angle(an)` (=
   `arctan2(Im, Re) * 1`) durch `np.arctan2(an.imag, an.real)` ersetzt: ein
   voller Ufunc-Pass + Temp-Array weniger, exakt gleiche Werte.

## 3. Belege

### 3.1 Bit-Identität der Salience-Kette
Fixer Input (6 s, 44,1 kHz, 83 Events über 4 Typen, ERB-Pfad aktiv), Ausgabe
= alle Severities, `perceptual_salience`-Metadaten und 20 Residuum-Salienzen:

```
vorher : SHA256 c3eaa3a127e13d0428647c2632716603f7061f8938ea539028f716809758cb2d
nachher: SHA256 c3eaa3a127e13d0428647c2632716603f7061f8938ea539028f716809758cb2d  (identisch)
```

### 3.2 Bit-Identität des **gesamten** Scan-Ergebnisses
Gleicher 33-s-Scan, Ausgabe-Hash über Materialtyp, alle Severities,
Confidences, Locations und Metadaten:

```
HEAD (vorher) : c599791d6acc04a1508c852fce077699e4e48c0b21a781ea665d572d514f3952
Arbeitsstand  : c599791d6acc04a1508c852fce077699e4e48c0b21a781ea665d572d514f3952  (identisch)
```

### 3.3 Messung

| Messung | vorher | nachher | Δ |
| --- | --- | --- | --- |
| Salience-Kette, 1 100 Events / 33 s (warm, Median-frei, deterministisch) | 3,57 s (6,50 s/min) | 2,91 s (5,30 s/min) | **−18,5 %** |
| Echter 33-s-Scan (s. o.), Wand | 16,57 s (30,13 s/min) | 16,19 s (29,43 s/min) | −2,3 % |
| Strategy-Golden-Gate | 3/3, 424 s (R19b) | **3/3, 402 s** | Schwellen 1,0 unverändert |
| Defect-Golden-Gate | 1/1 (R19b) | **1/1, 19,3 s** | — |

Tests: `tests/unit/test_residuum_masking.py` 16 grün (2 neue Vertragstests),
`tests/unit/test_perceptual_salience.py` 37 grün (2 neue Vertragstests:
Cache ≡ uncached bit-exakt; Determinismus §G5 (copilot-instructions.md)).

## 4. Rest-Gap (ehrlich)

Auf dem hier gemessenen Material liegt der Scanner bei **29,4 s/min** gegen das
normative Budget 4 s/min (7,3×). Die Register-Referenzmessung (16,0 s/min)
bezieht sich auf ein anderes Material mit ~1 100 statt 1 323 Events je 33 s;
auf dieser Skala entspricht R20 einem Schritt 16,0 → ~15,4 s/min (Rest 3,9×).

Verbleibende Blöcke (cProfile, gleicher Scan):

| Block | cum [s] | bit-identisch angreifbar? |
| --- | --- | --- |
| `_wf_cached_features` (Wow/Flutter-Basis) | 6,6 | teilweise (Feature-Ebene) |
| `_coherent_subband_fm` (Filterbank + Hilbert) | 5,5 | nein — `sosfiltfilt` je Band ist strukturell |
| Material-Erkennung | 4,65 | nein (läuft bereits 1×/Scan) |
| `_detect_transport_bump` | 3,58 | offen, nicht untersucht |
| Salience (nach R20) | ~3,4 | **am Boden** — Rest sind rfft + Median-Selektion |

**Der nächste große Schritt ist NICHT autonom:** Der getestete Batch-Pfad
`estimate_residuum_salience_batch` (ein Full-Song-STFT statt 2 STFTs je Event)
ist ~1,8× schneller, ändert aber die Maskierungs-Klassifikation
(mean |ΔSalienz| = 0,20; 978 → 790 „maskiert" von 1000 Events), weil die alte
Konkatenations-Semantik Junction-Frames mischt und kurze Events zero-paddet.
Diese Korrektur ist laut Code-Kommentar (2026-09-06) eine **HÖR-ENTSCHEIDUNG**
und braucht Golden-Set-Validierung + Hör-Abnahme — sie darf nur über das
§4-Verfahren (`scripts/flag_freigabe.py`) bzw. eine menschliche Hör-Session
erfolgen. **Kein Flag-Flip, keine Semantik-Änderung in diesem Schritt.**

## 5. Reproduktion

```bash
# Salience-Mikrobenchmark (deterministisch, 1 100 Events / 33 s)
PYTHONPATH=. nice -n 15 python3 -B <harness>     # Messaufbau: cProfile um annotate_defect_scores

# Bit-Identität der Salience-Kette
PYTHONPATH=. python3 -B <harness>                # SHA256 der Severity-/Salienz-Reprs

# Golden-Gates
.venv_aurik/bin/python -m pytest tests/unit/test_real_audio_strategy_golden_gate.py -q   # 3/3
.venv_aurik/bin/python -m pytest tests/unit/test_real_audio_defect_golden_gate.py -q     # 1/1

# Vertragstests der Änderung
.venv_aurik/bin/python -m pytest tests/unit/test_residuum_masking.py tests/unit/test_perceptual_salience.py -q
```
