# P1-2 A/B (KORRIGIERT) — SCNet-4-Stems vs. Demucs v4, faire Baseline

**Datum:** 2026-10-06 · **Status:** A/B neu gefahren · Verdikt revidiert · **C4/C5 weiter offen (menschlich)**
**Vorgänger (überholt):** `2026-10-04_p1_2_scnet_vs_mdx23c_ab.md`
**Auftrag:** `docs/TODOS_SOTA_ROADMAP.md` → TODO-P1-2 · `docs/reports/current/2026-10-04_offene_massnahmen.md` M2

---

## 1. Warum ein zweiter Lauf nötig war

Der Vorgänger-Report verglich SCNet gegen die Demucs-v4-Ketten-Stufe und meldete
für die Baseline `singer_identity_cosine` 0,45–0,63 und `separation_fidelity`
0,000–0,271 — **beide Gates verfehlt**, ∆SI-SDR „+10…+16 dB zugunsten SCNet".

Bei der Analyse des Artefakts `models/demucs/htdemucs_6s.onnx` (2026-10-06) zeigte
sich: Der ONNX-Export ist ein **hybrider Teilgraph** mit **zwei** Eingängen —
`input` (Wellenform) und `x` (STFT des Chunks) — und **zwei** Ausgängen
(`output` = Spektralzweig, `add_67` = Wellenformzweig). Der damalige Aufruf
verletzte den Vertrag an **vier** Stellen:

| # | Defekt | Wirkung |
| --- | --- | --- |
| D1 | `x` wurde mit **Nullen** gefüttert (als „unbenutzter State-Tensor" gedeutet) | Der Spektralzweig war **vollständig tot**: `output` lieferte exakt `0.0` |
| D2 | Nur `add_67` wurde verwendet | Die gültige Schätzung ist die **Hybrid-Summe** `add_67 + iSTFT(output)`; Stem-Summe erreichte nur 48 % der Mixture |
| D3 | Stem-Reihenfolge im **Plugin** permutiert (Index 0..3 = `drums, bass, other, vocals`, Aufrufer erwartet `[vocals, …]`) | „vocals" war tatsächlich Drums — **im Eval nicht wirksam** (dort korrekter Index 3) |
| D4 | Rate: der Plugin-Pfad resampelt auf **48 kHz**, das Modell ist ein **44,1-kHz**-Modell (`343980 = 7,8 s × 44100`) | −2,56 dB zusätzlich (isoliert gemessen) |

Gemessen an MUSDB18-HQ-Ground-Truth („Motor Tapes – Shore", 107 s, 7,8-s-Fenster):

| Aufrufvariante | Vocals-SI-SDR | Vocals-RMS (GT 0,0680) | Stem-Summe vs. Mixture |
| --- | --- | --- | --- |
| `x` = Nullen, nur `add_67` (Ist-Zustand) | **−1,80 dB** | 0,0002 | 48 % |
| `x` = STFT, `add_67 + iSTFT(output)` | **+11,59 dB** | 0,0639 | **98 %** |

**Der Vorgänger-Report war damit um rund 13 dB handicapiert**; seine Gate-Aussage
(„Baseline verfehlt beide Gates") beschrieb einen Aufrufdefekt, keine
Modelleigenschaft. Korrektur nach §G9 (copilot-instructions.md): ein kanonischer
Aufrufvertrag in `plugins/htdemucs_plugin.py`
(`htdemucs_onnx_stft_input`, `htdemucs_onnx_stems`, `resample_audio`), genutzt
von **Plugin und Eval** — eine Implementierung. Umsetzung: Commit `056f1ec1`,
Version 10.3.17.

## 2. Fairer Lauf — Aufbau

| Punkt | Umsetzung |
| --- | --- |
| Kandidat | `models/scnet_4stems/huge_scnet_4stems_v1.2.ckpt` — Apache-2.0, SHA-256 `807f470b13fe0734…`, 65.292.464 Parameter |
| Baseline | `models/demucs/htdemucs_6s.onnx` über den kanonischen Helfer (STFT-Eingang, Hybrid-Summe, korrekte Stem-Ordnung) |
| Songs | dieselben 3 MUSDB18-HQ-Test-Songs, dieselben Auto-Vocals-Fenster (165 / 169 / 107 s) — **identisch zum Vorgängerlauf** |
| Determinismus | Seed 42, `AURIK_FORCE_CPU=1` (GPU durch F7-Training belegt), beide Systeme CPU |
| Ehrlichkeit | `error_cases = 0`; quasisstille GT-Vocals würden als `null` gemeldet (§V6 (copilot-instructions.md)) |

## 3. Ergebnis-Matrix (fair)

| Song | System | SI-SDR [dB] | Singer-cos | Sep-Fidelity | Vocals-RMS | Laufzeit |
| --- | --- | --- | --- | --- | --- | --- |
| AM Contra | SCNet | **13,57** | 0,9782 | 0,9604 | 0,0806 | 65,9 s |
| AM Contra | Demucs v4 | 11,609 | 0,9478 | 0,9420 | 0,0789 | **9,5 s** |
| Al James | SCNet | **10,319** | 0,9922 | 0,9462 | 0,0874 | 68,6 s |
| Al James | Demucs v4 | 7,007 | 0,9728 | 0,9071 | 0,0784 | **9,9 s** |
| Motor Tapes | SCNet | **16,432** | 0,9875 | 0,9669 | 0,0841 | 72,1 s |
| Motor Tapes | Demucs v4 | 12,838 | 0,9735 | 0,9051 | 0,0819 | **10,1 s** |

### Vorher/Nachher der Baseline

| Song | Demucs v4 (handicapiert) | Demucs v4 (fair) | Δ |
| --- | --- | --- | --- |
| AM Contra | 1,54 dB · cos 0,6323 · fid 0,271 | **11,609 dB** · cos 0,9478 · fid 0,9420 | **+10,07 dB** |
| Al James | −7,00 dB · cos 0,5527 · fid 0,000 | **7,007 dB** · cos 0,9728 · fid 0,9071 | **+14,01 dB** |
| Motor Tapes | −20,07 dB · cos 0,4485 · fid 0,000 | **12,838 dB** · cos 0,9735 · fid 0,9051 | **+32,91 dB** |

### Gates (Never-Below, Hörordnung §62)

| Gate | Schwelle | SCNet | Demucs v4 (fair) | Demucs v4 (alt) |
| --- | --- | --- | --- | --- |
| `separation_fidelity` | ≥ 0,80 / 0,83 | 0,9604 / 0,9462 / 0,9669 ✅ | 0,9420 / 0,9071 / 0,9051 ✅ | 0,271 / 0,000 / 0,000 ❌ |
| `singer_identity_cosine` | ≥ 0,92 | 0,9782 / 0,9922 / 0,9875 ✅ | 0,9478 / 0,9728 / 0,9735 ✅ | 0,6323 / 0,5527 / 0,4485 ❌ |

## 4. Verdikt (revidiert)

1. **SCNet bleibt der bessere Kandidat** — aber der Vorsprung beträgt
   **+1,96 / +3,31 / +3,59 dB** (Mittel ≈ **+2,96 dB**), nicht „+10…+16 dB".
   Der Vorgängerwert war ein Artefakt der handicapierten Baseline.
2. **Demucs v4 erfüllt beide Gates** — es ist **kein** defekter Fallback, sondern
   nach der Aufrufkorrektur ein gültiger Separator. Die Produktions-Baseline
   gewinnt allein durch den Fix **+10 bis +33 dB**.
3. **Trade-off, der bisher nicht sichtbar war:** Demucs v4 ist auf CPU
   **~7× schneller** (9,8 s vs. 68,9 s je 30 s ⇒ 0,33× RT gegen 2,30× RT).
   SCNet kauft ~3 dB mit dem Siebenfachen an Rechenzeit.
4. Für die **Ketten-Stufe** bedeutet das: SCNet ist als Qualitätsstufe
   gerechtfertigt, der Aufruf-Fix wirkt aber **unabhängig davon** sofort.

## 5. Konsequenz für C4 / C5

- **C4** (menschliche Hörstichprobe SCNet vs. Kette) ist weiter **offen** — der
  Kandidat bleibt unverdrahtet, bis der Sign-off vorliegt. Die Hör-Artefakte
  liegen je Song bereit (`<song>__mix.wav`, `__gt_vocals.wav`,
  `__scnet_vocals.wav`, `__baseline_vocals.wav`), sodass die Stichprobe jetzt
  **durchführbar** ist.
- **C5** (VS-1/GSEP-Ziel) ist weiter als **ERSETZT durch SCNet-Rollenübernahme**
  bzw. **GESTRICHEN** einzutragen; „extern blockiert" ist kein gültiger Endzustand.
- Der Vorgänger-Report ist als **überholt** markiert; er bleibt als Beleg für den
  Defekt erhalten.

## 6. Limitationen (ehrlich)

1. **Stichprobe n = 3 Songs × 30 s** (MUSDB-Test-Trio). Keine Aussage über
   andere Genres/Ären; die Auto-Vocals-Fenster decken je Song einen
   gesangsenergiereichen Ausschnitt ab.
2. **`MUSDB18-HQ` ist 44,1 kHz** — die Rate-Korrektur D4 ist im Eval deshalb
   wirkungslos; sie wirkt im **Produktionspfad** (48 kHz), der getrennt
   vermessen wurde (−2,56 dB).
3. **CPU-Vergleich**: Die Laufzeiten sind CPU-gebunden (`AURIK_FORCE_CPU=1`);
   auf der ROCm-GPU verschiebt sich das Verhältnis (SCNet hat keinen
   paritätsverifizierten Torch-ROCm-Kern).
4. **MelBandRoformer-ONNX-Pfad** des BS-RoFormer-Plugins liefert duplizierte
   Instrument-Stems und entkoppelte Vocals — als Baseline unbrauchbar, der
   Defekt ist dokumentiert, aber **nicht** behoben.

## 7. Belege

| Artefakt | Pfad |
| --- | --- |
| Matrix (CSV) | `output/scnet_ab_2026-10-06_fair/matrix.csv` |
| Report (JSON) | `output/scnet_ab_2026-10-06_fair/report.json` |
| Lauf-Log | `output/scnet_ab_2026-10-06_fair/eval_scnet_ab.log` |
| Hör-Artefakte | `output/scnet_ab_2026-10-06_fair/<song>__{mix,gt_vocals,scnet_vocals,baseline_vocals}.wav` |
| Fix (Plugin) | Commit `056f1ec1`, Version 10.3.17, `plugins/htdemucs_plugin.py` |
| Fix (Eval) | `scripts/eval_scnet_vs_mdx23c.py` → `_demucs4_baseline` nutzt den kanonischen Helfer |
| Vorgänger (überholt) | `docs/reports/current/2026-10-04_p1_2_scnet_vs_mdx23c_ab.md` |
