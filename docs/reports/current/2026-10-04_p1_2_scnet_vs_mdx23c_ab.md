# P1-2 A/B — SCNet-4-Stems-Kandidat vs. Produktionsketten-Stand (Demucs v4)

**Datum:** 2026-10-05 (Ausführungssitzung; Nachtrag zum Auftrag M2 „Lane C")
**Status:** A/B gefahren · Verdikt liegt vor · **Integration offen** (menschliche
Hörstichprobe C4 + Sign-off)
**Auftrag:** `docs/reports/current/2026-10-04_offene_massnahmen.md` M2 (C1–C5);
Roadmap `docs/TODOS_SOTA_ROADMAP.md` → TODO-P1-2

---

## 1. Aufbau (strikt nach Vorgaben, Abweichungen belegt)

| Punkt | Umsetzung |
|---|---|
| Kandidat | `models/scnet_4stems/huge_scnet_4stems_v1.2.ckpt` (Aname-Tommy/Huge-SCNet-4stems, Apache-2.0, SHA-256 `807f470b13fe0734…`) |
| Architektur | **vendored** ZFTurbo MSST `models/scnet` (unverändert, MIT; `models/scnet_4stems/zfturbo_scnet/`) — der Autoren-Hinweis der Modell-Karte („If UVR not works, try ZFTurbo's msst") + der Key-Match zum Checkpoint bestätigen die Herkunft. Strict-Load: **0 missing / 0 unexpected**, 65.292.464 Params |
| Baseline | **Demucs v4** (`models/demucs/htdemucs_6s.onnx`, ONNX-CPU). Der Auftrag nennt „MDX23C-Stand" — MDX23C-Gewichte nach §v10.73 entfernt, „Demucs v5" existiert nie (htdemucs = v4). Die ursprünglich probierte MelBand-ONNX-Baseline wurde **verworfen** (Defekt, s. §5) |
| Songs | 3 MUSDB18-HQ-Test-Songs („AM Contra – Heart Peripheral", „Al James – Schoolboy Facination", „Motor Tapes – Shore") |
| Segmente | 30 s, Start = **vocals-energiereichstes Fenster** (auto: 165 / 169 / 107 s; §G5-deterministisch, 1-s-Raster — Intros mit quasistillem GT-Gesang sind für SI-SDR/Singer bedeutungslos, Befund GT-RMS 1,8e-5) |
| Determinismus | §G5: Seed 42; **Bit-Identität geprüft** (zwei Vorwärtsläufe, Stör-Seed dazwischen): `max|Δ| = 0.0` über alle Stems |
| CPU-Vertrag | `AURIK_FORCE_CPU=1` (GPU durch F7-Training belegt); beide Systeme ONNX-CPU/Torch-CPU; keine Produktions-Änderung, kein Flag-Flip |
| Ehrlichkeit | §V6: Fehler je Fall als `status="error"` + Exit ≠ 0; stille GT-Stems ⇒ Metrik `null` statt Rauschzahl. **0 Fehlerfälle** |

## 2. Ergebnis-Matrix (30 s, identische Fenster je System)

| Song | System | SI-SDR [dB] | Singer-cos | Sep-Fidelity | Vocals-RMS | Laufzeit |
|---|---|---|---|---|---|---|
| AM Contra | **SCNet** | **13,57** | **0,9782** | **0,9604** | 0,0806 | 129 s |
| AM Contra | Demucs v4 | 1,54 | 0,6323 | 0,271 | 0,0006 | 12 s |
| Al James | **SCNet** | **10,32** | **0,9922** | **0,9462** | 0,0874 | 80 s |
| Al James | Demucs v4 | −7,00 | 0,5527 | 0,000 | 0,0012 | 13 s |
| Motor Tapes | **SCNet** | **16,43** | **0,9875** | **0,9669** | 0,0841 | 130 s |
| Motor Tapes | Demucs v4 | −20,07 | 0,4485 | 0,000 | 0,0020 | 12 s |

Stem×Stem-SI-SDR (je Song, [drums, bass, other, vocals]):

- AM Contra — SCNet: `[10,9 · 2,9 · −9,0 · 13,6]` · Demucs: `[2,8 · 0,7 · −28,6 · 1,5]`
- Al James — SCNet: `[7,4 · 12,3 · 0,9 · 10,3]` · Demucs: `[−1,3 · 9,6 · −11,5 · −7,0]`
- Motor Tapes — SCNet: `[10,5 · 13,6 · 13,7 · 16,4]` · Demucs: `[4,6 · 4,7 · −16,6 · −20,1]`

**Gates (M2/C4):** `separation_fidelity` ≥ 0,80/0,83 → SCNet **erfüllt** (0,95–0,97);
`singer_identity_cosine` ≥ 0,92 → SCNet **erfüllt** (0,978/0,992/0,988). Baseline
verfehlt beide Gates in diesem Lauf.

## 3. Verdikt (Objektivmetriken)

Der SCNet-Kandidat **dominiert die getestete Baseline auf allen drei Songs in
allen drei Metriken** (ΔSI-SDR +9…+36 dB zugunsten SCNet). Der Kandidat liefert
plausible Stem-Level (geringe RMS-Differenz zum GT: 0,0806 vs 0,0818 usw.) und
erfüllt die C4-Schwellen der Objektivmetriken. **Empfehlung:** Kandidat in die
menschliche Hörstichprobe (C4) geben; bei GO als Kandidat für eine
Ketten-Stufe (Vocal-/4-Stem-Rolle) weiterverfolgen — **keine automatische
Integration** (C4/C5 bleiben menschliche Sign-offs).

## 4. Limitationen (ehrlich)

1. **Baseline-Vorbehalt:** Die Demucs-v4-ONNX-Ausgaben sind massiv leiser als
   das GT (Vocals-RMS ≈ −40 dB; Stem-Summe rekonstruiert die Mixtur kaum,
   Fidelity 0–0,27). SI-SDR/Singer sind skalierungsinvariant, aber der lokale
   ONNX-Export könnte eine Konvention/Projektion verlangen, die wir aus den
   verfügbaren Metadaten nicht bestätigen konnten ⇒ die Baseline ist evtl.
   **unterrepräsentiert**. Follow-up (nach F7): Baseline zusätzlich über die
   BS-RoFormer-317-Torch-Stufe (GPU) fahren.
2. **Fenster:** 30-s-Exzerpte (vocals-energiereichste Fenster), nicht ganze
   Songs; Session-Fortsetzung (Ketten-Kontext) nicht modelliert.
3. **3 Songs** (dokumentierter Stand-in des Auftrags), CPU-only (F7 belegt GPU;
   §III.9). CPU-Laufzeit SCNet ≈ 4,3× RT vs. Demucs ≈ 0,4× RT — **GPU-Beschleunigung
   des Kandidaten ist Integrationsauflage**.
4. Hörstichprobe (C4) steht aus; sie kann das objektive Verdikt überstimmen
   (Hörordnung: das Ohr entscheidet — gegen Ebene-1-Invarianten nie).

## 5. Zusatzbefund — MelBandRoformer-ONNX-Pfad DEFEKT (Bewertungs-relevant)

`plugins/bs_roformer_plugin.py` (CPU-Pfad `_separate_onnx`) liefert auf
identischem Material:

- **Vocal-Stems ohne Korrelation zum GT** (max|corr| bei ±2 s Lagsuche ≈ 0,003;
  SI-SDR −52…−71 dB) auf zwei verifizierten vocal-reichen Fenstern
  (`output/scnet_ab_smoke_20261005/`, `output/scnet_ab_smoke2_20261005/`).
- **Instrument-Stems = derselbe Residual mit festen Skalaren** (0,40/0,30/0,15/
  0,10/0,05 für drums/bass/guitar/piano/other) — keine echte Stem-Trennung
  (Code: `stems_out[...] = instruments_48 * k`); die gemeldete `SDRi=40 dB` ist
  plugin-interne Konsistenz, keine Qualitätsaussage.

Deshalb wurde der Pfad **nicht als Baseline** verwendet. Der Defekt betrifft
den CPU-/Fallback-Pfad der Separation (UVR/CLI-Pfad ohne GPU) und ist als
offener Punkt dokumentiert (Root-Cause-Suche: ONNX-Input-Konvention /
Normalisierung vs. reverse-engineerte MBR-Konstanten).

## 6. Artefakte & Reproduktion

**Artefakte:** `output/scnet_ab_2026-10-05/` — `matrix.csv`, `report.json`,
`eval_scnet_ab.log`, Hör-Artefakte je Song (`__mix` / `__gt_vocals` /
`__scnet_vocals` / `__baseline_vocals`, 32-bit-Float) für die C4-Hörstichprobe.

**Reproduktion:**

```bash
AURIK_FORCE_CPU=1 .venv_aurik/bin/python scripts/eval_scnet_vs_mdx23c.py \
  --seconds 30 --songs 3 --out output/scnet_ab_2026-10-05
# Determinismus (§G5): bit-identisch — zwei Vorwärtsläufe, Stör-Seed dazwischen, max|Δ| = 0.0
```

**Kette:** `scripts/eval_scnet_vs_mdx23c.py` (FILE_REGISTRY-Eintrag) ·
`models/scnet_4stems/zfturbo_scnet/` (vendored, MIT, README + LICENSE).
