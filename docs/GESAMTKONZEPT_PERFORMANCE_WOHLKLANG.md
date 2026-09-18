# GESAMTKONZEPT — Performance-Restaurierung ohne Wohlklang-Kompromisse

> **Stand:** 2026-09-15 · **Autor:** Aurik Testing Team · **Anlass:** Abschluss der
> SOTA-Roadmap-Umsetzungswelle P1–P5/P7 (Commit 74f4a5d5).
> **Geltung:** Dieses Dokument ist das verbindliche Gesamtkonzept dafür, wie die
> Restaurierung ALLER Importsongs zeitlich optimiert wird, ohne den Wohlklang
> für das menschliche Ohr zu kompromittieren. Einzelmaßnahmen bleiben in
> `docs/TODOS_SOTA_ROADMAP.md` und der normativen Kette (AGENTS.md §1) geführt.

---

## 1. Leitprinzipien

1. **Audibility-First (PSY-A1, Hörordnung §4).** Die Maskierungsschwelle ist das
   Reparaturziel: Was das Ohr nicht hört, wird nicht repariert — und kostet keine
   Rechenzeit. Jede reparierende Phase fragt: „Ist der Defekt über der
   Maskierungsschwelle hörbar?" Subaudible Defekte ⇒ Skip. Das senkt Rechenzeit
   UND Artefakt-Risiko **gleichzeitig** — der einzige Hebel, der nie abwägen muss.
2. **Never-worsen als Erhaltungssatz, nicht als Bremsklotz.** Gates sind
   delta-basiert: Hard-Fail nur bei Regression gegenüber dem Input, nie bei
   absoluten Produktions-Normalwerten (Guard-Kalibrierung AGENTS.md). Der
   P1-Befund (Input-af 0,7414 bei MP3-Quelle) belegt: absolute af-Schwellen
   sind auf degradierten Quellen unsinnig — nur Deltas zählen.
3. **Der Wohlklang entscheidet, Metriken sind Zeugen (Hörordnung §8a).** Keine
   Performance-Maßnahme darf eine hörbare Änderung erzwingen; Witnesses
   (Resemblyzer ≥ 0,92, MuQ-MOS, BMLD, ISO 226) dokumentieren jede Entscheidung.
4. **Determinismus (§G5 (GEBOTE.md)) und Song-Isolation (§V8) sind unantastbar.**
   Kein Zeitgewinn rechtfertigt `time.time()`-Abhängigkeiten oder Song-übergreifende
   Zustände. Parallelisierung/Scheduling läuft deterministisch (feste Seeds,
   feste Reihenfolge, gleiche Chunk-Geometrie).
5. **Ursache statt Symptom (§V7).** Stärke-Entscheidungen zentral über
   `global_scalar`; keine phasen-individuellen Schwellwerte, keine heuristischen
   „RT-Kills", die Phasen stumm abschalten.

---

## 2. Die vier Performance-Hebel

### H1 — Zeit-Ebene: nur rechnen, was der Song-Zeitpunkt verlangt

| Maßnahme | Stand | Wirkung |
|---|---|---|
| Song-globale Analytik (P0-1): GOAL_SCORECARD, End-Gate, HPI, Einladungs-Gate, Analytics, B2-Scan nur einmal je Song | ✅ umgesetzt | 8–9 End-Gate-Runden je Chunk entfallen |
| Audibility-First-Scheduling (R4): billige Detektion zuerst, teure Reparatur nur bei Hörbarkeit | ✅ Benchmark + phase_01-Zähler (`subaudible_skipped`); Ausweitung auf alle PSY-A1-Phasen = Folge-Slice | Rechenzeit sinkt dort, wo nichts zu hören ist |
| **Adaptive Phase-Rescheduling nach Hör-Impact (R7)** | 🔲 offen | Wall-Clock-Budget trifft die hörbarste Verbesserung zuerst; Impact-Schätzer speist sich aus den PSY-A1-Gates + defect_scores |
| Deferral mit Hör-Impact-Begründung (bestehendes `wall_budget_s`) | ⚙️ partiell | kein stummer Phasen-Skip; jeder Deferral loggt Warum + Hörbarkeits-Kontext |

**Ausbau R7 (nächster Schritt, CPU):** `backend/core/dsp/hearing_impact.py` —
Impact-Schätzer je Phase aus (a) PSY-A1-Zählern (hörbare Defekte), (b)
defect_scores-Schwere, (c) Never-worsen-Gate-Marge des Vorgängers. Sortierung
der Phasen nach erwartetem Hör-Gewinn je wall-clock-Budget; deterministisch
(feste Sortier-Schlüssel, keine Zufalls-Tiebreaks).

### H2 — Raum-Ebene: nur rechnen, wo der Defekt ist

| Maßnahme | Stand | Wirkung |
|---|---|---|
| R8 Sparse Repair: Defekt-Masken als Rechen-Masken (`sparse_windowed_repair`) | ✅ Infrastruktur + 17 Tests | Reparatur nur in Defekt-Nähe + Kontext, Hann-Crossfade |
| R8-Per-Phase-Rollout | 🔲 Pilot-Kandidaten aus P1-Befund: 07/17/19/38 (größte af-Abfälle) | Rechenzeit ↓ und Artefakt-Risiko ↓ gemeinsam |
| Forward-Masking-Zonen-Dämpfung (PSY-A5, ×0,6 in 01/27/64) | ✅ partiell | weichere Reparatur in Nachmaskierungs-Zonen — weniger Nacharbeit |
| Chunk-Lokalität: lokale DSP-/ML-Phasen je Chunk, song-globale Blöcke einmal (P0-1-Vertrag) | ✅ umgesetzt | 53× → Ziel 15–25× RT |

**Rollout-Reihenfolge R8:** erst die P1-Befund-Phasen mit Never-worsen-Fixes
stabilisieren (af-Delta ≥ 0), dann dieselben Phasen sparse machen — so ist jede
Sparse-Umstellung gegen den vorherigen Vollpfad messbar (af-Delta + R5-Zertifikat).

### H3 — Modell-Ebene: Modelle laden und teilen statt wiederholen

| Maßnahme | Stand | Wirkung |
|---|---|---|
| Modell-Residency & Warm-up (P1-1): LRU je Session, Warm-up einmal je Modell | 🔲 offen | ~5 min Ladezeit je Lauf entfällt |
| BSR auf GPU (Torch-ROCm-Pfad) | ✅ 41,8× | Stem-Trennung statt CPU-Minuten |
| **R3 ROCm-Ports aller ML-Modelle** (CQTdiff+, MuQ, BEATs, DeepFilterNet — Muster `bsr317_torch_rocm`) | 🔲 offen | echte GPU-Story auf AMD; ORT-ROCm-Kernel-Bug bleibt umgangen |
| Early-Exit-Pfade (phase_23 §v10.303: BW-Gewinn < 500 Hz ⇒ kein zweiter Pass) | ✅ umgesetzt (Fix 2026-09-15) | Wiederholte teure Pässe ohne Gewinn entfallen |
| Deterministisches Multi-Song-Batching desselben Modells (§G1-Seed-Isolation je Song) | 🔲 offen | GPU-Auslastung, gleiche Ergebnisse je Song |

### H4 — Mess-Ebene: nur messen, was entscheidet

| Maßnahme | Stand | Wirkung |
|---|---|---|
| GOAL_SCORECARD/End-Gate/HPI/VQI song-global statt je Chunk (P0-1) | ✅ umgesetzt | Qualität steigt, weil song-globale Größen nicht mehr auf 30-s-Ausschnitten laufen |
| JND-Gates statt Roh-dB-Toleranzen (PSY-A8 `hearing_jnd.py`, P1-3-Rollout) | ✅ Module + Guard-Rollout | weniger falsche Rollbacks ⇒ weniger teure End-Gate-Recovery-Runden |
| Analytics nur letzter Chunk (`analytics_last_chunk`) | ✅ umgesetzt | Reporting-Overhead einmal je Song |
| Budget-Wahrheit (P0-3): eine normative RT-Norm | 🔲 offen | Schein-Soll beseitigen, Deferral korrekt kalibrieren |

---

## 3. Psychoakustische Sicherungsnetze (Wohlklang-Garantie)

Jede Performance-Maßnahme muss eines oder mehrere dieser Netze **unverändert**
passieren — sonst wird sie verworfen:

1. **Ebene-1-Invarianten** (Hörordnung): Stimm-Identität ≥ 0,92 (Resemblyzer-
   Witness, P3 ✅ verfügbar), Konsonanten-Klarheit, Vibrato, Dynamikbogen, Atem.
2. **BMLD-/Equal-Loudness-Witnesses** (P2 ✅): Stereo-/EQ-Entscheidungen nach
   Hör-Freisetzung statt Mess-dB — auch wenn die Phase schneller läuft, ändert
   sich ihr Bewertungsmaßstab nicht.
3. **Never-worsen-Gates** (delta-basiert): jede Beschleunigung, die das af-Delta
   oder den Quality-Score senkt, ist per Definition falsch.
4. **Export-Vertrag §0c:** bestmögliches sicheres Ergebnis mit Status „degraded"
   statt Hardstop — Zeitdruck darf nie zu fehlender Ausgabedatei führen.
5. **Dither-Vertrag (§V5):** Beschleunigung nie über nacktes `astype`-Quantisieren.
6. **Silent-Failure-Verbot (§V6):** jeder Fallback loggt Warnung + Begründung —
   ein „schneller" stiller DSP-Fallback ist ein Wohlklang-Risiko, kein Gewinn.

---

## 4. Ziel-Budget-Modell

- **Normative Quelle:** eine einzige RT-Norm (P0-3-Nacharbeit), kalibriert nach
  P0-1. Referenzlauf 224 s ≤ 40 min als Zwischenziel; 32× RT als Endziel der
  CPU-Kette, darunter mit GPU (H3).
- **Modus-Staffel:** FAST (nur hörbare Pflichtreparaturen) → BALANCED → QUALITY →
  MAXIMUM (alle Zeugen aktiv). Die Staffel definiert nur das BUDGET, nie die
  Hörbarkeits-Schwellen — die bleiben in allen Modi identisch (Wohlklang-Primat).
- **Messgrößen je Maßnahme:** (a) RT-Faktor auf dem 224-s-Referenzlauf,
  (b) AURIK Quality Score + af-Delta (P1-Diagnose) auf „Elke Best"-Korpus,
  (c) R5-Zertifikat (bit-identisch), (d) Restorer-Suiten.

---

## 5. Umsetzungs-Roadmap (priorisiert)

**Phase 1 — CPU, sofort:**
1. Never-worsen-Fixes auf P1-Befund-Phasen 07/17/19/38 (af-Delta ≥ 0).
2. R8-Rollout in genau diesen Phasen (Vollpfad-Vergleich je Phase).
3. R7 Hearing-Impact-Rescheduling (wall_budget trifft hörbarste Verbesserung).
4. R4-Benchmark auf alle 16 PSY-A1-Phasen ausweiten (Sparquote je Phase als
   CI-tauglicher Report).

**Phase 2 — GPU (7900 XTX):**
5. P1-1 Modell-Residency/Warm-up + Multi-Song-Batching (Seed-Isolation je Song).
6. R3 ROCm-Ports (Muster BSR 42×; Paritäts-Test je Modell).
   ✅ **BANQUET (SOTA-ML-V5, 2026-09-18):** Torch-ROCm-Kern
   (`backend/core/dsp/banquet_torch_rocm.py`, 24-Zellen-BSRNN 1:1 aus den
   ONNX-Gewichten rekonstruiert, Parität ONNX-CPU max|Δ| ≈ 1,9e-6,
   deterministisch): ~160 ms/Fenster statt ~1,9 s ORT-ROCm (11,8×;
   B=4 ≈ 19,5×), End-to-End ~19×. Nebenfund: ORT-ROCm-LSTM-Kernels
   numerisch defekt (roh 0,35) — ONNX-Fallback läuft jetzt immer auf CPU
   (§V6-Warnung), der Torch-Kern ist damit auch der qualitätskorrekte Pfad.
   Re-Export mit dynamischer Batch-Dim (`scripts/export_banquet_batch_onnx.py`)
   bit-verifiziert; Mini-Batch-Empirie ~1,1× (Latenz-, nicht Durchsatzbindung).
   ⏳ **FCPE (Folge-Hebel, Messung 2026-09-18):** ORT-ROCm auch hier numerisch
   defekt (Salience max|Δ| ≈ 0,04, rel ≈ 0,19 vs. ORT-CPU — Softmax-/Attention-
   Kernels; dritter bestätigter Fall nach bs_roformer und BANQUET) → CPU-Policy
   im Registry bleibt korrekt. ROCm wäre 227 ms vs. 2082 ms CPU je 60-s-Mel
   (9,2×), aber falsch. Der GPU-Weg führt über einen Torch-ROCm-Port:
   torchfcpe-Quelle liegt im Repo (`models/fcpe/torchfcpe/`,
   `model_conformer_naive.py`), ONNX-Gewichte sind namentlich 1:1 zuzuordnen;
   es fehlen die Laufzeit-Deps `einops` + `local_attention` in der Venv.

**Phase 3 — strukturell:**
7. P2-1-Monolith-Split als Enabler für saubere Deferral-/Residency-Grenzen.
8. Deterministische Korpus-Parallelisierung (Shards mit festem Seed-Schema).

**Nicht tun (Verbotsliste):**
- Kein globales Stärke-Drosseln zugunsten von RT (Workaround §V7).
- Kein Überspringen hörbarer Reparaturen (Audibility-Gates bleiben fail-open, §V6).
- Keine bit-reduzierten Zwischenstufen ohne POW-r-Dither (§V5).
- Keine zufallsbehafteten Schnellpfade (Determinismus §G5 (GEBOTE.md)).

---

## 6. Akzeptanz-Protokoll je Maßnahme (Mess-Kadenz)

1. Baseline: P1-Diagnose-Skript (`scripts/artifact_freedom_diagnosis.py`) +
   AURIK Quality Score auf dem Korpus-Track.
2. Maßnahme umsetzen (mit Tests nach Test-Konvention).
3. Nachher: identische Messung — Score-Delta ≥ 0, af-Delta ≥ 0, RT-Faktor gemessen.
4. R5-Zertifikat: doppelter Lauf bit-identisch.
5. Nur bei Erfüllung aller Punkte: Merge. Sonst zurück (Never-worsen, §0).
