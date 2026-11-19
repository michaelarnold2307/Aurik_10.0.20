# P1-3 Restdeckung — Audibility-Guards: vollständiges Schwellwert-Guard-Inventar und Schließung

**Datum:** 2026-11-03
**Auftrag:** TODO-P1-3 (docs/TODOS_SOTA_ROADMAP.md) Restdeckung ausweisen und schließen
**Auslöser:** docs/reports/current/2026-10-28_vollstaendigkeit_tiefenanalyse.md E5:
„P1-3 nur teilweise: Audibility-Deckung für SCK/WBG/ATI/Formant umgesetzt — ob ALLE
Schwellwert-Guards abgedeckt sind, ist nicht ausgewiesen."
**Normativ:** §G10 (copilot-instructions.md) — Defekt-Freiheit unter der Hörbarkeitsschwelle,
universell; §G6 (copilot-instructions.md) — psychoakustische Präzision;
Hörordnung Ebene 2 (.github/instructions/hoerordnung.instructions.md) — Audibility =
Maskierungsschwelle statt Mess-Null; §SOTA-PSY-A1 (dsp.instructions.md);
§V7 (copilot-instructions.md) — keine phasen-individuellen Schwellwerte.

---

## 0. Ausführungsstand (Evidenz-Disclaimer)

Diese Arbeits-Session lief in einer Runtime **ohne Shell-/Exec-Werkzeug** (nur Datei-, Grep-,
Git-Inspektion). Aus diesem Grund ist der Ausführungsstand wie folgt — es wurde nichts
simuliert oder behauptet:

| Schritt | Status | Beleg |
|---|---|---|
| Guard-Inventar (file:line) | **erledigt** | §2 dieses Reports (grep-/read-basiert im Worktree) |
| Umsetzung fehlender Abdeckung | **erledigt** | §3, geänderte Dateien §6 |
| Unit-Tests ergänzt | **erledigt** | `tests/unit/test_p1_3_audibility_restdeckung.py` (14 Tests, statisch geprüft) |
| Unit-Tests grün (nice -n 19) | **NICHT AUSGEFÜHRT** | kein Shell-Tool in der Runtime |
| ruff --select F821,F601,B009,I001 | **NICHT AUSGEFÜHRT** | kein Shell-Tool in der Runtime |
| Conventional Commit | **NICHT AUSGEFÜHRT** | kein Shell-/git-write-Tool in der Runtime |

Nachzuholen (fokussiert, kein Voll-Suite-Lauf wegen parallelem B3-Messlauf):

```bash
nice -n 19 python3 -B -m pytest tests/unit/test_p1_3_audibility_restdeckung.py -q
nice -n 19 python3 -B -m pytest tests/unit/test_p1_3_masking_jnd_guards.py tests/unit/test_mikrodynamik_guard.py tests/unit/test_vibrato_guard.py tests/unit/test_hnr_guard.py -q
python3 -m ruff check <geänderte Dateien> --select F821,F601,B009,I001
git commit  # Conventional Commit, Vorschlag §7
```

Geänderte Dateien: §6. Die vorhandenen Tests der betroffenen Guards
(`test_mikrodynamik_guard.py`, `test_vibrato_guard.py`, `test_vocal_no_harm_gate.py`)
sind kompatibel erweitert (neue Parameter/Result-Felder sind optional/defaultbehaftet,
Bestandssignaturen unverändert).

---

## 1. Scope-Definition

**Schwellwert-Guard im Produktionspfad** = Mechanismus in `backend/core/`,
`backend/core/dsp/` oder `backend/core/phases/`, der

1. im Restaurierungs-Produktionspfad verdrahtet ist (von Phasen/UV3/Post-Phase-Hooks
   aufgerufen wird), **und**
2. eine Defekt-/Artefakt-Entscheidung an einer Mess-Schwelle trifft (dB, Korrelation,
   Ratio, Distanz) und daraufhin unterdrückt/rollt zurück/blendet/kappt.

**Abgedeckt (§G10/Hörordnung Ebene 2)** heißt: Die Schwelle kennt die lokale
Maskierungs-/JND-Dynamik — der maskierte Anteil eines Phasen-Deltas ist unhörbar und
löst keine Gegenmaßnahme aus (Muster: `estimate_delta_masking_jnd_db` in
`backend/core/residuum_masking.py:457`, `_DELTA_JND_CAP_DB = 6.0` ebenda :387,
Guard-Kontrakt „effektive Toleranz = max(fest, jnd_db)" in der Funktionsdoku).

Nicht im Scope (keine Defekt-/Artefakt-Schwellen): NaN/Inf-Schutz (§0a
(copilot-instructions.md)), Memory-/Budget-/Crash-Recovery-Guards, Streaming/Latenz-
Betriebslogik, Modell-Capability-Gates, reine Mess-/Witness-Module ohne
Suppress-Entscheid.

---

## 2. Vollständiges Guard-Inventar (file:line) mit Coverage-Status

### 2.1 A — Audibility-Machinerie (kanonische Muster, per Definition abgedeckt)

| Modul | file:line | Mechanismus | Status |
|---|---|---|---|
| `dsp/audibility_gate.py` | `defect_audibility` :89 | ISO-11172-3/Zwicker-Bandmaskierung, „subaudibel → überspringen" (§SOTA-PSY-A1) | abgedeckt |
| `defect_audibility_gate.py` | `evaluate_defect_audibility` :37 | Maskierungsschwelle je Defekt (Hörordnung Ebene 2) | abgedeckt |
| `defect_to_audibility.py` | Engine :50–344 | simultane + zeitliche Maskierung, adaptive Drift-Toleranz | abgedeckt |
| `residuum_masking.py` | `DeltaMaskingJNDResult` :391, `estimate_delta_masking_jnd_db` :457, `delta_masking_margin_db_per_band` :435 | P1-3-Maskierungs-JND für Phasen-Deltas | abgedeckt (P1-3-Kern) |
| `dsp/hearing_jnd.py` | `jnd()` :30 | Hör-JND-Tabelle (Ebene 1/2, §G6) | abgedeckt |
| `dsp/masking_model.py` | `compute_masking_threshold_db` :10 | Maskierungsschwelle-Schätzung | abgedeckt |
| `dsp/temporal_masking.py` | `ForwardMaskingGuard` :13/:23, `apply_to_strength` :206/:234 | Forward-Masking (zeitliche Hörbarkeit) | abgedeckt |
| `dsp/binaural_masking.py` | :156/:191/:226 | binaurale Maskierungsdifferenz | abgedeckt |
| `dsp/psychoacoustics.py` | `apply_psychoacoustic_masking_clamp` :18 | ISO11172-Clamp (in phase_19 genutzt) | abgedeckt |
| `dsp/erb_auditory_masking.py` | :213 | ERB-Maskierungsmodell | abgedeckt |
| `dsp/psychoacoustic_masking_model.py` | :62 | Psychoakustik-Loss-Masking | abgedeckt |
| `dsp/masking_analyzer.py` | :86 | Maskierungsanalyse | abgedeckt |
| `dsp/zwicker_loudness.py` | :82 | Lautheits-Domäne (Zwicker) | abgedeckt |
| `dsp/audibility_targets.py` | `residual_audibility` :38–238 | Zielwert-Formeln inkl. Maskierungsschwelle | abgedeckt |
| `dsp/perceptual_budget.py` | :18 | JND-Budget (wahrnehmungsbasierte Kontrolle) | abgedeckt |
| `dsp/perceptual_loudness_cap.py` | `PerceptualLoudnessCap` :66, `apply_perceptual_loudness_cap` :283 | Peak-STL in **Sone** (perzeptuelle Domäne, §SOTA-PSY-A7), materialadaptive Toleranz 0,15/5 % | abgedeckt (perzeptuelle Domäne) |
| `defect_scanner.py` | JND-Defektkatalog :736 (`jnd_db`/`masking_factor`) | Defekt→JND/Masking-Mapping | abgedeckt |

### 2.2 B — P1-3-Guards (Status „abgedeckt" lt. TODO-P1-3, hier verifiziert)

| Guard | file:line | Form | Status |
|---|---|---|---|
| SCK `dsp/spectral_color_guard.py` | Schwellen :41–52, `check_spectral_color_preservation` :76, JND :160/:177 | §SCK-Form: maskierte Abweichung relaxiert Korrelations-Schwelle, max. −0,20 bei voller 6-dB-JND | abgedeckt (P1-3) |
| WBG `dsp/warmth_guard.py` | `WARMTH_LOSS_THRESHOLD_DB` :24, `measure_warmth_band_delta` :78, JND :94/:114 | §WBG-Form: maskierter Verlust-Anteil zählt nicht kumulativ | abgedeckt (P1-3) |
| ATI `dsp/onset_guard.py` | `apply_onset_protection_mask` :117–169 | §ATI-Form: Toleranz = max(1,5 dB, JND) | abgedeckt (P1-3) |
| Formant `dsp/lpc_formant_tracker.py` | `check_formant_shift_db` :274, `resolve_formant_tolerance_db` :330/:438, Bark-Margen :485 | §0p-Form: Toleranz = max(fest/§V43-Frequenz-JND, lokale Bark-Marge) | abgedeckt (P1-3) |
| Gain-Step `dsp/temporal_continuity_guard.py` | :45/:103/:155 | §Gain-Step-Form: effektiver Schwellwert = max(1,5 dB, JND) | abgedeckt (P1-3) |

### 2.3 C — Weitere produktionsgebundene Guards MIT Audibility-Abdeckung

| Guard | file:line | Abdeckung | Status |
|---|---|---|---|
| Sibilant-Maskierungsgate phase_19 | `phase_19_de_esser.py:465–481` (Skips über `defect_audibility`), PSY-A1-Call-Sites phase_01/03/06/07/08/09/19/23/27 | §SOTA-PSY-A1: maskierte Sibilanten/Defekte → Skip | abgedeckt |
| JND-Gate phase_25 | `phase_25_azimuth_correction.py:172–177, 182–187` (ITD-JND + Pegel-JND-Floor), Reporting :119–130 | Hör-JND (`hearing_jnd`) als Schwellwert-Floor (§SOTA-PSY-A8) | abgedeckt |
| JND-Gate phase_31 | `phase_31_speed_pitch_correction.py:233–241, 245–258` (Frequenz-JND-Toleranz „<1 JND (unhörbar)"), :258 | Frequenz-JND (§V43 (VERBOTEN.md)) | abgedeckt |
| Additive-/Hybrid-/Stem-Gates | `dsp/additive_synthesis_gate.py:112–188`, `dsp/hybrid_denoise_fusion.py:258–375`, `dsp/stem_recombination_gates.py:219–341` | `compute_masking_threshold_db`-basierte Band-Hörbarkeit | abgedeckt |
| Defekt-Audibility-Call-Sites | phase_01 (:47 u. :2868), phase_06 (:409), phase_07 (:3565), phase_08 (:845), phase_09 (:1159), phase_23 (:562), phase_27 (:383) | PSY-A1-Skip-Muster | abgedeckt |
| HNR-Blend-Konsumenten | `vocal_no_harm_gate.py:388`, `dsp/hnr_guard.py:219` (apply_hnr_blend), `dsp/stem_level_restorer.py:891` (check_hallucination) | über nachstehend umgesetzte Guards (2.4) | abgedeckt (seit 2026-11-03) |

### 2.4 D — Restdeckung: Guards mit reinen Mess-Schwellwerten → **UMGESETZT (2026-11-03)**

Alle fünf Guards folgen dem P1-3-Muster (`estimate_delta_masking_jnd_db` je Guard-Modul,
keine phasen-individuellen Schwellwerte — §V7 (copilot-instructions.md)).

| Guard | file:line (Schwelle) | Form der Abdeckung | Umsetzung |
|---|---|---|---|
| **hnr_guard** | `dsp/hnr_guard.py:28` (`_HNR_DELTA_THRESHOLD_DB = 3.0`), `check_hnr_delta` :131 | §ATI-Form: effektive ΔHNR-Schwelle = max(3,0 dB, JND) | JND-Block :153–158, `effective_threshold_db`/`masking_jnd_db` im Result :178–179 |
| **mikrodynamik_guard** | `dsp/mikrodynamik_guard.py:26` (`MIKRODYNAMIK_THRESHOLD = 0.93`), :32–37 (Material-Floors), `recommend_mikrodynamik_wet` :48 | §SCK-Form: Korrelations-Schwelle relaxiert linear um max. `_JND_RELAX_CAP = 0.20` bei voller 6-dB-JND (Konstanten :38–43) | Optionale `pre`/`post`/`sr`-Parameter :48–52, Relaxations-Block :76–85; **12 Call-Sites** verdrahtet (phase_03/18/20/28/29/49/50/57/59/61/62/63) |
| **vibrato_guard** | `dsp/vibrato_guard.py:33` (`VIBRATO_MAX_REDUCTION_PCT = 10.0`), Entscheidung :145–154 | §SCK-Form: Toleranz = 10 % + max. `_RELAX_CAP_PP = 5.0` pp bei voller 6-dB-JND (Konstanten :35–39) | JND-Block :148–153; `VibratoDepthResult` erweitert (:52–53: `effective_tolerance_pct`, `masking_jnd_db`) |
| **hallucination_guard** | `dsp/hallucination_guard.py:37` (`_ROLLBACK_THRESHOLD_FLOOR = 0.15`), :38 (`_PENALTY_FLOOR = 0.08`), Schwellenlogik :254–278 | §SCK-Form: nur **hörbare** Spectral-Novelty zählt — linearer Faktor exponiert/(exponiert+maskiert); vollständig maskiert → 0 (kein Rollback) | Audibility-Block :192–204 (`spectral_novelty_audible`, `novelty_masking_jnd_db` im Meta); Schwellenvergleiche :258/:272 auf hörbare Novelty; Raw-Novelty bleibt als Messwert erhalten |
| **noise_texture_guard** | `dsp/noise_texture_guard.py:46` (`_MATERIAL_RESIDUAL_SLOPE_RANGES`), :65 (`_MAX_DEVIATION`), `compute_noise_texture_distance` :98, Whitening-Entscheid an den Call-Sites (0,25) | §SCK-Form: Textur-Distanz skaliert mit exponiertem Delta-Anteil; vollständig maskiertes Residual → Distanz 0 (unhörbares Whitening) | Optionale `pre`/`post`-Parameter :101–104, Skalierungs-Block :144–157; **17 Call-Sites** verdrahtet (phase_03/09/18/19/20/27/28/29/43/49/50/55/57/59/61/65/66) |

Call-Site-Verdrahtung vollständig verifiziert (grep 2026-11-03): keine
`compute_noise_texture_distance`-/`recommend_mikrodynamik_wet`-Aufrufstelle im
Produktionspfad ohne `pre=`/`post=`-Kontext (Alias-Formen `_ntNN_*_fn` inklusive;
phase_55/65 zunächst übersehen und nachverdrahtet).

### 2.5 E — Begründete Ausnahmen (keine Maskierungs-Relaxation)

| Guard | file:line | Begründung |
|---|---|---|
| **transient_guard** | `dsp/transient_guard.py:33` (`TRANSIENT_SHIFT_THRESHOLD_MS = 2.0`), `detect_transient_shifts` :122 | **Normative Spec-Grenze** (Attack-Zeiten ≤ ±2 ms, Spec 01/07) — Zeitdomäne, bereits strenger/jnd-nah kalibriert als die Onset-Timing-Wahrnehmung; eine JND-Relaxation würde die Spec-Grenze verletzen (Spec > Code, §1 AGENTS.md). §G10 wird hier über die Spez-Grenze abgebildet. |
| **noise_floor_guard** | `dsp/noise_floor_guard.py:42–46` (`_MATERIAL_FLOORS_DBFS`), `apply_noise_floor_minimum` :63 | **Materialphysikalische Floor-Kalibrierung** (§V (copilot-instructions.md) CD-Rauschprofil-Modell, §SOTA-PM-MAT): Der Floor ist keine Defekt-Toleranz, sondern die Echtheits-Vorgabe der Materialklasse (Verhindern von digitalem „Zu-Tot-Rauschen"). |
| **phrase_boundary_guard** | `dsp/phrase_boundary_guard.py:64` (`detect_phrase_boundaries`, −30 dBFS-Minima) | **Strukturelle Segmentierung** (Phrasengrenzen-Erkennung), keine Defekt-/Artefakt-Entscheidung — erkennt Grenzen, toleriert keine Defekte. |
| **edge_gain_cap** | `dsp/edge_gain_cap.py:17/:21` (2 dB Edge-Cap) | **Verarbeitungs-Cap** gegen Edge-Clicks (Gain-Limit, kein Defekt-Toleranz-Entscheid). Gain-Step-Hörbarkeit ist zentral über `temporal_continuity_guard` (§Gain-Step, 2.2) abgedeckt — doppelte Audibility wäre redundant. |
| **stereo_temporal_coherence_guard** | `stereo_temporal_coherence_guard.py:68` (`_MAX_DELAY_MS = 20.0`), :83 (`_CORR_DROP_TOLERANCE = 0.15`), :93/:145 | **Technische Latenz-Invariante** (§v10.13): physische Sample-Ausrichtung/Korrelations-Korrektur — Engineering-Korrektheit statt Hörbarkeits-Entscheid. |
| **comfort_guard** | `comfort_guard.py:26` (`COMFORT_THRESHOLD = 0.12`), `apply_comfort_guard` :143 | **Hörmüdungs-Kriterium** auf psychoakustischer Metrik (Sharpness, ISO 532): Kriterium ist ein Komfort-Target, nicht die Detektierbarkeit eines Defekts — Maskierungs-/JND-Relaxation ist hier kategorie-fremd (analog Sone-Domäne §SOTA-PSY-A7 bei `perceptual_loudness_cap`). |
| **harmonic_preservation_guard** | `harmonic_preservation_guard.py:334` (`build_harmonic_protection_masks`), :407 (`adaptive_harmonic_gain_floor`) | **Kein Schwellwert-Guard**: baut Schutzmasken/Gain-Floors (schützt Inhalte), trifft keine Defekt-Erkennungs-Entscheidung an einer Mess-Schwelle. |
| **artifact_freedom_guard** | `dsp/artifact_freedom_guard.py:22/:55–62` (`af_fast_never_worsen`, Click-/Pre-Echo-Proxy), :146/:239 | **Delta-basierter Regression-Guard** (Never-worsen-Prinzip, AGENTS.md „Guard-Kalibrierung"): vergleicht Artefakt-Scores pre↔post mit Toleranz, inkl. Maskierungs-Schätzung (`_estimate_noise_floor_db`). §G10 greift über das Nie-schlechter-als-Input-Prinzip (delta-basiert). |
| **vibrato_continuity_guard** | `dsp/vibrato_continuity_guard.py:61` (`PHASE_DISCONTINUITY_THRESHOLD_DEG = 90.0`) | **Technische Kontinuität an Chunk-Grenzen** (F0-Phasensprung → Crossfade): verhindert hörbare Nahtstellen präventiv; keine Defekt-Toleranz-Messung. Zudem nicht in den Phasen-Pfad verdrahtet (Singleton, 2.6). |
| **End-Level-Gates** (per_phase_musical_goals_gate :20–42/:208–541, `do_no_harm_guardian.py:65/:306`, `holistic_perceptual_gate.py:109`, export/mid_pipeline-Gates) | siehe file:line | **Delta-basierte Never-worsen-/Zielwert-Gates** auf Musical-Goals- und Qualitäts-Metriken (AGENTS.md Guard-Kalibrierung: Hard-Fail nur bei Regression gegenüber dem Input). Keine absoluten Defekt-Mess-Schwellen im §G10-Sinne; die Defekt-Audibility wird phasenlokal über §SOTA-PSY-A1 (2.1/2.3) abgebildet. Tiefe Einzelprüfung der Zielwert-Schwellen ist als Rest-Risiko notiert (§5). |

### 2.6 F — Nicht im Produktionspfad (keine Abdeckung erforderlich)

| Modul | file:line | Status |
|---|---|---|
| `dsp/spectral_*_guard.py` (16 Module: band_energy/centroid/crest/energy/entropy/flatness/flux/irregularity/kurtosis/peakiness/rolloff/roughness/skewness/slope/spread/variance/zero_crossing) | z. B. `spectral_centroid_guard.py:42` | DSPContract-Stubs/Analyse-Module — **von keinem Produktionspfad importiert** (grep 2026-11-03: nur Selbstreferenzen) |
| `dsp/over_dryness_guard.py`, `dsp/multiband_gate.py` | :42 | DSPContract-Stubs, keine Phasen-Verdrahtung |
| `dsp/human_hearing_comfort_guard.py` | `apply_human_hearing_comfort_guard` :36 | keine Aufrufer in `backend/` (grep 2026-11-03) |
| `dsp/transient_protection_guard.py` | `TransientProtectionGuard` | nur Regulator-Pfad (`regulator/_dsp_applier.py`), nicht der Phasen-Produktionspfad |

### 2.7 Mess-/Witness-Module (kein Suppress-Entscheid, daher keine Guard-Abdeckung nötig)

`dsp/vocal_register_detector.py` (:181/:248/:292/:337/:418/:485), `dsp/beats_onset_detector.py`
(`onset_witness_stats`), `dsp/silence_mask.py` (`apply_silence_preservation`, Struktur-
Bewahrung), `phases/nmr_feedback.py`, `phases/psychoacoustic_frame.py`,
`phases/hum_drift_tracker.py`, `phases/zwicker_metrics.py`, `phases/output_guard.py`
(OutputGuardDecision-Struktur), `defect_scanner`-Detektoren (:733/:734 liefern
jnd_db/masking_factor an die Audibility-Machinerie) — reine Messung/Erfassung.

---

## 3. Umsetzungsdetails (1:1 nach P1-3-Muster)

Muster (dsp.instructions.md §0p/§WBG/§ATI/§SCK, Guard-Kontrakt in
`residuum_masking.py:457`):

1. **§ATI-Form** (dB-Größe): `effektive_Schwelle = max(fest, jnd_db)` — hnr_guard.
2. **§SCK-Form** (keine dB-Größe): beschränkte lineare Relaxation,
   `_relax = clip(jnd_db * (cap / 6.0), 0, cap)` — mikrodynamik_guard (cap 0.20,
   identisch zur §SCK-Kalibrierung §v10.759), vibrato_guard (cap 5 pp),
   hallucination_guard/noise_texture_guard (Skalierung mit exponiertem Anteil).

Alle Änderungen sind zentral in den Guard-Modulen (keine phasen-individuellen
Schwellwerte, §V7 (copilot-instructions.md)); die Phasen reichen nur den
pre/post-Kontext durch (wie bei P1-3 bereits in `spectral_color_guard`/
`warmth_guard`/`onset_guard`).

**Rückwärtskompatibilität:** `compute_noise_texture_distance` und
`recommend_mikrodynamik_wet` erhalten keyword-only Optionen (Default `None` =
bisheriges Verhalten); `VibratoDepthResult` erhielt zwei Default-Felder;
`check_hnr_delta`/`check_hallucination` behalten ihre Signaturen und ergänzen
nur Result-/Meta-Felder. Bestehende Tests (`test_mikrodynamik_guard.py`,
`test_vibrato_guard.py`, `test_worldclass_quality_guards.py`,
`test_ml_dsp_fallback_paths.py`, `test_vocal_no_harm_gate.py`) bleiben kompatibel.

---

## 4. Tests

Neu: `tests/unit/test_p1_3_audibility_restdeckung.py` — 14 Tests, synthetische
Signale, deterministische JND-Steuerung via Monkeypatch (Muster:
`tests/unit/test_p1_3_masking_jnd_guards.py`):

- `TestHNRMaskingJND` (3): unmaskierter ΔHNR-Trigger; maskierte Schwelle max(3,6);
  feste Schwelle bleibt Floor.
- `TestMikrodynamikMaskingJND` (4): Baseline ohne Kontext unverändert; maskierte
  Relaxation (0,93→0,73) erlaubt Wet=1,0; unmaskiert strikt; Relaxation bei 0,20
  begrenzt.
- `TestVibratoMaskingJND` (3): 12,5 % Reduktion → unmaskiert not-ok / maskiert ok
  (Toleranz 15 %); Cap-Test (18,75 % auch bei voller JND not-ok).
- `TestHallucinationMaskingJND` (3): unmaskierte Novelty 0,30 → Rollback;
  vollständig maskiert → weder Rollback noch Penalty; partielle Exponierung skaliert
  auf 0,15.
- `TestNoiseTextureMaskingJND` (3): unmaskiertes Whitening geflaggt (Distanz
  unverändert); maskiert → 0,0; halbe Exponierung → halbe Distanz.

Ausführung: **siehe §0 — in dieser Runtime nicht ausführbar**. Vorgesehener
fokussierter Befehl: `nice -n 19 python3 -B -m pytest tests/unit/test_p1_3_audibility_restdeckung.py -q`.

---

## 5. Rest-Risiken / offene Punkte

1. **Ausführung fehlt** (§0): pytest/ruff/Commit sind nachzuholen; die Tests sind
   statisch gegen die tatsächlichen Signaturen geprüft (read/grep-Verifikation),
   aber nicht gelaufen.
2. End-Level-Gates (per_phase_musical_goals_gate u. a., 2.5): die dortigen
   Zielwert-/Delta-Schwellen wurden als Regression-/Never-worsen-Gates klassifiziert;
   eine tiefe Audibility-Analyse je Einzelwert steht aus und ist kein
   Defekt-Guard-Rest im §G10-Sinne (Folge-Task empfohlen, wenn gewünscht).
3. Schwellwert-Duplikate 0,25 vs. 0,10/0,18 (phase_18/55/65, Noise-Texture-Trigger):
   vorbestandene phasen-individuelle Trigger — bewusst **nicht** angefasst (Scope:
   Audibility-Deckung), §V7-Prüfung separat empfohlen.
4. Commit bewusst offen: Pre-Commit-Hooks (`aurik-unit-smoke` u. a.) laufen
   Voll-Suites — wegen des parallelen B3-Messlaufs muss der Hook-Lauf vom Manager
   terminiert werden; Vorschlag §7.

---

## 6. Geänderte/neue Dateien

Neu:
- `tests/unit/test_p1_3_audibility_restdeckung.py`
- `docs/reports/current/2026-11-03_p1_3_audibility_restdeckung.md` (dieser Report)
- `.github/FILE_REGISTRY.md` (2 Einträge, Write-Gate)

Geändert (Guards):
- `backend/core/dsp/hnr_guard.py`
- `backend/core/dsp/mikrodynamik_guard.py`
- `backend/core/dsp/vibrato_guard.py`
- `backend/core/dsp/hallucination_guard.py`
- `backend/core/dsp/noise_texture_guard.py`

Geändert (Call-Sites pre/post-Verdrahtung):
- `backend/core/phases/phase_03_denoise.py`, `phase_09_crackle_removal.py`,
  `phase_18_noise_gate.py`, `phase_19_de_esser.py`, `phase_20_reverb_reduction.py`,
  `phase_27_click_pop_removal.py`, `phase_28_surface_noise_profiling.py`,
  `phase_29_tape_hiss_reduction.py`, `phase_43_ml_deesser.py`,
  `phase_49_advanced_dereverb.py`, `phase_50_spectral_repair.py`,
  `phase_55_diffusion_inpainting.py`, `phase_57_print_through_reduction.py`,
  `phase_59_modulation_noise_reduction.py`, `phase_61_groove_echo_cancellation.py`,
  `phase_62_crosstalk_cancellation.py`, `phase_63_intermodulation_reduction.py`,
  `phase_65_vocal_naturalness_restoration.py`, `phase_66_stem_targeted_nr.py`

Write-Gate-Hinweis: `scripts/repo_search.py --before-create` war in der Runtime nicht
ausführbar; die Prüfung wurde grep-/glob-basiert nachvollzogen (keine Namens-/
Inhalts-Kollision; `tests/unit/test_p1_3_masking_jnd_guards.py` deckt die
JND-Schätzung ab, die neue Datei die Restdeckungs-Guards). Bei Gelegenheit
`python scripts/repo_search.py --before-create tests/unit/test_p1_3_audibility_restdeckung.py`
nachholen; FILE_REGISTRY-Einträge sind gesetzt (§2 AGENTS.md Write-Gate).

---

## 7. Conventional-Commit-Vorschlag (nicht ausgeführt)

```
feat(audibility): schließe P1-3-Restdeckung der Schwellwert-Guards (§G10)

- Guard-Inventar mit Coverage-Status je Guard (Report 2026-11-03)
- Audibility/Masking für hnr/mikrodynamik/vibrato/hallucination/noise_texture
  Guard nach P1-3-Muster (§ATI-/§SCK-Form, residuum_masking JND)
- 29 Call-Sites (17 noise_texture, 12 mikrodynamik) mit pre/post verdrahtet
- 14 Unit-Tests (maskiert vs. exponiert, deterministisch)
- Begründete Ausnahmen dokumentiert (Spec-Grenze transient, §V-Rauschprofil
  noise_floor, Struktur-/Technik-Guards)
```

Hinweis: `tests/unit/test_p1_3_audibility_restdeckung.py` und ruff müssen vor dem
Commit grün/clean sein (§0); `TASK_CHANGES.md` via
`python scripts/change_ledger.py snapshot` nachziehen.

---

## 8. Antwort auf E5 (Tiefenanalyse 2026-10-28)

„Ob ALLE Schwellwert-Guards abgedeckt sind, ist nicht ausgewiesen" → **jetzt
ausgewiesen** (§2, Inventar file:line) und **geschlossen**: 5 Guards mit reinen
Mess-Schwellwerten haben Maskierungs-/JND-Abdeckung erhalten (2.4), 10 Kandidaten
sind mit dokumentierter Spec-/Norm-Begründung als Ausnahmen klassifiziert (2.5),
der Rest ist abgedeckt (2.1–2.3) oder nicht im Produktionspfad (2.6). TODO-P1-3
gilt damit inhaltlich als geschlossen; die Rest-Evidenz (grüne Tests, ruff,
Commit) ist §0 zu entnehmen.
