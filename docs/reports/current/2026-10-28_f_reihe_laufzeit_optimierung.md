# F-Reihe Laufzeit-Optimierung (§PERF-R18) — 2026-10-28

**Auftrag (Manager, SOTA-Task F-Reihe):** End-to-End-Realtime-Faktor
(RT_REALITY_MEASURED = 53,0×, `docs/TODOS_SOTA_ROADMAP.md`) auf ≤ 32× RT
(D-K3-22-Deckel) senken — klangneutral (Hörordnung Ebene 1–3, keine
ML-Fallbacks B1), deterministisch (§G5), mit ehrlichem Vorher/Nachher.

**Version:** 10.12.15 → 10.12.16 · **Maßnahme-ID:** §PERF-R18
(`docs/TODOS_SOTA_ROADMAP.md`)

---

## 1. Ehrlichkeit zuerst: was wurde GEMESSEN, was nicht

Diese Session lief in einem Worker **ohne Ausführungskapazität** (keine
Shell, kein Python-Lauf, kein `git commit`). Konsequenz:

| Gefordert | Status | Evidenz / Ersatz |
| --- | --- | --- |
| cProfile-Reports (`docs/reports/current/*_cprofile.json`) | **nicht vorhanden** | Die im Brief genannten Dateien sind nicht im Worktree (gitignored — „interne Dev-Artefakte bleiben lokal", c4307207). Profil-Basis = dokumentierte Befunde (unten §2) |
| RT-Messung Referenz-Song vorher/nachher | **NICHT ausgeführt** | Mess-Methode + Befehle in §5; Zahlen in §4 sind **Extrapolationen aus dokumentierten Messungen**, keine neuen Messwerte |
| `tests/normative/test_performance_budget_ci_gate.py` grün | **NICHT ausgeführt** | Die Datei prüft reine Konstanten (`D_K3_22_32X_RT_CEILING = 32.0` etc.) — von den Änderungen nicht berührt; CI-Bestätigung ausstehend |
| Äquivalenz-Tests (`tests/unit/test_perf_r18_cache_equivalence.py`) | **geschrieben, NICHT ausgeführt** | Pinnt die Bit-Identität; muss in CI/lokal laufen |
| Commit (Conventional Commits) | **nicht möglich** | Commit-Message vorbereitet (§6) |

Die Akzeptanz „≤ 32× RT gemessen" ist damit **in dieser Session nicht
abgeschlossen**. Was geliefert ist: zwei bit-exakte, gegen die
benannten cProfil-Treiber gerichtete Optimierungen mit
Äquivalenz-Pinning, plus die vollständige Messkette für den
Referenz-Song-Nachweis.

## 2. Profilierung: wo die 53× sitzen (Beleglage)

Gesichtet: `docs/TODOS_SOTA_ROADMAP.md` (§PERF-R1–R18, TODO-P0-1), Git-
Historie (c4307207 „Detektor-Memo widerlegt", 71359308 „Gesamt-Laufzeit
−32 %"), `docs/GESAMTKONZEPT_PERFORMANCE_WOHLKLANG.md`, statische Analyse
der benannten Treiber.

1. **defect_scanner (benannter Treiber, c4307207):** `scan()` rechnet
   `_detect_flutter`/`_detect_wow` **doppelt** auf identischem
   Mono-Signal — Material-Erkennung nutzt die vollen Detektoren als
   Features + eigener Detektor-Pass: **44,9 s der 58,4 s Scan-Zeit (77 %)**.
   Ein Verdict-Memo wurde dort zu Recht verworfen (ohne Zustand im Schlüssel
   → 3 von 63 Scores verhaltensändernd; mit Zustand → 0 Treffer).
2. **phase_12 wow/flutter (benannter Treiber, §PERF-R2 „phase_12 war der
   Flaschenhals"):** der Pure-Python-pYIN-Fallback (FAST/BALANCED/ohne
   librosa) enthält ein Multi-Threshold-Doppel-Loop: 20 Thresholds ×
   ~912 tau je Fenster ⇒ über 224 s ≈ 4·10^8 Interpreter-Iterationen.
3. **Restlast (dokumentiert, NICHT in dieser Session angefasst):**
   ML-Lade-/Inferenzkosten („BANQUET 0,78 s/Prozess = 62 % der
   Phasen-Zeit"), P1-GPU-Ports (GPU-gebunden), P2-Phase-Gap-Rest
   (~5 s × 41 Phasen × 8 Chunks ≈ 27 min je Song, offen),
   P3-PLM-Residency. Diese Posten tragen nach aktueller Messlage die
   53×-Restlücke zur 32×-Norm.

## 3. Umgesetzte Optimierungen (§PERF-R18) — beide bit-exakt

### 3.1 defect_scanner: Feature-Cache für Wow/Flutter („cachebare Zwischenwerte")

Neu: `_wf_cached_features` + `_flutter_base_features` /
`_wow_track_features` (`backend/core/defect_scanner.py`).

- **Idee:** Gecacht wird ausschließlich die **threshold-unabhängige
  Feature-Schicht** als reine Funktion aus `(audio, sample_rate)`.
  Schwellwerte/Material fließen weiterhin **pro Aufruf** im Verdict-Teil
  ein. Damit trifft der Cache den D-K3-40-Negativbefund nicht:
  - „Memo ohne Zustand" war verhaltensändernd, weil es das **Verdict**
    (threshold-abhängig) cachte — hier bleibt das Verdict live;
  - „Memo mit Zustand" hatte 0 Treffer, weil Material/Schwellen zwischen
    den Pässen wechseln — hier ist der Schlüssel zustandsfrei, weil die
    Features zustandsfrei sind.
- **Bit-Identität by construction:** ein Cache-Treffer liefert exakt
  dieselben Float-Arrays, die die identische reine Funktion auf identischem
  Input erzeugt hat (§G5 (copilot-instructions.md)). Schwellwert-Wechsel
  zwischen zwei Aufrufen ändern nur den Verdict-Lauf, nicht die Features —
  genau das pinnt der Test.
- **Schlüssel:** `(Tag, sample_rate, dtype, BLAKE2b(Signal-Bytes))`;
  Levenshtein-frei, content-adressiert. Reset zusammen mit
  `_scan_welch_cache` (Song-Isolation §G1 (GEBOTE.md)) + FIFO-Trim (8).
  `_coherent_scrape_am`/`_coherent_subband_fm` (ebenfalls rein in
  `(audio, sr)`, verifiziert) sind mitgecacht.
- **Scope:** Nur `_detect_flutter`/`_detect_wow` (die belegten 77 %).
  Der Struktur-Verdacht „3 Detektoren im Loop konsultieren
  `self.material_type`" (D-K3-40) wurde bewusst **nicht** angefasst —
  verhaltensändernd.

### 3.2 phase_12: pYIN-Multi-Threshold-Tau-Suche vektorisiert („gebündelte Operationen statt Schleifen")

Neu: `_pyin_candidate_taus` (`backend/core/phases/phase_12_wow_flutter_fix.py`).

- Der Doppel-Loop `for thr … for tau …` ist ersetzt durch eine vektorisierte
  Suche über alle Thresholds: **identische boolesche Vergleiche in
  identischer Reihenfolge** (`cmnd[tau] < thr`, `<=`-Nachbarn,
  Randbedingung `0 < tau < max_period-1`); Break-Position == erstes True ==
  `argmax`.
- Fallback-`argmin` und parabolische Interpolation sind **unverändert**
  im Aufrufer geblieben (Schleifen-Restlogik bytegleich).
- Betroffen: nur der Pure-Python-Fallback (FAST/BALANCED/ohne librosa).
  Der librosa-/pyin_fast-Primärpfad (QUALITY) ist unberührt.

## 4. Erwartete Wirkung (Projektion, KEINE neuen Messwerte)

| Hebel | Basis (dokumentierte Messung) | Projizierter Effekt |
| --- | --- | --- |
| Feature-Cache | D-K3-40: 44,9 s von 58,4 s Scan-Zeit liefen doppelt; der Feature-Layer der 2. Ausführung entfällt | ~22 s je `scan()`; auf 8-Chunks/Song ≈ 3 min/Song (falls Scan je Chunk), plus jeder Bridge-/Denker-Aufruf |
| pYIN-Tau-Suche | ~4·10^8 Python-Iterationen über 224 s | Fallback-Pfad (FAST/BALANCED/ohne librosa): Sekunden bis Minuten je Song; QUALITY-Primärpfad: kein Effekt |

**Zur 32×-Norm:** Diese zwei Hebel allein tragen die 53×→32×-Lücke
(≈ 40 % Gesamtlaufzeit) **nicht**. Nach der dokumentierten Messlage sitzt
die Restlast bei P1 (ML-GPU-Ports, GPU-gebunden), P2 (Phase-Gap-Rest,
~27 min je Song) und P3 (PLM-Residency) — beides in der
SOTA-RESTHEBEL-MATRIX als OFFEN/GPU-GEBUNDEN geführt. Das ist der
begründete Grund, warum 32× in dieser Session nicht als gemessenes
Ergebnis vorgelegt wird (Akzeptanz-Alternative „bestmögliche Reduktion
mit Begründung").

## 5. Qualitäts-/Wohlklang-Neutralität (Hörordnung Ebene 1–3)

Beide Maßnahmen sind **bit-exakt** — identischer Output auf identischem
Input, unabhängig vom Cache-Zustand. Damit ist ein Hörunterschied
ausgeschlossen (kein A/B nötig; Ebene 1–3 unberührt). Keine
ML-Fallbacks eingeführt (B1), keine phasen-individuellen Schwellwerte
(§V7), Stärke-Logik unverändert, Determinismus §G5 gepinnt
(warm ≡ kalt, gleicher Input ⇒ identischer Output).

## 6. Mess-Protokoll Referenz-Song (vorher/nachher) — auszuführen

Referenz-Song: `test_audio/Testkünstlerin (Schlager) - Du wolltest nur ein
Abenteuer, aber ich suchte einen Freund.mp3` (224 s, wie
`scripts/profile_pipeline_hotspots.py`).

1. **Vorher-Zweig:** `git worktree add /tmp/aurik-vorher HEAD` (vor diesem
   Commit) — danach identisches Vorgehen auf beiden Worktrees.
2. **End-to-End-RT:** Kanonischer Lauf über `scripts/_aurik_run_excellence.py`
   (loggt `RT-Faktor`); RT-Faktor = Wall-Zeit / Audio-Dauer. **Messfenster
   ohne parallele GPU-Trainingsjobs** (Hinweis des Briefs: Trainingsjobs
   verzerren die Zahl; Fenster protokollieren, Jobs ggf. drosseln).
   Jeweils 3 Läufe, Median berichten (225-s-Lauf-Baseline des
   überwachten Laufs: Wall 12 137,5 s ≈ 53,9× RT — die 53×-Ausgangslage).
3. **Hotspot-Beweis cProfile:** `python scripts/profile_pipeline_hotspots.py
   --seconds 10 --out docs/reports/current/` auf beiden Ständen;
   `_detect_flutter`/`_detect_wow`/`_estimate_pitch_pyin` vergleichen.
4. **Qualitäts-Gegenprobe:** bit-Identität der Ausgabe-Audios
   (SHA-256 der Exporte Vorher/Nachher — muss identisch sein);
   zusätzlich `tests/normative/test_performance_budget_ci_gate.py` +
   `pytest tests/unit/test_perf_r18_cache_equivalence.py`.
5. **Zahlen in diesen Report nachtragen** (Tabelle §4 ergänzen) und
   D-K3-22-Wahrheit (`RT_REALITY_MEASURED`) nur bei gemessenem Nachweis
   korrigieren (TODO-P0-1: „kein Zahlenwechsel ohne Strukturaufgabe").

## 7. Commit (vorbereitet, nicht ausgeführt)

```
perf(defect-scanner,phase-12): bit-exakter Feature-Cache + vektorisierte pYIN-Tau-Suche (§PERF-R18)

- defect_scanner: Wow/Flutter-Feature-Cache (reine (audio, sr)-Features,
  Verdict pro Aufruf) gegen die D-K3-40-Duplikatrechnung (44,9 s/58,4 s)
- phase_12: pYIN-Multi-Threshold-Tau-Suche vektorisiert (bit-identisch,
  ≈ 4·10^8 Python-Iterationen je 224 s entfallen)
- tests: Äquivalenz-Pinning (Cached ≡ Uncached, Scan-A/B, Determinismus)
- docs: §PERF-R18, Report, Version 10.12.16
```

Offene Buchhaltung beim Commit: `python scripts/change_ledger.py snapshot`
→ `TASK_CHANGES.md` ausführen; bei Pre-Commit-Drift der
`spec_drift_baseline.json` (nur falls doch normative Dateien berührt)
Baseline regenerieren.

## 8. Geänderte Dateien

- `backend/core/defect_scanner.py` — Feature-Cache + Feature-Extraktion
  (`_flutter_base_features`, `_wow_track_features`, `_wf_cached_features`)
- `backend/core/phases/phase_12_wow_flutter_fix.py` —
  `_pyin_candidate_taus` + Einbau im pYIN-Fallback
- `tests/unit/test_perf_r18_cache_equivalence.py` — neu (Äquivalenz-Pinning;
  Write-Gate gilt nicht für `tests/`)
- `docs/TODOS_SOTA_ROADMAP.md` — §PERF-R18
- `CHANGELOG.md`, `README.md`, `pyproject.toml`, `backend/core/version.py`,
  `Aurik10/__init__.py` — Version 10.12.16
