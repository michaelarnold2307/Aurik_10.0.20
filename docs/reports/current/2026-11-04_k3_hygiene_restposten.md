# K3-Hygiene-Restposten — Abarbeitung 2026-11-04

**Auftrag:** Die offenen K3-Restposten im SOTA-Defizit-Register (`.github/SOTA_DEFICIT_REGISTER.md`)
schließen — strikt nach Register-Kriterien je Eintrag (Register-Regel: `geschlossen` nur mit
prüfbarem, existierendem Beleg-Pfad; kein Scheinschluss).

**Zweig:** `kun/k3-hygiene-restposten-owgg6v` · **Bearbeitet:** 12 offene K3-Einträge
(D-K3-8, 13, 21, 23, 31, 37, 38, 40, 62, 64, 65, 70 — die Auftragsliste ist damit vollständig;
D-K3-21 zählt laut Auftrag mit).

**Ergebnis in einem Satz:** 2 Einträge geschlossen (D-K3-8, D-K3-62 — beide mit existierenden,
inhaltlich geprüften Beleg-Pfaden), 10 Einträge offen **mit präzisiertem echten Blocker** —
alle 10 hängen an nicht-autonomen Entscheidungen (Hör-, Nutzer-, Spec- bzw. Korpus-A/B-Entscheidungen,
Aufgaben-Schritt 4) und wurden bewusst **nicht** geschlossen. Kein Code geändert, keine
klangverändernde Änderung, keine Flag-Flips, kein `docs/SOTA_HANDOVER.md`.

## Methodik und Verifikationsrahmen

- **Umgebungs-Hinweis (wichtig):** Diese Session verfügt über **kein Shell-/Execution-Tool**
  (nur Datei-Lese/-Edit- und Git-Lese-Tools). `pytest`, `ruff`, `scripts/sota_deficit_gate.py`
  und `git commit` konnten daher **nicht ausgeführt** werden. Verifiziert wurde, was in dieser
  Umgebung verifizierbar ist: **Existenz jedes zitierten Beleg-Pfades** (via `git ls-files`/Glob,
  Register-Regel „Beleg-Pfad muss existieren") und **Inhaltsprüfung** der Beleg-Dateien
  (Assertions der Vertragstests gelesen und mit den Register-Forderungen abgeglichen).
  Die exekutiven Nachweise (Testlauf, Gate-Lauf, Commit) sind unten als auszuführende Befehle dokumentiert.
- Keine neuen Code-Dateien ⇒ Write-Gate (`FILE_REGISTRY`) und Test-Pflicht greifen nicht;
  `ruff --select F821,F601,B009,I001` ist auf die geänderten Dateien anzuwenden (nur Markdown
  geändert → keine Python-Dateien in der Auswahl).
- Register-Struktur: 8-Spalten-Format, Status-Enum und Beleg-Pflicht unverändert; jede
  Status-Änderung auf `geschlossen` trägt existierende Beleg-Pfade (Gate-Logik `scripts/sota_deficit_gate.py`
  gelesen: (a) `geschlossen` ⇒ ≥ 1 existierender Backtick-Pfad in der Beleg-Spalte).

## Je Eintrag: Forderung → Handlung → Status/Grund

### D-K3-8 — 1269 layout-empfindliche Reduktionen ohne Layout-Entscheidung → **geschlossen**

- **Forderung (Register):** „Nächster Schritt: Einzelprüfung der Top-Module am Typ der
  durchgereichten Arrays — zuerst `stereo_authenticity_invariant.py`, `verify_import_export_guards.py`,
  `benchmark_stereo.py`"; Rest-Backlog per Entscheidung als L2-Bericht statt Gate.
- **Handlung:** Sichtung bestätigt: die Einzelprüfung der Top-3-Module ist 2026-10-25 erledigt
  (9 Treffer = False Positives) und in `tests/unit/test_dk38_layout_reductions_pin.py` gepinnt
  (Inhalt geprüft: 5 Vertragstests — MS-/LR-/Azimut-Korrelation layout-invariant (2, N) ≡ (N, 2),
  Mono-Input robust, channels-first-Vertrag von `scripts/benchmark_stereo.py`). Die
  L2-Entscheidung ist im Tool realisiert (`scripts/layout_invariant_check.py`:
  „Bericht/Triage-Backlog, kein Fail — Batch-Achse vs. Kanalachse ist statisch nicht entscheidbar").
- **Status/Grund:** `geschlossen`. Die eingeforderte Prüfung ist erfüllt und belegt; der
  Rest-Backlog ist **kein belegter Defekt**, sondern der dokumentierte Monitoring-Zähler
  (Lehre D-K3-5: Fail-Regel ohne Entscheidbarkeit = Lärm). Residual explizit benannt:
  die L2-Zahl nicht einzeln geprüfter Reduktionen bleibt als Bericht sichtbar.
- **Beleg:** `tests/unit/test_dk38_layout_reductions_pin.py`, `scripts/layout_invariant_check.py`.

### D-K3-62 — pre_echo-FP-Klasse auf Reparatur-Phasen → **geschlossen**

- **Forderung (Register):** „Duplikat-/Reparatur-Diskriminator in der Metrik (Korrelation Pre-Δ
  gegen Onset-Fenster) + Hörstichprobe vor jeder Verschärfung".
- **Handlung:** Sichtung bestätigt: der Diskriminator ist umgesetzt (10.12.27, report-only) und
  vertraglich gepinnt — `tests/unit/test_dk362_pre_echo_discriminator.py` (Inhalt geprüft:
  5 Vertragstests — codec-artige Onset-Kopie → `duplicate` (corr ≥ 0,30), lokale Füllung →
  `repair_fill` (corr < 0,30), kein Delta → `none`, Wrapper `pre_echo_ratio_db` bit-identisch (§G9),
  Witness-Report trägt `pre_echo_kind`/`pre_echo_corr` (§G8)). **Phantom-Beleg bereinigt:**
  die Beleg-Spalte zitierte `tmp_repro/pe_verify.py`, `tmp_repro/pe_worst.py`,
  `tmp_repro/witness_debug/` — diese Session-Artefakte existieren im Repo **nicht**
  (geprüft: kein `tmp_repro/`-Inhalt); die Verweise sind in Prosa-Schreibweise umgestellt,
  die Beleg-Spalte zitiert nur noch existierende Pfade.
- **Status/Grund:** `geschlossen`. Die Forderung ist erfüllt; die „Hörstichprobe vor jeder
  Verschärfung" ist keine ausstehende Arbeit, sondern die **dauerhafte Sperre** für eine
  künftige Verschärfung (Veto-Anschluss bleibt aus, §V7 (copilot-instructions.md)) — als
  Bedingung im Blocker und im Test-Docstring gepinnt.
- **Beleg:** `backend/core/dsp/pre_echo_model.py`, `backend/core/listening_witness.py`,
  `backend/core/unified_restorer_v3.py`, `tests/unit/test_dk362_pre_echo_discriminator.py`.

### D-K3-13 — 7 normative Tests rot (vorbestehend) → **offen, Blocker präzisiert**

- **Forderung (Register):** Ehrliche Ablöse des letzten roten Falls
  (`test_daily_gate_stored_recently`): UAT-Re-Run (Evidenz erneuern) **ODER** Test-Korrektur
  auf `latest.date` statt `generated_at`.
- **Handlung:** Ist-Stand gemessen/gesichtet: **beide** Ablösen sind zwischenzeitlich umgesetzt —
  (1) der Test prüft `latest.date` (Docstring + Assertion in
  `tests/normative/test_real_audio_restoration_quality_gate.py`, „Wirkung statt Formulierung",
  D-K3-34-Prinzip); (2) ein echter UAT-Lauf fand 2026-10-29 statt
  (`audit/uat_results_2026-10-29.json`; `audit/daily_real_audio_gate_status.json` trägt
  `latest.date` = 2026-10-29). Die Register-Notiz „aktuell `latest.date` 2026-05-11" war damit
  **stale** und ist korrigiert.
- **Status/Grund:** `offen` — echter Blocker: der Test misst die Evidenz-Frische (≤ 48 h) und
  ist zum Zeitpunkt dieser Welle wieder rot (UAT-Analyse 6 Tage alt). Das ist die gewollte
  Fail-closed-Wache (Kadenz), **kein** Struktur-Defekt. Frische Evidenz erfordert einen
  UAT-Re-Run (`audit/uat_report_generator.py` → `tests/test_uat_acceptance_criteria.py`,
  Real-Audio-Suite mit `--run-heavy-tests`, 180 s je Test, 1200 s gesamt) — bewusst **nicht**
  in dieser Welle gefahren (GPU-Kette läuft parallel, Manager-Vorgabe „keine Voll-Suites").
  Nebenbefund des Laufs 2026-10-29: R5–R12 stehen als ERROR (0/8) im Report —
  Harness-/Fixture-Fehler beim nächsten Lauf mit beheben. Sobald ein UAT-Lauf frische Evidenz
  liefert, ist der Test grün und der Eintrag schließbar.

### D-K3-21 — SNR-adaptive `CAUSE_PARAMS` inaktiv → **offen (nicht autonom, Schritt 4)**

- **Forderung:** Aktivierung erst nach A/B-Hörprobe auf eingefrorener Baseline + Faktor-Kalibrierung.
- **Handlung:** Einordnung dokumentiert (Register-Blocker präzisiert): Hör-Entscheidung, nicht
  autonom — die Aktivierung ist klangverändernd (Eingriffsstärke bis ×1,5, fünf Belege des
  Wohlklang-Vertrags). Gate `_SNR_ADAPTIVE_ENABLED` bleibt bewusst AUS (kein Flag-Flip).
- **Status/Grund:** `offen` — Blocker: Nutzer-Hörentscheidung.

### D-K3-23 — 327 Zeilen toter Code (RestaurierDenker) + Council-Duplikat → **offen (nicht autonom)**

- **Forderung:** Nutzer-/Spec-Entscheidung `decide()` implementieren **oder** entfernen;
  Council-Duplikat umleiten **oder** entfernen.
- **Handlung:** Einordnung dokumentiert + **neuer Befund**: der Phantom-Aufruf ist real
  verdrahtet — `backend/core/per_phase_musical_goals_gate.py` (Kommentar „§v10.6
  RESTAURIER-DENKER: Zentrale Entscheidungs-Intelligenz") ruft `_rd.decide(_ctx)` mit
  `# type: ignore[attr-defined]`, wertet `override_guard` als `regression × 0,3` und verschluckt
  den `AttributeError` in `except Exception: logger.debug` (stiller Fallback, §V6
  (copilot-instructions.md)) — dieselbe Defekt-Klasse wie D-K3-14. **Entscheidend:** ein
  implementiertes `decide()` würde über den Guard-Override klangwirksam werden ⇒
  „implementieren" ist eine klangverändernde Stufe (§V7 — kein Blind-Fix), „entfernen"
  kollidiert mit der Spec-11-Kanonizität („DIE EINE zentrale Entscheidung").
- **Status/Grund:** `offen` — Blocker: Spec-/Nutzer-Entscheidung; keine autonome Ablöse möglich.

### D-K3-31 — `snr_db` toter Parameter in `calibrate_phase_intensity` → **offen (nicht autonom)**

- **Forderung:** Nutzer-Entscheidung: (1) toten Parameter entfernen **oder** (2) belegte
  SNR-Senkung mit A/B-Hörprobe implementieren.
- **Handlung:** Einordnung dokumentiert: Variante (2) ist Hör-Entscheidung (klangverändernd),
  Variante (1) streicht den Vertragspunkt endgültig (API-/Test-Anpassung) — die Wahl ist nicht
  autonom. Der bestehende Vertragstest pinnt die belegte Lage (keine Eskalation +
  Risiko-Flaggung) und bleibt unangetastet.
- **Status/Grund:** `offen` — Blocker: Nutzer-Entscheidung (1) vs. (2).

### D-K3-37 — 32×-Guard in Standardpfaden nicht wirksam → **offen (nicht autonom)**

- **Forderung:** (a) Enforcement als Default (klangwirksam: Abbruch/Adaptieren ändert das
  Ergebnis) **oder** (b) Norm ehrlich als Monitoring-Ziel mit **benanntem Eigentümer** deklarieren.
- **Handlung:** Einordnung dokumentiert + Nebenbefund: `backend/core/performance_guard.py`
  (Docstring `PerformanceGuard`) formuliert „enforced … Garantiert: Max 32x RT" **bedingungslos**
  — bei Umsetzung von (b) oder im Entscheidungszug um die `enforce_limit`-Bedingung
  (Default AUS, `enforce_3x_rt` in `backend/core/unified_restorer_v3.py`) präzisieren.
- **Status/Grund:** `offen` — Blocker: (a) klangwirksam, (b) erfordert einen benannten
  Eigentümer (nicht erfindbar, §V7) — nicht autonom.

### D-K3-38 — Defekt-Scanner 35× über Per-Minute-Budget → **offen (nicht autonom)**

- **Forderung:** Budget-Lücke schließen (Limit 4 s/min vs. gemessene 35,3 s/min).
- **Handlung:** Einordnung dokumentiert: die verbleibenden Treiber sind an Entscheidungen
  gebunden — `estimate_residuum_salience` (Batch-Variante = **Hör-Entscheid**, mittlerer
  Salienz-Abstand 0,20) und der Konvolutions-/FFT-Ersatz (gemessen **nicht bit-identisch**,
  Äquivalenz-/Hör-Entscheidung); weitere Teilbehandlungen nur mit Bit-/Äquivalenz-Nachweis
  (§G5 (copilot-instructions.md)), keine verhaltensändernden Blind-Fixes (§V7).
- **Status/Grund:** `offen` — Blocker: Hör-/Äquivalenz-Entscheidungen; die bereits umgesetzten
  Teilbehandlungen (RTS-Rekursion, numba-Kalman, Flutter-Bündelung, `_uniform_smooth`) bleiben
  dokumentiert (Belege in der Register-Evidenz-Zelle).

### D-K3-40 — Scanner-Kosten zerlegt, Negativ-Befund → **offen (nicht autonom)**

- **Forderung:** „Nächste Schritte (Entscheidung nötig)": Per-Kanal-Block begrenzen,
  Material-Features wiederverwenden, §SR-CG8-Scan auf den Post-Scan-Mix ziehen.
- **Handlung:** Einordnung dokumentiert: alle drei Optionen ändern Verhalten bzw. sichtbare
  Metadaten (`channel_locations` wird in `Aurik10/ui/modern_window.py` angezeigt) und brauchen
  Korpus-A/B (§V7); ein verhaltensneutraler Gewinn ist gemessen **nicht auffindbar** (beide
  Versuche zurückgenommen: numpy-rfft-Hilbert in situ 1,7× langsamer; Detektor-Memo ohne
  Zustand ändert 3 Typen, mit Zustand 0 Treffer).
- **Status/Grund:** `offen` — Blocker: Korpus-A/B = Hör-/Daten-Entscheidung.

### D-K3-64 — Dither-Kaskade restore × Export → **offen (nicht autonom)**

- **Forderung:** Hörstichprobe + Entscheidung, wer der EINE Dither ist (restore-Vorstufe vs.
  Export-final) — bis dahin kein Blind-Fix (§V7).
- **Handlung:** Einordnung dokumentiert + Nebenbefund: die Ketten-Doku (§IV Export-Reihenfolge)
  nennt als Export-Dither nur `apply_dither`, obwohl restore-seitig `apply_final_polish`
  (`backend/core/dsp/final_polish.py`) zusätzlich dithert — Dokumentations-Abgleich im
  Entscheidungszug mitziehen.
- **Status/Grund:** `offen` — Blocker: Hör-/Entscheidungs-Blocker.

### D-K3-65 — ARC-/Goosebumps-/MDEM-Blöcke laufen auf jedem Chunk → **offen (nicht autonom)**

- **Forderung:** (a) Bewertungs-Blöcke nach §P0-1-Muster deferren, (b) audio-wirkende Pfade
  erst nach Hörstichprobe, (c) Regressionstest.
- **Handlung:** Code-Sichtung (UV3, ARC-Bewertung ab Z. 17725, Goosebumps-Wahl ab Z. 17945,
  MDEM-Messung/Morphing ab Z. 18276, ARC-Makro-Korrektur ab Z. 18383) + Einordnung dokumentiert:
  **Kopplungs-Befund** — die „Bewertungs-Blöcke" speisen die audio-wirkenden Pfade (die
  Goosebumps-Wahl liest den Checker-Wert, die ARC-Korrektur hängt an der ARC-Bewertung, das
  MDEM-Morphing an der MDEM-Messung). Ein Deferral von (a) ohne (b) würde auf Nicht-letzt-Chunks
  die Eingaben der audio-wirkenden Pfade entfernen = Verhaltensänderung ⇒ Blind-Fix verboten
  (§V7), die Fix-Spezifikation (a)–(c) bleibt unverändert maßgeblich.
- **Status/Grund:** `offen` — Blocker: Hörstichprobe vor Umsetzung; Umsetzung erst nach (b),
  dann inkl. Regressionstest (c).

### D-K3-70 — Golden-Gate rot: Metrik-Regression + Manifest-Annotation → **offen (nicht autonom)**

- **Forderung:** (A) produktseitige Reasoner-/Detektor-Kalibrierung und (B)
  Manifest-`accepted_causes` gegen die Ground-Truth korrigieren — erst danach ehrlich grün.
- **Handlung:** Einordnung dokumentiert: (A) ist klangwirksame Produkt-Kalibrierung
  (Phasen-/Ursachen-Routing) und braucht Nutzerentscheidung + Korpus-A/B; (B) ändert
  Golden-Expectations und ist per D-K3-28-Prinzip („Erwartung nur mit gemessenem Gegenbefund
  am Fixture") bewusst an (A) gekoppelt — bewusst **keine** Expectations-Relaxierung. Die
  injizierten Ground-Truth-Ursachen des Aurik-Corpus-Generators (hiss, wow, flutter in den
  Synthesefällen) sind als Entscheidungsgrundlage notiert, aber nicht ohne (A) umgesetzt.
  Hinweis: `test_audio/` ist in diesem Worktree nicht vorhanden (gitignored) — der Gate-Test
  skippt hier; die Prüfung ist korpusgebunden.
- **Status/Grund:** `offen` — Blocker: Nutzerentscheidung (A) + gekoppelte Umsetzung (B).

## Verifikation — was geprüft wurde und was nicht

**Geprüft (in dieser Session):**

- Vollständigkeit: alle 12 offenen K3-Zeilen (Status ≠ `geschlossen`/`bewusst-akzeptiert`)
  erfasst und abgearbeitet; keine weitere offene K3-Zeile übersehen (K3-Block Zeile 154–228
  vollständig gesichtet).
- Beleg-Pfad-Existenz aller in den geänderten Zeilen zitierten Pfade (`git ls-files`/Glob) —
  inkl. Bereinigung der drei Phantom-Pfade `tmp_repro/*` in D-K3-62.
- Inhaltsprüfung der beiden Schließungs-Belege (`tests/unit/test_dk38_layout_reductions_pin.py`,
  `tests/unit/test_dk362_pre_echo_discriminator.py`) — Assertions decken die Register-Forderungen.
- Register-Struktur nach jedem Edit (8 Spalten, Status-Enum, keine rohen `|` in Zellen);
  Gate-Logik von `scripts/sota_deficit_gate.py` gegen die Änderungen gegengelesen.

**Nicht ausführbar in dieser Session (kein Shell-Tool) — vom Auftraggeber/Manager auszuführen:**

```bash
# 1) Beleg-Tests der Schließungen (fokussiert, erwartet grün):
nice -n 19 python -m pytest tests/unit/test_dk38_layout_reductions_pin.py \
    tests/unit/test_dk362_pre_echo_discriminator.py -q

# 2) Register-Gate (erwartet Exit 0):
python scripts/sota_deficit_gate.py

# 3) Ruff auf geänderte Dateien (nur Markdown geändert → leerer Lauf):
ruff --select F821,F601,B009,I001 .github/SOTA_DEFICIT_REGISTER.md docs/reports/current/2026-11-04_k3_hygiene_restposten.md

# 4) Conventional Commit (vorbereitet, siehe unten):
git add .github/SOTA_DEFICIT_REGISTER.md docs/reports/current/2026-11-04_k3_hygiene_restposten.md
git commit -m "docs(register): K3-Restposten 2026-11-04 — D-K3-8/D-K3-62 geschlossen, 10 Blocker präzisiert"
```

Zusatz-Evidenz für D-K3-13 (nur wenn die GPU-Kette es zulässt, ~20 min):
`python audit/uat_report_generator.py` erneuert die UAT-Evidenz echt (kein Zeitstempel-Kosmetik).

## Resümee für den Manager (§2-Fortschreibung)

1. **Schließbar waren nur die Einträge, deren eingeforderte Behandlung bereits erledigt war**
   (D-K3-8: Top-Module geprüft/gepinnt; D-K3-62: Diskriminator umgesetzt/getestet) — beide
   Status-Wechsel tragen existierende Belege, D-K3-62 zusätzlich die Phantom-Beleg-Bereinigung.
2. **Die übrigen 10 sind „Hygiene/Phantome/Doku" nach Art des Befunds, aber nicht autonom
   abließbar:** jede verbleibende Forderung kippt in eine Hör-, Nutzer-, Spec- oder
   Korpus-A/B-Entscheidung (Schritt 4 der Aufgabe). Die Blocker sind je Eintrag präzisiert
   (inkl. zwei neuer Befunde: PMGG-Phantom-Aufrufer bei D-K3-23, Kopplung Bewertung/Audio bei
   D-K3-65; Nebenbefunde: `PerformanceGuard`-Docstring-Garantie bei D-K3-37, §IV-Doku-Drift
   bei D-K3-64, UAT-R5–R12-ERROR bei D-K3-13).
3. **Kein Scheinschluss:** D-K3-13 bleibt offen, obwohl beide Register-Ablösen erledigt sind —
   der Frische-Test ist real rot (UAT-Evidenz 6 Tage alt) und braucht einen echten UAT-Lauf.
4. **Keine klangverändernden Änderungen, keine Flag-Flips, kein `docs/SOTA_HANDOVER.md`** —
   Hygiene-Pflicht eingehalten; geändert wurden ausschließlich
   `.github/SOTA_DEFICIT_REGISTER.md` und dieser Report.
