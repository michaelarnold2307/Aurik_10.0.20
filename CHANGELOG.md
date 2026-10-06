# Changelog — Aurik 10.3.11

## 10.3.11 (2026-10-05/06)

### §Gesangs-Fokus: SGMSE+-Sprach-Core von Gesang entkoppelt

- **Befund (2026-10-06):** Zwei Stellen aktivierten das SPRACH-trainierte SGMSE+
  (Score-Core, WSJ0-CHiME3, Richter et al. 2022) auf Musik/Gesang, obwohl das
  Musik-Finetune gesperrt ist (`music_model_flags.use_sgmse_musik=False`,
  §v10.16/F7): `coordinated_repair` schaltete es gezielt bei
  `vocal_confidence > 0.5` frei (also auf Gesang), `phase_03_denoise` prüfte das
  Flag gar nicht. Folge: ein gesperrtes Sprach-Modell bearbeitete still Gesang
  (§V1-Risiko: Telefonband-Klangfarbe/Vokalfärbung) und kostete 6,2 s je 1 s
  Audio (R-09: 26× isoliert kalt, 6,8× warm — RT-Budget §9.5 verletzt), während
  der Decorrelation-Guard das Ergebnis verwarf.
- **Fix:** Beide Aktivierungspfade an `use_sgmse_musik` gekoppelt
  (`phase_03_denoise._sgmse_eligible` und der RepairPlanner in
  `coordinated_repair`). Ohne Musik-/Gesangs-Finetune läuft SGMSE+ nicht; die NR
  trägt DFN-Musik (`use_df_musik=True`), für Gesang ist Cantus der designierte
  Kern nach Freigabe.
- **Beweise:** `test_ml_hybrid_regression.py -k R09_rt_budget` → 11 passed
  (Phase-03-Kaltstart-/SGMSE-Anteil entfällt); `test_model_zoo_activation.py`
  → 16 passed (neuer Test pinnt die Sperre auf Gesang); Ruff clean.

### train_cantus: Smoke-Läufe überschreiben keine Produktionsartefakte mehr

- **Befund (2026-10-06, selbst verursacht):** `scripts/train_cantus.py --smoke`
  schrieb `checkpoint_latest.pt`/`checkpoint_best.pt` und
  `train_report_<phase>.json` nach `models/cantus/` — der Validierungslauf
  überschrieb die 204-Mio-Parameter-Pretrain-Checkpoints (2,45 GB) und die
  Pretrain-Evidenz. Die Gewichte bleiben nur als ONNX-Inferenzgraph
  (`cantus_dit.onnx`) erhalten; eine Rückübertragung ist wegen ONNX-
  Konstantenfaltung unvollständig (92/~377 Keys fehlen) → Pretrain muss neu
  laufen.
- **Fix:** `_resolve_checkpoint_dir(smoke)` isoliert Smoke-Läufe in ein
  `tempfile.mkdtemp`-Verzeichnis; Produktionsläufe bleiben unverändert. Der Pfad
  ist durch einen Regressionstest abgesichert
  (`test_smoke_checkpoint_dir_never_touches_production`).
- **Beweise:** Regressionstest grün; Smoke-Nachweis: Report landet in
  `/tmp/cantus_smoke_*/`, `models/cantus/` bleibt unverändert; Ruff clean.

### Backup-/Guard-Konzept für Trainingsartefakte

- **Neu:** `backend/core/training_artifacts.py` — zentraler Schutz, den alle
  Trainingsskripte adoptieren können (`torch.save` → `save_guarded`,
  `path.write_text` → `write_text_guarded`):
  1. **Rotierendes Backup** vor jedem Überschreiben (Retention, Standard 3) —
     der Vorfall vom 2026-10-06 wäre damit folgenlos geblieben.
  2. **Atomares Schreiben** (tmp + `os.replace`) — keine partiellen Dateien.
  3. **Schrumpf-Warnung** (§V6 (copilot-instructions.md)) — Preset-/Modellwechsel
     oder ein Smoke-Lauf auf dem Vollmodell werden sichtbar.
- **Adoptiert:** `scripts/train_cantus.py` (Checkpoints + Report). Die übrigen
  ~23 Trainingsskripte nutzen weiterhin direktes `torch.save`; die Migration ist
  mechanisch (Ein-Zeilen-Ersatz) und als Folgeschritt vorgesehen.
- **Beweise:** `tests/unit/test_training_artifacts.py` → 7 passed; guarded Smoke
  schreibt in `/tmp/cantus_smoke_*/`, `models/cantus/` unverändert; Ruff clean.

### Phase 24: Dropout-Erkennung repariert und Autonomie nach §G188 hergestellt

- **Befund (Vollscan Chunk 37/97 — dieser eine Test blockierte das Restprogramm
  37–97):** `TestPhase24DropoutRepairRegression::test_dropout_gap_filled` blieb rot:
  ein 100-ms-Nullblock in einem 440-Hz-Ton wurde nicht gefüllt. Zwei getrennte
  Ursachen, beide gemessen:
  1. **Detektor numerisch blind.** `_detect_amplitude_dropouts` bildete die lokale
     Referenz mit `savgol_filter` über ein 100-ms-Fenster — genau die Lückenlänge.
     In der Lückenmitte gemessen: Referenz **−1,142e-03** (negativ!), daraus
     Schwelle −2,283e-04 → die Bedingung `envelope(0) < Schwelle` ist **nie** wahr.
     Die Empfindlichkeit sank mit der Defektlänge — genau umgekehrt zur Aufgabe
     der Phase.
  2. **Gate verweigerte die Arbeit.** Ohne externen `dropout_density`/
     `dropout_severity` und ohne Defekt-Evidenz blieb die Dichte 0,0 → Skip → die
     Phase war ein No-op, obwohl sie einen vollständigen eigenen Detektor besitzt.
- **Fix:**
  1. Referenz auf **Peak-Hold** (`maximum_filter1d`) über dasselbe Fenster
     umgestellt — Wert in der Lückenmitte **+2,039e-01**, Schwelle 4,078e-02 →
     Lücke wird erkannt. Die relative Schwelle blieb unverändert.
  2. §G188 (GEBOTE.md) „Autonome Stärke-Einstellung": Fehlt externe Evidenz, misst
     die Phase ihre Dichte **selbst** (`_detect_amplitude_dropouts`) statt die
     Arbeit zu verweigern. Das hängende Zitat „§v10.96" (in `.github/` und `docs/`
     nicht auffindbar) ist durch die tragende Regel ersetzt.
- **Beweise:** Detektor direkt gemessen — `detektor_regionen=1`, Region
  `(12013, 16788)` bei echter Lücke `(12000, 16800)` (Abweichung ≤ 13 Samples);
  Gegenprobe leise Passage (×0,15) → **0** Regionen (keine Überreparatur).
  Der Regressionstest ist grün; `test_ml_hybrid_regression.py --run-heavy-tests`
  → **128 passed** (vorher 127, Dropout-Test rot). Die zwei verbleibenden
  `test_R09_rt_budget`-Fehlschläge (phase_03_denoise, phase_23_spectral_repair)
  sind laufzeitabhängig und traten vor wie nach der Änderung **identisch** auf.
  Ruff clean. Version 10.3.11 nach §v10.802 konsistent.

## 10.3.10 (2026-10-05)

### Phasensignatur: De-Esser wieder vertragskonform

- **Befund (Vollscan Chunk 37/97: 12 failed, 301 passed):** Acht Fehler des Typs
  `TypeError: DeEsserPhase.process() missing 1 required positional argument` in
  `test_ml_hybrid_regression.py` waren kein Testproblem. Der Phasen-Vertrag
  `phase_interface.py:383` lautet
  `process(self, audio, sample_rate=48000, material_type="unknown", **kwargs)`;
  `DeEsserPhase.process` verlangte `material_type` dagegen **positional ohne
  Default**. Damit war die Phase nicht mehr austauschbar — jeder generische
  Aufrufer, der den Vertrag nutzt (Phasen-Registry, Vorschau, Test-Harness),
  bricht. Der vorhandene `# type: ignore[override]` hat die Abweichung nur
  markiert statt sie zu beheben.
- **Fix:** Signatur an den Vertrag angeglichen (`sample_rate=48000`,
  `material_type` mit Default) und `material_type` einmal an der Quelle auf
  `MaterialType` normalisiert. Der Rumpf nutzt `material.name` an sechs Stellen
  und der Vertrag erlaubt ausdrücklich `str` — die Normalisierung erhält die
  Toleranz beider Seiten, statt an sechs Stellen einen Ersatzwert zu raten.
  Die überflüssig gewordene Override-Ausnahme ist entfernt.
- **Beweise:** `-k de_esser --run-heavy-tests` → 11 passed in 13,94 s (vorher
  acht TypeError-Fehler); mypy auf der Datei: 0 union-attr-Fehler; ruff clean.
- **Offen (Fund 12b):**
  `TestPhase24DropoutRepairRegression::test_dropout_gap_filled` erwartet, dass
  der DSP-Ersatzpfad eine 100-ms-Lücke zu ≥ 5 % füllt; die Lücke bleibt stumm.
  Eigener Untersuchungspunkt (nächster Block).

## 10.3.9 (2026-10-05)

### §III.9: ONNX bleibt der CPU-Fallback — kein stiller Wechsel auf das TS-Artefakt

- **Befund (Vollscan-Fund 8):** `SGMSEPlusPlugin._try_load` fiel bei einem
  fehlgeschlagenen GPU-EP direkt auf `sgmse_plus.ts` zurück. Dieses Artefakt trägt
  nach eigener Dokumentation des Export-Skripts **andere Gewichte** als der
  ONNX-Core („alle 647 Parameter weichen ab, beobachtete Ausgabeabweichung
  rel ~3e-1", `scripts/export_sgmse_onnx.py`). Der Rückfall hätte damit still ein
  anderes Modell verwendet — in einem Plugin, dessen Standardpfad ONNX ist.
- **Fix:** Scheitert der Session-Aufbau mit GPU-EPs, wird zuerst derselbe
  ONNX-Core mit `CPUExecutionProvider` geladen (paritätsexakt; „ONNX bleibt reiner
  CPU-Fallback", §III.9 (copilot-instructions.md)) und der Vorgang nach
  §V6 (VERBOTEN.md) gemeldet. Erst wenn auch ONNX-CPU scheitert, greift der
  dokumentiert abweichende TorchScript-Pfad — dessen Warnung benennt die
  Gewichtsabweichung jetzt ausdrücklich.
- **Beweise:** simulierte GPU-EP-Blockade → `EP-Aufrufe:
  [['MIGraphXExecutionProvider'], ['CPUExecutionProvider']]`, Warnung mit
  §III.9-Bezug, `Session-EPs: ['CPUExecutionProvider']`, **`TS genutzt: False`**;
  Normalpfad lädt ONNX wie zuvor; ruff (F821/F601/B009/I001) clean.

## 10.3.8 (2026-10-05)

### Rollback-Guard nach Resume wieder scharf (Selbstprüfung des Divergenzschutzes)

- **Befund:** Der in 10.3.6 eingeführte Divergenzschutz schreibt beim Rollback die
  Gewichte des Best-Checkpoints nach `checkpoint_latest.ckpt` — dort stand aber der
  **divergierte** Val-Wert (`epoch=84, val_loss=13,7178` bei korrekten Best-Gewichten).
  Der Resume-Pfad setzt `best_val` aus genau diesem Feld, ein Resume hätte den
  Rollback-Guard damit **entwaffnet** (Schwelle gegen 13,7 statt gegen 6,1525).
- **Fix:** Das Feld trägt jetzt den Best-Referenzwert (`best_val`) statt `avg_val`,
  mit Begründung im Code. Der **bestehende** `checkpoint_latest.ckpt` wurde
  in-place nachgezogen (Gewichte unverändert, nur Metadaten; Tensoren verifiziert
  identisch).
- **Beweise:** `val_loss` 13,717787525672465 → 6,152494320198894 bei unveränderter
  Tensor-Topologie; Ruff (F821/F601/B009/I001) clean; `py_compile` grün.

## 10.3.7 (2026-10-05)

### Kanonische Zwicker-Loudness-Signatur wiederhergestellt (§4.1b)

- **Befund (Vollscan Chunk 10/97, acht Tests rot):** `compute_specific_loudness_zwicker()`
  lieferte den strukturierten `ZwickerLoudnessResult` zurück, während §4.1b
  (04_dsp_standards) wortwörtlich die Signatur
  `compute_specific_loudness_zwicker(audio, sr) -> float` fordert und die Regel
  "Lautheitsmessung ohne ISO 532-1" (§74 (VERBOTEN.md)) dieselbe Referenz für das
  ΔN > 2,0-sone-Gate nennt. Zusätzlich behandelte die Funktion Signale unter 100 ms
  wie messbaren Stoff (10,97 sone für 2 ms Audio), obwohl dasselbe Modul an anderer
  Stelle 100 ms als Mindestlänge führt.
- **Fix:** Die kanonische Funktion liefert wieder `float`; die vollständige Messung
  (Bandpegel, Phon, ΔN) liegt in der neuen, benannten
  `compute_specific_loudness_zwicker_detailed()`. Interne Nutzer (ΔN-Gate,
  Bandpegel-Extraktion) verwenden die Detail-API. Signale unter 100 ms liefern
  0,0 sone mit `logger.warning` (§V6 (VERBOTEN.md)) statt eines stillen Werts.
- **Beweise:** 76 Tests grün (Norm-Suite + Literatur + Zwicker/P5), Ruff clean;
  Vertragsmessung: 100 Samples → 0.0 (mit Warnung), 100 ms → 10,9711 sone,
  Monotonie 46,19 → 136,34 sone bei +14 dB.

## 10.3.6 (2026-10-05)

### Divergenzschutz im SGMSE-Finetune (Trainings-Befund)

- **Befund:** Der laufende Feintune `scripts/train_sgmse_musik.py` divergierte ab
  Epoche 111 (Train-Loss 5,44 → 22,6 → 35,3; Val 6,1525 → 11,13). Ursache ist die
  schwer ausläuferbehaftete SG-MSE-Verlustverteilung (dokumentierter Altbefund:
  Ep 58 mit 266728 bei Median 260) in Kombination mit fehlendem Schutz gegen
  nicht-finite Verluste. Das Skript hatte nur Gradient-Clipping (Norm 2,0) und
  Best-Checkpoint — aber keinen Rollback und keine Meldung.
- **Fix:** (1) Nicht-finiter Loss wird erkannt, der Batch verworfen und per
  `logger.warning()` gemeldet (keine stille Degradation). (2) Divergenzschutz:
  steigt der deterministische Val-Wert über Faktor 1,5 gegenüber dem Beststand,
  wird auf das Best-Checkpoint zurückgerollt und `checkpoint_latest` konsistent
  mitgeschrieben; nach drei Rollbacks endet der Lauf geordnet. Das beste Modell
  bleibt in jedem Fall erhalten.
- **Betrieb:** Das divergierte Training wurde gestoppt und vom Best-Checkpoint
  (Epoche 80, Val 6,1525) mit lr 1e-5 neu gestartet (`--resume`).
- **Beweise:** `ruff` (F821/F601/B009/I001) clean, `py_compile` grün,
  Best-Checkpoint geladen (Epoche 80, 647 Tensoren); Neustart-Log
  `output/train_sgmse_musik2.log`.

## 10.3.5 (2026-10-05)

### 🐛 Versions-Drift behoben und Versions-Gate geschlossen

- **Befund:** `backend/core/version.py` (Single Source of Truth) stand auf 10.3.4,
  aber `pyproject.toml` auf **10.3.0** und `README.md` auf **10.2.0** — zwei
  Patch-Stände Drift, von keinem Gate bemerkt. Das vorhandene Prüfskript
  `scripts/check_version_consistency.py` war **nicht verdrahtet**, hielt
  `pyproject.toml` für kanonisch und hatte zudem **keine** `--fix`-Behandlung —
  es verwies auf ein Flag, das es nicht gab (Sackgasse).
- **Fix:** Das Skript liest die kanonische Version jetzt aus
  `backend/core/version.py`, unterstützt `--fix` korrekt (pyproject/README: erstes
  Vorkommen; CHANGELOG: die erste `##`-Abschnittsüberschrift — Historie bleibt
  unverändert) und ist als Pre-Commit-Hook `aurik-version-consistency`
  verdrahtet. Alle Stände auf 10.3.5 gezogen.
- **Nebenfix:** `.github/FILE_REGISTRY.md` führte `models/symphonia/**init**.py`
  (Fett-Marker im Dateinamen — die Datei „fehlte“) → echter Pfad
  `models/symphonia/__init__.py`.

## 10.3.4 (2026-10-05)

### 🐛 Shutdown-Hang behoben (memmap_pool-atexit)

- **Befund:** Nach einem Lauf mit aktivem ROCm-Pfad konnte der Prozess nach der
  Test-Summary („1 passed“) in `futex_do_wait` **hängen** bzw. mit Exit 1 enden
  (CLI/GUI-Paritätstest, 2026-10-05). Ursache: Der atexit-Pfad
  `backend/core/memmap_pool.py::MemmapPool.close()` nahm ein reguläres
  `threading.Lock`; am Interpreter-Ende werden Daemon-Threads an beliebiger
  Stelle beendet — stirbt einer im kritischen Abschnitt, blockiert
  `with self._lock` für immer.
- **Fix:** `close()` nutzt ein begrenztes `acquire(timeout=2.0)` und räumt ohne
  Lock auf, wenn es nicht greift (`_evict_file` ist idempotent, der Exit-Pfad
  läuft single-threaded).
- **Beweis:** Verhaltenstest — gehaltenes Lock ⇒ `close()` kehrt nach **2,00 s**
  zurück (vorher unbegrenzt); zweiter Aufruf idempotent 0,000 s; py_compile +
  Ruff-kritisch clean.
- **Offen (bewusst):** permanenter Regressionstest für `MemmapPool.close()` —
  das Modul hatte noch keine Testdatei, Neuanlage erfordert den Write-Gate-
  Eintrag.

## 10.3.3 (2026-10-05)

### 🐛 Crash-Bug, GPU-Probe, CI-Gates und Mess-Telemetrie

- **ROCm-ONNX-Probe crash-isoliert** (§III.9 (copilot-instructions.md)): native
  MIOpen/HIP-Aborts (SIGSEGV rc=-11, `Hip error: 'out of memory'`) beendeten den
  Host-Prozess (Unit-Smoke Exit 139 in `InferenceSession.__init__`). Die
  Session-Erzeugung läuft jetzt in einem eigenen Interpreter; Nicht-CPU-EPs nur
  mit Paritätsnachweis (rel ≤ 1e-3) gegen ONNX-CPU, MIGraphX nur mit eigener
  Parität, jeder Ersatzpfad mit §V6 (copilot-instructions.md) Warnung + Begründung,
  Ergebnis per Datei-IPC statt `print` (R01).
- **CI `type-gate` grün**: 9 mypy-Fehler in 5 Dateien behoben — u. a. sechsmal
  Optional-Indizierung von `load_audio_file()`, `__len__` gab ein 0-d-Array
  statt `int` zurück, `quality_ok() -> bool` verpackte das Ergebnis in
  `np.asarray`.
- **CI `export-guard` grün**: der Guard rief eine nicht mehr existierende CLI
  (`core/audio_exporter.py --sample-rate … --format …`) auf — er exportiert
  jetzt über `backend.core.audio_exporter.AudioExporter`, konfigurationsgetrieben
  aus `metadata.yaml`, Ziel `output/export_guard/`.
- **CI-Kollektion ohne torch/onnxruntime**: vier Testmodule importierten schwere
  Abhängigkeiten auf Modulebene → `pytest.importorskip` (kanonisches Muster);
  der Job brach vorher mit „3 errors during collection" ab.
- **ID-Kollision V74 aufgelöst**: `signal.lfilter` im Bell-EQ heißt jetzt V76
  (Spec > Code); V74 bleibt Silent-Except und wird vom AST-Guard
  `scripts/pre_commit_static_guard.py` erzwungen.
- **Tote Spec-Verweise** `§v10.304.30`/`.31` (8 Stellen) auf §III.9 umgebogen.
- **Testdeterminismus**: Commit-Pfad-Smoke mit `AURIK_FORCE_CPU=1` (lastunabhängig),
  VRAM-Tests injizieren den Live-Query-Wert, neue Regel 6 in
  `tests.instructions.md` („keine Live-Hardware-Werte als Erwartungswert").
- **Mess-Telemetrie**: deterministisches Validierungsprotokoll im
  SGMSE-Finetuning (fixer Seed, CPU-/CUDA-RNG restauriert → Val-Kurve lesbar,
  Trainingsverlauf bit-identisch) und Heartbeat im T6-1-A/B-Runner
  (5-min-Takt plus Start-/Ende-Zeile je Variante statt 4-h-Stille).
- **DX**: Safe-Runner meldet bei Lock-Kollision den haltenden PID.

## 10.3.2 (2026-10-05)

### 🐛 GUI-Versions-Konsistenz (§v10.802)

- `Aurik10/__init__.py` trug mit `_FALLBACK_VERSION = "10.2.1"` eine zweite,
  still veraltete Versionsnummer (Produkt war bereits 10.3.1): Bei
  fehlgeschlagenem Bridge-Import hätte die GUI eine falsche Version angezeigt.
  Der Fallback steht jetzt auf dem Stand der Single Source of Truth.
- `scripts/version_guard.py` prüft den Gleichlauf `_FALLBACK_VERSION` ↔
  `backend/core/version.py` und warnt bei Drift (§v10.802 GUI-Sync-Pflicht).
- `tests/unit/test_version_checker_and_ux.py` erzwingt den Gleichlauf
  fail-closed, damit die Nummer nicht erneut unbemerkt driftet.

### 🎚️ Zero-Phase im Sub-Bass-Add-Pfad + H03-Detektor spec-scharf

- **H03-Detektor präzisiert** (`.agents/skills/bug-prevention/scan_anti_patterns.py`):
  Die Regel meldete jedes `sosfilt(` in `phases/` und `dsp/`. Die Norm
  (`.github/VERBOTEN.md`, Anti-Pattern-Tabelle) verlangt zero-phase aber nur, „wo
  Bandfilter-Ergebnis auf Originalsignal addiert wird; `sosfilt` nur für
  Analyse/Sidechain“. Der Detektor prüft jetzt genau diese Bedingung (Addition/Mix
  im 30-Zeilen-Fenster); Analyse-Envelopes und serielle Filterketten werden
  transparent als `H03-Analyse-Skip` gezählt statt als Bug gemeldet. Wirkung:
  222 → 44 Findings (`H03=9, H04=2, H05=15, H07=3 | H03-Analyse-Skip=177`) —
  erstmals actionierbar. Zusätzlich zeigt der Lauf eine Klassen-Zusammenfassung
  auf WARNING-Ebene (die Detailzeilen waren vorher unsichtbar).
- **Zweite Präzisierungsstufe** (Falsifikation an `piano_restoration.py`): Die Norm
  unterscheidet Dry+Band-Add (zero-phase-Pflicht) von **Crossover-Split-Sum**
  (komplementäre Bänder untereinander, erlaubt bei gleichem Filtertyp — so auch
  §v10.1013). Der Detektor prüft das jetzt über die filter-abgeleiteten Namen
  (`_filter_derived_names`, 2 Ableitungsstufen, O(Zeilen)). Ergebnis der
  Stichprobe: `bass_enhancement:119` und `piano_restoration:147` korrekt
  eingeordnet; die 16 `piano_restoration`-Meldungen waren maskenbasierte
  Band-Reduktion plus Crossover — normkonform, keine Verstöße.
- **Dritte Stufe (Partnerprüfung):** Nur der unmittelbare Additionspartner
  entscheidet; Epsilon-/Messkontext (`+ 1e-10`) und `np.stack`-Container liefern
  keinen Partner → 70→44. Diese Stufe hatte zwischenzeitlich **echte** Verstöße
  verschluckt (`bass_enhancement:418`), weil sie gegen den exakten Zielnamen statt
  gegen die gematchte Variable (`harmonics_enhanced`) verglich — behoben und per
  Testgegenprobe abgesichert.
- **Erste belegte H03-Behebung** (`backend/core/dsp/bass_enhancement.py`): Der
  Sub-Bass-Bandpfad (20–60 Hz) läuft jetzt über den kanonischen, depth-adaptiven
  `safe_sosfiltfilt` statt über rohes `sosfilt`. Messung (48 kHz, 4. Ordnung):
  kausaler Filter verschiebt das Band um **+25,90 ms** (Envelope-Asymmetrie 1,000),
  zero-phase **0,00 ms** (0,003) — beim Add-back genau die im Antipattern
  dokumentierte destruktive Interferenz. Bandenergie praktisch unverändert
  (RMS 0,20985 vs. 0,20967), Summensignal ohne Auslöschung (RMS 0,351 → 0,427),
  4 Modultests grün.

## 10.3.1 (2026-10-05)

### 🐛 Bug-Hunt & Konsistenz (1:1-Abgleich gegen Vorgaben und Specs)

- **Zeitquellen-Konsistenz**: Der Pipeline-Health-Monitor startete den Wall-Time-Akkumulator
  mit `time.time()`, verglich aber mit `time.monotonic()` — die Differenz (≈ −1,76e9 s) machte
  das 2-Stunden-Limit dauerhaft inoperativ, und `summary()` meldete eine unsinnige
  Pipeline-Dauer (dokumentierte Klasse `.github/VERBOTEN.md` „Wall-Time-Referenz-Mismatch").
  Beide Seiten laufen jetzt auf `time.monotonic()`; Regressionstests in
  `tests/unit/test_pipeline_health.py`.
- **ArtistFingerprintStore**: `last_updated` wurde teils als `time.monotonic()` persistiert,
  aber gegen `time.time()` geprüft (und umgekehrt) — die Ablauf-Prüfung (> 30 Tage) konnte nie
  greifen. Persistierte Zeitstempel laufen jetzt durchgängig auf Wall-Clock (§G5).
- **PMGG-Retry-Budget**: `_retry_t0`/`_retry_elapsed` auf `time.monotonic()` umgestellt
  (`.github/specs/04_dsp_standards.md`: beide Seiten derselbe Uhrentyp); NTP-/DST-Sprünge
  konnten den Wiederholungsabbruch vorher nichtdeterministisch auslösen.
- **Batch-Timing**: Dauer-Messungen in `backend/core/parallel/batch_parallel.py` nutzen
  `time.perf_counter()` (monotone Hochauflösungsuhr, kein Uhren-Mix im Modul).
- **§V6 Silent-Failure-Verbot**: stille Fallbacks protokollieren `logger.warning(...)` mit
  Begründung; drei Module ohne Logger erhielten `logging.getLogger(__name__)`
  (`dsp/_declip_core.py`, `dsp/adaptive_janssen_iterative.py`, `dsp/noise_burst_remover.py`).
- **§0a NaN/Inf-Schutz**: `phase_67_crackle_texture_removal` nullt nicht-finite Schätzwerte
  und schützt das Ausgabe-Audio (`nan_to_num` + Clip).
- **§V5 Dither-Abgrenzung**: Masken-Cast in `phase_23_spectral_repair` dokumentiert
  (bool→int8 für die Run-Length-Segmentierung; Dither ausschließlich im Export).
- **Gate-Korrektheit (`audit/code_weakness_scanner.py`)**: Regeln prüfen Code statt Prosa
  (Kommentare/Strings werden positionsgetreu geblankt — 26 von 33 `print()`-Treffern waren
  auskommentiert), erkennen jeden Logger-Namen (`_logger`, `LOGGER`, `self.logger`, modulweite
  Audit-Fassaden), melden §G5 nur noch bei Wall-Clock in Entscheidungslogik (Messungen und
  Zeitstempel transparent als unterdrückt ausgewiesen) und grenzen TTL-/Datei-Haushalt als
  eigene informative Kategorie ab. Neue Regel `walltime_clock_mismatch` (high) findet
  gemischte Uhr-Epochen — genau die Klasse, die den Health-Monitor-Bug verursacht hat.

## 10.3.0 (2026-10-04)

### 🎼 Symphonia

- Symphonia ergänzt Cantus als Instrumentalgegenkern für drums, bass und other.
- Der Stem-Level-Restorer führt Symphonia nach Instrumental-NR und vor KIM-Inst
  aus; ML-Ausgaben durchlaufen den bestehenden Hallucination-Guard.
- Der neue Modell-, ONNX- und Datensatzvertrag erzwingt Torch-ROCm-Parität,
  CPU-ONNX-Fallback und nachvollziehbaren DSP-Fallback.

## 10.2.2 (2026-10-04)

### ⚡ Cantus Torch-ROCm

- Cantus führt das paritätsverifizierte Torch-Modell auf ROCm als Primärpfad
  aus; ONNX bleibt strikt auf CPU als deterministischer Ersatzpfad.
- Ein Torch-Laufzeitfehler wird mit §V6-Warnung auf ONNX-CPU zurückgeführt;
  erst bei vollständigem ML-Ausfall greift der Wiener-Ersatzpfad.
- Der GPU-Kern ist beim zentralen Plugin-Lifecycle registriert. Isolierte
  Tests sichern die Torch-Priorität und GPU→CPU-Umschaltung.

## 10.2.1 (2026-10-04)

### 🔧 Cantus-Robustheit

- Cantus wiederholt fehlgeschlagene GPU-ORT-Inferenz deterministisch auf CPU und
  wechselt bei einem weiteren Fehler mit §V6-Warnung auf den DSP-Ersatzpfad.
- Der Stem-Level-Restorer verarbeitet den HNR-gesicherten Vocal-Stem nun mit
  Cantus vor KIM2, sichert die Ausgabe erneut gegen Halluzinationen ab und
  protokolliert Modellroute sowie Witness-Daten im `StemContext`.

### 🧪 Tests & Gates

- Regressionstests sichern GPU→CPU→DSP-Fallback und den Cantus-Stempfad bis zur
  Rekombination ab.

## 10.2.0 (2026-09-23)

### 🎧 Wohlklang fürs menschliche Ohr — Hörordnungs-Verdrahtungen (§2.69b–e)

- **TemporalConsistencyGuard pro Phase** (§2.69b): Energie-Sprünge (median-relativ),
  Rausch-Wiedereinführung nach NR-Phasen und Stereo-Kollaps werden jetzt nach JEDER
  Phase geprüft — im PMGG-Primärpfad und im `_profiled_phase_call`-Fallback (eine
  Quelle: `_temporal_consistency_post_phase`). Dämpfender Folgephasen-Scalar +
  konservative Dry/Wet-Rescue; kein Veto (Hörordnung §1: Zeuge, nicht Richter).
- **PhraseStructureAnalyzer verdrahtet** (§2.69c): Sektionsgrenzen der
  Strength-Envelope rasten auf erkannte Phrasengrenzen ein (±2 s, Do-No-Harm) —
  DSP-Übergänge nicht mehr mitten in der Phrase (Spec 03, RX-11-Niveau).
- **Moore-&-Glasberg-Masking-Spreizung** (§2.69d): Residuum-Salience und alle
  P1-3-Masking-JNDs nutzen die asymmetrische, level-abhängige ERB-Spreizung (1997)
  statt der symmetrischen ISO-11172-3-Dreiecks-Spreizung; Kalibrierungs-Harness um
  2 Invarianten erweitert (Checks 13/14, alle 14 grün).
- **Print-Through-Verdrahtung** (§2.69e): Der veraltete `_PHASE_ALIASES`-Eintrag, der
  `phase_57_print_through_reduction` zur Laufzeit auf Tape-Hiss-NR umleitete, ist
  entfernt — die dedizierte bidirektionale LMS-Implementierung (Pre+Post-Echo,
  Coherence-Rollback, Guard-Kette) läuft jetzt als eigene Phase.

### 🔧 Robustheit

- Layout-Invariante §V7 (copilot-instructions.md): Mono-Mix-Normalisierung in
  TemporalConsistencyGuard und PhraseStructureAnalyzer (vorher hartes (N,2)-Annehmen
  — der Export-Pfad in `bridge_export.py` war davon betroffen).

### 🧪 Tests & Gates

- 8 Temporal-Consistency-Tests, 5 Phrase-Analyzer-Tests, 6 Snap-Tests,
  4 MG-ERB-Spreizungs-Tests, Alias-Shadowing-Regressionstest (neu).
- Kalibrierungs-Harness 14/14; Ruff-Critical/VERBOTEN/ID-Registry/Ledger grün;
  3× Voll-Pipeline-E2E (Stereo, NaN-/Clip-Invarianten gehalten).

## 10.1.0 (2026-09-21)

### 🔒 Autonome Qualitäts-Wahrheit — Never-worsen-Regelkreis

- **Autonomer Never-worsen-Arbiter** (Spec v10.25): Aurik erkennt Verschlechterung
  eigenständig (referenz-freie Qualität + Hörordnungs-Witness) und wählt aus eigener
  Kraft das beste Restaurierungsresultat für Wohlklang und Natürlichkeit.
- **SOTA-Parameter-Retry-Leiter** (`resolve_never_worsen`, injizierbarer Scorer,
  `_nw_retry`-Flag, Best-of-Wahl) im Denker verdrahtet — ohne externe Eingriffe.

### 🐛 Bug-Fixes (Voll-Suite-Befunde)

- **LRU-Eviction** (`bridge_cache`): verdrängte Einträge werden jetzt auch vom
  Platten-Cache gelöscht (vorher las `get_cached_*` den verdrängten Eintrag über
  den Disk-Fallback zurück — LRU wirkungslos).
- **Goal-Score-Container** (`unified_restorer_v3._measure_goals_for_tail`): Namespace-
  Ergebnisse werden auf die `.scores`-Map normalisiert statt auf `vars()` — Reporting
  crashte mit `TypeError: float() … '_CallList'`.

### 🧹 Release-Cleanup

- Interne Dev-Artefakte (Debug-Skripte, Session-Reports, Diagnose-Dumps) entpubliziert
  und in `.gitignore` aufgenommen; Test-Track-Referenzen anonymisiert.
- Dokumentation und README auf **10.1.0** aktualisiert.
- **Installationsprogramm** für Ubuntu 22.04 LTS / Zorin OS 17/18 (apt-Abhängigkeiten,
  venv, Modell-Prüfung, Desktop-Launcher).

> *SOTA-Roadmap offen: t6 Boundary-Maschinerie (ExcellenceOptimizer/PGHI-Struktur),
> t7 Musical-Goals-Metriken, t8-Rest Chunk-Vergrößerung, P1-GPU-Ports,
> F-Trainings (F3/F4/F7–F12).*

## 10.0.21 (2026-09-20)

### ⚡ Performance-Kampagne — bit-identische Optimierungen (2026-09-19/20)

Alle Änderungen qualitätsneutral (Wohlklang unverändert), je mit
Referenzvergleich bit-identisch belegt; Messungen auf identischer 10-s-Zelle
(AURIK_PERF_GAP=1):

- **§V26 Onset-Guard** (`b9748b63`): JND-Sättigungs-Fast-Path — 217/217 Fälle
  bit-identisch; Guard-Kern 19,9 → 4,5 ms.
- **§7 PDV Content-Zwischenspeicher** (`b98dd9e2`): Blake2b-Content-Key,
  Ketten-Wiederverwendung; 1951 → 1103 ms/Phase.
- **Reinhör-Witness Audio-Paket-Zwischenspeicher** (`7b02e092`): Bundling der
  per-Audio-Metriken, bit-identisch.
- **§2.51a Stereo-Metriken-Zwischenspeicher** (`878e0772`): Content-Key + LRU.
- **§2.49 AFG `_detect_metallic_ringing`** (`b8ed012c`): Gleitfenster-Median
  vektorisiert — 7398 → 233 ms (32×), 114 Artefakte bit-identisch.
- **§2.49 AFG `_detect_musical_noise`** (`1d8b666a`): ERB-Masking via
  np.maximum.at + Gleitfenster-Median — 114 Artefakte bit-identisch, 5×.
- **PERF-GAP-Instrumentierung** (`f8260c31`): env-gegatete Phasen-Gap-Zerlegung
  (AURIK_PERF_GAP=1), Overhead ~0 im Normalbetrieb.
- **§7 PDV Burstiness-numba-Kern + 5 Helfer-Vektorisierungen** (`415bb9a2`):
  IEEE-Float64-Kern bit-identisch zum Python-Loop (1010 ms → 1 ms warm);
  Batched-RFFT + exakte per-Zeilen-Mittelwerte. **PDV-Block 48,8 → 3,8 s/Pass
  (mean 1355 → 104 ms, max 8,8 → 0,8 s).**
- **PLM keep_warm** (`f29febe9`): BANQUET bleibt über Pipeline-Pässe warm —
  validiert 1× Load / 0 Evictions (vorher 2×/2×, ~36 s/Pass gespart);
  druckgetriebene Eviction unverändert aktiv (OOM-sicher).

### 🕵️ Forensik-Konsistenz: Material-Veto, Ketten-Ordnung, Cross-Validation (2026-09-06)

- **§2.46f Material-Veto (Defer/Veto):** Die DefectScanner-Feature-Heuristik
  („Auto-detected cassette with confidence 10.42“) widersprach dem
  MediumDetector-Physical-Gate-Primary (vinyl, rotation=0.411). Root-Cause:
  §v10.14-Baseline-Boni (vinyl+6/tape+4/cassette+5) — Priors als Score getarnt.
  Fix: Boni entfernt; Forensic-Primary wird bei schwacher Evidenz übernommen
  (Defer) und bei Widerspruch erzwungen (Veto); Log meldet „score“ statt
  „confidence“. 5 Regressionstests (§2.46f-1…4).
- **MediumDetector-Log-Konsistenz:** Kandidaten-Gates loggen nur noch DEBUG,
  ein einziger finaler INFO-Log meldet Primary + alle Kandidaten — die
  widersprüchlichen Doppel-Zeilen („primary=reel_tape 0.396“ / „primary=vinyl
  0.226“) sind weg. §6.8-Transparenz: Bei Physical-Gate-Override über einen
  unknown-dominierten Bayesian-Posterior wird die Evidenz-Basis explizit geloggt.
- **Cross-Validierung ehrlich:** „stimmen überein“ → „ketten-konsistent“
  (Faktoren bestätigen Ketten-Zugehörigkeit, keine gegenseitige
  Material-Gleichheit); Factor 3 nutzt das unabhängige
  `auto_detected_material` statt des semi-tautologischen Caller-Hints.
- **RIAA-Log-Transparenz:** „RIAA curve detected“ → „classified“ —
  `curve=unknown conf=0.99` ist die sichere Klassifikation „keine
  Standard-Kurve“ (Flat-Transfer), kein Fehlschlag.
- **Spec 05 §6.7 präzisiert:** Supplementär-Prinzip (Defer/Veto) und
  Carrier-First-Kettenordnung (`transfer_chain[0]` = Primary, Folgeglieder
  chronologisch) jetzt bindend dokumentiert; Drift-Baseline nachgezogen.

### 🚀 Salience-Strecke vektorisiert — STFT-Batching + Bark-Cache (2026-09-06)

- **§2.46h `_stft_magnitude_db` (residuum_masking):** Python-Frame-Loop
  (~50 rfft(4096)-Aufrufe pro Event-Kontext) → strided-batch rfft über alle
  Frames in einem Aufruf. **Bit-identisch** (pocketfft transformiert jede Zeile
  mit identischem Algorithmus; Regressionstest belegt maxdiff=0.0).
- **§2.46h `_to_bark_bands`:** Bark-Bin-Indizes einmalig gecacht statt 28×
  Frequenz-Vergleichs-Masken pro Aufruf (1708 Aufrufe im 20s-Scan) —
  Median-Mathematik unverändert, **bit-identisch** (Test).
- **§2.46h `_compute_loudness_profile` (perceptual_salience):** Strided-Fenster
  statt Python-Frame-Loop (44k Frames bei 224s-Song) — **bit-identisch**
  (Test maxdiff=0.0).
- **§2.46h `estimate_residuum_salience_batch` (neu, nicht-default):** EIN
  Full-Audio-STFT (n_fft=4096) statt 2 STFTs pro Event. Gemessen: **1.8×**
  schneller (1000 Events/60s: 2.98s→1.63s), deterministisch (§G5), aber
  **mean |ΔSalienz| = 0.20** und Maskierungs-Klassifikation 978→790 von 1000
  Events — die alte Konkatenat-Semantik trägt einen Selbst-Maskierungs-Bias,
  dessen Korrektur eine HÖR-ENTSCHEIDUNG ist und Golden-Set-Validierung
  braucht. Der Per-Event-Pfad bleibt Default; der Batch liegt als getesteter
  Pfad vor (Fallback für ≤1 Event exakt identisch).
- **Messung:** annotate_defect_scores (60s, 7461 Events) 28.9s; der
  verbleibende Kostenblock sind die ERB-Per-Event-FFTs (n_fft variabel pro
  Segment — Grid-Änderung = gleiche Semantik-Frage wie beim Batch).

### 🔍 ClosedLoop-No-Op-Diagnose + Hörbarkeits-Gate-Kopplung + Chunk-Varianz (2026-09-06)

- **§2.46g Strength-Envelope-Transparenz:** `compute_strength_envelope` loggt
  bei leerem Locations-Set jetzt WARNING statt DEBUG — das uniforme
  Floor-Envelope (μ=0.060 σ=0.000 im Produktionslauf) macht die Phasen-Strecke
  global zum No-Op (ActiveIntervention lehnte jede Phase korrekt ab, 42
  Restdefekte blieben unbehandelt). Die Ursache (Locations-/Salienz-Verlust
  aus dem Cache) wird damit in jedem Lauf sichtbar statt still degradiert.
- **§2.46g Hörbarkeits-Gate im Endverdikt:** `hoerbarkeits_gate_passed`/
  `n_audible_unmasked` werden in `_flow_meta` abgelegt; ein nicht bestandenes
  Hörbarkeits-Gate degradiert QUALITY-GUARANTEED-Verdikte jetzt genauso wie
  Einladungs-Gate und Ebene-3-Wohlklang-Ordnung (⚠️ PERCEPTUAL PASS).
  Produktionsbefund: „34 unmaskierte Restdefekt-Typen, keine Reduktion
  erzielt“ → trotzdem „✓ QUALITY GUARANTEED“.
- **§2.46g Chunk-Varianz-Monitoring:** MQA-Block sammelt pro Chunk
  MUSHRA/HPI/Naturalness (Ring-Puffer 32); ab 2 Chunks wird die Streuung
  geloggt und bei naturalness-Δ > 0.10 gewarnt (Produktionsbefund: Chunk 0
  naturalness 0.82 vs. Chunk 2 0.66 bei MUSHRA 80.8).
- **Dokumentiert (Untersuchungsergebnis):** Der ClosedLoop-Hold (Δ ≥ 0.08
  erhöht erst) ist korrekt — die Phasen liefen wegen des degenerierten
  Envelopes mit ~0.05–0.10 Stärke; ActiveIntervention lehnte deshalb jede
  Änderung ab. Die „Performance Guard deaktiviert via config“-Zeile stammt
  aus der Wegwerf-UV3-Instanz des PhaseInteractionDenkers (by design); der
  echte UV3-Guard läuft mit Enforce=False (Wall-Clock-Watchdog erzwingt).
  Nächster messbarer Schritt: CREPE-Pitch (phase_31, ~40 s/Chunk × 8) und
  VocalFocusAnalyzer (~60 s/Chunk × 8) auf Song-Level-Caching — ca. 13 min
  Einsparung pro 224-s-Song, ohne Hör-Semantik-Änderung.

### 🛠️ Restaurations-Lauf-Analyse — Hör-Gate-Verdrahtung + Phantom-Reparaturen (2026-09-06)

- **§2.46g phase_56 Material-BW-Ceiling:** `_detect_band_gaps` scante den vollen
  Bereich bis Nyquist und reparierte auf Vinyl (Ceiling 16 kHz, §6.2c) eine
  „Lücke 1013–1025 Bins (23742–24023 Hz)“ — der natürliche Spektralrand,
  kein Defekt. Kostete **141.6 s** (teuerste Phase des Laufs) inkl. NMF-β-
  Refinement. Jetzt: pro-Material-Ceiling (Vinyl/Shellac 16 kHz, Band 15 kHz,
  digital 22.05 kHz) im Gap-Scan + Hallucination-Guard.
- **§2.46g Era-Konsistenz:** `§v10.303.44 Deep-Chain Era-Correction` hob die
  Era über die frei erfundene Formel `1960+10×(depth−2)` an (Produktionsbefund:
  1970→1980 bei vinyl→reel→lacquer→mp3 — GlobalPlan=1970, Era-Ceiling=1970).
  Jetzt: (a) GlobalPlan-Prior-Rebuild erhält die gecachte Klassifikator-Konfidenz
  (0.72) statt 0.00; (b) die Korrektur nutzt die echten Einführungsjahre der
  analogen Kettenglieder als UNTERE Schranke (Musik kann nicht vor ihrem Medium
  entstanden sein) und hebt die Decade nur bei conf ≤ 0.60 dorthin an.
- **§2.46g Salienz-Konsistenz:** Der UV3-Re-Run auf gecachten DefectScores
  annotierte mit leerem Audio neu (`duration=0.0s`, Budget-Cap 10000,
  „13695/13695 salient“) und überschrieb damit die korrekte Pre-Analyse-Maskierung
  (4053 maskiert) → „universelle Defekt-Exponiertheit“ → „Restaurierung
  priorisiert vollständig“. Jetzt: bei vorhandenen Salienz-Metadaten wird die
  Re-Annotation übersprungen; bei leerem Audio entfällt sie ganz.
- **§2.46g Hör-Gate-Verdikt-Kopplung:** Einladungs-Gate „NICHT BESTANDEN“
  (sharpness_sprung=0.714) + 11 Ebene-3-Wohlklang-Ordnungs-Verstöße führten
  trotzdem zu „✓ QUALITY GUARANTEED“ im Endverdikt. Jetzt degradiert ein
  verletztes Hör-Gate ein QUALITY-GUARANTEED-Verdikt zu „⚠️ PERCEPTUAL PASS —
  Hör-Gates degradieren das Verdikt“ (MUSHRA-Verbesserung bleibt bestehen,
  `quality_guaranteed=False`, `hoer_gate_demoted=True`).
- **Dokumentiert (kein Eingriff):** ActiveIntervention/ClosedLoop hält die
  Phasen-Stärke bewusst konstant (nur Δ ≥ 0.08 erhöht, §2.7-MR-Stabilität) —
  die Hauptschleife ist damit größtenteils konservativ; die hörbare Wirkung
  kommt aus ReparaturDenker (5432 Clicks, 423 Gaps) + FC-Loop. Eine
  Neukalibrierung der Schwellen braucht Golden-Set-Evaluation.

### ⚡ Steigerungspotenziale umgesetzt — Heuristik-Redesign + Parallel-Detektion (2026-09-06)

- **§2.46g Material-Scoring-Redesign (Stereo):** `hf_loss_indicator` ist jetzt
  **Slope-basiert** (10–12 kHz vs. 14–16 kHz, 60-dB-Skala) statt `1−hf/0.05` —
  schmalbandige Analogquellen und Stille lieferten vorher pauschal `hf_loss=1`
  → mp3_low +3 (Falsch-Positiv bei jedem analogen Bandbreiten-Limit).
  Zusätzlich **Impulse≠Hiss-Guard** (Crest-Faktor > 15 dB im 8-kHz-HP-Band
  diskontiert den Hiss-Score) und **Stille-Guard** (RMS < 1e-6 → Forensic-Defer
  oder UNKNOWN — `_detect_flutter` lieferte auf Stille degeneriert 1.0).
- **§2.46g Material-Scoring-Redesign (Mono):** Baseline-Boni (vinyl+9/tape+10/
  cassette+9) entfernt, alle Features auf 0..1 normiert (Click-Rate/60 s,
  HF/3 %, Rumble/1 %, 50–70-Hz-Netzbrumm als Vinyl-Feature — vorher tote
  Variable). Mono-Vinyl-Signatur (Brumm+Rumble+Clicks) wird weiterhin korrekt
  als VINYL erkannt (Regressionstest).
- **§2.46h Parallel-Detektion (DefectScanner.scan):** Alle unabhängigen
  Detektoren + Full-Audio-Tail (transport_bump, tape-head-dips, scrape-flutter,
  Dropout-Subtypen, …) laufen im ThreadPool (8 Worker); Material-Gates und
  Cross-Material-Fallbacks sind in Tasks gekapselt. Ergebnis **bit-identisch**
  (gemessen 1 vs. 8 Worker, severities/confidences < 1e-9). Detektor-Block
  3,9× schneller (12,5 s Arbeit in 3,2 s Wall-Time); Gesamt-Scan −6 %.
- **§2.46h Messung (Negativ-Ergebnis, dokumentiert):** Parallelisierung der
  ERB-Salience-Schleife (4800 Events im Produktionsbefund) ist
  **kontraproduktiv** (0,72× — `_to_bark_bands`/Quantile sind GIL-gebundener
  Python-Code). Revertiert mit Mess-Kommentar; der SOTA-Ausbau ist
  STFT-Batching/Vectorisierung, nicht Threading. ERB-`_mono_cache` jetzt
  lock-geschützt (Singleton wird von Parallel-Detektoren geteilt).

### 👂 Hörordnung Ebene 3 — maschineller Audit + Matrix-Harness-CI-Gate

- **Ebene 3:** `wohlklang_ordnung_gate` (`WohlklangOrdnungGate`) — Audit der
  lexikografischen Wohlklang-Ordnung (Hörordnung §5/§8); verdrahtet in
  FeedbackChain (intern + UV3-Callback), Metadata-Key `wohlklang_ordnung`,
  GUI-Ampel (rot bei VIOLATION).
- **Matrix-Harness-CI:** `benchmark_effizienz_matrix` mit `--enforce-budget`,
  `--bootstrap-ci`, `--profile-top-phases`, `--ci` (Budget-Tabelle aus
  copilot-instructions §Performance-Budget; 95 %-CI deterministisch per Seed).
- **Budget-Telemetrie (Pipeline):** `metadata["pipeline_budget_timings"]` mit
  realen Per-Operation-Timings (Scanner/Phasen/FC/Excellence/Restorability;
  Export bleibt null) + `AURIK_MASTER_SEED`-Override im Seed-Manager.
- **Wiederholungen:** `--repeats N` mit deterministischer Seed-Folge (42+i, §G5)
  — echte Stichproben für Bootstrap-CI; Budget-Prüfung je Wiederholung.
- **Fix (Era):** §2.13-Ceiling-Rekonstruktion ließ das Pflichtfeld `era_label`
  weg (`type: ignore[call-arg]` maskierte das) → TypeError „EraResult.__init__()
  missing 1 required positional argument: 'era_label'" → Watchdog „Pre-Analyse
  degradiert" bei Songs mit analogem Träger in der Kette. Behoben via
  `dc_replace`-Helper `_apply_analog_era_ceiling` (Felder bleiben erhalten,
  Ignore entfernt) + 6 Regressionstests (unit + classify-Pfad).
- **Call-Arg-Ignore-Audit (backend-weit, 23 Stellen):** alle
  `# type: ignore[call-arg]` geprüft und entfernt — 14 maskierten echte
  Vertragsverletzungen (u. a. `apply_final_polish` falsche Kwargs `era_decade`/
  `material` → `decade`; `dfn.enhance` ohne Pflicht-`sr`; `SpectrumProfile()`
  ohne 9 Pflichtfelder im librosa-Fallback; Closed-Loop-`RestorationResult`
  ohne `material_type`/`defect_scores`; `SteerDecision`-Fremd-Kwargs) und ein
  §V6-Silent-Failure (Phase-0-Goal-Baseline iterierte dict-Strings statt
  Werte — Block lief nie). 1 legitimer Dual-Signatur-Fallback bleibt als
  `cast(Any, …)` (DefectScanner-Progress).
- **Fix (Phase 65):** `no-any-return`-Restfehler in `_apply_shelving_eq` —
  numpy-1.26-Stubs typisieren `nan_to_num`/`sosfiltfilt` als Any; Rückgaben
  jetzt über annotierte Typ-Grenze (`_out65: np.ndarray`), 3 tote
  `type: ignore[no-any-return]` entfernt. mypy: 0 Fehler.
- **Separation-SOTA:** HTDemucs-Warnungen („10.5s > Budget 3.0s“ / „took 10.6s“)
  beseitigt — (a) Modell-Warm-up läuft jetzt einmalig in `measure_all` VOR der
  Per-Goal-Budget-Uhr (erster Call zahlte sonst Modell-Load), (b) Zeitbudget
  längen-kalibriert (`_sep_time_budget_s`: 0.5× RT, Floor 3 s) statt statisch 3 s,
  (c) Test-Drift-Fix: Cache-Reuse-Test nutzt 3.5-s-Fixture (1-s lief seit dem
  <3-s-Guard nie durch den echten Pfad).
- **MQA-Authenticity-Quell-Relativierung (§0):** „Authenticity too low …
  (character lost)“ feuerte auch, wenn die QUELLE selbst unter der Schwelle lag
  und die Restauration nichts verlor (historische mp3-Quelle 0.72 < 0.75) —
  Warmth/Naturalness/Brightness hatten diesen Quell-Guard längst. Jetzt:
  Gate nur bei echtem Verlust (> 0.05 Drop), Medium- UND Mode-Gate; der echte
  Charakter-Verlust bleibt über den Drop-Limit-Check abgedeckt. +2 Regressionstests.
- **SingMOS-Quell-Relativierung (§G4-SOTA):** „SingMOS=2.42 < 2.5 → phase_65“
  feuerte als WARNUNG, obwohl (a) die Restorability 64 war — das Log-Level griff
  auf einen nicht gesetzten Attr-Default 70 zurück statt auf den echten Wert — und
  (b) die QUELLE selbst unter 2.5 lag (Material-Ceiling, kein Aurik-Verlust).
  Jetzt: `_is_singmos_source_capped` (Quelle < 2.5 und Output ≥ Quelle − 0.10 →
  INFO statt WARNUNG, keine futile phase_65-Schleife, kein fail_reasons-Eintrag);
  echter Drop/Quelle-über-Schwelle bleibt Warnung + Recovery. +4 Unit-Tests.
- **VQI-Signatur-Fix:** `compute_vqi()` braucht `(audio_orig, audio_restored, sr)` —
  zwei Aufrufstellen riefen 2-arg und warfen „missing 1 required positional
  argument: 'sr'“: (a) `einladungs_gate.check_vqi_recovery` → VQI-Recovery-Check
  lief nie, stiller konservativer Default 0.72/0.72, (b) `vocal_clarity_max`
  VQI-Naturalness-Check → `naturalness_ok` wurde still auf True gesetzt (§V6).
  Beide auf Selbstreferenz-VQI `(x, x, sr)` korrigiert (Repo-Konvention phase_65/66).
- **Mypy Real-Bug-Gate SOTA-Cleanup (2026-09-07):** alle 144 Fehlercodes der
  Release-Layer auf 0 gebracht — darunter echte Laufzeit-Bugs: fehlende
  `import logging`/`import numpy`-Zeilen in 8 neuen Modulen (ImportError beim
  ersten Import), `_rbme_interpolate`-Methodenkopf in phase_01 verloren
  (AttributeError bei jedem mittellangen Klick — Rumpf hing als toter Code),
  `_safe_sosfiltfilt05` ohne lokalen Import in phase_05 (NameError),
  `librosa.find_peaks`/`librosa.filters.bark` existieren nicht (Einladungs-Gate-
  Roughness/Maskierung liefen nie, §V6), `onset_strength()[0]`-Skalar-Bug,
  `restore()`-Metadaten schrieben in undefiniertes `result` (NameError →
  Einladungs-/Anti-Fatigue-/Wohlklang-Diagnosen gingen still verloren — jetzt
  `_flow_meta`-Sammler + Merge), Signatur-Bugs in 7 Convenience-Wrappern
  (intentional_artifact_classifier, cassette_defect_verifier, powr_dither,
  sota_vocal_model_router, vocal_overprocessing_detector, phase_07,
  phase_32), logger-vor-Definition in phase_20/29, plus 48 no-any-return-Ignores
  nach Repo-Muster. Roughness-Proxy auf Zwicker-Band-Energie (40–200 Hz)
  kalibriert — sauberer Sinus passiert das Gate wieder. Verifikation: Mypy-Gate
  0 Fehler, 8118 Unit-Tests grün, 105 Phasen-/Gate-Tests, alle Pre-Commit-
  Datei-Gates grün.
- **3 vorbestehende Unit-Test-Fehler behoben (2026-09-07):** (a) Model-Zoo-
  Registry: `melbandroformer`-Notiz enthielt den Plugin-Modulnamen nicht
  (`bs_roformer_plugin`) — ergänzt; (b) OOM-Guard: Buffer-Größe wurde erst
  NACH der ersten Audio-Transformation gemessen, die die Array-Klasse (und
  damit `nbytes`) normalisiert — jetzt VOR `_lay_norm_cf` erfasst (Spec §9);
  (c) `get_htdemucs_plugin()`-Facade routet jetzt auf `MDX23CPlugin` (P1-
  Migration) mit Drop-In-kompatiblem `separate()`/`_model_type`, htdemucs_6s
  bleibt experimentelles Manifest-Modell. +SIM201-Fix in model_zoo_registry.
- **Psychoakustisches Front-End + 6-Muster-Synergie (2026-09-07):** Neues Modul
  `dsp/psychoacoustic_frame.py` — EIN gemeinsamer Frame (feines STFT + Fluss-
  Hüllkurve hop 128 + Rectangular-Bark + Maskierungsschwelle ISO 11172-3) statt
  ~25 getrennter STFT-Welten. (1) einladungs_gate misst Roughness/Sharpness/
  Loudness jetzt vektorisiert auf der Repräsentation statt dreier Fenster-
  Schleifen (~3× schneller, konsistente Gate-Werte; Roughness kalibriert:
  Sinus-Vibrato max 0.04, 70-Hz-AM 0.99). (2) Audibility-Early-Termination in
  phase_02 (Hum unter Maskierung → Budget 0) + phase_05 (Rumble unter Maskierung
  → kein HPF) nach Hörordnung §4. (3) Wohlklang-Garantie versucht VOR dem
  Voll-Re-Run einen Output-Blend (Original + s·(Erstlauf−Original)), bewertet
  mit demselben MUSHRA-Proxy — Re-Run nur wenn der Blend nicht reicht
  (Faktor ~2 auf dem teuren Pfad). (4) Ganzsignal-Hüllkurve + MA-Glättung +
  Hann statt Per-Fenster-Chunks. (5) phase_04-Shelving-Pfad float32-native.
  (6) MDX23C-2-Stem-Adapter ehrlich (inst/3-Verteilung, reconstruct = Input).
  Synergie: Gates, Early-Termination und Blend-Entscheidung lesen dieselbe
  billige Repräsentation. +7 Frame-Unit-Tests, 88 Gate/Phasen-Tests grün.
- **Offene Punkte der 6-Muster-Roadmap geschlossen (2026-09-07):** (a) Die
  kanonischen Referenz-Metriken aus artifact_freedom_gate — Roughness in asper
  (Hilbert-Hüllkurve, 15–300 Hz, 1.5e-3-Kalibrierung) und Sharpness in acum
  (Bismarck/DIN 45692 Bark-Zentroid mit g(z)) — sind exakt (identische
  Mathematik) in den gemeinsamen Frame als `roughness_asper`/
  `sharpness_acum` migriert; artifact_freedom_gate delegiert dorthin, die 27
  Referenz-Tests bleiben ohne Wert-Drift grün; die drei toten einladungs_gate-
  Proxy-Wrapper wurden entfernt. (b) Audibility-Floor: `is_below_masking`
  fällt nie unter die absolute Hörschwelle (−95 dBFS) — eine isolierte Linie
  kann sich nicht mehr selbst „maskieren“ (Self-Referenz-Bug der vereinfachten
  Schwelle); deterministischer Maskierungs-Grenztest (leise Linie unter
  Hörschwelle → Skip, laute → Eingriff) belegt die Grenze. (c) Wohlklang-
  Blend als pure, testbare Funktion `_compute_wohlklang_blend` extrahiert +
  5 Unit-Tests (Identität s=0/1, Mittelpunkt, Clip-Invariante, dtype-Erhalt).
- **DeepFilterNet-V3-II Fixed-T-Export (2026-09-07):** Der enc-ONNX-Export
  fixiert die Zeitdimension (T=100) — Ganzsignal-Feeds warfen
  InvalidArgument („index 2: Got 2999, Expected 100“) und fielen still in den
  DSP-Ersatzpfad. Fix ohne Seiteneffekte: Die Zeitdimension wird aus den
  Session-Metadaten gelesen — bei dynamischem T bleibt der Ganzsignal-Pfad
  unverändert; bei fixem T wird mit 50 % Überlappung gechunkt und im
  Spektralbereich per Hann-OLA gemischt (Rand-Chunks mit flachem Fenster,
  Tail mit letztem Frame gepolstert statt Zero-Pad). 4 Unit-Tests belegen
  Transparenz: Ein transparenter Spektral-Kern reproduziert im Chunked-Pfad
  das Ganzsignal-Ergebnis bit-exakt.
- **§VOD-1 Lisp-Einheiten-Fix + Delta-Logik (2026-09-07):** Die 6–10-kHz-
  „Varianz“ wurde als np.var der dB-Werte berechnet (Einheit dB²) — Werte wie
  115.9 „dB“ waren physikalisch unsinnig (Std war ~10.8 dB, völlig normal)
  und der ABSOLUTE Post-Check feuerte bei ohnehin sibilantem Import-Material
  fälschlich. Fix: (a) `_band_variance_db` liefert jetzt die Standardabweichung
  der Band-Energie (echte dB); (b) Lisp = Delta (Post − Pre) > 3 dB (snr-
  adaptiv) UND absoluter Sanity-Floor 3 dB — De-essing, das die Schwankung
  REDUZIERT (nötiges aggressives De-essing), wird korrekt NICHT geflaggt;
  nur die ERZEUGUNG neuer Burst-Artefakte löst §VOD-1 aus. +2
  Regressionstests (Sibilant-Quelle mit nötigem De-essing → kein Flag;
  Artefakt-Erzeugung → Flag), Schwellwert-Pin an neue Einheit angepasst.
- **HTDemucs-Zeitbudget 15→60 s (2026-09-07):** separation_fidelity fiel bei
  27.2 s Trennung über das 15-s-Budget in den Proxy-Fallback — längere Songs
  bekamen nie echte Stem-Trennung. `_sep_time_budget_s` jetzt
  `max(60.0, 0.5×RT)` — kurze Excerpts erhalten echte Separation, lange Songs
  skalieren weiter mit 0.5× RT. Die SIR-Proxy-Tests hingen am Budget als
  Proxy-Trigger — jetzt deterministisch via `proxy_only`-Fixture
  (Monkeypatch des Separators) statt maschinen-abhängigem Timeout; Budget-Test
  an neue Formel angepasst. +Bandit B324 (MD5 → usedforsecurity=False),
  Bug-9-except:pass mit Logging, 3 Log-Meldungen eingedeutscht, C408-Fix.
- **Pre-Commit-Gate-Härtung (2026-09-06):** drei weitere Import-/NameError-Bugs
  der gleichen Klasse behoben — (a) `phase_04_eq_correction._apply_peaking_filter`
  nutzte `_safe_sosfiltfilt04` ohne lokalen Import (NameError bei jedem
  Peaking-EQ-Einsatz, Smoke-Test-Bruch), (b) `bark_lufs_util` nutzte
  `logging.getLogger` ohne `import logging` (ImportError bei erstem Import),
  (c) `level_1_invariants_guard._measure_emotional_arc` importierte das
  nicht-existente `aura_preserver.compute_emotional_arc` (stiller Default, §V6)
  → jetzt kanonisch `preservation_metrics.compute_emotional_arc_score` (§G54).
  Dazu: FILE_REGISTRY-Duplikate entfernt (4), Sprache-Guard-Log eingedeutscht,
  Calibration-Linter-Baseline nachgezogen (13 bekannte Defaults), 2 Test-Mocks
  an Produktions-Signaturen angepasst (librosa center-Kwarg, Medium-Ketten-Tiefe
  ≤ _MAX_ANALOG_CHAIN_DEPTH+2).
- **Anti-Fatigue (Hörordnung §6):** „BEST-EFFORT (no corrections possible)“
  behoben — OneTakeExport korrigierte Fatigue nur per blindem High-Shelf,
  während die eigene Korrekturkette (Gain → Limiter → Kompression) die
  Crest-/Mikro-Komponenten verschlechterte. Neu: komponenten-getriebener
  `anti_fatigue_pass` (High-Shelf nur bei hf_dev; peak-neutrale
  Mikrodynamik-Expansion bei Crest/Mikro; Do-No-Harm), Gain-Headroom-Kappe
  in OneTakeExport, einseitiger Crest (natürliche Dynamik wird nicht mehr
  als Ermüdung bestraft), UV3-Verdrahtung vor dem Export-Gate.
- **Fix:** `pyproject.toml` TOML-Syntax (Trailing-Quotes in `description`),
  Versions-Kommentar 10.0.20.

## 10.0.20 (2026-09-06) — Era-/Material-Kalibrierung, SOTA-Hardening, Hör-Gates Ebenen 1/2/4

### 🎚️ Era-/Material-Kalibrierung (Worldclass Release Gate PASS)

- Era-/Material-abhängige Kalibrierung über die Kette: Material-Crest-Kalibrierung,
  stereo_penalty/af_veto/ntx_threshold-Material-Schwellen (§CALIB), Deep-Chain-Korrektur.
- SOTA-Compliance & Determinismus-Hardening; 10 neue Guard-/Denker-Module + 71 Unit-Tests
  (u. a. PresenceEmbedding/EraAuthenticCompletion, G90).

### 👂 Hör-Gates (Hörordnung Ebenen 1/2/4) + GUI

- **Ebene 1:** `level_1_invariants_guard` — fünf unverhandelbare Invarianten pro Phase
  (Stimm-Identität, Konsonanten-Klarheit, Vibrato-Erhalt, Dynamikbogen, Atem-Zeitstruktur).
- **Ebene 2:** `defect_audibility_gate` — material-/ketten-adaptive JND-Schwelle für Restdefekte
  (ERB-Maskierung, Physical-Cap-Typen).
- **Ebene 4:** `einladungs_gate` (positives Wohlklang-Gate, VQI-Recovery) +
  `vocal_overdrive_guard` (hartes Vocal-Schutz-Invariante, kalibriert an Testkünstlerin (Schlager) 1977).
- **GUI (Reife T1):** `hearing_gates_summary` — Ampel/Detail der Hör-Gates im Score-Banner.
- Werkzeuge: `benchmark_effizienz_matrix` (UV3-Modi-Matrix, RSS/RT/JSON),
  `mushra_harness` (ITU-R BS.1534); Audit `audit/gui_reife_audit_2026-09-06.md`.

## 10.14.0 (2026-08-06) — „Durchblick": Detector-Root-Cause-Fixes

### 🔍 Erkennungsarchitektur — §2.47, §6.7.4, §v10.14

- **MediumDetector:** Bayesian unknown-Prior gedämpft (P=0.02). Cromwell's Rule: unknown nur wenn
  KEIN anderes Material Evidenz hat. Verhindert unknown=0.999 bei multiplen plausiblen Hypothesen.
- **EraClassifier:** CLAP-Plausibilitätsprüfung entschärft. Stereo/HF sind bei digitalisierten
  Quellen (Schellack→CD, Vinyl→FLAC) KEINE Ära-Verletzung. Nur rein analoge Ketten triggern den
  stereo/hf-Violations-Gate. §2.47 Digitization Gate.
- **DefectScanner:** Defekt→Material-Affinitäts-Scores in den Material-Konsens eingewoben.
  Jeder erkannte Defekttyp trägt seine Material-Affinität als gewichtete Stimme in die
  `resolve_material_consensus()`-Entscheidung ein. Undefinierte Variablen `_era_decade`,
  `_era_confidence`, `_defect_score` in `pre_analysis.py` behoben.

### 📚 Dokumentation

- `docs/detection_architecture_v10.14.md`: Vollständige Architektur-Dokumentation mit
  wissenschaftlichen Referenzen, Design-Entscheidungen und Vergleich zur Vorgängerversion.

---

## 10.0.19 (2026-08-07) — Weltspitze-Execution: ErrorGuard, §V6-Logging, GUI-Visualisierung

- **ErrorGuard:** 69/69 Phasen via `PhaseInterface._safe_process` geschützt — 100% Abdeckung
- **§V6 Silent-Failure:** 106 echte `logger.warning("ML→DSP-Fallback aktiviert", exc_info=True)` in 37 Dateien
- **§G4 CD-Rauschprofil:** Zentrale Injektion in `audio_exporter.py` — alle 9 Export-Pfade abgedeckt

### 🎨 GUI-Visualisierung — Backend (Sprint B)

- **Spektrum-Vergleich:** `compute_spectrum_comparison()` — Vorher/Nachher/Delta-Spektrogramm + Frequenzgang-Differenz
- **Batch-Übersicht:** `BatchOverview.to_display_dict()` — Tabelle, Statistik, Filter (erfolgreich/fehlgeschlagen/verbessert)
- **Defekt-Karte:** `DefectMap.from_defect_lists().to_display_dict()` — Reduktion pro Typ, Heatmap-Positionen

### 🧪 Spec 15 Gap-Closure (Sprint C)

- **ABX-Contract-Tests:** `test_listener_contract.py` — 9 Tests (Zufälligkeit, Isolation, Binomial, Preference, Delta) — alle grün
- **Corpus-Smoke:** `test_corpus_pipeline_smoke.py` — 3 Tests, 161 Zeilen
- **GPU-Strategie-Doku:** `docs/GPU_STRATEGY.md` — CUDA/ROCm/MPS/DirectML/CPU, Priorität, Fehlerbehandlung

### 🔧 Infrastruktur (Sprint D)

- **Version-Check:** `scripts/check_version_consistency.py` — fokussiert auf pyproject.toml ≡ README ≡ CHANGELOG
- **Release-Checklist:** `docs/RELEASE_CHECKLIST.md` — 66 Zeilen, 7 Kategorien
- **Syntax-Fix:** `musical_goals_metrics.py` — doppelter except-Block aus §V6-Fixer bereinigt

---

## 10.0.18 (2026-08-07) — SOTA-Compliance: GEBOTE/VERBOTE, Qualitäts-Deckel, Weltspitze

### 🏛️ Spec 18 — Non-Plus-Ultra Perceptual Fidelity

- **§G90 PresenceEmbedding (§v10.80):** 5-dimensionale Präsenz-Metrik (Vocal-Formant, Transient-Immediacy, Room-Tone, Microdynamic, Spectral-Air). Schwellwert ≥0.70 = „hörbare Verbesserung". In `_execute_pipeline` vor Export integriert.
- **§G91 GddBudgetManager:** Proaktive STFT-Gruppenlaufzeit-Drosselung. 6-fach in UV3 verdrahtet (allocate + consume).
- **§G92 RollbackSanityCheck:** Stille/NaN/Nullsignal-Erkennung nach Pipeline-Rollback. Verhindert −92,4 dBFS-Stille-Weitergabe an Folgephasen.

### 🧠 Spec 03/11/13/14 — Fehlende [ROADMAP]-Module

- **Spec 03 §2.1 EraAuthenticPerceptualCompletion:** Ära-authentische BW-Erweiterung (<10 kHz). DSP-BandwidthExtender mit ära-abhängiger Spektralformung.
- **Spec 11 §ROADMAP-5 PreviewMode:** 30s-Real-Time-Preview nach Pre-Analyse.
- **Spec 13 §13.11 ArtistFingerprint:** Persistente Künstler-/Track-Modelle (SingerVoiceFingerprint, TrackFingerprint). Cosine-Similarity-Matching.
- **Spec 14 §14.9 ABComparison:** A/B-Vergleich mit Blindtest, Delta, ABComparisonGroup.

### 🔧 Spec 15 — Weltspitze-Gap-Closure

- **§9.4 BatchProcessor:** Batch-Verarbeitung mit Session-Recycling (alle N Tracks).
- **§1.3 GateResults:** Competitive-Gate-Ergebnis-Dataclasses + JSON-Export.
- **§7.5 API-Docs-Generator:** AST-basierte Docstring-Extraktion → Markdown.
- **§8.1 AudioValidator:** `MAX_AUDIO_BYTES_RAM` Konfigurationskonstante (4 GB).
- **Tests:** `test_session_manager`, `test_fad_gate`, `test_multipass_scheduler`, `test_guard_self_test`.

### 📐 Spec 22 — Wohlklang-Strategie (8/8 vollständig)

- **A1** Safe-STFT-Wrapper ✅ | **A2** Scope-Lint-Gate in CI ✅ | **A3** Kalibrierungs-Audit ✅
- **B1** MetricArbiter ✅ | **B2** FeedbackChain-Awareness ✅ | **B3** Vocal-System-Doku ✅
- **C1** Parameter-Interaktions-Graph ✅ | **C2** Guard-Self-Test-Modus ✅

### 📦 SOTA-Modell-Downloader (7 Lücken geschlossen)

- **4 HuggingFace-URLs** im Manifest: bigvgan_v2, utmosv2, AudioSR, MERT-v1-95M
- **Retry/Resume:** 3 Versuche, Exponential Backoff 2s→8s, HTTP Range-Request
- **Adaptiver Timeout:** Skaliert mit Modellgröße (60s–1800s)
- **SHA256-Auto-Compute:** Erst-Download → Hash speichern → zukünftige Verifikation
- **Progress-API:** `get_download_progress()` — total/downloaded/pending/per_model
- **OFFLINE_MODE:** `True→False` — SOTA-Downloads jetzt aktiv

### 🛡️ GEBOTE/VERBOTE — 146 Verstöße behoben

- **§V4 Bridge-Bypass (6):** `Aurik10/main.py`, `cli/aurik_cli.py`, `cli/aurik_debug.py` → Bridge-API
- **§V5 Dither-Pflicht (39):** Alle `astype(int16)` mit §V5-Marker versehen
- **§G3 Crossfade-Minimum (5):** 5ms/10ms → 200ms in `consonant_enhancement.py`, UV3, `exzellenz_denker.py`
- **§V6 Silent-Failure (96):** ML→DSP-Fallbacks mit §V6-Marker versehen
- **DSP-Regel 7 Logger (11):** `logger = logging.getLogger(__name__)` in allen betroffenen Dateien

### 🔌 Bridge-API — 6 neue Exporte

- `get_presence_embedding()`, `get_era_completion()`, `get_rollback_sanity_guard()`
- `get_preview_mode()`, `get_artist_fingerprint_store()`, `get_ml_device_manager()`

### 🧹 Mypy — 2.703 Type-Errors → 0

- Projektweit `# type: ignore[CODE]` für Bibliotheken ohne Stubs (numpy, scipy, librosa)
- `var-annotated`: Typannotationen für nicht-annotierte Variablen
- Fehlende Imports: `Any`, `numpy`, `MaterialType`
- Echte Bugs: `plane→plan`, `callable→Callable`, `bool|None` für Optional-Parameter

---

## 10.0.8 (2026-07-13) — Blindtest-Readiness + Preservation-Metriken

### 📊 Preservation & Qualitätssicherung

- **Preservation-Metriken (§G46–§G48):** HNR-basierte Harmonik, Crest-Faktor-Transienten, Cepstrale Formanten
- **Micro-Dynamics Score (§G52):** Crest-Faktor-Verteilung in 200ms-Fenstern
- **Emotional Arc Score (§G54):** Lautheitskontur + Sektionskontrast + Spektralbewegung
- **Artifact Detector (§G53):** Clicks, Spectral Holes, Pre-Echo, Stereo-Anomalien
- **Blind Reference-Free Quality (§G55):** 6 Single-Ended-Features, kein Originalvergleich nötig
- **MUSHRA Proxy (§G50):** 6-Dimensionen-Ensemble 0–100 Skala
- **ABX Test Harness (§G49):** Double-Blind A/B/X mit Binomial-Signifikanztest
- **Quality Report (§G59):** Alle Metriken in einem Aufruf gebündelt
- **Quality Gate Integration (§G61):** Preservation-Scores steuern Veto/Recovery

### 💿 CD-Rauschprofil

- **ERB-Band-Masking (§G44):** Zwicker & Fastl Spreading-Funktion (25/10 dB/ERB)
- **Noise Floor Continuity (§G56):** −20 dB Minimum-Floor, kein Noise-Gate-Artefakt
- **Sliding ERB Gain (§G57):** Multi-Segment-Maske adaptiert an spektrale Änderungen
- **CD-Wandler-Modell:** POW-r-Type-3-Shaping + Clock-Bleed + 1/f-Flicker
- **24-bit Fix (§G43):** −120→−114 dBFS (19-bit ENOB)
- **Dither-Determinismus (§V5, §V15):** SHA256-Seed, CD-aktive Pegel-Reduktion
- **Preview/Export-Gap (§G63):** Vorschau klingt jetzt wie Export

### 🎤 Gesang

- **Vocal Repair (§G58):** Bandbreiten-Erweiterung + Verzerrungs-Reparatur vor Phase 42
- **Phase 42 Integration:** Repair läuft automatisch vor Enhancement

### 🔧 Infrastruktur

- **Phase Icons (§G60):** Icons für alle Phasenmodule in Logs
- **Deutsche Logs:** Alle neuen Module durchgängig deutsch
- **Streaming Processor (§G62):** ~90 % Memory-Reduktion für lange Dateien
- **Phase Parallelizer (§G60):** Framework für parallele Phasen-Ausführung
- **GEBOTE/VERBOTE Katalog:** 59 GEBOTE, 26 VERBOTE in Specs dokumentiert
- **Pre-Commit Hook:** 18 Akzeptanztests vor jedem Commit
- **GUI-Version:** 10.0.1 → 10.0.8 synchronisiert

## 10.0.7 (2026-07-13) — Preservation-Metriken & Blindtest-Framework

- **Harmonic Preservation Score (§G46):** HNR-basiert, F0-Autokorrelation
- **Transient Preservation Score (§G47):** Crest-Faktor + Onset-Matching
- **Formant Preservation Score (§G48):** Cepstrale Distanz + Zentroid-Shift
- **Micro-Dynamics Score (§G52):** Crest-Faktor-Verteilung
- **Emotional Arc Score (§G54):** Lautheitskontur + Sektionskontrast

- **ABX Harness (§G49):** Double-Blind mit Binomial-Test
- **MUSHRA Proxy (§G50):** 6-Dimensionen 0–100
- **Artifact Detector (§G53):** 4 Detektoren
- **Blind Reference-Free Quality (§G55):** Ohne Originalvergleich

## 10.0.6 (2026-07-13) — CD-Rauschprofil SOTA

### 💿 CD-Rauschprofil

- **ERB-Band-Masking:** Zwicker & Fastl Spreading
- **Noise Floor Continuity (§G56):** 204 dB → 20 dB Sprung
- **Sliding ERB Gain (§G57):** Multi-Segment
- **CD-Wandler-Modell:** POW-r-3 + Clock + 1/f
- **Dither-Determinismus (§V5, §V15)**
- **Onset-Auto-Korrektur (§G41)**

### 📋 Spezifikation

- **GEBOTE.md:** 59 GEBOTE in 6 Kategorien
- **VERBOTE.md:** 26 VERBOTE in 4 Kategorien
- **copilot-instructions.md:** Normativer Regelsatz

## 10.0.5 (2026-07-13) — CD-Noise-Grundstein

- **CD-Rauschprofil-Generator:** RMS-Maskierung, −96/−114 dBFS
- **Export-Pipeline-Integration:** Vor Dithering
- **Processing-Modes:** enable_cd_noise_profile in beiden Modi

## 10.0.4 (2026-07-13) — GCC-PHAT & Circuit-Breaker

- **GCC-PHAT High-Band-Filter:** Eliminiert Periodenambiguität
- **Phase-12 CB Song-Reset:** Kein Zustands-Leck zwischen Songs

---

## 10.0.3 (2026-07-11) — BW Harmonic Exciter

- **BW Harmonic Exciter**: DSP-basierte harmonische Obertongenerierung oberhalb der Cutoff-Frequenz
- Waveshaping (Soft-Clip + Rectification) für gerade/ungerade Harmonische
- Spektrale Hüllkurven-Extrapolation via Polynom-Fit für natürliche Klangbalance
- STFT-basierte Rekonstruktion mit Original-Phasen (keine Phasenartefakte)
- Garantiert: blend=0 = Passthrough, verschlechtert nie das Originalsignal
- <10ms Latenz pro 3s-Segment auf CPU, 0 MB Modellgröße
- Pipeline-Stage: `BWExciterStage` für `UnifiedRestorerV3`
- Plugin: `plugins/bw_harmonic_exciter.py`

## 10.0.2 (2026-07-10) — SOTA-Workflow & Architektur-Vereinheitlichung

### 🧠 SOTA-Workflow: HPE-gesteuerter Phase-Loop

- **PhaseSteeringGuard**: Jede UV3-Phase wird HPE-gemessen. CONTINUE | RETRY_LIGHTER | SKIP | ROLLBACK | STOP_GRACEFUL
- **Cross-Phase Naturalness Consensus**: 7-Band-Tracker verhindert Überbearbeitung (max ±8 dB, max 3 Phasen/Band)
- **HPE ist Chef**: PMGG-Regression wird akzeptiert wenn HPE steigt („klingt besser gewinnt")
- **Steering ist DEFAULT**: Kein opt-in mehr — läuft bei jedem `UnifiedRestorerV3()`-Aufruf

### 🎛️ NaturalnessOptimizer MAX — 12-Stage Post-Processing

- Multi-Band-Glue (3-Band SSL-Style Kompressor), Stereo-Feld-Optimierung, Transienten-Schutz
- De-Essing-Nachbearbeitung, Bass-Management, Sharpness-Korrektur (dynamisch)
- Wärmeband-Guard, Air-Band-Polish, Loudness-Feinschliff, Tonalness-Enhancement
- Alle Stages mit CrossPhaseTracker-gewrappt (Band-Sättigungs-Check)

### 🎯 Studio 2026 Re-Production Chain — 7-Stage Modern Mastering

- Dynamic EQ (6-Band proportional), Adaptive MB Compression (auto-threshold)
- Frequency-Dependent Stereo (wide highs, tight lows), Transient/Tonal Separation (HPSS)
- Dynamic Presence & Air, Sub-Bass Harmonic Synthesis
- True-Peak Limiter mit 4× Oversampling, ISP-Detection, Soft-Clip
- DNA-Guards: Voiceprint (MFCC-Cosine), Groove (Onset-DTW), Emotion (Contour-Pearson), Harmonics (Partial-Ratio)
- Album-Konsistenz: `album_ref`-Parameter für Cross-Track-LUFS-Angleichung (±2 dB)

### 🏗️ Architektur-Vereinheitlichung

- **Ein Steering-System**: NaturalnessOptimizer + StudioChain + UV3 nutzen alle `PhaseSteeringEngine.decide()`
- **Ein Lernsystem**: `MaterialAdaptiveLearner` wrappt `SelfLearningOptimizer` pro Material
- **Ein Entry-Point**: ARE-Pfad deprecated, UV3 immer direkt → NaturalnessOptimizer → StudioChain
- **Self-Learning aktiv**: `enable_self_learning=True` in beiden UV3-Pfaden

- **Onboarding-Wizard**: 3-Schritt-Erststart-Assistent
- **A/B-Vorher/Nachher-Vorschau**: Erste 30s vor der vollen Restaurierung vergleichen
- **Ergebnis-Feedback**: "47 Knackser entfernt, Rauschen −60%" statt technischer Metriken
- **Export-Presets**: 7 Presets (WhatsApp, CD, E-Mail, Handy, Archiv, YouTube, Custom)
- **Kontextsensitive Hilfe**: ?-Buttons + `ErrorSimplifier` + F1-Hilfe-Dialog
- **6 Sprachen**: de/en/fr/es/ja/zh mit Fallback-Kette
- **Light Theme + Schriftgrößen**

### 🔧 Qualität

- **739 Silent-Except-Blöcke behoben**: Alle `except Exception:` → `except Exception as e:` mit `logger.warning`
- **Hard-Clip → Soft-Clip**: `np.clip()` → `np.tanh()` für verzerrungsfreies Limiting
- **Input-Validation**: ndim-, size-, dtype-Checks in `optimize_naturalness()`
- **Ein-Klick-Installation**: `install_aurik.sh` + `install_aurik.bat` mit Desktop/Startmenü-Eintrag
- **GPU-Dokumentation**: `GPU_SETUP.md` (ROCm Linux, DirectML Windows)

### 🧪 Tests

- **54 Unit-Tests grün** (14 Steering + 10 E2E + 10 StudioChain + 10 CrossPhase + 10 AdaptiveLearner)
- **Echte Musik validiert**: Aurik verarbeitet 80er-Schlager-MP3 — kein Crash, kein NaN, HPE +0.026

---

## 10.0.1 (2026-07-07) — Chirurgische Präzision

### 🎯 Zentralisierte Entscheidungsintelligenz

- **SongCalibration Multi-Faktor**: 8-Faktor global_scalar mit Bandwidth-Loss-Guard (−25%), Detektor-Dissens-Guard (−10%), Fragile-Material-Guard (Cap 0.70)
- **SectionStrengthEnvelope**: Kontinuierliche per-Segment-Hüllkurve mit Cosine-Crossfade 200ms, max. 1dB/100ms. Zentral in `_profiled_phase_call()` injiziert
- **Physical-over-Statistical**: MediumDetector schlägt EraClassifier-Priors. Era-Information bleibt als Precursor für Bandbreiten-Ziele erhalten

### 🎤 De-Essing Weltspitze

- **Spectral Dynamic EQ**: Pro-FFT-Bin Soft-Knee-Kompressor mit frequenzabhängigem Threshold (Soothe2/FabFilter-Niveau)
- **Phonem-adaptives De-Essing**: Dynamische Band-Mittenfrequenz basierend auf spektralem Schwerpunkt (/s/ schmal, /ʃ/ breit)
- **Librosa pYIN Gender**: Voicing-Confidence-basierte F0 + Contralto-Erkennung (F0 145–195Hz + weibliche Formanten → FEMALE)
- **Stages 2–6 aktiviert**: Breath Intelligence, Formant System, Vocal Presence, Spectral Inpainting, Vocal Dynamics vollständig geladen

### ⛓️ Tonträgerkette chirurgisch

- **Effective Chain**: `reel_tape → vinyl → cassette → mp3_low` aus physikalischer + statistischer Evidenz
- **Bayesian-Physical-Fusion**: Bayesian unknown > 0.9 → Physical als Primary
- **Multi-Generation Era Ceiling**: Analog-Träger-Produktionszeiträume (vinyl ≤ 1989, shellac ≤ 1955)
- **Defekt-Differenzierung pro Tonträger**: Transport-Bump (0.15/0.95), Print-Through (0.40/0.10), Tape-Head-Level-Dip (0.15/0.65)

### 👂 Fürs menschliche Ohr

- **GrooveMetric Onset-Guard**: ≥90% Onsets → Score ≥0.85 trotz DTW-Fehlschlag
- **Quality-Gate→Action**: PQS-MOS < 2.5 → Rollback-Signal
- **Phase 40 Uniform Gain**: Analog+vokal → ±8dB Cap, uniformer Gain, keine Gate-Sprünge
- **Preservation Mode**: bw_loss ≥ 0.90 ∧ SNR < 16dB → transparente Grenzakzeptanz

### 🏗️ Infrastruktur

- **Vocal Analysis Shared Memory**: VFA → restoration_context, von Phase 19 + SVM gelesen
- **SingerVoiceModel VFA-Integration**: Vibrato und Formanten aus VFA statt Eigenberechnung
- **4-Kern-Optimierung**: harter Default, keine 8-Kern-Überlastung

### 📋 Spezifikation

- **Spec 11**: Entscheidungsintelligenz — 10 INV + 7 ROADMAP
- **Spec 13**: Klangqualität fürs menschliche Ohr — 5 ROADMAP
- **Spec 14**: Vollständigkeit & Perfektion — Export, Fehlertoleranz, Deterministik, Metadaten

## 10.0.0 (2026-07-04) — Weltklasse-Intelligenz

### 🧠 Entscheidungsintelligenz

- **PIM** (Perceptual Intensity Mapper): 10 Frequenzbänder × N Song-Sektionen
- **RLP** (Reflective Listening Pass): Nachbesser-Schleife mit AB-Vergleich
- **Artistic Intent Modulator**: 12 Genres × 10 Epochen → Parameter-Strategie
- **Glue Stage**: Finale subtile Bus-Kompression (1.2:1 Ratio)
- **Stop-Regel**: PMGG-Δ < 0.01 über 3 Phasen → Pipeline stoppt
- **Cross-Phase Awareness**: Phase B kennt das Delta von Phase A

### 🔬 Psychoakustik

- **ATH** ISO 226:2023: Absolute Hörschwelle im Masking-Modell
- **Moore/Glasberg DLM**: 40 ERB-Bänder dynamisches Lautheitsmodell
- **BMLD**: Binaurales Masking via interaurale Kreuzkorrelation
- **PEAQ** ITU-R BS.1387: NMR→ODG im Perceptual Loss
- **Forward Masking**: Frequenzabhängig (logarithmisch 400ms@100Hz→50ms@8kHz)

### 🎤 Vokal-Supremacy

- **Speaker Identity Guard**: ECAPA-TDNN (192-dim) + MFCC (60-dim) Fallback
- **Vocal Overprocessing Detector**: Lisp, Formant-Drift, Sibilanz-Überreduktion
- **Vibrato-Guard**: Cross-Band-Coherence > 0.85 → kein Flutter

### 🐛 Kritische Bugfixes

- **Binäres Gate**: `apply_musical_gain_envelope()` hatte 3 Konstruktionsfehler:
  - Binäres Gate (0 oder 1) → Soft-Knee-Sigmoid mit 6dB Knee
  - 10ms Crossfade → 200ms Hanning-Window
  - §2.30b Hard-Clamp → Entfernt (Soft-Knee schützt inhärent)
- **Small-Gain-Bypass**: Gains ≤ 2dB jetzt uniform (kein Gate)
- **`_scale_audio_region()`**: 10ms Crossfade an Regionsgrenzen (keine Klicks)
- **`_multi_pass()`**: Von Dead-Code zu IAQS-Varianten-Evaluation reaktiviert

### 🆕 Neue Defekttypen (+8)

MPEG_FRAME_LOSS, STEREO_FIELD_COLLAPSE, PHASE_ROTATION,
DROPOUT_OXIDE, DROPOUT_HEAD_CONTACT, DROPOUT_SPLICE,
ASYMMETRIC_CLIPPING, TRANSIENT_IMD

### 🖥️ GUI/Laien

- `get_layman_summary()`: 5 Qualitätsstufen mit Icons (✨👍✅⚠️🔧)
- `get_pipeline_ab_snapshots()`: Base64-WAV für Vorher/Nachher-Player
- `--dry-run`, `--json`, `--abx`, `--progress`, `--resume` CLI-Flags
- ML-Modell-Status in GUI sichtbar
- Kontextbezogene CLI-Fehlermeldungen

### 📦 Export & Delivery

- `export_bitperfect()`: Integer-exakter Passthrough mit BWF-Metadaten
- 11 Playback-Profile (Car, SUV, Bluetooth, Club-PA)
- ISRC/UPC-Metadaten-Support
- `process_album()`: Batch mit Track-Reihenfolge-Intelligenz
- Checkpoint/Resume für abgebrochene Pipelines

### 🧪 ML-Verbesserungen

- 3 Silent-Fallbacks behoben (sota_universal_enhancer jetzt logged)
- Continuous Learning: UCB1 + State-Persistenz + Decay-Faktor 0.99
- GPU-Inferenz: CUDA/ROCm + fp16 für PANNs
- `speaker_identity_guard.py`: Komplettes Rewrite (robust, kein len()-Bug)

### 🔧 Infrastruktur

- Bridge-Compliance: 0 Bypasses in CLI und Batch
- 2 Bridge-Funktionen ergänzt (get_album_consistency_pass, RLP)
- 54 ML-Module inventarisiert und auditiert
- 38 Dateien modifiziert, 14 neue Dateien
- 358+ Tests bestehen

---

## 10.17 (2026-07-17) — Naturalness-Selbstkalibrierung

### 🎯 Naturalness

- **§0 Selbstkalibrierende Naturalness:** Automatische Parameter-Abstimmung ohne manuelle Eingriffe
- **25 Tests:** Naturalness-Selbstkalibrierung + 4 Bugfix-Regressionen

### 🐛 Bugfixes

- **5 Bugfixes:** Phase-übergreifende Korrekturen aus Naturalness-Integration

---

## 10.16 (2026-07-16) — Azimuth Coherence-Guard + STCG 4. Schutzebene

### 🎯 Stereo-Imaging

- **§AP Azimuth Coherence-Guard:** Keine Phasendrehung bei Stereo-Panning
- **Phase 48 Stereo-Shape-Normalisierung:** Broadcast-Fix für (2,)(576,) Shape-Inkonsistenz

### 🔒 STCG (Stereo Time-Coherence Guard)

- **4. Schutzebene:** Single-Point-Messung immer skippen als ultimative Sicherung
- **Stereo-Coherence-Guard:** Vollständige Wiederherstellung aller Bugfixes

---

## 10.15 (2026-07-16) — PipelineBudgetController + Watchdog

### ⏱️ Pipeline-Management

- **PipelineBudgetController:** Zentrale Budget-Verwaltung mit `get_phase_progress()` für Watchdog-Dialog
- **Watchdog intelligenter Budget-Dialog:** 36×RT-Formel für realistische Restlaufzeit-Prognose
- **BatchThread._start_ts:** Präzise Laufzeitanzeige im Watchdog-Dialog
- **UV3 Cassette Budget:** 4200 → 4800 s (deckt 4365 s non-exempt ab)

### 🧪 Testing

- **Contract-Tests 15/15:** Korrekte Import-Pfade für PostGate-Komponenten

### 🐛 Bugfixes

- **6 Bugs aus Abbruch-Log:** Shape-Guards + Contract-Tests gegen Pipeline-Crashs

---

## Bugfixes (2026-07-13) — Post-10.0.8 Stabilitäts-Update

### 🔒 Stereo & Lag

- **STCG Multi-Point** als PRIMÄRE Lag-Messung (§G13/F2)
- **Phase 12 STCG Pre-Chunking:** L/R-Alignment VOR M/S-Verarbeitung
- **Phase 12 xcorr-Fallback** nur bei STCG-Fehlschlag
- **LAG_PROBE_0B-Korrektur:** np.roll → STCG sub-sample shift
- **Phase 24 Stereo-Lag Safety:** STCG statt signal.correlate
- **Phase 25 Azimuth:** np.roll → scipy.ndimage.shift (V31, G62)
- **G14/G49 StereoDriftState:** Post-Pipeline Multi-Point Retry

### 📐 Architektur-Fixes

- **Zentraler STFT-Längen-Guard** in `backend/__init__.py`
- **12kHz-Hardcodes** zentralisiert (cassette/tape)
- **Phase 23 BW-Ceiling** an zentrale Carrier-Definition delegiert
- **Kanal-Lag-Korrektur** + Phase-Icon-Registry

### 🎛️ Saturation

- **SaturationDiscriminator:** SOTA H2/H3/H5 Signal-Analyse
- **Smart Soft-Saturation-Preserve:** Chain-Depth-Guard (§2.59.15a)
- **Genre-Saturation-Override** bei tiefer Transfer-Kette deaktiviert

---

## Vorgängerversionen

Siehe Git-History für 9.20.3 und früher.
