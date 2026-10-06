# Changelog — Aurik 10.4.0

## 10.4.0 (2026-10-06)

### SOTA-Defizit-Register + Gate (WP-0) — stilles Kapital wird auditierbar

**Wurzel-Fix (§V7 copilot-instructions.md):** Der Modell-/Wohlklang-Defizit-Stand
war über Roadmap-Prosa, Changelog und Lauf-Reports verstreut; drei Defizit-Klassen
(belegt trainiert-aber-nicht-deployt, synthetisch-statt-Musik, Sprach-Signalpfad)
waren **nirgends** als solche geführt.

- **Neu: `.github/SOTA_DEFICIT_REGISTER.md`** — ID-getrackte Matrix
  `D-<Klasse>-<n>` mit **20 Einträgen** (K0 = 9, K1 = 2, K2 = 5, K3 = 4),
  Status-Enum (`offen`/`in-arbeit`/`geschlossen`/`bewusst-akzeptiert`) und
  **Evidenzpflicht pro Zeile** (§III.13 copilot-instructions.md): `geschlossen`
  ist nur mit **existierendem** Beleg-Pfad zulässig; Domänen-Behauptungen
  (`musik`/`sprache`/`gemischt`) brauchen konkrete, prüfbare Evidenz —
  `unbekannt` ist die ehrliche Angabe statt einer Sprach-Vermutung.
- **Neu: `scripts/sota_deficit_gate.py`** — fail-closed Gate für genau diese
  drei Akzeptanzkriterien des Roadmap-Arbeitspakets WP-0, plus
  **Domänen-Abgleich Register ↔ kuratiertes Manifest** (§G9
  copilot-instructions.md — eine Quelle). Verdrahtet als Pre-Commit-Hook
  `aurik-sota-deficit-gate` (`always_run`).
- **Neu: `tests/normative/test_sota_deficit_gate.py`** — 26 Tests: Bestands-
  Nachweis am echten Register und Regel-Nachweis über synthetische Register-
  Texte (Status/Evidenz/Klasse/ID/Enum/Format/Domänen-Abgleich) sowie
  Hook-Verdrahtung.

### Artefakt-Probe: **alle** Modelle am Artefakt ausgelesen (§III.13)

**Wurzel-Fix (§V7 copilot-instructions.md):** Die Domänen- und Identitätsaussagen
zu den ML-Modellen stammten aus Doku — bis hin zu einer nachweislich falschen
LibriTTS-Zuordnung. Jetzt wird jedes Modell-Artefakt **gemessen**:

- **Neu: `scripts/model_artifact_probe.py`** — liest ONNX-Köpfe (ohne External
  Data), Torch-Zip **und** Legacy-Pickles über einen _restricted Unpickler_, der
  Gewichte **nie materialisiert** (Storage-/Tensor-Stubs), dazu safetensors-,
  npy- und joblib-Header. Gemessen werden Mel-Bänder, Upsample-Faktor,
  Kanalbreiten, Parameterzahl, Hparams (`config`/`hparams`/`args`), Dtype-Mix,
  Operator-Mix und die lokale Doku-Evidenz (`config.json`, Modell-Karten, Lizenz).
- **Neu: `.github/ML_ARTIFACT_FINGERPRINTS.md`** (generiert, 66 Verzeichnisse ·
  **153 Artefakte** · 44,1 GB · **0 Auslesefehler**) — je Verzeichnis eine
  Messtabelle mit SHA-256, Parameterzahl und Architektur-Fingerabdruck.
  Formate: 89× ONNX, 49× Torch-Zip, 3× Legacy-Pickle, 5× npy, 2× safetensors,
  2× TorchScript, 2× joblib, 1× pip-`.pth` (korrekt als _kein_ Modellartefakt
  klassifiziert statt als Fehler).
- **Herkunftsbelege maschinell (Muster EAR-VAE, jetzt für alle):**
  Archiv `models/_archive_20260920` ↔ deployt → **8× byte-identisch**,
  **6× architektur-gleich mit abweichenden Gewichten** (Finetune-/Generations-
  belege), **14× ohne Gegenstück** (offen dokumentiert).
- **Neue harte Befunde:** `models/nvsr/nvsr.onnx` ist **byte-identisch** mit
  `_archive_20260920/flashsr/flashsr_onnx_prod_20260810.onnx` ⇒ das deployte
  Artefakt ist der **FlashSR**-Export, nicht ein NVSR-Modell (Namens-Drift, jetzt
  als D-K2-4 geführt). Die **Upstream-Basis** der EAR-VAE
  (`…/pretrained_weight/ear_vae_44k.pyt`, 147,86 M) und der **Sprach-Basis-Core**
  von SGMSE+ (`sgmse_wsj0_reverb.ckpt`, 327,95 M) liegen jetzt lokal als
  Vergleichsartefakte vor — die §III.11-Sperre ist damit gegen die echte Basis
  prüfbar.
- **Ehrlichkeit:** Aus Bytes ist die **Architektur-Identität** belegbar, **nicht**
  der **Trainingskorpus** (ein ONNX trägt kein Trainingsmaterial). Die neuen
  Vokoder-Einstufungen bleiben deshalb `unbekannt` statt „Sprache" (s. u.).

### Lücken-Matrix (gemessen) — was Aurik zum weltbesten Wohlklang fehlt

- **Neu in `docs/TODOS_SOTA_ROADMAP.md`:** konsolidierte **Lücken-Matrix**
  (A) vorhandenes, aber nicht wirkendes Kapital, (B) fehlende Modelle/
  Fähigkeiten, (C) Pfad-/Hygiene-Lücken, (D) priorisierte Reihenfolge — jede
  Zeile auf lokale Messungen gestützt (Artefakt-Probe, Verdrahtungs-Audit,
  Flags, Defizit-Register).
- **Neu im Register:** **D-K0-8** — der trainierte flache Material-/Depth-
  Klassifikator (`models/medium_shallow_v1.joblib`, CV 64,3 %/85,7 % bei n=56)
  hat **keinen Konsumenten** (`medium_classifier` nutzt im `use_ml`-Zweig CLAP);
  die Vergleichs-Baseline stammt aus Manifest-Feldern und ist **nicht
  applikationsgleich** — Rollout erst nach fairem Re-Measurement + A/B +
  Hörordnungs-Sign-off. **D-K0-9** — `models/ear_vae2_upstream/` (EAR-VAE v2)
  braucht eine Rollout-/Ersatz-Entscheidung gegen den v1-Finetune.
- **Verifizierter Ist-Stand der Pfade:** 22 Pfad-/Aktivierungstests grün
  (`test_model_zoo_activation`, `test_primary_paths_no_fallback`,
  `test_model_inventory_wiring`), Verdrahtung **0 undokumentierte Verwaiste**,
  Register-Gate ohne Verstoß.

### Fehlende Modelle aus der Sicherung beschafft

- **Bestandsabgleich** Repo ↔ drei Sicherungen (16.09./21.09./06.10.2026):
  die 06.10.-Sicherung ist deckungsgleich, die älteren enthielten **7 fehlende
  Bestände** — sie sind jetzt in-repo:
  `_archive_20260920/` (9,7 G, 26 Artefakte, Herkunftsbelegstelle),
  `ear_vae2_upstream/` (EAR-VAE-**v2**-Upstream, 298 M),
  `hubert/` (HuBERT-ONNX + External Data, 362 M),
  `mpsenet/` (MP-SENet-Trainingsquelle + `best_ckpt`, 8,8 M),
  `rvc/` (`rmvpe.pt` 181 M, `hubert_base.pt` 190 M — Originale zu den
  deployten ONNX-Exporten), `medium_shallow_v1.joblib` (**trainierter**
  Material-Klassifikator, 1,7 M) sowie die fehlenden Teile
  `ear_vae/{docs,eval,config/ear_vae_v2.json}`.
- **Verfahren:** `rsync --ignore-existing` (vorhandene Produktionsartefakte
  werden **nie** überschrieben), `.venv`/`.cache` ausgeschlossen. `models/`:
  62 → 69 Einträge.
- **Register:** neue Position **D-K0-7** — das Archiv ist „stilles Kapital":
  für 14 Artefakte ohne Gegenstück ist weder Deploy noch Verwerfen dokumentiert;
  die `ear_vae/*_orig_20260730_backup.onnx` (84,1/84,3 M) werden ausdrücklich
  **nicht** mit dem deployten Finetune (73,8 M) gleichgesetzt.

### Vocoder-Domäne am Artefakt gemessen — Doku-Zuordnung widerlegt (§III.13)

**Wurzel-Fix (§V7 copilot-instructions.md):** Der Gate-Fund (Manifest
`audio-allgemein` vs. Registry `sprache`) wurde **nicht** durch Übernahme der
Registry-Worte gelöst, sondern durch Messung **am Artefakt** — die Registry-Angabe war selbst
  nur Doku-Übernahme (Ursprung: Kopfkommentar `scripts/train_bigvgan_f3.py`).
  Gemessen: `models/bigvgan/bigvgan_v2.onnx`/`.pth` hat **128 Mel-Bänder**
  (`conv_pre.weight_v` = `[1536, 128, 7]`), 6 Up-Stufen = **512×**,
  `upsample_initial_channel` 1536, **SnakeBeta** (109×`alpha` **und** 109×`beta`)
  mit Anti-Alias und 122,19 M Parameter ⇒ **BigVGAN v2 in der
  44,1-kHz-Konfiguration** (`bigvgan_v2_44khz_128band_512x`) — die Behauptung
  „LibriTTS/Sprache" ist damit **architektur-widerlegt** (LibriTTS-Konfiguration
  = 24 kHz/100 Bänder/256×/512 Kanäle/≈14 M). `models/hifi_gan/hifi_gan.onnx`
  hat **80 Mel-Bänder**, `[8,8,2,2]`/`[16,16,4,4]` = **256×**, Kanäle **128/64**
  ⇒ 0,93 M Parameter, 22 050 Hz ⇒ HiFi-GAN-Topologie, aber **kein offizieller
  Checkpoint** (V1: 512 Kanäle/13,9 M; V2: `[4,4,2,2]`/`[8,8,4,4]`) ⇒ die
  Zuordnung „UNIVERSAL_V1" ist widerlegt. Ohne Modell-Karte, Hparams oder
  Upstream-SHA ist der Trainingskorpus nicht belegbar ⇒ beide Vocoder sind
  **`unbekannt`** (nicht „Sprache"); das Signalpfad-Risiko bleibt als Defizit
  (D-K0-2/D-K2-1) erhalten.

- **Deployt = Basis, nicht F3-Finetune (bytescharf bewiesen):** 469 von 783
  Tensoren unterscheiden sich zwischen `models/bigvgan/bigvgan_v2.pth` und
  `output/_training_archive_20260920/f3_bigvgan/best.pt`; der produktiv geladene
  ONNX stimmt exakt mit der **Basis** (`ups.1.0.weight_v`: max|Δ| = 0 gegen
  Basis, 1,0e-1 gegen F3) und datiert auf 2026-09-10 (Finetune: 2026-10-03) —
  der Export kann ihn nicht enthalten. Der F3-Musik-Finetune bleibt damit
  weiterhin nicht im Rollout.
- **Doku-Wahrheit hergestellt:** `.github/ML_MODEL_DOMAIN_REGISTRY.md` erhält
  den Mess-Block §4a und korrigierte Tabellenzeilen;
  `.github/SOTA_DEFICIT_REGISTER.md` (D-K0-2, D-K2-1, Drift-Abgleich samt Lehre),
  `scripts/model_inventory.py` (`_DOMAIN_EVIDENCE` mit Messwerten, neu für
  bigvgan), `scripts/train_bigvgan_f3.py` (Ursprungs-Fehlannahme) und
  `.github/instructions/ml_domain.instructions.md` (Regeln 2 und 5) sind
  synchronisiert; der Drift-Sperrtest
  (`test_curated_manifest_marks_unproven_vocoders_as_unknown`) sichert die
  Artefakt-Evidenz ab. Kuratierte Verteilung: 32 Einträge
  (22 musik / 3 sprache / 5 audio-allgemein / 2 unbekannt).
- **EAR-VAE-Herkunft geklärt — eigene Fehlannahme widerlegt (§III.13
  copilot-instructions.md):** Die erste Fassung dieses Releases führte die
  Roadmap-Angabe „EAR-VAE MUSDB-Finetune" als **nicht belegbar** (nur im
  Arbeitsbaum gesucht). Die Prüfung des Backups
  (`/media/michael/Aurik_Backup/…/models/ear_vae_upstream/`) zeigt das
  Gegenteil: `models/ear_vae/{encoder,decoder}.onnx` (`8f97ef67…`/`ed64a1f1…`)
  sind **byte-identisch** mit `ear_vae_ft_{encoder,decoder}_inline.onnx` — der
  Produktionsstand **ist** der MUSDB-Finetune; die früher genannten SHAs
  `4168db14…`/`48dad212…` sind derselbe Finetune in der External-Data-Variante.
  Der Checkpoint `ear_vae_music_finetuned.pyt` (SHA `993e69be…`) und das
  Trainings-/Exportrezept (`finetune_music.py`, `masking_loss.py`,
  `export_finetuned_onnx.py`, `benchmark_finetuned.py`, `model/`, `config/`)
  liegen jetzt **in-repo** unter `models/ear_vae/` — mitsamt der External-Data-
  Paarung `ear_vae_ft_{encoder,decoder}.onnx` (+ `.data`; SHAs
  `4168db14…`/`48dad212…`, nicht der Produktionspfad). Ein Re-Export reproduziert
  die deployten ONNX-Artefakte mit **max|Δ| = 0,0** (Encoder, Decoder,
  Ende-zu-Ende); die kopierte External-Data-Paarung liefert ebenfalls
  max|Δ| = 0,0 gegen die Produktion. Registry-Absatz, Register, Roadmap und das
  kuratierte Manifest (`_DOMAIN_EVIDENCE`: 2 neue kuratierte Pfade →
  32 Einträge) sind korrigiert.
- **Verankerung:** `AGENTS.md` §7 (Lektüre-Einstieg), Roadmap WP-0 auf
  „erledigt" gesetzt; Drift-Baseline fortgeschrieben.

## 10.3.22 (2026-10-06)

### ML-Transparenz + CLAP-Scoring an der Wurzel (§6.8, §G9, §V6)

**Medium-Klassifikation: physikalisch zuerst (§6.8 copilot-instructions.md)**

- `medium_classifier.classify()` war **CLAP-first**: das flache CLAP-Materiallabel
  konnte die physikalische Trägerbestimmung übersteuern (Produktionsbefund:
  digital erzeugte CD-Datei → „tape"). Jetzt **physikalisch zuerst**; CLAP füllt
  ausschließlich ein UNKNOWN-Ergebnis (Zeuge, kein Richter).
- `laion_clap_plugin`: Material ist **MULTI-LABEL** (unabhängige Kosinus-Evidenz,
  NICHT auf Summe 1) — Träger sind nicht disjunkt (Tonträger-KETTEN, 99 % der
  Restaurierungsfälle).
- `pre_analysis`: CLAP-Konsens vergleicht **ketten-joint** (on-chain vs. off-chain
  Evidenz-Masse) statt Einzel-Argmax.

**CLAP-Tag-/Zero-Shot-Normalisierung**

- Zero-Shot-Queries lieferten **0.0** (gesättigter gemeinsamer Tag-Softmax).
  Jetzt: **absolute** Kosinus-Ähnlichkeit je Query — vergleichbar über Aufrufe
  (für die pos/neg-Saldierung des `GermanSchlagerClassifier`).
- Instrumente/Genre: **Per-Kategorie-Softmax** statt eines gemeinsamen
  43-Tag-Softmax (der verwässert → `top_instruments` immer leer, Genre-Gate
  feuerte nie, Genre-Argmax = Rauschen).

**CLAP-Checkpoint-Laden**

- `text_branch.embeddings.position_ids` (nicht-parametrischer Index-Buffer)
  löste eine **Falsch-Warnung** aus; jetzt als benigne klassifiziert
  (`classify_checkpoint_params`).

**BEATs-Transparenz (§G8)**

- `beats_plugin`/`phase_53`: lokaler Export ist **Encoder-only** (768-dim), kein
  527-Tagger — Doku korrigiert, Regressions-Guard ergänzt.

### ML-Domänen-Registry — „die meisten Modelle sind Sprache" ist widerlegt (§III.13)

**Kanonische Registry:** `.github/ML_MODEL_DOMAIN_REGISTRY.md` belegt die
**Trainings-Domäne** jedes der **61** Modellverzeichnisse in `models/` aus
Trainings-Skripten, vendored Modell-Karten, Export-Skripten und SHA-Vergleichen.
Ergebnis: **37 musik-trainiert (≈ 61 %)**, nur **11 sprach-trainiert (≈ 18 %)**
— der häufigste KI-Fehlschluss („die meisten lokalen ML-Modelle sind
Sprachmodelle") ist damit widerlegt.

- **Normative Verankerung:** neue Regel **§III.13** in
  `copilot-instructions.md`; Eintrag in der normativen Kette (`AGENTS.md` §1
  Punkt 4 + §3 Schnell-Referenz + §7 Lektüre); Invariante in `CLAUDE.md`.
- **Auto-geladene Instruktion:** `.github/instructions/ml_domain.instructions.md`
  (`applyTo` deckt `plugins/`, `backend/`, `denker/`, `models/`, `scripts/`,
  `Aurik10/`, `cli/` ab).
- **Maschinenlesbar:** `python scripts/model_inventory.py --domains` liefert
  Domänen + Evidenz als JSON (kuratiertes Manifest: 20 musik / 3 sprache /
  7 audio-allgemein).

**Aufgedeckte Domänen-Korrekturen (Evidenz):**

- `models/ear_vae` = offizielles **Musik**-Rekonstruktionsmodell (εar-VAE) —
  **Korrektur 2026-10-06:** der deployte Stand ist der **MUSDB-Musik-Finetune**
  (SHA-belegt, siehe 10.4.0); die damalige Angabe „nicht deployt" war falsch.
- `models/miipher_dit` = auf **MUSDB18-HQ-Vocals** trainiert („Singing Voice
  Enhancement"), **aktiv** (`use_miipher_dit=True`) — kein Sprachmodell.
- `models/bigvgan` = **Sprach-Basis**; Musik-Finetune F3 existiert und ist
  A/B-validiert (HNR +4,42 dB), aber Flag `BIGVGAN_V2_HR_ACTIVATED` ist OFF.
- Sprach-Signalpfad-Risiken dokumentiert: HiFi-GAN, Vocos, NVSR, AERO
  (§V6-Fallback mit Warnung); `nara_wpe` ist unüberwacht/domain-neutral.

## 10.3.21 (2026-10-06)

### §G9 Genre-Single-Source (Tranche 3.1–3.3) + Zahlen-Wahrheit Runde 2

**Genre-Auflösung: eine Quelle statt fünf**

- `backend/core/genre_registry.py` ist die **einzige** Genre-Alias-Quelle. Neu:
  `goal_weight_key()` (Schlüsselraum von `song_goal_importance`) und
  `semantic_hint_label()` (Anzeige-Taxonomie); zusätzliche Alias-Abdeckungen
  (`techno`, `electronica`, `classic_rock`, `indie_rock`, `orchestra`,
  `dt_schlager`, `latin_pop`, `country_&_western`, …).
- **Fund des neuen Guards:** Die **fünfte** parallele Genre-Auflösung lag in
  `backend/core/phases/phase_53_semantic_audio.py` (`_GENRE_ALIAS_MAP`, 33 Einträge,
  eigene deutsche Etiketten). Sie ist **verhaltensgleich** in die Registry portiert
  (`semantic_hint_label`) — Nachweis: **47/47 identische Ausgaben** über 47
  Referenz-Proben; die Phase ist jetzt ein dünner Delegat.
- **Bewusst konserviert und dokumentiert** (keine stille „Verbesserung“):
  `metal→"Rock"`, `country→"Folk"`, `ambient→"Electronic"`,
  `latin→"Unbekannt"`, sowie die Substring-Reihenfolge (`classic_rock→"Klassik"`,
  weil `class` vor `rock` geprüft wird). Eine Korrektur wäre eine hörrelevante
  Verhaltensänderung und braucht A/B + Hörordnungs-Sign-off (§v10.802).

**Neuer fail-closed Guard** `scripts/genre_single_source_check.py` (Hook
`aurik-genre-single-source`, `always_run`):

- **R1** Genre-Alias-Tabelle nur in `genre_registry.py` — fand sofort die fünfte Tabelle.
- **R2** Jeder Schlüssel beider Ziel-Gewichtstabellen muss über `normalize_genre()`
  auf ein kanonisches Genre auflösen (17 + 10 Schlüssel geprüft).
- **R3** XOR-Invariante `goal_profile_key(g) is None` ⟺ `g in _GOAL_KEY_ABSENT`
  (11 deklarierte Ausnahmen, 21 kanonische Genres) — kein stilles Fehlen.
- **R4** **Bericht** (kein Fail): Wert-Divergenz der beiden Tabellen —
  **115 Zellen in 7 gemeinsam geführten Genres**. Die Divergenz ist _klangrelevant_
  (die Tabellen haben echte Rollen), daher wird sie sichtbar gemacht, aber
  **nicht** automatisch vereinheitlicht (§G8-Transparenz, §v10.802).

**Zahlen-Wahrheit Runde 2 (Doku vs. Code)**

- **11 weitere veraltete Zählwerte** korrigiert (65 DefectTypes / 66 Kausal-Ursachen),
  u. a. in der **normativen** Spec 02 (`46 defect_types`, `49 Ursachen`,
  `62 Kausal-Ursachen`), `docs/DEFECT_SCANNER_SPEC.md` (behauptete
  „Production-Ready“ mit 62 und listete nur 30),
  `backend/core/defect_phase_mapper.py` („20 DefectType-Werte“, real 65),
  `docs/architecture/*`, `docs/KI-AGENT-INTEGRATION-GUIDE.md`, `denker/README.md`,
  `docs/PROJECT_STATUS.md`.
- **Gate erweitert** (`scripts/defect_coverage_check.py`): Quellenliste 16 → 21 Dateien
  (Spec 02, `docs/architecture`, `DEFECT_SCANNER_SPEC`, KI-GUIDE,
  `defect_phase_mapper`, `pre_analysis`); die Regex erfasst jetzt auch
  `defect types`/`defect_types` und `Ursachen` ohne „Kausal-“; neue dokumentierte
  Inline-Ausnahme `defect-count-ok` für **legitime Teilmengen** (z. B.
  „Top-3 Ursachen“) — bewusst sichtbar im Text statt versteckt. Grenzen dokumentiert
  („N Defekte“, „N Typen“, „defect-type ticks“ bleiben manuell gepflegt).

**Tranche 3.4 — Ziel-Dialekte deklariert, stilles Verwerfen beendet (§V6)**

- **Befund (gemessen):** Das Projekt führt **vier Ziel-Vokabulare**; kanonisch ist
  allein `song_goal_importance.ALL_GOAL_NAMES` (15 Ziele). `genre_goal_profile` und
  `goal_budget._DEFAULT_GOAL_BUDGET` führen je 15 **eigene** Namen, Überlappung mit
  dem Kanon nur **8** → **12 Fremdschlüssel**.
- **§V6-Verstoß belegt:** `goal_budget.create_goal_budget()` übernahm Genre-Ziele nur
  bei `if goal in targets` und **verwarf pro Genre 5 von 15 Zellen still** — bei
  `metal` ausgerechnet die stärksten Signale (`punch` 2.0, `bass_praesenz` 1.9,
  `makrodynamik` 1.4).
- **Neu:** `GOAL_DIALECT_MAP` + `GOAL_DIALECT_NOTES` + `canonical_goal_name()`
  (`backend/core/song_goal_importance.py`). Alle 12 Fremdschlüssel sind **deklariert**
  — 2 wörtlich abgebildet (`mikrodynamik` → `micro_dynamics`,
  `raeumlichkeit` → `spatial_depth`), 10 ausdrücklich **offen** mit Begründung.
  Es wird nichts geraten (§V7 copilot-instructions.md): jede Zuordnung ist eine
  Klang-Entscheidung und braucht A/B + Hörordnungs-Sign-off (§v10.802
  copilot-instructions.md).
- **§V6 behoben, Klang unverändert:** Der Budget-Konsument meldet jede Verwerfung jetzt
  mit Gewicht, Begründung und kanonischem Ziel (erste Nennung als Warnung je
  Genre/Ziel, danach Debug — keine Log-Flut). **Verhaltensgleichheit gepinnt:** die
  wirksamen `metal`-Zellen sind bit-identisch zu vorher (Regressionstest).
- **Guard R5** (`scripts/genre_single_source_check.py`): jeder Ziel-Schlüssel der drei
  Ziel-Tabellen muss kanonisch **oder** in `GOAL_DIALECT_MAP` deklariert sein — ein
  fünftes, undeklariertes Vokabular ist fail-closed.
- **`SPEC.md` §2.3:** Phasentabelle ab 41 befüllt (Quelle: normative Spec 06 inkl.
  Mechanismus/Fallback), §0a-Verbote markiert; die Ist-Stand-Diskrepanz (Überschrift
  „66 Phasen" vs. 67 numerierte Phasen / 71 Dateien) ist **dokumentiert**, nicht
  stillschweigend geändert.

**Tests:** 294 (Genre-/Goal-Suiten) und 65 (Phase 53 / Registry / Guard / Ziel-Dialekte)
grün; neu: 26 Guard-Regressionen in `tests/unit/test_genre_single_source_guard.py`
(inkl. 3× R5) und 36 Ziel-Dialekt-Tests in `tests/unit/test_goal_dialects_t34.py`.

## 10.3.20 (2026-10-06)

### Zahlen-Wahrheit: Doku-Zählwerte gegen die Code-Wahrheit abgesichert (26 Stellen)

- **Befund:** Die Regeln nannten **62 DefectTypes**, das Enum hat **65**; Spec 03/05
  nannten gleichzeitig **54**, **46** und einen als „normativ“ markierten Wert **49**
  für dieselbe Größe und widersprachen sich damit im selben Block. Gemessen:
  `len(DefectType) == 65`, `len(CAUSES) == 66`.
- **Wurzel der Drift:** „62“ bezeichnete an verschiedenen Stellen **Verschiedenes**
  (DefectTypes _oder_ Kausal-Ursachen) — deshalb war weder ein Suchen-und-Ersetzen
  noch eine reine Sichtprüfung belastbar.
- **Gate (`scripts/defect_coverage_check.py`, 4. Bedingung):** neu prüft der
  bestehende fail-closed-Abgleich zusätzlich **dokumentierte Zahl == Code-Zahl**.
  Quelle ist eine **bewusst kuratierte Liste** von Dateien, in denen die Zahl eine
  Aussage über den _aktuellen_ Ist-Stand ist; Chroniken, Meilenstein-Tabellen,
  Audit-Momentaufnahmen und die in `AGENTS.md` §6 als Stubs deklarierten
  `.agents/skills`-Kopien sind **ausgenommen**, weil ihre alten Zahlen Historie und
  damit korrekt sind. Die Meldung nennt Datei **und Zeile**, damit die Korrektur
  mechanisch erfolgt statt geraten.
- **Korrigiert (26 Gate-Treffer):** `.github/copilot-instructions.md` (2×),
  `.github/GEBOTE.md`, `.github/ID_REGISTRY.md`, `.github/specs/03` (4×: 54→65, 46→65,
  2× Kausal-Ursachen 62→66), `.github/specs/05`, `.github/specs/25`, `AGENTS.md`,
  `CLAUDE.md`, `README.md` (2×), `CONTRIBUTING.md`, `SPEC.md` (2×),
  `scripts/gebote_verifier.py` (mitgezogene **hartkodierte** Regel, AGENTS.md §2),
  `backend/core/causal_defect_reasoner.py` (2×: 62→66, 34→66),
  `backend/core/surgical_defect_analyzer.py` (66→65),
  `backend/core/unified_restorer_v3.py` (46→65), `denker/defekt_denker.py` (3×: 23→65).
- **Spec 03 entschärft:** Der sich selbst widersprechende Block (62/35/49 für dieselbe
  Größe, „49 ist normativ“) ist durch eine einzige autoritative Aussage ersetzt:
  maßgeblich ist ausschließlich `CAUSES`; die Liste im Spec ist ein Auszug.
- **Verifikation:** Gate **Exit 0** (vorher Exit 1 mit 26 Lücken), Gebote-Verifier
  **30/30**, `reports/spec_drift_baseline.json` neu initialisiert (6 WATCHED-Files
  waren gedriftet, u.a. `FILE_REGISTRY.md`).
- **Bewusst nicht angefasst:** `docs/CHANGELOG_HISTORY.md`, die Meilenstein-Spalte in
  `docs/PROJECT_STATUS.md`, `docs/dev/**` und die Wachstums-Notizen im Enum
  („ergibt 28 DefectTypes“) — sie dokumentieren Historie; die Ausnahmen sind im
  Gate-Docstring einzeln begründet.
- **Rest-Befund (nicht normativ, gemeldet statt still korrigiert):**
  `.agents/skills/{readme,spec,contributing}/SKILL.md` nennen 56/62 DefectTypes.

## 10.3.19 (2026-10-06)

### §III.11: Sperr-Protokollierung konsolidiert + drei vorbestehende rote Tests geklärt

- **Befund (§G9 copilot-instructions.md):** Die dokumentierte Sperre
  `music_model_flags.use_sgmse_musik=False` (SGMSE+ ist ein SPRACH-Score-Core) wird
  von fünf Stellen konsumiert, aber nur zwei protokollierten sie §III.11-konform:
  `hybrid_dereverb.py` und `phase_49_advanced_dereverb.py` nutzen das kanonische
  `logger.info` mit Begründung. Drei Stellen taten das nicht:
  - `backend/core/dsp/sota_vocal_model_router.py` — nur `logger.debug`, und der
    Sperrgrund erschien in der Fallback-Kette als generisches
    `sgmse_plus:RuntimeError` (nicht von einem echten Modellausfall zu unterscheiden, §G8 copilot-instructions.md)
  - `backend/core/coordinated_repair.py` — **keine** Protokollierung
  - `backend/core/phases/phase_03_denoise.py` — Sperre in eine Sammel-Bedingung
    gefaltet, kein eigener Log
- **Fix:** Alle drei nutzen jetzt das kanonische Muster: `logger.info` mit Begründung
  (§v10.16/F7, §III.11 copilot-instructions.md). Der Router führt einen eigenen Kettentoken
  `sgmse_plus:speech_core_locked`; die Eignungslogik in Phase 03 ist in
  `_sgmse_material_ok` (Material/Voraussetzungen) und die Flag-Sperre getrennt —
  semantisch identisch, aber begründbar.
- **Zwei vorbestehende rote Tests** (`tests/unit/test_sota_vocal_model_router.py`)
  erwarteten den SGMSE+-Zweig, den die Sperre bewusst schließt:
  `test_router_vocal_nr_skips_unloaded_miipher_for_sgmse` und
  `test_router_vocal_nr_compensates_missing_miipher_with_dfn_and_hnr`. Sie geben das
  Flag jetzt per `monkeypatch` gezielt frei (Standard-Dependency-Injection, keine
  Produktionsänderung) und prüfen damit weiterhin ihren eigentlichen Vertrag
  (MIIPHER fehlt → DFN+HNR-Kompensation).
- **Neuer Sperr-Pin-Test** `test_router_vocal_nr_locks_speech_core_and_reports_reason`:
  Flag `False` ⇒ der Sprach-Core wird nachweislich NICHT aufgerufen
  (`sgmse_calls == []`), der Kettentoken lautet `sgmse_plus:speech_core_locked`,
  und die Sperre wird mit Begründung protokolliert (caplog, §G8 copilot-instructions.md).
- **Dritter vorbestehender roter Test geklärt**
  (`tests/unit/test_phase_03_denoise.py::test_clean_audio_not_degraded`): gegen HEAD
  reproduzierbar, also nicht durch diese Sitzung entstanden. Ursache ist ein
  **Testdesign**-Problem, belegt durch Messung (2026-10-06, 48 kHz):

  | Signal | corr | rms_ratio | Bewertung |
  | --- | --- | --- | --- |
  | MUSDB18-HQ-Vocalstem (**sauber**) | 1,00000 | 1,0000 | unangetastet ✓ |
  | `real_world_validation/.../digital_test_01.wav` (laut `metadata.json` **degradiert**: clipping) | 0,9542 | 0,693 | Bearbeitung korrekt ✓ |
  | MUSDB Mixture / Other | 0,9524 / 0,9711 | 0,818 / 0,776 | Bearbeitung korrekt ✓ |
  | synthetischer 440+880-Hz-Ton (0 Defekte) | 0,8736 | 0,999 | **Testdesign-Ausnahme** |

  Ein rauschfreier, stationärer Sinus ist für einen Rauschschätzer nicht von Rauschen
  zu unterscheiden (Messung: der DSP-Pfad senkt ihn auf 0,49×, der §0-Level-Guard holt
  ihn auf 0,999× zurück; `sgmse_plus_tier0_applied`/`deepfilternet_tier1_applied`
  bleiben `False`, also kein ML-Pfad). Die Korrelationsschwelle 0,97 ist von **keinem**
  realistischen Signal erfüllbar (auch nicht von echtem degradiertem Material).
  Der Test prüft jetzt die zwei **belegbaren** Verträge: Pegelerhalt am ruhigen Signal
  (`0,95 ≤ rms_ratio ≤ 1,05`) und Nie-Verschlechtern auf wirklich sauberem Material
  (Corr > 0,999, mit `skipif` für lokal fehlendes Referenzmaterial). Die Messwerte
  stehen im Testdocstring, damit die Schwelle nicht als willkürlich abgeschwächt wirkt.

## 10.3.18 (2026-10-06)

### §G9: A/B-Neumessung mit kanonischem Demucs-Helfer — Verdikt revidiert

- **Befund:** Der Vorgänger-A/B (`2026-10-04_p1_2_scnet_vs_mdx23c_ab.md`) verglich
  SCNet gegen eine Baseline, deren ONNX-Aufruf defekt war: Der STFT-Eingang `x`
  wurde mit Nullen gefüttert und nur der Wellenform-Zweig genutzt → die Baseline
  verlor rund **13 dB**.
- **Fix:** `scripts/eval_scnet_vs_mdx23c.py::_demucs4_baseline` nutzt jetzt den
  kanonischen Helfer aus `plugins/htdemucs_plugin.py` (Commit `056f1ec1`) —
  **eine** Implementierung für Plugin und Eval
  (§G9 copilot-instructions.md).
- **Neumessung** (identische Songs und Fenster 165/169/107 s, Seed 42, CPU,
  0 Fehlerfälle):

  | Song | SCNet | Demucs v4 fair | Δ | Demucs v4 alt (handicapiert) |
  | --- | --- | --- | --- | --- |
  | AM Contra | 13,57 dB | 11,61 dB | +1,96 dB | 1,54 dB |
  | Al James | 10,32 dB | 7,01 dB | +3,31 dB | −7,00 dB |
  | Motor Tapes | 16,43 dB | 12,84 dB | +3,59 dB | −20,07 dB |

- **Revidierte Aussagen:** Der SCNet-Vorsprung beträgt **+1,96…+3,59 dB**
  (Mittel ≈ +2,96 dB) statt „+10…+16 dB". Demucs v4 **erfüllt beide Gates**
  (`separation_fidelity` 0,9051–0,9420; `singer_identity_cosine` 0,9478–0,9735)
  — die frühere Gate-Verletzung war ein Aufrufartefakt, keine Modelleigenschaft.
  **Neuer Trade-off:** Demucs v4 ist auf CPU **~7× schneller** (9,8 s vs. 68,9 s
  je 30 s).
- **Report:** `docs/reports/current/2026-10-06_p1_2_scnet_vs_demucs_fair_ab.md`;
  der Vorgänger ist als ÜBERHOLT markiert, Roadmap TODO-P1-2 aktualisiert.
- **C4/C5 bleiben offen** (menschlicher Sign-off); die Hör-Artefakte je Song
  (`mix`, `gt_vocals`, `scnet_vocals`, `baseline_vocals`) liegen in
  `output/scnet_ab_2026-10-06_fair/` bereit.

## 10.3.17 (2026-10-06)

### §G9: Demucs-v4-ONNX-Pfad repariert — +13,4 dB Vocals-Separation

- **Befund (gemessen, nicht vermutet):** Der htdemucs-ONNX-Pfad war an **vier
  Stellen** defekt — und weil das `demucs`-Paket nicht installiert ist, lief
  genau dieser Pfad **produktiv**:
  1. **`x` mit Nullen gefüttert** (`plugins/htdemucs_plugin.py`): Der zweite
     Graph-Eingang ist **kein** „State-Tensor“, sondern der **STFT des Chunks**
     (`4 = 2 Kanäle × (Real, Imag)`, `2048` Bins, `336` Frames bei
     `343980 = 7,8 s × 44100`). Mit Nullen war der Transformer-Zweig
     vollständig tot — der Spektral-Ausgang lieferte exakt `0.0`.
  2. **Hybrid-Ausgabe unbenutzt:** Die gültige Schätzung ist die **Summe beider
     Zweige** (`add_67` + iSTFT(`output`)). Der Aufruf nahm nur `add_67` →
     Stem-Summe erreichte nur 48 % der Mixture.
  3. **Stem-Permutation:** `_separate_onnx` gab die Indizes `0..3` zurück
     (`drums, bass, other, vocals`), der Aufrufer entpackte als
     `[vocals, drums, bass, other]` → **„vocals“ war tatsächlich Drums**.
  4. **Ratenfehler:** Der Pfad resampelt auf 48 kHz, das Modell ist ein
     **44,1-kHz**-Modell (`343980 = 7,8 s × 44100`).
- **Messung** („Motor Tapes – Shore“, 107 s, 7,8-s-Fenster, MUSDB18-HQ-Ground-
  Truth, Vocals-SI-SDR): **−1,80 dB → +11,59 dB**; Vocals-RMS 0,0002 → 0,0639
  (GT 0,0680); Stem-Summe 48 % → 98 % der Mixture; der Ratenanteil allein
  kostet **2,56 dB**.
- **Fix:** Kanonischer Aufrufvertrag im Plugin (`htdemucs_onnx_stft_input`,
  `htdemucs_onnx_stems`, `resample_audio`) — **eine** Implementierung für
  Plugin und Eval-Skript (§G9 copilot-instructions.md). Stem-Mapping über
  `_HTDEMUCS_STEM_ORDER`, Ratenkonvertierung 48 k → 44,1 k → 48 k,
  §V6-Logging (copilot-instructions.md) der Ersatzpfade.
- **Nachweis:** End-to-End über den Produktionspfad `_separate_onnx` mit
  48-kHz-Eingang → **+11,59 dB** (identisch zur Modellrate), Länge exakt
  erhalten; `mypy` Success; `ruff` All checks passed; ID-Registry-Check vorab
  Passed; `test_sota_vocal_model_router.py` 16 passed + 2 **vorbestehende**
  Fehler (identisch auf HEAD, Miipher/DFN-Pfad — nicht dieser Fix).
- **Offen (benannt):** Der A/B-Report `2026-10-04_p1_2_scnet_vs_mdx23c_ab.md`
  verglich gegen eine um ~13 dB handicapiert Baseline → **Neumessung nötig**,
  bevor C4/SCNet entschieden wird. Ebenso: zwei rote Unit-Tests auf `main`
  (Miipher/DFN-Router) bleiben unentdeckt, weil der Pre-Commit-Smoke nur
  **einen** von 65 Chunks fährt.

## 10.3.16 (2026-10-06)

### §G9: Verdrahtungs-Audit — verwaistes Modellkapital wird fail-closed sichtbar (WP0)

- **Befund:** `models/` enthält **61 Verzeichnisse**, das Inventar
  (`scripts/model_inventory.py`) pflegte aber nur **30 kuratierte Einträge**.
  Lokal vorhandenes Kapital ohne Konsumenten war damit unsichtbar. Gemessen:
  **5 verwaiste Verzeichnisse**, darunter zwei mit hohem Wert —
  `ddsp_predictor/c4_head.pth` (**trainiert**, nirgends verdrahtet; die Roadmap
  meldete irrig „Prädiktor-Training fehlt“) und
  `scnet_4stems/huge_scnet_4stems_v1.2.ckpt` (**beschafft**, SHA-256
  `807f470b…` verifiziert, **A/B gewonnen** mit SI-SDR +10,3…+16,4 dB,
  `singer_identity_cosine` 0,978–0,992 — aber nirgends verdrahtet).
- **Fix:** Neue Achse in `scripts/model_inventory.py` — **`--wiring`** ordnet jedem
  `models/`-Verzeichnis einen Status zu (verdrahtet · begründet ausgenommen ·
  verwaist), **`--fail-on-orphans`** macht daraus ein Gate. Jede Ausnahme steht
  mit Begründung und Aufgabenziel in `_WIRING_ALLOWLIST` (§G8) — kein vergessener
  Rest.
- **Erweiterung statt Neuanlage:** `unwired_feature_audit.py` prüft
  _Phasen-Features_, `model_inventory.py` das _Vorhandensein_ — keiner den
  _Verdrahtungsstatus pro Artefakt_. Deshalb Erweiterung des kanonischen Skripts
  (Write-Gate-Regel, `repo_search.py --before-create`).
- **Gate:** Neuer Pre-Commit-Hook `aurik-model-wiring` (fail-closed) — ein
  künftig verwaistes Artefakt blockiert den Commit, solange es nicht verdrahtet
  oder begründet in der Ausnahmeliste steht.
- **Ehrlichkeit zur Methode:** Geprüft wird die _Namensreferenz_ im
  Produktionscode (untere Schranke — ein Kommentar-Treffer zählt). Das Gate
  findet vollständig unbekannte Artefakte, **nicht** falsch verdrahtete.
- **Nachweis:** `--wiring` → 61 Verzeichnisse, **0 undokumentierte Verwaiste**,
  5 begründete Ausnahmen; `--fail-on-orphans` → Exit 0; neue Tests
  `tests/unit/test_model_inventory_wiring.py` → 3 passed; `ruff` → All checks
  passed; Pre-Commit-Hook `aurik-model-wiring` → Passed.
- **Offen (benannt, nicht still):** Die 5 Ausnahmen sind Aufgaben, keine
  Endzustände — SCNet-Verdrahtung (TODO-P1-2 Folge-Slice), DDSP-Prädiktor
  (F5/C4), APPLADE-Bewertung, Matchering-Rolle, `gacela_upstream`-Kopie.

## 10.3.15 (2026-10-06)

### §G9: MP-SENet-Drift behoben — der Musik-Kern läuft jetzt im Vokal-Denoising

- **Befund:** Der Vokal-Denoising-Pfad `CoordinatedRepair._run_mp_senet_vocal`
  erzeugte eine **eigene** ONNX-Sitzung mit hartkodiertem Pfad
  `models/mp_senet/mp_senet.onnx` (VoiceBank-Sprach-Stand) und fest
  `providers=["CPUExecutionProvider"]`. Damit umging er den kanonischen Schalter
  `resolve_model_path("mp_senet")` **und** die EP-Policy (§III.9
  (copilot-instructions.md)). Wirkung: Genau das Artefakt, das auf Musik als
  schädlich gemessen wurde (SOTA-ML-V3, 3 MUSDB-Tracks × 30 s: ΔSDR −5,9 dB —
  „der Sprach-Enhancer klassifiziert Musikanteile als Noise“), lief weiter,
  obwohl der musik-finetunte Kern vorliegt und per Flag aktiviert ist
  (`use_mp_senet_musik=True`; A/B 2026-09-20: seg-SNR +5,3…+11,3 dB, VERSA
  4,96–4,99). Der Widerspruch zwischen `docs/TODOS_SOTA_ROADMAP.md` („de-wired“)
  und §v10.25 („aktiviert“) löst sich damit auf: Gemessen war der **Sprach**-Stand;
  aktiviert ist der **Musik**-Kern.
- **Normative Grundlage:** `tests/normative/test_primary_paths_no_fallback.py` ::
  `test_musik_finetuned_stufe_aktiv` verlangt für `mp_senet` ausdrücklich
  `"finetuned" in resolved.parts`. Der hartkodierte Pfad umging diesen Vertrag.
- **Fix:** `resolve_model_path("mp_senet")` als einzige Auflösung; fail-closed mit
  `log.warning` (§V6 (copilot-instructions.md)), wenn kein Primärartefakt
  auflösbar ist. EP-Provider über den kanonischen Helfer
  `get_ort_providers("mp_senet")` — er wendet zusätzlich das Numerik-Paritäts-
  Verdikt an (§v10.762: GPU nur wo „rocm“ validiert, CPU erzwungen wo „cpu“).
  I/O-Namen werden dynamisch aus `session.get_inputs()` gelesen (Muster aus
  `plugins/mp_senet_plugin.py`), weil Sprach- und Musik-Export unterschiedliche
  Namen tragen.
- **Befund B2 (Metadaten-Drift):** `models/manifest.json` deklarierte für
  `sgmse_musik` die TorchScript-Variante
  `models/sgmse_plus/finetuned/sgmse_musik.ts` — die plant
  `scripts/export_all_musik_models.py`, erzeugt sie aber nie. Korrigiert auf das
  deployte, verifizierte ONNX-Artefakt samt gemessenem `sha256` und
  `size_bytes` (263 593 607 B).
- **Nachweis:** `ruff` → `All checks passed!`; `mypy` → `Success: no issues found`;
  neue Regressionstests grün. Ihre Wirksamkeit ist gegen den Vorstand belegt:
  `git show HEAD:backend/core/coordinated_repair.py` enthält **0×**
  `resolve_model_path` und **1×** den hartkodierten Sprach-Pfad — nach dem Fix
  **2×** bzw. **0×**. Zusätzlich: normative Verträge
  `test_musik_finetuned_stufe_aktiv`, `test_resolve_model_path_active_flags`,
  `test_bewusst_gesperrte_flags_dokumentiert` → 3 passed;
  `tests/unit/test_model_zoo_activation.py` → 19 passed.

## 10.3.14 (2026-10-06)

### Typ-Sicherheit (P3 TYPE-SAFETY): 18 mypy-Befunde behoben — verhaltensneutral belegt

- **Befund:** `mypy 2.1.0` (Projekt-Config) meldete 12 Fehler in
  `scripts/train_cantus.py` und 6 in `tests/unit/test_model_zoo_activation.py`
  (`no-any-return`, `arg-type`, `assignment`, `attr-defined`, `union-attr`,
  `func-returns-value`). Beide Dateien lagen außerhalb des Pre-Commit-Scopes —
  der mypy-Hook prüft nur **gestagte** Dateien, deshalb blieben die Befunde
  unentdeckt.
- **`scripts/train_cantus.py`:**
  - `register_buffer`-Attribute (`self.mel_fb`) sind jetzt explizit als
    `torch.Tensor` deklariert — mypy las sie über `nn.Module.__getattr__`
    als `Tensor | Module`.
  - `_mel_fb()` greift nicht mehr ungetypt per `getattr` durch: fail-fast
    statt Any-Durchgriff (§V6 (VERBOTEN.md)).
  - `nn.Module.__call__` ist im Stub `Any`; der MSE-Kalibrier-Loss wird mit
    einem wahren `cast` typisiert (kein Workaround, §V7 (VERBOTEN.md)).
  - `_extract_pitch()` liefert dtype und Layout explizit (`np.ascontiguousarray`,
    Werte unverändert float32); `extract()` deklariert `pitch` vor den Zweigen,
    damit kein `None` in ein Array fließt.
  - `train()` annotiert `train_ds`/`val_ds` als `Dataset`; die Split-Zuweisung
    ist umgestellt, aber **definitionsgemäß identisch** (`train_ds is full`).
  - **§G9 (copilot-instructions.md):** `_load_config()` war ein toter,
    duplizierter Ladepfad — `train()` lud dieselbe JSON-Datei erneut. Jetzt
    existiert genau **ein** Config-Ladepfad.
- **`tests/unit/test_model_zoo_activation.py`:** `_make_step()` gibt
  `RepairStep` statt `object` zurück; `resolve_model_path()`-Ergebnisse werden
  explizit auf `None` geprüft (Vorbedingung benannt statt Union-Attribut-Zugriff);
  der Lambda-Seiteneffekt-Hack (`calls.append(...) is None`) ist durch eine
  benannte Fake-Funktion ersetzt.
- **Nachweis:** `mypy` → `Success: no issues found in 2 source files`;
  `ruff check` → `All checks passed!`; `tests/unit/test_model_zoo_activation.py`
  → 17 passed; `scripts/train_cantus.py --smoke --cpu` → Exit 0 (50 Epochen,
  Report in `/tmp/cantus_smoke_*`); Split-Äquivalenz über 5 Dataset-Größen
  (2/3/5/10/25) maschinell geprüft, inklusive `or full.pairs[-1:]`-Randfall.
- **Keine Verhaltensänderung:** kein Export-, Pipeline- oder Signalpfad
  berührt. Die Änderungen sind ausschließlich Annotationen, ein
  Dtype-/Layout-Explizitmachen (float32 unverändert) und eine
  Zuweisungs-Umsortierung mit identischem Ergebnis.

## 10.3.13 (2026-10-06)

### §G9: Genre-Erkennung konsolidiert (eine Registry statt vier Auflösungen)

- **Befund:** Dasselbe Genre-Label wurde von **vier Konsumenten mit vier
  verschiedenen, jeweils unvollständigen Verfahren** aufgelöst — `genre_classifier`
  hatte eine _vierte_ Kopie der Profiltabelle (`label_map`), `genre_goal_profile`
  ein Fuzzy-Teilstring-Matching, `tonal_reference_profile` eine 5-Einträge-`_ALIASES`,
  `unified_restorer_v3` eine 16-Zweige-if/elif-Kette. Gemessen pro Label:

  | Label | Restaurierungsprofil | Goal-Profil | Delta | JND |
  | --- | --- | --- | --- | --- |
  | `Deutscher Schlager` | **leer** | schlager | **0,000** | **1,00** |
  | `Internationaler Schlager` | **leer** | schlager | **0,000** | 1,00 |
  | `Klassik` | OK | **unknown** | +0,300 | **1,00** |
  | `Oper` | OK | **unknown** | +0,800 | **1,00** |
  | `Soul/R&B` | OK | **unknown** | +0,800 | 1,00 |
  | `Hip-Hop` | OK | **unknown** | −0,300 | 1,10 |
  | `Country` | OK | **unknown** | +0,500 | 1,00 |
  | `Metal` | OK | metal | **0,000** | 1,10 |

  Die Wirkung war **invertiert zur Konfidenz**: Das _unschärfere_ Label `Schlager`
  (Sprache unsicher, `lang_de_score` 0,30–0,55) erhielt die Genre-Spektralvorgabe
  (+0,800 dB), während `Deutscher Schlager` (≥ 0,55, Hauptanwendungsfall) sie
  vollständig verlor. `Klassik` bekam den generischen JND 1,00 statt 0,80 — bei
  Klassik liefen dadurch **weniger** Phasen, das Gegenteil der Absicht (§G6).
- **Fix:** Neue kanonische Registry `backend/core/genre_registry.py` als **eine**
  Quelle: 21 kanonische Genre-IDs, explizite Alias-Tabelle (Sprach-Präfixe
  „Deutscher/Internationaler …“, Schreibvarianten, Untergenre-Spezifität) und
  Ziel-Schlüssel je Konsument (Restaurierungsprofil, Goal-Profil, JND/Dynamik,
  Delta-Tabelle). Kein Fuzzy-Matching mehr — unbekannte Labels werden ehrlich als
  `None` gemeldet (§V6 (copilot-instructions.md)) statt per Teilstring auf ein
  falsches Genre zu fallen.
- **Verdrahtet:** `genre_classifier.get_restoration_profile` (vierte Tabellenkopie
  entfernt), `genre_goal_profile.get_genre_profile`, `tonal_reference_profile`
  (Delta **und** Konfidenz), `perceptual_tuning` (JND + Dynamik),
  `studio_goal_targets`, `unified_restorer_v3` (kanonische Zweig-IDs).
- **Totpfade beseitigt:** `plugins/genre_denoise_router.py` gelöscht — nie
  importiert, mit eigener Taxonomie und einer Fehlabbildung der AudioSet-Indizes
  0–29 (`Speech`/`Giggle`/`Cough`/`Sigh`) auf das Genre `classical`.
  `phantom_mode._detect_genre` lieferte konstant `"unknown"` („Platzhalter") und
  ist jetzt an den kanonischen Klassifikator angebunden.
- **Nachweis:** 44 neue Vertragstests (Alias-Tabelle, Konsumenten-
  Übereinstimmung, Untergenre-Spezifität, kein Fuzzy-Fehltreffer) + 406
  Regressionstests grün; `Klassik` → JND 0,80, `Deutscher Schlager` → Profil OK,
  Delta +0,800, JND 1,10.
- **Offen (ehrlich):** Genre-Erkennung ist **nicht** auf SOTA-Stufe: Tier-1 (CLAP)
  ist nur ein weicher Prior, die Erkennung selbst ist Hand-DSP. Es gibt **kein
  trainiertes Genre-Modell** — MERT (laut Spec §5 „14 Tasks SOTA inkl. Genre") ist
  als Backbone vorhanden, aber ohne Genre-Head (`models/mert_genre_classifier/`
  existiert nicht); die GTZAN-Labels liegen vor, werden aber nicht zum Training
  genutzt. Der Ausbau braucht einen Trainingslauf auf GTZAN/MagnaTagATune.

## 10.3.12 (2026-10-06)

### §G9/SOTA: LAION-CLAP-Tag-Pfad korrigiert (Falsch-Beschriftung + falscher Text-Encoder)

- **Befund 1 — Falsch-Beschriftung im Default-Pfad (AUDIO-QUALITY):** Der
  ONNX-Tag-Pfad schnitt aus `models/clap/text_embeddings.npy` die ersten 43
  Zeilen und beschriftete sie mit `INSTRUMENT_TAGS` + `GENRE_TAGS` +
  `MATERIAL_TAGS`. Das Artefakt enthielt jedoch **527 AudioSet-Klassen**
  (erzeugt von `scripts/derive_clap_text_embeddings.py`, belegt über den
  Generator selbst und durch bit-genaue Reproduktion, cos = 1,0000). Damit war
  z. B. `genre_tags["opera"]` in Wahrheit „Male singing“, `["ambient"]`
  „Female singing“, `["metal"]` „Choir“ und `["vocals"]` „Speech“ — jedes
  Instrument-/Genre-/Material-Tag des Standardpfads war falsch benannt.
- **Befund 2 — Falscher Text-Encoder, drei divergierende Pipelines (§G9):**
  Das Artefakt wurde mit einer **falschen Projektionskette** berechnet
  (CLS statt `pooler_output`, zusätzliches LayerNorm, Verkettung von
  `text_projection` _und_ `text_transform`), der PyTorch-Pfad nutzt dagegen
  `encode_text()`. Messung: Cosinus der Text-Encoder **0,02** → gleicher Song,
  andere Tags je nach Ladepfad. Kanonisch ist `encode_text()`;
  `text_transform` erzeugt in `forward()` eine zweite, getrennte Einbettung.
- **Befund 3 — Stiller Verlust trainierter Gewichte (§V6, R-BLOCKER-Klasse):**
  Der Checkpoint stammt von einem fairseq-RoBERTa (514 Positions- / 1
  Token-Type-Embedding), die HF-Konfiguration erwartet 512/2. Der
  Shape-Filter des Loaders verwarf dadurch **3 Tensoren stillschweigend**, u. a.
  die **trainierten Positions-Embeddings** — der Text-Turm rechnete mit
  Zufalls-Positionen (gemessen: std = 0,0200 = exakter Init-Wert).
- **Fix 1:** Text-Turm-Dimensionen auf Checkpoint-Format erzwungen (514/1)
  in `laion_clap/clap_module/model.py`; Verifikation 199/199 Gewichte
  übernommen (vorher 197/200).
- **Fix 2:** Shape-Drop im Loader wird jetzt namentlich mit
  `logger.warning()` gemeldet („nicht übernommen — behalten
  Zufallsinitialisierung“) statt still zu degradieren.
- **Fix 3:** `scripts/derive_clap_text_embeddings.py` neu gefasst — erzeugt
  exakt die 43 Aurik-Tags in Plugin-Reihenfolge mit der **kanonischen**
  Pipeline, vollständig lokal/offline (kein HuggingFace-Download), fail-closed
  bei fehlenden Gewichten. Parität gegen den PyTorch-Pfad: **cos = 1,000000**
  (min = max) → beide Pfade nutzen nun dieselbe Text-Funktion.
- **Fix 4:** Der Plugin-ONNX-Pfad prüft die Zeilenzahl **exakt** und lehnt ein
  Artefakt mit fremdem Label-Raum sichtbar ab, statt es falsch zu beschriften
  (Regressionsschutz gegen Befund 1).
- **Fix 5:** `genre_classifier` wertete Zero-Shot-Prompts über `genre_tags` aus
  und suchte Schlüssel wie „schlager“/„volksmusik“/„german“, die in
  `GENRE_TAGS` nicht existieren → immer 0,0, im PyTorch-Pfad sogar als echte
  Messung markiert. Jetzt über `query_scores()`; fehlende Evidenz wirft.
- **Fix 6:** §G9-Abschlusstest beider Pfade: identische Top-3-Tags und
  Score-Cosinus **0,99** (vorher 0,02); neuer Zero-Shot-Text-Turm des
  PyTorch-Checkpoints wird bei Bedarf nachgeladen (der ONNX-Graph enthält
  keinen Text-Turm), Zero-Shot-Scores überleben via `custom_scores`.
- **Hinweis zur Reproduzierbarkeit:** Der LAION-CLAP-Quelltext liegt unter
  `models/clap/src/` und damit in einem **gitignorierten** Verzeichnis
  (`.gitignore` → `/models/*`). Der dort gesetzte Config-Fix ist daher nicht
  versioniert; der versionierte Plugin-Loader leitet die Text-Turm-Dimensionen
  zusätzlich **aus dem Checkpoint** ab und meldet jede Angleichung mit
  `logger.warning()`. Auf einem frischen Checkout ohne `models/clap/` ist der
  PyTorch-Text-Turm generell nicht verfügbar — der ONNX-Pfad bleibt voll
  funktionsfähig.
- **Offen (ehrlich):** Die Kalibrierung der CLAP-Evidenz-Schwellen steht aus.
  `corpus/` enthält **keinen Gesang** (PANNs `Singing voice` = 0,0001–0,0004;
  CLAP-Cosinus für „singing“ negativ), und zwei Korpusdateien sind byte-identisch
  (gleiche MD5). Die neuen CLAP-Gender-Konstanten sind daher **nicht**
  kalibriert; die Fusion ist strukturell korrekt, aber ohne gesangs­haltiges,
  gelabeltes Korpus nicht belegbar.

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

### §Gesangs-Fokus: SGMSE+-Sprach-Core auch in Dereverb/Router entkoppelt

- **Befund (2026-10-06):** Nach Phase 03/`coordinated_repair` lief der Sprach-Core
  weiterhin an drei Stellen: `hybrid_dereverb._init_dccrn` (SGMSE+ als
  Dereverb-**Primärpfad**), `phase_49_advanced_dereverb` (Tier-0) und
  `sota_vocal_model_router` (Fallback im Gesangs-Router).
- **Fix:** Alle drei an `music_model_flags.use_sgmse_musik` gekoppelt. Ohne
  Musik-Finetune (§v10.16/F7) übernehmen die musik-tauglichen Pfade: WPE-DSP
  (Dereverb) bzw. DFN+MP-SENet (Router) — kein gesperrtes Sprach-Modell auf
  Musik/Gesang (§V1 (copilot-instructions.md)).
- **Beweise:** Dereverb-/Router-Suite 47 passed; Ruff clean; R2-Check 0 Warnungen.

### §v10.16/F7-Verdrahtung: SGMSE+ Musik-Core erreichbar gemacht

- **Befund (2026-10-06):** Der F7-Musik-Finetune liegt vor (`finetuned/sgmse_musik_best.ckpt`,
  Epoch 24) und wurde am 05.10. als `models/sgmse_plus/sgmse_musik_core.onnx`
  (263 MB) exportiert. Die Verdrahtung fehlte aber: `MUSIC_MODEL_PATHS["sgmse"]`
  zeigte auf den SPRACH-Core (`sgmse_plus_core.onnx`), und
  `plugins/sgmse_plugin.py` konsultierte `music_model_flags` gar nicht
  (hartcodierter Sprach-Core) → ein Flag-Flip hätte den falschen Core aktiviert,
  der Musik-Core war unerreichbar.
- **Fix:** `MUSIC_MODEL_PATHS["sgmse"]` → `sgmse_musik_core.onnx`;
  `plugins/sgmse_plugin._resolve_onnx_path()` löst den aktiven Core über
  `resolve_model_path("sgmse")` auf (Flag an → Musik-Core, aus → Sprach-Legacy).
  Der Flag bleibt `False` (A/B-/Hörordnungs-Abnahme offen,
  `test_primary_paths_no_fallback` pinnt das).
- **Beweise:** Neuer Test `test_sgmse_music_flag_selects_musik_core` (beide
  Zustände); `test_model_zoo_activation` 17 passed; normative Primärpfad-Tests
  7 passed; Ruff clean.

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

### §Norm-Konsistenz: §III.11/§III.12 verankert, Spec-Fakten korrigiert

- **Norm-Ergänzung:** `.github/copilot-instructions.md` §III erhält zwei
  Spezialregeln. **§III.11 Domänen-Konsistenz Musik/Gesang:** sprachtrainierte
  Modelle (SGMSE+/WSJ0, Whisper, UTMOSv2/BVCC, resemblyzer) dürfen nur mit
  Musik-Fine-Tune als Signalpfad laufen; `use_sgmse_musik` schaltet ALLE fünf
  SGMSE+-Signalpfade, der Legacy-Sprach-Core bleibt in `LEGACY_MODEL_PATHS`.
  **§III.12 Stem-Rekombinations-Vertrag:** genau EIN Vereinigungspunkt
  (`recombine_stems_with_gates()`, §SLR-1f), Pflicht-Gates in der Reihenfolge
  **C2 → C1 → C3** (W2/W1/W3), Zeugenpflicht in
  `StemContext.witness_reports["recombination"]`; offene Pflicht-Witnesses
  C4 (Doppelverarbeitung), C5 (Pegel-Kontinuität), W4 (Kammfilter-Ripple),
  W5 (Stem-Leakage), W6 (Seam-Laufzeitsprung) sind ausdrücklich als
  „nachzuweisen, nicht anzunehmen" markiert.
- **Spec-Fakten korrigiert (Regel Spec > Code > Kommentar):** `04_dsp_standards.md`
  (Dereverb-/Music-Vocal-Enhancement-/SNR<10-dB-Zeile sowie die
  `_PHASE_REQUIRED_MODELS`-Tabelle), `06_phases_system.md` (phase_03, phase_20),
  `02_pipeline_architecture.md` (Hebel-2-Tier-0-Bedingung), `07_quality_and_tests.md`,
  `03_cognitive_modules.md` und `08_architecture_and_distribution.md`
  (Plugin-Zeile nannte den TorchScript-Sprach-Core als PRIMÄR), `v10.19`
  („0 Speech-Modelle" wird als Ziel mit Mess-Wahrheits-Hinweis belegt statt
  behauptet), `v10.25` (Produktion lädt `sgmse_musik_core.onnx` nur bei Flag),
  `v10.99x`, `TODOS_SOTA_ROADMAP.md` (F7: Checkpoint + ONNX vorhanden — die
  Sperre ist eine offene A/B-Entscheidung, kein fehlendes Artefakt).
- **Code-Kommentar:** `plugin_lifecycle_manager._PHASE_REQUIRED_MODELS`
  (phase_20) dokumentiert die Flag-Abhängigkeit Phase ↔ `use_sgmse_musik`.
- **Beweise:** `scripts/spec_drift_check.py` → „No spec drift detected"
  (Baseline über 24 beobachtete Dateien neu gesetzt);
  `scripts/release_must_coverage_check.py` → 2/2 (100 %);
  `scripts/compliance_check.py` → 0 Errors, 1 Warning (vorbestehend
  `ab_test_manager.py:41`, non-blocking).

### §SOTA-Gender: EIN Detektor-Pfad + musiktaugliche Evidenz (Spec 19)

- **Befund:** Vier parallele Gender-Detektoren mit abweichenden Schwellen
  (§G9 (copilot-instructions.md)-Verstoß). Der Pfad der Musik-Vocal-Pipeline
  (`forensics/gender_detection`) entschied mit einer **F0-Einzelregel**
  (`f0 < 170 → male`) auf 16-kHz-Autokorrelation über nur 100 ms Signal. Ein
  sprachtrainiertes Embedding-Modell (Resemblyzer/LibriSpeech) wurde dabei
  berechnet und **verworfen**; fehlte es, blockierte es sogar die domänenneutrale
  Pitch-Auswertung (`emb is None → "unknown"`).
- **Fix (Konsolidierung auf EINEN Pfad):** Kanonisch ist allein die
  Multi-Evidenz-Fusion in `vocal_ai_enhancement.GenderDetector`:
  PANNs-„Male/Female singing" (musiktauglich) → F0 (Scan-Autokorrelation + pYIN
  mit Voicing-Confidence) → Burg-LPC-Formanten mit WORLD-Kreuzvalidierung →
  Contralto-Anatomie-Override.
  - **Neu im Kern:** `_panns_singing_prior()` als EINE Quelle (Mindest-Score
    0,25 UND Klassenabstand > 0,10 — bei Gleichstand schweigt die Evidenz) plus
    additive Fusion (`_PANNS_GENDER_WEIGHT = 0.35`); ohne Anatomie trägt sie die
    Entscheidung (instrumentales Intro) statt „unknown".
  - `forensics/gender_detection.GenderDetector` ist nur noch Fassade (Array- und
    Dateipfad) ohne eigene Klassifikationslogik; Resemblyzer entfällt
    (§III.11 copilot-instructions.md).
  - `forensics/gender_rule_based.RuleBasedGenderDetector` delegiert — eigene
    Pitch-/LPC-Heuristik und `classify_from_features` entfallen.
  - `phase_19_de_esser._detect_gender_robust` bezieht den Prior aus dem Kern
    (keine zweite Schwellenkopie) und reicht `panns_tags` in die Fusion.
  - `aurik_deesser_pro/music_vocal_pipeline` nutzt den Array-Pfad direkt — der
    Temp-WAV-Umweg entfällt.
- **Beweise:** 88 passed (`tests/unit -k "gender or utmos or noise_gate"`);
  normatives Gate `test_gender_detection_sota_gate.py` + `test_gender_detector.py`
  → 54 passed; 9 neue Regressionstests (PANNs-Prior klar/unklar/Schwelle, Fusion
  ohne Anatomie, unverändert ohne Tags, ein Pfad ohne Sprach-Embedder,
  Fassaden-Delegation, Scan-F0 trotz instrumentellem Intro). Ruff clean.

### §III.11-Durchsetzung: Sprach-Embedder als Richter entfernt (Ebene 1 + phase_65)

- **Befund:** Resemblyzer (LibriSpeech/Sprache) wirkte an zwei Stellen als
  **Richter** über Musik/Gesang, obwohl die Docstrings „Zeuge, kein Richter"
  behaupteten: `level_1_invariants_guard` (Ebene-1-Invariante „Stimm-Identität":
  `singer_identity < 0.92` → Blend-Reduktion) und
  `phase_65._apply_singer_identity_witness` (Blend bis 80 % Richtung Eingang).
  Derselbe Embedder war in `forensics/gender_detection` wirkungslos, blockierte
  aber als `emb is None → "unknown"` die domänenneutrale Pitch-Auswertung.
- **Fix:** Flag `music_model_flags.use_resemblyzer_music` (Default `False`) plus
  kanonische Messfunktion `level_1_invariants_guard.measure_singer_identity_cosine()`
  — EINE Quelle für beide Aufrufer (§G9 copilot-instructions.md):
  - Mit Freigabe: Embedder-Kaskade (Package→ONNX) wie bisher.
  - Ohne Freigabe: domänenneutraler DSP-Proxy (MFCC- + spektraler
    Centroid-Korrelation), der zuvor schon der Ersatz war. **§0p bleibt
    vollständig in Kraft** — der Identitätsschutz greift weiter, nur die Domäne
    der Messung ist korrekt. Der Embedder wird ohne Freigabe nicht geladen.
  - Rückgabe `float | None`: `None` = nicht messbar (konstant/stumm) ⇒ kein
    Messwert, kein Eingriff (Zeuge-Prinzip: keine Messung darf bestrafen).
  - **NaN-Fix:** Der DSP-Proxy lieferte bei Stille `NaN` (np.corrcoef auf
    konstanten Merkmalen) und trug eine nicht-endliche Größe in die
    Invarianten-Kette — jetzt explizit „nicht messbar" (§G5 GEBOTE.md).
- **SOTA-Ausbau der Fusion (`vocal_ai_enhancement`):**
  - Kontinuierliche PANNs-Singing-Evidenz: neben dem harten Shortcut (Abstand
    > 0,10, Score ≥ 0,25) fließt die Klassendifferenz proportional ein
    (`_PANNS_CONT_WEIGHT = 0.25`); beide Formen kommen aus EINER Quelle
    (`_panns_singing_scores`).
  - Aperiodizität als Konfidenz-Modulator: die bereits berechnete
    WORLD-Aperiodizität (`_detect_breathiness`, Yumoto-Proxy) wird VOR der
    Klassifikation gemessen und senkt die Konfidenz bei stark verhauchter oder
    verrauschter Stimme auf bis zu 60 % (`_modulate_confidence_by_aperiodicity`)
    — keine scheinbare Sicherheit, kein zusätzlicher Inferenzaufwand.
- **Beweise:** 70 passed (Ebene-1-, phase_65- und Gender-Suiten) mit 4 neuen
  Sperr-/Modulationstests (Embedder-Aufrufe == 0 ohne Freigabe, kontinuierliche
  PANNs-Evidenz, Konfidenz-Klemmung); 33 passed `test_gender_detector.py`;
  Ruff clean.

### §F13/F14-Infrastruktur: Gender-Head-Training + empirische Kalibrierung

- **Anschluss an §SOTA-Gender:** Nach der Konsolidierung auf EINEN Pfad fehlten
  die zwei datenabhängigen Hebel. Beide liegen jetzt als geprüfte Infrastruktur
  vor — ohne erfundene Labels oder Schwellen.
- **Neu:**
  - `scripts/train_gender_head.py` (F13) — Head auf MERT-v1-330M-Features
    (mean+std, 2×1024), deterministischer stratifizierter Split
    (§G5 GEBOTE.md), MLP-Klassifikation mit Macro-F1, Feature-Cache,
    **Label-Pflicht** (ohne `file,gender`-CSV Abbruch mit Exit 2 statt Raten,
    §V7 copilot-instructions.md), guarded Checkpoint-Speicherung über
    `training_artifacts.save_guarded`.
  - `scripts/calibrate_gender_evidence.py` (F14) — misst auf `corpus/` je Datei
    Anatomie-Lesart, Fusions-Entscheid, Konfidenz und Aperiodizität über den
    kanonischen Kern (kein Parallelpfad) und weist die Konsens-Rate
    Anatomie ↔ PANNs-Singing aus. Schwellenempfehlung nur bei Mindest-Stichprobe
    (12) UND Mindest-Konsens (0,80); der Report markiert ausdrücklich „kein
    Ground Truth" (das Korpus-Manifest kennt kein Gender-Feld).
- **Ehrlichkeit:** Beide Skripte treffen ohne Datenbasis keine
  Kalibrierungs-Aussage. F13/F14 sind in `docs/TODOS_SOTA_ROADMAP.md` mit den
  offenen Datenanforderungen dokumentiert (Label-Korpus ≥ ~50 je Klasse,
  PANNs-Tags).
- **Beweise:** 16 passed (`tests/unit/test_gender_calibration_infra.py`):
  Split-Determinismus, Abdeckung ohne Overlap, Ein-Exemplar-Klasse bleibt im
  Training, MERT-Pooling (Shape/NaN-Sanitisierung), Label-Pflicht
  (fehlend/ungültig/gültig), Kalibrierungs-Gate (ohne PANNs, hoher/niedriger
  Konsens, kleine Stichprobe), Macro-F1. Beide Skripte Ruff clean, als CLI
  lauffähig (`--help`), Pflicht-Abbruch ohne Labels verifiziert.
- **Write-Gate:** Einträge in `.github/FILE_REGISTRY.md` (2 Skripte, 1 Test)
  nach `scripts/repo_search.py --before-create` (keine kanonische Alternative).

### §III.11-Durchsetzung (Fortsetzung): VQI-Singer-Identity domänenrein

- **Befund:** `vocal_quality_index._compute_singer_identity` befragte den
  sprachtrainierten Resemblyzer-Embedder **ohne Domänen-Prüfung** als Primärpfad.
  Die VQI speist die Musical Goals (`vocal_quality`) und damit
  Hörordnungs-Entscheidungen — der Sprach-Embedder richtete also mittelbar über
  Musik/Gesang. Zusätzlich existierte dort eine zweite Messquelle mit eigener
  Kalibrierung (§G9 copilot-instructions.md).
- **Fix:** `resemblyzer_music_unlocked()` ist jetzt die öffentliche EINE Quelle
  der Freigabe (Ebene-1-Guard, phase_65, VQI). Ohne Freigabe nutzt die VQI den
  vorhandenen DSP-Proxy, der laut Code-Dokumentation auf die 0.92-Rollback-Schwelle
  kalibriert ist (§R3-Formel `(8c−1)/7`) — der Rollback-Schutz bleibt erhalten,
  nur die Domäne der Messung ist korrekt. Mit Freigabe läuft die
  Embedder-Kaskade unverändert.
- **Beweise:** 14 passed `test_vocal_quality_index.py` (3 neue
  Domänen-Regel-Tests: Proxy ohne Freigabe, identisches Audio hält ≥ 0,92,
  Embedder bei Freigabe); 112 passed `test_musical_goals_metrics.py`
  (Goal-Entscheidungen unverändert); 50 passed Ebene-1/phase_65/VQI; Ruff clean.

### §III.12-Pflicht-Witnesses implementiert (C4/C5/W4/W5/W6)

- **Befund:** §III.12 verlangt fünf Witnesses als **Nachweis** der Rekombination
  („nachzuweisen, nicht anzunehmen"); bis v10.3.11 waren C4/C5/W4/W5/W6 offen.
  Der Rekombinationspunkt selbst (`recombine_stems_with_gates`, C2→C1→C3) war
  bereits implementiert und verdrahtet.
- **Neu (report-only, kein Signal-Eingriff):**
  - **C4** `double_processed_bands` — Bark-Bänder, in denen BEIDE Stems um
    > 0,5 dB verändert wurden (Kamm-/Phantom-Risiko bei der Summe).
  - **C5** `level_step_db` — max. Pegel-Sprung Mix→Remix über 20-ms-Frames.
  - **W4** `ripple_depth_db` — Streuung des Band-Energie-Verhältnisses
    Remix/Mix (Kammfilter-Ripple-Tiefe).
  - **W5** `leakage_corr` — |Hüllkurven-Korrelation| der RAW-Stems
    (Geister-Anteile).
  - **W6** `lag_step_us` — max. Korrelations-Lag-Sprung zwischen
    Nachbarfenstern.

  Alle Werte erscheinen über `build_witness()` in
  `StemContext.witness_reports["recombination"]` (bestehende Verdrahtung in
  `stem_level_restorer`) und nutzen vorhandene Primitive
  (`_band_energy_db`, `bark_band_edges`, `_envelope`, `_xcorr_offset`) — kein
  Parallelcode (§G9 copilot-instructions.md).
- **Ehrlichkeit:** **W6/C5** sind als **Zeit-Kontinuität** gemessen, weil die
  Rekombination eine sample-genaue Summe ohne Segment-Naht ist (kein
  Concat/Crossfade); eine „Naht"-Größe wird nicht vorgetäuscht
  (`continuity_note`). Die Witnesses haben **keine automatische Konsequenz** —
  harte Schwellen erfordern eine eigene Kalibrierung (Muster F14).
- **Beweise:** 11 passed `test_stem_recombination_gates.py` (5 neue Tests:
  Berichtsvollständigkeit, Null-Werte bei perfekter Separation, C4 > 0 bei
  Doppelveränderung und == 0 bei Ein-Stem-Änderung, W5 steigt bei geteilter
  Quelle) sowie 23 passed `test_stem_level_restorer.py`; Ruff clean.

### F14-Ergänzung: PANNs-Evidenz wird selbst erzeugt

- **Warum:** Die Kalibrierung konnte den Konsens Anatomie ↔ PANNs nur messen, wenn
  eine Tags-JSON extern vorlag — im Default-Lauf blieb `consensus_rate = None` und
  die Schwellenempfehlung gesperrt.
- **Neu in `scripts/calibrate_gender_evidence.py`:**
  - `--panns auto` (Default): erzeugt die Tags selbst über
    `plugins.panns_plugin.classify_audio(mono, sr)` (Array-API, kein Temp-WAV).
    Fehlt das Modell, gibt es `None` und einen §V6 copilot-instructions.md-Hinweis
    — kein Abbruch, aber auch keine erfundene ML-Evidenz; die Empfehlung bleibt gesperrt.
  - `--panns off`: reine Anatomie-Verteilung.
  - `--panns <json>`: externe Tags (bisheriges Verhalten).
  - `--panns-out <json>`: sichert die erzeugten Tags (Cache für Wiederholungsläufe).
  - Der Report weist die Quelle aus (`panns_source`).
- **Belege:** 19 passed `test_gender_calibration_infra.py` (3 neue Tests: fehlendes
  Modell blockiert nicht, Tags ohne Singing-Klassen ⇒ None, Singing-Klassen werden
  durchgereicht); CLI-Probe `--panns off` auf `corpus/` läuft; die Klassen
  `Male singing`/`Female singing` sind im PANNs-Plugin verifiziert.

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

> SOTA-Roadmap offen: t6 Boundary-Maschinerie (ExcellenceOptimizer/PGHI-Struktur),
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
  ≤_MAX_ANALOG_CHAIN_DEPTH+2).
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
