# K2-Metrik-Lücken D-K2-2 / D-K2-3 / D-K2-5 — Messvertrags-Schluss (2026-11-05)

**Auftrag (Manager):** Die K2-Metrik-Lücken D-K2-2, D-K2-3, D-K2-5 im
`.github/SOTA_DEFICIT_REGISTER.md` strikt nach Register-Kriterien schließen —
Messverträge als Grundlage jeder Hör-Entscheidung. Umsetzung nach Muster der
erfolgreichen Messvertrags-Fixes 10.12.47 (`transient_energie`) und 10.12.48
(`transparenz`): **Metrik korrigieren statt Pipeline verbiegen** (§V7
(copilot-instructions.md)), Offenlegung im Report, A/B-Vorbereitung
(Vertrag alt/neu).

**Arbeitsrahmen:** Worker-Runtime ohne Shell — alle Messbefehle sind
vorbereitet und unten dokumentiert; die **Ausführung macht der Manager**.
Keine Flag-Flips (KP-5 `use_bigvgan_f3_studio` bleibt W-1-gesperrt),
kein `docs/SOTA_HANDOVER.md`, keine `models/`-/`scripts/train_*`-Anrührung
(D3-Training läuft auf der GPU). ID-Zitate mit Quelle (AGENTS.md §5).

---

## Übergeordneter Messvertrag-Befund: PQS Timing-Konflation (D-K2-2/3)

Der §1.4-Studio-Finisher (Kaskade Vocos→HiFi-GAN→PGHI, Sprach-Vocoder auf
Musik) wird über **PQS-MOS(Mix nach Schritt 11) < 4,3** entschieden. Prüfung
dieses Messvertrags gegen die echten Signaturen:

* **Vertragsbruch (Timing-Konflation):** `PerceptualQualityScorer.__init__`
  dokumentiert `align_signals=True` = „werden Signale vor Bewertung zeitlich
  ausgerichtet" — implementiert war nur Trunkierung auf die gemeinsame
  Länge. Ein reiner Latenz-Offset (Resample-/STFT-Ketten) wurde als
  Qualitätsverlust gemessen und konnte die Sprach-Vocoder-Kaskade unnötig
  auslösen. Belegt im Test `test_pqs_vertrag_alt_punished_reinen_timing_offset`.
* **Korrektur (Vertrag erfüllt, §V7 (copilot-instructions.md)):**
  `backend/core/perceptual_quality_scorer.py` — `score_audio` richtet in
  einem begrenzten Fenster (±`ALIGN_MAX_LAG_S` = 20 ms) aus und wählt
  `max(MOS(lag0), MOS(+k), MOS(-k))` (k = stärkster Kreuzkorrelations-Lag).
  **Never-worsen ist damit konstruktiv je Fenster garantiert** (Lag 0 ist
  immer Kandidat; Gleichstand gewinnt Lag 0). `align_signals=False`
  reproduziert exakt den Vertrag ALT (nur Trunkierung) — Alt-Pfad-Anker im
  Test `test_pqs_vertrag_alt_punished_reinen_timing_offset`.
* **Keine Formel-Korrektur:** Der LUFS-Term (konstant 1,0) ist im Code als
  bewusste Vereinfachung dokumentiert („keine Referenz-LUFS nötig"), die
  DLM-Lautheits-Ratio läuft als Zeuge (kein Hard-Fail) mit — hier lag kein
  belegter Vertragsbruch vor, also keine Änderung.
* **A/B vorbereitet:** `scripts/b3_pqs_mos_neverworsen.py` (Muster
  `scripts/b3_transient_energie_neverworsen.py` — **unverändert**, nur
  Vorbild). Vertrag alt (`align_signals=False`) vs. neu (`True`) auf
  denselben Corpus-Fenstern + deterministischen Proben (Seed 42),
  Sign-Test, JND 0,05, §1.4-Trigger-Flips (PQS-Norm-Schwelle 0,825).

---

## D-K2-2 — Vocos 24 kHz (sprache, LibriTTS belegt) → GESCHLOSSEN

### Kriterium (Register-Zeile)
„Vocoder-Stufe 24 kHz — §V6 (copilot-instructions.md)-Gate prüfen (WP-1)" —
K2-Regel: Sprach-Vocoder darf Musik nur als §V6-Fallback (Warnung + Grund)
verändern; WP-1-Deliverable: Vollständigkeitstest „keine unflagged
Sprach-Vocoder-Synthese auf Musik".

### Handlung
1. **Restoration-Hartsperre verifiziert:** `vocode(mode="restoration")` wirft
   `ValueError` („nur Studio-2026-Modus erlaubt") — im Restoration-Pfad läuft
   der Sprach-Vocoder gar nicht (WP-1-Kern).
2. **Ladepriorität verifiziert:** 48k → 44.1k → 24k in
   `plugins/vocos_plugin.py::__init__`; 24 kHz ist ausschließlich
   Fallback-Stufe 3 (Register-Verifikation 2026-10-07 bestätigt).
3. **Gate-Lücke gefunden und behoben:** Die §V6-Domänen-Warnung war über
   `if "44" not in bname` auf das 44.1-kHz-Tier beschränkt — ein
   nachgeladenes 44.1-kHz-Modell (nicht im Bestand, Domäne unbelegt) wäre
   als Stufe 1 OHNE Domänen-Warnung gelaufen (genau das „unflagged" der
   WP-1-Formulierung). Jetzt warnt jedes geladene Tier (24 kHz = LibriTTS,
   48 kHz = Sprach-Korpora ohne Gesang, 44.1 kHz = domänen-unbelegt) mit
   Begründung. Änderung ist logging-only (kein Signalverhalten, keine
   Metrik) ⇒ A/B-Pflicht nicht einschlägig; Never-worsen trivial.
4. **Trigger-Messvertrag geprüft:** siehe PQS-Abschnitt (Timing-Konflation
   behoben, A/B vorbereitet).

### Beleg (prüfbar)
`plugins/vocos_plugin.py` (§V6-Block + Ladepfad),
`backend/core/perceptual_quality_scorer.py` (Vertrag §2.6),
`tests/unit/test_k2_messvertrag_luecken.py` (3 Gate-Tests + 2 Vertragstests),
`scripts/b3_pqs_mos_neverworsen.py` (A/B-Harness), dieser Report.
Status `offen` → **`geschlossen`** (Register-Zeile D-K2-2).

---

## D-K2-3 — Vocos 48 kHz (sprache) → OFFEN, präziser echter Blocker

### Kriterium (Register-Zeile)
Evidenzpflicht §III.13 (copilot-instructions.md): Domäne nur mit
Trainings-Skript-Korpus, Modell-Karte, SHA-Identität oder
widerspruchsfreiem Plugin-Header — plus die offene **Primär-Rollen-Entscheidung**
(Musik-Ersatz BigVGAN-F3 vs. Musik-Finetune).

### Handlung
1. **Upstream-Beleg re-verifiziert (2026-11-05, echter Beleg):**
   * HF-API `https://huggingface.co/api/models/kittn/vocos-mel-48khz-alpha1`
     — Repo existiert (`usedStorage` 147 342 055 B ≡ `size_bytes`
     `pytorch_model.bin` in `models/manifest.json`; SHA `3315c87d…`);
   * HF-Discussion #2 (`…/discussions/2`, Titel „About Datasets"):
     Owner-Antwort (kittn, 2026-04-17) wörtlich: „I think it was mostly
     podcasts I had scraped, plus VCTK and Expresso" — reine
     Sprach-Korpora, **kein Gesang**; zusätzlich
     „reconstruction quality is quite poor". Das Zitat-Datum 2026-04-17
     im Plugin-Header ist das Antwortdatum des Owners (Fragestellung
     2026-04-07) — kein Zitatfehler.
   * Upstream `config.yaml` (48 kHz/128 Mel/hop 256): kein Korpus-Feld;
     `evaluate_pesq`/`evaluate_utmos` sind Sprach-Indikatoren (nur Indiz,
     kein Beleg).
   * Fazit: Domäne `sprache` bleibt belegt (Korpora ohne Gesang ⇒
     §III.11 (copilot-instructions.md) „Sprache muss Gesang sein" verneint).
     Die Aufgaben-Annahme „Domäne 'sprache' ohne existierenden Beleg" ist
     damit **widerlegt** — der Beleg existiert und ist heute erneut
     maschinell geprüft.
2. **Primär-Rolle NICHT autonom entscheidbar — echter Blocker:**
   * Musik-Ersatz BigVGAN-F3 (D-K0-1) erfordert die **C4-Hörstichprobe**
     (menschliche Urteile);
   * Musik-Finetune (MUSDB18-HQ, Hausroutine) = **D3-Training auf der GPU**
     (gesperrt, separater Auftrag);
   * KP-5 (`use_bigvgan_f3_studio`, SOTA-M6) ist vorbereitet, aber
     **W-1-gesperrt** — KEIN Flag-Flip in dieser Welle.
3. **Trigger-Messvertrag PQS-MOS abgesichert** (Timing-Konflation behoben,
   A/B vorbereitet) — der Studio-Pfad-Trigger, der das Domänen-Risiko real
   aktiviert, ruht jetzt auf einem erfüllten Messvertrag.

### Blocker (präzise, ehrlich)
Primär-Rolle entscheidbar erst nach C4-Hörstichprobe (menschlich) oder
Musik-Finetune (D3-Training, GPU); KP-5 bleibt W-1-gesperrt (SOTA-M6);
WP-3-Forensik 2026-10-25 (36 Hörordnungs-Verstöße, Gegenmaßnahme KP-5/W-13
vorbereitet) bleibt bestehen. Status `offen` **bleibt offen** — kein
Scheinschluss.

---

## D-K2-5 — AERO BWE-Challenger (sprache) → OFFEN, präziser echter Blocker

### Kriterium (Register-Zeile)
Designtreue: kein AERO im Produktionspfad, bevor eine Challenger-Runde
**korrekt** bestanden ist; Never-worsen auf beiden Seiten; Rest offen:
MUSHRA-Challenger-Runde (menschliche Urteile) **oder** Task-Korrektur am
Export (SOTA-M2 = „Sign-off-Entscheidungen §9" — Status „offen — Maintainer").

### Handlung (Messvertrag-Audit)
1. **Export-Zahlen konsistent (keine erfundenen Zahlen):**
   `docs/reports/current/2026-10-06_bwe_candidates.json` weist für
   shared-12k ΔSDR **min −2,446 / mean −1,638 dB** aus
   (`shared12_aero_minus_flash_min_db` / `…_mean_db`) — deckungsgleich mit
   den Register-Zitaten „min −2,4 / mean −1,6"; `aero_never_worsen` und
   `flash_never_worsen` beide `false` (wie zitiert).
2. **Vertrag geprüft — kein belegter Vertragsbruch:** Referenz =
   Volldband-Grundwahrheit (korrekt für BWE), 1-s-Segmente,
   min-ΔSDR-Never-worsen-Gate, beide Kandidaten identisch behandelt
   (gleiche Referenz, gleiche Trunkierung). Die in den Muster-Fixes
   typischen Bruchbilder (falsche Referenz / Impuls-Sets) liegen hier
   **nicht** vor; Timing: beide Kandidaten ohne Lag-Ausrichtung — relative
   Fairness nur bei vergleichbarer Latenz, als Hinweis für das
   MUSHRA-Protokoll notiert. **Daher keine Metrik-Korrektur** — die
   Akzeptanzregel ist bewusst streng und darf nicht aufgeweicht werden,
   nur damit ein Kandidat durchkommt (§V7 (copilot-instructions.md)).
3. **Struktureller Befund (offengelegt):** Die Regel „min ΔSDR ≥ 0 je
   Segment" gegen die Volldband-Referenz ist für generative BWE praktisch
   nie bestehbar — die Bandlimit-Baseline (20,1–21,6 dB) liegt über beiden
   Kandidaten (15,2–18,6 dB), weil jedes plausible, aber abweichende
   Hochband die Wellenform-SDR senkt. Objektiv-SDR kann die Rolle daher
   nicht validieren — genau deshalb ist die **menschliche Urteilsinstanz**
   (MUSHRA, Hörordnung) der vorgesehene Entscheidungsweg.
4. **Verdrahtungs-Invariante getestet:**
   `tests/unit/test_k2_messvertrag_luecken.py::test_aero_bleibt_unverdrahtet_bis_challenger_runde_bestanden`
   — kein AERO-Import/-Bezug in `backend/`, `plugins/` (außer
   `plugins/aero_plugin.py`), `cli/`, `Aurik10/`, `denker/`, `processing/`;
   Challenger-Infrastruktur (`scripts/benchmark_bwe_candidates.py`,
   `scripts/challenger_round.py`) bleibt die einzige Konsumentin.

### Blocker (präzise, ehrlich)
Adoption/Verwerfung erfordert entweder die **MUSHRA-Challenger-Runde**
(menschliche Urteile, n ≥ 30; WP-8 extern blockiert) oder das
**Maintainer-Sign-off SOTA-M2** („offen — Maintainer") — beides nicht durch
Worker-Arbeit ersetzbar. Kein ADOPT ohne bestandene Runde; keine
Verdrahtung (Designtreue). Status `offen` **bleibt offen**.

---

## Verifikationsbefehle für den Manager (Worker-Runtime ohne Shell)

```bash
# Fokussierte Unit-Tests (neu):
nice -n 15 python3 -B -m pytest tests/unit/test_k2_messvertrag_luecken.py -q
# Regression der bestehenden PQS-Konsumenten (align_signals=False-Anker):
nice -n 15 python3 -B -m pytest tests/unit/test_v97_cognitive_layer.py -q
# Statisches Gate:
ruff --select F821,F601,B009,I001 backend/core/perceptual_quality_scorer.py \
  plugins/vocos_plugin.py scripts/b3_pqs_mos_neverworsen.py \
  tests/unit/test_k2_messvertrag_luecken.py
# Register-Gate (fail-closed):
python3 -B scripts/sota_deficit_gate.py
# A/B Never-worsen (Vertrag alt/neu — B3-Muster, JND 0,05):
nice -n 19 python3 -B scripts/b3_pqs_mos_neverworsen.py run --label alt
nice -n 19 python3 -B scripts/b3_pqs_mos_neverworsen.py run --label neu
nice -n 19 python3 -B scripts/b3_pqs_mos_neverworsen.py compare \
  --alt docs/reports/current/b3_pqs_mos_alt.json \
  --neu docs/reports/current/b3_pqs_mos_neu.json
# Write-Gate-Rest + Task-Ledger:
python3 -B scripts/repo_search.py --before-create scripts/b3_pqs_mos_neverworsen.py
python3 -B scripts/change_ledger.py snapshot
```

Erwartet: pytest grün (7 Tests, davon 3 parametrisiert = 9 Fälle);
ruff clean; `sota_deficit_gate.py` ohne Befund (D-K2-2 `geschlossen` mit
existierenden Beleg-Pfaden); B3-Lauf ERFUELLT (Never-worsen konstruktiv).

## Geänderte/neue Dateien

* `backend/core/perceptual_quality_scorer.py` — Messvertrag §2.6 erfüllt
  (Timing-Ausrichtung, Never-worsen max-Auswahl; Altpfad via
  `align_signals=False` unverändert).
* `plugins/vocos_plugin.py` — §V6-Domänen-Warnung für jedes Tier
  (44.1-kHz-Lücke geschlossen) + Verifikationsvermerk im Header.
* `scripts/b3_pqs_mos_neverworsen.py` — **neu**, A/B-Harness (Muster
  `scripts/b3_transient_energie_neverworsen.py` unverändert).
* `tests/unit/test_k2_messvertrag_luecken.py` — **neu**, 7 fokussierte Tests.
* `.github/SOTA_DEFICIT_REGISTER.md` — D-K2-2 geschlossen, D-K2-3/D-K2-5
  Blocker präzisiert.
* `.github/FILE_REGISTRY.md` — 3 Write-Gate-Einträge.
* `docs/reports/current/2026-11-05_k2_metrik_luecken.md` — dieser Report.

## Quellen (prüfbar, abgerufen 2026-11-05)

* HF-API Modell-Meta: `https://huggingface.co/api/models/kittn/vocos-mel-48khz-alpha1`
  (`usedStorage` 147 342 055; SHA `abd1670e…`; Lizenz `mit`).
* HF-Discussion #2: `https://huggingface.co/kittn/vocos-mel-48khz-alpha1/discussions/2`
  (Frage 2026-04-07 „About Datasets"; Owner-Antwort kittn 2026-04-17).
* Upstream-Karte: `https://huggingface.co/kittn/vocos-mel-48khz-alpha1/raw/main/README.md`
  („Training details: TODO").
* Upstream-Config: `https://huggingface.co/kittn/vocos-mel-48khz-alpha1/raw/main/config.yaml`
  (48 kHz/128 Mel/hop 256, `evaluate_pesq`/`evaluate_utmos`).
* Artefakt-Identität: `models/manifest.json` (`pytorch_model.bin`, SHA-256
  `3315c87d130922df…`, `size_bytes` 147 342 055 ≡ HF `usedStorage`).
* BWE-Export: `docs/reports/current/2026-10-06_bwe_candidates.json`.

## Conventional-Commit-Vorschlag

```
fix(measurement): K2-Messvertraege — PQS Timing-Ausrichtung + §V6-Vocos-Tier-Gate (D-K2-2/3/5)

- perceptual_quality_scorer: align_signals-Vertrag erfuellt (±20 ms
  Lag-Ausrichtung, Never-worsen max(lag0,±lag)); Altpfad unveraendert
- vocos_plugin: §V6-Domaenen-Warnung fuer jedes Tier (44.1-kHz-Luecke
  geschlossen) — D-K2-2 geschlossen
- b3_pqs_mos_neverworsen: A/B-Harness Vertrag alt/neu (Muster
  b3_transient_energie_neverworsen, unveraendert)
- tests/unit/test_k2_messvertrag_luecken: Gate- + Vertragstests
  (D-K2-2/3/5), AERO-Verdrahtungs-Invariante
- Register D-K2-2 geschlossen; D-K2-3/D-K2-5 praezise Blocker
  (C4-Hoerstichprobe/D3-Training bzw. MUSHRA/SOTA-M2), Upstream-Beleg
  D-K2-3 re-verifiziert
```
