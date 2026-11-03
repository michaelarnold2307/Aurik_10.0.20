# EAR-VAE v1 ↔ v2 — Vergleichsmessung (D.1c / D-K0-9)

**Datum:** 2026-11-03
**Status:** Messung **nicht ausführbar** — Harness + Hör-Stimuli-Paket vorbereitet,
technischer Grund dokumentiert (Auftrags-Alternative „ehrlich dokumentieren statt
konstruieren").
**Messdatei:** `docs/reports/current/2026-11-03_ear_vae_v1_v2_vergleich.json`
**Kontext:** LUECKEN-MATRIX D.1c „EAR-VAE v1↔v2-Vergleich" (`docs/TODOS_SOTA_ROADMAP.md`)
+ Defizit D-K0-9 „EAR-VAE v1↔v2-Urteil" (`.github/SOTA_DEFICIT_REGISTER.md`,
menschlich-gated; Tiefenanalyse E3 in `docs/reports/current/2026-10-28_vollstaendigkeit_tiefenanalyse.md`)
+ LÜCKEN-MATRIX A Nr. 5. Vorbereitung der **menschlichen** Entscheidung —
**KEINE Aktivierung, KEINE Flag-Flips** (§v10.802 (copilot-instructions.md)).

## 1. Methodik (im Harness implementiert)

Angelegt nach dem B3-Muster (`scripts/b3_stufenmessung.py`):

* **Gleiche Corpus-Fenster:** Referenz-Korpus wie
  `scripts/release_candidate_pipeline.py` (`CORPUS`, `_existing_corpus`,
  `_load_clip`): 8 Clips aus `test_audio/` (tape/vinyl/digital/vocals),
  Fenster-Starts 0,0/2,5/5,0/7,5 s, Länge 2,5 s, Fenster nur bei Rest ≥ 1 s.
* **Determinismus:** Seed 42 (§G5 (copilot-instructions.md)); Blind-Verlosung je
  Fenster über `random.Random(f"{seed}:ear_vae_v1_v2")`, gleicher Seed ⇒
  identische Verblindung.
* **Blind-Paarung:** je Fenster v1-Ausgang vs. v2-Ausgang, Seiten verlost,
  Bewertung sieht nur „links"/„rechts" (`scripts/ear_vae_v1_v2_vergleich.py`,
  `vergleich_messen`).
* **Zielmetriken:** `musik_mos` + `defektlast` — beide **richtungszertifiziert**
  (`docs/reports/current/qualitaetsmetrik_richtungszertifikat.json`, Spearman
  0,95). Keine unzertifizierten Metriken.
* **Auswertung:** Never-worsen mit JND-Toleranz 0,05 (b3-Maß), exakter
  zweiseitiger Vorzeichen-Test (`scripts/ab_harness.py`,
  `sign_test_two_sided`), Signifikanz p < 0,05.
* **Delta-SDR:** nicht korpusfähig — das Referenz-Korpus hat keine sauberen
  Referenzen; im Messfile als solches vermerkt (kein konstruierter Wert).
* **Hör-Stimuli:** Blind-Paare v1/v2 je Fenster via `scripts/ab_harness.py
  prepare` (byte-identische Kopien unter neutralen Namen, Schlüssel getrennt;
  `key.json` ist das Schlüsselexemplar — **nicht** an die Hör-Instanz).

## 2. Technischer Befund: v2 ist nicht lauffähig

**B1 — Keine Gewichte (Blocker).** `models/ear_vae2_upstream/` ist der
Upstream-Klon (εar-VAE2, „Fourier is Frontier", Wang/Dai/Xu) und enthält laut
Verzeichnis-Inventar (2026-11-03, Bestands-Root) nur Code/Docs/Assets:
`inference.py`, `ear_vae2/{model,opt}.py`, `configs/ear_vae2.json`, `README.md`,
`LICENSE`, Demo-Seite `docs/` — **kein** `ear_vae2.pt`, kein Checkpoint in
keinem Format. Die Upstream-README fordert ausdrücklich
`huggingface-cli download earlab/EAR_VAE2 --local-dir checkpoints/` — dieser
Download ist nicht erfolgt. Der maschinell gemessene Artefakt-Bestand bestätigt
das: **0 Modell-Artefakte, 0 B** (`.github/ML_ARTIFACT_FINGERPRINTS.md`,
Sektion `ear_vae2_upstream`). Ohne Gewichte ist keine Inferenz möglich.

**B2 — Format-/Architektur-Bruch.** v2 lädt einen **monolithischen
Torch-Checkpoint** (`torch.load`, Schlüssel `gen`/`gen_ema`/`state_dict`) in
`EarVAE2` (Config C0=64, D=128, use_vae=true, banded refiner 256/1024/12 Lagen;
`configs/ear_vae2.json`), 48 kHz Stereo, Inferenz über
`encode_audio`/`decode_audio` (`models/ear_vae2_upstream/inference.py`).
v1 (Produktion) ist ein **ONNX-Paar** (Encoder/Decoder, latent 64 × 1024,
44,1 kHz), geladen von `plugins/ear_vae_denoiser.py` über ONNXRuntime.
Es existiert **kein** v2-ONNX-Export und **kein** Adapter im Produktionspfad —
ein Ersatz bräuchte zuerst einen Export/Adapter (§III.9/§III.13
(copilot-instructions.md)).

**B3 — Nebenbefund Bestandsangabe.** LÜCKEN-MATRIX A Nr. 5 nennt „298 M";
gemessen sind 0 B Modell-Artefakte (s. B1) — die Angabe bezieht sich offenbar
auf den Upstream-Klon insgesamt (inkl. Git-History/Demo-Assets) und sollte im
Register präzisiert werden.

**B4 — Umgebungs-Nebenbefund.** Diese Dispatch-Umgebung stellt keine
Shell/Ausführung bereit; der Harness konnte hier nicht gestartet werden
(zusätzlich zu B1). Der hier dokumentierte Befund stammt aus direkter
Bestands-Inspektion (Verzeichnis-Inventare, README, `inference.py`,
`configs/ear_vae2.json`, `.github/ML_ARTIFACT_FINGERPRINTS.md`) — nicht aus
einem Messlauf.

## 3. Ergebnis

* **Messung:** nicht ausgeführt (B1). Die Messdatei trägt `status:
  "nicht_ausfuehrbar"` mit Seed 42 und Quellen; `zeilen` leer, `never_worsen`,
  `vorteile`, `p_wert` = `null` — **keine erfundenen Zahlen** (§V7
  (copilot-instructions.md); Muster: `docs/reports/current/2026-11-02_b3_modelle_metrik.json`).
* **Harness:** `scripts/ear_vae_v1_v2_vergleich.py` — vollständig implementiert
  (Corpus-Fenster, Blind-Paarung, musik_mos/defektlast, Vorzeichen-Test,
  Never-worsen, Hör-Stimuli-Preparation, fail-closed Preflight).
* **Hör-Stimuli:** Paket-Mechanik vorbereitet (Zielordner
  `output/ear_vae_v1_v2/hoer_stimuli/`, `ab_harness.prepare`-Anbindung,
  neutrale Labels „Kandidat A/B"); das **Rendern der WAVs** ist ausstehend,
  weil v2-Gewichte fehlen (B1).

## 4. Nächster Schritt (Voraussetzung für den Messlauf)

1. **Gewichte beschaffen:** `ear_vae2.pt` von HF `earlab/EAR_VAE2`; vor
   Verdrahtung Lizenz- und SHA-Feststellung am Artefakt (§III.13
   (copilot-instructions.md)) + Eintrag in
   `.github/ML_ARTIFACT_FINGERPRINTS.md`.
2. **Adapter:** v2-Export (ONNX analog v1) **oder** dokumentierter Torch-Pfad
   über denselben Fenster-Interface — dann läuft der Harness ohne Änderung.
3. **Messlauf:** `python3 -B scripts/ear_vae_v1_v2_vergleich.py --root
   <Bestands-Root>` (das Skript schreibt die Messdatei neu und rendert die
   Hör-Stimuli); danach Hör-Entscheidung.
4. **Urteil:** menschliche Hör-Instanz (Hörordnung,
   `.github/instructions/hoerordnung.instructions.md`, Konfliktregel §7 dort —
   im Auftrag „§4 Hörordnung" genannt). **Metriken sind Zeugen, kein Urteil.**

## 5. Verifikation

* Modul-Code geschrieben; **Tests** `tests/unit/test_ear_vae_v1_v2_vergleich.py`
  (8 Fälle: Fail-closed ohne Gewichte, Fensterplan, Blind-Determinismus §G5,
  Never-worsen/JND-Toleranz, Vorzeichen-Test, Messfile-Schema) —
  **nicht ausgeführt:** diese Umgebung hat keine Shell
  (`nice -n 19 python3 -m pytest tests/unit/test_ear_vae_v1_v2_vergleich.py -q`
  ist der Befehl; `ruff`-Läufe analog ausstehend).
* Bestands-Befund (B1–B3) ist durch direkte Inspektion belegt (s. Abschnitt 2).

## 6. Expliziter Hinweis

**Urteil = menschliche Hör-Instanz** (§4/§7 der Hörordnung) — diese
Vergleichsmessung ist die Entscheidungsvorlage. **Keine Flag-Flips, keine
Aktivierung, kein Ersatz von v1** (D-K0-9 menschlich-gated;
§V7/§v10.802 (copilot-instructions.md)).
