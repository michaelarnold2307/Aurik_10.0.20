# K0/K1-Restlücken 2026-11-05 — D-K0-2, D-K0-8, D-K1-5 (+ Status D-K1-3)

> **Auftrag:** Die offenen K0-/K1-Defizite D-K0-2, D-K0-8, D-K1-5 (+ Status D-K1-3) im
> SOTA-Defizit-Register (`.github/SOTA_DEFICIT_REGISTER.md`) strikt nach Register-Kriterien
> bearbeiten und schließen — erfüllt mit prüfbarem Beleg ODER präziser echter Blocker
> (§V7 (copilot-instructions.md): keine erfundenen Belege; §III.13 (copilot-instructions.md):
> Evidenz am realen Artefakt).
>
> **Arbeitsstand:** Branch `kun/k0-k1-rest-lu-cken-schliessen-hunjdy`, Basis `9a8ba267`
> (2026-11-05 16:38). Diese Worker-Session hatte **kein Shell** — alle Ausführungs-,
> Beschaffungs- und Verifikationsbefehle sind in §6/§7 für den Manager dokumentiert; die
> Tests sind statisch gegen die realen Signaturen geschrieben (Gate-Parser
> `scripts/sota_deficit_gate.py`, Beleg-JSON, Flag-Modul `backend/core/music_model_flags.py`).
>
> **Nicht angefasst:** `docs/SOTA_HANDOVER.md`; **keine Flag-Flips** (§v10.802
> (copilot-instructions.md)); die laufende GPU-D3-Kette (`scripts/train_*` bestehend,
> `models/`, `output/f3_bigvgan`, `train_job_d3_*`) blieb unberührt.

## 0 · Ergebnis-Überblick

| Eintrag | Kriterium (Register) | Ergebnis | Status jetzt |
|---|---|---|---|
| D-K0-2 | K0 „stilles Kapital": F3-Rollout entscheiden; Beleg nur mit Pfad | **geschlossen** — K0-C Hör-Entscheidung (NO-GO) als Beleg gebucht | geschlossen |
| D-K0-8 | K0: Konsument braucht Label-Korpus ≥ ~50/klasse + Wiring-Entscheidung + Sign-off | **präziser echter Daten-Blocker** mit Beschaffungs-/Trainingspfad (kein Schein-Abschluss) | in-arbeit |
| D-K1-5 | K1 + F11: A1-HF-Gain-Gate ≥ 1,02, sonst Flag AUS | **Gate-Beleg gebucht** (bestanden 2026-10-27); verbleibender Blocker **menschlich** (Hör-Sign-off) | in-arbeit |
| D-K1-3 | K1: gelernte Repräsentation im Entscheidungspfad; Aktivierung über §4-Verfahren | **Status dokumentiert**: verdrahtet (28f2947d), Aktivierung menschlich-gated, **KEIN Flag-Flip** | in-arbeit |

---

## 1 · D-K0-2 — Basis-BigVGAN statt Musik-F3 (K0)

**Kriterium.** Register-Status-Enum: „geschlossen = behoben und belegt; Beleg-Pfad muss
existieren". Die Zeile fragte: Warum ist die Domäne-`unbekannt`-Basis aktiv statt des
musiktrainierten F3-Gegenstücks? Blocker (Alt): „Rollout des MUSDB-Finetunes D-K0-1
offen (Kostenarchitektur)" + Beleg-Lücke des Basis-Korpus.

**Handlung.** Die Hör-Instanz hat die Frage am 2026-10-30 entschieden (Strang H,
`docs/reports/current/2026-10-30_k0_signoff_paket.md` §K0-C): **K0-C = NO-GO**. Die
Paar-Matrix der C4-Stichprobe (jeweils 3 × 30 s, überwachte Stereo-Ausgabe,
`docs/reports/current/2026-10-29_c4_hoerprobe_bigvgan_f3.md`):

- `cassette_1980s_wow`: Referenz (Basis) deutlich besser → F3 schlechter
- `jazz_1950s_scratched`: F3 besser
- `reel_1940s_dropout`: F3 angenehmer, aber wesentlich dumpfer (Höhenverlust)

Urteil (wörtlich): „F3 ist nicht timbre-konsistent … 1× klar schlechter vs. 1× besser.
Kein konsistenter Gewinn + Timbre-Nachteil + Kosten +0,9 % ⇒ kein Rollout-Grund.
**Basis-BigVGAN bleibt aktiv.**" Damit ist die K0-Runde komplett (K0-A SCNet NO-GO →
D-K0-5, K0-B SGMSE NO-GO → D-K0-3, K0-C F3 NO-GO → D-K0-1) — die Buchung von D-K0-2
schließt die letzte nicht gebuchte Zeile dieser Runde, konsistent zum Stil von
D-K0-1/D-K0-3/D-K0-5 („geschlossen", (Alt-)Historie im Blocker-Feld, Beleg-Pfade).

**Beleg/Blocker.** Beleg: `docs/reports/current/2026-10-30_k0_signoff_paket.md` (§K0-C),
`docs/reports/current/2026-10-29_c4_hoerprobe_bigvgan_f3.md` (C4-Hörpaket), Mess-Block §4a
`.github/ML_MODEL_DOMAIN_REGISTRY.md`. Rest ohne Handlungsbedarf (im Blocker-Feld
dokumentiert): die Beleg-Lücke der Basis (Korpus ohne Modell-Karte/Hparams/SHA) ist ohne
Upstream nicht beschaffbar — die Domäne bleibt ehrlich `unbekannt` (§III.13
(copilot-instructions.md)); das F3-Gegenstück bleibt exportiert + paritätsbewiesen als
stilles Kapital. SOTA-M1 (Hörproben-Runde KP-1) ist durch die Runde erledigt und im
Blocker als „(Alt-)Offen (durch K0-C 2026-10-30 entschieden)" markiert.

## 2 · D-K0-8 — flacher Material-Klassifikator ohne Konsument (K0)

**Kriterium.** K0 „belegtes Training ohne Produktionskonsument" — schließbar nur mit
Konsumenten-Nachweis ODER bewusster Entscheidung. Der Register-Blocker (Alt) verlangt
**Daten** (Label-Korpus ≥ ~50 je Klasse), danach die menschliche **Wiring-Entscheidung**
(CLAP-Zeuge vs. trainierter Kopf) + **Hörordnungs-Sign-off** (§v10.802
(copilot-instructions.md)). Kein Beleg für „behoben" möglich ⇒ präziser echter Blocker
statt Schein-Abschluss (§V7 (copilot-instructions.md): kein Rollout ohne A/B).

**Handlung.** Blocker auf den realen Datenstand präzisiert (Status `offen → in-arbeit`,
analog D-K1-2, dessen Datenwelle diesen Blocker speist):

- **Bestand (WP-7, 2026-10-09):** 143 Labels, `shellac` 58 (Ziel ≥ 50 erreicht),
  63 Echt-Träger, `vinyl` **strukturell 0** (LP-Ära unter Schutz), Tape-Klassen dünn —
  Rechteinhaber-Freigabe (Lizenz §9/7 der A2-Liste) oder Synthetik, Maintainer-Entscheid
  offen (`docs/reports/current/2026-10-09_wp7_coverage.json`, D-K1-2).
- **Beschaffungspfad** vorbereitet: `scripts/generate_corpus_from_public_domain.py`
  (Lizenz-Gate PD/CC0, maschinenlesbar `--check`/`wp7_real_gap`).
- **Trainingspfad** vorbereitet: `python scripts/train_medium_classifier.py`
  (Korpus-Manifest + Golden-Listening-Set; §G5 (copilot-instructions.md) Determinismus —
  gleiche Seeds pro Session).
- **Sachverhalt Nachfolger:** der gelernte Kopf `models/material_crnn.pt` (D-K1-3:
  95,3 % auf echtem Korpus) ist hinter `use_material_crnn` verdrahtet;
  `models/medium_shallow_v1.joblib` bleibt ohne Konsument (`backend/core/medium_classifier.py`
  nutzt CLAP im `use_ml`-Zweig). Die Wiring-Entscheidung (CLAP-Zeuge vs. trainierter
  Kopf) + Hör-Sign-off bleiben **menschlich** — hier bewusst nicht vorweggenommen.

**Beleg/Blocker.** Beleg: `models/medium_shallow_v1_report.json` (CV n=56, 64,3 % —
unverändert nur diese eine Zahl bewertbar; die nicht reproduzierbare „77,5 % (Depth)"
blieb bewusst verworfen), `scripts/train_medium_classifier.py`,
`scripts/generate_corpus_from_public_domain.py`,
`docs/reports/current/2026-10-09_wp7_coverage.json`. Blocker: **Daten** (vinyl/Tape +
Rechte) → dann Wiring-Entscheidung + Hör-Sign-off; Tags SOTA-M18 behalten.

## 3 · D-K1-5 — BW-Reconstructor v5 (K1, synthetische Trainingsfamilie)

**Kriterium.** K1 „synthetisch trainiert statt Musik" + F11
(`docs/TODOS_SOTA_ROADMAP.md`): „Re-Training mit maskierungs-informiertem A1-Loss bis
Gate ≥ 1,02 — sonst bleibt das DSP-SBR/FlashSR-Paar der HF-Weg (Flag bleibt aus)";
Vertrag W-8 (`.github/WOHLKLANG_CLAIMS.md`) gesperrt. Status „geschlossen" nur mit
prüfbarem Beleg.

**Handlung.** Der technische Blocker ist **behoben + belegt**: A1-HF-Gain-Gate
**BESTANDEN** 2026-10-27 (kalibrierte Metrik `scripts/validate_bw_v5_hf_gate.py`):
HF-Gain (out/in > 3 kHz, milde HF-Dämpfung 6 dB/oct ab 2 kHz) neu = 51,0 ≥ 1,02
(Retrain `models/bw_reconstructor/retrain_a1_20261025/bw_reconstructor_v5.onnx`;
toene 1,952 → 85,756, breitband 0,808 → 50,981); die Kalibrierung reproduziert den
historischen 0,73-Wert der Alt-Modelle (gemessen 0,73–0,81, exakt 0,731 bei Seed 4242;
`docs/reports/current/2026-10-27_kp7_hf_gate_v2.json`, `summary.gate_passed: true`,
`kalibrierung.reproduktion: true`).

Trotzdem **kein „geschlossen"** — der verbleibende Blocker ist **menschlich**, und die
K0-Runde (K0-A/B/C) hat gezeigt, dass die Hör-Instanz Metrik-Gewinne verwerfen kann
(SCNet/SGMSE/F3 gewannen messbar und wurden per Ohr abgelehnt):

- **Offen (menschlich):** Hör-Sign-off (§v10.802 (copilot-instructions.md)) vor
  `use_bw_v5=True` wegen +17 dB HF-Boost (Artefakt-Risiko). Zusatz-Befund für das
  Sign-off: das A1-Gate ist **einseitig** (nur Untergrenze 1,02, keine Oberkante) — der
  massive HF-Boost ist genau die Risikostelle und muss gehört werden (Hörordnung §4
  (hoerordnung.instructions.md): Metriken sind Zeugen, die Hör-Instanz entscheidet).
- **Flag-Flip** nur über §4-Verfahren (`scripts/flag_freigabe.py`: Verdikt →
  Register-Buchung → Flag-Flip → Commit); `use_bw_v5` bleibt `False` (**KEIN Flag-Flip**
  in dieser Welle); der HF-Weg bleibt DSP-SBR/FlashSR (F11).
- Artefakt-Zeile korrigiert („nicht bestandenem A1-HF-Gain-Gate" → bestanden
  2026-10-27 nach Re-Training); SOTA-M7-Tag ergänzt (Register-Pflege: jede Zeile trägt
  ihren SOTA-M-Blocker). Synthetische Domäne wie D-K1-1 — darf nicht als musik-trainiert
  gelten.

**Beleg/Blocker.** Beleg: `scripts/validate_bw_v5_hf_gate.py` (kalibriert 2026-10-27),
`docs/reports/current/2026-10-27_kp7_hf_gate_v2.json`, `.github/WOHLKLANG_CLAIMS.md`
(W-8). Blocker: Hör-Sign-off (menschlich, +17 dB HF-Boost-Artefakt-Risiko) + §4-Verfahren.

## 4 · D-K1-3 — Material-CRNN: Register-Status (Dokumentation, KEIN Flag-Flip)

**Kriterium.** D-K1-3 „gelernte Material-Repräsentation im Entscheidungspfad statt
9 Skalar-Features; Ziel ≥ 80 % echte Korpus" + Aktivierung ausschließlich über
§4-Verfahren. Auftrag: Status dokumentieren, **KEIN Flag-Flip**.

**Handlung.** Register-Status vollständig gebucht (Stand 2026-11-05):

- **Verdrahtung abgeschlossen** (Commit `28f2947d`, 2026-11-03):
  `forensics/medium_detector.py` (`MediumDetector.detect`, Roh-Aufruf) hinter
  `use_material_crnn`, **Default AUS**; §V6 (copilot-instructions.md)-Fallback mit
  `logger.warning` + Begründung, getestet in `tests/unit/test_dk13_material_crnn_wiring.py`
  (11 Vertragstests). Messung 2026-10-31 (held-out echte Korpus): **95,3 %** vs.
  59–63 % Gauss/Mixture ⇒ SOTA-Pfad (1) diskriminativere Features + (2) gelernter
  Klassifikator erfüllt, (3) Echt-Träger bleibt D-K1-2.
- **Artefakt** `models/material_crnn.pt` liegt vor (seit 2026-11-01, 1,5 MB) —
  Korrekturhinweis `docs/reports/current/2026-11-04_sitzung_transient_messlauf_b3_status.md`
  §6; die „fehlt"-Meldung im 2026-11-03-Report ist worktree-abhängig veraltet.
- **Aktivierung menschlich-gated:** Flag-Flip nur über §4-Verfahren
  (`scripts/flag_freigabe.py`: Verdikt → Register-Buchung → Flag-Flip → Commit) mit
  Hörordnungs-Sign-off (§v10.802 (copilot-instructions.md)). **KEIN Flag-Flip** —
  `use_material_crnn` bleibt `False`, Default = bit-identisch zum bisherigen Verhalten.
- Offen: §4-Aktivierung + Echt-Träger ≥ 50/Klasse (D-K1-2, SOTA-M18).

**Beleg/Blocker.** Beleg: `docs/reports/current/2026-11-03_d_k1_3_material_crnn_wiring.md`
(Commit `28f2947d`), `docs/reports/current/2026-11-04_sitzung_transient_messlauf_b3_status.md`
§6, `docs/reports/current/2026-10-09_wp7_medium_validierung.json`. Status bleibt
`in-arbeit` (Aktivierung + Daten offen).

## 5 · Begleit-Korrekturen (§G9 (AGENTS.md) „eine Quelle")

Die überholte Behauptung „A1-Gate nicht bestanden (0,73 < 1,02)" stand seit dem
Gate-Bestand 2026-10-27 an **drei** Stellen und widersprach der Register-Belegkette.
Korrigiert (jeweils **ohne** Wert- oder Statusänderung):

1. `.github/WOHLKLANG_CLAIMS.md` W-8: Sperrgrund aktualisiert („A1-HF-Gain-Gate
   2026-10-27 bestanden — Beleg `docs/reports/current/2026-10-27_kp7_hf_gate_v2.json`;
   Hör-Sign-off offen, +17 dB HF-Boost-Artefakt-Risiko, §v10.802 (copilot-instructions.md))").
   Status-Zeile bleibt `gesperrt`, Soll bleibt `AUS` — reine Begründungs-Korrektur.
2. `backend/core/music_model_flags.py`: Kommentar zu `use_bw_v5` gezogen (Wert
   unverändert `False`; E5-Literal des Wohlklang-Gates unangetastet).
3. `tests/normative/test_primary_paths_no_fallback.py`: Fail-Meldung
   `_CONSCIOUSLY_GATED["bw_v5"]` aktualisiert (Assertion selbst unverändert).

## 6 · Vorbereitete Befehle (für den Manager; Worker-Env ohne Shell)

Beschaffung D-K0-8/D-K1-2 (Welle D, SOTA-M18 — **läuft**, hier nicht zu starten):

```bash
python3 -B scripts/generate_corpus_from_public_domain.py --check   # Abdeckung je Klasse (wp7_real_gap)
```

Re-Training Material-Klassifikator (erst nach Daten-Schwelle ≥ 50 je Klasse;
`scripts/train_medium_classifier.py`, Usage laut Modul):

```bash
python3 -B scripts/train_medium_classifier.py
```

CRNN-Nachlauf/Fingerprint (falls gewünscht; **berührt die laufende D3-GPU-Kette nicht**,
sondern liegt daneben):

```bash
python3 -B scripts/train_material_crnn.py --epochs 30 --out models/material_crnn.pt
python3 -B scripts/model_artifact_probe.py    # Artefakt-Fingerprint, §III.13 (copilot-instructions.md)
```

Flag-Aktivierungen (ausschließlich §4-Verfahren, jeweils mit menschlichem Hör-Sign-off —
hier bewusst **nicht** ausgeführt):

```bash
python3 -B scripts/flag_freigabe.py --flag use_material_crnn --verdikt <fit.json|ab_harness-Statistik> --signoff <hoer-sign-off>
python3 -B scripts/flag_freigabe.py --flag use_bw_v5       --verdikt <fit.json|ab_harness-Statistik> --signoff <hoer-sign-off>
```

## 7 · Tests + Verifikationsbefehle

Neu: `tests/unit/test_dk0_k1_rest_luecken.py` — 10 Vertragstests, statisch gegen reale
Signaturen (Gate-Parser `parse_register`/`audit`, Beleg-JSON, Flag-Modul):

1. `test_register_geht_durch_das_fail_closed_gate` — fail-closed Gate auf der gebuchten Fassung.
2. `test_dk0_2_ist_ueber_k0_c_entschieden_und_mit_beleg_geschlossen` — Status „geschlossen" + Beleg-Pfad existiert.
3. `test_dk0_2_beleg_traegt_das_hoer_urteil` — der Beleg trägt „NO-GO" + „Basis-BigVGAN bleibt aktiv".
4. `test_dk0_8_bleibt_datenblocker_mit_beschaffungs_und_trainingspfad` — Status „in-arbeit", Daten-Blocker + existierende Pfade (kein Schein-Abschluss).
5. `test_dk1_5_a1_gatebeleg_im_json_ist_gepint` — `gate_passed: true`, `gate_threshold_linear == 1.02`, Kalibrierung `reproduktion: true`.
6. `test_dk1_5_aktivierung_bleibt_hoer_signoff_und_flag_gesperrt` — Blocker nennt Hör-Sign-off, `use_bw_v5`, `flag_freigabe.py`.
7. `test_dk1_3_status_dokumentiert_verdrahtung_und_kein_flag_flip` — Commit `28f2947d`, `use_material_crnn`, §4-Verfahren, „KEIN Flag-Flip".
8. `test_kein_flag_flip_use_material_crnn` — Flag-Default AUS (bit-identisch).
9. `test_kein_flag_flip_use_bw_v5` — Flag-Default AUS.
10. `test_wohlklang_w8_sperrt_mit_aktueller_begruendung` — W-8 `gesperrt` + aktueller Sperrgrund (keine veraltete Gate-Behauptung).

Verifikation (Manager; Worker-Env hatte kein Shell, daher hier nicht ausgeführt):

```bash
nice -n 15 python3 -B -m pytest tests/unit/test_dk0_k1_rest_luecken.py -q
nice -n 15 python3 -B -m pytest tests/unit/test_dk13_material_crnn_wiring.py -q          # Nachbarsuite (11)
nice -n 15 python3 -B -m pytest tests/normative/test_sota_deficit_gate.py \
  tests/normative/test_primary_paths_no_fallback.py tests/unit/test_wohlklang_gate.py -q  # betroffene Gates
python3 -B scripts/sota_deficit_gate.py
python3 -B scripts/wohlklang_gate.py
ruff --select F821,F601,B009,I001 tests/unit/test_dk0_k1_rest_luecken.py \
  backend/core/music_model_flags.py tests/normative/test_primary_paths_no_fallback.py
python3 -B scripts/repo_search.py --before-create tests/unit/test_dk0_k1_rest_luecken.py  # Write-Gate, s. §8
python3 -B scripts/change_ledger.py snapshot                                              # TASK_CHANGES.md
```

## 8 · Beobachtungen außerhalb des Auftrags (unverändert)

- **Write-Gate:** `python3 -B scripts/repo_search.py --before-create` konnte workerseitig
  nicht laufen (kein Shell). Die inhaltliche Prüfung (keine Namens-/Zweck-Doppelung zu
  `tests/unit/test_dk13_material_crnn_wiring.py` bzw.
  `tests/normative/test_sota_deficit_gate.py`) erfolgte per Repo-Grep; der Gate-Befehl ist
  in §7 zur Nachholung notiert, der FILE_REGISTRY-Eintrag ist gesetzt.
- **§V6/§G5/§V8 (copilot-instructions.md):** keine neuen Laufzeit-Pfade in dieser Welle ⇒
  nicht einschlägig; die D-K1-3-Verdrahtung selbst ist §V6-konform (Fallback mit
  `logger.warning`, getestet).
- **Roadmap-Drift:** `docs/TODOS_SOTA_ROADMAP.md` SOTA-M1 steht noch „offen —
  Hörinstanz", ist aber durch die K0-Runde 2026-10-30 entschieden; SOTA-M7
  („in-arbeit — Gate BESTANDEN … offen: Hör-Sign-off") passt zur hier gebuchten
  D-K1-5-Lage. Nachzug dort separat empfohlen (nicht Teil dieser Welle).
- **A1-Gate-Form:** einseitig (nur Untergrenze 1,02) — für die W-8-Hörprobe ist der
  +17 dB HF-Boost die eigentliche Prüfstelle; als Hinweis in D-K1-5 gebucht.
- **Worktree-`models/`:** `models/medium_shallow_v1.*` und `models/material_crnn.pt`
  sind in diesem Worktree nicht abgelegt (gitignore/`models/`-Bestand ist
  worktree-abhängig) — die Gate-Pflicht „mindestens ein auflösbarer Pfad je Eintrag"
  erfüllen die zitierten Skript-/Report-Pfade; gegen Modell-Dateien wurde nichts gebucht.

## 9 · Conventional-Commit-Vorschlag

```
docs(sota): K0/K1-Restlücken buchen — D-K0-2 geschlossen (K0-C NO-GO), D-K0-8/D-K1-5 Blocker präzisiert, D-K1-3-Status dokumentiert

- D-K0-2: K0-C Hör-Entscheidung 2026-10-30 (L01) NO-GO gebucht → geschlossen mit
  Beleg docs/reports/current/2026-10-30_k0_signoff_paket.md (Basis-BigVGAN bleibt
  aktiv; K0-Runde komplett). Beleg-Lücke Basis-Korpus bleibt ehrlich unbekannt (§III.13).
- D-K0-8: echter Daten-Blocker präzisiert (WP-7: 143 Labels, shellac 58, vinyl
  strukturell 0) mit Beschaffungs-/Trainingspfad; Nachfolger-Kandidat Material-CRNN
  vermerkt; Wiring-Entscheidung + Hör-Sign-off bleiben menschlich. Status in-arbeit.
- D-K1-5: A1-HF-Gain-Gate BESTANDEN 2026-10-27 gebucht (51,0 ≥ 1,02, Kalibrierung
  reproduziert 0,73); verbleibender Blocker menschlich (Hör-Sign-off §v10.802, +17 dB
  HF-Boost, Gate einseitig); SOTA-M7 getaggt. Status in-arbeit, KEIN Flag-Flip.
- D-K1-3: Register-Status dokumentiert (verdrahtet 28f2947d hinter use_material_crnn,
  Default AUS, Artefakt seit 2026-11-01 vorhanden; Aktivierung nur §4-Verfahren +
  Hör-Sign-off). KEIN Flag-Flip.
- Begleitend (§G9 eine Quelle): W-8-Sperrgrund, use_bw_v5-Kommentar und
  _CONSCIOUSLY_GATED-Text auf den Gate-Stand 2026-10-27 gezogen (Werte/Status unverändert).
- Tests: tests/unit/test_dk0_k1_rest_luecken.py (10 Vertragstests) + FILE_REGISTRY-Eintrag.
```

---

## 9. Manager-Nachtrag (2026-11-05): Ausführung, Beleg-Checks, Integration

1. **Beleg-Stichproben (bestätigt):** D-K1-5-Behauptungswandel „Gate nicht
   bestanden → bestanden" ist belegt: `docs/reports/current/2026-10-27_kp7_hf_gate_v2.json`
   enthält `gate_passed = True` (kalibrierte Metrik); `use_bw_v5` bleibt
   ehrlich **AUS** (Hör-Sign-off offen, +17 dB HF-Boost-Artefakt-Risiko,
   §v10.802 (copilot-instructions.md)) — kein Flag-Flip. D-K0-2 = NO-GO ist
   die dokumentierte Hör-Entscheidung 2026-10-30 (Hör-Instanz L01).
2. **Verifikation (ausgeführt):** `test_dk0_k1_rest_luecken.py` +
   `test_dk13_material_crnn_wiring.py` + Normativ-Suiten
   (`test_sota_deficit_gate`, `test_primary_paths_no_fallback`,
   `test_wohlklang_gate`) **51/51 + 19/19 grün**; Gates grün
   (`sota_deficit_gate`, `wohlklang_gate`, `id_registry_check --strict`);
   ruff clean. Erste Fehlschläge waren Umgebungs-Artefakte des Worktrees
   (fehlende Modell-/Mess-Pfade — Symlinks nachgezogen, s. Handover §3).
3. **Zitat-Korrektur:** §G9-Bezug in der neuen Testdatei auf die kanonische
   Quelle (copilot-instructions.md) gebracht (ID-Registry R2).
4. **Empfohlener Nachzug ausgeführt:** `docs/TODOS_SOTA_ROADMAP.md` SOTA-M1
   Status „offen — Hörinstanz" → entschieden (K0-Runde 2026-10-30).
