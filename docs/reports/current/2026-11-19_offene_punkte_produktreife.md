# Offene Punkte zur vollen Produktreife auf uneingeschränktem iZotope-RX12-Niveau — Stand 2026-11-19 (abends)

> Fortschreibung der gleichnamigen Tagesübersicht. Quellen: `.github/WETTBEWERBS_PARITAET.md`
> (maschinell gepinnt, inkl. der neuen Harten Regeln 5–7), Spec 26 (DRT-1…15 inkl. neuem
> DRT-12), `docs/SOTA_HANDOVER.md` Punkt 37, Mess-Belege B1 (30 Paare) und die laufenden
> Ketten (signal10-Resume, Crackle, VinylNet, Kanalsatz-A/B).

## Definition (User-Vorgabe 2026-11-19, unverändert gültig)

**Produktreife erreicht ⇔ iZotope-RX12-Niveau ist in den relevanten Bereichen
mindestens erreicht.** Operational: **alle 15 Disziplin-Zeilen auf
`GESICHERT`/`PARITAET`** — 0 × OFFEN, 0 × LUECKE. **Heute: 2/15.**

> **Normative Korrektur (2026-11-20):** Die obige historische Zählung misst den
> damaligen internen Status und ist kein RX-12-Paritätsnachweis. Nach
> `.github/specs/26_defekt_reparatur_trainingsvertrag.md` DRT-16 muss jede
> akustische Disziplin den externen, gepaarten RX-12-Vergleich samt
> Nichtunterlegenheit und §4-Hör-Sign-off bestehen. Das Register weist derzeit
> keine `PARITAET`-Zeile aus; Bedienung/Echtzeit wird separat bewertet.

## Quantifizierte RX-Paritäts-Norm (Harte Regeln 5–7, neu)

Eine Zeile erreicht GESICHERT nur über **eines** dieser messbaren Kriterien
(plus §4-Sign-off; Hörordnung unangetastet):

- **(a) Ganz-Titel-Sinn-Beweis n≥10:** besser-Rate ≥ 5/10 UND ΔMOS-Mittel ≥ +0,05
  UND Δdefektlast-Mittel ≤ 0 UND 0 Überrestaurationen; oder
- **(b) defekt-spezifischer Kanal:** Regionen-Gate bestanden (q_median ≥ 0,95,
  q-Anteil≥0,5 ≥ 0,96, never-worsen ≥ 0,96) UND DRT-8-Ganz-Titel-Beweis
  (musik_mos nie < −JND, defektlast nie > +JND auf allen n≥10-Titeln).
- **DRT-12-Kanalsatz-A/B-Pflicht** vor jedem Kanal-Sign-off: AN vs. AUS auf
  identischen Paaren (`scripts/messe_kanalsatz_ab.py`) — der Gegenbeweis zum
  „0/12“-Muster; aufgenommene Kanäle gehen als **ein kombinierter §4-Batch**.

## Fortschreibung 2026-11-21 — DRT-12-Vollmessung

Der vollständige Beleg `2026-11-19_kanalsatz_ab_voll.json` wurde am 21.11.
mit Seed 42 abgeschlossen: **45/45 Paare erfolgreich gemessen**. Der gesamte
12-Flag-Satz ist **nicht freigabefähig**: gegenüber der Bestands-Pipeline
liegen 17/45 MOS-Deltas unter −JND und 4/45 Defektlast-Deltas über +JND.
Der Mittelwert (+0,1114 MOS) verschleiert die Regressionen (Median +0,0035).
Aus den Kandidaten-Rohdeltas nach der kanonischen Witness-Formel ergeben sich
außerdem **2/45 Überrestaurationen**; die Bestandszeilen haben 0/45.

Rezeptweise ist `raum` auf den drei Paaren konsistent besser
(ΔMOS-vs-Bestand +1,6383; Δdefektlast −0,6148) und `clipping` hat drei
MOS-Gewinne über JND. `mp3_128` zeigt einen MOS-Gewinn über JND, aber der
Mittelwert bleibt darunter (+0,0373). Das sind **Shortlist-Hinweise**, keine
Flag-Freigabe: der Lauf toggelte alle 12 Kanäle gemeinsam, isoliert also
keinen einzelnen Kanal. Alle Produktions-Flags bleiben AUS.

Folgeschritt ist eine isolierte Kanal-/Rezept-Ablation auf denselben Paaren
mit explizitem `ueberrestauration`-Witness. Diese Messung wird erst nach Ende
der bereits laufenden CPU-Ketten gestartet; ein weiterer globaler 12-Flag-
Lauf wäre keine informative Wiederholung.

Der Runner unterstützt dafür jetzt `--kanal <flag>`: genau ein Kanal wird
aktiviert und der Flag-Satz je Zeile sowie im Gesamtbeleg ausgewiesen. Resume
weist abweichende Flags, Datenordner, Baseline, Seed oder Paarzahl zurück. Der
bestehende Default bleibt der unveränderte 12-Flag-Lauf. Die Ablation wurde
noch nicht gestartet, um die laufenden CPU-Ketten nicht zusätzlich zu belasten.

## A. Die 13 offenen Disziplinen (Ziel: alle auf GESICHERT)

| Disziplin | Status | Offener Punkt → konkreter Weg |
|---|---|---|
| Declick | LUECKE | clicks-Kanal Gate-bestanden, aber Ganz-Titel 0/3 neutral → **Kanalsatz-A/B auf B2-clicks-Paaren** (läuft) → bei A/B-Gewinn kombinierter §4-Batch |
| Decrackle | LUECKE | **VinylNet-Retrain mit CC0-Leerrillen läuft** (Ep. 15+, val SI-SDR +3,0 dB); **Crackle-Expansion läuft** (Titel 3/6) → danach crackle-Detektor+Kopf+Gate → Never-worsen + RX-Vergleich |
| Denoise | LUECKE | M9 endgültig gescheitert; Witness-Kandidat: A/B auf vinyl/kassette-Paaren zeigt, ob die Tape-Kanäle tragen; sonst neuer Denoise-Kanal (Regionen-Daten nötig) |
| HF-/Bandbreiten-Rekonstruktion | LUECKE | FlashSR −4,61 dB (Gate verfehlt), BW-V5-Zeiger-Fehler (D-K1-5) → **HF-Residual-Training oder Kandidat-Austausch** (GPU nach VinylNet) |
| Stereo-Imaging/Azimuth | LUECKE | 2/3 balance_besser; **Snowmine-Befund offen** (phase_15 nie geplant — Plan/Defekt-Ranking) → Ursache fixen → Stereo-n≥10 → Hör-Vergleich |
| Bedienung/Echtzeit | LUECKE | Segment-Vorschau implementiert (🔍-Button, Bridge-Pfad); offen: **Echtzeit-Streaming-Architektur-Entscheidung**, 4-s-Scanner (Hör-Freigabe), CI-Smoke |
| Dehum | OFFEN | hum 0/3; Witness-hum verfehlt (q 0,61) → **Witness-Runde 2** (hyperparameter/Nacharbeit, GPU nach VinylNet) oder n≥10-Kriterium (a) |
| Wow/Flutter | OFFEN | wowflutter 0/3; Witness-wow verfehlt (q 0,95) → Witness-Runde 2 + WF4-Neubewertung |
| Pitch/Speed | OFFEN | pitch_drift 0/3 → n≥10 (B3 läuft an) → danach defekt-spezifischer Kanal oder Kriterium (a) |
| Dynamik-Rückbau | OFFEN | kompression 0/3 → n≥10 (Batch 4) + dynamic_compression_excess-Kanal-Gate ist bestanden (q 1,19) → A/B |
| Phasen-/Kammfilter | OFFEN | **n≥10 liegt vor: 3/10 besser, ΔMOS +0,021, 0 Überrestaurationen** — Kriterium (a) noch nicht erfüllt (≥5/10) → defekt-spezifischer Kanal (kammfilter hat keine k398-Regionen — Datenpfad fehlt) |
| Dropout/Riss | OFFEN | dropout 0/3; Witness-dropouts verfehlt (nw) → Witness-Runde 2; dropout_head_contact-Kanal Gate-bestanden (q 2,0, AUC-Caveat) → A/B |
| Master-Finish/Lautheit | OFFEN | lautheit 0/3 → n≥10 (B3) + Loudness-Gate-Evidenz |

**Muster der Lücke:** Die Bestands-Pipeline bleibt in den 45 DRT-12-
Vergleichspaaren ohne Überrestauration. Der gemeinsam aktivierte 12-Flag-Satz
zeigt dagegen 17 MOS- und 4 Defektlast-Regressionen sowie 2
Überrestaurationen. Der Abstand zu RX12 ist damit nicht durch pauschale
Aktivierung geschlossen; die nächsten Messungen müssen einzelne Kanäle
kausal isolieren.

## B. Maschinen-Arbeit in Flug (autonom, detached — überlebt Laufzeit-Neustarts)

| Kette | Stand | liefert für Zeile(n) |
|---|---|---|
| Signal10d-Schadensdaten | **CPU-AKTIV, Checkpoint alt** — 10 Titel × 4 Rezepte; zuletzt 01:35 CET aktualisiert, bis zum Check um 02:25 kein neuer Datei-Checkpoint. Ein früherer Batch-4-Log endete mit Exit 134; der aktuelle Generator ist ein separater Prozess. | Dynamik, Verzerrung, Vinyl, Kassette |
| Vocal-Harshness-Expansion | **CPU-AKTIV, Fortschritt nicht belegt** — 20 Titel, Seed 42; seit 22:24 CET kein neues Ausgabe-Artefakt, obwohl der Prozess CPU nutzt. Das Skript schreibt erst nach einer vollständigen Produktions-Restaurierung pro Rezept; daher nicht als abgeschlossen oder sicher hängend klassifizieren. | Gesangs-Härte |
| WQ4 Witness-Training | **ABGESCHLOSSEN, alle 3 Gates fehlgeschlagen** — Q-Median/Never-worsen-Anteil: Dropouts −0,2632/0,0179; Wow 1,0035/0,8387; Hum 0,8717/0,8750. Der Schutzarm lief mangels exakter Produktionszuordnung als Rohkopf-Diagnostik (`dropouts` ≠ `dropout_head_contact`; für `wow`/`hum` fehlt ein passendes DSP-Modul). Außerdem verglich der Trainings-Proxy beschädigten Eingang mit Reparatur-Ausgang statt Clean→Clean; `nw_min` war damit kein gültiger Schutzarm-Beleg. Reports speicherten weder `nw_min` noch je-Epoche-`nw`, daher ist die historische Auswahl nicht rekonstruierbar. Der Trainer misst nun Clean→Clean, erzwingt die Schranke ohne Fallback und protokolliert Auswahlwerte. Die alten Gates bleiben NO-GO; `WQ4_END=0` ist nur der Orchestrator-Exit. | Dropouts, Wow, Hum |
| Ketten-Watchdog | **AKTIV** als detached Prozess; `signal10-resume`- und `crackle-kette`-Ketten werden orchestriert. | Kettenfortschritt |
| Kanalsatz-A/B-Vollmessung | 45/45 Paare; 12-Flag-Gesamtsatz verworfen (17 MOS-Regressions-JND, 4 Defektlast-Regressions-JND, 2 Überrestaurationen) | Isolierte Kanal-/Rezept-Ablation |

**Resume-Berichtskonsistenz (2026-11-21):** Fertige Paare der
Vocal-Harshness-Expansion sowie fertige Titel der Crackle-Expansion gingen beim
Resume im Abschlussbericht verloren. Die Skip-Pfade bauen `eintraege` jetzt
vollständig wieder auf; Crackle rekonstruiert dafür deterministisch auch die
Rezeptparameter und stellt fehlende Clean-Symlinks wieder her. Beide
Generatoren akzeptieren ein Resume-Paar nur, wenn Schaden und Restaurat
nichtleer sind. Modellfreie Regressionstests decken beide Resume-Fälle ab. Der aktuell laufende
Vocal-Prozess hat den Fix bereits nicht mehr im Speicher. Nach seinem Ende
muss der Bericht daher mit einem Resume-Durchlauf aus den fertigen Artefakten
neu aufgebaut werden; ein Neustart der Restaurierungen ist dafür nicht nötig.

## C. Der kritische Pfad: kombinierter §4-Batch (menschlich)

Der vollständige 12-Kanal-Satz hat DRT-12 nicht bestanden; daraus wird kein
kombiniertes §4-Paket erstellt. Erst isolierte Kandidatenmessungen können eine
kleinere §4-Shortlist begründen. Zusätzlich bleiben Declip, Stereo, BW-V5,
material_crnn, MUSHRA n≥30 und die noch offenen Sign-offs im Hör-Batch
(`docs/guides/HOERBATCH_DECLIP_MUSHRA_W1.md`).

## D. Daten-/Lizenz-Lücken (folgen §9/7)

Vinyl↔Reissue ≥30, Echt-Träger, Genre-Labels, MUSHRA n≥30; lossless-Upgrade der
CC0-Leerrillen (freesound-Originale, IDs im Manifest).

## E. System-Hygiene

UAT-Neulauf (trainingsfreies Fenster; `test_daily_gate_stored_recently` rot bis dahin);
W-Stufen 0/13 aktiv (Bremser ohne Beschleuniger — der Witness/DRT-Zyklus ist der neue
Beschleuniger); cantus bleibt `status=training` (Gate ehrlich nicht bestanden).

## F. Reihenfolge bis „14+1 GESICHERT“

1. DRT-12-Vollmessung ist abgeschlossen und der globale Satz verworfen; nach Ende der aktiven CPU-Ketten isolierte Kanal-/Rezept-Ablation durchführen.
2. B2/B3-Bewertungen ernten → OFFEN-Zeilen mit n≥10 belegen (Kriterium (a) prüfen).
3. Crackle-Kette → crackle trainieren/gaten → Decrackle-Richtung GESICHERT.
4. VinylNet fertig → Never-worsen-Gate + RX-12-Vergleich → Decrackle.
5. Witness-Runde 2 (dropouts/dropout_oxide/wow/hum/tape_splice) im GPU-Fenster nach VinylNet.
6. HF-Residual-Entscheidung + Echtzeit-Preview-Architektur-Entscheidung (einzige Punkte,
   die eine Feature-Entscheidung brauchen).
7. **Fertig-Kriterium:** alle 15 Zeilen ≥ GESICHERT — 0 OFFEN, 0 LUECKE; der letzte
   Schritt je Zeile ist der §4-Sign-off.

## G. Datenlücken-Beschaffung (2026-11-20, eigenständig)

| Lücke | Status | Weg/Ergebnis |
|---|---|---|
| Genre-Labels | ✅ GESCHLOSSEN (maschinell) | `scripts/erstelle_genre_labels.py` (neu): 100 MUSDB-Train + Seed-42-Quellen werden mit dem Produktions-Klassifikator (PANNs-Prior + Schlager-Speziallogik + Open-Set) auf 20-s-Ausschnitten selbst gelabelt → `data/genre_labels.jsonl` + Beleg `2026-11-20_genre_labels.json` (Schema `aurik.genre_labels/1`, confidence je Titel; maschinell = kein §4-Ersatz, schließt die Label-Existenz-Lücke). |
| CC0-Leerrillen-Lossless | ⏳ BLOCKIERT (Credential) | Freesound-Download-API 401 ohne Token — der Lossless-Upgrade-Pfad braucht einen Freesound-API-Token (Konto). Beim Vorliegen: `curl -H "Authorization: Token <TOKEN>" -L https://freesound.org/apiv2/sounds/<id>/download/ -o <original.wav>` für die 6 IDs im `data/vinyl_noise_bank/MANIFEST.json`. |
| Echt-Träger / Vinyl↔Reissue ≥30 | ✅ GESCHLOSSEN (SOTA-Substitut, §9/7) | Musik-tragende Echt-Träger/Vergleichspaare sind lizenzrechtlich NICHT eigenständig beschaffbar (Urheberrechte Dritter). Das normkonforme Substitut steht bereits: echte CC0-Leerrillen-Bank (Träger-Oberfläche, materialkorrekt) + deterministisches Vinyl-Rezept auf CC0-Quellen (Volltitel-Schaden) — genau der Aufbau der VinylNet/Crackle-Ketten. |
| MUSHRA n≥30 | ⏳ MENSCHLICH | per Definition menschliches Hörurteil (§4) — keinem autonomen Weg zugänglich; der kombinierte §4-Batch ist vorbereitet. |

## H. Produktionsstabilität — BANQUET-Session-Race (2026-11-21)

Der Fehlerverlauf `/tmp/signal10_resume.log.fail_hist` enthält
`AttributeError: 'NoneType' object has no attribute 'run'` in
`BanquetVinylPlugin._infer_window`. Die Session konnte nach dem Verfügbarkeits-
Check durch parallele Quarantäne oder Model-Eviction auf `None` gesetzt werden.
Der Produktionspfad hält jetzt einen gelockten lokalen Session-Snapshot für
jedes bereits gestartete Fenster; Eviction und Reload-Zustand verwenden
denselben Lock. Ein deterministischer Thread-Test deckt genau dieses Timing ab.

**Status:** Der Python-seitige Session-Race ist behoben und durch 9 BANQUET-
Provider-Tests verifiziert. Ein separat protokollierter nativer Exit 139 ist
damit nicht kausal erklärt und bleibt offen. Der aktuelle signal10-Batch war
bei der letzten Prüfung noch aktiv; sein Ergebnis ist nicht vorweggenommen.
