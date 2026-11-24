# Aurik 10 — Spec 26: Defekt-Reparatur-Trainingsvertrag (DRT)

> **Zweck:** Normative Leitplanken für das Training defekt-spezifischer
> Reparatur-Köpfe (Witness-Kopf-Verfahren, D-K3-95-Folge). Verhindert
> unnötige Trainingsläufe, indem jede Verlustrunde der Vergangenheit
> (M9-V1/V2 2026-11-14, Witness-Kopf-V1 2026-11-15, Schutz-Arm-Versager
> 2026-11-15, Produktions-Paritäts-No-Op 2026-11-15) strukturell
> ausgeschlossen wird. Die externe RX-12-Abnahme ist in DRT-16 normiert und
> wird zusätzlich durch `aurik-wettbewerbs-paritaet` gepinnt.
> Maschinell erzwungen durch den Pre-Commit-Hook
> `aurik-drt-vertrag` (`scripts/compliance/drt_contract_gate.py`, fail-closed).
> Arbeitssprache: Deutsch. Regeländerung ⇒ Gate-Skript nachziehen (AGENTS.md §2).

## Warum diese Spec — die vier Verlustrunden

1. **M9-V1/V2 (TF-GridNet-Denoiser):** optimiert wurde Mask-MSE, gemessen wurde
   Wohlklang — Zielfunktion ≠ Gate-Ziel ⇒ zwei volle GPU-Läufe ohne Wirkung.
2. **Witness-Kopf-V1:** der Witness-Term war durch falsche Mel-Skalierung
   (log10 statt slaney + 10·log10 + amin) numerisch wirkungslos ⇒ Q=None ⇒ Neulauf.
3. **Schutz-Arm-Versager (6 Klassen):** der Kopf überrestaurierte auf clean-Regionen,
   weil das Training nie clean→clean gesehen hatte ⇒ Gate verfehlt ⇒ Nacharbeit.
4. **Produktions-Paritäts-No-Op:** k398-Regionen sind 0,1 s; Trainer/Gate padden auf
   1 s, die Produktions-Trigger-Schleife tat es nicht ⇒ Kanal war ein stiller No-Op.

Jede Runde hätte eine der folgenden Regeln abgefangen — **vor** dem GPU-Lauf.

## DRT-1 — Zielfunktion = Gate-Ziel

Jeder Trainer für einen defekt-spezifischen Reparatur-Kopf MUSS:

- ein **eingefrorenes Gate-Witness** nutzen (Parameter `requires_grad=False`;
  BatchNorm auf eval, GRU im Training-Modus — MIOpen-Backward-Befund 2026-11-15),
- den Witness als **differenzierbaren Loss-Term** führen (Gradient durch die
  Mel-Repräsentation: slaney-Norm + `10·log10` mit `amin`-Clipping wie
  `compute_mel_produktion` — jede andere Skalierung sättigt das Witness),
- die **Checkpoint-Auswahl nach dem Gate-Score beider Arme** treffen (DRT-5),
  niemals nach einem Einzel-Verlust.

## DRT-2 — Zwei-Arm-Loss (Never-worsen ist ein Trainings-Ziel)

Der Verlust MUSS zwei Arme enthalten:

- **Gewinn-Arm:** `L1(schaden→clean)` + λ_w · BCE(Witness(out), 0);
- **Schutz-Arm:** Identitäts-Anker `λ_id · L1(kopf(clean), clean)` mit λ_id > 0.

Der Never-worsen-Schutz des Gates ist damit eine TRAININGS-Eigenschaft, kein
Nach-Filter. Verfehlt ein Kanal den Schutz-Arm, ist die Ursache das Training
(§V7 (copilot-instructions.md): Ursache statt Symptom — niemals die
Gate-Schwelle senken).

## DRT-3 — Pre-flight vor jedem GPU-Lauf

Vor jedem vollen Lauf MUSS ein CPU-Smoke belegen:

- Witness-Kontrast auf den geladenen Regionen: `p_schaden − p_clean ≥ 0,5`
  (sonst kann der Witness das Gate-Ziel nicht tragen — Abbruch VOR dem Training),
- beide Loss-Arme numerisch aktiv (Kontrast-/Größen-Check),
- Determinismus (§G5 (copilot-instructions.md)) und Witness-Checkpoint
  fail-closed (§V6 (copilot-instructions.md)).

## DRT-4 — Parität Training/Gate/Produktion

Fenster, Zero-Padding auf 1 s, RMS-Normalisierung (`RMS_ZIEL` aus einer Quelle,
Import statt Duplikat — §V25 (copilot-instructions.md)), Mel-Repräsentation und
Resampling müssen zwischen Trainer, Gate und Produktions-DSP **identisch** sein.
Jeder Kanal braucht einen **Produktions-Paritäts-Test** (kurze Region triggert,
Ausgabe-Länge = Eingangs-Länge). Muster:
`tests/unit/test_witness_kopf_produktion_paritaet.py`.

## DRT-5 — Gate-Schwellen unantastbar, Gate-Ziel-Auswahl

- Aktivierungs-Gate: **Witness-Q ≥ 0,5** (Median, schaden-Regionen des
  Holdout-Titels, leak-frei) **UND** **Never-worsen ≥ 0,95** (clean-Regionen).
  Beide Schwellen sind normativ fix.
- Modellauswahl im Training über denselben Score (Gewinn- + Schutz-Arm).
- Verfehlen ⇒ Nacharbeit nur über Ursache (Training, Trigger-Schärfe), nie über
  Schwellwert-Senkung oder phasen-individuelle Ausnahmen.

## DRT-6 — Ehrliches Scheitern, kein toter Code

- Gate verfehlt ⇒ Kanal bleibt **report-only**: kein Flag, keine Verdrahtung,
  Q-Erstbeleg mit `status: gate_verfehlt`, Allowlist-Buchung „bewusst-verwaist"
  mit Beleg-Verweis (Modell-Verdrahtungs-Audit).
- Gate bestanden ⇒ Verdrahtung hinter `music_model_flags.use_witness_kopf_<klasse>`
  (Default AUS = bit-identisch; Flag-Flip NUR über §4-Hör-Sign-off), Produktions-
  Konsument in phase_07 mit Doppel-Guard (Witness-Trigger + Harmonik-Proxy gegen
  Input und Amtsinhaber), §4-Hörproben (nur wirksame Regionen, A≠B verifiziert).
- Trainings-/Gate-Infrastruktur bleibt wiederverwendbar (kein toter Code).

## DRT-7 — Produktions-Semantik-Gate für ALLE Reparatur-Kandidaten

Auch BESTEHENDE Kandidaten (Musik-Declip-Kopf, A-SPADE, de_crackle-CRNN,
Dritt-Denoiser) brauchen vor Verdrahtung bzw. Aktivierung den Gate-Lauf über
den PRODUKTIONS-Pfad (Trigger-Maske + Never-worsen-Guard, DRT-4-Semantik) —
ein Regionen-/LOO-Beweis allein ist nicht hinreichend. Befund 2026-11-15: der
„objektiv bestandene" Declip-Kopf verfehlt auf Ganz-Titel-Ebene (clipping
Δ defektlast +0,036) — sein Beweis deckte die Produktions-Ebene nicht ab.

## DRT-8 — Ganz-Titel-Sinn-Beweis als Aktivierungs-Beleg

Vor jedem §4-Sign-off MUSS der Ganz-Titel-Vergleich auf dem zuständigen
Rezept-Material belegt sein (`scripts/erstelle_schadens_proben.py` +
`scripts/bewerte_schadens_proben.py`: defektlast/musik_mos, pegelgematcht,
Seed 42). Regionen-Q (DRT-5) ist notwendig, nicht hinreichend — die
Aktivierungs-Entscheidung braucht beide Ebenen. Befund 2026-11-15 (voller
12-Paare-Lauf `2026-11-15_sinnbeweis_v2_12_paare.json`): 0/12 besser im
Zwei-Metrik-Maß; der 4-Paar-Teilbefund war ein Kontext-Artefakt (DRT-9-korrigiert).
Reproduzierbar: Boogie-kassette defektlast −0,079 (mos-neutral).

**Produktions-Semantik im mos-Arm (2026-11-24):** Der musik_mos-Arm des
Gate-Zeugen misst die PRODUKTIONS-Ausgabe (Trigger + Never-worsen-Guard),
nicht den rohen Kopf — Regionen, auf denen der Witness keinen Schaden sieht
(P ≤ Schwelle), werden durchgereicht und dürfen den mos-Anteil nicht belasten
(tdhl-Region-4-Befund: P(schaden)=0,0015, Modul Pass-Through, roher Kopf
−0,091; nach Korrektur mos_anteil 1,0).

## DRT-9 — No-Op-Erkennung, kein stilles Verschwinden

Kommt ein Kanal auf seinem Zielmaterial bit-identisch zurück (No-Op — Befund
2026-11-15: mp3_128-Paar), wird das im Beleg als „unwirksam (No-Op)" geflaggt
mit Ursachen-Vermerk (Material-Erkennung? Phase nicht aktiv? Flag AUS?).
Kein stiller No-Op, kein Schein-Erfolg, keine Aktivierung ohne Wirkungs-Beleg.

## DRT-10 — Ganz-Titel-Zwei-Metrik-Never-worsen mit Überrestaurations-Erkennung

Der Ganz-Titel-Sinn-Beweis (DRT-8) MUSS beide Arme prüfen: defektlast-Gewinn
UND musik_mos-Never-worsen. Ein Paar, das defektlast deutlich gewinnt (≥ JND)
und musik_mos deutlich verliert (≥ JND_MOS = 0,05), wird als **Überrestauration**
geflaggt (`ueberrestauration`-Feld in `bewerte_schadens_proben.py`) und ist
**nicht aktivierungsfähig** — Ursachen-Analyse ist Pflicht (§V7 (copilot-instructions.md)),
niemals eine Schwellwert- oder Mess-Anpassung. Befund 2026-11-15: Jets-vinyl
defektlast −0,203 bei musik_mos −0,551 — die Regionen-Gates (Witness-Q) sehen
diesen Wohlklang-Verlust nicht; erst die Ganz-Titel-Zwei-Metrik-Messung deckt ihn auf.
**Produktions-Verdikt — Invariante für JEDEN Lauf (M1, 2026-11-17):** Die
Überrestauration wird zur Laufzeit **unabhängig von aktiven Reparatur-
Kandidaten** geflaggt (`wohlklang_zeuge.ueberrestauration`) und der Lauf als
`degraded` mit `fail_reason=RESTORATION_DRT10_UEBERRESTAURATION` exportiert
(§0c: bestmögliches sicheres Ergebnis, kein Hardstop —
`drt10_verdict_in_metadata` in `backend/core/dsp/wohlklang_zeuge.py`,
Verdrahtung in `unified_restorer_v3._drt10_verdict_in_metadata` im Einzel-/
Options-/Chunked-Pfad). **Identitäts-Rollback:** Liegt keine hohe Hör-Instanz
vor (HPI < 0,75), wird das Ergebnis auf den Eingang zurückgesetzt
(`drt10_rollback=True`) — Ganz-Titel-Never-worsen als Produktions-Invariante.
Hält die Hör-Instanz hoch (HPI ≥ 0,75), gilt die Hörordnung-§7-Konfliktregel:
Messartefakt-Verdacht, KEIN Rollback (`_drt10_rollback_zulaessig`), nur
degraded + Vorlage an die Hör-Instanz.

## DRT-11 — Trigger-Kontext-Parität (Sperre gegen Fremd-Kontext-Fehltrigger)

Ein Reparatur-Kanal darf NUR auf Material feuern, dessen Kontext seinem
Trainings-Kontext entspricht. Die Witness-Köpfe sind auf k398-Regionen trainiert —
auf Rezept-Material (vinyl/kassette/clipping/mp3_128) triggern sie falsch und
blockieren die Bestands-Reparatur. Ablations-Befund 2026-11-15 (Boogie):
Bestands-Pipeline OHNE Kanäle 4/4 besser; MIT Kanälen 0/4. Vor jeder Aktivierung
braucht der Kanal eine Kontext-Sperre (nur auf detektiertem k398-Material feuern,
sonst Bypass) — die Sperre ist Teil der Verdrahtung, nicht des Trainings.

## DRT-13 — Phasen-Attribution & Teil-Rollback statt Ganz-Titel-Vernichtung

DRT-10/M1 schreibt bei Überrestauration den Identitäts-Rollback des ganzen
Titels vor — damit gehen belegte Defekt-Gewinne verloren (Jets-vinyl: defektlast
−0,203 bei musik_mos −0,551; das Ohr bekommt „alles oder nichts“). DRT-13
schärft das Verhältnis:

1. **Reparatur-Phasen-Snapshot:** Für Reparatur-Phasen (Declip, Denoise,
   De-Crackle, Witness-Köpfe) hält die Pipeline den Phasen-Eingang chunk-lokal
   als Snapshot (`backend/core/dsp/reparatur_phasen_attribution.py`) —
   Teil-Rollback = Snapshot statt Phasen-Ausgabe, nur in den Regionen, die die
   Phase verändert hat (Region-Maske, nie der ganze Titel).
2. **Attribution:** Der Verdict misst vor dem Voll-Rollback die Variante mit
   zurückgenommener(n) Reparatur-Phase(n) („Phase-AUS-Variante“) — wer den
   mos-Verlust verursacht, wird gemessen statt vermutet; das Ergebnis wird in
   `phase_attribution` berichtet.
3. **Teil-Rollback vor Voll-Rollback:** Ist die Phase-AUS-Variante frei von
   Überrestauration, wird SIE das Ergebnis (Teil-Rollback) — der
   Identitäts-Rollback (M1) bleibt NUR der Fallback, wenn keine Attribution
   greift oder auch die AUS-Variante überrestauriert.
4. **Song-Ebene und Chunk-Parität:** DRT-10 ist ein Ganz-Titel-Verdikt. Im
   Chunked-Pfad darf es deshalb nicht je Chunk über den endgültigen
   Song-Output entscheiden; der Verdikt und die DRT-13-Kandidatenmessung
   laufen nach der Chunk-Assembly auf dem vollständigen Song. Chunk-lokale
   Phase-AUS-Deltas werden mit denselben RMS-Match-Faktoren und
   Crossfade-Gewichten wie die Chunk-Assembly auf den Kandidaten übertragen.
   Samples außerhalb der erfassten Reparaturregionen bleiben bit-identisch.
   Die zusammengesetzte Phase-AUS-Variante wird gegen den vollständigen
   Eingang gemessen; nur ein verfügbarer Kandidat ohne Überrestauration darf
   den Identitäts-Rollback ersetzen. Fehlende, nicht ausgerichtete oder
   unmessbare Attribution führt fail-closed zu M1 und wird im Report begründet.
5. **Song-Isolation:** Snapshot-Zustand ist pro Ausführungskontext isoliert
   und wird vor jedem neuen Song bzw. Chunk zurückgesetzt (§G1/§V8
   (copilot-instructions.md)). Modul-globaler Zustand darf weder einen
   Folgesong noch parallele Restaurierungen beeinflussen.
6. **Pfadtransparenz:** Jeder DRT-10-Rollback berichtet DRT-13 als
   `applied` oder `unavailable` mit Grund. Der Options-Pfad ohne
   Reparatur-Phasen-Snapshots muss `unavailable` melden und M1 beibehalten;
   fehlende Attribution darf nicht als erfolgreicher Teil-Rollback erscheinen.

Kein stiller Flip: Teil-Rollback ist ein deklarierter, belegter Eingriff
(§V7 (copilot-instructions.md)); der M1-Fallback bleibt unverändert, bis die
Attribution nachweislich greift. Der bisherige Ein-Pfad-Stand (2026-11-17)
war Stufe 1; Stufe 2 verlangt die Song-Ebene, Chunk-Parität und Isolation
nach den obigen Punkten, bevor DRT-13 als pfadübergreifend vollständig gilt.

## DRT-14 — Wirksamkeits-Gebot: kein „neutral ist genug“

Die Norm kennt bisher nur Never-worsen — Wirksamkeit ist nicht gefordert;
der 0/12-Sinn-Beweis ist normativ legal. DRT-14 ergänzt den positiven Arm:

1. **Reparatur-Nachweis je Lauf:** Für jede erkannte Defektklasse mit
   Schaden-Score ≥ Schwelle MUSS der Lauf einen Reparatur-Nachweis führen
   (Regionen-Q bzw. defektlast-Delta aus `defekt_delta`/Wohlklang-Zeuge)
   ODER die Klasse als `unrepariert` mit Ursache berichten (Metadaten
   `reparatur_nachweis` / `unrepariert[]`, Helper im Wohlklang-Zeugen).
2. **Kein Hard-Fail:** Der Nachweis ist ein Zeuge (Hörordnung §7) — Eskalation
   über die Hör-Instanz; nie ein stiller Abbruch oder Rollback allein wegen
   fehlender Wirksamkeit.
3. **Ehrliche Bilanz:** Ohne Wirksamkeits-Bericht ist „besser/schlechter“ auf
   Klassen-Ebene nicht bewertbar — der Bericht ist die Voraussetzung für
   künftige Sinn-Beweis-Vergleiche.
4. **Abdeckungs-Pflicht (2026-11-17):** Der Ganz-Titel-Sinn-Beweis MUSS alle
   erzeugbaren Rezept-Klassen abdecken (nicht nur vinyl/kassette/clipping/
   mp3_128). `scripts/erstelle_schadens_proben.py` trägt mindestens die
   fünfzehn Rezept-Klassen vinyl, kassette, clipping, mp3_128, hum,
   wowflutter, kompression, kammfilter, verzerrung, clicks, pitch_drift,
   dropout, raum, lautheit, amp_drift; jede nicht gemessene erzeugbare
   Klasse ist im Beleg als `nicht_gemessen` auszuweisen — die Abdeckung wird
   maschinell gepinnt (DRT-14-Coverage im Pre-Commit-Gate).

## DRT-15 — B3: Ablehnung auf Substanz nur in der besten Komposition

1. **Kompositions-Pflicht:** Eine Einzel-Stufe darf auf Substanz ABGELEHNT
   werden, wenn ihre Messung in der besten gefundenen Komposition erfolgt ist
   (`scripts/b3_komposition_suche.py`) — sonst ist das Urteil auf
   „Einzel-Wirkung nicht signifikant“ zu beschränken.
2. **Sub-JND ≠ Ablehnung:** Liegt der Kompositions-Vorteil unter der JND,
   gilt „Messgrenze erreicht“ (kein Qualitätsurteil) — die Stufe bleibt
   mess-offen, nie still verworfen.
3. **Beleg-Pflicht:** Jeder B3-Beschluss trägt `kompositions_status`:
   `gemessen` / `sub_jnd` / `nicht_gemessen` (Schema `aurik.objektivbeschluss/1`).

## Normative Einordnung

Diese Spec konkretisiert für das Reparatur-Training: §G5 (copilot-instructions.md)
(Determinismus), §V6 (copilot-instructions.md) (kein Silent-Failure), §V7
(copilot-instructions.md) (Ursache statt Symptom), §V25 (copilot-instructions.md)
(kein hartcodierter Schwellwert), §III (copilot-instructions.md) (Soft-Knee
6 dB/200 ms), Hörordnung §4 (Sign-off für klangwirksame Aktivierungen).
DRT-16 konkretisiert zusätzlich den externen Wettbewerbsvergleich; die
interne Trainings- und Aktivierungsevidenz aus DRT-1…15 ersetzt ihn nicht.

## DRT-12 — Kontroll- und Kanalsatz-A/B vor dem §4-Sign-off (2026-11-19)

Befund: Die Ganz-Titel-Sinn-Beweise (2026-11-15: 0/12 besser; 2026-11-18:
6/45) wurden mit **deaktivierten** Witness-Kanälen gemessen (alle Flags
Default AUS) — sie belegen den Bestand, nicht den Kandidaten-Zustand. Der
12-Paare-Lauf MIT den 10 Kanälen zeigte 0/12; erst die Kontrolle OHNE Kanäle
(4/4 + 2/4 besser) offenbarte, dass Kanäle die Bestands-Reparatur blockieren.
Ohne identische Kontrolle wäre die Verschlechterung unsichtbar geblieben.

1. **Gepaarte Kontrolle:** DRT-8 MUSS dieselben Schadensdateien einmal mit
   Kandidat/Kanalsatz AN und einmal mit der Bestands-Pipeline AUS messen
   (gleiche Metriken, deterministische Auswahl, Seed 42). Das Delta ist nur
   zwischen diesen beiden Läufen zurechenbar.
2. **Aufnahme-Kriterium:** Nur Kandidaten, deren A/B auf den zugehörigen
   Rezept-Paaren nie schlechter als der Bestand um mehr als JND sind und
   mindestens ein Paar messbar verbessern, kommen in den §4-Batch. Ein
   verbleibendes Überrestaurations-Gate ist weiterhin ein Ausschlussgrund.
3. **Mess-only und Produktionsschutz:** `scripts/messe_kanalsatz_ab.py`
   setzt Flags nur prozess-lokal, schreibt keine Flag-Datei und löst keinen
   Produktions-Flip aus. Produktions-Defaults bleiben AUS, bis der §4-Batch
   freigegeben ist (§G1 (copilot-instructions.md)-Bit-Identität).
4. **Kombinierter §4-Batch:** Alle aufgenommenen Kanäle werden als EIN Satz
   gehört (eine Sitzung, ein Verdikt über die Menge).

## DRT-16 — Externer RX-12-Paritätsvergleich

DRT-1…15 belegen Trainingsqualität, interne Wirksamkeit, Schutz und
Aktivierungsfähigkeit. Diese Belege allein beweisen keine Wettbewerbsparität.
`PARITAET` im `.github/WETTBEWERBS_PARITAET.md` erfordert den folgenden
externen Vergleich; `GESICHERT` bleibt ein interner Status.

Der Vergleich ist ein perzeptueller Ergebnisvergleich, keine Forderung nach
sample-identischem Output, identischen Algorithmen oder gleicher Bedienung.
Der Holdout- und Panel-Aufwand ist die Nachweisschwelle für genau die
beanspruchte Disziplin, kein kurzfristiges Gesamtprodukt-Ziel. Der realistische
Fortschritt läuft stufenweise: internen Produktionspfad und Never-worsen
belegen, Verbesserungen pro Disziplin messen, dann externe RX-12-
Nichtunterlegenheit samt Hör-Sign-off nachweisen. Nicht belegte Disziplinen
bleiben offen; Erfolge werden nicht auf andere Defekte oder Referenzen
extrapoliert.

1. **Vergleichsidentität:** Für jede Disziplin und jede dort genannte
   Referenz wird ein eigener Beleg erzeugt. Er nennt Produkt, exakte Version
   und Build, verwendetes Modul, alle relevanten Einstellungen und deren Hash.
   Für RX ist die Referenz ausdrücklich **iZotope RX 12**; ein Vergleich mit
   einer anderen RX-Version wird nicht auf RX 12 übertragen.
2. **Gepaarter Produktionspfad:** Aurik und die Referenz verarbeiten denselben
   bit-identischen Schadens-Input. Der Aurik-Commit, die Version, der
   tatsächlich erreichte Produktionspfad und alle aktiven Flags werden
   aufgezeichnet. No-Op, nicht erreichte Phase, fehlender Kanal oder
   simulierte Referenzausgabe disqualifizieren den Lauf. DRT-12s interne
   Kandidat-AN/Bestand-AUS-Ablation bleibt zusätzlich erforderlich; sie ist
   nicht der RX-Vergleich.
3. **Holdout und Stichprobe:** Der objektive Ganz-Titel-Screen umfasst
   mindestens 10 unabhängige Holdout-Titel je Disziplin. Keine Quelle oder
   Titel-Identität darf zwischen Training/Kalibrierung und Holdout geteilt
   werden. Ein gehashter Manifest-Beleg weist Umfang und Split nach. Weniger
   Titel, Korpus-Leakage oder fehlende Titelzuordnung ergibt `OFFEN`, nicht
   `PARITAET`.
4. **Nichtunterlegenheits-Grenzen:** Pro Titel werden die gepaarten Deltas
   Aurik minus Referenz berechnet. Mit der kanonischen
   `wohlklang_zeuge.JND_TOLERANZ` muss die untere Grenze des zweiseitigen
   Student-t-95-%-Konfidenzintervalls (df = `n_titles - 1`) für `musik_mos`
   mindestens `-JND_TOLERANZ` und die obere Grenze für `defektlast` höchstens
   `+JND_TOLERANZ` sein. Das Gate berechnet beide Intervalle aus den
   veröffentlichten Einzeltitel-Deltas neu; die JSON-CI-Werte müssen bis auf
   1e-6 übereinstimmen. Jede Paar-Zeile bindet ihre beiden Metrik-Deltas an
   `source_id` und die bit-identischen PCM-Hashes; die Objective-Arrays müssen
   in derselben Reihenfolge mit diesen Paarwerten übereinstimmen. Ein bloß
   behauptetes `noninferiority_passed`, ein vom Datensatz abweichendes CI oder
   eine nicht gebundene Delta-Liste ist ungültig. Beide Grenzen müssen
   bestehen; Mittelwert, Einzelmetrik oder interne Aurik-Baseline genügt nicht.
   Überrestaurationen und Verletzungen der Hör-Invarianten sind unabhängig
   davon Ausschlussgründe.
5. **Menschliches Urteil:** Für `PARITAET` ist zusätzlich ein verblindeter,
   randomisierter, pegelgematchter MUSHRA-/§4-Vergleich mit mindestens
   30 Hörer:innen und dokumentiertem Sign-off erforderlich. Das Panel muss
   explizit `nicht_unterlegen` urteilen; ein uneindeutiges oder fehlendes
   Verdikt bleibt `OFFEN`. Die Hörordnung
   bleibt entscheidend; widersprechende Metriken werden nach Hörordnung §7
   behandelt, niemals still als bestanden gewertet.
6. **Kanonischer Beleg:** Das JSON-Schema `aurik.rx12_parity/1` enthält
   mindestens:
   `discipline`, `schema`,    `reference` (product/version/build/module/
   settings_sha256), `aurik` (version/commit/production_path/active_flags/
   settings_sha256),    `dataset` (paired_input_manifest_sha256/holdout/
   n_titles/pairs; each pair records source_id, identical
   input_pcm_sha256_aurik/input_pcm_sha256_reference, musik_mos_delta and
   defektlast_delta),
   `objective` (jnd/jnd_source/per-title musik_mos and defektlast deltas/
   musik_mos_delta_ci95/defektlast_delta_ci95/noninferiority_passed),
   `panel` (design/n_listeners/blind/randomized/level_matched/verdict/signoff) und
   `decision`. Der Bericht liegt unter `docs/reports/current/`; die
   Registerzelle `Evidenz` enthält den repo-relativen JSON-Pfad in Backticks.
   Das Wettbewerbs-Paritäts-Gate prüft Schema, Referenz, Disziplin,
   Holdout-Umfang, Konfidenzgrenzen und Panel-Felder, bevor `PARITAET`
   zulässig ist.
7. **Keine Extrapolation:** Ein bestandener Beleg gilt nur für die berichtete
   Disziplin, Referenz, Version, Konfiguration und Holdout-Abdeckung.
   `NICHT_MESSBAR`, fehlender lizenzierter Zugriff, ein synthetischer
   RX-Ersatz oder ein bloß geplanter Hörtest zählt nie als Parität.

## DRT-17 — Sechs Klang-Muster (Natürlichkeits-Kalibrierung, 2026-11-24)

Jeder Reparatur-Kandidat MUSS die sechs Klang-Muster der Hörordnung §8c
(§G192–§G197 (GEBOTE.md, Kategorie XXVII)) einhalten:

1. **Clean-Referenz-Kalibrierung:** Q-Ziel ist q ≈ 1,0 — Überschreitung ist
   Überrestauration und wird wie ein Defekt behandelt (Instrumente:
   `--lambda-nor`, `--q-ziel 1.0`).
2. **Zurückhaltung durch Detektor-Präzision:** Eingriff nur auf sicherem
   Witness-Trigger; Detektor-Verbesserung geht stärkerer Reparatur vor.
3. **Wahrnehmungs-Lehrer:** bei Metrik/Wahrnehmungs-Dissens kalibriert der
   musik_mos-Teacher im Training (`--lambda-mos`); Entscheidung bleibt bei
   Gate und §4.
4. **Audibility:** Ziel ist die Maskierungsschwelle, nicht Mess-Null.
5. **Determinismus:** last-/speicherabhängige Klang-Pfad-Wahl ist verboten
   (PLM-Pin; §G5 (copilot-instructions.md)).
6. **Daten-Authentizität:** reale Quellen; synthetische Ersatz-Rezepte nur mit
   Scanner-Nachweis der Defekt-Signatur.

Beleg: Trennung 2026-11-24 — 17/17 Witness-Kanäle Weizen
(`docs/reports/current/2026-11-24_spreu_weizen_trennung.json`; ts q 1,0893,
jitter 1,0063, dropouts 1,0027, dhc 1,0732). Kein Flag-Flip, keine
Schwellenänderung (DRT-5).
