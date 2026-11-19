# Aurik 10 — Spec 26: Defekt-Reparatur-Trainingsvertrag (DRT)

> **Zweck:** Normative Leitplanken für das Training defekt-spezifischer
> Reparatur-Köpfe (Witness-Kopf-Verfahren, D-K3-95-Folge). Verhindert
> unnötige Trainingsläufe, indem jede Verlustrunde der Vergangenheit
> (M9-V1/V2 2026-11-14, Witness-Kopf-V1 2026-11-15, Schutz-Arm-Versager
> 2026-11-15, Produktions-Paritäts-No-Op 2026-11-15) strukturell
> ausgeschlossen wird. Maschinell erzwungen durch den Pre-Commit-Hook
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

## DRT-12 — Kontroll-Bedingung (Ablation ohne Kanal) im Ganz-Titel-Beleg

Der DRT-8-Ganz-Titel-Beleg MUSS die Kontroll-Bedingung enthalten: dieselben
Paare einmal MIT und einmal OHNE den Kandidaten (Bestands-Pipeline), sonst ist
„besser/schlechter" nicht zurechenbar. Ablations-Befund 2026-11-15: Der
12-Paare-Lauf MIT den 10 Kanälen zeigte 0/12 — erst die Kontrolle OHNE Kanäle
(4/4 + 2/4 besser) offenbarte, dass die Kanäle die Bestands-Reparatur
blockieren. Ohne Kontrolle wäre die Verschlechterung unsichtbar geblieben.
Kontroll-Delta-Pflicht: Kandidat aktiviert nur, wenn MIT-Kanal ≥ OHNE-Kanal
(kein Paar darf durch den Kanal netto schlechter werden als die Bestands-Pipeline).

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

Kein stiller Flip: Teil-Rollback ist ein deklarierter, belegter Eingriff
(§V7 (copilot-instructions.md)); der M1-Fallback bleibt unverändert, bis die
Attribution nachweislich greift. Stufe 1 (2026-11-17): Snapshot + Attribution

- Teil-Rollback im Einzel-Pfad; Options-/Chunked-Pfad nutzen den M1-Fallback
(Stufe 2 folgt im nächsten freien GPU-Fenster).

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

## DRT-12 — Kanalsatz-A/B-Pflicht vor dem §4-Sign-off (2026-11-19)

Befund: Die Ganz-Titel-Sinn-Beweise (2026-11-15: 0/12 besser; 2026-11-18:
6/45) wurden mit **deaktivierten** Witness-Kanälen gemessen (alle Flags
Default AUS) — sie belegen den Bestand, nicht den Kandidaten-Zustand. Ein
§4-Sign-off ohne A/B auf identischen Paaren würde Kanäle freigeben, deren
Ganz-Titel-Wirkung nie gemessen wurde („0/12“-Muster).

1. **A/B-Pflicht:** Vor dem §4-Sign-off eines Witness-Kanals MUSS der
   Ganz-Titel-A/B auf IDENTISCHEN Paaren belegt sein — Kanalsatz AN vs.
   Bestand AUS (gleiche Schadensdateien, gleiche Metriken defektlast/
   musik_mos, deterministische Auswahl Seed 42). Werkzeug:
   `scripts/messe_kanalsatz_ab.py` (mess-only: setzt die Kanal-Flags
   prozess-lokal, schreibt KEINE Flag-Datei, kein Produktions-Flip).
2. **Aufnahme-Kriterium:** Nur Kanäle, deren Kanalsatz-Konfiguration den
   A/B auf den zugehörigen Rezept-Paaren gewinnt (nie schlechter als der
   Bestand um mehr als JND UND mindestens ein Paar messbar besser), kommen
   in den kombinierten §4-Batch.
3. **Kombinierter §4-Batch:** Die aufgenommenen Kanäle werden als EIN Satz
   gehört (eine Sitzung, ein Verdikt über die Menge) — der menschliche
   Aufwand skaliert nicht mit der Kanalzahl.
4. **Kein Schaden am Bestand:** Der A/B-Vergleich ist mess-only; die
   Produktions-Defaults bleiben AUS, bis der §4-Batch bucht (§G1
   (copilot-instructions.md)-Bit-Identität).
