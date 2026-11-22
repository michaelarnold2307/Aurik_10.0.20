# Wettbewerbs-Paritäts-Register — Aurik vs. iZotope RX · Cedar · Acon

> **Mission (Zielbild, normativ):** Aurik verbessert den Wohlklang für das
> menschliche Gehör systematisch und evidenzbasiert. Die langfristige
> Wettbewerbsambition gilt jeder Disziplin; sie ist weder eine Behauptung
> bereits erreichter Gesamtparität noch ein pauschales Release-Gate. Ein
> Ergebnis auf RX-12-Niveau wird nur für die konkret geprüfte Disziplin und
> Konfiguration behauptet, wenn DRT-16 vollständig bestanden ist. Dieses
> Register ist die **Pflichtlektüre** für jede
> Programmier-KI (SOTA_HANDOVER §0 verweist hierauf) und wird maschinell
> gepinnt (`scripts/compliance/wettbewerbs_paritaet_gate.py`,
> Pre-Commit `aurik-wettbewerbs-paritaet`, fail-closed).
>
> Status-Tokens (maschinell geprüft): `PARITAET` (externe Nichtunterlegenheits-
> Abnahme gegen jede in der Zeile genannte Referenz bestanden, einschließlich
> RX 12, wenn genannt, und menschlicher §4-Sign-off liegt vor),
> `GESICHERT` (interner Aurik-Nachweis bestanden; ausdrücklich noch kein
> Wettbewerbs-Paritätsnachweis),
> `LUECKE` (bekannte Lücke mit laufendem/geplantem Schritt),
> `OFFEN` (noch ohne Messung — Rezept/Messweg fehlt oder steht aus),
> `NICHT_MESSBAR` (kein lokaler Messweg — nur Hörtest/Beschaffung).
>
> **Beschaffung autonom geklärt (2026-11-21):** Der RX-Vergleichs-Weg ist bis
> auf den minimalen menschlichen Akt vorbereitet — DRT-16-Harness verifiziert,
> Paar-Materialien liegen bereit, Steckbrief: `docs/guides/RX12_BESCHAFFUNG.md`.
> RX läuft nur auf Windows/macOS (Linux-Host) — deshalb bleibt der letzte
> Schritt extern; danach läuft Bericht + Register-Beleg automatisch.

## Realistische Ziel- und Fortschrittsstufen

- **Erste Pflicht:** Hör-Invarianten und Never-worsen bleiben harte
  Sicherheitsgrenzen. Jede priorisierte, messbare Lücke erhält einen
  reproduzierbaren internen Vorher/Nachher-Nachweis; ein neutraler Befund oder
  fehlende Evidenz wird nicht als Verbesserung ausgegeben.
- **Schrittweiser Wettbewerb:** Interne Wirksamkeit (`GESICHERT`) und externe
  Wettbewerbsparität (`PARITAET`) sind unterschiedliche Stufen. Zuerst werden
  robuste, hörbar vorteilhafte Produktionspfade je Defektklasse belegt; RX-12-
  Vergleiche folgen für jene Disziplinen, für die ein fairer Referenzlauf,
  geeigneter Holdout und Panel tatsächlich verfügbar sind. Nicht gemessene
  Klassen bleiben `OFFEN`/`LUECKE`, nicht pauschal bestanden.
- **Kein 1:1-Output-Ziel:** Aurik muss nicht dieselben Samples, Bearbeitungen
  oder Bedienabläufe wie RX 12 erzeugen. Ziel eines externen Vergleichs ist
  die nach DRT-16 belegte perzeptuelle Nichtunterlegenheit bei geschützten
  Hör-Invarianten; aus einem einzelnen Rezept, Titel oder Modul wird keine
  Gesamtparität abgeleitet.
- **Parität ist Mindestnachweis, nicht Optimierungsziel:** `PARITAET` belegt
  Nichtunterlegenheit, nicht Überlegenheit oder maximalen Wohlklang. Auriks
  Optimierungsziel bleibt pro Song die bestmögliche menschliche Hörpräferenz
  unter Hör-Invarianten, Authentizität und Never-worsen; RX 12 ist Referenz,
  nicht die Obergrenze. Eine Überlegenheits-Behauptung braucht einen separaten,
  verblindeten Präferenznachweis; `nicht_unterlegen` allein genügt nicht.
  Hörpanels kalibrieren und validieren das autonome System, sind aber keine
  manuelle Laufzeit-Voraussetzung pro Importsong.
- **Ehrliche Reichweite:** Lizenz-, Daten-, Hardware- oder Panel-Zugang sind
  dokumentierte Abhängigkeiten und keine fingierten Erfolge. Ein bestmöglicher
  sicherer Aurik-Export und ein klarer Status bleiben zulässig, wenn eine
  externe Referenz nicht verfügbar ist; die Wettbewerbsbehauptung bleibt dann
  offen.

## Disziplin-Tabelle

| Disziplin | Amtsinhaber-Referenz | Aurik-Status | Evidenz | Nächster Schritt |
| --- | --- | --- | --- | --- |
| Declick (Knackser) | RX De-click / Cedar | LUECKE | Witness-Kanal clicks Gate-bestanden (Q 1,19, nw 1,0), verdrahtet, Flag AUS; **Sinn-Beweis 2026-11-18: clicks 0/3 besser, ΔMOS −0,018 (neutral); n=10 (signal10b, 2026-11-20): 2/10 besser, ΔDefektlast-Mittel −0,012, ΔMOS-Mittel +0,005 — Kriterium (a) verfehlt** | §4-Hör-Sign-off (Proben fertig) → Flip |
| Decrackle (Knistern) | RX De-crackle | LUECKE | VinylNet synthetisch trainiert; **CC0-Vinyl-Leerrillen-Bank vorhanden seit 2026-11-17** (`data/vinyl_noise_bank/`, 6 freesound-CC0-Dateien, §9/7-frei, Manifest /2); Spitzenwachstum-Never-worsen aktiv; Hörtest gegen RX 12 offen (D-K1-4) | VinylNet-Retrain mit `--noise-bank` (GPU-Fenster) + RX-Vergleich; ggf. Witness-Kanal |
| Denoise (Breitband) | RX Spectral Denoise / Cedar DNS | LUECKE | DFN3 aktiv; **M9-Gate 2026-11-18 (V2-Kandidat, 2.000-Paar-Retrain): gate_bestanden=false, vorteile_rauscharm=0** — TF-GridNet-Pfad mit V1+V2 gescheitert; „300-Paar-Retrain“ nicht belegbar | Witness-Kopf mit musik_mos-Loss (DRT-8/10, integriert) als nächster Kandidat im GPU-Fenster |
| Declip (Clipping) | RX De-clip | LUECKE | k398-ΔSDR-Gate BESTANDEN 2026-11-17 (Träumer +0,81 dB, never-worse 3/3); Kopf + A-SPADE verdrahtet; **n≥10-clipping-Rezept (2026-11-19, Batch 1): 1/10 besser, ΔMOS −0,001 — Kriterium (a) verfehlt; DRT-8-Ganz-Titel-Prüfung (2026-11-21, dieselben 10 Titel): nie-schlechter NEIN (mos_min −0,474, defektlast_max +0,397) — Kriterium (b) verfehlt**; Legacy-A-SPADE-Guard korrigiert (2026-11-22): Kandidat muss Proxy gegen Eingang UND PCHIP-Amtsinhaber senken; Regressionstest bestanden, **kein** neuer Qualitätsnachweis; DRT-5-Q-Erstbeleg 2026-11-21: 0/10 Regionen Q-bewertbar; Hörpaket (3 Paare) vorbereitet, §4-Sign-off offen | Ursache der 0/10 Q-Bewertbarkeit ermitteln und ≥6 bewertbare Schaden/Clean-Paare belegen; danach DRT-8 mit ≥10 Titeln gegen Bestands-Pipeline erneut bestehen, identischer-Paar-Kanalsatz-A/B und §4-Hör-Sign-off (Runbook `docs/guides/DECLIP_SIGNOFF_RUNBOOK.md`; `hoer_entscheidung.py`); RX-Parität bleibt separat nach DRT-16 |
| Dehum (Netzbrummen) | RX De-hum | OFFEN | Bestands-Phase aktiv (CRITICAL-Priorität); **Sinn-Beweis 2026-11-18: hum 0/3 besser, ΔMOS −0,026 (neutral); n=10 (signal10b): 1/10 besser, ΔDefektlast-Mittel −0,002, ΔMOS-Mittel −0,029 — Kriterium (a) verfehlt** | defekt-spezifischer Kanal (Witness/DRT-Zyklus) |
| Wow/Flutter | Cedar / RX Wow&Flutter | OFFEN | Bestands-Phase aktiv (CRITICAL); **Sinn-Beweis 2026-11-18: wowflutter 0/3 besser, ΔMOS −0,024 (neutral); n=10 (signal10b): 3/10 besser, ΔDefektlast-Mittel −0,013, ΔMOS-Mittel −0,008 — Kriterium (a) verfehlt** | WF4-Kopf-Neubewertung / Witness-Kanal |
| HF-/Bandbreiten-Rekonstruktion | RX (EQ-Match/Spectral Recovery) | LUECKE | FlashSR-Finetune −4,61 dB (Gate verfehlt, nicht im Signalpfad, kein Produktionsrisiko, D-K3-94); BW-V5-Retrain **bestätigt KEIN Kandidat** (2026-11-19-Nachprüfung von `2026-10-27_kp7_hf_gate_v2.json`: HF-Gain 51,0–85,8 linear ≈ +17 dB ggü. Eingang — physikalisch unplausibel für eine treue Rekonstruktion, der numerische Gate-Schwellwert ≥1,02 ist für diese Größenordnung blind; `use_bw_v5` bleibt korrekt AUS, Kommentar in `music_model_flags.py` bereits konsistent); aktiv DSP-SBR (sicherer Produktionspfad) | (a) BW-V5-Loss um eine Natürlichkeits-/Gain-Obergrenze erweitern + Gate um eine Plausibilitäts-Prüfung (z. B. HNR/musik_mos) ergänzen, dann Re-Training (GPU-Fenster); oder (b) bereits vorhandene musik-trainierte SR-Kandidaten im Modell-Zoo (z. B. AudioSR, FlashSR-Basis ohne Finetune) gegen DSP-SBR messen, bevor neues Training angesetzt wird |
| Stereo-Imaging/Azimuth | RX Azimuth / Acon | LUECKE | Bestands-Heuristik aktiv (Stereo-Invarianten); **Stereo-Kette implementiert + 3-Schicht-Fix 2026-11-18 (§III.15):** Phasen-Vorzeichen (else-Zweig), ActiveIntervention-Belohnung, Regelkreis-Balance-Metrik — **Sinn-Beweis-Beleg 3/3 `balance_besser`** (ANiMAL −3,50→−0,15 dB, Alexander Ross +4,55→0,00 dB, **Snowmine −4,09→+0,00 dB 2026-11-20** — End-to-End-Beweis nach Never-worsen-Arbiter-Notch-Metrik-Fix, v6-Akzeptanzlauf, Beleg `docs/reports/current/2026-11-20_snowmine_stereo_fix_v2.json`); §4-Hörpaket `study_stimuli/hoerproben/stereo_sinnbeweis/`; **n≥10-Kette läuft** (Watchdog `stereo-n10`, `data/schadens_proben_stereo10`, seed 43) | Stereo-n≥10-Auswertung (Balance + MOS-Deltas); danach Hör-Vergleich |
| Dereverb (Nachhall) | RX De-reverb / Acon DeVerberate | GESICHERT | WPE-DSP aktiv; SGMSE+-Musik abgelehnt (K0-B NO-GO); **Sinn-Beweis-Rezept: `raum` (Kriterium a): 6/10 besser, ΔDefektlast-Mittel −0,346, ΔMOS-Mittel +0,639, 0 Überrestaurationen** (Belege `docs/reports/current/2026-11-18_sinn_beweis_15_rezepte.md` + `data/schadens_proben_raum10/bewertung.json`; maschinell gegen Einzeltitel geprüft) | §4-Runbook: `docs/guides/GO_NO_GO_DECISION_PROTOCOL.md`; Hör-/RX-Vergleichshörtest als Bestätigung; danach Kandidat-Bewertung |
| Pitch/Speed-Korrektur | RX / Cedar | OFFEN | Bestands-Phase aktiv, unbelegt; **pitch_drift-Rezept 2026-11-18: 0/3 besser, ΔMOS −0,028 (neutral); n=10 (signal10c): 1/10 besser, ΔDefektlast-Mittel +0,034, ΔMOS-Mittel −0,062 — Kriterium (a) verfehlt** | defekt-spezifischer Kanal (Witness/DRT-Zyklus) |
| Dynamik-Rückbau (Überkompression) | RX | OFFEN | **Sinn-Beweis 2026-11-18: kompression 0/3 besser, ΔMOS −0,023 (neutral)** | mehr Titel je Klasse |
| Phasen-/Kammfilter-Artefakte | RX Phase / Acon | OFFEN | **Sinn-Beweis 2026-11-18: kammfilter 1/3 besser, ΔMOS −0,003 (neutral); n≥10 (2026-11-19, Batch 1, 10 Titel): 3/10 besser, ΔMOS-Mittel +0,021, 0 Überrestaurationen** | defekt-spezifischer Kanal (Witness/DRT-Zyklus); danach §4 |
| Dropout-/Riss-Reparatur | RX De-click (Riss) / Cedar | OFFEN | Erkannt, konservativ behandelt (§V6 (copilot-instructions.md)); **Sinn-Beweis 2026-11-18: dropout 0/3 besser, ΔMOS −0,024 (neutral); n=10 (signal10c): 0/10 besser, ΔDefektlast-Mittel +0,041, ΔMOS-Mittel −0,024 — Kriterium (a) verfehlt** | defekt-spezifischer Kanal (Witness/DRT-Zyklus) |
| Master-Finish/Lautheit | RX Loudness Control | OFFEN | Glue-Stage aktiv, Hör-Invarianten geschützt; **Sinn-Beweis 2026-11-18: lautheit 0/3 besser, ΔMOS −0,024 (neutral); n=10 (signal10c): 2/10 besser, ΔDefektlast-Mittel +0,025, ΔMOS-Mittel −0,016 — Kriterium (a) verfehlt** | defekt-spezifischer Kanal + Loudness-Gate-Evidenz |
| Bedienung/Echtzeit | RX (interaktiv) | LUECKE | 32× RT; **Segment-Vorschau implementiert 2026-11-19** (6-s-Ausschnitt über den Produktions-Pfad → A/B-Wiedergabe, `Aurik10/ui/preview_restorer.py` + Bridge `restore_segment_preview`); Scanner 14,9 s/min (Norm 16 ✓); **Architektur-Entscheidung getroffen (2026-11-19):** echtes sample-genaues Low-Latency-Streaming WIRD NICHT verfolgt — unvereinbar mit §G1/§G2 (viele Phasen brauchen Song-globalen Kontext, Section-Envelopes, Ganz-Titel-Analytik), kein RX-äquivalenter Interaktions-Modus (RX editiert Selektionen, Aurik verarbeitet autonom den ganzen Song). Die bereits vorhandene **progressive Snapshot-Live-Vorschau (§v10.999 Live-Mithören, `_play_restored_or_preview`, Ring-Puffer `_poll_audio_ring`, 30 Hz)** ist die funktionale Antwort: während eines laufenden Batches kann der zuletzt fertige Phasen-Zwischenstand jederzeit angehört werden (Toggle, Positionsgedächtnis, Scrubber). **Bugfix:** `_panel_play()` (Transport-▶-Knopf) rief zwei nicht existierende Methoden auf (`_play_restored`/`_play_live_preview`, AttributeError bei Bedienung während laufendem Batch, §V6-Verstoß durch fehlende Fehlerbehandlung) — auf die bereits korrekte `_play_restored_or_preview()` umgestellt; **Fix verifiziert 2026-11-21** (Vertragstest `tests/unit/test_modern_window_panel_play_fix.py` grün) | 4-s-Fast-Ziel (Hör-Freigabe Batch-Pfad §2.46h) als letzter offener Punkt dieser Zeile |

## RX-12-Paritätsvertrag (akustische Ergebnisparität)

Dieser Vertrag konkretisiert Spec 26 DRT-16. Er gilt für die akustische
Restaurierungsqualität, nicht für eine identische Bedienoberfläche oder
interaktive Editier-Workflows. `Bedienung/Echtzeit` bleibt eine separate
Produktdisziplin und kann keinen akustischen Paritätsstatus begründen.

- **Vergleichsgegenstand:** gleicher, bit-identischer Schadens-Input für Aurik
  und die exakt benannte RX-12-Version samt Build, Modul und dokumentierten
  Einstellungen. Gemessen wird der reale Aurik-Produktionspfad mit den im
  Beleg ausgewiesenen aktiven Flags; Simulationen, interne Aurik-Baselines oder
  Ergebnisse eines nicht erreichten Kanals sind kein RX-Vergleich.
- **Stichprobe:** mindestens 10 unabhängige Holdout-Titel je Disziplin als
  objektiver Ganz-Titel-Screen. Quellen-/Titel-Leakage zwischen Training,
  Kalibrierung und Holdout ist ausgeschlossen und durch Manifest-Hashes
  nachvollziehbar. Der Screen ersetzt nicht das menschliche Panel.
- **Nichtunterlegenheit:** pro Titel gepaarte Deltas Aurik minus RX. Für
  `musik_mos` muss die untere Grenze des zweiseitigen Student-t-95-%-
  Konfidenzintervalls (df = `n_titles - 1`) mindestens
  `-JND_TOLERANZ` betragen; für `defektlast` muss die obere Grenze höchstens
  `+JND_TOLERANZ` betragen. Das Paritäts-Gate berechnet beide Intervalle aus
  den Einzeltitel-Deltas neu; die Beleg-CIs müssen bis auf 1e-6 übereinstimmen.
  Jede Metrik-Differenz ist zusätzlich direkt im gehashten Quellenpaar
  hinterlegt; die Arrays müssen Reihenfolge und Werte der Paarzeilen abbilden.
  Ein behauptetes `noninferiority_passed` allein oder entkoppelte Deltas genügt
  nicht. Beide Metriken und Never-worsen-/Hör-Invarianten müssen bestehen. JND
  stammt ausschließlich aus `wohlklang_zeuge.JND_TOLERANZ`; Mittelwerte allein
  reichen nicht.
- **Hör-Sign-off:** zusätzlich verblindeter, randomisierter, pegelgematchter
  MUSHRA-/§4-Vergleich mit mindestens 30 Hörer:innen und dem expliziten
  Verdikt `nicht_unterlegen`. Metriken bleiben Zeugen gemäß Hörordnung; ein bestandener
  Zahlenvergleich ersetzt weder die Hör-Invarianten noch das Panel.
- **Beleg:** je Disziplin und je genannter Referenz ein maschinenlesbarer
  Bericht `aurik.rx12_parity/1` unter `docs/reports/current/`. Er enthält
  Referenzprodukt/-version/-build/-Modul/-Einstellungen, Aurik-Version und
  Commit, aktiven Produktionspfad und Flags, Holdout-Manifest-Hash und
  Titelzahl, gepaarte Metrik-Deltas samt 95-%-Intervallen, JND-Quelle,
  Panel-Parameter/-Verdikt und Sign-off. Der Registereintrag verlinkt den
  Bericht; fehlende oder nicht messbare Felder sind kein Bestehen.
- **Statusgrenze:** `GESICHERT` beschreibt nur den internen Aurik-Beleg aus
  Regel 5. `PARITAET` ist erst nach vollständigem DRT-16-Beleg zulässig und
  gilt ausschließlich für die konkret belegte Disziplin und Referenz.
  Sind mehrere Referenzen in einer Tabellenzeile genannt, müssen alle
  jeweils eigene Belege bestehen. `NICHT_MESSBAR` zählt nie als bestanden.

## Harte Regeln

1. **Kein Status-Downgrade ohne Beleg:** Ein Statuswechsel braucht Mess-Evidenz
   (Sinn-Beweis/Gate/Hörtest) — Prosa genügt nicht (WP-0-Muster).
2. **GESICHERT ⇒ §4-Pfad:** Jeder `GESICHERT`-Eintrag muss im Hör-Runbook
   verlinkt sein; ohne Runbook-Verweis blockt das Gate.
3. **LUECKE ⇒ Schritt-Pflicht:** Jeder `LUECKE`-Eintrag trägt einen konkreten
   nächsten Schritt (Trainings-/Mess-/Beschaffungsauftrag) — sonst blockt das Gate.
4. **Dieses Register ist Pflichtlektüre:** SOTA_HANDOVER §0, AGENTS.md und der
   Pre-Commit-Gate verweisen hierauf — jede Programmier-KI sieht den
   Paritäts-Auftrag bei Session-Start und bei jedem Commit.
5. **Interner GESICHERT-Nachweis (quantifiziert, 2026-11-19):** Eine
   OFFEN/LUECKE-Zeile erreicht `GESICHERT` erst, wenn EINES dieser Kriterien
   gegen Auriks Bestands-Pipeline messbar belegt ist:
   (a) **Ganz-Titel-Sinn-Beweis n≥10:** besser-Rate ≥ 5/10 UND ΔMOS-Mittel ≥ +0,05
   UND Δdefektlast-Mittel ≤ 0 UND 0 Überrestaurationen; oder
   (b) **defekt-spezifischer Kanal:** Regionen-Gate bestanden (q_median ≥ 0,95,
   q-Anteil≥0,5 ≥ 0,96, never-worsen ≥ 0,96 — DRT-Schwellen) UND DRT-8-Ganz-Titel-
   Beweis (musik_mos nie < −JND, defektlast nie > +JND auf allen n≥10-Titeln).
   Das ist ein interner Kandidaten-Nachweis, ausdrücklich kein RX-12-Beleg.
   Das Gate prüft den Nachweis fail-closed aus direkt verlinkten JSON-Berichten:
   Kriterium (a) verlangt `Sinn-Beweis-Rezept: <rezept>` und
   `aurik.schadensbewertung/1`; Kriterium (b) verlangt
   `DRT-5-Klasse: <klasse>` plus `DRT-8-Rezept: <rezept>`, den
   `aurik.witness_kopf_gate/2`-Bericht mit Q-/Never-worsen-Rohwerten und den
   passenden Ganz-Titel-Bericht. Das Gate rechnet die Aggregate aus den Rohwerten
   nach. Für Kriterium (b) werden mindestens sechs bewertbare Schaden-/Clean-
   Paare im Regionenbericht (übersprungene Paare zählen nicht) und mindestens
   zehn eindeutige Titel im DRT-8-Bericht verlangt; die Regionen-Anzahl ersetzt
   nicht die Titelzahl des Ganz-Titel-Belegs.
6. **Kanalsatz-A/B-Pflicht (DRT-12, 2026-11-19):** Vor dem §4-Sign-off eines
   Witness-Kanals MUSS der Ganz-Titel-A/B auf IDENTISCHEN Paaren belegt sein:
   Kanalsatz AN vs. Bestand AUS (gleiche Dateien, gleiche Metriken,
   `scripts/messe_kanalsatz_ab.py`). Nur Kanäle, die den A/B gewinnen, kommen
   in den §4-Batch — das ist der Gegenbeweis zum „0/12“-Muster.
7. **Kombinierter §4-Batch:** Kanäle, die Regel 6 erfüllen, werden als EIN
   Kanalsatz gehört (eine Sitzung, ein Verdikt über die Menge) statt einzeln —
   der menschliche Aufwand skaliert nicht mit der Kanalzahl.
8. **RX-/Wettbewerbs-Parität:** `PARITAET` verlangt den externen, gepaarten
   Referenzvergleich nach DRT-16 für jede Referenz in der Zeile. Regel 5,
   ein interner Score, ein Regionen-Gate oder ein §4-Test ohne RX-Referenz
   darf diesen Status nicht begründen.
9. **Akustische Produktreife als Langfristziel:** `PARITAET` ist je Zeile ein
   erreichbares, aber nicht pauschal behauptetes Ziel. Das Produkt darf
   Gesamtparität erst dann beanspruchen, wenn alle dafür ausdrücklich
   beanspruchten akustischen Disziplinen und Referenzen DRT-16 bestehen.
   `GESICHERT` belegt internen Fortschritt, nicht RX-Parität; offene oder
   mangels Referenz nicht messbare Disziplinen bleiben sichtbar offen. Die
   separate Zeile `Bedienung/Echtzeit` ist kein Bestandteil der akustischen
   RX-12-Abnahme.
