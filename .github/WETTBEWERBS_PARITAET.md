# Wettbewerbs-Paritäts-Register — Aurik vs. iZotope RX · Cedar · Acon

> **Mission (2026-11-17, normativ):** Aurik muss für das menschliche Gehör in
> **jeder Disziplin** mindestens die Stufe der Amtsinhaber iZotope RX, Cedar
> und Acon erreichen. Dieses Register ist die **Pflichtlektüre** für jede
> Programmier-KI (SOTA_HANDOVER §0 verweist hierauf) und wird maschinell
> gepinnt (`scripts/compliance/wettbewerbs_paritaet_gate.py`,
> Pre-Commit `aurik-wettbewerbs-paritaet`, fail-closed).
>
> Status-Tokens (maschinell geprüft): `PARITAET` (belegt auf Augenhöhe),
> `GESICHERT` (objektiv belegt, nur §4-Hör-Sign-off ausstehend),
> `LUECKE` (bekannte Lücke mit laufendem/geplantem Schritt),
> `OFFEN` (noch ohne Messung — Rezept/Messweg fehlt oder steht aus),
> `NICHT_MESSBAR` (kein lokaler Messweg — nur Hörtest/Beschaffung).

## Disziplin-Tabelle

| Disziplin | Amtsinhaber-Referenz | Aurik-Status | Evidenz | Nächster Schritt |
| --- | --- | --- | --- | --- |
| Declick (Knackser) | RX De-click / Cedar | LUECKE | Witness-Kanal clicks Gate-bestanden (Q 1,19, nw 1,0), verdrahtet, Flag AUS; **Sinn-Beweis 2026-11-18: clicks 0/3 besser, ΔMOS −0,018 (neutral)** | §4-Hör-Sign-off (Proben fertig) → Flip |
| Decrackle (Knistern) | RX De-crackle | LUECKE | VinylNet synthetisch trainiert; **CC0-Vinyl-Leerrillen-Bank vorhanden seit 2026-11-17** (`data/vinyl_noise_bank/`, 6 freesound-CC0-Dateien, §9/7-frei, Manifest /2); Spitzenwachstum-Never-worsen aktiv; Hörtest gegen RX 12 offen (D-K1-4) | VinylNet-Retrain mit `--noise-bank` (GPU-Fenster) + RX-Vergleich; ggf. Witness-Kanal |
| Denoise (Breitband) | RX Spectral Denoise / Cedar DNS | LUECKE | DFN3 aktiv; **M9-Gate 2026-11-18 (V2-Kandidat, 2.000-Paar-Retrain): gate_bestanden=false, vorteile_rauscharm=0** — TF-GridNet-Pfad mit V1+V2 gescheitert; „300-Paar-Retrain“ nicht belegbar | Witness-Kopf mit musik_mos-Loss (DRT-8/10, integriert) als nächster Kandidat im GPU-Fenster |
| Declip (Clipping) | RX De-clip | GESICHERT | k398-ΔSDR-Gate BESTANDEN 2026-11-17 (Träumer +0,81 dB, never-worse 3/3); Kopf + A-SPADE verdrahtet; **Hörpaket NEU GERENDERT 2026-11-18** (CPU-only via AURIK_FORCE_CPU=1, 3 Paare, Schlüssel getrennt, `study_stimuli/hoerproben/declip_produktionspfad/`); **n≥10-clipping-Rezept (2026-11-19, Batch 1): 1/10 besser, ΔMOS −0,001 (neutral) — mildes tanh-Rezept, keine Statuswirkung** | §4-Hör-Sign-off (Runbook `docs/guides/DECLIP_SIGNOFF_RUNBOOK.md`; `hoer_entscheidung.py`) |
| Dehum (Netzbrummen) | RX De-hum | OFFEN | Bestands-Phase aktiv (CRITICAL-Priorität); **Sinn-Beweis 2026-11-18: hum 0/3 besser, ΔMOS −0,026 (neutral)** | mehr Titel je Klasse (n≥10) |
| Wow/Flutter | Cedar / RX Wow&Flutter | OFFEN | Bestands-Phase aktiv (CRITICAL); **Sinn-Beweis 2026-11-18: wowflutter 0/3 besser, ΔMOS −0,024 (neutral)** | mehr Titel je Klasse; danach WF4-Kopf-Neubewertung |
| HF-/Bandbreiten-Rekonstruktion | RX (EQ-Match/Spectral Recovery) | LUECKE | FlashSR-Finetune −4,61 dB (Gate verfehlt, nicht im Signalpfad, kein Produktionsrisiko, D-K3-94); BW-V5-Retrain **bestätigt KEIN Kandidat** (2026-11-19-Nachprüfung von `2026-10-27_kp7_hf_gate_v2.json`: HF-Gain 51,0–85,8 linear ≈ +17 dB ggü. Eingang — physikalisch unplausibel für eine treue Rekonstruktion, der numerische Gate-Schwellwert ≥1,02 ist für diese Größenordnung blind; `use_bw_v5` bleibt korrekt AUS, Kommentar in `music_model_flags.py` bereits konsistent); aktiv DSP-SBR (sicherer Produktionspfad) | (a) BW-V5-Loss um eine Natürlichkeits-/Gain-Obergrenze erweitern + Gate um eine Plausibilitäts-Prüfung (z. B. HNR/musik_mos) ergänzen, dann Re-Training (GPU-Fenster); oder (b) bereits vorhandene musik-trainierte SR-Kandidaten im Modell-Zoo (z. B. AudioSR, FlashSR-Basis ohne Finetune) gegen DSP-SBR messen, bevor neues Training angesetzt wird |
| Stereo-Imaging/Azimuth | RX Azimuth / Acon | LUECKE | Bestands-Heuristik aktiv (Stereo-Invarianten); **Stereo-Kette implementiert + 3-Schicht-Fix 2026-11-18 (§III.15):** Phasen-Vorzeichen (else-Zweig), ActiveIntervention-Belohnung, Regelkreis-Balance-Metrik — **Sinn-Beweis-Beleg 2/3 `balance_besser`** (ANiMAL −3,50→−0,15 dB, Alexander Ross +4,55→0,00 dB; Snowmine noch offen); §4-Hörpaket `study_stimuli/hoerproben/stereo_sinnbeweis/` (automatisch nach der Runde) | Snowmine-Ursache + Stereo-n≥10; danach Hör-Vergleich |
| Dereverb (Nachhall) | RX De-reverb / Acon DeVerberate | GESICHERT | WPE-DSP aktiv; SGMSE+-Musik abgelehnt (K0-B NO-GO); **Sinn-Beweis 2026-11-18: 3/3 besser** (ΔDefektlast −0,395, ΔMOS +0,039) → **n=10-Bestätigung: 6/10 besser, ΔDefektlast-Mittel −0,346, ΔMOS-Mittel +0,639, 0 MOS-Verschlechterungen, 0 Überrestaurationen** (Belege `docs/reports/current/2026-11-18_sinn_beweis_15_rezepte.md` + `data/schadens_proben_raum10/bewertung.json`) | Hör-/RX-Vergleichshörtest als Bestätigung; danach Kandidat-Bewertung |
| Pitch/Speed-Korrektur | RX / Cedar | OFFEN | Bestands-Phase aktiv, unbelegt; **pitch_drift-Rezept 2026-11-18: 0/3 besser, ΔMOS −0,028 (neutral)** | mehr Titel je Klasse |
| Dynamik-Rückbau (Überkompression) | RX | OFFEN | **Sinn-Beweis 2026-11-18: kompression 0/3 besser, ΔMOS −0,023 (neutral)** | mehr Titel je Klasse |
| Phasen-/Kammfilter-Artefakte | RX Phase / Acon | OFFEN | **Sinn-Beweis 2026-11-18: kammfilter 1/3 besser, ΔMOS −0,003 (neutral); n≥10 (2026-11-19, Batch 1, 10 Titel): 3/10 besser, ΔMOS-Mittel +0,021, 0 Überrestaurationen** | defekt-spezifischer Kanal (Witness/DRT-Zyklus); danach §4 |
| Dropout-/Riss-Reparatur | RX De-click (Riss) / Cedar | OFFEN | Erkannt, konservativ behandelt (§V6 (copilot-instructions.md)); **Sinn-Beweis 2026-11-18: dropout 0/3 besser, ΔMOS −0,024 (neutral)** | mehr Titel je Klasse; ggf. Witness-Kanal |
| Master-Finish/Lautheit | RX Loudness Control | OFFEN | Glue-Stage aktiv, Hör-Invarianten geschützt; **Sinn-Beweis 2026-11-18: lautheit 0/3 besser, ΔMOS −0,024 (neutral)** | mehr Titel je Klasse + Loudness-Gate-Evidenz |
| Bedienung/Echtzeit | RX (interaktiv) | LUECKE | 32× RT; **Segment-Vorschau implementiert 2026-11-19** (6-s-Ausschnitt über den Produktions-Pfad → A/B-Wiedergabe, `Aurik10/ui/preview_restorer.py` + Bridge `restore_segment_preview`); Scanner 14,9 s/min (Norm 16 ✓); **Architektur-Entscheidung getroffen (2026-11-19):** echtes sample-genaues Low-Latency-Streaming WIRD NICHT verfolgt — unvereinbar mit §G1/§G2 (viele Phasen brauchen Song-globalen Kontext, Section-Envelopes, Ganz-Titel-Analytik), kein RX-äquivalenter Interaktions-Modus (RX editiert Selektionen, Aurik verarbeitet autonom den ganzen Song). Die bereits vorhandene **progressive Snapshot-Live-Vorschau (§v10.999 Live-Mithören, `_play_restored_or_preview`, Ring-Puffer `_poll_audio_ring`, 30 Hz)** ist die funktionale Antwort: während eines laufenden Batches kann der zuletzt fertige Phasen-Zwischenstand jederzeit angehört werden (Toggle, Positionsgedächtnis, Scrubber). **Bugfix:** `_panel_play()` (Transport-▶-Knopf) rief zwei nicht existierende Methoden auf (`_play_restored`/`_play_live_preview`, AttributeError bei Bedienung während laufendem Batch, §V6-Verstoß durch fehlende Fehlerbehandlung) — auf die bereits korrekte `_play_restored_or_preview()` umgestellt | Fix verifizieren (GUI-Smoke, sobald Hostlast sicher ist); danach 4-s-Fast-Ziel (Hör-Freigabe Batch-Pfad §2.46h) als letzter offener Punkt dieser Zeile |

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
5. **RX-Paritäts-Norm (quantifiziert, 2026-11-19):** Eine OFFEN/LUECKE-Zeile
   erreicht GESICHERT erst, wenn EINES dieser Kriterien messbar belegt ist:
   (a) **Ganz-Titel-Sinn-Beweis n≥10:** besser-Rate ≥ 5/10 UND ΔMOS-Mittel ≥ +0,05
   UND Δdefektlast-Mittel ≤ 0 UND 0 Überrestaurationen; oder
   (b) **defekt-spezifischer Kanal:** Regionen-Gate bestanden (q_median ≥ 0,95,
   q-Anteil≥0,5 ≥ 0,96, never-worsen ≥ 0,96 — DRT-Schwellen) UND DRT-8-Ganz-Titel-
   Beweis (musik_mos nie < −JND, defektlast nie > +JND auf allen n≥10-Titeln).
   In beiden Fällen zusätzlich §4-Sign-off (Hörordnung bleibt unangetastet).
6. **Kanalsatz-A/B-Pflicht (DRT-12, 2026-11-19):** Vor dem §4-Sign-off eines
   Witness-Kanals MUSS der Ganz-Titel-A/B auf IDENTISCHEN Paaren belegt sein:
   Kanalsatz AN vs. Bestand AUS (gleiche Dateien, gleiche Metriken,
   `scripts/messe_kanalsatz_ab.py`). Nur Kanäle, die den A/B gewinnen, kommen
   in den §4-Batch — das ist der Gegenbeweis zum „0/12“-Muster.
7. **Kombinierter §4-Batch:** Kanäle, die Regel 6 erfüllen, werden als EIN
   Kanalsatz gehört (eine Sitzung, ein Verdikt über die Menge) statt einzeln —
   der menschliche Aufwand skaliert nicht mit der Kanalzahl.
