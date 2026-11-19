# Offene Punkte zur vollen Produktreife auf uneingeschränktem iZotope-RX12-Niveau — Stand 2026-11-19 (abends)

> Fortschreibung der gleichnamigen Tagesübersicht. Quellen: `.github/WETTBEWERBS_PARITAET.md`
> (maschinell gepinnt, inkl. der neuen Harten Regeln 5–7), Spec 26 (DRT-1…15 inkl. neuem
> DRT-12), `docs/SOTA_HANDOVER.md` Punkt 37, Mess-Belege B1 (30 Paare) und die laufenden
> Ketten (signal10-Resume, Crackle, VinylNet, Kanalsatz-A/B).

## Definition (User-Vorgabe 2026-11-19, unverändert gültig)

**Produktreife erreicht ⇔ iZotope-RX12-Niveau ist in den relevanten Bereichen
mindestens erreicht.** Operational: **alle 15 Disziplin-Zeilen auf
`GESICHERT`/`PARITAET`** — 0 × OFFEN, 0 × LUECKE. **Heute: 2/15.**

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

**Muster der Lücke:** Die Bestands-Pipeline ist „neutral, nicht schlechter“
(0 Überrestaurationen über alle Messungen) — der Abstand zu RX12 entsteht dort,
wo die **defekt-spezifischen Kanäle noch nicht gemessen/aktiviert** sind. Der
DRT-12-Kanalsatz-A/B ist genau dafür gebaut.

## B. Maschinen-Arbeit in Flug (autonom, detached — überlebt Laufzeit-Neustarts)

| Kette | Stand | liefert für Zeile(n) |
|---|---|---|
| signal10-Resume (B2→B3) | B2 clicks/hum/wowflutter, Resume ab 13/60 | Declick, Dehum, Wow (n≥10) |
| Crackle-Kette | Titel 3/6 → Paare → Manifest | Decrackle (Daten) |
| VinylNet-Retrain | GPU, Ep. 15+/100, `--noise-bank` CC0 | Decrackle (Modell) |
| Kanalsatz-A/B-Smoke | 4 Paare, nice 19 | DRT-12-Erstbeweis (alle Kanal-Zeilen) |

## C. Der kritische Pfad: kombinierter §4-Batch (menschlich)

12 Kanäle objektiv Gate-bestanden (9 Alt + groove_echo-mos + dynamic_compression_excess
+ dropout_head_contact), Pakete fertig; dazu Declip, Stereo, BW-V5, material_crnn,
MUSHRA n≥30, §9/7-Vinyl-Lizenz. Nach DRT-12 wird der Kanalsatz als **ein** Verdikt
gehört (Reihenfolge: `docs/guides/HOERBATCH_DECLIP_MUSHRA_W1.md`).

## D. Daten-/Lizenz-Lücken (folgen §9/7)

Vinyl↔Reissue ≥30, Echt-Träger, Genre-Labels, MUSHRA n≥30; lossless-Upgrade der
CC0-Leerrillen (freesound-Originale, IDs im Manifest).

## E. System-Hygiene

UAT-Neulauf (trainingsfreies Fenster; `test_daily_gate_stored_recently` rot bis dahin);
W-Stufen 0/13 aktiv (Bremser ohne Beschleuniger — der Witness/DRT-Zyklus ist der neue
Beschleuniger); cantus bleibt `status=training` (Gate ehrlich nicht bestanden).

## F. Reihenfolge bis „14+1 GESICHERT“

1. Kanalsatz-A/B-Smoke ernten → bei Gewinn voller 45-Paar-A/B → kombinierter §4-Batch vorbereiten.
2. B2/B3-Bewertungen ernten → OFFEN-Zeilen mit n≥10 belegen (Kriterium (a) prüfen).
3. Crackle-Kette → crackle trainieren/gaten → Decrackle-Richtung GESICHERT.
4. VinylNet fertig → Never-worsen-Gate + RX-12-Vergleich → Decrackle.
5. Witness-Runde 2 (dropouts/dropout_oxide/wow/hum/tape_splice) im GPU-Fenster nach VinylNet.
6. HF-Residual-Entscheidung + Echtzeit-Preview-Architektur-Entscheidung (einzige Punkte,
   die eine Feature-Entscheidung brauchen).
7. **Fertig-Kriterium:** alle 15 Zeilen ≥ GESICHERT — 0 OFFEN, 0 LUECKE; der letzte
   Schritt je Zeile ist der §4-Sign-off.
