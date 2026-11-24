# Beschleunigungsstrategie zur maximalen Produktreife — 2026-11-24

Ziel: maximale Produktreife **deutlich schneller**, ohne Abkürzungen — und mit
**steigender** Qualität. Jeder Hebel ist unten mit Beleg verifiziert.

## 1. Diagnose — was bisher bremst

| # | Bremse | Beleg |
| --- | --- | --- |
| D1 | Witness-Kopf-Trainings laufen auf CPU (nice 15/19) bei ~3,5 h je 30 Epochen, während die RX 7900 XTX **frei** ist (6 % Last, 1/24 GB VRAM belegt, 2026-11-24 01:55 gemessen) | `rocm-smi`, Laufzeit `nor_tape_head_level_dip.log` |
| D2 | Dropout-EXP STEP1 erzeugt je Song einen kompletten Produktions-Restaurations-Pass (~1,5–2 h × 7 ≈ 12 h) — **dessen Ausgabe (`_restauriert.wav`) kein Kettenschritt konsumiert** (STEP2 extrahiert Regionen aus Schaden↔Clean; STEP3–6 nutzen nur Manifest/Regionen) | `kette_dropout_drt8.sh`; `erstelle_schadens_proben.py` Zeile 446 |
| D3 | Messungen laufen, obwohl die Messmaschine selbst nicht deterministisch ist (D-K3-GoNoGo: Go/No-Go-Flip ±0,36) — heutige A/B-Deltas sind teils wertlos ⇒ Gefahr doppelter Messkampagnen | Beleg `2026-11-23_determinismus_go_nogo_befund.json` |
| D4 | Einzelschuss-Rezepte statt Kalibrierungs-Sweeps: bei 3,5 h/Training war ein λ-nor-/Trigger-Sweep unbezahlbar — dadurch landet man „im Fenster“ statt nahe q=1,0 | Trainingsprotokolle der Nacht |
| D5 | Zwei A/B-Kampagnen nacheinander geplant (11er jetzt, 14er später) — die erste wird durch den finalen Satz obsolet | Fahrplan-Stand vor dieser Strategie |

## 2. Hebel (verifiziert)

- **H1 — GPU für alle Witness-Kopf-Trainings.** `train_witness_kopf.py` hat `--device`
  (Default cpu, keine choices-Einschränkung) und bewegt Modelle **und** Tensoren
  via `.to(device)` (Zeilen 394–553) — `cuda` funktioniert auf dem ROCm-Torch des
  Repos. Erwartung: 30 Epochen ≈ 10–20 min statt 3,5 h. **Umsetzung läuft:**
  jitter-v2 und STEP5 der Dropout-Kette trainieren mit `--device cuda`.
- **H2 — STEP1 ohne Restaurations-Pass.** Das existierende Flag `--nur-schaden`
  („Nur beschädigen, KEINE Restauration (sofort)“, Zeile 368) erzeugt nur die
  Schaden-WAVs — exakt das, was die Kette braucht. STEP1: ~12 h → Minuten.
  **Umsetzung läuft:** Kette v6 mit `--nur-schaden`.
- **H3 — Determinisums-Fix (D-K3-GoNoGo) als SOFORT-Ingenieur-Aufgabe, parallel
  zu den Trainings.** Messungen werden erst wieder gestartet, wenn die Pipeline
  auf identischem Input identisch entscheidet. Das macht jede Messung **einmal
  richtig statt mehrfach unsicher** — Zeitgewinn und Beweiskraft-Gewinn zugleich.
  Ansatzpunkte: async pre_Analyse/PLM-Konsum deterministisch in den Phasenplan;
  Grenzwert-Verhalten des §0h-VETO dokumentieren und stabilisieren (KEINE
  Schwellenänderung).
- **H4 — Kalibrierungs-Sweeps statt Einzelschüsse.** Mit GPU-Trainings im
  Minutenbereich werden λ-nor ∈ {0, 0.5, 1.0, 2.0} × Q-Ziel {0.9, 1.0} ×
  Trigger-Varianten je Klasse messbar. Auswahlkriterium bleibt das unveränderte
  Gate (q≈1,0, q-Anteil, nw, mos) — es wird nur **mehr Evidenz pro Zeit**
  erzeugt. Qualität steigt, weil die Kanäle näher an die 1,0-Kalibrierung
  rücken statt „irgendwo im Fenster“ zu landen.
- **H5 — Keine redundanten Messläufe.** Der pausierte 11er-A/B wird NICHT
  fortgesetzt. Es gibt genau **einen** gepaarten DRT-12-Lauf (Bestand+Kanalsatz
  in einer Session) auf dem **finalen** Satz — nach Determinisums-Fix und
  finalen Gates. Das spart eine komplette 25-h-Kampagne und liefert die
  belastbarere Evidenz.
- **H6 — (nächster Schritt) Detektor-Trainer GPU-fähig machen.**
  `train_witness_detektor.py` ist bewusst CPU-only verdrahtet („die GPU gehört
  den großen Ketten“) — diese Begründung ist obsolet, die GPU ist frei. Die
  drei anstehenden Detektor-Trainings (STEP4, FU2, post_exp) kosten zusammen
  ~6–9 h CPU; auf GPU ≈ 20 min. Umsetzung: `--device`-Plumbing + Tensoren +
  Vertragstests, CPU bleibt Default. Ist der Aufwand unerwartet groß, bleibt
  CPU (kein Qualitätsverlust, nur weniger Zeitgewinn).

## 3. Warum schneller hier Qualität STEIGERT (keine Abkürzung)

- Kein Gate, keine Schwelle, kein Flag wird verändert; alle Kanäle bleiben
  Default-AUS bis §4 + Hör-Sign-off (Spez-26-DRT, Hörordnung unangetastet).
- Gleiche Daten, gleiche Rezepte, gleiche Seeds — die Artefakte sind identisch,
  nur schneller erzeugt (H2) bzw. auf anderem Gerät trainiert (H1; die CPU-Gates
  bleiben die Schiedsrichter über das Ergebnis).
- Deterministische Messung (H3) hat **höhere Beweiskraft** als die heutigen
  kontaminierten Deltas — die Qualität der EVIDENZ steigt.
- Sweeps (H4) verbessern die Kalibrierung Richtung q≈1,0 — das ist messbar
  **mehr Wohlklang-Treue**, nicht nur Geschwindigkeit.

## 4. Neue Zeitachse (ehrliche Schätzungen)

| Schritt | Vorher | Nachher |
| --- | --- | --- |
| Dropout-EXP STEP1 (Daten) | ~12 h | Minuten (H2, läuft) |
| Witness-Kopf-Trainings je Klasse | ~3,5 h | ~10–20 min (H1, läuft) |
| Detektor-Trainings (STEP4/FU2/post_exp) | ~2–3 h je Lauf | ~20 min nach H6 |
| Determinisums-Fix (D-K3-GoNoGo) | „später“ | parallel ab jetzt, 1–3 Arbeitstage |
| Gepaarter DRT-12-A/B (finaler Satz) | 2 Kampagnen (~50 h) | EINE (~8–10 h) |
| Danach | — | Trennung → §4-Pakete → Hör-Sitzung (~1–2 h Hörzeit) |

Erwartung: Die **Maschinen-Kampagne schrumpft von ~2–3 Tagen auf ~1 Tag**,
und die Ingenieur-Arbeit (Determinismus) läuft parallel, statt danach.

## 5. Umsetzungsreihenfolge

1. ✅ H1/H2 in den laufenden Ketten (jitter-v2, Dropout-EXP v6) — erledigt 02:00.
2. H6 (Detektor-GPU) — nächster Ingenieur-Schritt, klein, mit Tests.
3. H3 Determinisums-Fix — ab sofort parallel (async-Konsum, §0h-Grenzverhalten).
4. H4 Sweeps, sobald die ersten GPU-Trainings laufen (Mess-Tabelle je Klasse).
5. H5: finaler Satz → EINE gepaarte A/B → Trennung → §4.

## 6. Risiken

- `--device cuda` im Kopf-Trainer: falls der ROCm-Pfad unerwartet scheitert,
  sofortiger Fallback auf CPU (Kette patched zurück) + Trainer-Fix als Task.
- UAT-Frische: der UAT-Neulauf ist frisch (2026-11-23, 29/30); während GPU-Trainings
  ist kein UAT-Neulauf möglich — Terminierung beachten.
- H2 ändert NICHT die Datenqualität: `--nur-schaden` nutzt denselben Rezept-Generator
  und dieselben Seeds — die Trainings-Regionen sind bit-gleich.
