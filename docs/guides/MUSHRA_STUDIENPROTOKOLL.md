# MUSHRA-Studienprotokoll (extern, ITU-R BS.1534-3)

Version: 1.0
Stand: 2026-04-08
Geltung: RELEASE_MUST fuer Schwellwert-Aenderungen in Musical Goals und OQS/PQS-Gates

## 1. Ziel und Hypothese

Ziel: Externe, statistisch belastbare Validierung der Aurik-Qualitaetsziele mit ITU-R BS.1534-3-konformer Hoerstudie.

Primarhypothese (H1):
Aurik ist in den definierten Szenarien gegen die Baseline ueberlegen.

Nullhypothese (H0):
Kein signifikanter Unterschied zwischen Aurik und Baseline.

## 2. Studiendesign

- Protokoll: ITU-R BS.1534-3 MUSHRA, double-blind
- Randomisierung: Latin-Square oder vollstaendig randomisiert pro Teilnehmer
- Teilnehmerzahl (Minimum): n >= 30
- Empfohlen fuer finale Freigaben: n >= 50
- Materialumfang: 10 AMRB-Szenarien, je 30-60 s
- Modi: `restoration` und `studio2026`
- Baselines:
  - interne Vorversion
  - mindestens ein externer Wettbewerber
  - Hidden Reference + Low Anchor

## 3. Ein-/Ausschlusskriterien

Einschluss:

- Hoertest-Selbstauskunft ohne bekannte akute Hoerschaeden
- Training-Block bestanden

Ausschluss:

- Antwortmuster ohne Trennschaerfe (z. B. Hidden-Reference systematisch niedrig)
- Unvollstaendige Sessions

## 4. Testumgebung

- Abhoere: kalibrierte Kopfhoerer oder kontrollierter Monitorraum
- Pegelkalibrierung vor Start dokumentiert
- Dateiformat: WAV/FLAC, 48 kHz
- Lautheitsabgleich: ITU-R BS.1770-5, keine pegelbedingte Bevorzugung

## 5. Bewertungsachsen

Pflichtachsen `restoration`:

- Natuerlichkeit
- Authentizitaet
- Artefaktfreiheit
- Tonale Treue

Pflichtachsen `studio2026`:

- Klarheit/Presence
- Punch/Bass-Kraft
- Raumwirkung
- Artefaktfreiheit

## 6. Statistikplan (vorab festgelegt)

- Signifikanzniveau: alpha = 0.05
- Mehrfachtests: Holm-Bonferroni
- Effektstaerke: Cohen d + 95 %-Konfidenzintervall
- Unsicherheit: Bootstrap-CI (mind. 10.000 Resamples)
- Reporting pro Szenario und aggregiert

## 7. Abnahmekriterien

Ein Release-Kandidat gilt als extern validiert, wenn alle Punkte erfuellt sind:

1. Primarmetrik signifikant besser als Baseline (korrigiertes p < 0.05)
2. Effektstaerke mindestens klein-mittel (d >= 0.3) in der Aggregation
3. Keine systematische Verschlechterung in P1/P2-nahen Achsen
4. Artefaktfreiheit ohne Veto-Ereignis

## 8. Pflichtartefakte

- Praeregistriertes Statistikprotokoll (unveraenderbar versioniert)
- Rohdaten (anonymisiert)
- Auswerteskripte inkl. Seed
- Abschlussbericht mit CI, p-Werten, Effektstaerken
- Mapping auf betroffene Schwellwerte in Specs

## 9. Governance

- Ohne externen Studienreport: keine Schwellwertverschaerfung oder -lockerung in `.github/specs/01` und `.github/specs/07`
- Jede Aenderung muss auf einen Studienreport verweisen (Dateipfad + Datum + Commit)

## 10. Vorbereitung 2026-09 — konkreter Studienplan (P1-4)

> Ergänzung vom 2026-09-14; ersetzt nicht §2–§9, sondern konkretisiert den Einstieg.

**Stimuli-Set (Pilot, 6 Szenarien × 30 s, Seed 42):**

1. Vinyl worn (rock_1970s_worn) — Restoration
2. Cassette Wow (cassette_1980s_wow) — Restoration
3. MP3 64 kbps (mp3_64kbps_artifacts) — Restoration
4. CD clipped (cd_clipped_2000s) — Restoration
5. Reel dropout (reel_1940s_dropout) — Restoration
6. Studio2026-Szenario (e2e_jazz_studio2026) — studio2026

Quellen: `test_audio/` (Inputs) + `output/supervised_run/` (Aurik-Ausgaben,
2026-09-Batch, Exit 0). Bedingungen je Szenario: Hidden Reference, Low Anchor
(3,5-kHz-Tiefpass, −6 LUFS), Aurik-Ausgabe, interne Vorversion, externer
Wettbewerber (erst nach Lizenzklärung, sonst 4 Bedingungen).

**Ablauf:**

- Lautheitsabgleich BS.1770-5 (existiert im Export-Pfad) + 500-ms-Crossfades.
- Training-Block (2 Szenarien) vor dem Testblock; Reihenfolge Latin-Square.
- Pilot n=12 (intern) → Hauptstudie n≥30; Ausschlusskriterien nach §3.

**Tooling (lokal vorhanden):**

- `tests/test_blindtest_metrics.py` — Auswertemetriken (MUSHRA-konforme
  Differenz-/CI-Statistik) bereits im Repo und per Hook verdrahtet.
- Fehlend: Stimuli-Builder (Cut/Normalisierung/LUFS) + Web-MUSHRA-Deployment —
  als eigene Teilschritte vor dem Pilot.

**Meilensteine:** Stimuli-Builder → Pilot (n=12) → Auswertung + ggf.
Design-Fixes → Hauptstudie (n≥30) → Studienreport nach §8.
