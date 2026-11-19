# WP-7 — Echt-Träger-Beschaffung: Vollzug und Validierung (2026-10-09, 10.12.33)

**Auftrag:** „Beschaffe optimalen WP-7-Träger eigenständig und proaktiv und validiere das Optimum nach Fertigstellung."
**Ergebnis in einem Satz:** 51 neue Echt-Träger, Korpus 92 → 143 Labels (63 real statt 12); Shellac-Ziel ≥50 erreicht (58);
der neue Bestand ist durch 49 Tests abgesichert — und die Abschluss-Validierung legt eine reale Detektor-Schwäche offen (D-K1-3).

---

## 1. Vorgehen (ohne Abkürzungen)

1. **Lizenzvertrag strikt:** `corpus/README.md` verlangt **Public Domain oder CC0** — die erste Kurationsrunde (v2) lieferte
   überwiegend CC BY-NC(-ND/SA) (vertragswidrig) und wurde verworfen; v3–v5 filtern lizenzstrikt.
2. **Fünf Kurationsrunden** (Internet Archive Advanced Search + Metadaten-API): Trefferzahlen → Filter (Jahr, Lizenz,
   Sprech-Abweisung, Musikkontext) → Metadaten → Datei-Auswahl (Größenfenster, Original vor Derivative).
3. **Manuelle Einzelprüfung jeder Klasse** vor der Aufnahme (u. a. Aussortierung von Uploader-„CC0" auf kommerziellen
   Releases — Boots Randolph, Atlantic Records, Chrome, Concert For Bangladesh: **verworfen**, keine Schein-Provenienz).
4. **Aufnahme über den kanonischen Generator** (`scripts/generate_corpus_from_public_domain.py`) — inklusive der in 10.12.32/33
   gefixten Merge-Semantik, des neuen **Lizenz-Gates** (PD/CC0 mechanisch erzwungen) und des **Real-Träger-Zählwerks** (`--check`).

## 2. Quellenlage und Ausbeute je Klasse (gemessen)

| Klasse | Suchraum (PD/CC0-gefiltert) | Aufgenommen (real) | Befund |
| --- | ---: | ---: | --- |
| shellac | 78rpm-Collection ≤1927 (441 allein für 1927) | **47** | PD gesichert (US-Sound-Recording-PD ≤1927, MWA-Frist 2026); Oper/Operette/Arien 1902–1927 |
| digital | Netlabels CC0/PD (2.397 Treffer) | **10** | Selbst-Dedikation der Labels (z. B. treetrunk, SEAM, MARAV) — legitim |
| cassette | Cassette-Kultur PD/CC0 (10–297 Treffer) | **4** | DIY-Selbstdedikationen (1987 Canadian Cassette Culture, Home-Tap-Kollektive) |
| reel_tape | Spulband + PD/CC0 (140 Treffer) | **1** | dokumentierte Transferkette (Reel > PCM > FLAC) |
| tape | Tape-CC0 (344 Treffer) | **1** | Tape-Art-Projekt (CC0) |
| vinyl | LP-Ära + PD/CC0 (40k+ Treffer) | **0** | **strukturell:** LP-Ära (ab 1948) unter Schutz (Bundesrecht ab 1972; Vor-1972 einzelstaatlich); Uploader-CC0 auf Kommerz-Releases unglaubwürdig → KEINE Aufnahme |
| **Summe** | | **63 real** | davon 51 in diesem Lauf beschafft |

**Verworfene Fehl-Provenienzen (Beispiele, dokumentiert):** Boots Randolph 1963, Atlantic-Records-Kompilation 1976,
Chrome 1980, Concert For Bangladesh 1971, japanische Kommerz-Singles 1971/1979 — alle mit Uploader-CC0/PD markiert,
faktisch geschützt. **Falsch-positive Kandidaten** (falsche Klasse durch Suchbegriff): `KeenAge` („reel" = Irish Reel),
`riego-de-secano` („reel" = Bewässerungs-Film). Beides abgelehnt.

**Totquellen (bleiben als Leads):** `derhonvedhusar` und `NotAfricaineOdeon` lieferten reproduzierbar HTTP 503/Node-Timeout
(4 Versuche, 2 Laeufe); Ersatz durch 2 verifizierte 1927-Träger (`weine-nicht-liu-o-7536`, `fausttriodeduel`).

## 3. Code-Änderungen (Teil des Commits 10.12.33)

- **Merge-Manifeste** statt Overwrite (kein Datenverlust, Dedupe per `file`; unlesbares Bestands-Manifest ⇒ Schreiben
  übersprungen, §V6) — 10.12.32/33-Basis, hier erstmals im vollen Lauf wirksam (vinyl 21 synthetische Einträge blieben erhalten).
- **Lizenz-Gate** (`_license_allowed`, PD/CC0-Allowlist): erzwingt den corpus/README-Vertrag mechanisch; Verstöße werden
  mit `logger.warning` übersprungen (§V6).
- **Slug-Schema** `<material>_<era>_<slug>.<ext>`: stabile, kanonische Dateinamen (47/47 Shellac-Quellen); 9 Alt-Namens-Orphans
  hash-identisch verifiziert und entfernt.
- **Real-Träger-Zählwerk:** `check_coverage()` liefert `per_material_real` (source_url-basiert), `wp7_real_gap`,
  `wp7_real_met` — die K1-2-Lücke („mostly synthetic labels") ist damit maschinenlesbar (Ziel ≥10 real je Klasse).

## 4. Validierung („das Optimum nach Fertigstellung")

1. **Tests:** 49 grün (42 `tests/corpus/` + 7 neue Vertragstests `tests/unit/test_corpus_generator_merge.py`).
2. **Coverage (`--check`):** 143 Labels; `wp7_gap`: shellac **0** (58 ≥ 50 erreicht); real: shellac 47, digital 10, cassette 4,
   reel_tape 1, tape 1, vinyl 0 (dokumentiert). Rohdaten: `docs/reports/current/2026-10-09_wp7_coverage.json`.
3. **MediumDetector-Validierung an 9 Echt-Trägern** (Roh-Aufruf wie `pre_analysis.py:322`, 30-s-Fenster):
   1/9 hart korrekt (Caruso-78er shellac 0,97), 1/9 `mp3_high` (0,45), **5/9 `vinyl`** (0,60–0,99; u. a. 1987-Cassette mit 0,99),
   1 `unknown`, 1 `lacquer_disc`. Ergebnis deckt sich mit §v10.19 (Era-Prior im Produktionsaufruf nie aktiv; nachgelagerte
   Era-Konsistenz `pre_analysis.py:755–830` dämpft nur teilweise) → **neu registriert als D-K1-3** (Kalibrierung des
   Roh-Detektors gegen den jetzt verfügbaren Real-Korpus). Rohdaten: `docs/reports/current/2026-10-09_wp7_medium_validierung.json`.

## 5. Offene Punkte

- **Vinyl-Rechtelage:** ohne Rechteinhaber-Freigabe-Politik (z. B. Unlocked-Recordings) bleibt vinyl ohne Echt-Quellen —
  Maintainer-Entscheid (Register D-K1-2, Spalte Blocker).
- **Weitere Netlabel-Runden** möglich (2.397 CC0/PD-Kandidaten; aktuell 10 genutzt) — Ausbau bei Bedarf.
- **D-K1-3 Kalibrierung** mit dem neuen 63-Träger-Bestand (Grundlage für Material-Klassifikator-Training).
