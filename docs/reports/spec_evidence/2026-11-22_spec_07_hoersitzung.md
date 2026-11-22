# Spec-Evidenz: Hörsitzung (Spec 07 §8.8)

## Evidenzblock

- Spec-Datei: `.github/specs/07_quality_and_tests.md`
- Abschnitt: Spec 07 §8.8 (neu, [RELEASE_MUST]) — Hörsitzung: verblindete, richtungsexplizite Hör-Evidenz
- Aenderungstyp: Sonstiges — Prozess- und Evidenzvertrag für menschliche Hörurteile; kein Schwellwert mit Audio-Wirkung, keine Pipeline-Änderung
- Alte Regel: Hörpakete wurden einzeln und uneinheitlich manuell abgehört; `flag_freigabe.verdikt_ok` akzeptierte `aurik.ab_stats/1` richtungsblind (ein signifikantes Votum für den Bestand wäre buchbar gewesen); statistisch unentscheidbare Blöcke (n < 6) wurden zum Hören angeboten.
- Neue Regel: Ein Befehl (`scripts/hoersitzung.py`) baut verblindete Sitzungen mit Positionsbalance, Pegelabgleich und Kontrollen, prüft vorab die Machbarkeit (n ≥ 6), schließt byte-identische Paare aus und erzeugt `aurik.ab_stats/2` mit expliziter Richtung. `flag_freigabe` bucht nur bei `siege_kandidat > siege_bestand`; `/1` nur mit `kandidat_seite`; simulierte/ungültige/unvollständige Belege nie.
- Hypothese: Eine fehlerarme, verblindete Ein-Befehl-Sitzung erhöht den Nutzen jeder Hörerminute und verhindert Fehlbuchungen durch Richtungsverwechslung; sie ersetzt keinen Hörer und kein Panel.
- Referenzen/Standards: hoerordnung.instructions.md §7; ITU-R BS.1534 (Anker); Spec 07 §8.1.5; §G5, §G8, §V7 (copilot-instructions.md).
- Datensaetze/Szenarien: (1) Synthetische Pakete in tmp_path im Layout der echten Pakete mit Test-Hörer (nur tmp, `simuliert`). (2) Realitätscheck: `hoersitzung.py bauen` auf den echten, lokalen Paketen (kein Hören, keine Audioausgabe).
- Umfang (n): 37 Tests in `test_hoersitzung.py`, 14 in `test_flag_freigabe.py`; Realitätscheck n = 1 Bauvorgang.
- Primarmetrik: Vertragseigenschaften (bestanden/nicht bestanden). Realitätscheck: 74 Proben (8 Kontrollen), ≈ 13 min; 6 Blöcke zurückgestellt (declip 3, jitter 1, transport 1, stereo 2, dereverb 5 Paare, c4_vocos 0 nach Ausschluss); 14 Paare ausgeschlossen, davon alle 12 C4-Vocos-Paare byte-identisch (SHA-256 verifiziert) — ein Paketdefekt, den das Werkzeug vor dem Hören aufdeckt.
- Effektstaerke: N/A — keine Effektmessung; es wird kein Klangeffekt behauptet.
- 95 %-CI: N/A — deterministische Vertragstests.
- Signifikanztest + p-Wert: N/A für diese Änderung; das Werkzeug implementiert den exakten zweiseitigen Vorzeichentest (Gleichstände zählen nicht), getestet gegen bekannte Werte (6/6 → 0,03125).
- Seed(s): 42 (Plan-Seed, deterministisch; gleicher Seed ⇒ gleicher Plan, getestet).
- Commit: Basis `2770cb49`; die Änderungen sind Teil des noch offenen Worktrees.
- Skript/Befehl: `AURIK_FORCE_CPU=1 python3 -B -m pytest -q tests/unit/test_hoersitzung.py tests/unit/test_flag_freigabe.py`
- Artefaktpfade: `scripts/hoersitzung.py`; `scripts/flag_freigabe.py`; `scripts/hoer_entscheidung.py`; `tests/unit/test_hoersitzung.py`; `tests/unit/test_flag_freigabe.py`; `docs/guides/HOERSITZUNG.md`
- Risiko fuer P1/P2: Keines für Audio: keine Pipeline-Änderung. Die Richtungsregel macht die Buchung strenger (zuvor buchbare, richtungsblinde `/1`-Belege sind nun abgewiesen).
- Risiko fuer Artefakte: Keines (Werkzeug schreibt nur nach `output_audio/hoersitzung/`).
- Bekannte Unsicherheiten: Pegel-Toleranz 0,5 dB und Anker-Mindestquote 0,75 sind unkalibrierte Vorbelegungen. Das Hör-Interface (ffplay/aplay/paplay, Einzeltasten) wurde nur über injizierte Player getestet, nicht mit echter Audioausgabe. Ein Hörer ist kein Panel; Declip/Stereo/Jitter/Transport/Dereverb bleiben bis zum Nachbau weiterer Paare unentscheidbar; C4-Vocos-Pakete müssen neu erzeugt werden. Es wurde noch kein echtes Hörurteil erhoben.
- Rollback-Kriterium: Rücknahme, falls die Sitzung ein Urteil einem falschen Kandidaten zuordnet oder ein simulierter/ungültiger Beleg buchbar wird.
- Entscheidung: CONDITIONAL — Werkzeug und Richtungsregel implementiert und getestet; kein Hörurteil, keine Flag-Aktivierung.
- Maintainer Sign-off: AUSSTEHEND — ein Maintainer-Sign-off wird nicht fingiert.
- Datum: 2026-11-22

## Nachtrag (erster realer Hörversuch, 2026-11-22)

- Befund: Die 12 Witness-Pakete (60 von 74 Proben) sind isolierte Defektregionen von 0,05–0,44 s ohne Musik; der Hörer meldete, dass nur ein kurzer Ausschnitt zu hören war. Die Machbarkeits-Vorprüfung kannte nur die Statistik, nicht die Hörbarkeit.
- Änderung: `MIN_HOER_DAUER_S = 2,0` (Vorbelegung, Hörpanel offen) — Blöcke mit kürzerem Median je Hälfte werden zurückgestellt, mit Grund und Nachbau-Hinweis; Spec 07 §8.8 ergänzt. Die Sitzung umfasst jetzt 12 Proben (M5, 6 Paare + 6 Kontrollen, ≈ 5 min).
- Zusätzliche Unsicherheit: Ob M5 (A_linear vs B_vocos_ft) eine sinnvolle Entscheidung abbildet, ist nicht geprüft; die Hörbarkeit der Witness-Klassen ist nur mit Kontext-Paaren prüfbar.

## Validierung

- `tests/unit/test_hoersitzung.py` und `tests/unit/test_flag_freigabe.py`: bestanden (siehe Befehl oben).
- Beim Testlauf gefundene und behobene Fehler: Zusammenfassung der Auswertung ohne `gueltig`/`hoerer`/`simuliert`; Tests mit falscher Erwartung zu „zurück“ und zum Anker-Test.
- Dies ist ein Prozess-/Vertragsnachweis, keine Hörabnahme.
