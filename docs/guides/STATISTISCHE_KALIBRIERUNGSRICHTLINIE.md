# Statistische Kalibrierungsrichtlinie fuer Zielschwellen

Version: 1.0
Stand: 2026-04-08
Geltung: Alle Schwellwerte in Musical Goals, PQS/OQS und Gates

## 1. Zweck

Diese Richtlinie stellt sicher, dass jede Schwellwert-Aenderung datengetrieben, reproduzierbar und auditierbar ist.

## 2. Pflichtdaten je Kalibrierung

- Datensatzbeschreibung (Quelle, Umfang, Materialmix, Modusmix)
- Stichprobengroesse und Power-Begruendung
- Vorverarbeitungsregeln
- Seeds und Versionsstaende
- Ausreisserregeln (vorab festgelegt)

## 3. Pflichtstatistik

Fuer jeden betroffenen Schwellwert:

1. Punktschaetzer (Mittel/Median)
2. 95 %-Konfidenzintervall (Bootstrap)
3. Sensitivitaet/Robustheit ueber Materialklassen
4. Effekt auf Fehlentscheidungen (False Accept / False Reject)
5. Vergleich alt vs. neu (Delta inkl. Unsicherheit)

## 4. Entscheidungsregel fuer neue Schwelle

Eine neue Schwelle darf nur uebernommen werden, wenn:

- sie die Zielmetrik signifikant verbessert oder stabilisiert,
- sie keine nachweisbare Verschlechterung bei P1/P2 verursacht,
- der Effekt konsistent ueber relevante Materialklassen ist,
- und die Unsicherheit dokumentiert ist.

## 5. Pflicht-Template pro Schwellwert

- Zielname:
- Alte Schwelle:
- Neue Schwelle:
- Datengrundlage:
- n:
- 95 %-CI alt:
- 95 %-CI neu:
- Delta (neu-alt):
- Material-Robustheit:
- Risikoanalyse:
- Entscheidung:
- Reviewer-Freigabe:

## 6. Verbotene Praktiken

- Schwellwert-Aenderungen ohne Datennachweis
- Nachtraegliches Aendern des Statistikplans
- Entfernen unguenstiger Szenarien ohne dokumentierte Ausschlussregel
- Heuristische Fixwerte ohne CI/Effektstaerke

## 7. Ablagepflicht

Jede Kalibrierung muss enthalten:

- Kalibrierungsreport unter `docs/reports/`
- Rohdaten-Hinweis (anonymisiert)
- Reproduktionsskript (mit Seed)
- Changelog-Eintrag mit Verweis auf den Report

## 8. Minimaler CI-Check fuer Kalibrierungen

- Parser validiert Vollstaendigkeit der Pflichtfelder
- Seed vorhanden
- 95 %-CI vorhanden
- Reviewer-Sign-off vorhanden
