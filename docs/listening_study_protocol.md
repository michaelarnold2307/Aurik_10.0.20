# Hörstudien-Protokoll für Aurik

> §15.10: Formale perzeptuelle Validierung nach ITU-R BS.1116 und BS.1534-3.
> Status: Protokoll — Studie noch nicht durchgeführt.

## Studiendesign

### Ziel
Vergleich der perzeptuellen Qualität von Aurik-Restaurierungen mit:
1. **Original-Referenz** (Hidden Reference)
2. **iZotope RX 11/12** (Industriestandard)
3. **3.5-kHz-Tiefpass-Anchor** (ITU-R BS.1534 Kalibrierung)

### Design
- **Verfahren**: MUSHRA (ITU-R BS.1534-3)
- **Design**: Within-subjects, doppelblind
- **Bedingungen**: 4 (Hidden Ref, Aurik, RX 11, Anchor)
- **Szenarien**: 12 (4 Materialien × 3 Defektstufen)
- **Trials**: 12 × 4 × 3 Wiederholungen = 144 Trials
- **Dauer**: ~45 Minuten pro Teilnehmer

### Material

| Material | Defektstufe | Beispiel-Defekte |
|----------|------------|------------------|
| Schellack | Leicht/Mittel/Schwer | Klicks, Rauschen, Bandbreite |
| Vinyl | Leicht/Mittel/Schwer | Knackser, Rillenrauschen |
| Tonband | Leicht/Mittel/Schwer | Rauschen, Dropouts, Gleichlauf |
| Kassette | Leicht/Mittel/Schwer | Rauschen, Höhenverlust |

## Teilnehmer

- **Anzahl**: ≥12 (Minimum für ANOVA)
- **Einschluss**: Normales Hörvermögen (selbstberichtet)
- **Ausschluss**: Professionelle Audio-Engineers (Voreingenommenheit)
- **Rekrutierung**: Hochschule, Tonstudio-Netzwerk
- **Ethik**: Informed Consent, Anonymisierung, Widerrufsrecht

## Ablauf

1. **Training** (5 min): 3 Übungs-Trials mit Feedback
2. **Hauptstudie** (35 min): 144 Trials in randomisierter Reihenfolge
3. **Pause** nach 72 Trials (Pflicht, 2 min)
4. **Debriefing** (5 min): Offene Fragen, Feedback

## Technische Umsetzung

- **Audio-Interface**: Kopfhörer (Sennheiser HD 650 oder äquivalent)
- **Pegel**: Kalibriert auf 73 dB(C) SPL (Pink Noise -20 LUFS)
- **Wiedergabe**: 48 kHz, 24-bit, kein Sample-Rate-Conversion
- **Software**: Aurik MUSHRA Listener (backend/core/mushra_listener.py)

## Auswertung

### Primäre Endpunkte
- MUSHRA-Score (0–100) pro Bedingung
- Aurik vs. RX 11: Gepaarter t-Test (α = 0.05, Bonferroni-korrigiert)

### Sekundäre Endpunkte
- Aurik vs. Hidden Reference: Äquivalenztest (Δ < 5 MUSHRA-Punkte)
- Anchor-Konsistenz: Score < 30 (Validierungscheck)
- Inter-Rater-Reliability: ICC (Intraclass Correlation)

### Statistische Methoden
```
# Analyse mit scripts/analyze_listening_study.py
python scripts/analyze_listening_study.py \
    --results results/listening_study_2026.json \
    --output results/analysis_2026.json \
    --alpha 0.05
```

## Ergebnisdokumentation

Template: `docs/PERCEPTUAL_VALIDATION_REPORT.md`

## Ethik & Datenschutz

- Keine personenbezogenen Daten außer anonymisierter Listener-ID
- Ergebnisse nur aggregiert (keine Einzel-Scores publiziert)
- Teilnehmer können jederzeit abbrechen
- Daten werden nach 5 Jahren gelöscht

## Geplante Publikation

- ArXiv-Preprint (Open Access)
- AES Convention Paper (Peer-Review)
- GitHub: Rohdaten + Analyse-Skripte (CC0)
