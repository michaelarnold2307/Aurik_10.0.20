# PR Hoervalidierung Template (Mini-MUSHRA / Blindtest)

Version: v10.0.0
Datum:
PR/Commit:
Autor:
Reviewer:

## 1. Scope

Kernaenderung:
Betroffene Module/Phasen:
Testziel in 1 Satz:

Modus-Abdeckung:

- [ ] restoration
- [ ] studio2026

## 2. Pflicht-Setup (Pre-Check)

Hoereranzahl (Pflicht >= 8):
Szenarien gesamt (Pflicht >= 6):
Vocal-Szenarien (Pflicht >= 2):
Abhoerumgebung (Kopfhoerer/Monitore, Pegel, Raum):
Blindtest-Protokoll (A/B/X oder MUSHRA-like):
Datumsfenster aller Szenarien (<= 24h):

Pre-Check Ergebnis:

- [ ] Vollstaendig
- [ ] Unvollstaendig (NO-GO)

## 3. Szenarienmatrix (Pflicht: 3x restoration, 3x studio2026)

- S1: Material=, Defektklasse=, Modus=restoration, Input=, Restored=, Reference=, Kommentar=
- S2: Material=, Defektklasse=, Modus=restoration, Input=, Restored=, Reference=, Kommentar=
- S3: Material=, Defektklasse=, Modus=restoration, Input=, Restored=, Reference=, Kommentar=
- S4: Material=, Defektklasse=, Modus=studio2026, Input=, Restored=, Reference=, Kommentar=
- S5: Material=, Defektklasse=, Modus=studio2026, Input=, Restored=, Reference=, Kommentar=
- S6: Material=, Defektklasse=, Modus=studio2026, Input=, Restored=, Reference=, Kommentar=

## 4. Bewertungsachsen

### Restoration (Pflichtachsen)

- Natuerlichkeit (>= 3.5):
- Authentizitaet (>= 3.5):
- Artefaktfreiheit (Artifact-Veto beachten):
- Tonale Treue (>= 3.5):

### Studio2026 (Pflichtachsen)

- Frische/Presence (>= 3.8):
- Punch/Bass-Kraft (>= 3.9):
- Klarheit (>= 3.9):
- Artefaktfreiheit (Artifact-Veto beachten):

## 5. Ergebnis je Modus

### restoration

- Mini-MUSHRA/Blindtest Mittelwert:
- Konfidenzintervall:
- Delta zur Vorversion:
- Artifact-Veto ausgeloest (>= 2/8 hoeren neues Artefakt):

  - [ ] nein
  - [ ] ja (BLOCKER)

- P1/P2-Verschlechterung gegen Input:

  - [ ] nein
  - [ ] ja (BLOCKER)

### studio2026

- Mini-MUSHRA/Blindtest Mittelwert:
- Konfidenzintervall:
- Delta zur Vorversion:
- OQS >= 88 erreicht:

  - [ ] ja
  - [ ] nein (BLOCKER)

- PQS_MOS Ziel je Szenario erreicht:

  - [ ] ja
  - [ ] nein (BLOCKER)

- OQS/Hoerurteil konsistent:

  - [ ] ja
  - [ ] nein (BLOCKER bis Root-Cause)

## 6. GO/NO-GO Entscheidung

Release-Entscheidung:

- [ ] GO
- [ ] Conditional GO (mit Auflagen)
- [ ] NO-GO

Blocker (falls vorhanden):
Root-Cause-Hinweise:
Risiko nach Release:
Auflagen bei Conditional GO:

## 7. Reviewer Sign-Off

Reviewer Handle:
Datum:
Freigabe:

- [ ] Genehmigt
- [ ] Abgelehnt

## 8. Artefakte

Rohdaten/Anhaenge (Pfad/Link):
Audio-Snippets (Pfad):
Plots/Metriken (Pfad):
Zusatznotizen:
