# Spec-Evidenzblock-Template

Version: 1.0
Stand: 2026-04-08
Zweck: Pflichtblock fuer jede normative Aenderung in `.github/specs/*`

## Verwendung

Diesen Block unterhalb der geaenderten Spec-Stelle einfuegen oder im zugehoerigen Report referenzieren.
Alle Felder muessen ausgefuellt sein. Nicht anwendbare Messungen brauchen eine
konkrete Begruendung; leere Felder, Platzhalter und ein ausstehender Sign-off
bestehen das Gate nicht. Eine DRT-16/RX-12-Paritaetsbehauptung muss auf den
kanonischen JSON-Beleg unter `docs/reports/current/` verweisen.

---

## Evidenzblock

- Spec-Datei:
- Abschnitt:
- Aenderungstyp: Schwelle | Gate | Algorithmus | Fallback | Sonstiges
- Alte Regel:
- Neue Regel:

### 1. Wissenschaftliche Begruendung

- Fachliche Hypothese:
- Referenzen (Paper/Standard):
- Warum ist die Aenderung kausal plausibel?

### 2. Datengrundlage

- Datensaetze/Szenarien:
- Umfang (n):
- Material- und Modusabdeckung:
- Ausschlusskriterien:

### 3. Statistik

- Primarmetrik:
- Effektstaerke:
- 95 %-CI:
- Signifikanztest + p-Wert:
- Multiple-Testing-Korrektur:

### 4. Reproduzierbarkeit

- Seed(s):
- Commit:
- Skript/Befehl:
- Artefaktpfade:

### 5. Risikoanalyse

- Risiko fuer P1/P2:
- Risiko fuer Artefakte:
- Bekannte Unsicherheiten:
- Rollback-Kriterium:

### 6. Entscheidung

- Entscheidung: APPROVED | REJECTED | CONDITIONAL
- Maintainer Sign-off:
- Externer Reviewer (optional):
- Datum:

---

## Minimalanforderung fuer Merge

Ein Spec-Merge ist nur zulaessig, wenn Abschnitt 1-6 vollstaendig ausgefuellt ist und ein Maintainer-Sign-off vorliegt.
