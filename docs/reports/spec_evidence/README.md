# Spec-Evidenzberichte

Dieser Ordner enthaelt verpflichtende, ausgefuellte Evidenzberichte fuer jeden
Push auf `main` (einschliesslich gemergter PRs), wenn Dateien unter
`.github/specs/` geaendert wurden. Der Bericht muss im selben Push wie die
Spec-Aenderung enthalten sein. Fuer PRs bleibt zusaetzlich der vollstaendige
Evidenzblock im PR-Body erforderlich.

Pflicht bei Spec-Aenderungen:

- Mindestens eine neue/aktualisierte Markdown-Datei in diesem Ordner
- Das Template unter `docs/guides/SPEC_EVIDENZBLOCK_TEMPLATE.md` ist
  auszufuellen; Marker allein reichen nicht.
- Nicht anwendbare Messungen brauchen eine konkrete Begruendung. Fehlende
  Daten, Seed oder Konfidenzintervalle duerfen nicht erfunden werden.
- Der Maintainer-Sign-off muss eine tatsaechliche Freigabe mit Name und Datum
  dokumentieren; `pending`, leere Felder oder Platzhalter bestehen das Gate
  nicht.
- Eine DRT-16/RX-12-Paritaetsbehauptung muss auf einen kanonischen JSON-Beleg
  unter `docs/reports/current/` verweisen. Die JSON-Validierung erfolgt ueber
  den bestehenden Wettbewerbs-Paritaets-Vertrag; ein Markdown-Bericht allein
  belegt keine Produktparitaet.

Referenz-Template:

- `docs/guides/SPEC_EVIDENZBLOCK_TEMPLATE.md`

Namensschema:

- `YYYY-MM-DD_<kurzthema>.md`

Enforcement: `scripts/spec_evidence_gate.py` wird im PR-, Nightly- und
Main-Push-Gate mit denselben Inhaltsregeln ausgefuehrt.
