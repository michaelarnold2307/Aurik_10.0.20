# Spec-Evidence: Spec 25 Ambience-Match — Natürliche Raumhülle (Entwurf)

Datum: 2026-09-22 | Spec: `.github/specs/25_ambience_match_plugin.md` (+ Index `.github/specs/00_SPEC_INDEX.md`) | Version: 10.1.0

---

## Evidenzblock

### 1. Änderungsgegenstand

- **Spec-Datei:** `.github/specs/25_ambience_match_plugin.md` (neu), `.github/specs/00_SPEC_INDEX.md` (Eintrag + Zähler 94 → 95)
- **Abschnitt:** gesamt (neue Entwurfs-Spec nach Vorlage `XX_measure_template.md`)
- **Änderungstyp:** Sonstiges — reine Design-/Roadmap-Spec (Status: Entwurf); **keine** Laufzeit-, Schwellen-, Gate- oder Algorithmus-Änderung
- **Alte Regel:** Keine Ambience-Klasse im Plugin-Bestand; die Rausch-/Raumbehandlung der Phasen 03/18/20/26 entfernt nur, stellt nichts wieder her.
- **Neue Regel:** Normativ verankerter Entwurf für ein deterministisches DSP-Plugin (`plugins/ambience_match_plugin.py`), das eine raumprofil-treue Hülle unter der Maskierungsschwelle (σ = 6 dB Sicherheitsabstand) synthetisiert — mit psychoakustischen Invarianten H1–H4, Determinismus-Vertrag (§G5 (copilot-instructions.md)), Blend-Garantie und GO/NO-GO-Hörprotokoll als Erfolgskriterium.

#### 1. Wissenschaftliche Begründung

- **Fachliche Hypothese:** Aggressive Restaurierung entfernt die psychoakustische Grundhülle (Raumtönung, Saalrauschen, Bandgrund); das Ohr nimmt die entstandene „schwarze Stille“ als Künstlichkeit wahr (Hörordnung, Ebene 1: Audibility an der Maskierungsschwelle statt Mess-Null). Eine additive, unter der Maskierungsschwelle kalibrierte Hülle stellt Natürlichkeit wieder her, ohne das Nutzsignal zu verändern.
- **Referenzen (Paper/Standard):** Johnston (1988) Maskierungsmodell; Glasberg & Moore (2002) Lautheits-/Maskierungsforschung; `.github/instructions/hoerordnung.instructions.md` (Maskierungsschwelle, lexikografische Wohlklang-Ordnung); iZotope RX „Ambience Match“ als industrielles Referenzkonzept (kein Code-Bezug).
- **Warum kausal plausibel:** Die Hülle ist rein additiv und liegt je Bark-Band mindestens σ unter der Schwelle — per Konstruktion unhörbar als Rauschzuwachs (Invariante H1), wirkt aber gegen den „Totklang“ nach der Restaurierung. Blend=0 ist bit-identischer Passthrough.

#### 2. Datengrundlage

- **Datensätze/Szenarien:** entfällt — Entwurfs-Spec ohne Laufzeit-Änderung; es wurden keine Audiodaten verarbeitet.
- **Umfang (n):** n=0 Audio-Läufe; verifiziert wurden statische Registrierungs-Gates (siehe §4).
- **Material- und Modusabdeckung:** entfällt (Entwurf).
- **Ausschlusskriterien:** entfällt.

#### 3. Statistik

- **Primärmetrik:** Gate-Sauberkeit der Registrierung: spec_index-Konsistenz (Scanner: 0 Fehler), ID-Registry (0 Warnungen), markdownlint (0 Befunde).
- **Effektstärke:** entfällt (keine Laufzeit-Änderung).
- **95 %-CI:** Deterministische Prüfskripte — CI = Punktwert (0 Varianz über Wiederholungen).
- **Signifikanztest + p-Wert:** entfällt (deterministische Checks, keine Stichprobe).
- **Multiple-Testing-Korrektur:** entfällt.

#### 4. Reproduzierbarkeit

- **Seed(s):** entfällt (keine stochastischen Anteile; deterministische Skripte). Für die spätere Implementierung ist der Session-Seed-Vertrag in der Spec fixiert (§G5 (copilot-instructions.md)).
- **Commit:** `01c9807c` (Spec 25 + Index-Eintrag; in Push `ece18a60..062c29df` auf main enthalten).
- **Skript/Befehl:** `python audit/spec_integration_scanner.py --fail-on error`; `python scripts/id_registry_check.py`; `node_modules/.bin/markdownlint -c .markdownlint.json .github/specs/25_ambience_match_plugin.md`.
- **Artefaktpfade:** `audit/spec_integration_report.md` (INFO: Spec nur im Index verankert — dokumentierter Deferred-Status wie Spec 22/24).

#### 5. Risikoanalyse

- **Risiko für P1/P2:** Keines — keine Laufzeit-Änderung; die Spec verbietet Verdeckung aktiver Defekte (Invariante H3, Defect-Map-Kopplung).
- **Risiko für Artefakte:** Keines zum jetzigen Zeitpunkt; für die Implementierung sind H1–H4 sowie das GO/NO-GO-Protokoll als Gate fixiert.
- **Bekannte Unsicherheiten:** Kein Plug-in-Code vorhanden (Status Entwurf); die psychoakustischen Invarianten sind normative Zusagen für die Umsetzung.
- **Rollback-Kriterium:** Spec ist reines Dokument — Rollback = Zeile im Index entfernen.

#### 6. Entscheidung

- **Entscheidung:** APPROVED (Entwurf verankert, Implementierung getrennt freizugeben)
- **Maintainer Sign-off:** Michael Arnold (Solo-Maintainer) — 2026-09-22
- **Externer Reviewer (optional):** —
- **Datum:** 2026-09-22

---
