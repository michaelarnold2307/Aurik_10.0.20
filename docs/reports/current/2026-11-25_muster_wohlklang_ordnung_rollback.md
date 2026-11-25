# Muster-Befund: Die Pipeline verstößt selbst gegen die Wohlklang-Ordnung — und rollt dann komplett zurück (2026-11-25)

**Frage des Auftraggebers:** „Ist der aktuell eingeschlagene Weg das Optimum
für maximalen Wohlklang für das menschliche Gehör?"

**Antwort:** Nein — und die Pipeline selbst belegt es. Sie protokolliert ihre
eigenen Verstöße gegen die normative Wohlklang-Ordnung (Hörordnung §8a,
lexikografische Ordnung) und korrigiert sie anschließend mit **vollständigem
Rollback auf den unverarbeiteten Eingang** (`blend_factor 0.000`). Das heißt:
die Standard-Verarbeitung erzeugt systematisch Verbesserungen der Stufen 3–4
auf Kosten der **Stufe-1-Ziele** und verwirft danach die gesamte Arbeit.

## 1. Messung (bestehender Beleg-Log `2026-11-05_b3_w2w3_nachmessung.log`)

| Kennzahl | Wert |
| --- | --- |
| §Ebene-3-Verletzungs-Warnungen gesamt | **1.797** |
| Pipeline-Ende-Enforcement-Ereignisse | **24** |
| Actions | 24 × `ROLLBACK` |
| blend_factor | min/median/max **0.000 / 0.000 / 0.000** (vollständiger Rückbau) |

**Häufigste Verletzungs-Muster** (Gewinn → Kosten):

| Gewinn (Stufe 3–4) | Kostet (Stufe 1–2) | Anzahl |
| --- | --- | --- |
| transparenz | authentizitaet (**Stufe 1**) | 110 |
| waerme | authentizitaet (**Stufe 1**) | 106 |
| transparenz | emotionalitaet | 102 |
| brillanz | authentizitaet (**Stufe 1**) | 98 |
| brillanz | emotionalitaet | 83 |
| waerme | emotionalitaet | 78 |
| brillanz | tonal_center | 67 |
| waerme | natuerlichkeit | 66 |

**Struktur:** Die Anreicherungs-Phasen (Brillanz/Wärme/Transparenz) kaufen
ihre Gewinne konsistent mit den **höchstrangigen** Zielen (Authentizität,
Timbre-Authentizität, Emotionalität) — exakt das, was die lexikografische
Ordnung verbietet. Der Rollback (blend 0.0) macht daraus in der Praxis:
**24 Läufe ohne jegliche Restaurations-Wirkung** (Eingang = Ausgang), obwohl
der Defekt-Scanner Verarbeitung geplant hatte.

## 2. Muster zur Steigerung des maximalen Wohlklangs (Meldung an den Auftraggeber)

Die Ordnungs-Konformität muss **konstruktiv** hergestellt werden, nicht
korrektiv: Statt „verarbeiten → verstoßen → komplett zurückrollen" muss die
**zentrale Stärke-Logik** (§V7 (copilot-instructions.md): zentral, keine
phasen-individuellen Schwellwerte) die Anreicherungs-Phasen so begrenzen,
dass der Ausgang die Ordnung **nativ** erfüllt. Das ist der höchste
Wohlklang-Hebel, weil:

1. Jeder verhinderte Rollback ein verlorenes Restaurations-Ergebnis
   zurückgewinnt (heute: Eingang = Ausgang trotz geplantem Defekt-Eingriff).
2. Jeder nativ konforme Lauf den Blend-Schritt erspart — der Blend
   (verarbeitet ↔ unverarbeitet) mischt zwei spektral/phasen-verschiedene
   Signale; sein Wegfall entfernt eine Artefakt-Quelle.
3. Das Erfolgskriterium ist maschinell messbar: Verletzungs-Zähler und
   Rollback-Zähler müssen auf dem Korpus gegen Null gehen — ohne dass die
   Stufe-1-Deltas sinken. Die Hör-Instanz validiert anschließend (§4).

## 3. Vorgehen (Reihenfolge, ohne Schaden)

1. **Ordnungs-Konformitäts-Sweep** auf `data/schadens_proben` (Produktionspfad,
   CPU-only): Verletzungs-/Rollback-Rate je Rezept + die verursachenden
   Phasen identifizieren (aus `_phase_metadata_accumulator`).
   **Start NACH Abschluss der laufenden DRT-12-Kampagne** (Parallel-Last
   würde die Kampagnen-Kinder über die Live-Notbremse stören — der
   deterministische RAM-Planer schützt den Normalbereich, nicht den Notfall).
2. **Zentrale Stärke-Rekalibrierung** auf Basis der Sweep-Daten
   (Kandidaten: adaptive Thresholds der Anreicherungs-Phasen, Brillanz-Kappen,
   Wärme-/Transparenz-Ziele) — zentral, mit Vertragstests, delta-basiert
   (keine absoluten Normalwerte als Hard-Fail, AGENTS.md §3).
3. **Never-worsen-Beweis:** B3-äquivalente Messung auf dem Korpus (nie
   schlechter als Status quo, Stufe-1-Ziele unangetastet) + Verletzungs-
   /Rollback-Zähler ≈ 0.
4. **§4-Hör-Validierung** der betroffenen Materialien (freigegebener Plan
   „Hörbarkeit zuerst" bleibt komplementär — Blöcke B/C Wow/Flutter + Brumm).

## 4. Einordnung in die Gesamt-Strategie

- Die laufende DRT-12-Kampagne bleibt der ehrliche Satz-Beleg (bisher:
  Satz inert, Δ 0.000/0.000) — aber sie ist nicht der Wohlklang-Hebel.
- Die Substrat-Fixes (Determinismus, RAM-Planung) sind dafür die Voraussetzung:
  erst mit reproduzierbarer Pipeline ist der Verletzungs-Zähler ein
  kalibrierbares Steuer-Signal.
- Der §4-Plan „Hörbarkeit zuerst" bleibt gültig; der Ordnungs-Sweep ergänzt
  ihn um die maschinell messbare Seite derselben Wahrheit.
