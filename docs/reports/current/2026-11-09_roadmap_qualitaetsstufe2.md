# Roadmap — Qualitätsstufe 2 (Hebel A/B/C) getaktet gegen GPU-Ketten & Panel

**Datum:** 2026-11-09 · **Ausgangslage:** Die Specs sind 2026-SOTA-geschärft
(L1–L3 geschlossen); die nächste Qualitätsstufe kommt aus **gelerntem
Ohr-Urteil** (B) und **gelerntem Musik-Wissen** (A/C). Dieses Dokument taktet
die drei Hebel gegen die limitierte Ressource GPU (eine Karte, sequenzielle
Ketten) und die menschlichen Abhängigkeiten (Hör-Labels, Panel).

**Ressourcen-Wahrheit:** 1 GPU · Ketten laufen sequenziell mit 6-h-Timeout +
Resume · die Kette-2 läuft (FlashSR, 94 % VRAM) · Panel-Zeit ist der
kritische menschliche Pfad (D-K4-9: Protokoll fertig, keine Stimuli).

## Phase 0 — jetzt, GPU-frei (parallel zu Kette 2)

| # | Schritt | Aufwand | Ergebnis |
|---|---|---|---|
| 0.1 | **B-Stimuli-Job definieren:** MUSHRA-Session-Dateien aus dem fertigen Builder erzeugen (Referenz-Songs + Defekt-Klassen), Panel-Anleitung dokumentieren — der Mensch startet sofort | autonom, Stunden | Panel läuft parallel zu allen GPU-Arbeiten |
| 0.2 | **A-Design-Dokument:** Architektur-Wahl (Diffusions-/Flow-Matching-Prior, DiffWave/BigVGAN-Bausteine), Daten-Pipeline auf MUSDB18-HQ (vorhanden), deterministisches Sampling als §G5-Kompromiss, Never-worsen-Gate-Skizze | autonom, GPU-frei | fertiges Design vor Phase 1 |

## Phase 1 — nach Kette 2 (erstes freies Fenster)

Reihenfolge: **kurze offene Jobs zuerst** (Warteschlange abbauen), **dann Hebel A**.

| # | Job | Dauer | Herkunft |
|---|---|---|---|
| 1.1 | Cantus-fine_tune (Kette 1) echt fahren — No-Op-Fix D-K3-84 steht | ~1–2 h | offener Job |
| 1.2 | UAT-Evidenz-Lauf (Commit-Blocker) | ~20 min | offener Job |
| 1.3 | B3-Neumessung M12-WF4/M12-TPV2 (Modellpfad-Fix D-K3-85 steht) | ~2–3 h | offener Job |
| 1.4 | **Hebel A: Kette 5 — Musik-Rekonstruktions-Prior** (Training auf MUSDB18-HQ, deterministisches Sampling, Anker je Epoche nach D-K3-72-Standard, Never-worsen-/B3-Gate als Abschluss) | **Tage (größter Block)** | neu |

## Phase 2 — parallel zu A (menschlicher Pfad)

| # | Schritt | Verantwortung |
|---|---|---|
| 2.1 | MUSHRA-Panel-Durchführung (Stimuli aus 0.1) | **menschlich** |
| 2.2 | **Hebel B: Preference-Orakel F15 trainieren** auf den Panel-Labels (kleiner Head, GPU ~1–2 h — passt in ein A-Resume-Fenster) | autonom nach 2.1 |

## Phase 3 — nach A + B

| # | Schritt | Abhängigkeit |
|---|---|---|
| 3.1 | **Hebel C: Stem-Pfad mit dem B-Orakel bewerten** — SCNet/SGMSE-W-Stufen neu messen (die früheren Ablehnungen galten den heuristischen Proxies) | B fertig |
| 3.2 | Ggf. Trenn-Modelle feintunen (Kette-6-artiger Job) + C-Rollout über B3 | B + Messung |

## Reihenfolge-Begründung

- **B zuerst anstoßen** (0.1), weil Panel-Zeit der längste serielle Pfad ist —
  sie läuft parallel zu allem anderen.
- **A nach den Kurz-Jobs**, weil A Tage belegt und die Kurz-Jobs sonst
  unbestimmt warten müssten.
- **C zuletzt**, weil es das B-Orakel als Bewertungskriterium braucht —
  ohne B wäre C nur eine Wiederholung der bereits abgelehnten W-Stufen-Messung.

## Risiken & ehrliche Grenzen

- **A-Determinismus:** deterministisches Sampling kostet etwas generative
  Qualität (§G5-Kompromiss); eine bewusste §G5-Ausnahme für einen klar
  abgegrenzten Pfad wäre eine Grundsatzentscheidung des Nutzers.
- **B-Label-Qualität:** Das Orakel ist nur so gut wie die Panel-Daten
  (MUSHRA n≥30 je Bedingung ist die dokumentierte Mindestanforderung).
- **A-Plausibilität ≠ Wiederherstellung:** Der Prior erzeugt das
  musikalisch Plausibelste, nicht das Original — das Never-worsen-Gate bleibt
  die harte Sicherheitsgrenze.
