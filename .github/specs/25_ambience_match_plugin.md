# Spec 25: Ambience-Match — Natürliche Raumhülle für restauriertes Material

> **Version:** Aurik 10.1.0 · **Scope:** Wohlklang
> **Status:** Entwurf
> **Erstellt:** 2026-09-22 · **Abgeschlossen:** —

## Prämisse

Jede aggressive Restaurierung (Denoise, Dereverb, Declip, Spectral Repair) entfernt
zwangsläufig die psychoakustische Grund-„Hülle“ einer Aufnahme: Raumtönung, Saal­
rauschen, Bandgrund. Das Ergebnis ist messtechnisch sauber, klingt aber tot und
klinisch — und genau diese Künstlichkeit nimmt das menschliche Ohr als Artefakt wahr
(Hörordnung, Ebene 1: Audibility an der Maskierungsschwelle statt Mess-Null;
lexikografische Wohlklang-Ordnung: Natürlichkeit vor technischer Reinheit).
Aurik hat aktuell kein Plugin dieser Klasse: Die Bestandsprüfung (Plugins, Phasen
01–66, DSP-Module) ergibt keine Ambience-Synthese; die Rausch-/Raumbehandlung der
Phasen 03/18/20/26 entfernt nur, stellt nichts wieder her. Ziel: eine deterministische,
unter der Maskierungsschwelle kalibrierte Raumhülle, die die Natürlichkeit des
Ausgangsmaterials wiederherstellt, ohne das Nutzsignal zu verändern.

## Maßnahme

Neues DSP-Plugin `plugins/ambience_match_plugin.py` (Vorbild: BW Harmonic Exciter —
reiner DSP, kein Modell, keine Gewichte, keine Latenz jenseits eines 3-s-Segments).
Es schätzt die spektrale Hüllkurve des Quellmaterials aus den Stille-/Zwischen­
segmenten (Minimal-Statistik nach dem im Haus etablierten IMCRA-Prinzip, vgl.
Phase 20), synthetisiert daraus eine **raumprofil-treue** Hülle (kein weißes Rauschen)
und mischt sie blend-gesteuert unter das Signal — kalibriert gegen die
psychoakustische Maskierungsschwelle, nicht gegen einen Messwert.

### Implementierung

1. **DSP-Kern:** Profil-Schätzung (Bark-Band-Hüllkurve, zeitlich geglättet) aus
   Stille-Segmenten; Synthese über geseedeten PRNG + Formfilter + optionale
   Faltung mit vermessenen Raum-IRs (Basis-Set intern, deterministisch).
2. **Maskierungskalibrierung:** Pegel je Bark-Band = geschätzte
   Maskierungsschwelle − Sicherheitsabstand σ (Default σ = 6 dB). Damit ist die
   Hülle per Konstruktion unhörbar als Rauschzuwachs, wirkt aber gegen die
   „schwarze Stille“ nach der Restaurierung.
3. **Blend-Garantie:** `blend=0` ⇒ bit-identischer Passthrough (kein Anteil am
   Signal, kein Dither, kein Resampling). Stärke zentral über `global_scalar`
   (§G1 (copilot-instructions.md), §V7 (copilot-instructions.md): keine
   phasen-individuellen Schwellwerte).
4. **Verdrahtung:** Aufruf ausschließlich über `backend/api/bridge.py` (Bridge-Verbot
   §V4 (copilot-instructions.md)); Anwendung als Politur unmittelbar **vor** der
   Glue Stage — die Glue Stage bleibt vorletzte Phase (§III (copilot-instructions.md));
   schaltbar über Materialprofil (Default: an für Vintage-Klassen wie Schellack,
   Kassette, Rumpel-/Hiss-lastige Bänder; aus für bereits „lebendiges“ Material).
   Kein neuer DefectType (§III (copilot-instructions.md): 62 DefectTypes sind fix).
5. **Tests:** Unit-Suite (Passthrough, Determinismus, Invarianten H1–H4), dazu
   GO/NO-GO-Hörentscheidung nach `docs/guides/GO_NO_GO_DECISION_PROTOCOL.md`.
   Implementierungs-PR beachtet Write-Gate, FILE_REGISTRY-Eintrag und Task-Ledger.

### Psychoakustische Invarianten (Hörordnung, Ebene 1 — nie verletzbar)

- **H1 Audibility:** Der Hüllen-Anteil liegt in jedem Bark-Band mindestens σ unter
  der Maskierungsschwelle — es entsteht kein hörbarer Rauschzuwachs (ABX-Nachweis).
- **H2 Timbre-Neutralität:** Die Hülle ist additiv; der Betragsfrequenzgang des
  Nutzsignals ändert sich um ≤ ±0,05 dB je Dritteloktave. Kein EQ-, kein
  Kompressions-Anteil.
- **H3 Kein Verdecken:** In Frequenzbändern mit aktiven Defekt-Markern der
  Pipeline (Defect-Map) wird die Hülle um ≥ 12 dB zusätzlich abgesenkt — Ambience
  darf Defekte nie maskieren (§V7 (copilot-instructions.md): Ursache statt Symptom).
- **H4 Reversibilität:** Hülle und Signal werden getrennt geführt; der Export kann
  jederzeit blend=0 (Originalbitstrom) reproduzieren.

### Maskierungskalibrierung (deterministisch)

Vereinfachtes psychoakustisches Modell: Bark-Skalierung, Ausbreitungsfunktion
(spreading) mit −25 dB/Bark-Flanke, Schwelle aus der Hüllkurve des lokalen Signals.
Alle Konstanten liegen als benannte Kalibrierungskonstanten vor und unterliegen
dem Calibration-Guard (delta-basiert, kein absoluter Hard-Fail). Es gibt genau
einen freien Nutzparameter (σ), alle übrigen folgen dem Modell.

### Determinismus-Vertrag (§G5 (copilot-instructions.md))

- Seed pro Session; **kein** `time.time()`/`random` ohne Seed in der
  Entscheidungslogik.
- Gleicher Input + gleiche Version ⇒ bit-identischer Export, mit und ohne Hülle.
- Zwei identische Läufe ⇒ bit-identische Ausgabe (eigener Determinismus-Test,
  analog `test_full_pipeline_determinism`).
- Pro Song vollständiger State-Reset (§G1 (copilot-instructions.md), §V8
  (copilot-instructions.md)).

### Erfolgskriterium

Binär: (a) Unit-Suite grün — `blend=0` bit-identisch, Determinismus bit-identisch,
Invarianten H1–H4 bestanden; (b) GO/NO-GO-Protokoll absolviert mit positiver
Hör-Entscheidung in mindestens zwei Materialklassen; (c) ABX-Gegenprobe zeigt
keinen signifikanten Rauschzuwachs (p ≥ 0,05).

### Aufwand

14h | **Wohlklang-Wirkung:** ⬆️⬆️⬆️

### Risiken & Gegenmaßnahmen

| Risiko | Eintrittswkt. | Gegenmaßnahme |
|--------|---------------|---------------|
| Hülle wird hörbar (Rauschzuwachs) | Mittel | Maskierungskalibrierung H1 + σ-Marge + GO/NO-GO-Gate |
| Klingt synthetisch („Digitalrauschen“ statt Raum) | Mittel | Raumprofil-treue Synthese + IR-Faltung; kein weißes Rauschen |
| Verdeckt Defekte statt sie zu reparieren | Niedrig | H3 (Defect-Map-Kopplung) + Hörordnung-Konfliktregel |
| Drift zwischen Hör-Instanz und Metrik | Niedrig | Hörordnung: Metriken sind Zeugen, die Hör-Instanz entscheidet — nie gegen Ebene 1 |
| Materialabhängige Fehlkalibrierung | Niedrig | Profil-Schätzung je Song (§G1 (copilot-instructions.md)), kein globaler Default-Pegel |

---

## Ziel-Matrix

| Ziel | Betroffen? | Wie? |
|------|-----------|------|
| Hörbarer Wohlklang | Ja (primär) | Stellt die natürliche Raumhülle wieder her; beseitigt den „toten“ Klang nach der Restaurierung |
| Systemische Stabilität | Nein | Additives DSP-Plugin ohne Modell/ONNX/GPU; kein Einfluss auf bestehende Phasen |
| Nachhaltige Wartbarkeit | Indirekt | Reiner DSP ohne Gewichte — keine Vendor-, keine Update-Last; Kalibrierung über benannte Konstanten |

> **Regel:** Eine Maßnahme adressiert GENAU EIN Ziel als primäres Ziel.
> Die anderen beiden dürfen als sekundäre Ziele profitieren, aber nicht
> im Fokus stehen. Keine Maßnahme adressiert alle drei gleichzeitig.
