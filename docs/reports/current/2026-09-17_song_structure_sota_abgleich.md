# Songaufbauanalyse — SOTA-Abgleich & Upgrade (2026-09-17)

> Arbeitsauftrag 2026-09-17: Ist die Songaufbauanalyse auf maximaler
> SOTA-Ausbaustufe? Deckt sich das Analyseergebnis mit einer unabhängigen
> Analyse des Songaufbaus?
> Referenzmaterial: Testkünstlerin (Schlager), voller Song (225,3 s @ 48 kHz, Export des
> überwachten Laufs `output/supervised_run/elke_225s_supervised_v1020.wav`).

## Befund VOR dem Upgrade

`SongStructureAnalyzer.analyze_structure` (volle Länge, PANNs 0,5):

| Zeit | Label (alt) |
|---|---|
| 0,5–42,4 s | **intro** (42 s „Intro“ — faktisch Strophe 1) |
| 42,4–60,9 s | verse |
| 60,9–98,3 s | **bridge** (37 s „Bridge“ — faktisch Strophe 2) |
| 98,3–116,6 s | verse |
| 116,6–155,1 s | verse |
| 155,1–218,7 s | verse |
| 218,7–225,3 s | outro |

**0 Chorus, 0 Klimax** — trotz klarem, wiederkehrendem Refrain-Motiv.

## Unabhängige Analyse (Chroma-Wiederholungs-Scan, 8-s-Fenster, Kosinus ≥ 0,70)

- **Refrain-Motiv (~8 s) wiederholt sich bei ≈44 s, ≈100 s, ≈156 s und ≈192 s**
  (Fenster-Paare: 43,9–51,9 ↔ 99,7–107,7 / 155,6–163,6 / 191,5–195,5).
- **Intro/Outro-Klammer**: 8,0 s ↔ 215,4 s (gleiches Material).
- Energieprofil steigt monoton zum Ende (Peak 195–210 s) → Klimax am Schluss.

**Fazit: Das alte Analyseergebnis deckte sich NICHT mit dem tatsächlichen
Songaufbau.** Ursachen:

1. **Keine Wiederholungs-Evidenz** — Labels waren reine Position/Energie-Heuristik.
   Ein Refrain ist per Definition der WIEDERKEHRENDE Abschnitt; ohne
   Wiederholungs-Nachweis ist Chorus-Erkennung unmöglich.
2. **Intro/Outro-Heuristik etikettierte das GESAMTE erste Segment** (42 s) als
   „intro“ und das gesamte drittletzte als „bridge“ — nur wegen Position.
3. **Klimax über Segment-MITTELWERT** der Energie — lange Segmente (63 s)
   verwässern den Peak → nie erkannt.
4. **Segment-Mittelwert-Chroma ist schlüsseldominiert** (alle paarweisen
   Korrelationen ≈ 0,65–0,73, gemessen) — Wiederholung zeigt sich erst auf
   **Fenster-Niveau** (8 s).

## Upgrade (deterministisch, rein librosa/numpy, §G5)

- `_window_repetition_counts()`: 8-s-Chroma-Fenster (Schritt 4 s), Kosinus ≥ 0,70,
  Match über Segment-Grenzen ⇒ „wiederholt“; ≥ 2 wiederholte Fenster ⇒ Chorus-Kandidat.
- Label-Logik: wiederkehrend + Energie **>** Median ⇒ `chorus`; wiederkehrend + leise ⇒ `verse`;
  einmalig + kontrastierend ⇒ `bridge`; Intro/Outro nur für KURZE Rand-Segmente (≤ 15 % der Dauer).
- Klimax: `chorus` mit Segment-p90-RMS ≥ 98 % des Song-Maximums (Spec §2.52b: „Energy-Peak“).

## Befund NACH dem Upgrade (gleiches Material)

| Zeit | Label (neu) |
|---|---|
| 0,5–42,4 s | verse |
| 42,4–60,9 s | **chorus** (Refrain ≈44 s) |
| 60,9–98,3 s | verse |
| 98,3–116,6 s | **chorus** (Refrain ≈100 s) |
| 116,6–155,1 s | verse |
| 155,1–218,7 s | **chorus + KLIMAX** (Refrain ≈156/192 s, Energie-Peak) |
| 218,7–225,3 s | outro |

**Deckung mit der unabhängigen Analyse: vollständig** (alle 4 Refrain-Positionen
chorus, Klimax am Energie-Peak, Intro/Outro-Klammer korrekt). Laufzeit 4,83 s /
225 s = 1,29 s/min ≤ 2 s/min-Budget (§2.52b).

## Wirkung auf die Pipeline

`get_strength_scalar` (§2.52b-Tabelle) erhält jetzt ECHTE chorus-Segmente:
Refrains bekommen die schützenden Skalare (NR × 0,85, Kompression × 0,70) —
vorher liefen die Skalare praktisch nur für „verse“ (der Default-Fall).

## Analogie-Korrektur 2026-09-17 (zweite Iteration): kanonische SSM-Grenzen

Dieselbe Fehlerklasse fand sich AUCH bei der Grenz-Erkennung selbst: Der
§2.52b-Analysator nutzte eine agglomerative k-Heuristik (1 Grenze / 30 s),
während die definierende Evidenz — die Novelty-Kurve der Self-Similarity-
Matrix (Foote 2000) — bereits im §2.17-MusicalStructureAnalyzer implementiert
war. Zwei Analysatoren, zwei Methoden, potenziell widersprüchliche
Sektions-Karten in EINEM Lauf (§V7 (copilot-instructions.md)).

**Korrektur:** `backend/core/dsp/ssm_segmentation.py` ist jetzt die EINE
kanonische SSM-Methode (Checkerboard-Novelty, Gauß-Glättung, Peak-Picking);
§2.17 delegiert (verhaltensidentisch, 37 Tests grün), §2.52b nutzt sie primär
(agglomerativer Fallback bleibt). Zusätzlich: Intro/Outro nur noch für das
ERSTE/letzte Segment (die Positionsregel `relative_pos ≥ 0,85` verschluckte
den 156-s-Refrain als „outro“).

Ergebnis Test-Track-225s: **12 evidenz-basierte Segmente** (statt 7
Heuristik-Segmente) — Intro 0–7,5 s (Klammer mit dem Outro-Material),
Refrain-Segmente ≈39,5/96/129,5/137/151,5/180,5 s decken alle vier
unabhängig belegten Refrain-Positionen ab, Klimax am Energie-Peak;
Laufzeit unverändert 1,28 s/min ≤ Budget. Neuer Test:
`test_ssm_boundaries_detect_aba_transitions` ([A|B|A] → Übergänge erkannt).

## Verbleibende SOTA-Lücke (dokumentiert, GPU-gebunden)

- **ML-Boundary-Detektor** (MSAF/SALAMI-Klasse oder MERT-basierter
  Boundary-Modelle, 2024/25) als optionale Verbesserung ÜBER der jetzt
  kanonischen SSM-Methode — benötigt Modell-Gewichte + GPU-Port
  (Muster: BSR-Torch-ROCm).
- **Beat-synchrones Downbeat-/Bar-Tracking** (Spec §2.52b erwähnt Beat-Tracking;
  heute nicht implementiert) — librosa-beat als DSP-Zwischenschritt möglich
  (CPU, Folge-Slice).
- **Chunk-Modus**: `analyze_structure` läuft pro 30-s-Chunk (k=2) statt einmal
  auf der ganzen Länge — Segment-adaptive Stärke ist im Chunk-Modus grob.
  Rezept: Struktur EINMAL vor dem Chunk-Loop auf einem dezimierten
  Ganz-Song-Probe berechnen und wiederverwenden (Folge-Slice, siehe Roadmap).

## Tests

- `tests/unit/test_song_structure_analyzer.py`: 18 Tests grün, davon neu:
  `test_ssm_boundaries_detect_aba_transitions` (kanonische SSM-Methode),
  `test_assign_label_chorus_from_repetition` (Entscheidungslogik),
  `test_window_repetition_counts` (Fenster-Evidenz [A|B|A] ⇒ Segmente 0/2),
  `test_deterministic` (§G5 (GEBOTE.md)).
- `tests/unit/test_musical_structure_analyzer.py`: 37 Tests grün
  (Delegation verhaltensidentisch).
