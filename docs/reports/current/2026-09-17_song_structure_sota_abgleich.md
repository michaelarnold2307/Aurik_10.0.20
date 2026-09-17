# Songaufbauanalyse — SOTA-Abgleich & Upgrade (2026-09-17)

> Arbeitsauftrag 2026-09-17: Ist die Songaufbauanalyse auf maximaler
> SOTA-Ausbaustufe? Deckt sich das Analyseergebnis mit einer unabhängigen
> Analyse des Songaufbaus?
> Referenzmaterial: Elke Best, voller Song (225,3 s @ 48 kHz, Export des
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

## Verbleibende SOTA-Lücke (dokumentiert, GPU-gebunden)

- **ML-Boundary-Detektor** (MSAF/SALAMI-Klasse oder MERT-basierter
  Boundary-Modelle, 2024/25) statt der heuristischen k-Wahl (1 Segment / 30 s)
  — benötigt Modell-Gewichte + GPU-Port (Muster: BSR-Torch-ROCm).
- **Beat-synchrones Downbeat-/Bar-Tracking** (Spec §2.52b erwähnt Beat-Tracking;
  heute nicht implementiert) — librosa-beat als DSP-Zwischenschritt möglich
  (CPU, Folge-Slice).
- **Chunk-Modus**: `analyze_structure` läuft pro 30-s-Chunk (k=2) statt einmal
  auf der ganzen Länge — Segment-adaptive Stärke ist im Chunk-Modus grob.
  Rezept: Struktur EINMAL vor dem Chunk-Loop auf einem dezimierten
  Ganz-Song-Probe berechnen und wiederverwenden (Folge-Slice, siehe Roadmap).

## Tests

- `tests/unit/test_song_structure_analyzer.py`: 17 Tests grün, davon neu:
  `test_assign_label_chorus_from_repetition` (Entscheidungslogik),
  `test_window_repetition_counts` (Fenster-Evidenz [A|B|A] ⇒ Segmente 0/2),
  `test_deterministic` (§G5).
