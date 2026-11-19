# D3-3-Gate: FlashSR-Musik-Finetune besteht sein eigenes Gate nicht (2026-11-10)

**Klasse:** K1/K2 (Modell-Bewertung, kein Klangpfad-Eingriff) · **Register:** D-K3-94 ·
**Kein Flag-Flip, kein Deploy, kein Training.**

## 1. Auftrag und Herkunft

`models/flashsr/report.json` trägt seit dem Lauf vom 2026-11-10 die eigene
Auflage: `gate_hint = "MUSDB-ΔSDR ≥ +2 dB (Bandbreiten-Extension) NACH dem Lauf
messen"`. Der Trainer misst nur seinen Validierungs-Loss (A1 + MR-STFT), nicht
den Hörtrag gegenüber der 16-kHz-Basis. Diese Messung holt das nach — mit dem
**trainierten** Checkpoint, nicht mit dem deployten ONNX.

Werkzeug: `scripts/verify_flashsr_d3_gate.py` (neu, 9 Vertragstests).
Aufruf:

    CUDA_VISIBLE_DEVICES="" HIP_VISIBLE_DEVICES="" \
        .venv_aurik/bin/python -B scripts/verify_flashsr_d3_gate.py \
        --tracks 3 --seconds 20 --mit-pretrained

CPU-only und fail-closed (sichtbare GPU ⇒ Exit 2), damit die laufende Kette 2
(D3-2 BigVGAN) unangetastet bleibt. Seed 42, feste Track-Auswahl, `_sdr` und
`_segmentwise` aus `benchmark_bwe_candidates.py` importiert (eine Wahrheit für
SDR und Segmentierung).

## 2. Ergebnis (3 Tracks × 20 s, MUSDB18HQ-Test, Seed 42)

Checkpoint `models/flashsr/best.pt` (Epoche 15, sha256 `97403806…`), Basis
`models/flashsr/models/upsampler.pth` (sha256 `62c70874…`).
Beleg: `docs/reports/current/2026-11-10_flashsr_d3_gate.json`.

| Variante | mean ΔSDR | min ΔSDR | Gate (≥ +2 dB, never-worsen) |
| --- | --- | --- | --- |
| `raw` (Konsum-Vertrag: Modellausgabe) | **−4,61 dB** | −10,60 dB | **NICHT BESTANDEN** |
| `residual` (Mess-Option: Basis-Passband + modelliertes HF) | −0,03 dB | −0,09 dB | nicht bestanden (neutral) |
| `pretrained` (Referenz, deploytes Modell) | −6,44 dB | −11,93 dB | nicht bestanden |

Je Track (`raw`): SDR 17,40 / 16,75 / 17,80 dB gegen Basis 22,41 / 22,40 /
20,33 dB.

**Der Finetune ist besser als das Vortrainings-Modell (+1,8 dB), aber er
verliert gegen „nichts tun" (Bandlimit) um 2,1–6,2 dB.**

### 2b. Robustheitslauf (6 Tracks × 30 s, unabhängige Stichprobe)

Weil 3 Tracks dünn sind, derselbe Lauf mit `--tracks 6 --seconds 30` (anderes
Track-Subset, Seed 42): `docs/reports/current/2026-11-10_flashsr_d3_gate_2.json`.

| Variante | mean ΔSDR | min ΔSDR | Spanne je Track |
| --- | --- | --- | --- |
| `raw` | **−9,04 dB** | −27,72 dB | −3,06 … −16,07 dB |
| `residual` | −0,08 dB | −1,22 dB | −0,02 … −0,26 dB |
| `pretrained` | −10,67 dB | −35,26 dB | −5,41 … −22,94 dB |

Auch hier ist die Peilung artefaktfrei (Lag 0/−1 Sample, Gain 0,92–1,00), und
die Passband-SDR der Ausgabe (12,1–18,2 dB) liegt weit unter der Basis
(18,3–24,4 dB). **Zwei unabhängige Stichproben, beide deutlich unter der
Schwelle** (−4,6 dB und −9,0 dB): das Urteil hängt nicht an der Track-Auswahl.
Der Unterschied zwischen den Läufen (3 vs. 6 Tracks) ist Material-Streuung,
kein Widerspruch — die Richtung und die Größenordnung sind gleich.

## 3. Diagnose — kein Mess-Artefakt

Die Lehre aus D-K3-85 („erst den Pfad prüfen, dann urteilen") ist hier
eingebaut; alle drei Artefakt-Klassen sind ausgeschlossen:

| Artefakt | Messwert | Bewertung |
| --- | --- | --- |
| Peilung/Latenz | beste Verzögerung **−1 Sample** (≈ −0,02 ms) | ausgeschlossen |
| Pegel/Skala | optimaler Gain **0,990–0,995**, RMS-Verhältnis ≈ 1,0 | ausgeschlossen |
| Bandlage | HF-SDR der Schätzung **−0,49 / −0,22 / −0,01 dB** | HF ist praktisch unkorreliert |

**Die Ursache sitzt im Passband:** die Passband-SDR der Modellausgabe liegt bei
17,4/16,8/17,8 dB gegen 22,4/22,4/20,3 dB der Basis — das Modell re-synthetisiert
das gesamte Signal (Vocoder-Klasse), statt das Passband zu erhalten. Deshalb
ist auch die `residual`-Variante neutral: sie erhält das Passband
konstruktionsbedingt (bit-nah) und fügt nur ein Hochband hinzu, das nichts
beiträgt (−0,03 dB).

## 4. Was das bedeutet (und was nicht)

- **F4/Roadmap-Ziel „ΔSDR ≥ +2 dB" ist mit diesem Modell und diesem Rezept
  nicht erreicht** — auch nicht mit mehr Epochen, weil der Fehler nicht im
  Training, sondern in der Aufgabenstellung liegt (16k→48k-Sprach-BWE als
  Vollsignal-Resynthese auf Musik).
- **Kein Produktionsrisiko heute:** der Pfad ist nicht im Signalpfad.
  `HFExtender` (einziger Konsument von `FlashSRPlugin.process`) hat außerhalb
  seines eigenen Moduls **keine Aufrufstelle**; `unified_restorer_v3.py:25866`
  instanziiert `HybridNVSR()` nur für die Status-Payload (`active: True`), ohne
  Audio zu verarbeiten. Es gibt deshalb **nichts abzuschalten** und keinen
  Flag-Flip.
- **Die `residual`-Variante ist kein Kandidat:** sie ist neutral, nicht
  besser. Sie bleibt als Mess-Option im Skript, **nicht verdrahtet**.
- **Deploymentsperre bleibt:** ein Deploy wäre klangwirksam und liefe über §4 —
  nach dieser Messung gibt es dafür keine Grundlage.

## 5. Nächster Schritt (nicht autonom, weil Modell-/Rezeptur-Entscheidung)

Drei Wege, alle mit GPU-Bedarf und ohne Hörbeleg nicht entscheidbar:

1. **Aufgabe ändern:** HF-Residual *trainieren* (Ziel = Hochband-Differenz,
   Passband einfrieren) statt Vollsignal-Resynthese — die einzige Variante, die
   konstruktiv never-worsen-fähig ist.
2. **Modellklasse wechseln:** Musik-trainiertes BWE/SR (statt Sprach-BWE).
3. **F4 zurückstellen:** die Rolle bleibt beim deterministischen SBR-Kern in
   Phase 06 (kein ML-Risiko).

Eine Vorauswahl ist eine SOTA-Entscheidung mit Hör-Impact — sie gehört der
Hör-Instanz (§4/B3), nicht diesem Lauf.

## 6. Reproduzierbarkeit

- Report-JSON enthält Checkpoint- und Basis-sha256, Seed, Sekunden, Track-Namen
  und alle Diagnosegrößen.
- Determinismus: Seed 42, keine Zeitquelle in der Auswahl (§G5); CPU-only, damit
  das Ergebnis nicht von der GPU-Auslastung abhängt.
- Grenze: 3 Tracks × 20 s (60 s Material) plus ein unabhängiger Robustheitslauf
  mit 6 Tracks × 30 s (180 s) — zusammen 9 Track-Läufe. Für ein *endgültiges*
  Urteil über die Modellklasse wären mehr Tracks nötig; die Größenordnung
  (−4,6 dB bzw. −9,0 dB, min bis −27,7 dB) lässt aber keinen Grenzfall erwarten.
- Der zweite Lauf überschreibt den ersten **nicht**: `_zielpfad()` vergibt einen
  numerischen Beleg-Namen (`…_2.json`), eine identische Wiederholung bleibt
  idempotent (2 Tests).
