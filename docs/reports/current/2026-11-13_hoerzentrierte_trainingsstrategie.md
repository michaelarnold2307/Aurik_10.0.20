# Strategie: Hör-zentrierte Trainingserfolge ohne tagelange Fehlversuche

Datum: 2026-11-13 · Status: verbindliche Trainingsstrategie (Roadmap-Pfeiler 8,
Hörordnung Ebene 2, §G13 (copilot-instructions.md)). Kein Flag-Flip.

## 0. Problem — die Evidenz der bisherigen Fehlversuche

| Befund | Konsequenz |
|---|---|
| **D3-2 BigVGAN (D-K3-96):** Tage Training, Gate NICHT BESTANDEN — mit Produktions-Mel HNR −6,56/−6,39/−1,03 dB. Ursache (a) **Mel-Drift**: Training `log(abs-STFT)` (Mittel ≈ −3,2) vs. Produktion `power_to_db(ref=max)` (Mittel ≈ −40,7); (b) Finetune degradiert die Basis auch mit korrektem Mel. | Trainingsziele waren Signal-Nähe im FALSCHEN Format — der Fehler war in Minuten erkennbar, wurde aber erst nach dem Gate sichtbar. |
| **D3-3 FlashSR (D-K3-94):** Tage Training, Gate NICHT BESTANDEN — ΔSDR −4,61 dB, Ursache **Passband-Resynthese**; „mehr Epochen beheben das nicht". | Mehr Rechenzeit heilt kein falsches Rezept. |
| **F10 SingMOS (D-K4-6):** 320-Paar-Kalibrierung am Ceiling, 144/320 Ordnungsverletzungen. | Eine Metrik ohne Sensitivität für die Zielschäden taugt nicht als Trainingsziel. |
| **§G13-Q-Erstbelege (2026-11-12):** 0/65 Klassen erreichen Q ≥ 0,5; verfehlt (Q≈0): compression_artifacts, dynamic_compression_excess, groove_echo, hum, jitter_artifacts, sibilance, sticky_shed_residue, tape_head_clog, tape_head_level_dip, transport_bump. | Die Verbesserungslücke ist jetzt BENANNT — sie ist zugleich die Prioritätenliste fürs Training. |

**Gemeinsames Muster:** Trainingsläufe optimierten Wellenform-Nähe statt
Hörbarkeit, starteten ohne Produktions-Paritätsnachweis, und ihr Scheitern
wurde erst nach Tagen sichtbar.

## 1. Prinzip — das Ohr ist die Zielfunktion, nicht die Wellenform

1. **Trainingsziel = Hörbarkeits-Marge.** Ein Modell ist gut, wenn der Rest
   unter der Maskierungsschwelle − 6 dB liegt (`objective_met`, Hörordnung
   Ebene 2) und die Reparaturquote Q ≥ 0,5 erreicht (§G13) — nicht wenn der
   SDR/Mel-Abstand minimal ist.
2. **Eine Mel-Wahrheit (D-K3-96/§G9).** Trainer und Produktion konsumieren
   exakt `compute_mel_produktion` — keine Kopie, kein anderes Format. Der
   Mel-Drift ist als Klasse verboten, nicht als Entdeckung nach Tagen.
3. **Never-worsen wird Verlust-Term, nicht nur End-Gate.** Jeder Batch trägt
   Kontrollregionen (Δ ≈ 0 erzwingen, §G13(c)) — das Modell lernt
   chirurgisches Eingreifen statt Flächenänderung.

## 2. Fail-fast statt tagelang — drei mechanische Stufen

- **Stufe A — Flugschein-Gate VOR dem GPU-Job (Minuten, CPU-only):**
  Produktions-Parität (Mel-Statistik, `gate_hint`, Domänen-Check der
  Trainingsdaten, Lader fail-closed) + ein 10-min-Rezeptprobe-Lauf auf EINEM
  Track gegen das existierende Produktions-Gate. Kein Flugschein ⇒ kein
  GPU-Job. (Die D3-Gates `verify_bigvgan_d3_gate.py` /
  `verify_flashsr_d3_gate.py` liefern die Mess-Vorlage.)
- **Stufe B — Epoch-Level-Gates im Lauf (Minuten je Prüfung):**
  Alle N Epochen (Start: N=2) läuft das CPU-only-Produktions-Gate auf einem
  Track. Verschlechterung gegen die Basis ⇒ Abbruch mit Befund
  (D-K3-96-Muster: Epoche 15 wäre Epoche 2 gewesen).
- **Stufe C — Budget-Disziplin:** max. 1 GPU-Tag pro Rezept ohne
  Gate-Fortschritt; Zeitbudget-Status (`ergebnis: zeitbudget`, D-K3-87) ist
  der gesunde Abbruch, nicht das Versagen.

## 3. Ungewöhnliche Wege — für maximalen Hör-Erfolg

1. **Residual-Training statt Vollband.** Trainiere nur, was dem Ohr fehlt:
   HF-Residual (D-K3-94-Empfehlung), Defekt-Rest, nicht das ganze Signal.
   Kleinere Modelle konvergieren in Stunden statt Tagen und degradieren die
   Basis nicht (D-K3-96(b)-Lehre).
2. **Rezept-abgeleitete Trainingsdaten je Defektklasse.** Die deterministischen
   Schadens-Rezepte (`erstelle_schadens_proben.py`: clipping/kassette/vinyl/
   mp3_128, Seed 42) + Clean-Originale sind bereits überwachte Paare. Je
   Defektklasse ein **kleiner Per-Klassen-Kopf** (CRNN/DNN-Challenger-Muster
   wie `use_defect_crnn`) — Priorität exakt die Q≈0-Klassen der
   §G13-Erstbelege. Kleine Köpfe = Minuten–Stunden Training, klassenlokale
   Verantwortung, kein Flächenrisiko.
3. **Produktion-im-Loop.** Die Evaluierungs-Funktion des Trainings ist der
   echte Produktionsschritt (z. B. `apply_hr_v1_additive`), nicht ein
   Trainings-Rekonstruktor — damit ist „Trainingserfolg" per Definition
   „Produktionserfolg" (D3-Gate-Muster, jetzt im Loop).
4. **Hörbarkeits-Curriculum.** Trainingsdaten nach Hörbarkeits-Marge
   sortiert (lauteste Defekte zuerst); Fortschritts-Maß ist der
   `objective_met`-Anteil je Epoche, nicht der Loss.
5. **Kontrafaktische Mini-Läufe vor jedem Rezeptwechsel.** Jede
   Rezeptur-Hypothese (z. B. „mehr Epochen", „größere Batch") wird zuerst als
   30-min-Lauf auf einem Track gegen das Gate getestet — die D-K3-94-Lehre
   („mehr Epochen beheben das nicht") kostet dann 30 Minuten, nicht 30 Epochen.
6. **Gegner-Paare als Stopp-Signal.** Die Schadens-Rezepte dienen im Training
   zugleich als Gegner: verschlechtert ein Checkpoint die Gegner-Klasse, wird
   er verworfen — never-worsen wird Teil der Modellauswahl.

## 4. Messlatte & Verantwortung

- Jeder Trainingslauf **bucht** ein §G13-Q-Ergebnis + `objective_met` je
  Klasse — kein „Trainingserfolg" ohne Hör-Evidenz (Metriken sind Zeugen,
  Hörordnung §7).
- Stärke-/Schwellen-Änderungen aus Trainingsergebnissen laufen ausschließlich
  über das §4-Verfahren (`flag_freigabe.py`).
- Die Iteration-1-Prioritätenliste sind die Q≈0-Klassen der Erstbelege
  (Abschnitt 0) — jede Klasse hat damit Ziel, Messung und Verantwortung.

## 5. Erste Umsetzungs-Schritte (autonom, CPU-only)

1. Flugschein-Gate als Skript + Pre-Commit (Produktions-Parität + Rezeptprobe).
2. Epoch-Level-Gate-Verdrahtung in die Trainer (N=2, Abbruch bei Basis-Degradation).
3. Per-Klassen-Datensatz-Generator aus den Schadens-Rezepten (Klasse = Scanner-Typ).
4. Erster Per-Klassen-Kopf auf der Q≈0-Spitzenklasse (compression_artifacts).
