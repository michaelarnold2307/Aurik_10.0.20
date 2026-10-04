# Symphonia — Instrumentalrestaurierung

Symphonia ist das Gegenstück zu Cantus für Instrumentalstems. Es restauriert
`drums + bass + other` nach der Separation und vor KIM-Inst.
Bei rein instrumentalen Importen ohne erkannten Gesang wird der komplette Input
als Instrumentalstem an Symphonia geführt; die Gesangserkennung darf diesen Pfad
nicht sperren.

## Architektur

- MERT frameweise: Instrumentierung, musikalische Form und Textur
- Rhythmus-Track: lokale Energie und positive Hüllkurvendelta für Onsets
- MuQ-MuLan: globaler Harmoniekontext
- Multi-Scale Flow-Matching-DiT: tiefe Harmonien und lokale Transienten

Der Torch-ROCm-Kern ist der Primärpfad. ONNX darf ausschließlich über CPU als
paritätsgeprüfter Ersatzpfad laufen (§III.9 (copilot-instructions.md)). Ein
vollständiger ML-Ausfall führt mit Warning auf den DSP-Ersatzpfad (§V6
(copilot-instructions.md)).

## Daten- und Evidenzvertrag

Training benötigt deterministische Paare aus MUSDB18-HQ-Instrumentalstems;
die saubere Referenz ist `drums + bass + other`, nie der Vocalstem. Synthetische
Degradationen decken Codec, Rauschen, Hall, Clipping, Bitreduktion und
Bandlimitierung ab. Ein produktives Gewicht benötigt vor Aktivierung:

1. vollständiges Pretraining und Fine-Tuning auf getrennten MUSDB-Tracks;
2. Domain-Adaptation auf lizenzierten realen Instrumentalrestaurierungspaaren;
3. Torch-ROCm↔ONNX-CPU-Parität `rel ≤ 1e-3` auf strukturierten Feeds;
4. Instrumental-Listening-Witness und Blindtest-Evidenz ohne Regression.

Bis diese Evidenz vorliegt, bleibt Symphonia korrekt im dokumentierten
DSP-Fallback; der kanonische Plugin-Pfad verlangt dafür den Modell-Zoo-Status
`active`. Es werden keine zufällig initialisierten oder Vocal-Checkpoint-
Gewichte als Instrumentalmodell ausgegeben.
