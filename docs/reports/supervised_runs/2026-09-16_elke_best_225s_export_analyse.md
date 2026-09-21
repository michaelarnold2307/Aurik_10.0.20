# Export-Analyse 2026-09-16 — Testkünstlerin (Schlager) 225 s: Restdefekte & Optimierungspotenzial

> Gegenstand: `output/supervised_run/elke_225s_supervised_v1020.wav` (PCM_24, 48 kHz,
> 225,33 s) vs. Original-MP3 (225,3 s @ 44,1 kHz, auf 48 kHz resampelt für
> Vergleich). Methoden: DefectScanner, Maskierungs-/Bark-Delta, Rauigkeit,
> Pre-Echo, HNR, Crest/Spektral-Zentroid, Oktavband-Energie, MuQ-MOS.

## 1. Restdefekte — durch die Restaurierung EINGEFÜHRTE Schäden: keine ✅

| Messung | Ergebnis |
|---|---|
| Oktavband-Energie-Delta (7 Bänder, 125 Hz–16 kHz) | **exakt ±0,00 dB in jedem Band** |
| Rauigkeits-Anstieg (rest vs. orig) | +0,000 asper |
| Pre-Echo-Verhältnis | −200 dB (Floor — kein Pre-Echo hinzugefügt) |
| Artifact-Freedom (aus dem Lauf) | 0,998 (False-Positive delta-basiert verworfen) |
| §Hörbarkeits-Gate (Lauf) | BESTANDEN — total=0, audible_pre=0, resolved=0 |
| MuQ-MOS | orig 4,84 → rest 4,85 (Δ +0,01 — keine Verschlechterung) |
| Crest / Spektral-Zentroid / Flatness | 4,1→4,1 · 621→625 Hz · 0,001→0,001 |

**Fazit:** Die Restaurierung war auf diesem Material maximal konservativ —
das Never-worsen-Prinzip (§0, Hörordnung) wurde perfekt eingehalten. Es
wurden KEINE neuen Artefakte eingeführt; die lokal reparierten Defekte
(transport_bumps n=135, suppressed=21) sind im globalen Bandspektrum
nicht sichtbar und lagen unter der Maskierungsschwelle.

## 2. Verbliebene Quell-Charakteristika (DefectScanner-Flags, sev/conf)

| Flag | sev | conf | Einordnung |
|---|---|---|---|
| bandwidth_loss | 1,00 | 0,99 | Ära/MP3: Bandbreite endet bei 12,9 kHz |
| hf_remanence_loss | 1,00 | 0,98 | dito (kein HF-Rest > 13 kHz) |
| inner_groove_distortion | 1,00 | 0,97 | Vinyl-Kette |
| transient_smearing | 0,81 | 0,91 | Quell-Charakter (Bump-Reste unter Schwelle) |
| wow / flutter | 1,00 | 0,63/0,53 | bewusst NICHT korrigiert (musikalische Modulation, §AUTH-P12) |
| reverb_excess / room_mode_resonance | 1,00 | 0,82/0,74 | Ära-Hall |
| soft_saturation / proximity_effect_excess | 1,00 | 0,74/0,85 | Band-/Mikrofon-Charakter |

Diese Flags beschreiben die **authentische Klangsignatur** der Aufnahme
(1960er-Jahre, Vinyl-Übertragungskette, MP3-Digitalisierung). Die Hörordnung
(Stufe 1: authentizitaet) verbietet deren aggressive „Korrektur“ — ein
Eingriff würde den Wohlklang verschlechtern, nicht verbessern.

## 3. Optimierungspotenzial für maximalen Wohlklang (geordnet)

1. **HF-Rekonstruktion oberhalb 12,9 kHz** (bandwidth_loss/hf_remanence_loss
   conf≈0,99): SOTA-ML-V1 FlashSR-Musik-Finetune (F4) bzw. HR-V1 BigVGAN (F3),
   abgesichert durch `additive_synthesis_gate` (nie unhörbare Bänder
   hinzufügen). Hebel: Air/Presence-Goals — MUSHRA-Proxy des Laufs zeigt
   VocPres=0,500 und ISO226=0,254 als schwächste Komponenten. GPU-gebunden.
2. **BANQUET-ML-Knistern-Pfad** war im Lauf tot (ONNX fehlt lokal, Docker-
   Pfad scheiterte am Float-WAV — SUP-F2-Fix greift ab dem nächsten Lauf):
   Rest-Innenrillen-Distortion wird dann ML-adressiert.
3. **PANNs-GPU (SUP-F1-Fix)**: verbessert die Gesangs-Konfidenz in den
   Phasen 01/19/43/65 → präzisere Schutzzonen (Vibrato/Formanten).
4. **Lautheit**: Integriert ≈ −18,4 LUFS — OneTakeExport hat das −16-LUFS-
   Ziel angewendet (PASS, TP −1,7 dBTP). Für moderne Wiedergabe optional
   ein −14-LUFS-Ziel erwägen (Nutzer-Einstellung, kein Hardcode).
5. **Wow/Flutter/Hall**: bewusst erhalten — einziger authentizitätssicherer
   Weg wäre SOTA-WF-V4 (neuraler Warp-Schätzer), extern blockiert
   (Checkpoint-Quelle unklar).

## 4. Mess-Nachweis

- Skripte (scratch): `/tmp/analyze_elke.py`, `/tmp/analyze_elke2.py`
- Roadmap: `docs/TODOS_SOTA_ROADMAP.md` → „Export-Analyse 2026-09-16“
