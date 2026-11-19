# Musik-Declip-Kopf: Durchbruch-Beweis (2026-11-13, erzwungen)

**Kontext:** D-K4-1 — A-SPADE (MUSDB-Vocal-trainiert) hatte auf dem k398-
Sättigungs-Clipping **0/3 Titel SDR-Gewinn** (Ursachenanalyse: Modell ohne
Gewinn auf Musik-Mischungen + 16-kHz-Erkennungsschwäche). Dieser Versuch
erzwingt den Durchbruch mit einem auf **Musik-Mischungen** trainierten
Waveform-Declip-Kopf.

## Methode

- **Daten:** 61 MUSDB18-HQ-Train-Mischungen, 48-kHz-downmixt, mit dem
  Produktionsrezept `clipping` beschädigt (dieselbe Rezeptur wie die
  k398-Gegner-Proben) + Clean-Originale. Zusätzlich Clip-Kulturen aus den
  k398-Clean-Titeln (Level 0,80/0,85/0,90, Peak-normiert).
- **Modell:** identity-initialisierter Residual-Conv1d-Encoder-Decoder
  (~60k Parameter, 345 KB — `build_model` aus `scripts/train_klassen_kopf.py`),
  Fenster 8.192 Samples, L1 + MR-STFT, Seed 42, 40 Epochen × 400 Schritte,
  Batch 16, RX 7900 XTX (~4 min/Lauf).
- **Evaluation:** die 30-s-Schlimmstfenster der drei k398-Clipping-Titel
  (SDR best-lag-alignt gegen das Clean-Original, FFT-Korrelation);
  Overlap-Add mit Rand-Padding (die naive OLA hatte eine Rand-Explosion:
  Division durch Hanning²→0 — behoben).
- **Leak-Freiheit:** Leave-one-out — je Titel ein Modell, dessen Training
  den Clean-Inhalt DIESES Titels nie gesehen hat (nur MUSDB + die anderen
  beiden k398-Titel).

## Befund (leak-frei)

| Titel | SDR Input | SDR Kopf (LOO) | Δ |
| --- | ---: | ---: | ---: |
| Boogie Woogie Mama | −9,99 dB | **+5,60 dB** | **+15,6 dB** |
| Träumer (Dreamer) | −13,72 dB | −5,30 dB | +8,4 dB |
| Me I'm So Beautiful | −11,83 dB | −4,93 dB | +6,9 dB |

In-Domäne (MUSDB): −9,09 dB → +27,0 dB. Der Kopf invertiert die komplette
Rezeptur einschließlich des 5×-Gain-Boosts (Ausgangs-RMS 0,104 = Clean-RMS
0,103).

## Konsequenz und nächste Schritte

1. **Erster Kandidat mit echtem, generalisierendem Declip-Gewinn auf dem
   D-K4-1-Gegenmaterial.** Kein Flag-Flip — der Kopf ist (noch) nicht in
   Phase 07 verdrahtet.
2. **Vor der Verdrahtung zwingend:** Clip-Masken-Blend (der Kopf darf nur in
   geclippten Regionen wirken — auf sauberem Material würde er den
   Rezeptur-Gain rückgängig machen und Musik dämpfen) + Never-worsen-Witness,
   dann offizieller ΔSDR-Gate-Lauf mit `verify_aspade_declipper_gate.py`
   und das §4-Verfahren (blindes A/B + Hörordnungs-Sign-off) für den Flag-Flip.
3. **Leak-Kompromiss dokumentiert:** die Modelle sahen die k398-DAMAGE-Proben
   (geclippt) nicht als Trainingsziel — nur die Clean-Kulturen der jeweils
   anderen Titel; der LOO-Test schließt Clean-Memorierung aus.
