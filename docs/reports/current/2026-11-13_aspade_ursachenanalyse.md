# A-SPADE-Ursachenanalyse: Warum das ΔSDR-Gate scheitert (2026-11-13)

**Kontext:** D-K4-1-Aktivierungsvoraussetzung für `use_aspade_declipper` ist ein
ΔSDR-Gate auf geclipptem Real-Material. Der Vollpfad-Lauf
(`scripts/verify_aspade_declipper_gate.py`, echte k398-Ganz-Titel,
30-s-Schlimmstfenster, SDR best-lag-alignt) ergab **0/3 Titel SDR-Gewinn**.
Diese Analyse trennt „Witness zu konservativ“ von „Modell ohne Gewinn“.

## Methode

Je Titel das am stärksten geclippte 30-s-Fenster (Clip-Schwelle 0,85, da die
k398-Rezepte bei ±0,90 klippen). Gemessen wurden die **Roh-Kandidaten** direkt
(ohne den Phase-07-Never-worsen-Guard): `_declip_pchip` (Amtsinhaber),
`declip_aspade_class` (Klassen-Challenger hinter dem Flag) und der ONNX-APPLADE-
Zweig (`plugins/aspade_declipper_plugin.py`). SDR gegen das Clean-Original,
FFT-Korrelation (±50 Samples Lag), deterministisch.

## Befund

| Titel | SDR Input | Δ PCHIP | Δ Klasse | Δ ONNX | SDR-Gewinn |
| --- | ---: | ---: | ---: | ---: | --- |
| Boogie Woogie Mama | −9,988 | 3e-05 | 0,0406 | 0,0945 | keiner |
| Träumer (Dreamer) | −13,722 | 0,0 | 0,0 | 0,0 | keiner |
| Me I'm So Beautiful | −11,827 | 3e-05 | 0,0 | **0,3966** | keiner |

Δ = maximale Spitzenänderung gegenüber dem Input; alle SDR-Werte bleiben auf
3 Dezimalen identisch mit dem Input.

## Interpretation

1. **Der Witness ist NICHT das Problem.** Selbst der rohe ONNX-Kandidat (ohne
   Phase-Guard) verändert auf Boogie (0,094) und Jets (0,397) die Wellenform
   sichtbar — die geclippten Plateaus werden angefasst —, aber der SDR gegen das
   Clean-Original bleibt identisch. Die Rekonstruktion bewegt sich nicht in
   Richtung Original: **das Modell hat auf diesem Sättigungs-Clipping ganzer
   Musik-Mischungen keinen messbaren Gewinn.** Die Never-worsen-Ablehnung in
   Phase 07 (Boogie/Jets bit-identisch) ist damit korrektes Verhalten des
   Musikschutzes, kein Defekt.
2. **Zweiter, echter Defekt gefunden (Modellreparatur nach Resampling):**
   Die APPLADE-Erkennung läuft bei 16 kHz (`resample_poly 48k→16k`); die
   Anti-Aliasing-Filterung dämpft die ±0,90-Plateaus unter die interne
   Schwelle — auf Träumer fand der Zweig deshalb **gar kein** Clipping
   (Δ = 0,0). Auf Boogie/Jets griff er (Plateaus blieben über der Schwelle),
   ohne Gewinn. Das ist eine echte Erkenntnis-Schwäche des 16-kHz-Pfads bei
   schwelle-nahen Clip-Pegeln.
3. **PCHIP bleibt formtreu** (Δ ≤ 3e-05, D-K3-97 bestätigt) — der Amtsinhaber
   ist auf diesem Material ebenfalls wirkungslos, aber unschädlich.

## Konsequenz

- **Aktivierungsvoraussetzungen bleiben NICHT erfüllt.** Flag bleibt AUS.
- Nächster Schritt ist Modell-seitig, nicht Witness-seitig: A-SPADE-Finetune
  auf Musik-Sättigungs-Clipping (ganze Mischungen, nicht nur Vocals) ODER
  Ersatz durch einen Musik-declippenden Kandidaten; dabei die 16-kHz-
  Erkenntnisschwelle gegen schwelle-nahe Clips härten (Detektion in der
  48-kHz-Domäne vor dem Resampling).
- Beleg: `docs/reports/current/aspade_declipper_gate.json` (Vollpfad) und
  diese Analyse (Roh-Kandidaten).
