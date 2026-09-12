# Knistern-Entfernung: ML-Modell-Analyse — kommt das optimale Modell zum Einsatz?

> Status: Analyse + Root-Fix (2026-09-12) · Bezug: `phase_09_crackle_removal.py`,
> `quality_mode.py`, BANQUET (2023), DeepFilterNetV3
> Datum: 2026-09-12

## 1. Fragestellung

> „Warum kommt bei der Entfernung des Knisterns nicht BANQUET zum Einsatz —
> ist es hier nicht geeignet? Kommt das optimale ML-Modell zum Einsatz oder
> wäre banquet.onnx das besser geeignete Modell für Vinyl?"

## 2. Was für Knistern zur Verfügung steht (Ist-Zustand)

| Pfad | Modell/Verfahren | Spezialisierung | Rolle |
|---|---|---|---|
| phase_03 | DeepFilterNetV3 (ONNX) | Breitband-NR für Noise/**Click** | ML-Rauschentfernung (Mix) |
| **phase_09** | **BANQUET** (`banquet_vinyl_final.onnx`, 92 MB) | **Blind-Audio-Enhancement für VINYL** (Knistern/Impulse/Fläche) | **designierter ML-Primärpfad für Vinyl-Knistern** |
| phase_09 | SpectralDecrackler + `surgical_repair._repair_crackle` (Median >4 kHz) | Deterministischer DSP-Fallback | Ersatzpfad, sample-exakt |

BANQUET ist also **das** für Vinyl trainierte Modell (BANQUET 2023: blind audio
quality enhancement auf Vinyl-Korpus) — es IST das besser geeignete Modell für
Vinyl-Knistern gegenüber DeepFilterNet (Breitband-Denoiser ohne
Impuls-Spezialisierung) und gegenüber dem DSP-Median (generischer
Kurzfenster-Glätter ohne Texturmodell).

## 3. Befund: Warum BANQUET trotzdem nie lief

Produktionsbefund aus den überwachten Vinyl-Läufen
(`output/supervised_run/rock_1970s_worn.log`, `classical_1960s_hiss.log`):

```
INFO: ✅ BANQUET ONNX geladen: banquet_vinyl_final.onnx
INFO: ✅ Vinyl ML-Chain: BanquetVinylPlugin geladen (92 MB ONNX)
INFO: PLM: Entlade 'BanquetVinyl' (0.80 GB) … released
```

Das Modell wurde **geladen und ungenutzt wieder entladen** — keine einzige
Zeile „BANQUET ML-Modell (Vinyl-Entknacken) aktiv". Ursache (Code-Beweis):

```python
use_banquet = QUALITY_MODE_AVAILABLE and _is_vinyl and is_phase_ml_enabled(9)
```

und in `quality_mode.py`:

```python
_CRITICAL: frozenset[int] = frozenset({3, 23, 24, 29, 55, 66})   # 9 fehlte!
```

`is_phase_ml_enabled(9)` war damit **permanent False** → das Anwendungs-Gate
konnte niemals durchlassen. BANQUET war an der Verdrahtung **dead code**; jede
Vinyl-Knistern-Entfernung lief über den DSP-Ersatzpfad.

## 4. Root-Fix (umgesetzt)

- `quality_mode.py`: Phase 9 in die ML-CRITICAL-Liste aufgenommen
  (`{3, 9, 23, 24, 29, 55, 66}`) mit Produktionsbefund-Kommentar.
- Schutzschichten bleiben aktiv: ML-Speicherbudget (0.8 GB Allocation),
  B11-HF-Rauschfloor-Rollback (Banquet-Schaden sichtbar → Dry),
  bw_loss-Graduierung, Zeugen-Gates — BANQUET kann nur gewinnen oder neutral
  bleiben.

**Neben-Befund (dokumentiert, nicht Teil dieses Fixes):** Dasselbe Gate-Muster
blockiert auch `is_phase_ml_enabled(18)` (VAD im Noise-Gate) — Phase 18 steht
nicht in `_CRITICAL` und nutzt ML damit ebenfalls nie. Falls VAD im Noise-Gate
gewollt ist, gehört 18 analog ergänzt (separate Entscheidung, da ein
VAD-Gate stille Passagen fälschlich absenken kann).

## 5. Antwort auf die Frage

1. **BANQUET ist für Vinyl das besser geeignete Modell** — und es ist als
   Primärpfad verdrahtet; es kam nur wegen des Gate-Defekts nie zur Anwendung.
2. **KIM2/KIM-Inst sind für Knistern ungeeignet:** Sie sind für harmonische
   Klarheit/Brillianz von Stimme/Instrument trainiert (langzeitige
   Spektral-Synthese); 1–5-ms-Impuls-Bursts würden verschmiert statt entfernt.
3. Nach dem Fix gilt: Vinyl-Kette → BANQUET ONNX (Quality/ML-Modus), sonst
   DeepFilterNet (Noise/Click) bzw. DSP-Fallback — jede Stufe mit
   Never-worsen-Schutz. Verifikation erfolgt im nächsten überwachten
   Vinyl-Lauf (Log-Marker: „BANQUET ML-Modell (Vinyl-Entknacken) aktiv —
   ONNX Direct Inference" + `method=ml_banquet_vinyl_onnx_direct`).
