# D3-2-Gate — BigVGAN-F3-Finetune gegen die Basis

**Datum:** 2026-11-11 · **Beleg:** `2026-11-11_bigvgan_d3_gate.json` ·
**Werkzeug:** `scripts/verify_bigvgan_d3_gate.py` ·
**Test:** `tests/unit/test_verify_bigvgan_d3_gate.py` ·
**Register:** D-K3-96

## 1. Auftrag und Befund

D3-2 war die letzte offene D3-Verifikation: Der Finetune
`output/f3_bigvgan/best.pt` (Epoche 7) war fertig, aber es existierte **kein
Gate**. Die im Trainer hinterlegte Auflage lautete „MUSDB-ΔSDR ≥ +2 dB via
`scripts/validate_hr_v1.py`“. Das ist aus zwei Gründen nicht messbar:

1. `validate_hr_v1.py` misst **af und HNR**, keinen SDR.
2. Ein Mel-Vocoder erzeugt eine **neue Phase**; der zeitliche SDR liegt für
   Basis und Finetune selbst nach Peilung bei ≈ 0 dB (gemessen: 0,0 dB bei
   bestem Lag). SDR ist damit kein gültiges Gate für diese Modellklasse.

Das Gate wurde deshalb neu als **falsifizierbarer A/B-Test gegen die Basis**
definiert und am 2026-11-11 CPU-only gemessen. Ergebnis: **NICHT BESTANDEN** —
sowohl mit Produktions-Mel als auch mit Trainings-Mel. Die Basis bleibt aktiv;
es gab **keinen Flag-Flip** und keine Verdrahtung des Finetunes.

## 2. Verfahren

- **Pfad:** echter Produktionsschritt `apply_hr_v1_additive` (5-%-Deckel,
  `additive_synthesis_gate`, 100-ms-Rampen) — kein Nachbau.
- **Tracks:** 3 MUSDB18HQ-Testtracks, deterministisch (Seed 42), je 10 s,
  Segmentstart = energiereichstes 10-s-Fenster der ersten 60 s. Damit fällt
  kein Messfenster in Stille (erster Lauf mit 3-s-Segmenten verworfen, siehe
  `2026-11-11_bigvgan_d3_gate_verworfen_messfenster.json`).
- **Zwei Mess-Ebenen:**
  - `roh_norm` — Rohsynthese des Modells, RMS-auf-Eingang normiert. Das ist
    die **Modellgüte** vor dem additiven Schutz; Entscheidungsgrundlage.
  - `fenster_ausgang` — der tatsächliche Produktionsmix als Sicherheitszeuge.
    Er ist erwartungsgemäß fast eingangsfest (das Gate gibt nur 4–22 Bänder
    frei) und kann allein keine Modellentscheidung tragen.
- **Toleranzen** (dokumentiert, §G5 (copilot-instructions.md)):
  af ±0,02 · HNR ±0,5 dB · Wohlklang-JND 0,05 (musik_mos/defektlast).
- **Urteil:** Never-worsen gegen die Basis auf **allen** Tracks und ein
  Vorteil über der Toleranz auf **≥ 2/3** Tracks; ohne Produktionswirkung
  (`applied=false`) kein Bestehen — fail-closed.
- **Checkpoints (am Artefakt gemessen):**
  - Basis `models/bigvgan/bigvgan_v2.pth` —
    `d9fe7ec6bd0b44ed9d66973d5012d8181c1570b01e5c72df51973e241dccd357`
  - F3 `output/f3_bigvgan/best.pt` —
    `25642f9571edff764a344a1a8f40bcc89a009acd28f4fe6f64e2cb9a37dc11db`

Aufruf:

```bash
CUDA_VISIBLE_DEVICES="" .venv_aurik/bin/python -B \
  scripts/verify_bigvgan_d3_gate.py --tracks 3 --seconds 10 --seed 42
```

## 3. Ergebnisse

`roh_norm`, je Track; Δ = F3 minus Basis (bei defektlast ist kleiner besser).

| Track | Variante | af | HNR dB | musik_mos | defektlast | Δaf | ΔHNR | Δmos | Δdef | Vorteil |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Al James — Schoolboy Facination | Basis | 1,0000 | −1,19 | 4,9264 | 0,0074 | — | — | — | — | — |
| | F3 prod-Mel | 0,5455 | −7,75 | 4,6280 | 0,4897 | −0,4545 | −6,556 | −0,2984 | +0,4823 | nein |
| | F3 train-Mel | 0,4466 | −5,05 | 4,6597 | 0,4860 | −0,5534 | −3,864 | −0,2667 | +0,4786 | nein |
| Bobby Nobody — Stitch Up | Basis | 0,6591 | −2,44 | 4,7809 | 0,4632 | — | — | — | — | — |
| | F3 prod-Mel | 0,6591 | −8,83 | 4,6635 | 0,4950 | ±0,0000 | −6,390 | −0,1174 | +0,0318 | nein |
| | F3 train-Mel | 0,4935 | −0,40 | 4,7741 | 0,3918 | −0,1656 | +2,034 | −0,0068 | −0,0714 | ja |
| The Easton Ellises — SDRNR | Basis | 0,5455 | −6,20 | 4,7876 | 0,4709 | — | — | — | — | — |
| | F3 prod-Mel | 0,5455 | −7,23 | 4,4627 | 0,4932 | ±0,0000 | −1,034 | −0,3249 | +0,0223 | nein |
| | F3 train-Mel | 0,4935 | −1,64 | 4,8048 | 0,4975 | −0,0520 | +4,556 | +0,0172 | +0,0266 | ja |

**Gate-Urteil:**

- `f3_prod`: **NICHT BESTANDEN** — Never-worsen auf 3/3 Tracks verletzt,
  Vorteil 0/3.
- `f3_trainmel`: **NICHT BESTANDEN** — Vorteil 2/3 (HNR +2,03/+4,56 dB,
  defektlast −0,071 auf Track 2), aber **af-Regression auf 3/3**
  (−0,5534/−0,1656/−0,0520) und Track 1 verliert auf allen vier Zeugen.
- `gesamt`: `bestanden=false`, `deploybar_heute=false`,
  `kandidat_mit_mel_adapter=false`.

## 4. Befunde (Ursachen, nicht Symptome — §V7 (copilot-instructions.md))

1. **Mel-Drift ist strukturell.** Der Trainer füttert
   `bigvgan.meldataset.mel_spectrogram` (`log(abs(STFT))`, `center=False`,
   Mittel ≈ −3,2, Std 2,1). Die Produktion füttert
   `librosa.power_to_db(ref=np.max)` (Mittel ≈ −40,7, Std 16,6, Bereich
   −80…0). Mit Produktions-Mel erzeugt F3 1,5–2,2× so viel RMS wie die Basis,
   peakt bei 1,0 und verliert 6,4–6,6 dB HNR. Das ist kein Tuning-, sondern
   ein **Train-/Inference-Vertragsbruch** (Klasse D-K3-85: das Artefakt wird
   mit einem anderen Eingabe-Vertrag gemessen als trainiert).
2. **Auch mit Trainings-Mel schlägt F3 die Basis nicht.** Es gewinnt HNR auf
   2/3 und defektlast auf 1/3, verliert aber af auf 3/3 und Track 1 auf allen
   Zeugen. Der Finetune (13/30 Epochen, Early Stop bei Patience 5, 0,5-s-Crops,
   nur A1 + MR-STFT, kein Diskriminator) **degradiert** die Basis. Ein
   HNR-Gewinn bei af-Verlust ist kein Never-worsen-Erfolg.
3. **Der Produktionsschutz kann keinen Vorteil erzeugen.** Das additive Gate
   gibt 4–22 Bänder frei und hält den Mix eingangsfest; die Fenster-Metriken
   sind von der unveränderten Eingabe dominiert. Der Schutz verhindert Schaden,
   aber ein schlechterer Synthesizer wird durch ihn nicht besser.
4. **Das alte Gate war unerfüllbar.** ΔSDR ≥ +2 dB gegen die Basis ist für
   einen phasen-unabhängigen Mel-Vocoder mathematisch nicht erreichbar; der
   `gate_hint` im Trainer wurde auf das neue, messbare Gate korrigiert und
   durch einen Test gepinnt.

## 5. Entscheidung und offener Rest

- **Keine Aktivierung.** `BIGVGAN_V2_HR_ACTIVATED` bleibt auf dem Basispfad;
  F3 wird nicht verdrahtet. Es gab keinen `flag_freigabe.py`-Lauf, weil keine
  Flag-Änderung beantragt wird.
- **D3-2-Verifikation ist geschlossen** (negatives Ergebnis, belegt). Der
  offene Rest ist eine **Rezept-/Modellentscheidung**, keine Messlücke:
  Produktions-Mel im Training, Diskriminator/längere Schedule, oder Basis
  behalten. Das ist GPU- und Hör-Instanz-gebunden (B3/§4) und nicht autonom
  entscheidbar.
- Für jeden künftigen F3-Checkpoint ist `scripts/verify_bigvgan_d3_gate.py`
  das Gate; ein grünes Ergebnis ersetzt **nicht** die Hör-Instanz.

## 6. Geänderte Artefakte

| Artefakt | Änderung |
|---|---|
| `scripts/verify_bigvgan_d3_gate.py` | neu: D3-2-Gate (Rohsynthese + Produktionszeuge, fail-closed) |
| `tests/unit/test_verify_bigvgan_d3_gate.py` | neu: 14 Vertragstests (Toleranzen, 2/3-Vorteil, fail-closed, Determinismus, CPU-Guard) |
| `scripts/train_bigvgan_f3.py` | `gate_hint` + Docstring auf das messbare Gate korrigiert; kein ΔSDR-Versprechen mehr |
| `.github/SOTA_DEFICIT_REGISTER.md` | D-K3-96 |
| `docs/TODOS_SOTA_ROADMAP.md`, `docs/SOTA_HANDOVER.md` | D3-2-Status nachgezogen |
