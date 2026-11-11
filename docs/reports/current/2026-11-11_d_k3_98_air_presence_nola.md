# D-K3-98 — Air-Presence-Enhancer: NOLA-Fenster-Mismatch + Hard-Clip behoben

Datum: 2026-11-11 · Status: **geschlossen** (Code + Vertragstests) · Kein Flag-Flip

## 1 Befund (vor dem Fix gemessen)

`backend/core/dsp/air_presence_enhancer.py` analysierte mit **symmetrischem**
`np.hanning(2048)` und synthetisierte über `scipy.signal.istft(window="hann")`
(**periodisch**) bei 75 % Overlap (n_fft 2048, hop 512); normiert wurde mit
`Σwin` (symmetrisch) gegen scipys `Σw²`-Konvention.

Messung (1-s-Sinus 440 Hz, system-python, deterministisch):

| Größe | vorher |
| --- | --- |
| STFT→ISTFT-Roundtrip, rel. Fehler | **8,93e-2** (9 %) |
| scipy-Warnung | `NOLA condition failed` |
| 440-Hz-Ton (keine Luftband-Energie ⇒ Maske 0), max. Abweichung | **0,25** (volle Amplitude) |
| 12-kHz-Sinus 0,99 mit `strength=1.0` | **23.804 Samples auf 1,0** geklippt |

Beide Aufrufstellen waren betroffen: `phase_19_de_esser.py` (Air-Band) und
`stem_level_restorer.py` (§SLR-1e2b).

## 2 Fix (Auflage a/b/c aus dem Register)

- **(a) Identisches Fenster:** `_hann_periodic(n_fft)` (scipy-Konvention
  `fftbins=True`) in Analyse **und** Synthese; das symmetrische `np.hanning`
  entfällt vollständig.
- **(b) scipy-Konvention:** `_stft_frames` nutzt `scipy.signal.stft` mit
  `window=_hann_periodic`, `detrend=False`, `boundary="zeros"`, `padded=True`
  (Frames mit `win/Σwin` skaliert); `_istft_frames` invertiert mit demselben
  Fenster. Signale kürzer als ein Fenster werden auf `n_fft` genullt (§V6,
  kein Crash).
- **(c) Eingangs-Decke statt Hard-Clip:** neu
  `_begrenze_auf_eingangsdecke(orig, enhanced, ceiling)`. Der **Zusatz**
  (`enhanced − orig`) wird linear zurückgenommen, bis
  `max|orig + k·delta| ≤ min(0,999, peak_in)`; die Basis bleibt unangetastet.
  Liegt der Eingang selbst über der Decke, ist `k = 0` ⇒ bit-identischer
  Passthrough (Never-worsen, §0).

## 3 Verifikation

| Größe | vorher | nachher |
| --- | --- | --- |
| Roundtrip 1 kHz, rel. Fehler | 8,93e-2 | **1,78e-8** |
| NOLA-Warnung | ja | **nein** |
| 440-Hz-Ton, max. Abweichung | 0,25 | **1,49e-8** |
| 12-kHz 0,99, `strength=1.0` | 23.804 Samples @ 1,0 | Peak **0,9900**, **0** Samples @ 1,0 |
| Kurzsignal 500 Samples | Crash/0 Frames | finit, Form erhalten |

- Tests: `tests/unit/test_air_presence_enhancer.py` **10/10** — 4 neue
  Vertragstests (COLA-Rekonstruktion ohne Ripple, bitnahes Passthrough ohne
  Luftband-Energie, Eingangs-Decke statt Hard-Clip, Kurzsignal);
  `tests/unit/test_stem_level_restorer.py` 17/17;
  Phase-19-Suite (`test_phase_19_de_esser.py`,
  `test_phase19_de_esser_profile.py`, `test_phase19_crest_sculpting.py`)
  81/81.
- Performance: 3-min-Stereo 2,54 s (≈ 71× Echtzeit für die Stufe; das
  32×-Gesamtbudget bleibt unkritisch). Die alte Variante brauchte 1,37 s nur
  für STFT/ISTFT — bei 9 % Rekonstruktionsfehler.
- `ruff check` und `py_compile` clean.

## 4 Grenzen / offen

- Die **Wirkung** auf die Clipping-Verschlechterung der D-K3-95-Rezepte ist
  damit **nicht** bewiesen: der Enhancer läuft nur im Stem-Pfad, `phase_19`
  ist zusätzlich witness-guarded. Der eigene Messzyklus (Schadens-Rezepte
  clipping/kassette) nach dem Reproduzierbarkeits-Nachweis D-K3-97 §10 steht
  aus.
- Kein Flag-Flip, keine Verdrahtung geändert; Aufrufstellen unverändert.
