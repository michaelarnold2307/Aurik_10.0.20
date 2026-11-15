# D-K3-92 geschlossen — `use_bw_v5` ist kein No-Op (indirekter Dispatch)

**Datum:** 2026-11-15 · **Lücke:** D-K3-92 („Schalter ohne Lesestelle" /
stiller No-Op) · **Status:** geschlossen, **kein Flag-Flip**, bit-identisches
Bestandsverhalten.

## 1. Befund (Ursache statt Symptom, §V7)

Das Flag-Wirkungs-Gate vom 2026-11-10 führte `use_bw_v5` als **einzige
Ausnahme**: „keine Produktions-Lesestelle, BW-v5 hängt an `_use_nvsr`". Diese
Diagnose ist **falsch** — sie war ein Falsch-Negativ des Gates:

```
backend/core/music_model_flags.py:211   flag_map = {..., "bw": use_bw_v5, ...}
backend/core/music_model_flags.py:226   use_music = flag_map.get(model_key, False)
plugins/bw_reconstructor_plugin.py:65   _p = _resolve_model_path_flag("bw")
backend/core/phases/phase_06_frequency_restoration.py:1298
                                        BWReconstructorStage(cutoff_hz=None, ...)
```

`phase_06` konstruiert die BW-Stufe **ohne** `model_path`; der Plugin-Konstruktor
löst den Pfad über `resolve_model_path("bw")` auf, und diese Funktion liest
`use_bw_v5` über ihre Dispatch-Tabelle **innerhalb** der Flag-Datei. Das Gate
schloss die Flag-Datei als Lesestelle aus und sah nur direkte Namens-Loads —
der indirekte Dispatch blieb unsichtbar.

## 2. Messung: der Schalter wählt Artefakt **und** Algorithmus-Zweig

Gemessen 2026-11-15, CPU-only, ohne Flag-Flip (`use_bw_v5` bleibt `False`):

| Schalter | aufgelöster Pfad | SHA-256 | ONNX-Inputs |
| --- | --- | --- | --- |
| AUS | `models/bw_reconstructor/bw_reconstructor.onnx` | `4a4e4f58b9679abe…` | `input` (Mel-Domäne) |
| AN | `models/bw_reconstructor/bw_reconstructor_v5.onnx` | `7888e65678234fa3…` | `waveform`, `cutoff` |

Die unterschiedliche Input-Signatur schaltet in
`plugins/bw_reconstructor_plugin.py` den Zweig `_is_v5` →
`_reconstruct_v5_waveform` (Waveform- statt Mel-Domäne). Ein AN/AUS-Lauf misst
damit **zwei verschiedene Modelle**, nicht den Pfad mit sich selbst.

## 3. Fix (Gate, nicht Klang)

- **`scripts/flag_wirkung_gate.py`:** neue Stufe „indirekte
  Dispatch-Lesestelle" — ein Flag, das in einer Dispatch-Tabelle im
  Flag-Modul steht, deren Tabelle dieselbe Funktion per `.get(...)`/Subscript
  liest und zurückgibt, wirkt im **Aufrufer** der Funktion. Der Aufrufer
  entscheidet über Stufe A/B; Literal-Schlüssel werden gefiltert
  (`resolve_model_path("bw")` zählt nur für `use_bw_v5`), dynamische Schlüssel
  werden konservativ zugerechnet und im Report als `[dyn]` markiert.
  Die Ausnahmeliste `AUSNAHMEN` ist **leer**.
- **`scripts/b3_komposition_suche.py`:** die W-8-Begründung ist korrigiert —
  W-8 ist produktion-wirksam und bleibt nur deshalb aus der Suche, weil der
  Stufe ein **Ladeweisen-Zeuge** fehlt (BW-Pfad läuft nur bei Rolloff ≥ 7 kHz
  und nicht-tapeartigem Material ⇒ ein Lauf ohne Zeuge könnte Basis gegen
  Basis vergleichen). Flip weiterhin nur §4 + Hör-Sign-off (D-K1-5).

## 4. Belege (reproduzierbar)

```
python3 -B scripts/flag_wirkung_gate.py --verbose
  → Flag-Wirkungs-Gate — 34 Schalter geprüft
  → ✅ 34 Schalter, alle wirksam (0 Verstöße)   [vorher: 1 Ausnahme use_bw_v5]

python3 -B -m pytest tests/unit/test_flag_wirkung_gate.py \
    tests/unit/test_dk0_k1_rest_luecken.py \
    tests/unit/test_f9_identity_gate_und_w8.py -q
  → 34 passed
```

Neue Vertragstests (5): Dispatch-Tabelle zählt als Lesestelle; Schlüssel
filtert fremde Flags; dynamischer Schlüssel zählt konservativ; Tabelle ohne
Produktionsaufruf bleibt Verstoß; Tabelle ohne Rückgabe zählt nicht.
Zusätzlich: `use_bw_v5` AUS→Basis / AN→v5 und Quelltext-Pin auf den
Produktions-Ladepfad.

## 5. Ehrlicher Rest (nicht D-K3-92, gebucht an D-K1-5)

Der **Zielpfad** des Schalters zeigt auf das A1-**durchgefallene**
Alt-Artefakt (`7888e656…`, HF-Gain 0,8079 < 1,02); das A1-bestandene
Retrain-Artefakt liegt in `models/bw_reconstructor/retrain_a1_20261025/`
(SHA `548c0127…`) und ist **untracked** (`/models/*` ist in `.gitignore`).
Ein §4-Flip würde heute das falsche Artefakt laden. Die Korrektur ist eine
**Modell-Deployment-Entscheidung** (16-MB-Artefakt ins Repo tragen oder
kanonischen Pfad ersetzen) und deshalb **nicht autonom** — als Blocker an
D-K1-5 vermerkt. Kein Flag-Flip, keine Klangänderung in diesem Lauf.
