# D-K3-98-Folgemessung: Air-Presence-Wirkung auf den Schadens-Rezepten

Datum: 2026-11-12 · Werkzeug: `scripts/messe_air_presence_wirkung.py` ·
Belege `2026-11-12_air_presence_wirkung.json` (+ Stem-Variante). Kein Flag-Flip, kein Deploy.

## 1 Auftrag und Vorbedingung

`2026-11-11_d_k3_98_air_presence_nola.md` §4: „Die **Wirkung** auf die
Clipping-Verschlechterung der D-K3-95-Rezepte ist damit **nicht** bewiesen …
Der eigene Messzyklus (Schadens-Rezepte clipping/kassette) nach dem
Reproduzierbarkeits-Nachweis D-K3-97 §10 steht aus."

Die Vorbedingung ist erfüllt: der Reproduzierbarkeits-/§V8-Leck-Nachweis
(solo 3× bit-identisch `be0e87cb…`, Kontext-Lauf wieder `be0e87cb…` = solo)
ist seit 2026-11-11 geschlossen (Handover §2 Zeile 13).

## 2 Messdesign

- Proben: die 6 degradierten D-K3-95-Proben (3 Titel × Rezepte
  `clipping`, `kassette`, 5 s Stereo 48 kHz, `~/Musik/schadens_proben`).
- Ebene **Kern** (Produktions-DSP `enhance_air_presence`, strength 0,15,
  Listening-Witness-Gate, Eingangs-Decke): `applied`, Witness-Kennzahlen,
  Decken-Vertrag `max|out| ≤ min(0,999, max|in|)`, RMS-Delta, Delta-Metriken
  musik_mos/defektlast (JND 0,05).
- Ebene **Stem-Pfad** (`--stem-pfad`): Produktions-Trigger
  `StemLevelRestorer.restore` (panns_singing 0,8) — `air_presence_used` +
  Zeuge der echten Produktionsstufe.
- Deterministisch (§G5 (copilot-instructions.md)), CPU-only, kein Zufall.

## 3 Befund (Kern-Ebene, 6/6 Proben)

| Titel · Rezept | decke_in | decke_out | rms_delta |
| --- | --- | --- | --- |
| Boogie · clipping | 0,899994 | 0,899994 | **0,0** (bit-identisch) |
| Träumer · clipping | 0,900024 | 0,900024 | **0,0** (bit-identisch) |
| Jets · clipping | 0,900024 | 0,900024 | **0,0** (bit-identisch) |
| Boogie · kassette | 0,700287 | 0,700287 | 4,8e-06 (STFT-Numerik) |
| Träumer · kassette | 0,759827 | 0,759827 | 4,3e-05 (STFT-Numerik) |
| Jets · kassette | 0,848895 | 0,848895 | 3,8e-05 (STFT-Numerik) |

- `witness_ok` 6/6, `decke_ok` 6/6, `nie_schlechter_als_jnd` 6/6 — der
  Never-worsen-Vertrag hält auf allen Proben.
- **Aber: die Wirkung ist Null.** Alle Deltas liegen bei 0 bzw. im
  STFT-Numerik-Rauschen (≤ 4e-5), alle Metrik-Deltas exakt 0,000.

## 4 Ursache (Code-verifiziert, kein Raten)

`air_presence_enhancer.py::_begrenze_auf_eingangsdecke` (D-K3-98c) setzt
`ceiling = min(0,999, peak_in)`. Für reale Pegel (< 0,999) ist die Decke also
**identisch zum Eingangs-Peak**. Die Luftband-Anhebung ist am globalen
Peak-Sample (breitbandige Transienten tragen Luftband-Energie) positiv ⇒
`min((ceiling − orig)/delta) = 0` ⇒ `k = 0` ⇒ der **gesamte** Zusatz wird
linear auf Null zurückgenommen. Der Enhancer ist damit auf realer Musik
strukturell ein No-Op — solange der Peak nicht zufällig rein tieffrequent ist.

Konsistenz: Die D-K3-98-Verifikationstabelle („12-kHz 0,99: Peak 0,9900,
0 Samples @ 1,0") ist mit `k = 0` erfüllbar — sie hat die Anhebung selbst nie
belegt. Der Befund ist also kein Widerspruch zu D-K3-98, sondern genau die
dort als unbewiesen markierte Lücke.

Konsequenz für D-K3-95/97: Der Enhancer kann die Clipping-Verschlechterung
(0,478 → 0,716) **nicht** verursacht haben — er tut auf diesen Proben nichts.
Das bestätigt die D-K3-97-Ursachenkette (auch nicht der Declipper) und
verengt die Suche weiter auf die Pegel-/Entscheidungskette.

## 5 Befund (Stem-Pfad, 6/6 Proben)

`StemLevelRestorer.restore` (panns_singing 0,8, CPU): 6/6 gelaufen,
`air_presence_used = true` und Zeugen-`applied = true` auf **allen** Proben —
während der Kern exakt dasselbe Signal zurückgibt (Abschnitt 3). Die
Produktionsstufe **meldet Anwendung, obwohl nichts passiert** (No-Op-Artefakt
mit aktivem Zeugen, D-K3-85-Klasse: „Basis gegen Basis gemessen"). Drei
Proben enden mit `success = false` (vocal-gate rollback vqi_below_floor bzw.
miipher_model_not_loaded unter CPU-only — §V6-nicht-blockierend) — am
Enhancer-Zeugen ändert das nichts.

## 6 Grenzen

- Ein Fix der Decken-Logik (z. B. per-Frame-Decke oder `ceiling = 0,999`
  statt `peak_in`) wäre **klangwirksam** ⇒ nur über §4-Verfahren mit
  Hör-Sign-off — bewusst NICHT eigenmächtig geändert.
- Die Stem-Pfad-Ebene (Produktions-Trigger) läuft als Ergänzungsbeleg
  (`--stem-pfad`); die Kern-Ebene ist der maßgebliche Befund, weil der
  Enhancer-Eingang im Stem-Pfad derselben Decken-Logik unterliegt.
- 5-s-Fenster, kein Qualitätsurteil über ganze Titel (gleiche Grenze wie
  D-K3-95).
