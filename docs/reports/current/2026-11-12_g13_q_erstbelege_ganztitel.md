# §G13-Q-Erstbelege auf Ganz-Titel-Artefakten (2026-11-12)

Werkzeug `scripts/bewerte_defekt_delta.py` (defekt_delta, pegel-gematcht p25,
JND 0,05, `q_ziel_erfuellt` je Defektklasse gegen `REPARATURQUOTE_ZIEL` 0,5).
Artefakte: die 6 Ganz-Titel-Paare (clipping/kassette × 3 Titel,
`~/Musik/schadens_proben_ganz`), Clean-Originale `~/Musik/k398_quellen`
(Längenangleich). Kein Flag-Flip, keine Verdrahtung geändert — reine Messung.

## Befund (9 Belege, 6 Paare)

- **Keine Defektklasse erreicht Q ≥ 0,5** auf diesem Material. Die
  §G13-Verbesserungs-Pflicht ist damit je Defektklasse **gemessen verfehlt**
  bzw. nicht bewertbar — die Lücke ist jetzt benannt statt unbewiesen.
- **Verfehlt (Q ≈ 0, keine Rückgewinnung in den Scanner-Regionen):**
  `compression_artifacts`, `dynamic_compression_excess`, `groove_echo`,
  `hum`, `jitter_artifacts`, `sibilance`, `sticky_shed_residue`,
  `tape_head_clog`, `tape_head_level_dip`, `transport_bump` (je Titel
  variierend); vereinzelt leicht negativ (Q −0,003…−0,011: Restauration
  minimal schlechter als Schaden in diesen Regionen).
- **Nicht bewertbar (Q=None):** die übrigen Klassen — kein messbarer
  Clean-Kontrast in den Regionen (Clean-Delta ≈ 0 ⇒ Q undefiniert, fail-safe).
- **Kontext:** Das Aggregat der Ganz-Titel-Messung bleibt gültig — Jets·clipping
  ist auf dem vollen Titel klar repariert (defektlast 0,3104 → 0,0299,
  Δ −0,2805; musik_mos +0,2556, `2026-11-12_ganze_titel_wirkung.json`). Die
  Per-Klassen-Quote Q ist die strengere §G13-Sicht: sie misst nur die
  Scanner-annotierten Regionen und verlangt die Hälfte des Clean-Deltas.

## Einordnung

§G13 wirkt wie entworfen: der Messrahmen liefert je Defektklasse eine
ehrliche Bilanz statt eines Pauschalurteils. Die nächste Iterationsrunde
(Reparatur-Nachweis je Klasse) hat damit konkrete Ziele: die verfehlten
Klassen sind die Prioritätenliste. Stärke-/Schwellen-Änderungen daraus
laufen ausschließlich über das §4-Verfahren — Q-Befunde sind Zeugen
(Hörordnung §7).
