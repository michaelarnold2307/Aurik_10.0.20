# Q≈0-Metrik-Kanal-Forschung — Ergebnis (2026-11-14)

**Frage:** Welche der §G13-Q≈0-Klassen (Erstbelege: Q ≈ 0 auf dem
Ganz-Titel-Gegner-Material) sind mit einem gain-invarianten Skalar-Kanal
messbar — so wie der Sättigungskanal (1/Crest) die Clipping-Klassen
messbar gemacht hat (Q_saett 0,70–0,75)?

**Messung:** `scripts/q0_witness_forschung.py` — je Klasse deterministische
Stichprobe (Seed 42, Ziel 300 Regionen) aus `data/klassen_paare/manifest.jsonl`
(16.779 Regionen mit clean/schaden/restauriert-WAVs), Richtungskriterium:
Median(schaden − clean) > JND 0,02 UND > 50 % Regionen über JND.
Beleg: `2026-11-14_q0_witness_forschung.json`.

## Ergebnis je Klasse×Feature

| Klasse | Feature | Δ Median (schaden−clean) | > JND | Urteil |
| --- | --- | --- | --- | --- |
| **jitter_artifacts** | **Seitenband-Ratio** | **+0,180** | **98,3 %** | ✅ **KANAL AKTIVIERT** (Schema /4) |
| sibilance | Crest 4–12 kHz | +0,022 | 100 % | ⚠️ marginal (Δ ≈ JND) — nicht aktiviert |
| groove_echo | Cepstral-Echo-Peak | 0,0 | 0,7 % | ✗ unsichtbar |
| groove_echo | Spektral-Ripple-Tiefe | 0,0 | 0 % | ✗ unsichtbar |
| sticky_shed_residue | Impulsdichte | 0,0 | 0 % | ✗ unsichtbar |
| tape_head_level_dip | Hüllkurven-Modulation | 0,0 | 42,8 % | ✗ Richtung unklar |
| transport_bump | LF-Stoß-Energie | ≈ 0 | 2,1 % | ✗ unsichtbar |

## Konsequenz

1. **Jitter-Kanal aktiviert:** `jitter_kanal` (Seitenband-Energie-Verhältnis
   um die stärksten Spektrallinien) ist kanonisch in
   `backend/core/dsp/wohlklang_zeuge.py`, Report-Only-Zweitkanal im
   Wohlklang-Zeugen (alle drei UV3-Pfade) und in `defekt_delta` Schema /4
   (`jitter_*`-Felder, Kanal-JND 0,02, `reparaturquote_q_jitter`,
   Aggregat `q_jitter_ziel_erfuellt`). /3 ist veraltet (fail-closed).
   **Kein Gate, kein Flag** — Zeuge, kein Richter (§G191 (GEBOTE.md)).
2. **Ehrliche Negativbefunde:** groove_echo (4.072 Regionen) ist mit den
   zwei physikalisch naheliegendsten gain-invarianten Features nicht
   messbar — die Echo-Struktur ist auf diesem Material nicht als
   Cepstral-/Ripple-Signal separierbar. sibilance-Crest ist richtungskorrekt,
   aber Δ nur knapp über JND — keine Fake-Aktivierung.
3. **Nächster Schritt:** für groove_echo/sticky_shed wäre ein
   rezept-/klassen-supervidiertes Witness (kleiner Detektor auf den
   Regionen-Paaren) der Weg — kein Wellenform-Kopf, kein Gate.
