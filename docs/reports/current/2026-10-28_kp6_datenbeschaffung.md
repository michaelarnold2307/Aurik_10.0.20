# KP-6 Datenbeschaffung — BANQUET Real-Vinyl-Finetune (F12) + D2 Leerrillen-Noise-Bank

> Auftrag (2026-10-28): Fehlende KP-6-Daten autonom über lizenzfreie
> Online-Quellen beschaffen, verifiziert und passgenau. Beleg-Manifest:
> `docs/reports/current/2026-10-28_kp6_vinyl_noise_bank_manifest.json`.

## Was fehlte (SOTA-Trainingskonzept §4 KP-6, Welle D)

1. **Echte Vinyl↔Reissue-Paare ≥30** (Crackle/Hiss/Wow/Oberflächenrauschen) —
   lokal nur 16 damaged/6 clean, alle synthetisch CC0.
2. **Leerrillen-Noise-Bank (D2)** — „Noise-Bank = 0, größter Hebel".
3. **Provenienz-Hygiene:** SHA-Kette Quelle → Artefakt (§III.13).

## Beschafft (data/vinyl_noise_bank/ — 9 Dateien, je mit SHA-256 + Probe)

| Rolle | Herkunft | Lizenz-Status |
| --- | --- | --- |
| Leerrillen-Geräusch (4× 5 s, Ein-/Auslauf) | Great 78 / archive.org (2 Nadelabtastungen, 96 kHz/24bit) | PD pre-1930 (Kollektionsannahme) — **§9/7-Freigabe erforderlich** |
| Echt beschädigt (2× volle Seiten) | Great 78 / archive.org | wie oben |
| PD-Mark-Audio (3) | archive.org, `licenseurl=publicdomain/mark/1.0` (pro Item maschinell belegt) | **lizenzverifiziert** |

**Eignungsnachweis (gemessen):** Lead-in/Lead-out der Nadelabtastungen tragen
HF-Anteil >6 kHz von 0,16–0,82 bei RMS 0,003–0,008 — echtes Träger-/Rillen-
Rauschen (Crackle/Hiss), nicht synthetisch. Quell-URLs + IA-sha1 + lokale
sha256 + Probe-Werte je Datei im Manifest.

## Rezept für ≥30 Paare (Hybrid laut Konzept zulässig)

Echtes PD-Material (obige beschädigte Seiten + PD-Mark-Musik) + **echtes**
Defekt-Overlay aus der Leerrillen-Bank → Paare erzeugen mit
`scripts/train_banquet_vinyl_finetune.py` (Default-Korpus austauschbar).
Die Bank skaliert durch weitere Great-78-Seiten (187.034 Items verfügbar,
Download-Muster im Manifest dokumentiert).

## Offene Punkte (nicht autonom)

- **§9/7 Lizenzpolitik (User):** Great-78-Items tragen kein maschinelles
  `licenseurl` (nur Kollektions-/pre-1930-Annahme) — Freigabeentscheidung
  nötig, bevor das Material ins Training geht.
- Echte **Vinyl-LP**-Paare (post-1948, Wow-Charakteristik) sind über
  lizenzfreie Quellen kaum deckend verfügbar; die Hybrid-Paare + die
  78er-Trägerfläche decken Crackle/Hiss/Oberfläche, Wow teils.
