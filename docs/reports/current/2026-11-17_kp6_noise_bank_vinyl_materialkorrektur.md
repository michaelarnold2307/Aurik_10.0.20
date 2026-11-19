# KP-6/D2 Materialkorrektur — Vinyl-Rauschbank: Schellack raus, echtes Vinyl rein (CC0)

> Auslöser: User-Korrektur (2026-11-17) — „vinyl gab es 1929 noch nicht".
> Der Altbestand der `data/vinyl_noise_bank/` (KP-6, 2026-10-28) stammte aus
> **Great-78-Aufnahmen (Schellack, pre-1930)**. Schellack (78/min, grobe
> Rille, anderes Material und Abtastverhalten) ist für ein **Vinyl**-Modell
> (Mikrorille, LP-Ära ab ~1948) das akustisch falsche Trägermaterial.

## Befund Altbestand (2026-10-28, jetzt entfernt)

| Datei | Problem |
| --- | --- |
| 4× `leerrille_*` (Great 78) | **Schellack**, nicht Vinyl; zudem Lizenz nur „PD pre-1930 (Kollektionsannahme)" — **§9/7-Freigabe stand offen und wurde nie erteilt** |
| 2× `echt_beschaedigt_*` (163/178 s) | volle **78er-Musikseiten** — Musik, kein Rauschen; dieselbe offene §9/7-Freigabe |
| 3× `pdmark_*` | Musik-Dateien (kein Rauschen), 2 davon nicht lesbar, 1 ohne HF-Anteil |

## Ersatz (data/vinyl_noise_bank/, Schema aurik.vinyl_noise_bank/2)

**6 CC0-Vinyl-Rauschaufnahmen von freesound.org** (Filter `license=CC0 1.0`,
je Datei per Sound-Seite maschinell verifiziert). Entscheidender
Lizenz-Unterschied: Es sind **Recordist-eigene Aufnahmen** des
Rillenrauschens — der Aufnehmende kann CC0 rechtsgültig vergeben; **kein
fremdes Musikwerk ist betroffen**, §9/7 wird vollständig umgangen.

| Datei | Rolle | Dauer | Eignungsnachweis |
| --- | --- | --- | --- |
| `vinyl_fs_837986_Artem_Zaitsev.wav` | Oberflächenrauschen | 60,1 s | Breitband, tonal_peak_med 3,7 — reinstes Rauschen |
| `vinyl_fs_725597_WhiteNoiseSleeper.wav` | Crackle+Rauschen | 206,7 s | Breitband, tonal_peak_med 7,1 |
| `vinyl_fs_719879_TheSurfboardingGiraffe.wav` | Rauschfläche | 207,0 s | Breitband, tonal_peak_med 8,2 |
| `vinyl_fs_520147_f-r-a-g-i-l-e.wav` | Statik/Rustling | 27,4 s | tonal_peak_med 22,6 |
| `vinyl_fs_500975_nervousneal.wav` | leiser Rauschboden | 18,3 s | tonal_peak_med 31,8 |
| `vinyl_fs_266786_deleted_user_2906614.wav` | dichtes Crackle/Pop | 62,0 s | impulshaft (peak_med 87,3 = Knistern, kein Ton) |

**Abgelehnt** (ebenfalls CC0, aber kein reines Rauschen — im Manifest
dokumentiert): `506282` „Vinyl" (tonal_peak_med 11 752 = Musikanteil),
`509960` „vinyl_loop 2" (tonaler Loop), `684957` „Vinyl Cut 04"
(Nadel-Cut mit Tonalanteil).

SHA-256 + Probe-Werte (rms, HF-Anteil >6 kHz, Tonalitätsmaß) je Datei:
`data/vinyl_noise_bank/MANIFEST.json`.

## Format-Hinweis

Freesound liefert ohne Account nur die **128-kbps-MP3-Previews** (lokal als
PCM_16/44,1 kHz WAV). Der Rausch-Charakter ist vollständig erhalten;
Lossless-Upgrade-Pfad: Original-WAVs über einen freesound-Account
(dieselben Sound-IDs, im Manifest hinterlegt).

## Code-Anpassungen (Konsumenten)

- `scripts/quality_metric_direction_gate.py::_lade_rauschen`:
  `leerrille_*.wav` → `vinyl_*.wav` (mit Begründungskommentar).
- `scripts/erstelle_corpus_vinyl.py::_vinyl_rezept`:
  Scratch-Quelle `"echt_beschaedigt" in p.name` → `p.name.startswith("vinyl_")`.
- `scripts/erstelle_cantus_real_paare.py`: Docstring-Rolle aktualisiert
  (Loader globt `*.wav` — erfasst die neue Bank automatisch).

## R5-Neuerzeugung (2026-11-17, User-Freigabe)

Die 60 R5-Paare (`data/cantus_real/`, 2026-11-12) wurden mit der neuen
CC0-Vinyl-Bank neu erzeugt: `erstelle_cantus_real_paare.py` (Seed 42,
20 MUSDB-Vocal-Stems × vinyl/rir/vinyl_rir, 10 s, Exit 0).
**Verifiziert:** 60/60 sha256_pcm konsistent (Datei ↔ Manifest), **57/60
geändert** (alle 40 vinyl/vinyl_rir-Paare neu; die 3 unveränderten sind
reine `rir`-Paare ohne Vinyl-Anteil — RNG-Stream-Verschiebung betrifft nur
die Vinyl-Ketten), **60/60 Clean byte-identisch** (gleiche MUSDB-Ausschnitte),
`noise_quelle=data/vinyl_noise_bank` auf 40/40 Vinyl-Paaren;
`test_erstelle_daten_autark.py` 5/5 grün. Alt-Manifest (Schellack-Stand)
archiviert als `2026-11-17_cantus_real_manifest_alt_schellack.jsonl`
(hash-konsistent gegen den Altbestand geprüft).

**Retrain SAUBER (User-Entscheid 2026-11-18):** Ein erster domain_adapt-Versuch auf dem November-Stand hätte Schellack- (Epochen 1–20) mit Vinyl-Adaptation (21–40) gemischt und wurde verworfen. pre-domain_adapt- und pretrain-Checkpoints existieren nicht mehr (im November überschrieben; ONNX = pre-KP8-Stand, Parity stale) ⇒ volle Neu-Leiter als `_v2`-Jobs: `pretrain(50) → fine_tune(30) → domain_adapt(20, NUR Vinyl-Paare)`.
**Befund unterwegs:** pretrain (Ketten-Befehl, Batch 8) scheiterte reproduzierbar mit CUDA-OOM (22,77 GiB Peak, torch 2.11/ROCm 7.2 — D-K3-103-Erbe; Kette-R3 trug nie Batch-Flags) ⇒ pretrain_v2 läuft mit dem kanonischen D-K3-103-Rezept `--batch-size 2 --grad-accum 4` (effektive Batch 8 ⇒ Trainingsdynamik äquivalent; Hash-Reproduzierbarkeit gegen `78ec1319` bleibt prüfbar). Nach Abschluss je Phase: Hash-Abgleich gegen die Alt-Manifeste (`78ec1319` pretrain, `c6132ce8` fine_tune), danach Never-worsen-Gate/Hörabnahme. Manifeste: `train_job_c2_cantus_{pretrain,finetune}_v2.json` + `…_domain_vinylv3.json`.

Provenienz des Altbestands bleibt in
`docs/reports/current/2026-10-28_kp6_vinyl_noise_bank_manifest.json`.
