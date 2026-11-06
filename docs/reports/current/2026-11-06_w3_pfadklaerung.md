# 2026-11-06 — Pfad-Klärung W-3: Warum der Schalter `use_sgmse_musik` im Messpfad wirkungslos war

> **Auftrag:** Autonomer Lückenschluss nach `docs/SOTA_HANDOVER.md` §2 (priorisiert),
> Pflicht-Wiederherstellungs-Check §3, „Fallback-/Mess-Läufe fortsetzen wo sie
> endeten", AGENTS.md-Regeln (Write-Gate, Zitierdisziplin §5, deutsche Logs,
> Determinismus §G5 (copilot-instructions.md)), Tests + Pre-Commit-Gates grün,
> Conventional Commit. Keine Hör-/Lizenz-Entscheidungen, kein Flag-Flip ohne
> §4-Verfahren (§5 SOTA_HANDOVER, §V7 (copilot-instructions.md)). Quelle je
> Zeile. Sitzung 2026-11-06 (04:30–14:55 CET).

## 1. Pflicht-Wiederherstellungs-Check (SOTA_HANDOVER §3)

| Prüfung | Befund | Beleg |
| --- | --- | --- |
| (a) 04:30 CET | Ketten-Warter lief (PIDs 1109714/1109716, gestartet 2026-11-05T17:07:59, `/tmp/ketten_warter.log`); D3-BigVGAN-Resume aktiv (Runner PID 746833, Kind `train_bigvgan_f3.py` PID 1674812, seit 04:23); zusätzlich A-SPADE/M8-Lauf ab 05:24 (PID 1723401) über `scripts/gpu_kette3_k4_rest.sh` (Commit `7b1d0929`) — **parallel** zum BigVGAN-Training (Verstoß gegen §5 SOTA_HANDOVER „keine parallelen Trainings“, nicht von dieser Sitzung gestartet; dokumentiert, nicht gestört) | Prozessliste, `docs/reports/current/gpu_kette3_2026-11-06.log` |
| (a) 14:40 CET (nach Session-Abbruch) | **Keine Kette, kein Training** mehr aktiv. Manifest-Stände: `train_job_d3_bigvgan_f3_fortsetzung_20261105.json` = fehlgeschlagen (4× Timeout 21 600 s, exit −15; Auto-Resume bis Epoche 16, bestes Val-A1 0,5682, Checkpoints `output/f3_bigvgan`); `train_job_m8_aspade.json` = 1× Timeout (bis Epoche 41/60, Loss 0,15); `train_job_d3_diffwave_vokal_fortsetzung_20261105.json`/`train_job_d3_flashsr_musik.json` = ohne Versuche (Queue nie erreicht) | `docs/reports/current/train_job_*.json` |
| **Konsequenz (a)** | Pflicht-Check greift (keine Kette + offene Runden) ⇒ **Ketten gestartet 14:44 CET**: `nohup bash scripts/gpu_kette1_welle_c.sh > /tmp/k1.log 2>&1 &` (PID 2254039, R1 `m8_aspade` mit Auto-Resume) und `nohup bash scripts/gpu_kette2_matrix_d3.sh > /tmp/k2.log 2>&1 &` (PID 2254102). Kette 2 wartet über die eigene Sequenz-Schleife (`while pgrep -f "gpu_kette1_welle_c\.sh"`, `scripts/gpu_kette2_matrix_d3.sh` Z. 7) ⇒ **keine parallelen Trainings** | `/tmp/k1.log`, `/tmp/k2.log`, Prozessliste |
| Resume-Pfad-Fix | `scripts/gpu_kette2_matrix_d3.sh` D3-2 resumete auf `models/bigvgan/checkpoint_latest.pt` — **existiert nicht** (dort liegen nur die vortrainierten Artefakte `bigvgan_v2.pth`/`.onnx`; `scripts/train_bigvgan_f3.py` speichert unter Default `--out-dir output/f3_bigvgan`, Z. 209 + Z. 346–354). Korrigiert auf `--ckpt-dir output/f3_bigvgan` + `--resume output/f3_bigvgan/checkpoint_latest.pt` (identisch zur bewährten Warter-Invocation „Resume-Pfad korrigiert“ 2026-11-05). Ohne Korrektur hätte der nächste D3-2-Lauf bei Epoche 0 begonnen statt bei Epoche 16 fortzufahren | Manifest `train_job_d3_bigvgan_f3_fortsetzung_20261105.json`, Dateisystem-Check 2026-11-06 |
| Beobachtung (offen) | 30 Epochen × ~5 000 s ≫ `--timeout-s 21 600` ⇒ die Jobs laufen abschnittsweise über Auto-Resume (beobachtet: Epoche 4→41 bzw. 4→16 je 6-h-Versuch); ein Abschluss in einem Versuch ist strukturell ausgeschlossen. Entscheidung über Timeout-/Epochen-Zahl ist offen und **nicht** Teil dieses Auftrags | Manifest-`versuche`-Blöcke |
| (b) Mess-/Fallback-Läufe | „4-Modelle“-B3-Lauf weiterhin **nicht rekonstruierbar** (Treiber ad-hoc, kein Blob in der History) — kein Nachbau, da unklare Datenlage (Auftragsteil 3: abbrechen und begründen; dokumentiert 2026-11-04_sitzung_transient_messlauf_b3_status.md §6–7). RT-Budget-Nachweis W-2 fenster-blockiert (trainingsfreies Fenster; seit 14:44 läuft wieder die Kette). Keine weiteren unterbrochenen Mess-Läufe | ebd.; `docs/reports/current/2026-11-05_sitzung_w2w3_b3_nachmessung.md` §3.3 |

**Nicht angefasst:** Die menschliche Hör-Instanz hat Block C entschieden (EAR-VAE
v2 = NO-GO, v1 bleibt; Ziel-Koordinate „heller UND weniger Rauschen“) — als
Commits `7d2e3441`/`5690c08c` durch die parallele Sitzung gebucht. Ebenfalls
unberührt: die parallel in Arbeit befindlichen Dateien
`scripts/prepare_tfgrid_pairs.py`, `tests/unit/test_prepare_tfgrid_pairs.py`,
`.github/FILE_REGISTRY.md` (offener Stand der parallelen Sitzung).

## 2. Lückenwahl (§2 SOTA_HANDOVER)

- #4 Laufzeit: geschlossen (RT-Messung 32,19×; Verifikation nach D3-Lauf
  fenster-blockiert, läuft über die Ketten).
- #5 EAR-VAE: menschlich entschieden (NO-GO v2, Commit `7d2e3441`).
- **#6 Wohlklang W-1…W-13 — verbleibend autonom abarbeitbar: „Pfad-Klärung
  (W-3)“ ⇒ gewählt.** (C4-Hörpaare = menschlich; RT-Budget-Nachweis W-2 =
  fenster-blockiert.)
- #7 Modelle (WF-V4/TP-V2/BEATs-/MERT-/Gender-Head): extern-/daten-blockiert
  (2026-11-05_sitzung_w2w3_b3_nachmessung.md §4).
- #8 Hör-/Lizenz-Entscheidungen: menschlich, nie eigenmächtig.

## 3. Kausalkette: „Schalter wirkt im Messpfad nicht messbar“

Die W-3-Nachmessung 2026-11-05 stellte Δ ≈ 0 zwischen Schalter AUS/AN fest
(`b3_stufenmessung_W-3.json`, z. B. `cassette_1980s_wow__w0`: mos_an 4,796 =
mos_aus 4,796) und wertete das als „Schalter ohne messbare Pfad-Wirkung“
(2026-11-05_sitzung_w2w3_b3_nachmessung.md §3.2). Die Klärung der Ursache:

### 3.1 Setz-Weg und Konsumenten — Verdrahtung funktioniert

- `scripts/b3_stufenmessung.py` setzt das Flag pro Fenster via
  `_schalter_setzen` (Z. 64–66) auf `backend.core.music_model_flags`
  (Mapping „W-3“ → `use_sgmse_musik`, Z. 54); `stufe_messen` (Z. 110–117) ruft
  pro Fenster erst AUS- dann AN-Restore auf.
- Alle Konsumenten lesen das Flag **funktionslokal**
  (`from backend.core.music_model_flags import use_sgmse_musik as …`):
  `backend/core/phases/phase_03_denoise.py` Z. 1650,
  `backend/core/phases/phase_49_advanced_dereverb.py` Z. 581–584,
  `backend/core/hybrid/hybrid_dereverb.py` Z. 135–138,
  `backend/core/coordinated_repair.py` Z. 509–513,
  `backend/core/dsp/sota_vocal_model_router.py` Z. 510–512. Die Setzung wirkt
  also zur Laufzeit (kein Import-by-Value-Fehler).
- **Empirischer Beleg:** Im Log der W-2/W-3-Nachmessung
  (`docs/reports/current/2026-11-05_b3_w2w3_nachmessung.log`, 31 468 Zeilen)
  erscheint die Sperrmeldung „Verarbeitungsschritt 49: SGMSE+ … gesperrt →
  WPE-DSP“ **ausschließlich** in Schalter-AUS-Restores der W-3-Bühne:

| Restore (W-3-Bühne) | Flag | Zeilen-Bereich | „gesperrt“-Meldungen |
| --- | --- | --- | --- |
| #13 | AUS | 16038–17312 | 1 (Z. 16569) |
| #14 | AN | 17313–18583 | 0 |
| #15 | AUS | 18584–20065 | 2 (19070, 19866) |
| #16 | AN | 20066–21542 | 0 |
| #17 | AUS | 21543–22974 | 2 (22047, 22733) |
| #18 | AN | 22975–24405 | 0 |
| #19 | AUS | 24406–25796 | 2 (24893, 25567) |
| #20 | AN | 25797–27149 | 0 |
| #21 | AUS | 27150–28354 | 1 (27573) |
| #22 | AN | 28355–29552 | 0 |
| #23 | AUS | 29553–30562 | 1 (29841) |
| #24 | AN | 30563–31468 | 0 |

  (In der W-2-Bühne #1–#12 blieb `use_sgmse_musik` unangetastet = False ⇒ dort
  18 Sperrmeldungen, konsistent.)

### 3.2 Konsument A — Schritt 03 (Denoise-Zweig): Gate im Messpfad doppelt zu

- Die Schaltstelle `_sgmse_eligible = _use_sgmse_musik_p03 and
  _sgmse_material_ok` (`phase_03_denoise.py` Z. 1662–1667) erfordert u. a.
  `quality_mode in ("quality", "maximum")` (Z. 1654–1660).
- Der Vertrags-Messpfad stellt `QualityMode.FAST`
  (`scripts/b3_stufenmessung.py` `_restore_fenster`, Z. 85–93; identisch zur
  Release-Candidate-Pipeline `scripts/release_candidate_pipeline.py` Z. 136/142
  — der Messpfad bildet den Produktionspfad also korrekt ab).
- **Beleg:** „Modus=QualityMode.FAST“ im Log (Z. 138, 1504, 2788, 4327 …);
  die Gate-Meldung „Phase 03: SGMSE+ … gesperrt“ (setzt `_sgmse_material_ok ==
  True` voraus) kommt in **0 von 24 Restores** vor (`grep -c "Phase 03"` = 0).
  Zusätzlich sperrt `not _dfn_applied` den Zweig auch bei quality/maximum,
  solange DFN-Musik (`use_df_musik=True`) greift — SGMSE+ ist dort nur
  Ersatzpfad (dokumentiertes Design, Z. 1645–1652).

### 3.3 Konsument B — Schritt 49 (Dereverb-Tier-0): Dauer-Achsen-Defekt

- Das Flag-Gate funktioniert (3.1); dahinter lag aber ein weiterer Skip:
  Die Mindest-Signallängen-Prüfung berechnete die Dauer **nach**
  `to_channels_last` (Eingangs-Normalisierung auf (N, C),
  `phase_49_advanced_dereverb.py` Z. 272) mit `audio.shape[-1]` — der
  **Kanalzahl 2** bei Stereo. Ergebnis: `_audio_dur_mono_s ≈ 0,00004 s` <
  1,0 s ⇒ `_sgmse_kipped_short` (Z. 566–570) ⇒ der SGMSE+-Tier-0 war für
  **jedes Stereo-Fenster** unerreichbar, auch bei gesetztem Flag.
- **Belege:** (i) Im W-3-Log gibt es 0× „Tier-0 SGMSE+ OK“, 0× „SGMSE+
  verbessern fehlgeschlagen“, 0× „ML→DSP-Ersatzpfad aktiviert“ — der Zweig
  wurde nie betreten; die Schalter-AN-Restores laufen still auf WPE
  (z. B. Restore #14: „Verarbeitungsschritt 49 WPE-Dereverb: strength=0,05“
  Log Z. 17841, ohne Sperrmeldung davor). (ii) Im selben Modul berechnet das
  Reverb-Presence-Gate die Dauer korrekt über `len(audio)` (Z. 404) — der
  Widerspruch zweier Achsen-Ausdrücke im selben Funktionskörper. (iii)
  **Rot→Grün-Beleg:** `tests/unit/test_phase_49_advanced_dereverb.py::
  TestSgmseTier0PfadW3::test_sgmse_tier0_erreichbar_stereo` scheitert am alten
  Ausdruck (Factory nicht aufgerufen) und ist nach dem Fix grün.
- Das SGMSE-Plugin gibt Ausgaben layout-konform zurück (Round-Trip
  (2,N)↔(N,2), `plugins/sgmse_plugin.py` Z. 511–513/567–568) — der nun
  erreichbare Zweig ist layout-sicher.

## 4. Umgesetzt

1. **Fix** `backend/core/phases/phase_49_advanced_dereverb.py`: Dauer-Berechnung
   layout-sicher über `sample_axis` aus `backend.core.audio_layout`
   (§G9 (copilot-instructions.md): Achsen-Entscheidung aus audio_layout),
   Z. 572. **Flag-Verhalten unverändert:** `use_sgmse_musik` bleibt Default
   `False` (`backend/core/music_model_flags.py` Z. 27–28) ⇒ bit-identisches
   Bestandsverhalten (§G5 (copilot-instructions.md)); bei gesetztem Flag wird
   der Vertrags-Schalter künftig wirksam.
2. **Tests** `tests/unit/test_phase_49_advanced_dereverb.py` +
   `TestSgmseTier0PfadW3` (3 Tests): (a) Tier-0 bei Flag + 2,5-s-Stereo
   erreichbar, (b) ohne Flag gesperrt (§V7 (copilot-instructions.md)), (c)
   Kurzsignal-Skip (< 1 s) bleibt erhalten. Deterministisch: RandomState(7),
   RAM-Budget-Entscheidung gestubbt.
3. **Fix** `scripts/gpu_kette2_matrix_d3.sh` D3-2: Resume-Pfad
   `models/bigvgan` → `output/f3_bigvgan` (siehe §1).
4. Keine neuen Dateien ⇒ kein FILE_REGISTRY-Eintrag nötig (Write-Gate
   `scripts/repo_search.py --before-create` nur bei Neuanlage).

## 5. Bewertung der W-3-Datenlage

- Der W-3-B3-Beschluss 2026-11-05 (`b3_objektivbeschluss_W-3.json`,
  „ABGELEHNT (stärkstes Nein)“, 0/6) verglich zwei **strukturell identische**
  Ausgänge: In beiden Armen lief WPE-Dereverb, da der Schalter im Messpfad
  keine Pfad-Wirkung entfalten konnte (§3). Der Beschluss ist damit ein
  **No-Op-Artefakt und kein Qualitätsurteil** über SGMSE+-Dereverb;
  `never_worsen: true` ist trivial (Δ = 0).
- Die Vertragsspalte „Pfad gesperrt“ (`.github/WOHLKLANG_CLAIMS.md`, W-3)
  bleibt richtig: `use_sgmse_musik` bleibt **AUS** (kein Flag-Flip).
- Eine aussagekräftige W-3-Messung erfordert vorher zwei **Vertrags-/§4-
  Entscheidungen** (menschlich, nicht autonom): (i) Messprofil — erst
  quality/maximum öffnet den Schritt-03-Zweig, das Vertrags-RT-Budget misst
  FAST; (ii) Vertrags-Referenz-Songs statt Korpus-Clips
  (2026-11-05_sitzung_w2w3_b3_nachmessung.md §3.1). Bis dahin gilt die
  Datenlage als geklärt, aber die Stufe als nicht messend entschieden.

## 6. Bewusst nicht angefasst

- Keine Flag-Aktivierung/-Flips (nur §4-Verfahren, fail-closed; §5 SOTA_HANDOVER).
- Keine neue B3-Messung (Vertrags-/Verfahrensfrage; Auftragsteil 3).
- Laufende Ketten/Trainings nicht gestört; der A-SPADE-Parallel-Lauf vom
  Vormittag nur dokumentiert.
- Menschliche Hör-Dokumente und die parallel in Arbeit befindlichen Dateien
  der parallelen Sitzung (§1) unberührt.

## 7. Offene Reste (gefunden, nicht behoben)

1. `phase_49_advanced_dereverb.py` Z. 988–989: `_n_samp_p49`/`_n_in_p49`
   nutzen im channels-last-Fenster ebenfalls `shape[-1]` ⇒
   `_n_blend_p49 = min(2, 2)` ⇒ die [RELEASE_MUST]-Atemzonen-Rückblendung
   (§0p Vocal-Supremacy) ist de facto ein No-Op. Eigenständiger Fix +
   Never-worsen-Beleg nötig (Verhaltensänderung auf dem Default-Pfad).
2. `plugins/tape_restoration_plugin.py` Z. 20–40: Die Tape-ML-Chain nutzt
   denselben SGMSE+-Sprach-Core (Diffusion-Inpainting für Dropouts)
   **unabhängig** von `use_sgmse_musik` — Rollen-Frage (Inpainting vs.
   Enhancement) berührt §III.11 (copilot-instructions.md); Klärung durch
   Domänen-/Hör-Instanz nötig, kein autonomer Eingriff.
3. W-2 (`use_scnet_music`): Δ ≈ 0 mit 2/6 Vorteilen im selben FAST-Messpfad —
   analoge Gate-Kette ungeprüft (eigene Pfad-Klärung möglich).
4. W-3-Neumessung erst nach §5 (i)/(ii); Timeout-/Epochen-Frage der
   Trainings-Jobs (§1) offen.

## 8. Reproduzierbarkeit

- Seeds: Tests RandomState(7); B3-Messung seed 42 (Harness-Default).
- Umgebungen: Tests/Messung `python3` (torch 2.7.0+cpu, GPU-frei,
  `nice -n 15` Host-Schutz); GPU-Ketten
  `/home/michael/.local/share/aurik/venv_rocm72/bin/python` (torch
  2.11+rocm7.2).
- Befehle: `python3 -B -m pytest tests/unit/test_phase_49_advanced_dereverb.py
  -q` (11 passed, davon 3 neu; Rot→Grün gegen den alten Ausdruck verifiziert),
  `python3 scripts/wohlklang_gate.py` („Vertrag erfüllt"; Phantom-Referenzen 30
  = dokumentierter P2-Bestand), Pre-Commit-Gauntlet im Commit `c0ff1a26`:
  **alle Hooks grün** (u. a. `aurik-unit-smoke` tests/unit, Coverage-Gate,
  ID-Registry R1/R2 fail-closed, Gebote-Verifier 25/25, File-Lifecycle,
  Wohlklang-Vertrag, Layout-EINE-Quelle §G9 (copilot-instructions.md),
  Conventional Commits).
- Commits dieser Sitzung: `0c02c90d` (Ketten-Resume-Pfad Kette 2), `c0ff1a26`
  (W-3-Pfad-Klärung: Fix + Tests + Report + §2/Roadmap).
