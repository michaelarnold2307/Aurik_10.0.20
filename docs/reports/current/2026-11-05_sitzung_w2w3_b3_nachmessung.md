# Sitzung 2026-11-05 (Nachmittag) — W-2/W-3-B3-Nachmessung fortgesetzt + Ketten-Queue

> **Auftrag:** Autonomer Lückenschluss nach `docs/SOTA_HANDOVER.md` (§2
> priorisiert), Pflicht-Wiederherstellungs-Check, „Fallback-/Mess-Läufe
> fortsetzen wo sie endeten", AGENTS.md-Regeln, alle Pre-Commit-Gates grün,
> Conventional Commit. Hör-/Lizenz-Entscheidungen nie eigenmächtig (§5
> SOTA_HANDOVER). Quelle je Zeile (Zitierdisziplin `AGENTS.md` §5).

## 1. Wiederherstellungs-Check (Pflicht, SOTA_HANDOVER §3)

| Prüfung | Befund (2026-11-05 16:29–16:35 CET) | Beleg |
| --- | --- | --- |
| `pgrep -f gpu_kette` | **keine** `gpu_kette*`-Kette; der erste Treffer war der Prüfprozess selbst (Selbst-Match des Prüfbefehls, per `ps -fp` widerlegt) | Prozessliste |
| Echte GPU-Last | **D3-Fortsetzungs-Kette läuft** (Inline-Warter PIDs 613694/613696, gestartet 07:41 CET): 1/3 `d3_bigvgan_f3` Resumption aktiv (Runner PID 746833, Kind 1043142 `train_bigvgan_f3.py --resume output/f3_bigvgan/checkpoint_latest.pt`, GPU 64 %); danach 2/3 `d3_diffwave_vokal`, 3/3 `d3_flashsr_musik` (Warter-Log `docs/reports/current/d3_fortsetzung_2026-11-05.log`) | Prozessliste, `rocm-smi`, Manifest `docs/reports/current/train_job_d3_bigvgan_f3_fortsetzung_20261105.json` (Versuch 1: Timeout 21 600 s, Resume läuft) |
| Trainings-Runden offen | 6 fehlgeschlagen ohne „ergebnis: erfolgreich": `m8_aspade`, `m9_tf_gridnet`, `c2_cantus_pretrain/finetune/domain` (Befehls-Bugs, in `scripts/gpu_kette1_welle_c.sh` korrigiert), `d3_bigvgan_f3` (Alt-Manifest, Timeout); erfolgreich: `c3_rt60`, `d3_diffwave_vokal` | `docs/reports/current/train_job_*.json` |
| Konsequenz | **Kein Ketten-Direktstart** (Regel a greift nicht: eine Kette läuft; ein Direktstart von `gpu_kette1_welle_c.sh` würde M8/M9/c2-Cantus **parallel** zum BigVGAN-Training starten = Verstoß gegen §5 „keine parallelen Trainings") → **Ketten-Warter** (siehe §2) | diese Sitzung |

## 2. Ketten-Kontinuität: sequenzielle Start-Queue (keine parallelen Trainings)

Statt Direktstart wurde ein **Warter-Daemon** gestartet (PID 1108790,
`nohup`, Log `/tmp/ketten_warter.log`) — Muster wie der D3-Warter der
Sitzung 2026-11-05 07:41:

1. Wartet per `pgrep -f "[t]rain_job_runner"` bzw. `[t]rain_(bigvgan|diffwave|
   flashsr|cantus|aspade|tf_gridnet|symphonia|rt60)` (Klammern-Trick gegen
   Selbst-Match), bis **keine** Trainingsprozesse mehr laufen; 120 s Ruhe +
   Doppel-Check.
2. Startet exakt die Auftrags-Befehle sequenziell:
   `nohup bash scripts/gpu_kette1_welle_c.sh > /tmp/k1.log 2>&1 &` und nach
   deren Ende `nohup bash scripts/gpu_kette2_matrix_d3.sh > /tmp/k2.log 2>&1 &`.
3. Doppelstart-Schutz: Abbruch, wenn `gpu_kette1_welle_c`/`gpu_kette2_matrix_d3`
   bereits laufen; Doppelläufe zusätzlich über Job-Locks
   (`models/*/.train_job.lock`, `scripts/train_job_runner.py`).
4. **pgrep-Sichtbarkeit:** Die Warter-Kommandozeile enthält die Ketten-Skript-
   namen — `pgrep -f gpu_kette` findet den Warter, solange die Queue läuft.
   Damit startet der Scheduled-Task „SOTA-Lückenschluss" (6-h-Raster) keinen
   parallelen Lauf mehr (zuvor bestand diese Gefahr: der Inline-D3-Warter ist
   für das pgrep-Muster `gpu_kette` unsichtbar).

Begründung für Verzögerung: SOTA_HANDOVER §5 „Laufende Ketten/Hintergrund-Läufe
nicht stören; Trainings sequenziell". Der Pflicht-Check (a) verlangt den Start
nur „falls keine Kette läuft" — die D3-Kette läuft.

## 3. Lücke #6 (W-1…W-13): W-2/W-3-B3-Nachmessung fortgesetzt

Der vorbereitete Lauf aus Sitzung 2026-11-04 (Commit `45384a3`,
`scripts/b3_stufenmessung.py` mit W-2/W-3-Mapping) wurde **gestartet**; der
Ergebnis-Platzhalter im Report `2026-11-04_sitzung_w2_w3_b3_messung.md` §4
war nie gefüllt, `b3_stufenmessung_W-2.json`/`_W-3.json` fehlten.

**Lauf:** `nice -n 15 python3 -B scripts/b3_stufenmessung.py --stages W-2,W-3`
(CPU-only `python3` = torch 2.7.0+cpu, seed 42, host-sicher parallel zur
GPU-Kette; Log `docs/reports/current/2026-11-05_b3_w2w3_nachmessung.log`).
Umfang: 3 Clips × 2 Fenster je Stufe, blindes A/B Restore mit/ohne
Vertragsschalter (`use_scnet_music` W-2 bzw. `use_sgmse_musik` W-3),
Bewertung musik_mos + defektlast (beide richtungszertifiziert),
B1-Zählung ungeplanter ML→DSP-Rückfälle (`backend/core/fallback_ledger.py`).

### 3.1 Ergebnis W-2 (SCNet-4-Stems)

**Fertig 2026-11-05 ~17:55 CET** (Laufzeit ~80 Min). Belege: `docs/reports/current/b3_stufenmessung_W-2.json` + `docs/reports/current/b3_objektivbeschluss_W-2.json` (Quelle: `scripts/b3_stufenmessung.py`, seed 42, blind ausgewertet, Schalter `use_scnet_music`).

| Prüfung (Vertrag B3) | Messwert | erfüllt? |
| --- | --- | --- |
| Never-worsen (G-4) | `never_worsen: true` — in keinem der 6 Fenster Verschlechterung (Δ musik_mos ≤ 0,0008; Δ defektlast ≤ +0,0043) | ✓ |
| Signifikanter Vorteil (G-2) | `vorteile: 2/6`, `p_wert: 0.6875` (Vorzeichen-Test) — musik_mos Δ ≈ 0 je Fenster | ✗ **nein** |
| B1 ungeplante Rückfälle | `defekt_fallbacks: 0`, `b1_sauber: true` | ✓ |
| Blindheit | `blind_ausgewertet: true` (Harness-intern verlost) | ✓ |
| **Ergebnis** | `erfuellt: false` → Beschluss **ABGELEHNT** (`b3_objektivbeschluss_W-2.json`) | — |

**Struktur-Befund (Vertrags-Abgleich):** Das Harness misst auf dem Referenz-Korpus der Release-Candidate-Pipeline (`_existing_corpus()[:3]`): `cassette_1980s_wow`, `reel_1940s_dropout`, `jazz_1950s_scratched` (6 Fenster). Die Vertragszeile W-2 (`.github/WOHLKLANG_CLAIMS.md`) nennt als „≥ 3 echte Songs" jedoch **AM Contra (165 s), Al James (169 s), Motor Tapes (107 s)**. Die Beschluss-JSON zählt `referenz_songs: 3` (Korpus-Clips) — für einen künftigen Weg-B-Beschluss muss die Messung auf den Vertrags-Songs erfolgen oder die Vertragszeile begründet angepasst werden („Regeländerung ⇒ Skript nachziehen", `AGENTS.md` §2). Da der Beschluss ohnehin nicht signifikant ist, ändert die Diskrepanz nichts am Ergebnis — sie ist als offener Punkt dokumentiert.

### 3.2 Ergebnis W-3 (SGMSE+ Musik-Core)

**Fertig 2026-11-05 ~19:22 CET** (Laufzeit ~87 Min). Belege: `docs/reports/current/b3_stufenmessung_W-3.json` + `docs/reports/current/b3_objektivbeschluss_W-3.json` (Quelle: `scripts/b3_stufenmessung.py`, seed 42, blind ausgewertet, Schalter `use_sgmse_musik`).

| Prüfung (Vertrag B3) | Messwert | erfüllt? |
| --- | --- | --- |
| Never-worsen (G-4) | `never_worsen: true` — Werte praktisch identisch (Δ musik_mos ≤ 0,0001; Δ defektlast ≤ 0,0001) | ✓ |
| Signifikanter Vorteil (G-2) | `vorteile: 0/6`, `p_wert: 0.03125` — **kein** Fenster vorteilhaft; der Schalter wirkt im Messpfad nicht messbar | ✗ **nein** |
| B1 ungeplante Rückfälle | `defekt_fallbacks: 0`, `b1_sauber: true` | ✓ |
| Blindheit | `blind_ausgewertet: true` | ✓ |
| **Ergebnis** | `erfuellt: false` → Beschluss **ABGELEHNT** (`b3_objektivbeschluss_W-3.json`) | — |

**Befund:** Δ ≈ 0 zwischen „Schalter an" und „Schalter aus" — konsistent mit der Vertragsspalte „Pfad gesperrt" (`.github/WOHLKLANG_CLAIMS.md`, W-3): Die Wirkung von `use_sgmse_musik` ist im gemessenen Produktionspfad nicht nachweisbar (Zeilen 1–6 fast identisch, z. B. `cassette_1980s_wow__w0`: mos_an 4.796 = mos_aus 4.796). Bis die Pfad-Wirkung geklärt ist, ist die Datenlage unklar (§5 SOTA_HANDOVER).

### 3.3 Aktivierungs-Entscheidung (fail-closed, §4 SOTA_HANDOVER)

**Keine Flag-Aktivierung.** Beide Wege sind versperrt bzw. nicht erfüllt:

1. **Weg B (Objektivbeschluss) ausgeschlossen:** `scripts/flag_freigabe.py` prüft beide Beschlüsse und bricht fail-closed ab (Trockenlauf ohne `--commit`, 2026-11-05 ~19:25 CET):

       ABBRUCH: Objektivbeschluss ungueltig — Vorteil nicht signifikant (G-1/G-2/G-3)   # use_scnet_music
       ABBRUCH: Objektivbeschluss ungueltig — Vorteil nicht signifikant (G-1/G-2/G-3)   # use_sgmse_musik

2. **Weg A (Hör-Abnahme)** ist menschlich (Hörordnung §4/§7, D-K0-5: „bis C4-Hörstichprobe + Hörordnungs-Sign-off" bzw. W-3 „bis Drei-Wege-A/B + Hörabnahme") — SOTA_HANDOVER §5: nicht eigenmächtig.

3. **Zusätzlich unklare Datenlage** (je §5 „abbrechen und begründet berichten"):
   - W-2: Vertragsspalten „Budget" (CPU ~7× langsamer als Demucs v4 — Zielkonflikt mit dem F-Reihe-RT-Budget 32×, §2/4) und „Blindes A/B (C4)" offen; Hörpaare liegen bereit (`output/scnet_ab_2026-10-06_fair`).
   - W-2/W-3: Messung auf Korpus-Clips statt der Vertrags-Referenz-Songs (§3.1).
   - W-3: Schalter ohne messbare Pfad-Wirkung (§3.2).

**Ergebnis:** `use_scnet_music` und `use_sgmse_musik` bleiben **AUS** (Vertragssoll „AUS" eingehalten). Die Nachmessung schließt den „Mess-Lauf"-Teil von §2/6 ab; die Aktivierungs-Entscheidungen bleiben menschen-/vertragsgebunden (C4-Hörstichprobe W-2, Budget-Nachweis W-2, Pfad-Klärung W-3, Hör-Abnahmen W-1/W-4/W-5).

### 3.4 Parallele Aktivität (Kun-Scheduled-Task, dokumentiert)

Während dieses Laufs arbeitete der Scheduled-Task „SOTA-Lückenschluss"
parallel am selben Repo (Commits `40086bbd` 17:51, `4361777f` 18:18,
`63c9ff9f` 18:55). Befunde, die diesen Bericht berühren:

- `63c9ff9f` (`docs(reports)/2026-11-05_k4_wohlklang_stufen_objektivierung.md`)
  wertete die um 17:55 fertige `b3_stufenmessung_W-2.json` mit aus
  („objektiv ABGELEHNT", p = 0,6875) — identisches Ergebnis wie §3.1; W-3
  galt dort noch als „ohne Messung", da der W-3-Lauf erst 19:22 endete.
  Der dadurch entstandene Widerspruch in `docs/SOTA_HANDOVER.md` §2/6 ist
  von dieser Sitzung konsolidiert („W-3 damals noch ohne Messung …
  Nachmessung am selben Tag abgeschlossen: 0/6 ⇒ ebenfalls ABGELEHNT").
- Zwei gestaged, aber **nicht** von dieser Sitzung committed:
  `2026-11-05_hoer_runde_gesamtpaket.md` + `2026-11-05_hoer_signoff.md` —
  eine **unausgefüllte** Hör-Runden-Vorlage für die menschliche Hör-Instanz
  („ohne Flag-Flip ohne Ihr explizites GO"). Keine Hör-Entscheidung wurde
  getroffen; die Dateien blieben unberührt (git-Index unangetastet).
- Keine Konflikte mit der laufenden GPU-Kette; der Scheduled-Task startete
  **keine** parallelen Trainings (der Ketten-Warter war pgrep-sichtbar,
  §2 — genau der Zweck der Sichtbarkeit).

## 4. Bewusst nicht angefasst

- **Laufende GPU-Kette:** unberührt (kein paralleles Training, keine
  Drosselung; Mess-Lauf CPU-only `nice -n 15`).
- **Flag-Aktivierungen/Flag-Flips:** siehe §3.3 — nur §4-Verfahren
  (`scripts/flag_freigabe.py`, fail-closed).
- **Hör-/Lizenz-Entscheidungen:** keine (W-1/W-4/W-5 Hör-Sessions, EAR-VAE-
  Rollout D-K0-9, Great-78/Material — menschlich, SOTA_HANDOVER §5).
- **§2/7 Modelle** (WF-V4, TP-V2, BEATs-Head, MERT-Head, Gender-Head): laut
  Roadmap „Was fehlt"-Tabelle vollständig blockiert (Quelle/Download unklar,
  Modell + Quelle fehlen, GPU/Trainingslauf fehlt bzw. ≥ ~50 Labels je Klasse);
  die GPU-gebundenen Teile laufen über die Ketten-Queue.
- **§2/4 Verifikation nach D3-Lauf:** fenster-blockiert (trainingsfreies
  Fenster erst nach `d3_flashsr_musik`), läuft automatisch über Warter/
  Scheduled Task.
- **`study_stimuli/`** (untracked, `model_ab/`): Hör-Studien-Material —
  menschlicher Bereich, nicht versioniert in dieser Sitzung.

## 5. Reproduzierbarkeit

- Seeds: Mess-Lauf seed 42 (Harness-Default); Warter deterministisch
  (Zustands-Checks, keine Zeit-Entscheidungen) (§G5 (copilot-instructions.md)).
- Umgebungen: Messung `python3` (torch 2.7.0+cpu, GPU-frei); GPU-Ketten
  `/home/michael/.local/share/aurik/venv_rocm72/bin/python` (torch
  2.11+rocm7.2); `nice -n 15` (Host-Schutz).
- Ketten-Warter: `/tmp/ketten_warter.log`, Start-Ketten-Logs `/tmp/k1.log`,
  `/tmp/k2.log`.
