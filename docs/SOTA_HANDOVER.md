# SOTA-Handover — Weiterarbeiten ohne Vorkenntnis (jede Programmier-KI)

> **Zweck:** Dieses Dokument macht den Zustand von Aurik session-unabhängig
> lesbar. Jede KI (Kun, Claude Code, Copilot, Codex, Cursor …) soll nach dem
> Lesen **sofort** wissen, was zu tun ist — auch nach Abbruch, Reboot oder
> Wochen Pause. Stand: 2026-11-03.

## 0. Mission

Aurik 10 = autonomes Audio-Restaurierungsprogramm für Musik mit **weltbestem
Wohlklang für das menschliche Gehör**. Normative Kette: `AGENTS.md` §1
(Pfeil: `AGENTS.md` liest zuerst, dann `.github/copilot-instructions.md`,
`.github/VERBOTEN.md`, Hörordnung, Domain-Regeln, Specs).

## 1. Sofort-Start (3 Minuten)

1. `AGENTS.md` lesen (Repo-Regeln — Write-Gate, Zitierdisziplin §5,
   Determinismus §G5, deutsche Logs).
2. **Dieses Dokument** + `docs/reports/current/2026-10-28_vollstaendigkeit_tiefenanalyse.md`
   (das Lückenbild) + `docs/TODOS_SOTA_ROADMAP.md` (Bestand).
3. Zustand prüfen (siehe §3) und die nächste offene Lücke aus §2 schließen.

## 2. Rest-Lücken (priorisiert) — WAS zu tun ist

| # | Lücke | Typ | Wo dokumentiert |
| --- | --- | --- | --- |
| 1 | ~~WP-2-Hygiene~~ **ERLEDIGT 2026-10-28:** Stubs (D-K3-3), Phantome (D-K3-5), D-K2-4 Namens-Drift und Domänen-Status sind register-seitig geschlossen; die P2-INFO-Findings des Gates sind dokumentierter Akzeptanz-Bestand | — | Register D-K3-3/D-K3-5/D-K2-4/D-K0-10 |
| 2 | Referenzfreie Qualitätsmetrik — **ERLEDIGT 2026-10-28**: `musik_mos` (UTMOS-Ersatz, `plugins/musik_mos_plugin.py`) + `defektlast` beide **zertifiziert** (Spearman 0,95); UTMOS/PQS blind gesperrt; offen nur noch absolute MOS-Kalibrierung (KP-12/MUSHRA-Proxy, menschlich) | — | Zertifikat-JSON + `plugins/musik_mos_plugin.py` |
| 3 | ~~P1-7 (m1b-Stufe-2 intern) + P1-10 (Defekt-Chips)~~ **ERLEDIGT 2026-11-03:** Spec-Verifikation + 15 Verkabelungs-Invarianten, Roadmap-Status geschlossen (Commit `e12a49c7`) | — | Roadmap TODO-P1-7/P1-10 + Report `2026-11-03_p1_7_p1_10_status_audit.md` |
| 4 | Laufzeit 53×→32× RT (F-Reihe optimieren) | autonom (GPU-gebunden) | **IN ARBEIT 2026-11-03/04:** GPU-D3-Kette läuft (DiffWave→BigVGAN→FlashSR, Manifeste `docs/reports/current/train_job_d3_*.json`); §PERF-R18-Äquivalenz-Tests + Budget-Gate **21/21 grün** (2026-11-04, zuvor „nicht ausgeführt"), Ketten-Bugfixes M8/M9/c2-Cantus per Smoke verifiziert (Report `2026-11-04_sota_sitzung_ear_vae_messung_ketten_checks.md`); offen: nur noch Verifikation nach D3-Lauf; **RT-Messung 2026-11-05 ABGESCHLOSSEN: Median 32,19× RT (3 Läufe, Wall 7248,7–7259,2 s / 225,33 s), Baseline 53,9× — Lücke geschlossen** (Report `2026-10-28_f_reihe_laufzeit_optimierung.md` §9, Log `rt_messung_2026-11-04.log`; Hinweis: der geloggte rt_factor ist auf 32,0 gekappt, Spec §9.5; `RT_REALITY_MEASURED` = 32,2); ursprüngliches **RT-Mess-Protokoll 2026-11-04** (Report `2026-10-28_f_reihe_laufzeit_optimierung.md` §6: `scripts/_aurik_run_excellence.py` auf Referenz-Song „Testkünstlerin…“ (224 s), 3 Läufe/Training-frei, Baseline 53,9× RT — Bericht `2026-11-04_sitzung_w2_w3_b3_messung.md` §5) |
| 5 | ~~F3-BigVGAN-Rollout~~ **ENTSCHEIDEN 2026-10-30 (K0-C NO-GO, Hör-Instanz)** + ~~EAR-VAE v1↔v2-Vergleichsmessung~~ **GEMESSEN 2026-11-04:** HF-Gewichte `earlab/EAR_VAE2` (apache-2.0, `c87215f8…`) beschafft; Blind-Messung seed 42: **never_worsen ✓, 16/17 Trials v2-vorteilhaft, Sign-Test p = 0,000275 signifikant, erfuellt_objektiv ✓** (Messdatei `2026-11-03_ear_vae_v1_v2_vergleich.json`, Δ musik_mos +0,10 / Δ defektlast −0,12); **Rollout-Entscheidung v1↔v2 bleibt menschlich (D-K0-9/SOTA-M2 — Hör-Abnahme vor jedem Austausch, §V7 (copilot-instructions.md))** | — | LÜCKEN-MATRIX D.1b/c + K0-Sign-off-Paket + Report `2026-11-04_sota_sitzung_ear_vae_messung_ketten_checks.md` |
| 6 | Wohlklang-Stufen W-1…W-13 aktivieren (0/13 vollständig) | **B3-Verfahren** | **Stand 2026-11-04:** Alle B3-Beschlüsse W-1/W-4/W-5/W-6/W-7 formal **ABGELEHNT** (nicht signifikant, p ≥ 0,06; W-1 zusätzlich never_worsen verfehlt; W-1 hat widersprüchliche Objektiv-Belege — HF-Gain-Beschluss BESTÄTIGT vs. musik_mos-Beschluss ABGELEHNT ⇒ die Hör-Instanz entscheidet, Hörordnung §7); B1-Zähler präzise (Commit `ef3795b7`); Hör-Sessions W-1/W-4/W-5 vorbereitet = **menschlich offen**; W-2/W-3 gesperrt (Budget/Pfad) — **B3-Nachmessung läuft 2026-11-04** (Harness für W-2/W-3 erweitert, Commit `d5483a43`; `b3_stufenmessung.py --stages W-2,W-3`, host-sicher parallel zur Kette; Report `2026-11-04_sitzung_w2_w3_b3_messung.md`); „4-Modelle"-B3-Lauf (Material/EAR/Defekt/Loudness) offen **begründet** — Treiber ad-hoc, nicht rekonstruierbar (Report `2026-11-04_sitzung_transient_messlauf_b3_status.md` §6–7) |
| 7 | Modelle: WF-V4, TP-V2, BEATs-Head, MERT-Head, Gender-Head | teils Daten-blockiert | Tiefenanalyse E2 |
| 8 | Hör-/Lizenz-Entscheidungen (§9/5–7, AERO, MUSHRA n≥30) | **menschlich** | Katalog A2 |
| 9 | ~~P1-3-Restdeckung (Audibility-Guards)~~ **ERLEDIGT 2026-11-03:** HNR/Mikrodynamik/Vibrato/Hallucination/Noise-Texture mit Maskierungs-Abdeckung (§G10), 19 Phasen verdrahtet (Commit `2c5e5522`) | — | Report `2026-11-03_p1_3_audibility_restdeckung.md` |
| 10 | ~~D-K1-3 Material-CRNN-Verdrahtung (95,3 % statt 40,2 %)~~ **ERLEDIGT 2026-11-03:** im Material-Entscheidungspfad hinter `use_material_crnn` (Default AUS — Flag-Flip offen, nur §4-Verfahren) (Commit `28f2947d`) | — | Report `2026-11-03_d_k1_3_material_crnn_wiring.md` |

**Regel:** Punkte 1–5 ohne Rückfrage erledigen. Punkt 6 NUR über das
B3-Verfahren (§4). Punkt 8 nie eigenmächtig — abbrechen und begründet
berichten.

## 3. Zustand & Wiederherstellung nach Abbruch/Reboot

- **GPU-Trainingsketten** (dauerhaft im Repo, jederzeit neu startbar):
  - Kette 1 (Welle C + M8/M9): `nohup bash scripts/gpu_kette1_welle_c.sh > /tmp/k1.log 2>&1 &`
  - Kette 2 (Matrix D.3, startet nach Kette 1): `nohup bash scripts/gpu_kette2_matrix_d3.sh > /tmp/k2.log 2>&1 &`
  - **Kein Doppelläuft-Schutz nötig:** `scripts/train_job_runner.py` hält
    Job-Locks (`models/*/.train_job.lock`) und schreibt Manifeste
    (`docs/reports/current/train_job_*.json`) inkl. Auto-Resume-Befehl.
  - Status prüfen: Manifeste lesen (`ergebnis`, `versuche`) + `rocm-smi`.
    Laufende Kette erkennen: `pgrep -f gpu_kette`.
  - GPU-Umgebung: `/home/michael/.local/share/aurik/venv_rocm72/bin/python`
    (torch 2.11+rocm7.2, Radeon 7900 XTX). System-python ist CPU-only.
- **Dauer-Automation:** Kun-Scheduled-Task „SOTA-Lückenschluss" (ID
  `73cc09ac-1ce4-48ae-851c-5022f637f534`, alle 6 h) arbeitet §2 ab und führt
  die Wiederherstellung aus. Fällt Kun aus, übernimmt die nächste KI manuell
  nach diesem Dokument.
- **Mess-/Trainingsstand:** alles liegt in `docs/reports/current/` (mit Quelle
  je Zeile) und `data/`-Manifesten (`vinyl_noise_bank/`, `rir_real/`) —
  Reproduzierbarkeit über Seeds + sha256.
- **Worker-Worktrees (Manager-Erfahrung 2026-11-05, ADE-Dispatches):** Frisch
  erzeugte Task-Worktrees (`/home/michael/.kun/worktrees/tasks/…/tws_*`) können
  auf einem **veralteten Stand** hängen (beobachtet: `65c6ef4d` vom 2026-10-08
  statt aktuellem main). Symptom: der Worker meldet fehlende Quellen
  (Register/Reports/Harnesses) und blockert korrekt. **Abhilfe (Manager):**
  `git -C <tws> stash push -u && git -C <tws> merge --ff-only main &&
  git -C <tws> stash pop` (bei unverändertem Worktree genügt `merge --ff-only`),
  danach den Worker mit `worker_send` fortsetzen. **Hook-/Gate-Umgebung in
  Worktrees:** `.venv_aurik`-Symlink auf das Haupt-Checkout, `chmod +x
  run_tests_chunked_safe.sh`, und unverfolgte Modell-/Mess-Pfade
  (`models/sgmse_plus/…`, `models/harmonic_inpainting`, `models/vocos_48khz`,
  `output_audio/mushra/…`, `output/scnet_ab_…`) als Symlinks nachziehen — sonst
  scheitern Wohlklang-/SOTA-Gates mit „zitierte Pfade fehlen" (Umgebungs-Artefakt,
  kein Code-Defekt).

## 4. Aktivierungs-Verfahren (Wohlklang-Stufen/Flags) — WIE

Nur zwei legale Wege (fail-closed, `scripts/flag_freigabe.py`):

    # Weg A: Hör-Abnahme (menschlich)
    python3 -B scripts/flag_freigabe.py --flag use_x --verdikt <fit.json> --signoff <signoff.md> --commit
    # Weg B: Formeller Objektivmetrik-Beschluss (Wohlklang-Vertrag B3)
    python3 -B scripts/flag_freigabe.py --flag use_x --verdikt <messung.json> --objektivbeschluss <beschluss.json> --commit

Beschluss-JSON (Schema `aurik.objektivbeschluss/1`) MUSS enthalten:
`flag`, `referenz_songs` ≥ 3, `never_worsen: true`, `signifikant: true`,
`defekt_fallbacks: 0`, `blind_ausgewertet: true`, `messung` (existierende
Messdatei), `zielmetrik`. **Hör-Vorbehalt:** spätere Hör-Abnahme gilt
vorrangig (Widerspruch ⇒ Rollback). Vertrag: `.github/WOHLKLANG_CLAIMS.md`.

## 5. Harte Regeln (Verstöße = Rückabwicklung)

- Flag-Flip NIE ohne §4 (§V7 (copilot-instructions.md): kein stiller Flip).
- Keine Hör-/Lizenz-Entscheidungen eigenmächtig (§9-Liste, Great-78/Material).
- Jede Änderung: Tests + alle Pre-Commit-Gates grün + Conventional-Commit;
  „Regeländerung ⇒ Skript nachziehen" (AGENTS.md §2).
- Laufende Ketten/Hintergrund-Läufe nicht stören; Trainings sequenziell
  (Job-Locks); Host-Schutz (nice, Thread-Caps, RAM-Guard) respektieren.
- Keine Funktion ohne adäquaten Ersatz entfernen (User-Vorgabe 2026-10-28).

## 6. Verifikations-Befehle

    python3 scripts/wohlklang_gate.py          # Vertrag + Phantome (fail-closed)
    python3 -m pytest tests/unit -q            # Unit-Smoke
    python3 .venv_aurik/bin/python scripts/model_inventory.py --fail-on-orphans
    python3 -B scripts/release_report.py --version $(python3 -c "import backend.core.version as v;print(v.__version__)") --out docs/reports/current/release_report_aktuell.md

## 7. Aktualisierungspflicht

Wer eine Lücke schließt: Roadmap-Checkbox mit Beleg abhaken,
`docs/reports/current/`-Report ablegen, **dieses Dokument §2 aktuell halten**
(sonst verliert die nächste KI den Faden).
