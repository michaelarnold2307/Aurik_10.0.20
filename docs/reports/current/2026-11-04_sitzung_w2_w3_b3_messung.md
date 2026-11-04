# Sitzung 2026-11-04 (3. Fenster) — W-2/W-3-B3-Messung + RT-Mess-Vorbereitung

> **Auftrag:** Autonomer Lückenschluss nach `docs/SOTA_HANDOVER.md` (§2
> priorisiert), Pflicht-Wiederherstellungs-Check, „Fallback-/Mess-Läufe
> fortsetzen wo sie endeten“, AGENTS.md-Regeln, alle Pre-Commit-Gates grün,
> Conventional Commit. Hör-/Lizenz-Entscheidungen nie eigenmächtig (§5
> SOTA_HANDOVER). Quelle je Zeile (Zitierdisziplin `AGENTS.md` §5).

## 1. Wiederherstellungs-Check (Pflicht, SOTA_HANDOVER §3)

| Prüfung | Befund (2026-11-04 12:43 CET) | Beleg |
| --- | --- | --- |
| `pgrep -f gpu_kette` | Kette 2 läuft (PID 3560120, `bash scripts/gpu_kette2_matrix_d3.sh`) | Prozessliste |
| Trainings-Runden | `d3_diffwave_vokal` **erfolgreich** (Commit-Gate bestanden); `d3_bigvgan_f3` läuft (run_id `matrix_d3-20261104T092240Z-b734d7`, Timeout 21 600 s ⇒ Ende ≤ ~16:22 CET); **D3-3 `d3_flashsr_musik` folgt danach** (Phase 3 der Kette, `scripts/gpu_kette2_matrix_d3.sh`) | `docs/reports/current/train_job_d3_*.json`, `scripts/gpu_kette2_matrix_d3.sh` |
| Konsequenz | **Kein trainingsfreies Fenster** (erst nach D3-3) ⇒ kein Kettenstart (Regel a greift nicht: Kette läuft), kein paralleles Training (§5); Mess-Läufe CPU-only (`python3` = torch 2.7.0+cpu) mit `nice` | diese Sitzung |

## 2. Gewählte Lücke und Begründung

**#4 (F-Reihe 53×→32× RT)** bleibt GPU-gebunden: Die offenen Punkte „RT-Messung
Referenz-Song“ und „Verifikation nach D3-Lauf“ brauchen das trainingsfreie
Fenster (SOTA_HANDOVER §2/4). **#6 (W-1…W-13)** hat als einzigen autonom
messbaren Rest die vom Manager-Bericht 2026-11-03 §5 empfohlene
**W-2/W-3-Nachmessung** („nach dem B3-Verfahren erneut messen, wenn
Budget/Messlatenz erlaubt“) — W-1/W-4/W-5 warten auf die Hör-Instanz,
W-6/W-7 sind nachgemessen und abgelehnt. **#7** ist laut Roadmap
„Was fehlt“-Tabelle vollständig blockiert (WF-V4 „Quelle/Download unklar“,
TP-V2 „Modell + Quelle fehlen“, BEATs-/MERT-Head „GPU/Trainingslauf fehlt“,
Gender-Head „≥ ~50 Labels je Klasse fehlen“); **#8** menschlich.

## 3. Umgesetzt: B3-Harness für W-2/W-3 (Commit `45384a3`)

- **Verdrahtungs-Nachweis:** Die Vertragsschalter `use_scnet_music` (W-2) und
  `use_sgmse_musik` (W-3) sind im Produktionspfad verdrahtet
  (`backend/core/dsp/stem_separator.py` §P1-2, `plugins/scnet_plugin.py`,
  `backend/core/coordinated_repair.py`, `backend/core/dsp/sota_vocal_model_router.py`,
  `backend/core/hybrid/hybrid_dereverb.py`,
  `backend/core/phases/phase_49_advanced_dereverb.py`).
- **Erweiterung** `scripts/b3_stufenmessung.py`: `_schalter_lesen`-Mapping und
  Beschluss-Flag-Mapping um W-2/W-3 ergänzt (Default `--stages` unverändert).
- **Tests** `tests/unit/test_wohlklang_gate.py`: Mapping deckt alle
  Vertragsschalter W-1…W-7 ab (Attribut existiert am Modul) + Beschluss-Flag
  bleibt konsistent zum Schalter-Mapping (E5 Schalter-Abgleich,
  WOHLKLANG_CLAIMS.md). Ergebnis: **55/55 grün** (inkl.
  `tests/unit/test_fallback_auditor_wiring.py`), `ruff --select
  F821,F601,B009,I001` **sauber**. Alle Pre-Commit-Gates grün (Commit `45384a3`).

## 4. Mess-Lauf W-2/W-3 — Ergebnis (B3-Verfahren, §4 SOTA_HANDOVER)

**Lauf:** `nice -n 15 python3 -B scripts/b3_stufenmessung.py --stages W-2,W-3`
(host-sicher parallel zur GPU-Kette, CPU-only, seed 42; Docstring
`scripts/b3_stufenmessung.py`). Erwartung: 3 Clips × 2 Fenster je Stufe,
blindes A/B Restore mit/ohne Schalter, Bewertung musik_mos + defektlast
(beide richtungszertifiziert), B1 = ungeplante ML→DSP-Rückfälle.

<!-- ERGEBNIS WIRD NACH MESSABLAUF NACHGETRAGEN -->

**Hinweis zur Latenz (Budget-Frage W-2):** SCNet-4-Stems und SGMSE+ laufen im
FAST-Profil des Produktionspfads auf CPU; die Messlatenz ist hoch (mehrere
Stunden), die Ausführung ist aber host-sicher (`nice`, CPU-only) und schreibt
ihr Ergebnis je Stufe sofort (`b3_stufenmessung_W-*.json` +
`b3_objektivbeschluss_W-*.json`). Ein Flag-Flip erfolgt **nicht** (nur
§4-Verfahren `scripts/flag_freigabe.py`, fail-closed).

## 5. RT-Messung Referenz-Song (Lücke #4) — vorbereitet, fenster-blockiert

Das verbindliche Mess-Protokoll steht (Report
`docs/reports/current/2026-10-28_f_reihe_laufzeit_optimierung.md` §6):

1. Referenz-Song `test_audio/Testkünstlerin (Schlager) - Du wolltest nur ein
   Abenteuer, aber ich suchte einen Freund.mp3` (224 s).
2. End-to-End-RT über `scripts/_aurik_run_excellence.py` (loggt `RT-Faktor`
   = Wall-Zeit / Audio-Dauer), **3 Läufe, Median**; Baseline des überwachten
   Laufs: Wall 12 137,5 s ≈ **53,9× RT** (die 53×-Ausgangslage,
   `RT_REALITY_MEASURED = 53,0` in `denker/strategie_denker.py`).
3. **Messfenster ohne parallele GPU-Trainingsjobs** (Protokoll-Hinweis:
   Trainingsjobs verzerren die Zahl) — das Fenster öffnet erst nach D3-3
   `d3_flashsr_musik`.
4. Danach Hotspot-Beweis (`scripts/profile_pipeline_hotspots.py --seconds 10`),
   Qualitäts-Gegenprobe (bit-Identität + `tests/unit/test_perf_r18_cache_equivalence.py`)
   und Nachtrag in den F-Reihe-Report; `RT_REALITY_MEASURED` nur bei gemessenem
   Nachweis korrigieren (TODO-P0-1: „kein Zahlenwechsel ohne Strukturaufgabe“).

**Status:** Start durch die nächste Sitzung bzw. den Scheduled-Task
„SOTA-Lückenschluss“ im nächsten trainingsfreien Fenster — bewusst nicht
während der Kette gestartet (§5 SOTA_HANDOVER, Protokoll-Punkt 3).

## 6. Bewusst nicht angefasst

- **Laufende GPU-Kette 2:** unberührt (keine parallelen Trainings, keine
  Drosselung; alle Mess-Läufe CPU-only).
- **Flag-Aktivierungen/Flag-Flips:** keine (nur §4-Verfahren,
  `scripts/flag_freigabe.py`, fail-closed).
- **W-1/W-4/W-5 Hör-Sessions:** menschlich offen (Hörordnung §4/§7).
- **EAR-VAE-Rollout (D-K0-9):** menschlich (Hör-Abnahme vor jedem Austausch).
- **§2/7 Modelle:** Roadmap-Bestätigung, vollständig blockiert (siehe §2).
- **„4-Modelle“-B3-Lauf:** begründet offen (Treiber ad-hoc, Report
  `2026-11-04_sitzung_transient_messlauf_b3_status.md` §6).
- **Pipeline-Baustein U4 / Spec-Revision V27:** offene Reste des
  Transient-Reports (`2026-11-04_transient_energie_ursache_fix.md` §9) —
  nicht Teil der §2-Priorisierung, bewusst dieser Sitzung nicht aufgeschaltet.

## 7. Reproduzierbarkeit

- Seeds: Mess-Lauf seed 42 (Harness-Default); Unit-Tests deterministisch
  (§G5 (copilot-instructions.md)).
- Umgebungen: Messung `python3` (torch 2.7.0+cpu, GPU-frei); Tests
  `.venv_aurik/bin/python -B -m pytest`; `nice -n 15/19` (Host-Schutz).
- Commits dieser Sitzung: `45384a3` (feat: B3-Harness W-2/W-3 + Tests).
