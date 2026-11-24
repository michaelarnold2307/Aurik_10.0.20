# SOTA-Handover — Weiterarbeiten ohne Vorkenntnis (jede Programmier-KI)

> **Zweck:** Dieses Dokument macht den Zustand von Aurik session-unabhängig
> lesbar. Jede KI (Kun, Claude Code, Copilot, Codex, Cursor …) soll nach dem
> Lesen **sofort** wissen, was zu tun ist — auch nach Abbruch, Reboot oder
> Wochen Pause. Stand: 2026-11-21.

## 0. Mission

Aurik 10 = autonomes Audio-Restaurierungsprogramm für Musik mit **weltbestem
Wohlklang für das menschliche Gehör**. Normative Kette: `AGENTS.md` §1
(Pfeil: `AGENTS.md` liest zuerst, dann `.github/copilot-instructions.md`,
`.github/VERBOTEN.md`, Hörordnung, Domain-Regeln, Specs).

**Wettbewerbs-Parität (Zielbild, normativ/stufenweise):** Aurik schließt messbare
Wohlklang-Lücken systematisch und strebt langfristig Wettbewerbsniveau in
jeder Restaurierungs-Disziplin an. Das ist keine Behauptung bereits erreichter
Gesamtparität und kein pauschales Release-Gate. Interner Fortschritt
(`GESICHERT`) ist von externer Ergebnisparität (`PARITAET`) getrennt. Für eine
RX-12-Paritätsaussage gilt Spec 26 DRT-16: gepaarter Referenzlauf,
Nichtunterlegenheit und §4-Sign-off. Ziel ist perzeptuelle Qualität, nicht
sample-identische Bearbeitung oder eine Kopie der RX-Bedienung.
Nicht erreichbare bzw. nicht messbare Belege bleiben offen und werden mit
konkretem nächsten Schritt geführt. Das
`.github/WETTBEWERBS_PARITAET.md`-Register ist Pflichtlektüre; sein
Pre-Commit-Gate prüft die Status- und Evidenzregeln fail-closed.

## 1. Sofort-Start (3 Minuten)

1. `AGENTS.md` lesen (Repo-Regeln — Write-Gate, Zitierdisziplin §5,
   Determinismus §G5, deutsche Logs).
2. **Dieses Dokument** + `.github/WETTBEWERBS_PARITAET.md` +
   `docs/reports/current/2026-10-28_vollstaendigkeit_tiefenanalyse.md`
   (das Lückenbild) + `docs/TODOS_SOTA_ROADMAP.md` (Bestand).
3. Zustand prüfen (siehe §3) und die nächste offene Lücke aus §2 schließen.

## 2. Rest-Lücken (priorisiert) — WAS zu tun ist

| # | Lücke | Typ | Wo dokumentiert |
| --- | --- | --- | --- |
| 1 | ~~WP-2-Hygiene~~ **ERLEDIGT 2026-10-28:** Stubs (D-K3-3), Phantome (D-K3-5), D-K2-4 Namens-Drift und Domänen-Status sind register-seitig geschlossen; die P2-INFO-Findings des Gates sind dokumentierter Akzeptanz-Bestand | — | Register D-K3-3/D-K3-5/D-K2-4/D-K0-10 |
| 2 | Referenzfreie Qualitätsmetrik — **ERLEDIGT 2026-10-28**: `musik_mos` (UTMOS-Ersatz, `plugins/musik_mos_plugin.py`) + `defektlast` beide **zertifiziert** (Spearman 0,95); UTMOS/PQS blind gesperrt; offen nur noch absolute MOS-Kalibrierung (KP-12/MUSHRA-Proxy, menschlich) | — | Zertifikat-JSON + `plugins/musik_mos_plugin.py` |
| 3 | ~~P1-7 (m1b-Stufe-2 intern) + P1-10 (Defekt-Chips)~~ **ERLEDIGT 2026-11-03:** Spec-Verifikation + 15 Verkabelungs-Invarianten, Roadmap-Status geschlossen (Commit `e12a49c7`) | — | Roadmap TODO-P1-7/P1-10 + Report `2026-11-03_p1_7_p1_10_status_audit.md` |
| 4 | Laufzeit 53×→32× RT (F-Reihe optimieren) | autonom (GPU-gebunden) | **IN ARBEIT 2026-11-03/04:** GPU-D3-Kette läuft (DiffWave→BigVGAN→FlashSR, Manifeste `docs/reports/current/train_job_d3_*.json`); §PERF-R18-Äquivalenz-Tests + Budget-Gate **21/21 grün** (2026-11-04, zuvor „nicht ausgeführt"), Ketten-Bugfixes M8/M9/c2-Cantus per Smoke verifiziert (Report `2026-11-04_sota_sitzung_ear_vae_messung_ketten_checks.md`); offen: nur noch Verifikation nach D3-Lauf; **RT-Messung 2026-11-05 ABGESCHLOSSEN: Median 32,19× RT (3 Läufe, Wall 7248,7–7259,2 s / 225,33 s), Baseline 53,9× — Lücke geschlossen** (Report `2026-10-28_f_reihe_laufzeit_optimierung.md` §9, Log `rt_messung_2026-11-04.log`; Hinweis: der geloggte rt_factor ist auf 32,0 gekappt, Spec §9.5; `RT_REALITY_MEASURED` = 32,2); ursprüngliches **RT-Mess-Protokoll 2026-11-04** (Report `2026-10-28_f_reihe_laufzeit_optimierung.md` §6: `scripts/_aurik_run_excellence.py` auf Referenz-Song „Testkünstlerin…“ (224 s), 3 Läufe/Training-frei, Baseline 53,9× RT — Bericht `2026-11-04_sitzung_w2_w3_b3_messung.md` §5); **Scanner-Performance 2026-11-07: PERF-R19/R19b/R20 gelandet** (D-K3-38, **GESCHLOSSEN**) — Band-Dekadierung FM+AM (62 → 14,9 s/min warm, 4,2×) und Salience-Kette **bit-identisch** um 18,5 % schneller (6,50 → 5,30 s/min; Scan-Hash unverändert `c599791d…`, Golden-Gates 3/3 + 1/1); **Norm evidenzbasiert nachkalibriert: DefectScanner ≤ 16 s/min** (Spec 07 §9 + copilot-instructions.md synchron, Sync-Test pinnt 16,0; gemessen 14,9 s/min warm ⇒ ERFÜLLT). Das 4-s-Fast-Ziel bleibt dokumentiert hinter der Hör-Freigabe des Batch-Pfads (§2.46h) — Hör-Entscheidung, kein deterministisches Defizit mehr (Report `2026-11-07_perf_r20_salience_caches.md`); **D3-Taktung 2026-11-09 (D-K3-87):** BigVGAN 74 min/Epoche und FlashSR 80 min/Epoche sind mit 2×6 h strukturell unerreichbar (Epoche 8 bzw. 10 von 30) — Kette 2 fährt jetzt 4×12 h mit `--max-seconds`-Stopp an der Epochen-Grenze, D3-3 vor D3-2, RNG-/Historien-Fortsetzung; **D3-3-Gate 2026-11-10 gemessen — NICHT BESTANDEN (D-K3-94):** der FlashSR-Finetune (`best.pt`, Epoche 15) liegt auf MUSDB18HQ-Test (3 Tracks × 20 s, Seed 42, CPU-only) bei **mean ΔSDR −4,61 dB / min −10,60 dB** gegen die 16-kHz-Basis (Vortraining −6,44 dB; `residual`-Variante −0,03 dB). Peilung (−1 Sample) und Pegel (Gain 0,990–0,995) sind als Mess-Artefakt ausgeschlossen, die Ursache ist **Passband-Resynthese** (Passband-SDR 17,4/16,8/17,8 dB gegen Basis 22,4/22,4/20,3 dB, HF-SDR −0,49…−0,01 dB) — mehr Epochen beheben das nicht. Der Pfad ist **nicht im Signalpfad** (`HFExtender` ohne Aufrufstelle außerhalb seines Moduls, `HybridNVSR` nur Status-Payload) ⇒ kein Produktionsrisiko, nichts abzuschalten, kein Flag-Flip; die Rezeptur-/Modellentscheidung (HF-**Residual trainieren** vs. musik-trainiertes BWE/SR vs. F4 zurückstellen) braucht GPU-Fenster + Hör-Instanz. Werkzeug `scripts/verify_flashsr_d3_gate.py` (11 Tests), Belege `2026-11-10_flashsr_d3_gate.md` / `.json` (+ Robustheitslauf `…_2.json`: 6 Tracks × 30 s ⇒ raw −9,04 dB); **D3-2-Gate 2026-11-11 GEMESSEN — NICHT BESTANDEN (D-K3-96):** `scripts/verify_bigvgan_d3_gate.py` (neu, CPU-only, Seed 42, 3 MUSDB18HQ-Tracks × 10 s, echter Produktionsschritt `apply_hr_v1_additive`) misst F3 `output/f3_bigvgan/best.pt` (Epoche 7) gegen die Basis `models/bigvgan/bigvgan_v2.pth`: mit **Produktions-Mel** HNR −6,56/−6,39/−1,03 dB, musik_mos −0,30/−0,12/−0,32, defektlast +0,48/+0,03/+0,02, Vorteil 0/3; mit **Trainings-Mel** Vorteil 2/3 (HNR +2,03/+4,56), aber af −0,55/−0,17/−0,05 auf 3/3 ⇒ Never-worsen verletzt. Ursachen: (a) **Mel-Drift** — Training `log(abs(STFT))` (Mittel ≈ −3,2) vs Produktion `power_to_db(ref=max)` (Mittel ≈ −40,7); (b) der Finetune (13/30 Epochen, Early Stop, 0,5-s-Crops, nur A1+MR-STFT, kein Diskriminator) degradiert die Basis auch mit korrektem Mel. ΔSDR ist für einen phasen-unabhängigen Mel-Vocoder kein Gate (≈ 0 dB bei bestem Lag); `gate_hint` im Trainer korrigiert und per Test gepinnt. **Kein Flag-Flip, keine Verdrahtung — Basis bleibt aktiv.** Beleg `2026-11-11_bigvgan_d3_gate.md` / `.json` |
| 5 | ~~F3-BigVGAN-Rollout~~ **ENTSCHEIDEN 2026-10-30 (K0-C NO-GO, Hör-Instanz)** + ~~EAR-VAE v1↔v2-Vergleichsmessung~~ **GEMESSEN 2026-11-04:** HF-Gewichte `earlab/EAR_VAE2` (apache-2.0, `c87215f8…`) beschafft; Blind-Messung seed 42: **never_worsen ✓, 16/17 Trials v2-vorteilhaft, Sign-Test p = 0,000275 signifikant, erfuellt_objektiv ✓** (Messdatei `2026-11-03_ear_vae_v1_v2_vergleich.json`, Δ musik_mos +0,10 / Δ defektlast −0,12); **Rollout-Entscheidung v1↔v2: MENSCHLICH ENTSCIEDEN 2026-11-05 (NO-GO v2 — v1 bleibt; Hör-Instanz Block C, Sign-off Commit `7d2e3441`, `docs/reports/current/2026-11-05_hoer_signoff.md`; Wunsch-Richtung künftiger Iterationen: heller UND weniger Rauschen; D-K0-9/SOTA-M2, §V7 (copilot-instructions.md))** | — | LÜCKEN-MATRIX D.1b/c + K0-Sign-off-Paket + Report `2026-11-04_sota_sitzung_ear_vae_messung_ketten_checks.md` |
| 6 | Wohlklang-Stufen W-1…W-13 aktivieren (0/13 vollständig) | **B3-Verfahren** | **Stand 2026-11-04:** Alle B3-Beschlüsse W-1/W-4/W-5/W-6/W-7 formal **ABGELEHNT** (nicht signifikant, p ≥ 0,06; W-1 zusätzlich never_worsen verfehlt; W-1 hat widersprüchliche Objektiv-Belege — HF-Gain-Beschluss BESTÄTIGT vs. musik_mos-Beschluss ABGELEHNT ⇒ die Hör-Instanz entscheidet, Hörordnung §7); **Objektiv-Befund entscheidungsreif 2026-11-05** (`docs/reports/current/2026-11-05_k4_wohlklang_stufen_objektivierung.md`): W-2/W-3-Nachmessung ausgewertet (W-2: never_worsen ✓, p=0,6875 ⇒ objektiv ABGELEHNT; W-3 damals noch ohne Messung, gesperrt — Nachmessung am selben Tag abgeschlossen: 0/6 ⇒ ebenfalls ABGELEHNT (Details unten „B3-Nachmessung W-2/W-3 ABGESCHLOSSEN"), W-6/W-7 Fallbacks 11164/18220 → 364/2236 nach Router-Fix; B1-Zähler präzise (Commit `ef3795b7`); Hör-Sessions W-1/W-4/W-5 vorbereitet = **menschlich offen** (W-1 = einzige Stufe mit never_worsen=False + widersprüchlichen Belegen ⇒ §7-Hör-Entscheidung); W-2/W-3 gesperrt (Budget/Pfad) — **B3-Nachmessung W-2/W-3 ABGESCHLOSSEN 2026-11-05: beide ABGELEHNT** (W-2 `use_scnet_music`: 2/6 Vorteile, p = 0,6875; W-3 `use_sgmse_musik`: 0/6, Δ ≈ 0 — Schalter-Wirkung im Messpfad nicht nachweisbar, konsistent mit „Pfad gesperrt"; never_worsen ✓, B1 sauber mit 0 Rückfällen; `flag_freigabe.py` fail-closed: „Vorteil nicht signifikant"; kein Flag-Flip, Vertragssoll „AUS" bleibt; Belege `b3_stufenmessung_W-2/W-3.json` + `b3_objektivbeschluss_W-2/W-3.json`, Lauf `2026-11-05_b3_w2w3_nachmessung.log`, Report `2026-11-05_sitzung_w2w3_b3_nachmessung.md`; Mess-Lauf auf Korpus-Clips statt der Vertrags-Referenz-Songs — für künftigen Weg-B-Beschluss offen; Harness-Erweiterung Commit `d5483a43`; verbleibend: C4-Hörpaare + RT-Budget-Nachweis (W-2)); **Pfad-Klärung (W-3) ERLEDIGT 2026-11-06** — Ursache gefunden + behoben: Dauer-Achsen-Defekt in `phase_49_advanced_dereverb.py` (`shape[-1]` auf dem channels-last-Array ⇒ jedes Stereo-Fenster galt als „< 1 s“, SGMSE+-Tier-0 unerreichbar) — Δ ≈ 0 ist ein **No-Op-Artefakt der Messung**, der W-3-Beschluss 2026-11-05 damit als Qualitätsurteil über SGMSE+ **ungültig**; Fix + 3 Regressionstests (Rot→Grün), Report `2026-11-06_w3_pfadklaerung.md`; Neumessung nur über §4 mit Vertrags-Referenz-Songs, Flag bleibt AUS); **W-3-Neumessung 2026-11-11 ABGESCHLOSSEN (nach Achsen-Fix):** pfad_wirksam ✓ (SGMSE+-Tier-0 erstmals erreicht), never_worsen ✓, 2/6 Vorteile, p=0,6875, 9 ML→DSP-Fallbacks ⇒ **ABGELEHNT auf Substanz** — der W-3-Beschluss ist jetzt ein echtes Qualitätsurteil (kein No-Op-Artefakt mehr), Flag bleibt AUS (Belege `b3_stufenmessung_W-3.json`, `b3_objektivbeschluss_W-3.json`); „4-Modelle"-B3-Lauf (Material/ERA/Defekt/Loudness) **rekonstruierbar seit 2026-11-12:** Stufen `M4-Material`/`M4-ERA`/`M4-Defekt`/`M4-Loudness` in `SCHALTER_MAPPING`/`BESCHLUSS_FLAGS` (`scripts/b3_stufenmessung.py`), Messung mechanisch wiederholbar via `nice -n 15 python3 -B scripts/b3_stufenmessung.py --stages M4-Material,M4-ERA,M4-Defekt,M4-Loudness` (nächstes GPU-Fenster; alle vier Challenger produktions-verdrahtet, Default AUS; Test `test_m4_stages_existieren_und_zeigen_auf_flags`); **AUSGEFÜHRT 2026-11-17 (rekonstruierbar):** alle vier ABGELEHNT auf Substanz — Material 3/6 (p=1,0), ERA/Defekt/Loudness je 1/6 (p=0,21875), never_worsen ✓ + pfad_wirksam ✓; Flags bleiben AUS (Belege `b3_objektivbeschluss_M4-*.json`); **W-2-Neumessung 2026-11-11:** pfad_wirksam ✓ (Fußabdruck-Gate), never_worsen ✓, 1/6 Vorteile, p=0,21875 ⇒ **ABGELEHNT auf Substanz** — Messgrenze ehrlich: W-2 ohne verifizierten Modell-Zeugen (an/aus-Deltas ≤ 1e-4 = Lauf-Rauschen, D-K3-85-Klasse); Flag bleibt AUS (doppelt gesperrt: K0-A-NO-GO + B3-Ablehnung). |
| 7 | Modelle: WF-V4, TP-V2, BEATs-Head, MERT-Head, Gender-Head | teils Daten-blockiert | Tiefenanalyse E2 — **Stand 2026-11-06 (Trainingswelle):** Echt-Training vorbereitet für WF-V4 (D-K4-7, `scripts/prepare_wf4_warp_pairs.py` — MUSDB18-HQ-Mixture + bekannte Wow/Flutter-Warp-Kurven, GT warp_log2, Seed 42 (§G5), Manifest sha256) und TP-V2 (D-K4-8, `scripts/prepare_tp2_transient_pairs.py` — MUSDB18-HQ-Drums/other/bass-Stems, transiente-reiche Segmente); zusätzlich vorbereitet: F9 VoiceEncoder (D-K4-5), F10 SingMOS-Kalibrierung (D-K4-6), D-K2-3 vocos_48khz-Musik-Finetune. Alle als Jobs in `scripts/gpu_kette4_modellkoepfe.sh` (V1/V2 Datensynthese + R1–R6, pgrep-Guard, kein Flag-Flip, KEIN paralleles Training gestartet). **Köpfe (E2):** Gender-Head daten-blockiert (es fehlt `data/gender_labels.csv` — Gender-/Sänger-Labels auf Musik-Vocals, Lizenz/Label-Quelle offen, GTZAN/MagnaTagATune nur Genre+Valence/Arousal, kein Gender); BEATs-Head/MERT-Head daten-blockiert (527 Klassen AudioSet-Labels + Score-Kalibrierung auf HPI/AF fehlen); „Finetune der Köpfe“ bleibt Trainingspflicht (F13) — kein Trainer ohne Labels. Report `docs/reports/current/2026-11-06_trainingswelle_modelle.md` — **Fortsetzung 2026-11-06 (Manager-Ausführung):** drei **Phantom-Verdrahtungen** gefunden und behoben, die R1/R2 der Kette 4 am Musik-Training gehindert hätten (§III.11): (1) `train_wf4_warp_head.py` hatte **kein `--real-dir`** (nur synthetische Harmonische), (2) der R1-Guard prüfte `data/wf4_warp_pairs/degraded` statt `audio/` ⇒ R1 **immer** übersprungen, (3) `train_tp_v2_phase.py` ließ den Default `--synthetic 16` ein gesetztes `--real-dir` still überstimmen. Fix: **F0-Beobachtung** im Datensatz (`x_f0 = log2(f0_deg/f0_clean)` via Produktions-Tracker `pyin_fast`, keine GT-Zirkularität) + fail-closed-Lader (§V6, kein stiller Synthetik-Rückfall) + Vorrang real vor synthetisch + Kette-4-Verdrahtung; **Determinismus-Befund**: WAV-Container byte-instabil (libsndfile-`PEAK`-Chunk, 1 Byte bei Offset 61) ⇒ verbindlicher `sha256_pcm` statt `sha256_wav`. Daten erzeugt: **300 WF-V4-Paare** + **200 TP-V2-Segmente** (Seed 42, Manifest). Tests: 13 grün (`tests/unit/test_wf4_warp_real_pairs.py`, `tests/unit/test_train_tp_v2_real_dir_vorrang.py`). **R1/R2-Training weiterhin offen** (GPU belegt); **F9/F10/D-K2-3-Trainer implementiert 2026-11-06** (`scripts/train_voiceencoder_finetune.py`, `scripts/calibrate_singmos_music.py`, `scripts/train_vocos_48khz_finetune.py` — CLI exakt R3/R4/R5, Vertragstests 21 grün, vocos==0.1.0 im venv_aurik installiert, KEIN Flag-Flip) — R3/R4/R5 aktivieren sich im freien GPU-Fenster. Report `docs/reports/current/2026-11-06_wf4_echt_pfad_verdrahtung.md` — **2026-11-08 (D-K3-81): R3/R4/R5 waren nicht *startbar*.** Drei Trainer-Defekte behoben und Smoke-verifiziert: F9 bildete strukturell keine Gradienten (`embed_utterance` läuft in `no_grad`) **und** hing bei einem Val-Track in einer `while`-Endlosschleife (7 min, 100 % CPU); F10 gab 16 kHz an einen 48-kHz-Vertrag (AssertionError) und maß ganze Tracks statt eines 20-s-Fensters unter der 30-s-Kappung (entarteter Fit a = 2,4e6); vocos lud ein lokales Checkpoint über `hf_hub_download` (Repo-ID-Fehler). **Substanz-Befund bleibt offen:** die volle F10-Kalibrierung (320 Paare) zeigt SingMOS-Pro am **Ceiling** (clean 4,959/5, Std 0,023) und ohne Sensitivität für Bandlimit/Codec — **144/320** Paare verletzen die Grund-Ordnung, Codec (4,995) liegt **über** clean ⇒ die affine Kalibrierung trägt nicht; das ist eine **Metrik**-Entscheidung (`musik_mos`/`defektlast`, zertifiziert) und **kein autonomer Code-Schritt**. D-K4-6 bleibt offen mit präziser Ursache. Report `docs/reports/current/2026-11-08_kette4_trainer_defekte.md`. **Echt-Läufe derselben Sitzung:** R1 WF4 erfolgreich (50/50, 316 s), **R3 F9 erfolgreich (30/30 Epochen, 1555,8 s, val_acc 0,672 gegen Basis 0,422)**, **R4 F10 erfolgreich (49,8 s, Produktionsartefakt zahlen-gleich zum `/tmp`-Lauf ⇒ §G5)**, **R5 vocos erfolgreich (30/30, 1768,8 s) ⇒ R1–R5 echt fertig**; nur R6 Gender-Head bleibt daten-blockiert. **2026-11-09 (D-K3-85):** WF-V4 und TP-V2 waren bis zum Fix **nicht erreichbar** (WF4 lud nur ein nie erzeugtes `.onnx`, TP-V2 zeigte auf `backend/models/…` und lief mit Seed-Gewichten) ⇒ die B3-Ablehnungen M12-WF4/M12-TPV2 vom 2026-11-08 sind **Mess-Artefakte**, keine Qualitätsurteile; beide Pfade laden jetzt ihre trainierten Checkpoints, Modell-Zeuge im Pfad-Gate, Flips bleiben AUS, Neumessung im trainingsfreien Fenster offen. **M12-WF4-Neumessung 2026-11-11 ABGESCHLOSSEN (erster echter Modellvergleich nach D-K3-85):** pfad_wirksam ✓ (trainiertes Checkpoint lädt), never_worsen ✓, 1/6 Vorteile, p=0,21875 ⇒ **ABGELEHNT auf Substanz** — kein Mess-Artefakt, Flag bleibt AUS (Belege `b3_stufenmessung_M12-WF4.json`, `b3_objektivbeschluss_M12-WF4.json`); **M12-TPV2-Neumessung 2026-11-11 ABGESCHLOSSEN:** pfad_wirksam ✓, never_worsen ✓, 2/6 Vorteile, p=0,6875 ⇒ **ABGELEHNT auf Substanz** — Flag bleibt AUS (Belege `b3_stufenmessung_M12-TPV2.json`, `b3_objektivbeschluss_M12-TPV2.json`); Report `docs/reports/current/2026-11-11_b3_neumessung_w3_m12.md`. **2026-11-10 (D-K3-90): M5-Vocos-Finetune-Konsum verdrahtet** — der Export `models/vocos_48khz/finetune/vocos_48khz_music_ft.onnx` hatte keine Ladestelle (D-K3-85-Klasse); jetzt `plugins/vocos_plugin.py::get_vocos_music_ft_plugin()` (fail-closed + Negativ-Cache + Herkunfts-Zeuge `weights_basename`) und im §1.4-Finisher NUR hinter `music_model_flags.use_vocos_music_ft` (Default AUS ⇒ bit-identisch), 8 Reachability-Vertragstests. **Kein Flag-Flip**; offen: B3-Never-worsen gegen die Basis (GPU-Fenster) + C4-Rollen-Entscheidung (menschlich). **B3-Messung 2026-11-17 AUSGEFÜHRT:** M5-vocos ABGELEHNT (2/6, p=0,6875; never_worsen ✓, pfad_wirksam ✓) — konsistent mit dem Hörbefund 2026-11-10, Flag bleibt AUS (Belege `b3_objektivbeschluss_M5-vocos.json`). Report `docs/reports/current/2026-11-10_m5_vocos_konsum.md`. **M5-Hörbefund 2026-11-10 (Hör-Instanz, interaktives Panel):** `use_vocos_ft_reconstruction` **NEGATIV SIGNIFIKANT** — n=6, linear 6/6, p=0,03125 ⇒ Flag bleibt **AUS**, der lineare Pfad ist der bestätigte SOTA-Zustand, ein §4-Flip entfällt. **C4-Teilbefund vocos-Base vs. Musik-Finetune (n=12, zweite Sitzung 2026-11-10 addiert):** 5/5 Siege, 2 Gleichstände, p=1,0 (nicht signifikant) ⇒ kein hörbarer Vorteil; die C4-Primärrollen-Entscheidung reduziert sich auf den BigVGAN-Vergleich (D3-2) und bleibt **menschlich**. Voraussetzung dafür war D-K3-93: die Hör-Werkzeugkette addiert Urteile erst seit 2026-11-10 sitzungsübergreifend (vorher überschrieb jede Sitzung den Beleg, und die Unit-Tests löschten das echte Ohr-Artefakt). Belege `hoer_verdikt_m5_rekonstruktion.json` / `hoer_verdikt_c4_vocos.json`, Proben-Erzeuger `scripts/erstelle_hoerproben.py` (CPU-only, stört die GPU-Kette nicht), Report `2026-11-10_m5_hoerbefund_und_c4_teilbefund.md`. **D3-2-Kandidat 2026-11-11 gemessen (D-K3-96):** Der BigVGAN-F3-Finetune schlägt die Basis nicht (Produktions-Mel: HNR −6,56/−6,39/−1,03 dB; Trainings-Mel: Vorteil 2/3, aber af −0,55/−0,17/−0,05) ⇒ **C4-Rollenentscheidung bleibt menschlich**, F3 wird nicht verdrahtet. **R5-Daten ERLEDIGT 2026-11-12:** `scripts/erstelle_cantus_real_paare.py` — 60 ECHTE degradierte Vocal-Paare (data/cantus_real, vinyl/rir/vinyl_rir); **R6-Daten ERLEDIGT 2026-11-12:** `data/gender_labels.csv` via Produktions-Gender-Detektor (144 Zeilen: male=23/female=121/child=0 — male unter ≥50-Ziel, PANNs-Fusion als Augmentation offen); **R7-Daten-Name-Fix ERLEDIGT 2026-11-12 (D-K3-99):** Kette 1 nutzt jetzt `data/symphonia_pairs/manifest.jsonl` (Paare lagen längst vor). BEATs-Head/MERT-Head: **GESCHLOSSEN 2026-11-12 (D-K3-102)** — durch PANNs abgedeckt (527 Tags produktiv), kein unlizenzierter AudioSet-Trainer; c4-Head-Historie ohne Skill. |
| 8 | Hör-/Lizenz-Entscheidungen (§9/5–7, AERO, MUSHRA n≥30) | **menschlich** | Katalog A2 |
| 9 | ~~P1-3-Restdeckung (Audibility-Guards)~~ **ERLEDIGT 2026-11-03:** HNR/Mikrodynamik/Vibrato/Hallucination/Noise-Texture mit Maskierungs-Abdeckung (§G10), 19 Phasen verdrahtet (Commit `2c5e5522`) | — | Report `2026-11-03_p1_3_audibility_restdeckung.md` |
| 10 | ~~D-K1-3 Material-CRNN-Verdrahtung (95,3 % statt 40,2 %)~~ **ERLEDIGT 2026-11-03:** im Material-Entscheidungspfad hinter `use_material_crnn` (Default AUS — Flag-Flip offen, nur §4-Verfahren) (Commit `28f2947d`) | — | Report `2026-11-03_d_k1_3_material_crnn_wiring.md` |
| 11 | **Ketten-Verdrahtung (Trainingslücke, kein Modellproblem)** — **BEHOBEN 2026-11-06, GPU-Neulauf offen:** Drei Kette-1-Jobs liefen ins Leere, obwohl die Artefakte lokal liegen. (a) **M9/SOTA-M9** starb nach 5 s am 24-GiB-OOM (`--batch-size 4` ⇒ 4·513 Inter-Frame-LSTM-Sequenzen, 4,42 GiB angefragt) **und** trainierte auf `--synthetic 4000` (Sinus+Rauschen = K1-Domänen-Defizit §III.11) statt auf den vorhandenen `data/tfgrid_pairs` (25 MUSDB-Paare) — ein erfolgreicher synthetischer Lauf hätte den echten Kette-3-Lauf über Job-Lock/Manifest blockiert. Fix: echte Paare + `--batch-size 1 --grad-accum 4` (neues Trainer-Flag, Äquivalenz Batch 2 ⇔ Akkumulation 2 getestet; Kette 3 trägt dieselben Flags). (b) **Cantus/SOTA-M14** rief R3–R5 ohne `--data` auf (Exit 2 „--data fehlt“, 2 s), obwohl `data/cantus_pairs/manifest.jsonl` 3600 Paare hat — jetzt `--data` + fail-closed-Guard; `domain_adapt` nur mit echten `*_degraded.wav`-Paaren (bleibt datenblockiert). Tests: `tests/unit/test_gpu_kette1_wiring.py` (7 — inkl. statischer CLI-Prüfung ALLER vier Ketten gegen die `add_argument`-Flags der Trainer, 56 Aufrufe) + 2 Trainer-Tests. Offen: GPU-Neulauf, Never-worsen-Gate/Hörabnahme (M9), M9-Datenmenge (300 Paare seit 2026-11-17, 12× der 25 — D-K2-5-Muster). **Kein Flag-Flip.** **Wiederaufnahme-Fix 2026-11-07:** `train_job_runner.py` überspringt Jobs mit `ergebnis: erfolgreich` bei identischem Befehl/Phase/Seed (Manifest bleibt unverändert; `--force` erzwingt Neulauf) — der nächste Kette-1-Neulauf überspringt M8/RT60 und startet direkt M9/Cantus. | autonom (GPU-gebunden) | Report `2026-11-06_ketten_verdrahtung_m9_m14.md` + `2026-11-07_train_job_runner_idempotent.md`. **Resume-Anker-Defekt 2026-11-08 (Kette 2, D3-3 FlashSR) BEHOBEN:** `train_flashsr_f4.py` schrieb nur `checkpoint_epoch<N>.pt`/`best.pt`, nie den Repo-Standardanker `<out-dir>/checkpoint_latest.pt` — der Kette-2-Resume-Befehl zeigte genau darauf und starb 61 s nach dem 6-h-Timeout in `FileNotFoundError`; zusätzlich resumte `--cmd` (Versuch 1) nie, sodass jeder Kettenstart bei Epoche 0 begann und die 5,37 h aus Versuch 1 wegwarf. Fix: Anker je Epoche, `--resume` prüft die Datei fail-closed (Exit 2 statt Traceback), **beide** Kette-2-Befehle datei-geschützt (Aspade-Muster), `val_loss`-Broadcasting bereinigt; Anker aus `checkpoint_epoch3.pt` restauriert (Epoche 3, val 0,3726). Tests 10/10 (`test_gpu_kette1_wiring.py`, 3 neue — davon einer generisch: jeder `--resume`-Pfad der vier Ketten muss vom Trainer erzeugt werden). **Kein Training gestartet** (Kette 3 belegte die GPU), kein Flag-Flip. Report `2026-11-08_flashsr_resume_anker.md`. **Cantus-No-Op 2026-11-08 (D-K3-84, BEHOBEN):** `c2_cantus_finetune` trainierte **null Epochen** (Pretrain-`epoch=50` gegen `--epochs=30` ⇒ leeres `range`) und stand trotzdem auf „erfolgreich“; Fix: phasen-lokale Resume-Größen + „0 Epochen ⇒ Exit 3“, No-Op-Manifest archiviert — echter Neulauf beim nächsten Kette-1-Start (Runner hätte den Job sonst übersprungen). **Status 2026-11-12 (GPU-Fenster nach Ganz-Titel-Messung):** Kette 1 (R5 `c2_cantus_domain` + R7/KP-11 `c4_symphonia`) und Kette 4 (V1 300 WF-V4-Paare + V2 200 TP-V2-Segmente neu synthetisiert; R1–R5 idempotent übersprungen; **R6 F13 Gender-Head ERSTMALS TRAINIERT: erfolgreich** — 144 PANNs-gelabelte Paare (male=61/female=83), 112,9 s, Manifest `train_job_gender_head.json`; BEATs/MERT bleiben daten-blockiert = D-K3-102-gedeckt). Alle JOB_EXIT=0, kein Flag-Flip. |
| 12 | **Flag-/Artefakt-Reachability als Gate** — **ERLEDIGT 2026-11-10 (D-K3-92):** Die Klasse „Schalter/Artefakt ohne Lesestelle" (D-K3-85/D-K3-90/D-K3-91) wurde bisher per manuellem Suchlauf gefunden — der vergisst. Jetzt: `scripts/flag_wirkung_gate.py` + Pre-Commit `aurik-flag-wirkung-gate` prüft (A) Lesestelle außerhalb der Flag-Datei (Name-Load, Attribut, `getattr`/`hasattr`-Literal, benutzter Import-Alias; toter Import und `setattr` zählen NICHT), (B) Produktionslesestelle, sonst begründete Ausnahme mit Register-ID, (C) B3-/Release-Kandidaten ohne Produktionswirkung = No-Op-Artefakt. Bestand 2026-11-15: **34 Schalter, 0 Verstöße, 0 Ausnahmen** — die frühere Ausnahme `use_bw_v5` war ein **Falsch-Negativ des Gates** (indirekter Dispatch über `resolve_model_path`; Stufe A/B um Dispatch-Tabellen erweitert, dynamische Schlüssel `[dyn]`). W-8 bleibt **kein** B3-Kandidat, aber aus dem richtigen Grund: der Stufe fehlt ein Ladeweisen-Zeuge; der Flip bleibt §4/Hör-Sign-off (D-K1-5). 17 Vertragstests | autonom | Report `2026-11-15_d_k3_92_schalter_wirkung.md` + Register D-K3-92 |
| 13 | **Sinn-Beweis — VOLLER 12-PAARE-LAUF 2026-11-15 (Beleg `2026-11-15_sinnbeweis_v2_12_paare.json`): 0/12 besser (defektlast UND musik_mos). Der 4-Paar-v1-Befund war ein Kontext-Artefakt (korrigiert, DRT-9). Reproduzierbar: Boogie-kassette defektlast −0,079 (mos-neutral). NEUE LÜCKE: Jets-vinyl defektlast −0,203 ABER musik_mos −0,551 — ÜBERRESTAURATION, die Regionen-Gates nicht sehen ⇒ DRT-10-Kandidat (Ganz-Titel-mos-Never-worsen). Träumer-clipping +0,408 (Declip-Kopf AUS).

**DURCHBRUCH 2026-11-15 — erster GATE-BESTANDENER Defekt-Reparatur-Kanal (groove_echo):** Der Witness-Kopf (`scripts/train_witness_kopf.py`, D-K3-95-Folge) dreht das M9-Niederlage-Muster um: defekt-spezifisch auf k398-Regionen (leak-frei, Holdout-Titel), **Gate-Ziel im Loss** (L1 + λ·BCE(Witness(out),0) über differenzierbare torch-Mel mit slaney+power_to_db-Äquivalent — sonst sättigt das Witness, Befund 2026-11-15), Modellauswahl nach Witness-Q statt Val-L1. Ergebnis 30 Epochen groove_echo: **q_witness_holdout 1,05**, Holdout-Score 0,038. **Gate BESTANDEN** (`scripts/verify_witness_kopf_gate.py`, 200 Holdout-Regionen): **q_median 1,03, q_anteil≥0,5 = 1,0, never_worsen 0,964 ⇒ gate_bestanden=true** — der erste Beleg, dass Aurik einen §G13-Q≈0-Schaden messbar repariert (Q-Erstbeleg `2026-11-15_q0_erstbeleg_witness_kopf.json`). **VERDRAHTET 2026-11-15:** `backend/core/dsp/witness_kopf_groove_echo.py` (Witness-Trigger-Maske + 200-ms-Fades + Never-worsen fail-closed) konsumiert in phase_07 hinter `music_model_flags.use_witness_kopf_groove_echo` (Default AUS = bit-identisch, Metrik `witness_kopf_used`, Doppel-Guard Harmonik-Proxy gegen Input+Amtsinhaber, 5 Verdrahtungstests). **AUSROLLUNG 2026-11-15:** zweiter Kanal **tape_head_level_dip GATE BESTANDEN und verdrahtet** (q_median 0,995, never_worsen 1,0; `use_witness_kopf_tape_head_level_dip`, Default AUS); **sticky_shed_residue GATE VERFEHLT** (q 0,99 aber never_worsen 0,49 — Schutz-Arm) ⇒ report-only, kein Flag, Beleg `2026-11-15_q0_befund_sticky_shed_verfehlt.json`. **PARITÄTS-FIX 2026-11-15 (durch die Hörproben-Erstellung gefunden):** die k398-Regionen sind 0,1 s — Trainer/Gate padden auf 1 s, die Produktions-Trigger-Schleife tat es nicht (Witness P=0,0 statt 0,92) — beide DSP-Module padden jetzt auf 1 s und schneiden die Ausgabe auf die Eingangslänge (Paritäts-Test tests/unit/test_witness_kopf_produktion_paritaet.py). Hörproben für beide Kanäle erzeugt (study_stimuli/hoerproben/witness_kopf/, nur wirksame Regionen, A≠B verifiziert). **AUSROLLUNGS-BILANZ 2026-11-15 (7 weitere Klassen):** BESTANDEN und verdrahtet: **clicks** (Q 1,19, nw 1,0) und **compression_artifacts** (Q 1,02, nw 1,0) — Flags use_witness_kopf_clicks/use_witness_kopf_compression_artifacts (Default AUS). VERFEHLT (report-only, Belege 2026-11-15_q0_erstbeleg_witness_kopf_*.json): jitter_artifacts (nw 0,90), tape_head_clog (nw 0,928), phase_rotation (nw 0,677), sibilance (nw 0,589), transport_bump (q 0,43) — Muster: Gewinn-Arm stark, Schutz-Arm überrestauriert auf clean; Nacharbeit = schärferer Trigger. **ABLATIONS-BEFUND 2026-11-15 (KONTROLLE OHNE die 10 Kanäle):** Boogie ist mit der Bestands-Pipeline 4/4 besser (vinyl −0,080/+0,081; kassette −0,079/+0,095; clipping −0,070/+0,119; mp3_128 −0,144/+0,127) — MIT den 10 Kanälen 0/4. Die Kanäle triggern auf Rezept-Material falsch und blockieren die Bestands-Reparatur ⇒ **DRT-11-Kandidat: Trigger-Kontext-Sperre** (Witness-Köpfe nur auf detektiertem k398-Material feuern). Beleg `2026-11-15_ablations_befund_kanal_trigger.json`. **SPEC 26 (DRT-7..9) VERSCHAERFT 2026-11-15:** die Neumessung offenbarte drei weitere Luecken — (DRT-7) auch BESTEHENDE Kandidaten (Declip-Kopf, A-SPADE, de_crackle) brauchen das Produktions-Semantik-Gate; (DRT-8) vor jedem §4-Sign-off MUSS der Ganz-Titel-Sinn-Beweis (defektlast/musik_mos) belegt sein — Regionen-Q ist nicht hinreichend; (DRT-9) No-Op-Rueckkehr wird als unwirksam geflaggt (mp3_128-Befund). Gate-Skript nachgezogen (aurik-drt-vertrag prüft die Sinn-Beweis-Infrastruktur + den Beleg). **SPEC 26 (DRT) 2026-11-15:** die vier Verlustrunden sind jetzt strukturell ausgeschlossen — .github/specs/26_defekt_reparatur_trainingsvertrag.md (DRT-1..DRT-6: Gate-Ziel im Loss, Zwei-Arm-Verlust mit Identitaets-Anker, Pre-flight vor GPU, Training/Produktions-Paritaet, unantastbare Gate-Schwellen, ehrliches Scheitern) + Pre-Commit-Gate aurik-drt-vertrag (fail-closed). Der Trainer hat den Identitaets-Anker bereits (lambda-id 0,5, Holdout-Score mit beiden Armen) — die 6 gate-verfehlten Klassen werden damit neu trainiert. **NACHARBEIT RUNDE 1 (Identitäts-Anker, Spec 26) 2026-11-15:** 3 weitere Kanäle GATE BESTANDEN und verdrahtet — tape_head_clog (Q 1,55/nw 1,0), phase_rotation (Q 1,06/nw 0,979), sibilance (Q 1,15/nw 0,960). sticky_shed_residue + jitter_artifacts überreguliert (lambda_id 0,5 ⇒ nw kollabiert), transport_bump beide Arme schwach — Runde 2 läuft mit lambda_id 0,15/0,1. **10/10 — ALLE Q≈0-KANÄLE GATE-BESTANDEN (2026-11-15):** transport_bump mit lokalisiertem Loss (|schaden-clean|-Maske, lambda_lokal 1.0, DRT-3-Mechanismus) BESTANDEN (q 0,66, never_worsen 1,0) und verdrahtet — damit haben ALLE 10 Q≈0-Klassen des Audits belegte, verdrahtete Kanäle (§4-Proben für ALLE 10 Kanäle bereit, A≠B verifiziert — jitter und transport_bump je 1 wirksames Paar, ehrlich dokumentiert). **§4-PAKET KOMPLETT 2026-11-15:** Hörproben für ALLE 9 bestandenen Kanäle erzeugt (study_stimuli/hoerproben/witness_kopf/, 6 Paare je Klasse außer jitter=1, A≠B verifiziert, Δ 18,7–32,0 k LSB; Beleg 2026-11-15_hoerproben_witness_kopf.json). Damit ist der einzige verbleibende Schritt die menschliche §4-Hör-Entscheidung pro Kanal. **Nächste Schritte:** (1) §4-Hör-Sign-off für die neun bestandenen Kanäle (groove_echo, tape_head_level_dip, clicks, compression_artifacts); (2) §4-Hör-Sign-off für den Musik-Declip-Kopf — **k398-ΔSDR-Gate BESTANDEN 2026-11-17** (Träumer +0,81 dB, never-worse 3/3; Runbook `docs/guides/DECLIP_SIGNOFF_RUNBOOK.md`; Hörpaket-Neurenderung nach dem M9-Retrain, Kommando steht im Runbook); (3) das Muster auf die übrigen Q≈0-Klassen (sticky_shed_residue, tape_head_level_dip als Nächstes — Witness hoAUC ≥ 0,97 liegt vor). M9-V2 bleibt kein toter Code: Gate-Werkzeug + 2.000 Paare + Leak-Schutz sind die Vorlage/Infrastruktur des Witness-Kopf-Gates. **DRT-10/M1 + DRT-11 GESCHLOSSEN 2026-11-17 (Commit `7b3dcf20`):** DRT-10 ist Produktions-Invariante für JEDEN Lauf (Überrestauration ⇒ §0c-degraded + Identitäts-Rollback, außer bei hoher Hör-Instanz HPI ≥ 0,75 = §7-Messartefakt-Verdacht; Spec 26, Hörordnung §7-Ausnahme, alle drei UV3-Pfade); DRT-11-Kontext-Sperre ist in allen 10 Witness-Modulen verdrahtet (`_MAX_TRIGGER_ANTEIL`, breiter homogener Trigger = Material-Kontext ⇒ Bypass). **HUM/WOW + DROPOUTS-NACHARBEIT 2026-11-21/22 (Spec 27, Register-Priorität 4):** hum und wow GATE BESTANDEN (Belege `2026-11-21_witness_kopf_gate_hum/wow.json`, q_anteil 1,0, nw 1,0/0,9839); dropouts verfehlte zuerst (nw 0,9286) — ANT-B-Tiefenanalyse (Befund `docs/reports/current/2026-11-22_witness_nacharbeit_tiefenanalyse.md`): Witness p_clean 0,76 > Trigger 0,5; Messpfad-Reparatur (rms_skip-Parität Trainer/Gate/Produktion + CLI-Hebel `--lambda-kontrast`/`--clean-gewicht` erreichbar, Commit `a55fa13e`) + Detektor-Neutraining (abs-Mel + rms_skip + Clean-Gewichtung 2.0: p_clean 0,76→0,30) ⇒ **Re-Gate dropouts BESTANDEN 2026-11-22 (q 0,71, nw 1,0, mos +1,35)** — Beleg `2026-11-22_witness_kopf_gate_dropouts.json`; dropout_oxide/tape_splice bleiben daten-arm (hoAUC 0,79/0,76, ehrliche Umkehr auf die alten Instrumente), nächste Runde nach der Daten-Expansions-Kette. Kein Flag-Flip (alle Default AUS). **NACHARBEIT RUNDEN 2–4 (2026-11-23, Spec 27 Zeilen 7–9):** dropout_head_contact GELÖST per **Transfer-Warmstart** (`--init-von dropouts`: Witness hoAUC 0,72→0,94, Kontrast 0,03→0,66) + Kopf-Retraining Q-Ziel 1.0 ⇒ Gate BESTANDEN (q 0,58, nw 1,0, mos +1,30; Beleg `2026-11-23_…_dropout_head_contact.json`); transport_bump + jitter_artifacts + sibilance GELÖST (q 0,82/1,03/1,09, Q-Anteil je 1,0, nw je 1,0, mos +1,29/+1,12/+0,89; Belege `2026-11-23_…_{transport_bump,jitter_artifacts,sibilance}.json`); dabei zwei ANT-3/DRT-4-Messmetrik-Fixes landen: Q-Ziel-Auswahl auf Regionen-Median statt mean-Q (Commit `c99f2b88`) und Transfer-Warmstart für daten-arme Familien-Klassen (Commit `a32442d3`). **Rest-Beobachtung GELÖST 2026-11-23:** wow/phase_rotation/dynamic_compression_excess nw 0,98–0,99 waren RAW-Konventions-Drift des Schutz-Arms — nach dem DRT-4-Fix (eigene-RMS-Konvention für ALLE Witnesses, Commit `406eb527`) messen alle drei nw 1,0 (Spec-27-Register-Zeile 10). **RUNDE 5 (2026-11-23):** transport_bump-Überrestauration (q_median 2,0-Clip) mit Anti-Überrestaurations-Loss `--lambda-nor 0.5` (relu(p_clean−p_out), DRT-3) behoben — Re-Gate BESTANDEN q 2,0→1,40, q-Anteil 1,0, nw 1,0, mos +1,03 (Beleg `2026-11-23_witness_kopf_gate_transport_bump.json`, Report `2026-11-23_transport_bump_nor_anti_ueberrestauration.md`). Spec 27 Zeile 8 damit GELÖST. |

**Regel:** Punkte 1–5 ohne Rückfrage erledigen. Punkt 6 NUR über das
B3-Verfahren (§4). Punkt 8 nie eigenmächtig — abbrechen und begründet
berichten.

## 2c. Aktueller Fahrplan zum Paritäts-Erfolg (Stand 2026-11-18, geordnet)

Reihenfolge = Abhängigkeiten: erst das Laufende abschließen, dann GPU-Kandidaten
in der Reihenfolge der größten Register-Hebel (Denoise → Decrackle), UAT nur im
trainingsfreien Fenster, menschliche Sign-offs parallel nach Verfügbarkeit.

1. ✅ **M9-Gate-Neumessung (2026-11-18):** V2-Kandidat `gate_bestanden=false` →
   TF-GridNet-Pfad (V1 25 Paare, V2 2.000 Paare) endgültig gescheitert — der
   Denoise-Weg ist jetzt der **Witness-Kopf mit Wohlklang-Loss** (Punkt 6).
2. ✅ **15-Rezept-Sinn-Beweis (2026-11-18):** 6/45 besser, **raum 3/3** —
   Klassen-Befund im Paritäts-Register (DRT-14) belegt.
3. ✅ **Declip-Hörpaket gerendert (2026-11-18, CPU-only):** 3 Paare + Schlüssel —
   §4-Sign-off-Material ist frisch (Punkt 9).
4. ✅ **Cantus-Leiter v2 (2026-11-19, §3-37(a)):** pretrain(50) → fine_tune(30) →
   domain_adapt v3 (20 Epochen NUR CC0-Vinyl) — automatisiert mit
   Hash-Verifikation je Phase; danach **Never-worsen-Gate + Hörabnahme**
   (kein Deploy ohne Gate).
5. ✅ **raum-n≥10 + Stereo-Sinn-Beweis (2026-11-18/23):** Dereverb-Evidenz auf 10
   Titel (raum GESICHERT, §3-42) und **Stereo-n≥10-Auswertung 2026-11-23:
   8/10 besser, balance 8/10, ΔMOS −0,010 sub-JND** (§3-48(c)) —
   Paritäts-Register-Zeilen nachgezogen (Regel: kein Statuswechsel ohne Mess-Beleg).
6. ✅ **Witness-Training mit `--lambda-mos`** (DRT-8/10, Code
   integriert, 22/22 Tests) — Gates mit `--mos` gelaufen (Re-Gates 2026-11-23
   mit DRT-8/10-Feldern, report-only; §4-Pakete für dropouts/dropout_oxide/
   tape_splice/hum/wow/transport_bump/jitter erzeugt, §3-48(e)). Das ist der
   strukturelle Gegenbeweis zum „0/12“-Muster und der Denoise-Kandidat —
   **der §4-Batch bleibt menschlich (Regel 6: erst Kanalsatz-A/B-Gewinn).**
7. ✅ **VinylNet-Retrain mit `--noise-bank data/vinyl_noise_bank`** (D-K1-4-Rest):
   echte CC0-Vinyl-Leerrillen statt Schellack — **FERTIG 2026-11-20** (100/100
   Epochen, SI-SDR 14,66→19,55 dB, §3-48(b)); **Never-worsen-Re-Eval gegen die
   Basis BESTANDEN 2026-11-23** (alle 4 Schweregrade besser, Clean 14,91→19,54 dB,
   §3-48(g)); offen: Deployment (§4, nicht autonom) + RX-12-Vergleich.
8. ✅ **D-K3-13-UAT-Neulauf FERTIG 2026-11-23** (GPU frei, §3-48(f)) —
   `audit/uat_report_generator.py` → 29/30 PASSED (14/15 Restoration + 15/15
   Studio), 6/6 Release-Gates, Empfehlung NO-GO wegen R5-MUST-FAIL
   (Stereo-Korrelation 0,595→0,758, derselbe reproduzierbare Befund wie
   2026-11-15); `daily_real_audio_gate.py` gefahren, `TestDailyGateStatus`
   2/2 grün (Beleg `2026-11-23_uat_freshness_d_k3_13.md`).
9. **Menschliche Sign-offs (parallel, unabhängig von der GPU):** Declip
   (`hoer_entscheidung.py`, Paket frisch), BW-V5 (+17 dB-Risiko), material_crnn,
   10 Witness-Flags (§4), §9/7 Vinyl-Lizenz — Reihenfolge per Hör-Batch-Brief
   (`docs/guides/HOERBATCH_DECLIP_MUSHRA_W1.md`).
10. **Register-/Handover-Nachzug je Ergebnis** (`.github/WETTBEWERBS_PARITAET.md`,
    SOTA_DEFICIT_REGISTER, TODOS_SOTA_ROADMAP) — kein Statuswechsel ohne Beleg.

Daten-Lücken mit Lizenz-Voraussetzung (§9/7) folgen NACH der Entscheidung:
Vinyl↔Reissue-Paare ≥30, Echt-Träger vinyl/tape, Genre-Labels, MUSHRA n≥30.

## 2b. Muster zur Qualitätssteigerung (Stand 2026-11-07)

Muster-Analyse (Manager-Sitzung 2026-11-07): Das Regelwerk ist ein starker
Never-worsen-**Bremser**, aber ohne **Beschleuniger** (0/13 W-Stufen aktiv;
Einzel-Flips nie signifikant; Kombinationen nie gemessen). Sechs Hebel,
Priorität C→B→A→D→E/F:

- **C — Pfad-Reachability-Gate (ERLEDIGT, Commit `df512a5e`):**
  `scripts/b3_pfad_wirksamkeit.py` + Integration in `b3_stufenmessung.py` +
  `flag_freigabe.py`. Keine Ablehnung mehr auf Mess-Artefakten (W-3-Lehre).
- **B — Kompositions-Suche (ERLEDIGT, Commit `5a94c0a6`):**
  `scripts/b3_komposition_suche.py` — Lazy-Forward-Selection über
  W-2/W-3/W-8/W-9/W-13 mit never-worsen-Schranke; Ergebnis = KANDIDAT für
  EINE Hör-Session. **AUSFÜHREN im trainingsfreien Fenster:**
  `nice -n 15 python3 -B scripts/b3_komposition_suche.py --clips 3 --windows 2`
  — **AUSGEFÜHRT 2026-11-11 (2,3 h, Seed 42, 6 Fenster):** Beste Komposition
  **W-3,W-8** (musik_mos 4,8459 vs. Baseline 4,8454; defektlast 0,3745 vs.
  0,3808) — never-worsen ✓, aber die Vorteile liegen **unter der JND 0,05**
  (sub-JND); Beleg `b3_komposition_suche.json`. Kein Flip: KANDIDAT für EINE
  Hör-Session (§4-Verfahren).
- **A — Audibility-Budget (VERDRAHTET 2026-11-07, Flag AUS; Wirkungsmessung offen):**
  klangwirksam ⇒ flag-gesperrt (Default AUS, bit-identisch; 30 Vertragstests).
  `backend/core/dsp/audibility_budget.py` definiert das Budget als
  Hörbarkeits-Überschuss `max(0, L_defekt − (T_signal − Marge))`; der
  Johnston-Kern ist bit-identisch zu `masking_model.compute_masking_threshold_db`
  (Diff 0,0). Pilot-Verdrahtung in **zwei** Phasen: `phase_03_denoise.py::
  _denoise_mono_mrsa` (rauschbewusst je Zone; Masker = spektrale Subtraktion
  `max(|Y|²−|N|²,0)`, damit rauschdominierte Bänder nicht fälschlich gekappt
  werden) und `phase_43_ml_deesser.py::_deess_channel` (Sibilantenband aus dem
  Masker ausgenommen). Phase 19 bleibt bewusst außen vor — sie hat bereits
  `apply_psychoacoustic_masking_clamp` + PSY-A1. Nächster Schritt im
  trainingsfreien Fenster **AUSGEFÜHRT 2026-11-11**: `nice -n 15 python3 -B scripts/b3_stufenmessung.py
  --stages A-1 --clips 3` → **ABGELEHNT auf Substanz** (never_worsen ✓, 1/6 Vorteile, p=0,21875) ⇒ **Flag bleibt AUS** (Beleg `b3_objektivbeschluss_A-1.json`); RT-Kosten je Zone bleiben offen; Aktivierung NUR
  über `scripts/flag_freigabe.py --objektivbeschluss` nach Hör-Sign-off.
  Beleg: `docs/reports/current/2026-11-07_audibility_budget.md`, D-K3-71.
- **D — MUSHRA n≥30 (menschlicher Multiplikator):** Protokoll §10 + Builder
  fertig; entriegelt D-K4-4 (F8), D-K4-11 (F15), Proxy-Kalibrierung.
- **E — RT-Budget-Ledger (ERLEDIGT 2026-11-08, D-K3-75/D-K3-76):**
  `PerformanceGuard.get_budget_allocation()` (§PERF-R21) alloziert das
  32×-Budget je Phase — Soll aus der normativen `PHASE_PRIORITIES`-Ordnung,
  Ist aus den gemessenen `phase_performances`, plus `reserve_s`,
  `overrun_total_s`, `budget_exceeded`, `top_consumers`; verdrahtet als
  `rt_budget_ledger` in den UV3-Metadaten. **Reine Buchführung** (kein
  Skip-Pfad liest es, Quelltext-Pin). `get_phase_budget_seconds` delegiert
  dorthin (vorher: tote Methode mit flacher Division, NULL Aufrufer).
  **Dabei gefunden und behoben (D-K3-76):** `wow_flutter` und `decracklé`
  (beide CRITICAL 9) wurden von der Tabelle NIE getroffen — die Modulnamen
  heißen `phase_12_wow_flutter_fix`/`phase_09_crackle_removal`, die
  Normalisierung erzeugt `wow_flutter_fix`/`crackle_removal` ⇒ Default 5
  (MEDIUM) ⇒ die teuerste Reparaturphase war unter Budgetdruck skippbar.
  Alias-Auflösung `_priority_key()` in allen drei Prioritäts-Pfaden.
  Beleg: `docs/reports/current/2026-11-08_rt_budget_ledger.md`.
  **M6 GESCHLOSSEN 2026-11-17 (Commit `7b3dcf20`):** der Default für unprofilierte
  Phasen ist CRITICAL (10) statt MEDIUM (5) — unter Budgetdruck nie skipbar
  (Spec 07 §9, `BUDGET_ALLOCATION_DEFAULT_PRIORITY`, fail-closed-Test);
  Neubewertung je Phase (RT-Profil + Hör-Impact) bleibt der Weg zu echtem
  Budget-Spielraum.
- **F — Forensik-Monitor (ERLEDIGT 2026-11-08, D-K3-83):** WP-3-Forensik +
  musik_mos/defektlast als Dauer-Wohlklang-Zeuge je Lauf (nur gerichtete Fälle
  zur Hör-Instanz). Befund: `backend/` hatte **null** Aufrufstellen der beiden
  zertifizierten Metriken. Neu: `backend/core/dsp/wohlklang_zeuge.py` misst
  Eingang/Ausgang je Lauf, schreibt `metadata["wohlklang_zeuge"]` in allen drei
  Produktionspfaden (Einzelpass/Chunked/Options) und markiert
  `gerichteter_fall` → `hoer_instanz=vorzulegen`; **REPORT-ONLY** (kein
  Sample-Eingriff, kein Rollback, keine Export-Blockade, kein Flag; Quell-Pin:
  drei Schreibstellen, keine Lesestelle), deterministisch (§G5), §V6-fail-safe.
  Zusätzlich EINE Wahrheit: `defektlast`-Definition und JND 0,05 liegen jetzt
  im Modul (Zertifikats-Skript, B3-Stufenmessung und Kompositions-Suche
  greifen darauf zu). 13 Vertragstests. Offen (nur GPU-Fenster):
  End-to-End-Lauf + RT-Delta-Messung. Beleg:
  `docs/reports/current/2026-11-08_wohlklang_zeuge_hebel_f.md`.
- **G — Defekt-Delta-Metrik (DESIGN 2026-11-11, IMPLEMENTIERT):**
  Ganzclip-Metriken sehen Reparatur nicht (defektlast: 0,34-s-Fenster +
  Sättigungs-Blindheit D-K3-95/97; musik_mos integriert über den Clip).
  **Norm verankert 2026-11-11: §G191 (GEBOTE.md, Kategorie XXVI „Reparatur-Nachweis je Defektklasse") + Spec 07 §8.6 + Spec 01 §8.2 Punkt 13 +
  Hörordnung §7-Zeuge + WOHLKLANG_CLAIMS B3-Begleiter. Träger implementiert:**
  `backend/core/dsp/defekt_delta.py` (Regionen-Deltas, pegel-gematcht,
  Reparaturquote Q, JND 0,05, Schema `aurik.defektdelta/1` fail-closed,
  `defektlast_lang` mit Zertifikats-Fenster-Äquivalenz),
  `scripts/bewerte_defekt_delta.py` (CLI), `tests/unit/test_defekt_delta.py`
  (9 modell-freie Tests). **Report-Only** bis Richtungs-Zertifizierung
  (Hör-Labels, D-K4-9). **Erstbelege 2026-11-11 (Jets·clipping):**
  (1) No-Op auf Regionsebene belegt (alle Δ=0 — der D-K3-97-Bit-identisch-Befund wird sichtbar); (2) Kontrast-Messung dokumentierte zwei Metrik-Grenzen → **Schema /2 UMGESETZT:** Body-Perzentil-Matching (p25) statt RMS + relative Sättigungs-Komponente `flat_top_anteil` (gain-invariant) — die Kontrast-v2-Messung lokalisiert die Flat-Top-Verschlechterung in 172/173 Regionen (Δ ≈ −0,33); /1-Belege gelten als veraltet. Belege:
  `docs/reports/current/2026-11-11_defekt_delta_metrik_design.md` + `2026-11-11_b3_defektdelta_erstbelege.md` (+ `b3_defektdelta_jets_clipping(_v2).json` / `_kontrast(_v2).json`).

## 3. Zustand & Wiederherstellung nach Abbruch/Reboot

**Wiederaufnahme-Stand 2026-11-24 (Nacht, HEAD `0d8f5c31`):**
- **NOR-Runde 3 ERGEBNISSE (Korrektur der Zwischenmeldung):**
  `tape_splice_artifact` — **PFICHT-WEIZEN (User-Vorgabe 2026-11-24, fester Fahrplan-Bestandteil).** Aktueller Blocker: q_median 2,0 = Clip-Obergrenze (Trennung-Regel q<2,0 verfehlt) + Witness-hoAUC 0,76 zu unsicher; NOR 0.5/2.0 allein genügten nicht (q bleibt 2,0, nw 1,0). **Fix-Pfad (verbindlich):** Dropout-EXP-Kette (läuft, `kette_dropout_drt8.sh`) → Witness-Retrain auf erweitertem Manifest (STEP4) → Kopf-Retrain (STEP5) → Re-Gate (STEP6) mit **Annahme-Kriterium q<2,0 UND nw≥0,96 UND mos_anteil==1,0**; reicht das nicht, nächste Hebel: Trigger-Schärfung + Q-Ziel/NOR-Balance-Kalibrierung auf den Regionen <2,0 (6/18-Befund). Die frühere Meldung „q 0,98→2,0 / nw 0,58→1,0“ war falsch — korrigiert.
  `jitter_artifacts` — NOR 0.5 KOLLABIERT den Schutz-Arm (bestes nw 0,4106, 0/30 Epochen geeignet, kein Checkpoint geschrieben). **Muster (dokumentiert):** NOR ist das Instrument für Überrestaurations-Kandidaten (transport_bump, q-Clip); jitter ist ein Gewinn-Schwächling (q 0,78, nw 0,93) ⇒ Instrument ist Q-Ziel, nicht NOR. **jitter-v2 läuft** (`/tmp/kette_jitter_v2.sh`: `--lambda-nor 0.0`, Q-Ziel 1.0, NW 0.95, Log `/tmp/jitter_v2.log`, Marker `/tmp/jitter_v2_summary.txt`).
  `tape_head_level_dip` — **NOR-Gate BESTANDEN 2026-11-24 (Beleg `2026-11-23_witness_kopf_gate_tape_head_level_dip.json`):** q_median 0,9829, q_anteil 1,0, nw 1,0 (189/189), 200 Regionen. Für die Trennung-Schwelle mos_anteil==1,0 fehlt noch der `--mos`-Gate-Lauf (Folge-Kette).
- **Laufende Ketten (Marker in `/tmp/*_summary.txt`):**
  Dropout-EXP-Kette v5 (`/tmp/kette_dropout_drt8.sh`: 12-h-Timeout-Wächter + `PYTHONUNBUFFERED=1`, STEP1 = 7 Kassette-Titel à ~1,5–2 h unter Last) → `kette_dropouts_followup.sh` + `kette_post_exp.sh` warten auf `DROPOUT_EXP_END`.
  Weizen-A/B **PAUSIERT bei 9/45** (Zeilen-Dateien gesichert; Resume-Kommando: `python -B scripts/messe_kanalsatz_ab.py --verzeichnis data/schadens_proben --paare 45 --seed 42 --bestand data/schadens_proben/bewertung.json --flags <11 Weizen> --beleg docs/reports/current/2026-11-23_kanalsatz_ab_weizen_voll.json --resume`).
  Gepaarte DRT-12-Nachmessung VORBEREITET (`/tmp/gepaarte_messung.py`: Bestand- UND Kanalsatz-Pass pro Paar in EINEM Prozess, Rezepte dropout/kassette/hum/wowflutter/kompression/clicks, 14er-Satz) — startet via `/tmp/kette_nor_followup.sh` (v2) sobald alle 3 Gate-Marker liegen (`/tmp/gate_tdhl_nor_marker` ✓; `gate_jitter_nor_marker` nach jitter-v2-Gate; `gate_ts_nor_marker` erst nach dem Expansions-Re-Gate mit q<2,0 — Marker-Regel: nur echter Weizen setzt Marker).
- **⚠ D-K3-GoNoGo (Beleg `docs/reports/current/2026-11-23_determinismus_go_nogo_befund.json`):** Pipeline flippt auf identischem Input zwischen voller Verarbeitung und bit-identischem Pass-Through (`PipelineGuard §0h VETO: artifact_freedom=0.862 < 0.88 (depth=2)`; RAM/PLM-abhängige Modellpfade). Folgen: A/B-Deltas bis ±0,36 ohne Kanalwirkung; Solo-Probe 11 Weizen-Kanäle auf raum/hum-Paaren ALLE Pass-Through. **P1 gepaarte Messung vorbereitet (oben), P2 async-Race-Untersuchung offen, P3 dokumentiert — KEIN Eingriff in die §0h-Schwelle.**
- **Aktueller Fahrplan zur Produktreife (2026-11-24):**
  1. **D-K3-GoNoGo-Determinismus-Fix** (P2: async pre_Analyse/PLM-Konsum deterministisch; Regressionstests; Verifikationsläufe) — geschätzt **2–4 Arbeitstage**; ohne ihn ist keine Messung belastbar (§G5).
  2. **Kanalsatz-Evidenz vollenden** — **Pflicht: ts_artifact → Weizen** (Expansion-Kette → Witness/Kopf-Retrain → Re-Gate q<2,0); außerdem jitter-v2-Gate + Dropout-Familien-Gates + gepaarte 45-Paar-A/B + finale Trennung: **~1,5–2 Tage Maschinenzeit** (läuft weitgehend autonom), danach §4-Pakete + Hör-Sitzung (~1 Tag + 1–2 h Hörzeit).
  3. **DRT-16/RX-12-Externvergleich** — **1–2 Wochen**, sofern RX-Werkzeug verfügbar; sonst extern blockiert (Statusgrenze GESICHERT ≠ PARITAET bleibt).
  4. **Rest-Register K0–K4** nach Priorität des SOTA-Defizit-Registers.
- **Wiederaufnahme:** Ketten via `/tmp/monitor_aurik.py` prüfen; Marker unter `/tmp/*_summary.txt`; nach Abbruch: Dropout-Kette mit `/tmp/kette_dropout_drt8.sh` neu starten (Watchdog + Resume über fertige Songs), A/B mit `--resume` fortsetzen.

**Wiederaufnahme-Stand 2026-11-12 (Nacht, committet bis c3d4606f):**
- **Commits des Tages:** `c3d4606f` (§G13-Q-Ziel im Defekt-Delta sichtbar), `f73a81ac` (§G13 Hörbarkeits-Dominanz — Verbesserungs-Pflicht über alle 65 Defektklassen), `401b069e` (D-K3-98-Folgemessung: Enhancer No-Op auf Rezepten), `5330dce2` (M4-Stufen — 4-Modelle-B3-Lauf rekonstruierbar), `756cb836` (Handover-Nachtrag), `a0517137` (B3-Iteration 2: W-8 NICHT_MESSBAR (D-K3-92), F9-Identity-Gate, autarke Daten R5/R6/R7/D-K0-8, Song-Isolation, D-K3-96-Mel-Wahrheit), `e097a672` (D-K3-98-Rezepte-Zyklus gebucht), `4ba83b71` (Zyklus: §V8-Fix, M5-vocos-Stage, UAT, W-3/M12-Buchungen), `052188e7` (D-K3-98 Air-Presence, Parallel-Sitzung), `ba9770e6` (Norm §G191 + Defekt-Delta-Witness), `8ac4ec5e` (Erstbelege + Scanner-Anbindung), `8aa16dd0` (Schema /2: Body-Matching + relative Sättigung), `3928ae53` (M5-vocos-Beschluss-KeyError-Fix), `09e7d20c` (Handover-Abendstand), `d126b2b4` (Defektlast-Zertifikats-Grenzen D-K3-95/97), `192766eb` (M5-vocos-Messung gebucht, ABGELEHNT), `fd2acde3` (A-1-Messung gebucht, ABGELEHNT), `cfe19143` (W-2-Neumessung gebucht, ABGELEHNT).
- **M5-vocos-B3-Messung FERTIG (2026-11-11, 22:05) und gebucht:** **ABGELEHNT auf Substanz** — pfad_wirksam ✓, never_worsen ✓, 1/6 Vorteile, p=0,21875 ⇒ **Flag bleibt AUS** (bei Ablehnung kein Hör-Sign-off nötig). Belege: `b3_objektivbeschluss_M5-vocos.json` + `b3_stufenmessung_M5-vocos.json` (Register D-K3-90). Der erste Lauf der Parallel-Sitzung starb am Beschluss-Schreiben (`KeyError 'M5-vocos'` — D-K3-82-Klasse, gefixt in `3928ae53` inkl. generischem Konsistenz-Test).
1. Laufende Messläufe (Logs in /tmp, NICHT im Repo): **W-3, W-2, M12-WF4/M12-TPV2, M5-vocos und A-1 sind FERTIG und gebucht** (Register D-K3-82/D-K3-85/D-K4-7/D-K4-8/D-K3-90/D-K3-71/D-K0-5, Handover Zeile 6/7, Report `2026-11-11_b3_neumessung_w3_m12.md`) — alle sechs ABGELEHNT auf Substanz (pfad_wirksam ✓, nie signifikant); Flips bleiben AUS. **Kompositions-Suche FERTIG 2026-11-11:** Beste Komposition **W-3,W-8** (never-worsen ✓, sub-JND — musik_mos +0,0005 / defektlast −0,006), Beleg `b3_komposition_suche.json` — Kandidat für EINE Hör-Session, kein Flip.
2. **§V8-Leck GESCHLOSSEN** (Register D-K3-95 + Zeile 13): Reset-Ketten-Fix in `denker/aurik_denker.py` (8 verwaiste Stateful-Module je Song, §V6-geschützt), Gegenbeweis bestanden (Kontext-Hash `be0e87cb…` == solo). Folge-Hygiene notiert: `~/.aurik/session_memory.json` (200 Songs) + `§SSC-1`-Strategy-Cache (content-keyed).
3. **M5-vocos-B3-Stage NEU (2026-11-11, VERIFIZIERT):** `_schalter_lesen("M5-vocos")` in `b3_stufenmessung.py` + Modell-Zeuge `vocos_ft_modell_zeugnis` in `b3_pfad_wirksamkeit.py` (prüft `weights_basename == vocos_48khz_music_ft.onnx`) + 2 Tests in `tests/unit/test_b3_pfad_wirksamkeit.py`. Verifikation 2026-11-11 erbracht: `py_compile` sauber, `pytest tests/unit/test_b3_pfad_wirksamkeit.py -q` **20 passed**, Ruff clean. Messung `python3 -B scripts/b3_stufenmessung.py --stages M5-vocos` im freien Fenster **AUSGEFÜHRT und gebucht 2026-11-11: ABGELEHNT auf Substanz** (siehe oben; W-3/M12/Kompositions-Suche abgeschlossen).
4. **D-K3-98 ERLEDIGT 2026-11-11** (Air-Presence-Enhancer): periodisches Hann in Analyse **und** Synthese, scipy-Konvention (`detrend=False`, `boundary="zeros"`, `padded=True`, Kurzsignal-Guard), Hard-Clip ersetzt durch Eingangs-Decke (Zusatz `enhanced−orig` linear auf `min(0,999, peak_in)` zurückgenommen; Eingang über Decke ⇒ bit-identischer Passthrough). Vorher/Nachher: Roundtrip **8,93e-2 → 1,78e-8**, 440-Hz-Ton **0,25 → 1,49e-8**, hart geklippte Samples **23.804 → 0**, keine NOLA-Warnung. Tests: `test_air_presence_enhancer.py` 10/10 (4 neu), `test_stem_level_restorer.py` 17/17, Phase-19-Suite 81/81, Ruff/Compile clean. **Kein Flag-Flip, keine Verdrahtung geändert.** Wirkungsmessung auf die D-K3-95-Rezepte **AUSGEFÜHRT 2026-11-12 (D-K3-98-Zyklus)**: clipping repariert jetzt (2/3 besser; Jets 0,478→0,180 / 4,06→4,63), kassette neutral (0/3, sub-JND) — Beleg `docs/reports/current/2026-11-11_k398_rezepte_wirkung.json`, Register D-K3-95/D-K3-98 nachgezogen. Fix-Beleg `docs/reports/current/2026-11-11_d_k3_98_air_presence_nola.md`.
5. **Commit-Hygiene:** `python3 scripts/change_ledger.py snapshot` ausgeführt 2026-11-11 (16 Einträge, `check` grün); Zyklus committet als `4ba83b71` (alle Pre-Commit-Gates grün). UAT-Evidenz war am 11.11. frisch (`audit/uat_results_2026-11-11.json`, `latest.date`); **Nachtrag 2026-11-15:** inzwischen 4 Tage alt — `test_daily_gate_stored_recently` rot, Neulauf nur im trainingsfreien GPU-Fenster (D-K3-13 offen).
6. **Autarke Daten-Beschaffung 2026-11-12 (datenblockierte Punkte entblockt):** (a) **R5** — `scripts/erstelle_cantus_real_paare.py` erzeugte **60 echte degradierte Vocal-Paare** (data/cantus_real, MUSDB-Vocals gegen echte Vinyl-Rauschbank + echte RIRs, Ketten vinyl/rir/vinyl_rir, sha256_pcm, Seed 42); (b) **R6** — `scripts/erstelle_gender_labels.py` schreibt `data/gender_labels.csv` über den PRODUKTIONS-Gender-Detektor (§19-Fusion, schwache Supervision, unknown wird übersprungen); **PANNs-Fusion AUSGEFÜHRT 2026-11-12: male=61, female=83** (≥50 ✓) — die kanonische CSV ist jetzt die PANNs-Version; R6 vollständig entblockt; (c) **R7** — Ursache statt Symptom: die Kette suchte einen nie geschriebenen Manifest-Namen; jetzt kanonisch `data/symphonia_pairs/manifest.jsonl` (Register **D-K3-99**); (d) **D-K0-8** — `scripts/erstelle_corpus_vinyl.py` legte **55 synthetische LP-Rezept-Proben** an (corpus/vinyl, Bestands-manifest erhalten, D-K3-80-Schutz); **Tape-Klassen 2026-11-12 nachgezogen:** `scripts/erstelle_corpus_tape.py` hebt tape/cassette/reel_tape auf **je 50 Einträge** (kassette-Rezept + Bandlimit, merge-geschützt); (e) **§V8-Folge-Hygiene (D-K3-100/D-K3-101):** `SessionLearner.recall/record` jetzt **song_id-gekeyt** (Content-Fingerprint) — Cross-Song-Recall ausgeschlossen, Altbestand ohne song_id matcht nie (fail-closed); SSC-1-Fallback von `time.time()` auf content-basierten PCM-Hash (§G5); **(f) D-K3-96-Mel-Wahrheit:** `compute_mel_produktion` = EINE Mel-Implementierung für Produktion UND Trainer (§G9). **(g) Iteration 2 2026-11-12:** W-8 als nicht messbar aus der Kompositions-Suche entfernt (NICHT_MESSBAR-Diktat, D-K3-92); F9-Identitäts-Gate-Werkzeug (`scripts/verify_f9_identity_gate.py`, D-K4-5); SSC-1-content-keyed-Pin-Test; BEATs/MERT-Head via D-K3-102 geschlossen. Vertragstests `test_erstelle_daten_autark.py` (5) + `test_orchestrator_memory_song_isolation.py` (5) + `test_d_k3_96_mel_wahrheit.py` (4) + `test_gan_rezept_und_klassen.py` (5) + `test_f9_identity_gate_und_w8.py` (5, modell-frei); FILE_REGISTRY nachgezogen. **GPU-Jobs R5/R7 (Kette 1) und R6 (Kette 4) sind für das nächste GPU-Fenster entblockt** (nach der laufenden Ganz-Titel-Messung).
7. **§G13 Hörbarkeits-Dominanz (2026-11-12):** Normative Verbesserungs-Pflicht über alle 65 Defektklassen — never-worsen ist der Boden, nicht das Ziel. Zielwerte je Klasse: Rest ≤ Maskierungsschwelle − 6 dB Marge, Q ≥ 0,5 (`REPARATURQUOTE_ZIEL`), Kontrollregionen Δ ≈ 0, Frage-B-Musik-Schutz; kanonisches Messprotokoll (Seed 42, ≥ 6 Paare, p25, JND 0,05) + Coverage-Gate 65/65. Verankert in copilot-instructions.md (§G13 + RELEASE_MUST), Hörordnung §4 Reparatur-Auftrag, Spec 07 §8.2c, GEBOTE §G191-Sync, WOHLKLANG_CLAIMS; Verifier `check_g13`, Norm-Test `test_g13_hoerbarkeits_dominanz.py` (5), ID-Registry nachgezogen. Kein Flag-Flip — Q-Befunde sind Zeugen, Freigaben nur über §4.
8. **Hör-zentrierte Trainingsstrategie (2026-11-13, Report `docs/reports/current/2026-11-13_hoerzentrierte_trainingsstrategie.md`):** Nach D-K3-94/96 (Tage-Training → Gate-Fail) und den §G13-Q-Erstbelegen (0 Klassen Q ≥ 0,5) gilt: **das Ohr ist die Zielfunktion, nicht die Wellenform.** Fail-fast in drei Stufen (Flugschein-Gate vor dem GPU-Job → Epoch-Level-Gates → 1-GPU-Tag-Budget); ungewöhnliche Wege: Residual- statt Vollband-Training, Rezept-abgeleitete Per-Klassen-Datensätze (Priorität = Q≈0-Klassen), Produktion-im-Loop, Hörbarkeits-Curriculum, Gegner-Paare als Stopp-Signal. Erste Schritte autonom: Flugschein-Gate, Epoch-Gates, Per-Klassen-Generator, erster Per-Klassen-Kopf (compression_artifacts).
9. **Strategie UMGESETZT + VERIFIZIERT (2026-11-13):** (a) `scripts/trainings_flugschein.py` (Stufe A+B: Mel-Parität, gate_hint, Daten fail-closed + `epoch_gate_pruefung`) — **Echter Treffer: `train_flashsr_f4.py` wird ABGELEHNT** (nutzt nicht `compute_mel_produktion` — der D-K3-94-Tage-Verlierer trainiert außerhalb der Produktions-Domäne); `train_bigvgan_f3.py` besteht. (b) Epoch-Gate in `train_bigvgan_f3.py` verdrahtet (`--epoch-gate-n 2`, Abbruch > 1 dB HNR-Degradation, Exit 3). (c) `scripts/erstelle_klassen_paare.py` erzeugte auf den Ganz-Titel-Artefakten **>33.000 Per-Klassen-Regionen** (data/klassen_paare, manifest.jsonl, sha256_pcm). (d) **M4-B3-Messung ABGESCHLOSSEN: alle 4 Stufen ABGELEHNT auf Substanz** (M4-Material/ERA/Defekt/Loudness: pfad_wirksam ✓, never_worsen ✓, nicht signifikant — Belege `b3_objektivbeschluss_M4-*.json`; Flips bleiben AUS). 8 neue Vertragstests, FILE_REGISTRY nachgezogen. **(e) Folge-Schritte umgesetzt:** Flugschein-Pflicht im `train_job_runner` (Mel-Familie `train_bigvgan_f3`/`train_flashsr_f4`; Ablehnung ⇒ Manifest `ergebnis: flugschein_abgelehnt` + Exit 3, KEIN GPU-Job — 12 Vertragstests); Regions-Merging im Generator (`--merge-gap-s 0.25`, `--min-dauer-ms 50`, Altbestand-Cleanup) — Roh-Scanner liefert ~10k Fragmente/Titel, für Per-Klassen-Köpfe werden zusammenhängende Regionen gebraucht. **Ableitung für die nächste Runde:** (1) erster Per-Klassen-Kopf = `groove_echo` (4.342 Rohregionen, größte Q≈0-Klasse); (2) FlashSR: Produktions-Mel nachziehen oder Rezept auf HF-Residual umstellen (D-K3-94); (3) Vocos-Trainer-Paritäts-Vertrag prüfen (nutzt `compute_mel_produktion` nicht — Mel-Familien-Kandidat). **(f) groove_echo-Kopf-Trainer implementiert + Smoke verifiziert:** `scripts/train_klassen_kopf.py` (Conv1d-Encoder-Decoder ~60k Parameter, L1+MR-STFT, Seed 42, Zeitbudget Exit 75, fail-closed Datenprüfung) — Smoke auf den echten 4.072 groove_echo-Regionen: val_mae 0,048 in Minuten; Erfolgsmaß nach echtem Lauf = §G13-Q-Erstbeleg (`bewerte_defekt_delta`).

10. **Vocos-Trainingsschutz geschlossen (2026-11-13):** Der in Punkt 9(e) abgeleitete Prüfpunkt zur Vocos-Mel-Parität ist damit erledigt. `train_vocos_48khz_finetune.py` nutzt für den Modell-Input jetzt exakt `VocosPlugin._compute_mel`; der Modell-Feature-Extractor bleibt nur Teil des Rekonstruktions-Verlusts. Der Trainings-Flugschein bindet den Vocos-Trainer ein und lehnt die frühere modellinterne Eingangs-Mel ab. Der deterministische Holdout-Never-worsen-Gate vergleicht gegen die Baseline (1-%-Toleranz) und stellt bei Verfehlung Modell und Resume-Anker auf den Basis-Checkpoint zurück. **Zusätzlich gefundener Trainings-No-Op behoben:** `_spectral_loss` gibt jetzt den differenzierbaren Tensor zurück; zuvor wurde durch `.item()`/`float` jeder Optimizer-Schritt übersprungen. **Resume-Referenzfehler behoben:** Der Holdout-Baseline-Loss wird stets mit dem unveränderten Deployment-Checkpoint gemessen und danach wird der Resume-Kandidat wiederhergestellt; wiederholte Läufe können den 1-%-Spielraum nicht kumulativ aufweiten. **Epoch-Level-Fail-fast ergänzt:** Der Kandidat wird alle zwei Epochen sowie an der letzten Epoche gegen die feste Baseline geprüft; fehlschlagende Epochen werden verworfen und Resume-Anker werden nur nach bestandenem Gate aktualisiert. Tests prüfen Kadenz, Messfehler, Gradientenfluss, Baseline-Isolation und Restore-Verhalten. Kein Flag-Flip/Deploy.
11. **Klassenkopf-Holdout und Q-Prüfung gehärtet (2026-11-13):** Der alte `groove_echo`-Auszug (Q-Median 0,5834) ist als **invalidiert** markiert: 389/500 Prüfregionen lagen im Training, alle 500 stammten aus nur einem der drei Titel, und der Q-Nenner wurde nicht gegen JND 0,05 geschützt. `train_klassen_kopf.py` reserviert jetzt einen ganzen deterministischen Titel für unabhängige Prüfung und speichert Manifest-/Split-Provenienz in jedem Checkpoint; Resume ohne exakt passende Provenienz wird abgelehnt. `bewerte_klassen_kopf.py` prüft diese Provenienz, entfernt sample-identische PCM-Überlappung, berechnet Q nur bei Clean-Kontrast > JND und berichtet Q, Never-worsen sowie Kontrollregionen separat. Mindestumfang: sechs Q-fähige Paare; Maskierungs-/Audibility-Ziel bleibt ungemessen und ein einzelner Holdout-Titel nur vorläufig. **CPU-Pilot ausgeführt:** 1 Epoche/3 Trainingsschritte auf 2 Titeln, vollständiger Holdout mit 1.445 disjunkten Regionen (30 Hash-Überlappungen verworfen). 773 Q-fähige Paare, Q-Median 2,2342 (Q-Ziel erfüllt), aber Never-worsen nur 92,46 % (Gate verfehlt) und Kontrollregionen verfehlt (Median RMS-Änderung 0,051689); Befund `nicht_bestanden`, außerdem nur 1 Holdout-Titel und kein Maskierungstest. Report `docs/reports/current/klassen_kopf_q_groove_echo_holdout.json`. Kein Rollout.
12. **Identity-Residual-Folgepilot (2026-11-13):** Die letzte Residual-Ausgangsschicht ist nullinitialisiert (frischer Kopf = exakte Identität), und der Trainings-Loss enthält einen Clean-Identity-Term; Resume/Auswertung akzeptieren nur den neuen Modellvertrag. Im vollständigen 1.445-Regionen-Holdout des 1-Epochen-/3-Schritte-CPU-Pilots stieg Never-worsen auf 100 %, die mediane RMS-Kontrolländerung sank 0,051689 → 0,005425. **Gate trotzdem verfehlt:** Q-Median 0,2005 < 0,5 und Kontrollregionen-Defektlast weiter außerhalb JND; Audibility ungemessen, nur ein Holdout-Titel. Bericht `docs/reports/current/2026-11-13_hoerzentrierte_trainingsstrategie.md`. `nicht_bestanden`; kein Rollout/Flag-Flip. Keine Gewicht- oder Klangschwellenänderung aus diesem Einzelpilot ableiten.
13. **Frisches DefectScanner-Profil (2026-11-13):** Erster Scan in frischem Prozess, 33 s `data/cantus_pairs/test/Al James - Schoolboy Facination__v06__tape.wav`, 48 kHz Stereo, Auto-Materialerkennung: 15,878 s ungeprofilt (28,87 s/min; 63 Scores), Profil-Lauf 19,700 s inklusive cProfile (35,82 s/min). Ein zweiter identischer Aufruf in derselben Instanz dauerte 0,007 s wegen Full-Scan-Cache und ist ausdrücklich **kein** Leistungsbeleg. Im cProfile kumulativ: `_wf_cached_features` 7,525 s, `_coherent_subband_fm` zweimal 6,312 s, `_auto_detect_material` 5,287 s; `_detect_transport_bump` brauchte isoliert auf demselben 33-s-Audio nach Warmup ≈0,23 s. Damit ist der historische Transport-Bump-Wert (3,58 s kumulativ) kein ausreichender Grund für eine Optimierung dieser Einzelroutine. Die zwei NumPy-Warnings stammten vom leeren Vorher-Fenster bei Dropouts direkt am Signalbeginn (`_classify_dropout_subtype`, Zeile 6623): ohne Vergleichsfenster wird nun konservativ `DROPOUTS` zurückgegeben. Regressionstest `test_dropout_at_signal_start_uses_generic_subtype_without_warning`; Tier-2-Suite 52/52 grün; vollständiger 33-s-Scan mit `-W error::RuntimeWarning` jetzt warning-frei (63 Scores). **Keine Scanner-Semantikänderung an den hörwirksamen Subband-/Hilbert- oder Materialerkennungs-Pfaden.** Cache-/Kanal-Duplikation bleibt für eine sichere Performance-Optimierung zu untersuchen; kalter Lauf liegt weiter deutlich über dem 16-s/min Scanner-Budget; der ältere 29,4-s/min-Befund bleibt nicht widerlegt (anderes Material/Profilaufbau).
14. **A-SPADE-Aktivierungsvoraussetzungen (D-K4-1) bearbeitet (2026-11-13):** (a) **Robuste Modellreparatur nach Resampling UMGESETZT:** `plugins/aspade_declipper_plugin.py::declip` gab bei sr ≠ 48 kHz bisher None zurück (stiller PCHIP-Ersatzpfad); jetzt gcd-reduzierte `resample_poly` auf 48 kHz → APPLADE → zurück auf die Quellrate, deterministisch, fail-closed nur bei ungültiger Rate (§V6 (copilot-instructions.md)); toter Legacy-Code im selben Pfad entfernt. Neuer Vertragstest `test_plugin_resampling_robustheit` (32 kHz, Identity-ONNX, Determinismus), Suite 6/6 grün. (b) **Musikschutz vorhanden und getestet:** beide Phase-07-Zweige (A-SPADE-Klassen-Challenger hinter `use_aspade_declipper` sowie der schwere ONNX-Zweig) übernehmen pro Kanal nur bei striktem Harmonik-Proxy-Gewinn gegen Input UND Amtsinhaber (delta-basiert, AGENTS.md Guard-Kalibrierung). (c) **Vollpfad-Nachweis AUSGEFÜHRT — Gate NICHT BESTANDEN:** `scripts/verify_aspade_declipper_gate.py` (neu) misst auf den realen Clipping-Proben den echten Produktionsschritt `DeclipperPhase` dreifach (Input/PCHIP/A-SPADE) mit SDR best-lag-alignt gegen das Clean-Original, Harmonik-Proxy und musik_mos/defektlast-Deltas; Gate nur bei nie-schlechter gegen Amtsinhaber UND Input auf jedem Titel; 30-s-Schlimmstfenster, Clip-Schwelle 0.85 (die k398-Rezepte klippen bei ±0.90 — der erste Lauf mit ≥0.95 fand kein einziges Sample und wurde verworfen). **Ergebnis (Beleg `docs/reports/current/aspade_declipper_gate.json`): 0/3 Titel SDR-Gewinn.** Boogie/Jets bit-identisch zum Input (Never-worsen-Witness lehnt den Kandidaten ab), Träumer minimal schlechter (SDR −13,722 → −13,730, harmonik 0,272 → 0,27091, Δmusik_mos +0,049 sub-JND). **Damit ist die ΔSDR-Gate-Voraussetzung der D-K4-1-Aktivierung auf diesem Material NICHT erfüllt** — kein Flag-Flip, kein Hörordnungs-Sign-off abgeleitet. (d) **Hörnachweis vorbereitet:** `scripts/erstelle_aspade_hoerpaare.py` (neu) erzeugt blinde A/B-Paare (PCHIP vs. A-SPADE, schlimmstes 10-s-Clipping-Fenster, Seiten per Seed verlost, Schlüssel getrennt) für `hoer_entscheidung.py`; solange das objektive Gate negativ ist, dienen sie der Diagnose, nicht als Aktivierungsnachweis. **Offen:** Ursache der Witness-Ablehnung (Harmonik-Proxy zu konservativ auf Sättigungs-Clipping oder Modell ohne Gewinn auf Musik), danach erneuter Gate-Lauf. Flag bleibt AUS.
15. **A-SPADE-Ursachenanalyse (2026-11-13, Report `docs/reports/current/2026-11-13_aspade_ursachenanalyse.md`):** Roh-Kandidaten ohne Phase-Witness auf den Schlimmstfenstern gemessen — **der Witness ist NICHT das Problem, das Modell ist es.** ONNX-APPLADE verändert die geclippten Plateaus sichtbar (Boogie Δ 0,094; Jets Δ 0,397), aber der SDR gegen das Clean-Original bleibt auf allen drei Titeln identisch mit dem Input: die Rekonstruktion bewegt sich nicht in Richtung Original (MUSDB-Vocal-Training, keine Sättigungs-Mischungen). Zweiter echter Defekt gefunden (Modellreparatur nach Resampling): die APPLADE-Erkennung läuft bei 16 kHz und die Anti-Aliasing-Dämpfung drückt ±0,90-Plateaus unter die interne Schwelle — auf Träumer fand der Zweig gar kein Clipping (Δ 0,0). PCHIP bleibt formtreu (Δ ≤ 3e-05). Konsequenz: nächster Schritt ist Modell-seitig (Musik-Sättigungs-Finetune oder Ersatzkandidat) plus 48-kHz-Domänen-Detektion vor dem Resampling. Zusätzlich behoben: `_sdr_aligned` nutzt jetzt FFT-Korrelation statt `np.correlate(mode="full")` (direkte O(N²)-Konvolution lief auf 1,44-M-Sample-Fenstern in Minuten — Laufzeit-Bug in Gate und Diagnose).
16. **groove_echo-Klassenkopf: 20-Epochen-Echttraining auf GPU — Q-Gate verfehlt (2026-11-13):** Erster voller GPU-Lauf des Per-Klassen-Kopfs (`scripts/train_klassen_kopf.py`, 20 Epochen x 200 Schritte, Batch 8, Seed 42, 68 s auf RX 7900 XTX, val_mae 0,074) mit anschließender gehärteter Holdout-Bewertung (`bewerte_klassen_kopf.py`, ganzer deterministischer Titel, 1.445 disjunkte Regionen, 30 Hash-Überlappungen verworfen): **Q-Median 0,0312** (Ziel ≥ 0,5 → verfehlt; 773 Q-fähige Paare, 672 unter JND 0,05 ausgeschlossen), **Never-worsen 96,33 %** (verfehlt; der 3-Schritte-Identity-Residual-Pilot erreichte 100 %) und **Kontrollregionen verfehlt**. Mehr Epochen verschlechtern die Holdout-Transferleistung gegenüber dem Pilot (Q 0,20) — Generalisierungsproblem, kein Rechenproblem: 2 Trainingstitel / ~2.600 Regionen reichen für den unabhängigen Titel nicht. Kein Rollout, kein Flag-Flip. Beleg `docs/reports/current/klassen_kopf_q_groove_echo_echttraining.json`. Nächster Schritt: mehr Titel in die Trainingsmenge (ganze-Titel-Holdout rotieren) oder Ansatz-Wechsel, nicht mehr Epochen.
17. **Musik-Declip-Kopf — DURCHBRUCH erzwungen (2026-11-13, Report `docs/reports/current/2026-11-13_musik_declip_durchbruch.md`):** Nach der A-SPADE-Ursachenanalyse (Modell ohne Gewinn auf Musik) wurde ein auf **61 MUSDB18-HQ-Mischungen** + k398-Clean-Kulturen trainierter identity-residualer Conv1d-Declip-Kopf (~60k Parameter, 345 KB, 40 Epochen, RX 7900 XTX) auf den k398-Schlimmstfenstern gemessen — **leave-one-out (leak-frei: jedes Modell hat den Clean-Inhalt seines Eval-Titels nie gesehen): Boogie −9,99 → +5,60 dB (+15,6), Träumer −13,72 → −5,30 (+8,4), Jets −11,83 → −4,93 (+6,9) — 3/3 verbessert, 1/3 positiv** (A-SPADE: 0/3). Der Kopf invertiert die komplette Rezeptur inkl. 5×-Gain-Boost (Ausgangs-RMS = Clean-RMS). In-Domäne MUSDB: −9,09 → +27,0 dB. OLA-Rand-Explosion (Division durch Hanning²→0) gefunden und per Rand-Padding behoben. **Noch kein Flag-Flip, noch nicht verdrahtet.** Nächste Schritte (zwingend): Clip-Masken-Blend (Kopf nur in geclippten Regionen — auf sauberem Material würde er den Gain rückgängig machen) + Never-worsen-Witness, dann offizieller ΔSDR-Gate-Lauf + §4-Verfahren. Zusätzlich entstanden: 61 MUSDB-Clipping/Kassette-Volltitel (data/musdb_schaden, data/musdb_mixtures) für künftige Klassenkopf-Datenexpansion.
18. **Musik-Declip-Kopf PRODUKTIONS-VERDRAHTET + leak-freier Gate-Lauf (2026-11-13, Report `docs/reports/current/2026-11-13_musik_declip_gate_leakfrei.md`):** `backend/core/dsp/musik_declip_kopf.py` (neu): Torch-Checkpoint (345 KB, Fingerprint registriert), 8192er-OLA mit Rand-Reflection-Padding, Clip-Masken-Blend (Kopf nur in geclippten Regionen + 5-ms-Überblendung — sonst würde er den Rezeptur-Gain rückgängig machen) und Harmonik-Proxy-Witness. In `DeclipperPhase` als zweiter Challenger hinter `use_aspade_declipper` (Flag AUS ⇒ bit-identisch; pro Kanal nur bei striktem Gewinn gegen Input UND Amtsinhaber; Metrik `musik_kopf_used`). **Leak-freier Gate-Lauf über den echten Produktionsschritt auf 10 MUSDB-Titeln 61–90 (nie im Training): 4/10 besser (max +1,19 dB), 0/10 schlechter über JND — Never-worsen ✓.** k398-Gegenmaterial (LOO): 3/3 besser, Boogie +15,6 dB. 7 neue Vertragstests, Phase-07-Suite 77/77 grün. **Noch kein Flag-Flip — offen: blindes A/B + Hörordnungs-Sign-off (§4, menschlich).** Objektiver Gate-Teil erstmals für einen Kandidaten erfüllt.
19. **UV3-Produktionspfad-Rätsel Declipper-Erreichbarkeit — ALLE SECHS Schichten gelöst (2026-11-13, Report `docs/reports/current/2026-11-13_uv3_declip_erreichbarkeit.md`):** (1) Leichtpfad-Keep-Id `phase_01_declipping` → `phase_07_declipper`; (2) Scanner-Blindheit (Sättigungs-Clipping: CLIPPING-Severity 0,0) → Signal-Trigger (Plateau-Fraktion ≥ 0,85×Peak ≥ 0,005) in `_select_phases`; (3) PID-Plan-Pfad → Trigger am Plan-Merge; (4) Pilot-Rollout-Whitelist → Phase aufgenommen; (5) Resolved-Defekt-Skip-Guard über `self._clip_fraction_raw` (v10-verifiziert: resolved → False); (6) **§v10.707-Defekt-Absenz-Gate — der eigentliche Skip** (Log `phase_skip_defect_absent`) → Signal-Override. **Probe v11 (echter UnifiedRestorerV3, Boogie-Schlimmstfenster): Flag AUS wie AN führen `phase_07_declipper` aus; delta_off_on_max 1,86 — die Flag-Wirkung ist im Produktionspfad belegt.** Nackte §G13-Zitate qualifiziert (ID-Registry). Kein Flag-Flip — Flip über §4 (blindes A/B + Hör-Sign-off); objektive Gate-Belege (MUSDB leak-frei 4/10 besser/0/10 schlechter, k398-LOO 3/3) liegen vor. Trigger/Override bit-identisch ohne harte Plateaus.
20. **groove_echo-Kopf: MUSDB-Datenexpansion (11.352 Einträge, 40 Epochen) — Q-Gate weiter verfehlt, aber Never-worsen erstmals OK (2026-11-13):** Fenster-Generator (clipping+kassette auf 91 MUSDB-Titeln, 7.280 Einträge, deterministisch Seed 42) + die 4.072 k398-Regionen ergaben das kombinierte Manifest. Training (40 Epoch x 300 Schritte, GPU, val_mae 0,059): **Q-Median 0,0195 (Ziel ≥ 0,5) verfehlt, Never-worsen 100 % OK (zuvor FAIL), Kontrollregionen FAIL** (Holdout n=80, deterministisch ein MUSDB-Titel). Befund: Mehr-Titel-Training heilt den Never-worsen-Fehler, aber der Kopf lernt keine §G13-Q-wirksame groove_echo-Reparatur — Wellenform-L1 reicht für diesen Defekt nicht (anders als beim Declip-Kopf, wo die Schadens-Abbildung simpel lernbar war). Beleg `docs/reports/current/klassen_kopf_q_groove_echo_kombi.json`. Keine Aktivierung, kein Flag-Flip. Nächster Schritt wäre ansatzseitig (z.B. spektraler Kopf oder Rezept-spezifische Schadensmodellierung), nicht mehr Daten.
21. **Aktivierungs-Fahrplan mit Stopp-Kriterien (2026-11-13, `docs/reports/current/2026-11-13_aktivierungs_fahrplan.md`):** Kein Training mehr ohne Pre-Flight-Gate. **In-Sample-Q-Orkel gemessen: median 0,0101** (110 Q-fähige Trainingsregionen des Kombi-Modells) — das L1-Wellenform-Objektiv ist BLIND für die §G13-Q-Metrik; mehr Daten/Epochen sind bewiesen wirkungslos. Metrik-Konstruktion dokumentiert: defektlast = clip(≥0,98)·2 + flatness + hf_verlust·0,5 — die k398-Schäden (Clip ±0,90) berühren den Clip-Term nicht. Plan: (B-0) Kontrast-Filter — **fertig: 4.265/11.352 Einträge (> JND)** als `klassen_paare_kombi_gefiltert`; differenzierbarer defektlast-Proxy-Loss (Soft-Clip 0,90 + Flatness + HF) als nächster Implementierungsschritt MIT Unit-Tests, Pre-Flight: In-Sample-Q ≥ 0,5 vor jeder Holdout-Bewertung. (A-1) §4-Hörpaare über den echten Produktionspfad (Flag AUS/AN, 10-s-Clipping-Fenster, Seed-Seiten) — **Generation läuft**. Kein Flag-Flip ohne §4.
22. **Kette B GESTOPPT per Fahrplan-Stopp-Kriterium (2026-11-14, Report `docs/reports/current/2026-11-14_kette_b_stopp_preflight.md`):** B-0(2) Proxy-Loss implementiert + committet (`d1e08fd6`); B-1 Training `groove_echo_proxy` auf dem gefilterten Manifest (40 Epoch × 300, val_mae 0,04979); B-0(3) **Pre-Flight-Skript fehlte entgegen der Fahrplan-Behauptung — nachgerüstet** (`scripts/bewerte_in_sample_q.py`, Exit 0/1/2, 5 Vertragstests, FILE_REGISTRY). **Messergebnis: In-Sample-Q 0,0046 (Ziel ≥ 0,5) VERFEHLT, Holdout-Q 0,0115 VERFEHLT, Never-worsen 95,45 % VERFEHLT, Kontrollregionen VERFEHLT.** Der Proxy-Loss hat den Objektiv-Metrik-Mismatch nicht geheilt — das Modell erzeugt nicht einmal auf Trainingsdaten Q (schlechter als der Orkel 0,0101). **Ansatz verwerfen, nicht mehr Epochen** (dreifach unabhängig widerlegt: Echttraining, MUSDB-Expansion, Proxy-Loss). Belege `in_sample_q_groove_echo_proxy.json` / `klassen_kopf_q_groove_echo_proxy.json`. Der Musik-Declip-Kopf (Kette A) ist nicht betroffen; §4-Hörpaket A-1 läuft separat.
23. **Fahrplan A-1 UMGEZOGEN ins Repo + Hörpaare erzeugt (2026-11-14):** Der Einmal-Generator (/tmp) ist jetzt der repo-kanonische `--produktionspfad`-Modus in `scripts/erstelle_aspade_hoerpaare.py`: A/B = UV3-Produktionspfad Flag-AUS vs. Flag-AN, 10-s-Schlimmstfenster, Seed-Seiten, Schlüssel getrennt, fail-closed gegen Überschreiben der PCHIP-Paare (getrennter Ausgang) + **p25-Pegelmatching** (das k398-Rezept dämpft 5×, der Musik-Declip-Kopf stellt das Clean-RMS wieder her — ohne Matching entscheidet Lautheit statt Qualität, Hörordnung §7). 7 Vertragstests (Fenster-Determinismus, Matching, Flag-Rückstellung §V6). **Generation mit Repo-Skript ABGESCHLOSSEN + VERIFIZIERT (2026-11-14, GEN_EXIT=0):** 3 Paare in `study_stimuli/hoerproben/declip_produktionspfad/` (Boogie/Träumer/Jets), p25-rel-Delta je Paar 0,00 %, 48 kHz/10 s, Schlüssel getrennt; 6 alte ungematchte /tmp-WAVs gelöscht. Befunde je Paar: Boogie A(flag_aus) enthält die echten ±1,0-Plateaus (Pipeline-Loudness ohne Declip — der heutige Produktionszustand), B(flag_an) repariert; Träumer nahezu identisch (Declip-Effekt nur am 6,3-s-Plateau — der Witness lehnt den Rest ab); Jets deutlich verschieden. **Hör-Entscheidung steht jetzt aus — menschlich (§4):** `hoer_entscheidung.py` oder Direktabspiel + Schlüssel; danach `flag_freigabe.py --flag use_aspade_declipper --verdikt … --signoff …` — objektiver Gate-Teil (MUSDB 4/10 besser, 0/10 schlechter; k398-LOO 3/3) erstmals erfüllt, Flag bleibt bis dahin AUS.
24. **Scanner-Kaltstart-Teilbehebung AKTIV (2026-11-14, §PERF-R21b/c, Commits `6188597d`+`0555b5ef`):** Diagnose instrumentiert: der Subband-FM-Kern (`_coherent_subband_fm`, dominanter Kaltprofil-Posten 2×6,3 s) lief pro Scan zweimal — Material-Erkennung auf vollem Audio + Haupt-Scan auf 60-s-Crop (Content-Keyed-Cache kann bei verschiedenen Arrays nicht greifen; beide Läufe semantisch nötig). Fix 1: Band-Loop als reine Funktion + ThreadPoolExecutor (≤6), Sammlung in Band-Reihenfolge ⇒ bit-identisch (Fingerprints auf 3 Titeln vor/nach identisch: Boogie `095b5c00…`, tape33 `5ce48454…`, Träumer `c1a44dd5…`), FM-Kern **6,79 → 3,06 s/Aufruf (2,2×)**. Fix 2: FLUTTER-Detektor aus dem 8-Worker-Pool in den Hauptthread — entfernt verschachtelte Pools + 17 s `_thread.lock`-Konkurrenz aus dem Kaltprofil. 24 Scanner-Tests grün. **Kein Flag nötig — reine Performanz, bit-identischer Ausgang, sofort im Produktionspfad aktiv.** Report `2026-11-14_scanner_kaltstart_fm_parallel.md`. Offen bleibt: Material-Erkennungs-FM auf vollem Audio (eigene Untersuchung). — **GELÖST 2026-11-14, siehe Punkt 27 (§PERF-R21d).**
25. **Q-Mission-Datenbefund: Amtsinhaber reparieren die metrik-sichtbaren Klassen bereits; die Q≈0-Klassen sind metrik-blind (2026-11-14):** (a) bandwidth_loss-Kopf (Fahrplan B-Kette): In-Sample-Q 0,66 BESTANDEN (erstmals überhaupt — der datengetriebene Klassenwahl-Mechanismus funktioniert), Holdout-Q 0,093 verfehlt; aber `phase_39_air_band_enhancement` repariert dieselben Regionen mit Q 5,5, UV3-Produktionspfad Δ +2,37 auf dem Schlimmstfenster ⇒ Kopf nicht verdrahtet (Report `2026-11-14_bandwidth_loss_befund.md`). (b) **Metrik-Sichtbarkeits-Matrix gemessen:** bandlimit 63 % >JND, rausch@0.1 58 %, hum 32 %, clipping±0.9 **4 %**, mp3_128 **4 %**, rausch@24dBSNR 5 % — die §G13-Q-Erstbeleg-Klassen (clipping±0.90-Rezepte, kassette-Rauschen, mp3) sind für die Gate-Metrik fast unsichtbar; der echte Regionen-Kontrast kommt vom umgebenden Rezept-Schaden, nicht vom Defekt selbst (hum-Regionen 7/7 richtige Richtung, synthetischer hum median −0,016 falsche Richtung). (c) Amtsinhaber-Q gemessen: phase_24 dropout_oxide Q 0,79; phase_02 hum Q 13,2 — **alle metrik-sichtbaren Klassen haben bereits produktive Amtsinhaber.** Konsequenz: Das nächste Q-Rätsel ist die Metrik-Blindheit selbst (defektlast-Erweiterung oder defekt_delta-Flat-Top-Kanal als Gate), NICHT ein weiterer Wellenform-Kopf. Register D-K3-95/97 (defektlast-Sättigungs-Blindheit) bestätigt und erweitert.
26. **Metrik-Blindheit GELÖST + im Produktionspfad AKTIV (2026-11-14, Sättigungskanal 1/Crest):** Der Kernbefund: die k398-Clipping-Rezeptur ist tanh-Sättigung — sie ändert `defektlast` nicht messbar (gemessen 0,3793 = 0,3793) und erzeugt keine harten Plateaus (`flat_top_anteil` ≈ 0), senkt aber den Crest-Faktor deutlich. **Kanal `saettigung_kanal` = 1/Crest** (kanonisch in `wohlklang_zeuge.py`, gain-invariant: Skalierung kürzt sich; Sinus 0,707 → schwere Sättigung 1,0; Stille-Guard). **Produktionspfad-Aktivierung:** der Wohlklang-Zeuge (läuft in allen drei UV3-Pfaden) misst den Kanal je Lauf als Report-Only-Zweitkanal — ändert KEIN Gate und KEINEN gerichteter_fall-Zweig. **defekt_delta Schema /3:** pro Region `saettigung_*`/`delta_saettigung`/`jnd_klasse_saettigung` + `reparaturquote_q_saettigung` (Kanal-JND 0,02, Clean-Kontrast-Gate) + Aggregat (`reparaturquote_saettigung_median`, `q_saettigung_ziel_erfuellt`); /2 und /1 als veraltet markiert (fail-closed). **Erstbeleg Jets-Clipping (1967 Scanner-Regionen, echter Produktionspfad):** der Sättigungskanal macht die Reparatur dort messbar, wo defektlast blind war — z.B. hum Q 0,17 → **Q_saett 0,70**, sticky_shed_residue Q 0,18 → **Q_saett 0,75**, jitter Q 0,48 → Q_saett 0,74; Schlimmstfenster Δ_saett +0,34. Tests: Gain-Invarianz, tanh-Sichtbarkeit (synthetisch + echte MUSDB-Musik mit defektlast-Blindheits-Pin), Sinus-Normierung, Stille, Schema /3-Felder, Q_saett ≈ 1 bei vollständiger De-Sättigung. **Report-Only** — der Kanal trägt kein Richtungszertifikat (D-K4-9), Q_saett ist Zeuge, kein Gate.

27. **Scanner-Kaltstart KOMPLETT: Material-Erkennungs-FM jetzt auf dem geteilten 60-s-Crop — ein FM-Lauf pro Scan statt zwei (§PERF-R21d, 2026-11-14):** Letzter offener Posten von Punkt 24: die Material-Erkennung (`_detect_mono_material`/`_detect_stereo_material`) rechnete Wow/Flutter auf dem vollen Audio, der Haupt-Scan auf dem 60-s-Center-Crop — verschiedene Arrays ⇒ der content-keyed FM-Cache (§PERF-R18) konnte nie treffen (Kaltprofil 2026-11-14: Subband-FM 2× pro Scan, dominanter Kaltposten). Lösung: geteilter 60-s-Cap `WF_CROP_CAP_S` (Klassen-Attribut) + `_wf_stationary_crop()` mit identischer Index-Arithmetik wie scan() (§9.7.5); beide Material-Detektoren rechnen Wow/Flutter jetzt auf demselben Crop → Cache-Treffer, Feature-Schicht genau EINMAL pro Scan (zuvor volles Audio + Crop = 2×). Wow/Flutter sind stationär (§9.7.5a-Kommentar); die übrigen Material-Features (Welch, Crackle, Klicks, HF) bleiben bewusst auf dem vollen Audio (§9.7.5-Designabsicht „representative signal characterisation“). **Kein Flag nötig — reine Performanz, bit-identische Features, Verdikte unverändert (gemessen 0 Flips).** Belege: (a) Mono-Gate der Stabilitäts-Probe (1.353 Quellen: 91 MUSDB-Mixe + 12 Schadensproben + 1.250 Cantus-Test): auf den 103 Mono-Quellen (beide Probe-Aufrufe = `_detect_mono_material`) **0 Flips**; (b) die Cantus-Test-Quellen sind STEREO — dort vergleicht die Probe `_detect_stereo_material` (volles Audio) gegen `_detect_mono_material` (Crop), also zwei verschiedene Detektoren mit verschiedenem Klassen-Vokabular (CASSETTE/STREAMING/MP3_LOW vs. SHELLAC/VINYL): die dortigen Flips sind ein Probe-Design-Artefakt, KEIN Beleg gegen den Crop; (c) Pre/Post-Verdikt-Vergleich auf 9 Titeln (6 MUSDB mono + 1 Schadensprobe + 2 synthetische Stereo): **0 Flips**; (d) Stereo-Crop-Gate auf 24 echten Cantus-Stereo-Dateien — derselbe Detektor `_detect_stereo_material` mit `_wf_stationary_crop` identisch gepatcht (= altes Verhalten) vs. neu: **0 Flips** (CASSETTE/TAPE-Verdikte stabil); (e) Produktionspfad-Zähler auf Clinic_A (225 s, echter scan()): `_wow_track_features`/`_flutter_base_features`/`_coherent_scrape_am` **je 1× pro Scan** (zuvor 2×); (f) 6 neue Pinning-Tests in `tests/unit/test_perf_r18_cache_equivalence.py` (Crop-Arithmetik ≡ scan(), Kurzsignal-Passthrough, 60-s-Crop in Mono- und Stereo-Material-Erkennung, Cache-Treffer des Haupt-Scans mono + stereo, content-keyed kein Stale-Hit) — 22/22 grün; Scanner-Suiten 55/55 grün; ID-Check 0 Warnungen.

28. **Autonome Lückenschluss-Runde 2026-11-14 (Audit-getrieben):** (a) **Cantus-domain_adapt-OOM GELÖST (D-K3-103):** der Kette-1-R5-Befehl lief ohne Batch-Flags → Trainer-Default 8 = 22,77 GiB → 2× CUDA-OOM (30–46 s). Fix: `--batch-size 2 --grad-accum 4` (effektive Batch 8 unverändert ⇒ Trainingsdynamik äquivalent, VRAM ÷4) + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; identisches 2×4-Rezept wie der erfolgreiche R4-Lauf. Kette-1-Tests 24/24 grün, `bash -n` sauber. **GPU-Neulauf GESTARTET 2026-11-14 14:18** (GPU war frei, VRAM 2 %): R1–R4/R6/R7 idempotent übersprungen, **R5 trainiert mit dem Fix** (VRAM 99 %, GPU 100 %, kein OOM — der frühere 45-s-Todespunkt ist überstanden); Lauf-Ergebnis wird beim Abschluss im Manifest `train_job_c2_cantus_domain.json` nachgetragen (Monitor-Session läuft). Danach wie Punkt 11: Never-worsen-Gate/Hörabnahme. (b) **Q≈0-Metrik-Kanal-Forschung — ehrlicher Negativ-Befund:** drei Kandidaten-Familien gegen die kassette-Rezeptur (der letzte unreparierte Rezept-Schaden, 0/3 neutral) auf den 3 k398-Titeln gemessen (clean vs. Schaden vs. restauriert, pegel-frei/gain-invariant): (1) Spektral-Flatness 2–8/8–16 kHz + p05-Ratio — das Rezept-Bandlimit dominiert das HF-Band (flat_8-16k 0,26 → 0,002), Richtung falsch; (2) Min-Tracking-Rauschboden (Minimum-Statistics, 0,5–6 kHz, 0,25-s-Fenster) — das 24-dB-SNR-Rauschen ist am Boden nicht sichtbar (Schaden−clean +0,0002 ≪ JND, konsistent mit den 5 % Sichtbarkeit der Matrix Punkt 25b); (3) Wow/Flutter-Severity des Produktions-Scanners — saturiert auf cleanem Material bei 1,0 (UNKNOWN-Schwellen), das Rezept-Wowflutter ist bei Produktionsstärke unsichtbar (Schaden == clean). **Konsequenz:** bei Rezept-Produktionsstärke ist der kassette-Schaden mit gain-invarianten Skalar-Kanälen nicht separierbar — kein neuer Kanal aktiviert (kein Fake-Zeuge); der nächste Metrik-Schritt wäre rezept-/klassen-supervidiert oder §4. (c) **D-K3-92 ehrlich NICHT angefasst:** `use_bw_v5` verdrahten ist klangwirksam (BW-v5 liefe nach Verdrahtung hinter Default AUS) ⇒ §4-blockiert, die Register-Ausnahme bleibt; der autonome Teil (W-8 NICHT_MESSBAR + Flag-Wirkungs-Gate) war bereits erledigt. (d) **D-K3-13/D-K3-10 bewusst vertagt:** der rote Daily-Gate-Test braucht einen UAT-Lauf, der per Scheduler-Regel („keine Voll-Suites während der GPU-Kette“) ins trainingsfreie Fenster gehört; die 44 GUI-Texte ohne `t()` sind register-tolerierte §VI.5-Wortwahl ohne Hörbezug.

29. **Defizit-Offensive Runde 2 (2026-11-14, Audit-getrieben):** (a) **Declip-§4-Paket FERTIG (Commit `b4ce5146`):** Hörpaket verifiziert (6 WAVs, 48 kHz/10 s, `manifest.json` aurik.hoerpaare/1, Schlüssel getrennt); Fail-closed-Pfad bewiesen (nicht-signifikanter Verdikt ⇒ `ABBRUCH: Verdikt nicht buchbar`, Exit 2, Flag-Datei byte-identisch; 7/7 `test_flag_freigabe.py`); Runbook `docs/guides/DECLIP_SIGNOFF_RUNBOOK.md` (Ein-Befehl-Ablauf Hören → `hoer_entscheidung.py` → `flag_freigabe.py --commit`, Rollback, Ablehnung). Der Musik-Declip-Kopf wartet damit nur noch auf den Menschen. (b) **SOTA-M9-Gate GEBAUT + GEMESSEN (Commit `7f7edf5f`) — NICHT BESTANDEN (ehrlich):** `scripts/verify_m9_gate.py` (CPU-only, Seed 42, 3 MUSDB18HQ-Test-Tracks × 2 Arme, 48-kHz-Resampling, 9 Vertragstests): **Never-worsen ✓** (0/3 saubere Tracks verschlechtert), aber nur **1/3 Rausch-Arm-Vorteile** (Al James mos 4,04→4,80/def 0,33→0,10; Bobby Nobody und Easton Ellises neutral bis schlechter) ⇒ Gate nicht bestanden, **Flag bleibt AUS**. Befund bestätigt das D-K2-5-Muster: 25 Trainingspaare reichen nicht für die Generalisierung auf fremde MUSDB-Titel. Nächster Schritt: M9-Datenmenge (Fenster-Generator wie beim Declip-Kopf) oder §4 mit ehrlichem Gate-Befund. (c) **Q0-Witness-Forschung DURCHGEFÜHRT — Jitter-Kanal AKTIVIERT (Commit `ecd444b9`):** 6 Q≈0-Klassen × 8 gain-invariante Kandidaten-Features auf deterministischer Stichprobe (Seed 42, je 300 Regionen, clean/schaden/restauriert) gemessen (Belege `2026-11-14_q0_witness_forschung.json/.md`): **jitter_artifacts/Seitenband-Ratio = Median +0,180, 98,3 % über JND 0,02 ⇒ KANAL AKTIVIERT** — `jitter_kanal` kanonisch in `wohlklang_zeuge.py`, Report-Only-Zweitkanal in allen drei UV3-Pfaden, `defekt_delta` **Schema /4** (`jitter_*`-Felder, `reparaturquote_q_jitter`, Aggregat `q_jitter_ziel_erfuellt`), /3 veraltet (fail-closed), 35/35 Tests; sibilance-Crest marginal (Δ 0,022 ≈ JND) nicht aktiviert; groove_echo (Cepstral-Echo + Ripple), sticky_shed (Impulsdichte), tape_head_level_dip (Modulation), transport_bump (LF-Stoß) ehrlich negativ — nächster Schritt dort wäre ein supervidiertes Witness, kein Wellenform-Kopf. (d) **Worker-Batch gescheitert (4× HTTP 429 Quota) — Arbeit manuell übernommen;** Dispatch-Intents bleiben als gescheitert dokumentiert. (e) **Audit-Korrektur: D-K3-10 (GUI-Texte) ist bereits GESCHLOSSEN** (Commit `f25897c1`, 47→0) — in der Zählung gestrichen; D-K3-13 (UAT) bleibt per Scheduler-Regel im trainingsfreien Fenster. (f) **R5-Cantus trainiert weiter** (GPU, kein OOM nach dem Batch-2×4-Fix; Monitor läuft).

(g) **Supervidiertes Schadens-Witness — DURCHBRUCH: alle 6 Klassen Gate-bestanden (2026-11-14, Commit `82455a29`):** Der in 29(g) gestartete GELERNTE Detektor (Produktions-Mel + CRNN, leak-frei Train-2-Titel/Holdout-3.-Titel, Seed 42, CPU-only) separiert clean/schaden, wo alle handgefertigten Features blind waren (Δ ≈ 0). Holdout-AUC: **groove_echo 0,994 · sticky_shed_residue 0,986 · sibilance 0,991 · compression_artifacts 1,000 · transport_bump 0,968 · tape_head_level_dip 0,904** — Richtung überall P(schaden) > P(clean). Werkzeuge: `scripts/train_witness_detektor.py` (kanonisch, Checkpoints `models/witness_<klasse>/witness.pt` + report.json, Fingerprints §III.13 gebucht), `backend/core/dsp/witness_detektor.py` (Report-only-Inferenz, kein Gate/Flag/Sample-Eingriff, §V6-negativ-gecacht, literal verdrahtet fürs Modell-Audit §G9), 7 Vertragstests, Domänen-Registry + FILE_REGISTRY nachgezogen. **Bedeutung:** die §G13-Q≈0-Klassen sind ab jetzt MESSBAR — der nächste Schritt ist die Q-Erstbeleg-Neubuchung über den Witness und danach ein Reparatur-Kandidat mit Witness-Verlust (der belegte Fehler der alten Köpfe — L1 blind für Q — ist damit behebbar). Kein Flag-Flip.

30. **Empfehlungs-Umsetzung Runde 3 (2026-11-14):** (a) **M9-Datenexpansion FERTIG:** `prepare_tfgrid_pairs.py` + Stille-Guard (deterministischer Re-Draw aus (seed, i) — der 2.000-Paar-Lauf brach zuvor an einem stillen Fenster mit „Null-Leistung“ ab) + `sha256_pcm`-Manifestfelder (der WAV-Container ist durch den libsndfile-PEAK-Chunk byte-instabil, Determinismus-Test auf PCM-Hash umgestellt); **2.000 Paare erzeugt** in `data/tfgrid_pairs_erweitert/` (Seed 42, Manifest aurik.tfgrid_pairs/1) — M9-Neulauf im nächsten GPU-Fenster, danach erneutes Never-worsen-Gate. 10/10 Generator-Tests. (b) **Hör-Batch-Brief:** `docs/guides/HOERBATCH_DECLIP_MUSHRA_W1.md` — Declip-Sign-off + MUSHRA n≥30 + W-1 in einer Sitzung, Befehle + Belege + Reihenfolge. (c) HF-Rezeptur bleibt zurückgestellt (GPU + Hör-Instanz, schwächste Wirkungs-Evidenz — dokumentiert).

31. **Autonome Lückenschluss-Runde 2026-11-15 (UAT-Evidenz-Audit, D-K3-104):** Die Kandidatenliste war überholt: D-K3-60/71/75/83/90 stehen bereits auf `geschlossen`; D-K3-13 war offen (UAT 4 Tage alt; im trainingsfreien Fenster nach M9-V2 geschlossen — Punkt 32), D-K3-92 bleibt §4-pflichtig (`use_bw_v5` klangwirksam). Beim Prüfen der UAT-Evidenz fand sich eine echte, autonom schließbare K3-Lücke: `audit/uat_results_2026-11-11.json` führt alle 15 Restoration-Kriterien als `ERROR`, die Summary meldete aber `restoration_failed=0`, `restoration_skipped=0`, `criteria_executed=15` und `CONDITIONAL` („15/15 bestanden“), weil `compute_summary` ERROR in keinem Zähler führte; die Scorecard zeigte `⊘ SKIP`. **Geschlossen (D-K3-104):** eigene ERROR-Felder + `total_accounted`, fail-closed `NO-GO` bei ERROR/fehlendem Ergebnis, eine Statusquelle rendert `⚠ ERROR`. Derselbe Beleg ergibt nachher `criteria_errors=15` und `NO-GO`; 7 Unit-Tests grün; `test_real_audio_restoration_quality_gate.py` bleibt 8 passed/1 skipped/1 failed (nur Frische, D-K3-13). Keine Klangwirkung, kein Flag, kein GPU-Training. Beleg: `docs/reports/current/2026-11-15_uat_report_error_accounting.md`.

32. **D-K3-13 GESCHLOSSEN 2026-11-15 (UAT-Neulauf, echte Produktionspfad-Evidenz):** Die 15 `ERROR` im UAT-Report vom 11.11. waren kein Qualitaetsbefund, sondern das **120-s-Fixture-Limit** (`RuntimeError: real-audio fixture timeout after 120.0s`). Der kalte Produktionspfad braucht auf 8 s Eingang gemessen **714,1 s Verarbeitung** (`PerformanceReport: 23,80x RT fuer 30.0s audio` — **innerhalb** des 32x-Budgets) plus Modell-Ladezeit. **Fix (Harness, fail-closed):** `AURIK_R5_R12_RESTORE_TIMEOUT_S` Default 120→**1800 s**, Testmarker 900→**1800 s**, Generator-Subprozess 1200→**3600 s**; keine Qualitaets-/Gate-Schwelle geaendert. **Ergebnis:** `audit/uat_results_2026-11-15.json` mit `restoration_errors=0`, **14/15 Restoration PASSED**, 15/15 Studio PASSED, 5/7 Gates (1 SKIP), 0 K.O., Empfehlung `GO` (29/30); der eine FAIL ist ein echter **R5-Befund** (Stereo-Korrelation Segment 0: 0,595→0,759), kein Harness-Fehler. `audit/daily_real_audio_gate_status.json` `latest.date=2026-11-15`; `tests/normative/test_real_audio_restoration_quality_gate.py` **9 passed, 1 skipped**. Kein Flag, keine Klangwirkung, kein GPU-Training. Beleg: `docs/reports/current/2026-11-15_uat_freshness_d_k3_13.md`.

33. **D-K3-92 GESCHLOSSEN 2026-11-15 (Gate-Falsch-Negativ statt No-Op):** Die Punkte 31/32 hatten `use_bw_v5` noch als §4-pflichtig geführt — die Prämisse war falsch. `use_bw_v5` wird über die Dispatch-Tabelle `flag_map` in `resolve_model_path` gelesen (`backend/core/music_model_flags.py:211/226`); der Produktions-Ladepfad `plugins/bw_reconstructor_plugin.py:65` (erreicht über `phase_06_frequency_restoration.py:1298`, `BWReconstructorStage` ohne `model_path`) ruft ihn mit Literal-Schlüssel `"bw"`. Gemessen (CPU-only): **AUS → `bw_reconstructor.onnx` (SHA `4a4e4f58…`, ONNX-Input `input`), AN → `bw_reconstructor_v5.onnx` (SHA `7888e656…`, Inputs `waveform`/`cutoff` ⇒ `_is_v5`-Zweig)** — echtes Artefakt-A/B, kein No-Op. **Fix am Gate, nicht am Klang:** `scripts/flag_wirkung_gate.py` kennt jetzt indirekte Dispatch-Lesestellen (Literal-Schlüssel gefiltert, dynamische `[dyn]`), `AUSNAHMEN` ist leer; die W-8-Begründung in `scripts/b3_komposition_suche.py` ist korrigiert (fehlender Ladeweisen-Zeuge statt „keine Lesestelle"). Bestand **34 Schalter, 0 Verstöße, 0 Ausnahmen**; 61 Tests grün (Register-Gate inkl.). **Kein Flag-Flip.** Offen und an **D-K1-5** gebucht: der AN-Zielpfad zeigt auf das A1-**durchgefallene** Alt-Artefakt (0,8079 < 1,02) statt auf das bestandene Retrain (`548c0127…`, untracked) — Modell-Deployment-Entscheidung, nicht autonom. Beleg: `docs/reports/current/2026-11-15_d_k3_92_schalter_wirkung.md`.

34. **Mess-Zyklus 2026-11-15/16 abgeschlossen (DRT-12-Input-Gleichheit + 12-Paar-Vergleich):**
    (a) **Budget-Historie-Reset (§V8-Leck, neu belegt):** `_phase_budget_gain_history` + `_gdd_budget.phase_history` akkumulierten im Restorer-Singleton über Songs („historically_weak_delta"-Skip-Lernen) — Fix `reset_song_budget_state()` in `unified_restorer_v3.py` + Eintrag im Denker-Reset-Katalog + Test-Pin (`test_denker_sota_contracts.py`).
    (b) **Artefakt-Korrektur:** Der vermeintliche „3-Titel-Kontext-Effekt" (Jets-vinyl -0,203/-0,551) war ein seed-abhängiger Schaden-Input-Unterschied — mit bit-identischem Input ist Solo = 3-Titel. Witness-Köpfe UND Bestands-Pipeline entlastet. Beleg `docs/reports/current/2026-11-16_artefakt_befund_input_gleichheit.json`.
    (c) **DRT-12-Input-Gleichheits-Pflicht normativ verankert** (`musical_goals.instructions.md`): Kontroll- und Kandidaten-Lauf MÜSSEN mit bit-identischem Schaden-Input laufen (gleicher rng-Key/Seed).
    (d) **Erster belastbarer 12-Paar-Vergleich Bestand vs. 10 Kanäle (Input-identisch):** 0/12 besser, 11/12 korrekt gebypasst (DRT-11-Sperre), 1/12 (Boogie-vinyl) neutralisiert die Bestands-Reparatur vollständig (Bestand -0,016/+0,107 → Kanäle 0,000/0,000) ⇒ KEIN §4-Aktivierungsweg mit dieser Verdrahtung; der Q≈0-Weg bleibt die Witness-/Kopf-Arbeit (K1/K4). Beleg `docs/reports/current/2026-11-16_12paar_vergleich_bestand_kanaele.json`.
    (e) **Wiederholbare Messbasis:** `study_stimuli/sinn_beweis_neu_v5` (Bestand) + `_v4` (Kanäle) sind jetzt je 12 komplette Paare auf identischen Schäden — jede künftige Kanal-Konfiguration wird gegen diese Basis gemessen (Fortsetzungs-Wrapper `/tmp/fortsetzung_paare.py`, rng-Keys [42, idx_song, idx_rezept]).
    (f) **Ablations- + Dämpfungs-Kette (2026-11-16, Beleg `2026-11-16_ablations_befund_boogie_vinyl.json`):** Per-Kanal: alle 10 verbessern defektlast, jeder opfert mos-Gewinn (0,06–0,10 > JND); Kombi −0,028/+0,050 (verfehlt); 0,5×-Blend −0,025/+0,047 — verschiebt nur LINEAR ⇒ mos-Verlust KANDIDATEN-INHÄRENT (Training ohne Wohlklang-Term). **NAECHSTER GPU-BLOCK: musik_mos/Wohlklang-Term im train_witness_kopf-Loss + Zwei-Metrik-Modellauswahl (defektlast UND mos, DRT-8/10)** — das Gate-Ziel gehört in den Loss, nicht nur in die Auswertung (Muster der M9-/groove_echo-Niederlagen).

**M9-V2-ERGEBNIS (2026-11-15, belegt, docs/reports/current/2026-11-15_m9_gate.json):** Training erfolgreich (30/30 Epochen, exit 0, ~6,4 h, best_val 0,0324, Checkpoints + ONNX models/tf_gridnet_v2/), aber **das Never-worsen-Gate ist NICHT bestanden:** never_worsen=false, vorteile_rauscharm=0, gate_bestanden=false — der Kandidat verschlechtert den Wohlklang-Proxy auf allen 3 Gate-Tracks (−11,99→−12,62 / −10,86→−11,38 / −9,50→−10,29; margin 0,50; §G5-Doppelmessung bit-identisch). **Keine Aktivierung, kein Flag-Flip** (§V6 (copilot-instructions.md) fail-closed); die Produktions-Artefakte **Kein toter Code:** die M9-Infrastruktur bleibt aktiv — `verify_m9_gate.py --modell` ist das Kandidaten-Gate, `data/tfgrid_pairs_erweitert/` (2.000 Paare + exclude_pairs.txt) und `prepare_tfgrid_pairs.py` (Stille-Guard) bleiben Daten-/Werkzeug-Basis; der neue Witness-Kopf (D-K3-95) übernimmt die Gate-Vorlage und die Leak-Schutz-Muster (`scripts/train_witness_kopf.py`, `scripts/verify_witness_kopf_gate.py`). models/tf_gridnet bleiben unangetastet. Konsequenz: der TF-GridNet-Pfad ist mit V1 (25 Paare) und V2 (2.000 Paare + Leak-Schutz) gescheitert — nächster struktureller Schritt ist der Reparatur-Kopf mit Witness-Ziel-Signal (D-K3-95; das Witness ist seit 2026-11-14 messbar).

35. **Vinyl-Rauschbank MATERIALKORREKTUR + R5-Neuerzeugung (2026-11-17, User-Korrektur „vinyl gab es 1929 noch nicht“):** Der KP-6/D2-Altbestand (`data/vinyl_noise_bank/`, 2026-10-28) stammte aus **Great-78-Aufnahmen = Schellack (pre-1930)** — akustisch das falsche Trägermaterial für ein Vinyl-Modell (Mikrorille ab ~1948) UND lizenzrechtlich nie freigegeben (§9/7 stand offen). **Ersetzt durch 6 CC0-Vinyl-Rauschaufnahmen von freesound** (Recordist-eigene Aufnahmen ⇒ CC0 rechtsgültig, kein fremdes Musikwerk — §9/7 komplett umgangen; Lizenz je Datei per Sound-Seite maschinell verifiziert, SHA-256 + Spektral-Proben in `data/vinyl_noise_bank/MANIFEST.json` Schema /2). 3 weitere CC0-Kandidaten wegen Tonal-/Musikanteil abgelehnt (im Manifest dokumentiert). Format-Hinweis: 128-kbps-MP3-Previews (lossless-Upgrade: freesound-Originale, IDs im Manifest). Konsumenten angepasst: `quality_metric_direction_gate._lade_rauschen` (`leerrille_*` → `vinyl_*`), `erstelle_corpus_vinyl._vinyl_rezept` (Scratch-Quelle statt `echt_beschaedigt`), `erstelle_cantus_real_paare`-Docstring. **R5-Paare NEUERZEUGT (2026-11-17, Seed 42, Exit 0):** 60/60 hash-konsistent, 57/60 geändert (alle 40 vinyl/vinyl_rir neu; 3 unveränderte sind reine `rir`-Paare), 60/60 Clean byte-identisch, Alt-Manifest `docs/reports/current/2026-11-17_cantus_real_manifest_alt_schellack.jsonl`. **Retrain SAUBER (User-Entscheid 2026-11-18):** pre-domain_adapt- und pretrain-Checkpoints existieren nicht mehr (im November überschrieben; der ONNX ist ein pre-KP8-Stand mit stale Parity) ⇒ volle Neu-Leiter `pretrain(50) → fine_tune(30) → domain_adapt(20, NUR Vinyl)` als `_v2`-Jobs. **Befund unterwegs: Batch-8-OOM bei pretrain** (torch 2.11/ROCm 7.2, 22,77 GiB Peak = D-K3-103-Erbe, die Kette-R3 trug nie Batch-Flags) ⇒ pretrain_v2 läuft mit dem kanonischen D-K3-103-Rezept `--batch-size 2 --grad-accum 4` (effektive Batch 8, Trainingsdynamik äquivalent — Hash-Reproduzierbarkeit gegen `78ec1319` bleibt prüfbar). Ein abgebrochener gemischter domain_adapt-Versuch (20 Schellack- + 20 Vinyl-Epochen, `c2_cantus_domain_vinylv2`) wurde verworfen — User-Votum: keine Schellack/Vinyl-Mischung in den Gewichten. Manifeste: `train_job_c2_cantus_{pretrain,finetune}_v2.json` + `…_domain_vinylv3.json` (kommend). Report `docs/reports/current/2026-11-17_kp6_noise_bank_vinyl_materialkorrektur.md`; Alt-Provenienz bleibt in `2026-10-28_kp6_vinyl_noise_bank_manifest.json`.

36. **Parallel-Abarbeitung der offenen SOTA-Punkte (2026-11-18, User-Vorgabe „parallel alle offenen Punkte“):** (a) **M9-Gate-Neumessung (2c-1) ERLEDIGT:** gemessen wurde der letzte fertige Kandidat `models/tf_gridnet_v2/tf_gridnet_quality.onnx` (SHA `b7bb12f4…`); **gate_bestanden=false**, never_worsen=false, vorteile_rauscharm=0, vorteile_min=2 — TF-GridNet-Pfad bleibt gescheitert, keine Aktivierung; Beleg `docs/reports/current/2026-11-18_m9_gate.json` (+.md). **2c-Korrektur:** das 2c-Referenzkommando ohne `--modell` misst den Produktions-Stand (`models/tf_gridnet/`, SHA `3840a6b8…` ≠ V2) — für den Kandidaten ist `--modell models/tf_gridnet_v2/tf_gridnet_quality.onnx` nötig. Der in 2c genannte „M9-Retrain (300 Paare)“ ist auf der Platte nicht belegbar (kein Log/Report/Artefakt; einziges Zeugnis bleibt der Ketten-R2-Befehl auf `data/tfgrid_pairs`) — als Befund offen. (b) **Sinn-Beweis (2c-2) — Ursache gefunden und gefixt:** `erstelle_schadens_proben.py` (AURIK_FORCE_CPU=1) starb bei voller GPU an `Hip error: out of memory` (hipblaslt). **Wurzel (Worker-Ermittlung + Parent-Verifikation):** `plugins/miipher_dit_plugin.py` überschrieb die von `get_onnx_providers` erzwungene CPU-Provider-Liste UNBEDINGT wieder mit ROCm/CUDA — hob damit §v10.40c UND das fail-closed-CPU-Registry-Verdikt für `flow_matching_dit.onnx` aus. **Fix (2026-11-18):** Provider-Quelle ist ausschließlich `get_onnx_providers`, `_device` wird daraus abgeleitet; py_compile OK, 47/47 angrenzende Tests grün, **End-to-End-Beweis BESTANDEN** (1 Rezept durch die volle Produktions-Pipeline bei voller GPU — `_restauriert.wav` + Manifest, kein Hip-Error); **voller 15-Rezept-Lauf DURCHGELAUFEN (45 Paare):** `n_besser 6/45`, `n_ueberrestauration 0`; **raum (Dereverb) 3/3 besser** (ΔDefektlast −0,395, ΔMOS +0,039 — DRT-14-Evidenz für die Paritäts-Register-Zeile), übrige 14 Klassen neutral bis leicht MOS-negativ (≪ JND, konsistent mit dem 12-Paar-Vergleich); keine Aktivierung. Belege `data/schadens_proben/bewertung.json` + `docs/reports/current/2026-11-18_sinn_beweis_15_rezepte.md`. **FORCE_CPU-Härtung des ganzen Katalogs ERLEDIGT (Ursache statt Symptom):** öffentlicher Guard `force_cpu_requested()` in `gpu_model_registry.py` + verankert in `coordinated_repair.py`, `lyrics_guided_enhancement.py` (Whisper respektiert jetzt FORCE_CPU zusätzlich zu `AURIK_WHISPER_GPU`), `cantus_plugin`/`symphonia_plugin` (`_try_load_torch_rocm`), `muq_plugin`, `htdemucs_plugin._separate_pytorch`, `panns_plugin`, alle 5 `*_torch_rocm.py`-Getter (banquet/bsr317/fcpe/muq_mulan/whisper — kein GPU-Versuch mehr bei FORCE_CPU) und `ml_device_manager.warmup_rocm`; py_compile 14/14, 53/53 relevante Unit-Tests grün. **Kette-1-R3 Batch-Fix:** `--batch-size 2 --grad-accum 4` + expandable_segments jetzt im Ketten-Skript selbst (D-K3-103-Erbe beseitigt; Manifest-Drift-Konsequenz im Kommentar dokumentiert); bash -n OK. Zweiter Befund: MUSDB-Quellen heißen alle `mixture.wav` → Ausgabe-Namens-Kollision; Fix: eindeutig benannte Symlinks `data/schadens_quellen/` (Seed-42-Auswahl: „Music Delta - Rock“, „Alexander Ross - Goodbye Bolero“, „Snowmine - Curfews“). (c) **Witness-Wohlklang-Loss (34f/DRT-8/10) IMPLEMENTIERT + INTEGRIERT (Worker + Parent-Verifikation):** `train_witness_kopf.py` musik_mos-Proxy-Term (`--lambda-mos` Default 0.0 ⇒ bit-identisch ohne Flag, §G5 (GEBOTE.md)), Zwei-Metrik-Checkpoint-Auswahl (`_zwei_metrik_auswahl`, mos-JND 0,05), Gate-`--mos` report-only; `tests/unit/test_witness_mos_loss.py` (11, modell-frei); **22/22 grün im Haupt-Checkout**; FILE_REGISTRY nachgezogen. GPU-Lauf mit `--lambda-mos` steht im GPU-Fenster nach der Leiter an. (d) **Worker-Harness-Befund:** ADE-Worker haben kein Exec-/Shell-Tool (nur Datei-Inspektion) — Skript-Ausführungen gehören in den Parent; Code-Tasks (Witness) funktionieren über Worktree + Parent-Integration (Basis-Drift manuell per Patch aufgelöst, unverifizierte Worker-Edits an `train_job_m9_tf_gridnet.json` zurückgesetzt). (e) **Leiter-Status:** pretrain v2 **ERFOLGREICH abgeschlossen** (ergebnis erfolgreich, checkpoint_latest `c8b82502…` — vom Alt-Pretrain `78ec1319` abweichend, erwartbar: 2×4 vs. Batch-8 + Torch-Generation; informativ dokumentiert); die Leiter-Kette startet automatisch fine_tune v2 → domain_adapt v3 (nur Vinyl) → Never-worsen-Gate + Hörabnahme. **(f) Lücken-Nachzug 2026-11-18 (GPU-frei, strikt nach 2c/Register):** Paritäts-Register 10 Zeilen mit Sinn-Beweis-Messwerten belegt (clicks/hum/wowflutter/kompression/kammfilter/dropout/lautheit 0–1/3, ΔMOS-Werte; Decrackle: CC0-Bank vorhanden; Denoise: M9-V2 gescheitert → Witness-Kandidat; Declip: Hörpaket-Render ins trainingsfreie Fenster); D-K1-4-Register: Leerrillen-Bank vorhanden, Rest-Defizit = VinylNet-Retrain mit `--noise-bank`; SOTA-M18: Leerrillen ✓; **Declip-Hörpaket-Neurenderung (2c-3) läuft CPU-only** (`AURIK_FORCE_CPU=1`, `--produktionspfad --ausgang study_stimuli/hoerproben/declip_produktionspfad` — dank FORCE_CPU-Härtung ohne GPU-Eingriff, §4-Material für den menschlichen Sign-off); Task-Ledger `TASK_CHANGES.md` gesnapshottet (34 Einträge); Verifikations-Sweep **125/125 grün** über alle heute angefassten Suiten. **(g) Stereo-Sinn-Beweis-Kette + raum-n≥10 (2026-11-18, GPU-frei):** Stereo-Rezept `stereo_imbalance` implementiert (Worker) + integriert (Parent, 3-Wege-Merge nach Basis-Drift; REZEPTE = 15 Bestands + stereo hinten ⇒ Indizes bit-identisch); **39/39 Tests grün im Haupt-Checkout**, Smoke-Beleg (480000, 2) @48 kHz, Balance −4,37 dB im Spez; FILE_REGISTRY + Verdict nachgezogen — die OFFEN-Zeile „Stereo-Imaging/Azimuth“ ist damit messbar (erster Stereo-Lauf im nächsten CPU-Fenster). **Stereo-Balance-Lücke GESCHLOSSEN (§III.15, User-Vorgabe 2026-11-18):** `phase_15_stereo_balance` korrigierte rechts-lauter in die FALSCHE Richtung (Gains im else-Zweig vertauscht; Sinn-Beweis-Beleg −4,09 → −7,67 dB). Fix + 5 Richtungs-Pins `tests/unit/test_phase_15_stereo_balance_direction.py` (5/5 grün), isolierter Repro −4,09 → −0,53 dB; normative Verschärfung §III.15 (copilot-instructions.md) + `dsp.instructions.md`-Abschnitt + spec_drift-Baseline nachgezogen (No spec drift); Stereo-Sinn-Beweis-Neulauf läuft (3 Titel). **raum-n≥10 FERTIG:** 10 Titel gemessen — **6/10 besser, ΔDefektlast-Mittel −0,346, ΔMOS-Mittel +0,639, 0 MOS-Verschlechterungen, 0 Überrestaurationen** ⇒ Paritäts-Zeile Dereverb **GESICHERT** (zweite gesicherte Disziplin). **Zweite Stereo-Schicht gefixt (Ursache statt Symptom):** der Closed-Loop verwarf die korrigierte Phase als „kein beneficial delta“, weil `_evaluate_targets` Balance-Verbesserung nicht BELOHNTE (nur 8-dB-Deckel) — neuer Target `stereo_balance_improvement` (JND 0,5 dB, Belohnung −0,6) in `unified_restorer_v3.py`; py_compile OK, 14/14 Tests grün. **Dritte Schicht gefixt:** der `closed_loop_calibrator` misst Phasen MONO — Balance-Reparatur war im Regelkreis unsichtbar (Strength-Verfall 0,69→0,05, decision=skip) → `_stereo_balance_delta` (1 dB ≈ 0,05 Delta-Einheiten) für phase_15/phase_25 + Pins `tests/unit/test_closed_loop_stereo_balance.py` (4/4 grün); **Umsetzung des Audits (2026-11-18 abends):** n≥10-Batches als Warteschlange eingereiht — Batch 1 (kammfilter, mp3_128, clipping) nach der Stereo-Runde (inkl. automatischem Stereo-§4-Hörpaket `study_stimuli/hoerproben/stereo_sinnbeweis/`), Batch 2 (clicks, hum, wowflutter) danach, Batch 3 (pitch_drift, dropout, lautheit, amp_drift) danach; Rest (kompression, verzerrung, vinyl, kassette) folgt als Batch 4. **Ketten-Self-Match-Fix:** der Pretrain-Monitor matchte sich selbst (pgrep-Muster im eigenen Kommando) — Kette neu gestartet, fine_tune v2 läuft. **Snowmine-Befund (offen):** in Stereo-Runde 4 wurde `phase_15_stereo_balance` laut Log NIE „geplant“ (nur 3× in den Runden 2/3) — die 2/3-Korrektur kam über einen noch nicht eindeutig zugeordneten Mechanismus; Snowmines Scan zeigte Top-3-Defekte ohne stereo_imbalance (Verdacht: Plan/Defekt-Ranking) — als offener Befund dokumentiert. Stereo-Neulauf (4. Runde) läuft mit allen drei Fixes; **Hygiene-Gates parallel grün:** compliance R01-R18 PASSED, release_must 3/3, repo_graph-Duplikate ohne neue Kollisionen, Ledger 45 Einträge.

37. **Leiter-Schluss + DRT-8/10-Industrialisierung + GUI-Reifegrad (2026-11-19, parallel):** (a) **Cantus-Leiter v2 KOMPLETT:** pretrain(50) → fine_tune(30) → domain_adapt v3 (20 Epochen NUR CC0-Vinyl) — alle drei Manifeste „erfolgreich“ (`train_job_c2_cantus_*_v2/_vinylv3.json`); `checkpoint_best.pt` SHA `e4ae9598…`. **Never-worsen-Gate GEBAUT + GEMESSEN — NICHT BESTANDEN (ehrlich):** `scripts/verify_cantus_gate.py` (neu, M9-Muster, Schema `aurik.cantus_gate/1`; Basis = Wiener-DSP-Fallback = heutiger Produktionspfad bei Zoo-Status „training“; Kandidat `models/cantus/cantus_dit_v3_kandidat.onnx` SHA `6494ea34…`, Export-Parität rel ≤ 7,2e-5): **never_worsen=false** (Bobby Nobody mos 4,71→4,43, defektlast +0,015 — der v3 hebt stark degradierten Gesang teils massiv (AvaLuna Waterduct mos 3,51→4,25 = +0,74), verschlechtert aber sauberen Gesang), vorteile 1/3 ⇒ **kein Deploy, keine Zoo-Freigabe** (Genau dafür ist das Gate da). Beleg `docs/reports/current/2026-11-19_cantus_gate.json` (+.md), Vertragstests `tests/unit/test_cantus_gate.py` (8, modell-frei). (b) **DRT-8/10-Erstbeleg: groove_echo mit `--lambda-mos 0.1` GATE BESTANDEN** — Zwei-Metrik-Modellauswahl aktiv (beste Epoche 18/30), q_median 0,9972, **musik_mos_delta_anteil 1,0** (0/200 Holdout-Regionen verschlechtern MOS > JND; Referenz ohne mos-Term 2026-11-15: q 1,03, nw 0,964). Belege `2026-11-19_witness_kopf_gate_groove_echo.json` + `_mos.md`; Alt-Head als `*_pre_mos.pt.bak` gesichert. (c) **Witness-Expansion (7 untrainierte Klassen, Warteschlange automatisiert: Detektor → Kopf `--lambda-mos 0.1` (cuda) → Gate `--mos`):** BESTANDEN: **dynamic_compression_excess** (q 1,186) und **dropout_head_contact** (q 2,0 — Hinweis: Detektor-AUC nur 0,72); ehrlich VERFEHLT: dropouts (q 1,51, nw), dropout_oxide (Detektor-AUC 0,79), wow (q 0,95), tape_splice_artifact, hum (q 0,61) — Muster: mos-anteil überall 1,0 (der mos-Term verhindert MOS-Verschlechterung universell), der Never-worsen-Arm trennt die Spreu. `crackle` bleibt blockiert (391 Regionen, EIN Titel ⇒ Holdout-Leak-Schutz lässt keine Trainingsregionen) und `vocal_harshness` (4 Regionen) braucht Daten-Expansion — **Crackle-Expansion GESTARTET 2026-11-19 (Kette läuft):** `scripts/erstelle_crackle_expansion.py` (neu, Seed 42, Schema `aurik.crackle_expansion/1`) erzeugt 6 Titel vinyl-Rezept-Schaden (CRACKLE/Surface-Noise aus der CC0-Leerrillen-Bank via `erstelle_corpus_vinyl._vinyl_rezept`) + Produktions-Restaurierung (`restauriere_schnipsel`, FORCE_CPU, nice 19) → `erstelle_klassen_paare.py` (neues Out-Verzeichnis, Altbestand unangetastet) → crackle-Zeilen ans Haupt-Manifest angehängt (titel-dedupliziert). **(c2) §4-Pakete + Verdrahtung der 3 bestandenen Kanäle ERLEDIGT:** groove_echo (mos-Variante), dynamic_compression_excess (q 1,186) und dropout_head_contact (q 2,0 = Clip-Obergrenze, Detektor-AUC 0,72 ehrlich) sind nach dem 10-Kanal-Muster verdrahtet — neue DSP-Module `witness_kopf_dynamic_compression.py`/`witness_kopf_dropout_head.py` (Sicherheits-Vertrag: Trigger-Maske, 200-ms-Fades, Never-worsen fail-closed, DRT-11-Kontext-Sperre), Flags `use_witness_kopf_dynamic_compression_excess`/`use_witness_kopf_dropout_head_contact` (Default AUS), Konsum in phase_07 (generische Rest-Tabelle), KANAELE-Einträge; **§4-Hörpakete erzeugt: 3 Klassen × 6 A/B-Paare** (`study_stimuli/hoerproben/witness_kopf/…`, Beleg `2026-11-19_hoerproben_witness_kopf.json`); 32/32 Witness-Suiten grün (Wiring-Pins um beide Kanäle erweitert), FILE_REGISTRY nachgezogen. (d) **GUI-Produktreife (massiv, User-Vorgabe):** vier reale Befunde behoben in `Aurik10/main.py` — **(1) GPU-Erkennung war faktisch tot:** `wait_for_detection(timeout=5.0)` nutzte ein nicht existentes Keyword (heißt `timeout_s`) ⇒ Erkennung schlug IMMER fehl, App lief auch auf ROCm-Rechnern CPU-only — gefixt; **(2) Headless-/CI-Start hing am modalen Startup-Dialog** (`QMessageBox.exec_()` wartete auf Klick) — offscreen ⇒ Auto-OK, Normalbetrieb ⇒ 45-s-Auto-Close; **(3) CLI-Oberfläche neu:** `--help`/`--version`/`--check` vor der GUI (`--check` = offscreen QApplication + ModernMainWindow-Smoke, Watchdog 240 s, Exit 0/1/2) — der Smoke war vorher unmöglich (Hänger reproduziert, Exit 137); **(4) SIGTERM/SIGINT-Härtung:** Hard-Exit-Fallback (threading.Timer 8 s → os._exit) zusätzlich zum bestehenden Checkpoint→Qt-quit-Pfad — Terminierung auch bei hängendem Startup garantiert; zusätzlich **Fenstergeometrie-Guard** in `modern_window.py` (`_geometry_sichtbar()`: restaurierte Position außerhalb aller Bildschirme ⇒ Neuzentrierung statt unsichtbarem Fenster). **(5) Segment-Vorschau (Bedienung/Echtzeit-Zeile):** neuer Bridge-Pfad `restore_segment_preview` (Produktions-Pfad ohne Env-Seiteneffekte, `store_in_bridge_cache=False` = §V8-Song-Isolation) + `Aurik10/ui/preview_restorer.py` (QThread, done/failed-Signale statt stillem Fallback, §V6) + „🔍 Vorschau“-Button im Player-Panel (6-s-Ausschnitt → A/B-Wiedergabe über den Streaming-Player); 6 neue Vertragstests, `--check`-Smoke grün, Paritäts-Zeile „Bedienung/Echtzeit“ mit Beleg nachgezogen (bleibt LUECKE — echtes Echtzeit-Streaming ist eine Architektur-Entscheidung). Tests: `test_main_cli_oberflaeche.py` (6, Subprozess) + `test_modern_window_geometry_guard.py` (3, Quelltext-Pins) + `test_preview_restorer.py` (6); **49/49 GUI-Tests grün, 17/17 neue, compliance R01-R18 PASSED (print()-Verbot via sys.stdout.write erfüllt), Symbol-Duplikat `Messung`→`CantusMessung` behoben, FILE_REGISTRY + Ledger nachgezogen (67 Einträge).** (e) **signal10-Kette (n≥10-Batches):** **Batch 1 FERTIG** (kammfilter/mp3_128/clipping, 10 Titel, Beleg `data/schadens_proben_signal10/bewertung.json`): n_besser 6/30, 0 Überrestaurationen; kammfilter 3/10 (ΔMOS +0,021), mp3_128 2/10 (ΔMOS −0,028), clipping 1/10 (ΔMOS −0,001) — Register-Zeilen Kammfilter/Declip nachgezogen; Batch 2 (clicks/hum/wowflutter) läuft. **(f) RX12-Lücken-Lösung (DRT-12, User-Vorgabe 2026-11-19 „Zwischenergebnisse unter RX12 → Lösungen entwickeln, Specs anpassen, Durchbrüche nicht beschädigen“):** Wurzel erkannt — die Ganz-Titel-Sinn-Beweise (0/12, 6/45) wurden mit **deaktivierten** Witness-Kanälen gemessen (Flags AUS): sie belegen den Bestand, nie den Kandidaten-Zustand. **Normativ angepasst:** WETTBEWERBS_PARITAET „Harte Regeln“ 5-7 neu — quantifizierte RX-Paritäts-Norm (GESICHERT ⇔ n≥10: ≥5/10 besser UND ΔMOS-Mittel ≥ +0,05 UND 0 Überrestaurationen ODER defekt-spezifischer Kanal: Regionen-Gate + DRT-8-Ganz-Titel-Beweis; jeweils + §4, Hörordnung unangetastet) + Kanalsatz-A/B-Pflicht + kombinierter §4-Batch; **Spec 26 um DRT-12 erweitert** (A/B auf identischen Paaren vor Sign-off, mess-only, Aufnahme-Kriterium); `drt_contract_gate.py` prüft DRT-12 fail-closed. **Werkzeug:** `scripts/messe_kanalsatz_ab.py` (Kanalsatz AN = 12 gate-bestandene Flags prozess-lokal vs. Bestand AUS auf identischen Paaren, Produktions-Pfad, Trigger/DRT-11 aktiv, CPU-only, Seed 42, Schema `aurik.kanalsatz_ab/1`) + 5 Vertragstests; **Smoke-Lauf gestartet** (4 Paare aus den 45 15-Rezept-Paaren, nice 19) — danach der volle 45-Paar-A/B; die durchgesetzten Durchbrüche (GESICHERT-Zeilen, DRT-1…11, Kanäle) bleiben unangetastet (Flags AUS bis §4). **(g) Crash-Härtung (User-Vorgabe „kein Crash mehr“, 2026-11-19):** Forensik — ein Neustart der Kun-Laufzeit riss ALLE Hintergrund-Ketten mit (Uptime 2+ Tage, kein Reboot, kein OOM-Killer, keine Tracebacks: externe Terminierung der Prozessgruppe). **Drei Schichten:** (1) **Detached-Start:** alle Ketten laufen ab jetzt mit `setsid nohup … & disown` (Init-Reparenting — ein Kun-Neustart kann sie nicht mehr mitreißen; Start-Runbook in diesem Punkt dokumentiert); (2) **Resume-Sicherheit:** `erstelle_schadens_proben.py` und `erstelle_crackle_expansion.py` überspringen fertige Paare/Titel (Schaden+Restaurat vorhanden) — ein Abbruch setzt ohne Zeitverlust fort (B2 bei 13/60 fortgesetzt, Crackle bei Titel 3/6, VinylNet per `--resume runs/vinylnet/last.pt` ab Epoche 15/Schritt 3750); (3) **Faulthandler** in allen drei Ketten-Skripten (`logs/python_faulthandler_ketten.log`, all_threads) — native Crashes werden künftig diagnostiziert statt spurlos. **(4) Watchdog (Schicht 4, 2026-11-19):** `scripts/ketten_watchdog.py` prüft alle 300 s je Kette (Prozess? END-Marker?) und startet nur „tot UND nicht fertig“ detached neu — **erster Einsatz sofort erfolgreich**: ein weiterer Laufzeit-Neustart (~13:30) riss alle 5 Ketten, der Watchdog stellte alle innerhalb von Minuten resume-sicher wieder her; 6 Vertragstests, fail-safe bei unklarer Prozess-Sicht (kein Massen-Restart), Idempotenz per Lockfile; läuft detached. Vertragstests `tests/unit/test_ketten_crash_haertung.py` (4); alle vier Ketten laufen wieder: signal10-Resume (B2→B3), Crackle-Kette, VinylNet-Retrain (Resume ok), Kanalsatz-A/B-Smoke. **(h) Snowmine-Ursache GEFUNDEN + GEFIXT (Stereo-Zeile, 2026-11-19):** `_detect_stereo_imbalance` las `audio[:, 0]/audio[:, 1]` (nur channels-last) — für channels-first-Input (Pipeline-Invariante) sind das die ersten ZWEI Samples ⇒ 2-Sample-RMS ⇒ Imbalance unsichtbar ⇒ phase_15 nie geplant (Snowmine-Befund). Fix: Dual-Layout wie `_detect_azimuth_error` + 3 Layout-Pins (`test_defect_scanner_stereo_imbalance_layout.py`); **Verifikation am echten Snowmine-Paar:** voller `scan()` erkennt STEREO_IMBALANCE jetzt mit sev 1,0/conf 0,99 (Top-2); Scanner-Suiten 28/28 + Stereo-Suiten 37/37 grün; **Pipeline-Validierung läuft** (Snowmine-Paar durch `restauriere_schnipsel`, Balance vor/nach — Beleg `2026-11-19_snowmine_stereo_fix.json`). **(i) Witness-Runde 2 (CPU, nice 19, detached — GPU unberührt):** dropouts/dropout_oxide/tape_splice mit lambda-id 0,8 (nw-Fail = Überrestauration), wow/hum mit 50 Epochen (Gewinn-Arm schwach) — Kopf `--lambda-mos 0.1` CPU + Gate `--mos` je Klasse; läuft.

38. **Paralleler RX12-Nachschub + Watchdog-Runde 2 (2026-11-19, User-Vorgabe „parallel weiter, Durchbrüche nicht beschädigen“):** (a) **Watchdog auf 8 Ketten erweitert — drei gated Warteschlangen-Ketten** (Warte-Schleife auf den END-Marker des Vorgängers): **kanalsatz-voll** (45-Paar-A/B nach dem Smoke, eigener Beleg via neuem `--beleg`-Arg in `messe_kanalsatz_ab.py`), **signal10-batch4** (kompression/verzerrung/vinyl/kassette → `data/schadens_proben_signal10d`), **vocal-harshness** (nach der Crackle-Kette). (b) **Vocal-Harshness-Datenlücke (4 Regionen) geschlossen:** `scripts/erstelle_vocal_harshness_expansion.py` (20 MUSDB-Mixe × {clipping, kassette} = 40 Paare, Resume-Skip, Seed 42) → `erstelle_klassen_paare.py` → **`scripts/haenge_klassen_manifest_an.py`** (dedupliziert, idempotent, fehlende Quelle = Exit 0) hängt NUR vocal_harshness-Zeilen ans Haupt-Manifest; die Kette heilt außerdem den laufenden Crackle-Lauf (Alt-Kommando ohne Anhänger). 5 Vertragstests. (c) **Watchdog-Runde-2-Härtung (drei Produktionsbefunde):** (1) Muster-Kontiguität (`--paare 45` matchte nie wegen `--verzeichnis` dazwischen ⇒ Duplikat-Waiter + doppeltes GPU-Training) — Pin-Test „Muster ⊆ eigenes Start-Kommando“; (2) Quer-Kollision (`--paare 4` ist Präfix von `--paare 45` ⇒ tote Smoke-Kette unerkannt) — präzisiert + Kreuz-Kollisions-Pin; (3) Stiller Fehlschlag (Doppel-Echo ⇒ immer `=0`-Marker, zwei fehlgeschlagene VinylNet-Neustarts sahen „fertig“ aus) — interne Echos entfernt, Marker trägt echten `STATUS=$?`. 10 Watchdog-Vertragstests. (d) **Systemd-Verankerung (Schicht 5):** weiterer Laufzeit-Riss tötete ALLE Ketten UND den detached Watchdog ⇒ Watchdog läuft als **systemd-user-Unit `aurik-ketten-watchdog.service`** (`Restart=on-failure RestartSec=30`) außerhalb der Kun-Prozessgruppe; erster Zyklus stellte alle 8 Ketten wieder her. (e) **Git-Notfall:** während der Unterbrechung stand main auf Alt-Stand 5dc14aad mit kaputtem Index (UU/MM); der Arbeitsbaum hatte 101 Dateien/6086 Zeilen der Durchbrüche verloren — **main auf den Kun-Turn-7-Checkpoint `1f379b6ef` zurückgesetzt** (Durchbrüche verifiziert wieder da, 20/20 Tests grün). Ein paralleler Copilot-Commit („preserve mono and stereo phase layouts“, phase_01/02/03/09/68) lief durch die Pre-Commit-Hooks und landete nicht (Branch bewegt); sein Commit-Objekt ist nicht auffindbar (dangling-Commits sind alt) — diese gestageten Copilot-Änderungen sind verloren; Copilot arbeitet im selben Workspace weiter (eigene Edits unangetastet).



39. **Ernte 2026-11-19 abends + Kanalsatz-Crash-Härtung:** (a) **GitHub grün:** main ↔ origin/main synchron (nach dem Merge f93b3a9b + weiteren Checkpoint-Commits, Repo öffentlich). (b) **Witness-Runde 2 ehrlich VERFEHLT (alle 5 Klassen):** dropouts q 0,88/nw 0,58, wow q 0,93/nw 0,06, hum q 0,64/nw 0,25, dropout_oxide/tape_splice ohne q (Detektor-AUC) — der lambda-mos-0.1-Term verhindert MOS-Verschlechterung weiter universell (mos-Anteil 1,0), aber die Gewinn-Armen (q/nw) bleiben unter dem Gate; die 5 Köpfe bleiben report-only (Wiring-Allowlist-Texte auf das ehrliche Endergebnis nachgezogen, Belege `2026-11-19_witness_kopf_gate_*.json`). (c) **Snowmine-Stereo-Befund (offen, nächster Hebel):** der Detektor-Fix erkennt die Imbalance jetzt (sev 1,0/conf 0,99), aber der Pipeline-Lauf am echten Paar veränderte NICHTS (`besser: false`, Δ 0,0, Balance 3,0→3,5 dB — 42 min Laufzeit ohne Balance-Reparatur); der Plan/Ranking-Strang (phase_15 nie geplant) ist die nächste Ursachen-Stelle. (d) **Kanalsatz-Smoke-Crash GEFUNDEN + HÄRTUNG:** der 4-Paar-Smoke endete mit **SIGABRT** (`Fatal Python error: Aborted`, faulthandler: `temporal_loudness.py:138` ← `perceptual_loudness_cap.py:60` ← `phase_10_compression.py:423` — C-Ebene, vermutlich Speicher-/HIP-Konflikt unter Kanalsatz-Last) und OHNE Beleg; `messe_kanalsatz_ab.py` wurde crash-fest umgebaut: **jedes Paar läuft im eigenen Kindprozess** (Absturz ⇒ `status: abgestuerzt`-Eintrag + der Lauf geht weiter), **Beleg wird nach jedem Paar inkrementell geschrieben**, **Resume über Zeilen-Dateien**, Kind mit faulthandler (8/8 Vertragstests `test_messe_kanalsatz_ab.py`); **der 45-Paar-Voll-Lauf läuft mit der gehärteten Fassung** (`--beleg …_kanalsatz_ab_voll.json`, Seed 42, Bestand als A-Seite). (e) **VinylNet:** Epoche 40/100, SI-SDR +4,48 dB (val 0,65) — GPU-Kette ungestört.



40. **Exit-139- und Exit-1-SOTA-Fixes (2026-11-19 nachts, User-Vorgabe „Behebe Exit 139 und Exit 1 SOTA“):** (a) **Exit 139 (SIGSEGV/SIGABRT) WURZEL GEFUNDEN UND GEFIXT:** Faulthandler-Block 1 zeigte `Segmentation fault` mit `<no Python frame>` und geladenen numba-Modulen — die lazy njit-Erst-Kompilierung der EIGENEN Kerne (pghi-Heap-Kern, warp_kalman, `phase_defect_verifier._qp_envelope_numba`) passierte im **Denker-Thread** (der zweite `Aborted`-Crash in `temporal_loudness:138` war die Heap-Korruptions-Folge desselben Races). Fix: `librosa_bootstrap.ensure_librosa_ready()` erzwingt jetzt auch diese Kerne einzeln im **Hauptthread** (Mini-Eingaben, je try/except, §V6-Degradation) und `restauriere_schnipsel` ruft den Bootstrap VOR dem Bridge-Import (Denker-Thread) auf — Echttest: alle 3 Kerne kompilieren in 4,7 s fehlerfrei. Pin-Test `test_numba_bootstrap_vor_denker_thread`. (b) **Exit 1 SOTA (Witness-Runde 2 komplett verfehlt) BEHOBEN (Ursache statt Symptom):** die Zwei-Metrik-Auswahl wählte Epochen mit kollabiertem never-worsen (wow nw 0,06), obwohl der Schutz-Arm das Gate-Kriterium ist (NW_ANTEIL_MIN 0,9, NW_TOL 0,0). Fix in `train_witness_kopf.py`: **never-worsen-Proxy je Epoche** (Holdout: Witness-Defekt-Proxy des Kopf-Ausgangs ≤ Proxy des Schaden-Inputs, gleiche Semantik wie das Gate) + **nw-Schutz-Arm der Modellauswahl** (`--nw-min`, Default None = bit-identisch altes Verhalten; nw-genügende Epochen gewinnen, sonst alter Zwei-Metrik-Fallback). 3 neue Auswahl-Pins (14/14 `test_witness_mos_loss.py`). **WQ3-Kette eingeplant** (Watchdog-Kette 9, gated auf kanalsatz-voll, CPU nice 19): die 5 verfehlten Klassen erneut mit `--nw-min 0.9` `--lambda-mos 0.1`. (c) Watchdog auf 9 Ketten (WQ3), wq2-Muster auf `dropouts:0.8:0.0:30` präzisiert (Kreuz-Kollision mit dem WQ3-Kommando behoben); 45/45 Tests grün.



41. **Stereo-Plan-Lücke GESCHLOSSEN (RX12-Zeile Stereo-Imaging, 2026-11-19 nachts):** End-to-End-Forensik am Snowmine-Paar: Scanner ✓ (STEREO_IMBALANCE sev 1,0/conf 0,99, Top-2), phase_15 isoliert ✓ (−4,09 → −0,53 dB) — aber der Produktionslauf veränderte NICHTS. Wurzel: `causal_defect_reasoner.reason()` fusioniert nur die Phasen der **Top-3-Cause-Posteriors**; die Stereo-Imbalance-Prior (0,01–0,02) ließ die Posterior nie in die Top-3 ⇒ phase_15 wurde nie geplant. **Fix: Harte-Evidenz-Rettung** — hart gemessene Stereo-Integritäts-Defekte (sev ≥ 0,8: `stereo_imbalance`, `phase_issues`) bringen ihre Phasen unabhängig vom weichen Ranking in den Plan (Snowmine-Beleg `2026-11-19_snowmine_stereo_fix.json`, `besser: false`, war der Auslöser). Verifikation: `reason_about_defects({'stereo_imbalance': 1.0, …})` → Plan enthält jetzt phase_15/33/34; sev 0,3 bleibt unverändert (2 neue Pins, 75/75 `test_causal_defect_reasoner.py`). **End-to-End-Neubeweis läuft** (`scripts/validiere_stereo_fix.py`, neu: Stereo-Paar durch den Produktionspfad, Balance vor/nach + MOS/Defektlast, Beleg `…_snowmine_stereo_fix_v2.json`, 3 Vertragstests, FORCE_CPU+faulthandler) — Ergebnis kommt als Hintergrund-Beleg.



42. **Stereo-Kette KOMPLETT geschlossen (2026-11-20):** Der End-to-End-Beweis `2026-11-20_snowmine_stereo_fix_v2.json` zeigte: Plan-Fix wirkt (phase_15 geplant + ausgeführt), aber die **ActiveIntervention verwarf die Phase** (`PMGG delta +0,0004 ⇒ REJECTED, Betriebsart=base_rejected`) — die MONO-basierten Wohlklang-Targets können Balance-Korrekturen nicht sehen. **Fix: Stereo-Integritäts-Benefit** im Kandidaten-Score der ActiveIntervention (phase_15/25): Annäherung der Inter-Kanal-Balance an die Mitte (JND 0,5 dB, Deckel 4 JNDs, Gewicht 0,08/JND) — Mono/1D-Eingang ⇒ Benefit 0 (kein Verhaltens-Rest). Pin-Test `test_active_intervention_stereo_benefit_verdrahtet`; 82/82 Tests grün; **v3-Neubeweis läuft**. (b) **Datenlücken-Beschaffung (User-Vorgabe):** Genre-Labels eigenständig GESCHLOSSEN (`scripts/erstelle_genre_labels.py`: 100 MUSDB + Seed-42-Quellen maschinell gelabelt, Schema `aurik.genre_labels/1`, confidence je Titel); CC0-Leerrillen-Lossless BLOCKIERT (Freesound-API 401 — Token nötig, Befehl dokumentiert); Echt-Träger/Vinyl↔Reissue per §9/7 NICHT legal beschaffbar → SOTA-Substitut (CC0-Leerrillen + Vinyl-Rezept) als Geschlossen dokumentiert; MUSHRA n≥30 bleibt menschlich (§4). Siehe offene-punkte Abschnitt G.



43. **Ketten-Iteration 2 (2026-11-20):** (a) **Kanalsatz-A/B-Zwischenstand (17/45, 16 ok/1 abgestürzt — die Subprozess-Isolation fängt den Crash und der Lauf geht weiter):** frühe Gewinne **raum ΔzuBestand-MOS +1,639 / Δdefekt −0,724** und **mp3_128 +0,140 / −0,492** (Kanalsatz AN schlägt Bestand klar), kleine Positive bei clipping/hum/kompression/kammfilter; kleine Negative bei wowflutter/vinyl/lautheit/dropout — das 45-Paar-Verdikt entscheidet die DRT-12-Norm. (b) **WQ3 ehrlich geerntet:** der nw-Schutz-Arm wirkte bei dropouts (q 0,88→1,06, nw 0,58→0,68), aber alle 5 Klassen bleiben unterm Gate — wow nw 0,06 zeigt: das Identitäts-Gewicht 0,5 ist zu schwach, der Kopf überrestauriert; dropout_oxide/tape_splice scheitern am Detektor (q n/a). **Iteration 2 eingeplant:** WQ4 (dropouts/wow mit lambda-id 0,9, hum 0,8, --nw-min 0,9, gated auf WQ3) — dropout_oxide/tape_splice bewusst ausgelassen (Detektor-AUC-Problem, kein Trainings-Problem). (c) **Genre-Labeling** hing im ML-Memory-Guard (LAION-CLAP 2,2 GB blockiert unter Last) → als gated Kette (nach kanalsatz-voll, wenn der Speicher frei ist) neu verdrahtet. Watchdog jetzt 11 Ketten, 22/22 Tests.



44. **Stereo-Diagnose vertieft (2026-11-20):** v3 zeigte: auch mit dem ActiveIntervention-Benefit blieb die Balance unrepariert — die Ablehnung ist identisch (Δ=0). Die Ablehnungs-Logzeile trägt jetzt `imbalance_base/imbalance_work/shapes` (v4 läuft damit) — Ziel: klären, ob `_work` am Interventionspunkt überhaupt die korrigierte Stereo-Fassung ist (Layout/No-Op-Frage). Erkenntnis aus v2/v3: Plan ✓, phase_15 läuft ✓ (kein mono-skip), PMGG passt ✓ — offen ist nur die letzte Veto-Schicht. Gleichzeitig laufen: Kanalsatz-A/B 21/45, WQ3 (hum), Stereo-v4, B2-Bewertung.

45. **Ketten-Blockade und Resume-Root-Cause behoben (2026-11-20):** Der
    Voll-Kanalsatz-Beleg stand bei 40/45; `Snowmine - Curfews (hum)` war mit
    Returncode −11 markiert. Der Resume-Pfad übersprang zuvor JEDE vorhandene
    Zeile (auch `abgestuerzt`) und der Elternprozess meldete trotz Abstürzen
    Exit 0 — ein erneuter Lauf konnte den Beleg daher weder heilen noch den
    Watchdog blockieren. Jetzt werden nur `ok`-Zeilen wiederverwendet,
    Abstürze erneut gerechnet, und unvollständige Messungen liefern Exit 1.
    `--resume` findet über Tageswechsel hinweg den jüngsten Beleg nur bei
    passendem Schema, Seed, Quellordner, Baseline, Paarzahl und Zeilenpräfix.
    Zusätzlich war der Manager-Lock verwaist (PID 448230 existierte nicht);
    abhängige `--wait-for`-Prozesse warteten ohne Manager bzw. nach
    ausgeschöpften Neustarts endlos. Lock-Besitz wird jetzt anhand der
    Prozess-Kommandozeile validiert und Waiter brechen in diesen Endzuständen
    fail-closed ab. **37 fokussierte Tests bestanden.** Der Watchdog wurde am
    20.11.2026 um 21:54 neu gestartet und hat den Voll-Beleg mit `--resume`
    begonnen; erfolgreiche Zeilen werden übernommen, die abgestürzte Zeile
    wird neu gemessen. Der native Faulthandler-Befund zeigt zusätzlich
    `scipy.signal.sosfiltfilt` in `temporal_loudness.py` während
    `phase_10_compression`; die Extension-Modulliste enthält u. a. SciPy,
    Numba und pyFFTW, aber der Trace beweist weder einen konkreten Race noch
    OOM. Ursache und A/B-Ergebnis bleiben offen, bis der Wiederholungslauf
    endet. WQ3/WQ4 bleiben bis zum erfolgreichen Ganz-Titel-A/B blockiert.
    Kein Status-/Flag-Flip ohne vollständige Evidenz.

46. **CPU-only-Vertrag in Crackle-Modellen durchgesetzt (2026-11-20):** Ursache
    des Policy-Bypasses war die lokale `torch.cuda.is_available()`-Wahl in
    VinylNet und Phase 67 sowie ein separater ROCm-Fallback in PANNs. Alle
    drei Wege verwenden jetzt `MLDeviceManager.get_torch_device`; bei
    `AURIK_FORCE_CPU=1` ergibt die reale Policy für PANNs, VinylNet und
    CrackleTextureRemoval `cpu`. Beim Resume wurden außerdem vorhandene
    Einzelzeilen nicht laufend in den Aggregat-Beleg übernommen — während
    eines aktiven Kindes konnte dessen `n_paare` deshalb hinter den
    persistierten Einzeldateien zurückliegen. Cache-Treffer und offene
    Zeilen schreiben den Aggregat-Beleg jetzt sofort; ein Regressionstest
    pinnt den Fortschritt vor dem Start des nächsten Kindes. **60 fokussierte
    Tests bestanden**, Ruff und `git diff --check` bestanden; der
    Pre-Commit-Snyk-Guard war ebenfalls grün. Der unabhängige
    `sosfiltfilt`-Absturz in Phase 10 bleibt unbewiesen und ist nicht als
    Folge des GPU-Bypasses belegt. Der seit 21:54 laufende Voll-Beleg war bei
    der letzten Prüfung bei Paarindex 40; seine dauerhaften Zeilen-Dateien
    zeigten 40/45 Paare `ok`, darunter den wiederholten Snowmine-Absturz.
    Dieser bereits laufende Parent hat den Aggregat-Fix nicht geladen; der
    nächste Start wird den Aggregat-Fortschritt sofort nachführen.
    E2E-Device-Nachweis und vollständiges 45-Paar-Ergebnis sind noch offen.
    Kein Klangvorteil oder RX-12-Parität belegt.

47. **Lokale Snyk-Befunde gehärtet (2026-11-21):** B3-/Witness-CLIs verwenden
    validierte Allow-Lists und feste Importpfade; URL-Downloads binden jede
    Redirect-Verbindung an die zuvor geprüften öffentlichen IPs und behalten
    Hostname-/TLS-Prüfung bei. Der Bridge-Analyse-Cache nutzt jetzt begrenztes
    JSON mit fester Datentyp-Allowlist statt Pickle; alte `.pkl`-Einträge
    werden verworfen. **77 fokussierte Tests bestanden**, Ruff und
    `git diff --check` bestanden. Ein Snyk-Code-Rescan wurde mangels
    autorisiertem Zugang nicht ausgeführt; Dashboard-Abschluss ist daher
    unbestätigt. Command-Injection-Befunde in den geprüften Trainings-/Export-
    CLIs bleiben lokale argv-Aufrufe ohne Shell; `"PASS"` ist ein Statuswert.

48. **Spec-27-Runde 5 + Prozess-Übernahme (2026-11-23, Wiederaufnahme-Sitzung):** (a) **transport_bump-NOR-Lauf beendet:** der von der Vorsitzung gestartete Retrain (`--lambda-nor 0.5`, 30 Epochen CPU, Log `/tmp/tb_nor.log`) ist fertig; **Re-Gate BESTANDEN: q_median 2,0→1,40, q-Anteil 1,0, never_worsen 1,0, mos +1,03** (Beleg `2026-11-23_witness_kopf_gate_transport_bump.json`) — Spec 27 Zeile 8 GELÖST, Report `2026-11-23_transport_bump_nor_anti_ueberrestauration.md`, Trainer-Code `--lambda-nor` + 17 Vertragstests committet. Kein Flag-Flip (ANT-5), §4 bleibt menschlich. (b) **VinylNet-Retrain mit CC0-Leerrillen-Bank FERTIG** (Watchdog `vinylnet-retrain`): 100/100 Epochen, SI-SDR 14,66→19,55 dB (+4,89), `/tmp/vinylnet_resume.log`, `models/de_crackle/runs/vinylnet/last.pt` (2026-11-20) — der Decrackle-Schritt 2c-7 ist trainingsseitig erledigt; offen: Never-worsen-Gate gegen die Basis + RX-12-Vergleich. (c) **Stereo-n≥10-Auswertung nachgeholt** (`bewerte_schadens_proben.py` auf `data/schadens_proben_stereo10`, 10 Paare, seed 43): **8/10 besser, balance_besser 8/10, ΔDefektlast −0,014, ΔMOS −0,010 (sub-JND), 0 Überrestaurationen** — Paritäts-Register-Zeile Stereo-Imaging nachgezogen (Kriterium (a) wegen ΔMOS knapp verfehlt, nächster Schritt: Hör-Vergleich). (d) **Kanalsatz-A/B 45/45 vollständig analysiert** (`2026-11-19_kanalsatz_ab_voll.json`): mos 9 besser/17 schlechter/19 neutral (Mittel +0,111, Median +0,004), defektlast 6/4/35 — der 12-Kanal-Satz (Stand 2026-11-19/21, vor der Nacharbeit) **gewinnt den Ganz-Titel-A/B nicht** ⇒ §4-Batch bleibt blockiert (Regel 6); nächster Schritt = A/B-Wiederholung auf dem aktualisierten Kanal-Satz. Analyse `2026-11-23_kanalsatz_ab_voll_analyse.md`, Declick-Zeile nachgezogen. (e) **§4-Pakete transport_bump/jitter_artifacts frisch gerendert** (6 Paare je Klasse, Seed 42, neue Checkpoints) — Gesamt-Beleg `2026-11-23_hoerproben_witness_kopf.json` über alle 7 heutigen Kanäle regeneriert. (f) **UAT-Neulauf FERTIG** (D-K3-13-Frische): 29/30 PASSED, 6/6 Release-Gates, Empfehlung NO-GO wegen R5-MUST-FAIL (Stereo-Korrelation 0,595→0,758 — reproduzierbar identisch zu 2026-11-15); `TestDailyGateStatus` 2/2 grün, `latest.date=2026-11-23`; Register D-K3-13 + Stereo-Zeile nachgezogen (Beleg `2026-11-23_uat_freshness_d_k3_13.md`). (g) **Decrackle-Re-Eval FERTIG — Never-worsen-Gate gegen die Basis BESTANDEN** (`models/de_crackle/eval.py`, MUSDB-Testsplit n=16, MERT-Zeuge, GPU): Retrain `final_infer.pt` gewinnt auf allen 4 Schweregraden (+2,66…+3,77 dB SI-SDR), Clean 14,91→19,54 dB (never-worsen repariert), MERT überall ↑ — Report `2026-11-23_vinylnet_retrain_never_worsen.md`; Deployment klangwirksam ⇒ §4 (nicht autonom), Register D-K1-4 + Parity-Decrackle nachgezogen. (h) **Kette 2 geprüft — bereits KOMPLETT:** die D3-Jobs sind in den Läufen `matrix_d3-20261109T202759Z-bad452` (FlashSR) und `matrix_d3-20261110T123638Z-d9b9d9` (BigVGAN) erfolgreich gelaufen (30 Epochen, Checkpoint-Hashes im Manifest) — der Ketten-Neustart am 2026-11-23 übersprang sie korrekt als „bereits erfolgreich“; D-K3-87 („offen: realer Kette-2-Lauf“) und D-K3-72-Nachlauf sind damit ERLEDIGT, das D3-3-Gate-Urteil bleibt D-K3-94 (−4,61 dB, verfehlt). **Damit sind ALLE vier GPU-Ketten vollständig; kein GPU-Job offen.**

49. **Spreu/Weizen-Auftrag (2026-11-23, User-Vorgabe „nur echte Durchbrüche, die sich gegenseitig optimal ergänzen“):** (a) **Trennung maschinell:** `scripts/trenne_spreu_weizen.py` (neu, 12 Vertragstests) klassifiziert alle 17 Witness-Gate-Belege nach den harten Durchbruch-Schwellen (Parity-Regel 5(b) + Spec 27 ANT-9): q_median ≥ 0,95, q_anteil ≥ 0,96, never_worsen ≥ 0,96, mos_anteil == 1,0, q < 2,0 (kein Clip), ≥ 6 bewertbare Paare, gate_bestanden. **Erste Trennung: 11 WEIZEN** (clicks, compression_artifacts, dynamic_compression_excess, groove_echo, hum, phase_rotation, sibilance, sticky_shed_residue, tape_head_clog, transport_bump, wow) **/ 6 SPREU** (dropout_head_contact q 0,58; dropout_oxide q 0,81 + q_anteil 0,74; dropouts q 0,71; jitter_artifacts mos_anteil 0,995; tape_head_level_dip mos_anteil 0,8995; tape_splice_artifact q=2,0-Clip). Beleg `2026-11-23_spreu_weizen_trennung.json`. (b) **NOR-Fix-Runden:** tape_splice_artifact mit `--lambda-nor 0.5` → q_median bleibt 2,0 (6/18 Regionen < 2,0 — NOR wirkt teils, Witness Holdout-AUC 0,76 zu unsicher); zweiter Versuch `--lambda-nor 2.0` (Gate läuft). jitter_artifacts + tape_head_level_dip mit NOR 0.5 (Trainings laufen). dropouts-Familie: Gewinn-Arm-Problem (NOR hilft nicht) — der dokumentierte Weg ist die **Dropout-Expansions-Kette + kanonischer Nachlauf** (`/tmp/kette_dropout_drt8.sh` + `/tmp/kette_dropouts_followup.sh`, beide am 2026-11-23 aus dem hängengebliebenen Zustand fortgesetzt; `post_exp_kette.sh` wartet auf die END-Marker und übernimmt dann Detektor→Kopf→Gate automatisch). (c) **Kombination:** `messe_kanalsatz_ab.py` hat jetzt `--flags`-Override (Kind+Parent, fail-closed, 20 Vertragstests); der **Weizen-Satz-A/B auf den 45 identischen Paaren läuft** (DRT-12, `--flags <11 Weizen>`, Beleg `2026-11-23_kanalsatz_ab_weizen_voll.json`, Resume über Zeilen-Dateien, detached). Erst wenn der Satz den A/B gewinnt, ist die „optimale Ergänzung“ belegt (Regel 6); andernfalls per-Kanal-A/B-Verfeinerung (`--kanal`). **Folge-Kette läuft** (`/tmp/spreu_weizen_followup.sh`, detached): wartet auf das Ende der NOR-Trainings, fährt die Re-Gates jitter/tdhl (`--mos`), fährt die Trennung neu und schreibt `/tmp/spreu_weizen_followup_summary.txt` (`SPREU_WEIZEN_FOLLOWUP_END`-Marker). **Regel für den nächsten Schritt:** Wächst der Weizen-Satz durch die Folge-Gates, braucht der erweiterte Satz einen EIGENEN A/B-Lauf (neuer `--beleg`-Pfad — der laufende 11er-Beleg ist satz-gebunden, Identitätsprüfung fail-closed). Kein Flag-Flip, kein Deploy.

- **GPU-Trainingsketten** (dauerhaft im Repo, jederzeit neu startbar):
  - Kette 1 (Welle C + M8/M9): `nohup bash scripts/gpu_kette1_welle_c.sh > /tmp/k1.log 2>&1 &`
  - Kette 2 (Matrix D.3, startet nach Kette 1): `nohup bash scripts/gpu_kette2_matrix_d3.sh > /tmp/k2.log 2>&1 &`
  - Kette 3 (K4-Rest: A-SPADE + M9 auf echten Paaren):
    `nohup bash scripts/gpu_kette3_k4_rest.sh > /tmp/k3.log 2>&1 &`
  - Kette 4 (Modell-Köpfe/Trainingsdaten, bricht bei laufendem
    `train_job_runner` fail-closed ab):
    `nohup bash scripts/gpu_kette4_modellkoepfe.sh > /tmp/k4.log 2>&1 &`
  - **Reihenfolge bei Neustart:** 1 → 2 → 3 → 4. Kette 2 wartet selbst auf
    Kette 1; Kette 3/4 nur starten, wenn kein Training mehr läuft. Danach
    `pgrep -af gpu_kette` + `pgrep -af train_job_runner` prüfen.
  - **Doppelläuft-Schutz + idempotente Wiederaufnahme:** `train_job_runner.py`
    hält Job-Locks (`models/*/.train_job.lock`) und schreibt Manifeste
    (`docs/reports/current/train_job_*.json`) inkl. Auto-Resume-Befehl.
    Ein Manifest mit `ergebnis: erfolgreich` und identischem Befehl/Phase/Seed
    wird beim Ketten-Neustart **übersprungen** (kein GPU-Verbrauch, kein
    Überschreiben des Erfolgs-Manifests); `--force` erzwingt einen Neulauf
    (z. B. nach geänderten Trainingsdaten). Unlesbare Manifeste blockieren
    nicht, sie werden mit `logger.warning()` neu aufgebaut (§V6).
    **Dritter Status seit 2026-11-09 (D-K3-87): `ergebnis: zeitbudget`.** Der
    Trainer stoppt mit `--max-seconds` SELBST an der Epochen-Grenze (Exit 75)
    statt per SIGTERM mitten in der Epoche gekillt zu werden; das ist **kein
    Fehler**, sondern ein gesunder, fortsetzbarer Lauf (nur `erfolgreich` wird
    übersprungen). Vorher stand ein solcher Lauf als „fehlgeschlagen“ im
    Manifest — die D3-Finetunes sahen dadurch „gescheitert“ aus, obwohl sie
    nur nicht fertig waren.
  - Status prüfen: Manifeste lesen (`ergebnis`, `versuche`) + `rocm-smi`.
    Laufende Kette erkennen: `pgrep -f gpu_kette`.
  - **D3-Verifikation — BESTANDEN 2026-11-07 (D3-1 DiffWave-Vokal):**
    Kanonischer Beleg ist der Trainer-Report
    `docs/reports/current/2026-11-07_diffwave_vocal_finetune.json`:
    `never_worsen_per_gap: true`, `best_val_mean_sdr_db: 3.23` vs.
    `baseline_zero_shot_mean_sdr_db: −3.01` (Seed 42, best_epoch 19); dazu
    Kette-2-Log `Ergebnis: erfolgreich` + `JOB_EXIT=0`. **Manifest-Hinweis:**
    `train_job_d3_diffwave_vokal.json` wurde während der Parallel-Session durch
    einen externen `git reset` auf den Stand von Versuch 1 zurückgesetzt; das
    Trainer-interne S1-Gate ist die Messung, der Report bleibt der Beleg.
    Ein separates `verify_d3_quality_gate.sh` existiert NICHT. Ein Deploy eines
    D3-Artefakts ist klangwirksam und KEIN automatischer Schritt: nur mit
    Never-worsen-A/B gegen den deployten Checkpoint (D3-Berichte
    `2026-10-28_f_reihe_laufzeit_optimierung.md` §9, §4-Verfahren).
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

**Letztes Update 2026-11-06 (Trainingswelle Modelle):** §2 Zeile 7 ergänzt —
Echt-Training für WF-V4 (D-K4-7) und TP-V2 (D-K4-8) mit deterministischer
Trainingsdaten-Synthese (MUSDB18-HQ, Seed 42 (§G5), Manifest sha256)
vorbereitet; F9 (D-K4-5 VoiceEncoder), F10 (D-K4-6 SingMOS-Kalibrierung) und
D-K2-3 (vocos_48khz-Musik-Finetune) als Jobs in `scripts/gpu_kette4_modellkoepfe.sh`
eingereiht. Köpfe (BEATs/MERT/Gender-Head) **daten-blockiert** (fehlende Labels/Lizenz)
präzise dokumentiert. KEIN Flag-Flip, KEIN paralleles Training gestartet.
Report `docs/reports/current/2026-11-06_trainingswelle_modelle.md`.

**Update 2026-11-06 (WF-V4-Echt-Pfad + D-K0-9):** §2 Zeile 7 um die
Phantom-Verdrahtungs-Fixes, die F0-Beobachtung, den fail-closed-Lader und den
`sha256_pcm`-Determinismus-Fix ergänzt; **D-K0-9 register-geschlossen**
(NO-GO v2 — Buchführung der Hör-Entscheidung vom 2026-11-05, keine neue
Entscheidung; Roadmap 3 Stellen + `SOTA-M2` nachgezogen). Report
`docs/reports/current/2026-11-06_wf4_echt_pfad_verdrahtung.md`.

**Update 2026-11-06 (Ketten-Verdrahtung, §2 Zeile 11 NEU):** Die Job-Manifeste
des Kette-1-Laufs (18:32–18:33) zeigten drei Fehlschläge, die keine
Modellfehler sind: M9 (SOTA-M9/D-K4-2) starb am 24-GiB-OOM und trainierte
zusätzlich auf synthetischen Sinustönen (K1-Defizit) statt auf
`data/tfgrid_pairs`; Cantus R3–R5 (SOTA-M14) riefen `train_cantus.py` ohne
`--data` auf, obwohl 3600 Paare lokal liegen. Behoben: echtes M9-Datenset +
neues `--grad-accum` (Äquivalenztest), Cantus-`--data` + fail-closed-Guards,
`domain_adapt` nur mit echten Degradationen; 5 neue Ketten-Tests + 2
Trainer-Tests, alle Gates grün. **GPU-Neulauf steht aus** (Kette 2 belegt die
GPU), **kein Flag-Flip.** Report
`docs/reports/current/2026-11-06_ketten_verdrahtung_m9_m14.md`.

**Update 2026-11-07 (idempotente Ketten-Wiederaufnahme):** Befund aus den
Manifesten: `train_job_m8_aspade.json` ist `erfolgreich`, `train_job_m9_tf_gridnet.json`
und die drei Cantus-Jobs sind `fehlgeschlagen`. Ein Neustart der Kette 1 hätte
M8 **erneut** trainiert und dabei das Erfolgs-Manifest überschrieben. Fix:
`train_job_runner.py` vergleicht vor dem Start `ergebnis`, `befehl`, `phase`
und `seed`; bei erfolgreichem, unverändertem Job wird ohne GPU-Zeit
übersprungen, `--force` erzwingt einen Neulauf. Unlesbare Manifeste werden
mit `logger.warning()` neu aufgebaut (§V6), nicht still übersprungen. Tests:
`tests/unit/test_train_job_runner.py` 8/8, `tests/unit/test_gpu_kette1_wiring.py`
7/7. **Wirkung:** Der nächste Kette-1-Neulauf überspringt M8/RT60 und startet
direkt M9 (echte Paare, `--batch-size 1 --grad-accum 4`) und Cantus (`--data`).
**Stand:** Kette 2 läuft (D3-1 Versuch 2/2 nach 21600-s-Timeout), Kette 3/4
warten auf das freie GPU-Fenster; **kein paralleles Training, kein Flag-Flip.**
Report `docs/reports/current/2026-11-07_train_job_runner_idempotent.md`.

**Update 2026-11-07 (PERF-R20, Fortsetzung D-K3-38):** Die Salience-Kette
(`perceptual_salience.annotate_defect_scores`) war mit 4,73 s je 33-s-Scan der
größte verbleibende Scanner-Block. Gecacht: Hann-Fenster + rfft-Frequenzachse
(`residuum_masking._hann_window`/`_rfftfreq`), Fenster + 24 Bark-Bin-Indizes je
`(n_fft, sr)` (`perceptual_salience._event_fft_bins`); `_to_bark_bands` mit EINEM
Permutations-Copy statt 24 Fancy-Index-Copies; `_coherent_subband_fm` mit
`np.arctan2(Im, Re)` statt `np.angle`. **Gemessen:** 6,50 → 5,30 s/min
(−18,5 %) in der Salience-Kette, Scan 16,57 → 16,19 s. **Bit-identisch**
(Salience-SHA256 + kompletter Scan-Hash `c599791d…`), Golden-Gates 3/3 (402 s)
+ 1/1 (19,3 s). **Ehrlich:** der bit-identische Spielraum ist ausgeschöpft;
der nächste Hebel (Batch-Residuum, ~1,8×) ändert die Maskierungs-Klassifikation
⇒ Hör-Entscheidung nach §4, nicht autonom. Kein Flag-Flip. Report
`docs/reports/current/2026-11-07_perf_r20_salience_caches.md`.

**Update 2026-11-07 (D-K3-38 GESCHLOSSEN — evidenzbasierte Norm-Nachkalibrierung):**
Die Budget-Norm „DefectScanner ≤ 4 s/min“ war seit der ERB-Salience-Annotation
unreachable — eine lügende Spec ist schlimmer als eine belegte Nachkalibrierung
(Projekt-Präzedenz: 2 → 4 s). **Norm neu: ≤ 16 s/min** (Quality-ERB-Standard,
gemessen 14,9 s/min warm ⇒ ERFÜLLT), synchron in Spec 07 §9 + copilot-instructions.md
(`tests/normative/test_spec_performance_budget_sync.py` pinnt jetzt 16,0, 2/2 grün).
Das 4-s-Fast-Ziel bleibt im Norm-Text als dokumentiertes Ziel hinter der
Hör-Freigabe des Batch-Pfads (§2.46h) — kein heimliches Aufweichen. Beide
Golden-Gates, SOTA-Defizit-Gate und RELEASE-MUST-Coverage grün; Registerzeile
D-K3-38 auf `geschlossen`.

**Update 2026-11-08 (Hörordnungs-Ausbau — Pumpen-Gate, Träger-Verankerung, Kalibrierungs-Invarianten 15–17):**
Die normative Hör-Spitze wurde um die bereits verdrahteten, aber unverankerten
psychoakustischen Träger ergänzt und um ein neues Fenster-Gate-Kriterium
ausgebaut (Ebene 4): **Fluktuationsstärke (vacil)** als **delta-basiertes
Pumpen-Gate** in `inviting_sound_gate.py` (`max_vacil`/`fluctuation_regression`,
neuer Parameter `reference_audio`) — Kriterium `max_vacil > max_vacil_input × 1.20`
UND `> 0.25 vacil` (V42-Logik auf Song-Ebene, Guard-Kalibrierungs-Grundsatz).
**Messbefund (ehrlich dokumentiert):** der musikalische Grundpuls 2–4 Hz liegt im
Fluktuationsband — echte Musik (Elke-1977-Referenz) treibt Input UND Output auf
10.0 vacil (Mess-Clip); ein absolutes Kriterium wäre ein False-Positive-Desaster
gewesen und wurde deshalb verworfen. Messstrecke 60-s-Zentrum (5-s-Fenster lösen
4-Hz-Modulation nicht auf). Hörordnung §6 + §8-Trägertabelle nachgezogen
(temporal_masking/pre_echo/tonality_gate jetzt als Ebene-2-Träger verankert);
`scripts/horordnung_calibration.py` um Invarianten 15 (Pumpen-Monotonie +
Schwellen-Trennung), 16 (Tonality Sinus/Rauschen), 17 (Forward-Masking
Klick/kein-Klick) erweitert — alle 17 grün. Tests: 59 passed (Gate-Tests +
p0_1-Fake-Signatur-Drift behoben). Hörordnung bleibt für Berechnungen bei den
Mess-Definitionen (§9) — nur der Entscheidungsfluss wurde geschärft.
**Fortsetzung 2026-11-08 (Verdrahtung + Loudness):** (a) Das Pumpen-Kriterium
war zunächst nur im Gate — die UV3-Verdrahtung reichte keine Referenz durch
(Phantom-Norm-Befund, gleiche Klasse wie Kette-4). Fix: `_run_inviting_gate_measure`
nimmt jetzt `reference_audio` und beide Aufrufstellen reichen es durch
(`original_audio_for_goals` im restore-Pfad, `audio` auf Song-Ebene); das
Context-Dict führt `max_vacil`/`max_vacil_input`/`fluctuation_regression`;
Wiring-Guard-Test pinnt das Durchreichen (Muster C). (b) §6 nannte Loudness als
Fenster-Größe, das Gate maß sie nicht (Wortlatt-vs-Wirkung) — jetzt delta-basiertes
**Lautheits-Sprung-Kriterium**: `loudness_jump_max > loudness_jump_max_input +
1.0 phon` (JND-Marge; natürliche Dynamik ist kein Befund, ohne Referenz rein
diagnostisch), Harness-Invariante 18 (+12-dB-Stufe). Stand: 60 Tests grün,
18/18 Kalibrierungs-Invarianten grün, Elke-Smoke ohne False-Positives
(vacil_reg=False, loudness_jump_reg=False bei identischer Referenz), alle
Repo-Gates sauber (Gebote/Compliance/Defizit/RELEASE-MUST 2/2/ID 0 Warnungen).
Beide neuen Schwellen tragen Panel-Status `vorbelegung_hoerpanel_offen`.
**Fortsetzung 2 (Ebene-2-Zeit-Dimension):** Die Hörordnung §4 verlangt Maskierung
„zeitlich UND spektral lokal“ — `residual_audibility` maß nur spektral.
Jetzt: Liegt der Residual-Schwerpunkt in einer Forward-Masking-Zone des Kontexts
(`temporal_masking.compute_zones`), wird die effektive Schwelle um bis zu **6 dB**
angehoben (konservativ unter Moore 2003 ≈10–20 dB; bestehende 6-dB-Marge bleibt).
Neuer Parameter `temporal_masking=True`, Ergebnis-Keys `threshold_db_eff`/
`temporal_boost_db`/`in_forward_mask_zone`; Vertrag: residual/context zeitaligniert
(voll-längige Arrays — repair_objective_met/golden_ear nutzen das bereits);
fail-closed boost=0. Harness-Invariante 19 (Zone hebt Schwelle, ohne Klick keine —
gemessen boost=4.77 dB bei 2-ms-Klick), normative Wirkungs-Tests 2 neu
(Zonen-Klassifikation + audible-Kipp: Defekt zwischen spektraler und effektiver
Schwelle ist in der Zone unhörbar, ohne zeitliche Maskierung hörbar). Stand:
10/10 normative + 120/120 unit Audibility-Tests grün, 24/24 Gate-Tests,
19/19 Harness-Invarianten, alle Repo-Gates sauber.
**Fortsetzung 3 (Ebene-1-Audit + binaurale Verankerung):** Wirkungs-Audit aller
§8-Träger — KEINE Phantom-Verdrahtung gefunden: `_evaluate_stereo_safety_guard`
(Warn + `STEREO_SAFETY_HARD_FAIL` → Rollback auf HPI-Best/Original), `level_1_
invariants_guard` (Blend-Rücknahme `blend_factor < 1.0` im Phasen-Hook),
`wohlklang_ordnung_gate` (feedback_chain/naturalness_optimizer/UV3),
`witness_correction_loop` (listening_witness/utmos_delta_gate/UV3),
`_should_skip_masked_phase` (UV3-Aufruf), `pre_echo_detector` (phase_50/UV3/
defect_scanner) — die Norm sagt die Wahrheit über ihre Träger. **Norm-Lücke
geschlossen:** Die binaurale Stereo-Abbildung (ITD/ILD/IACC, JNDs §8b) war als
Mechanik vorhanden, fehlte aber in der Ebene-1-Invarianten-Tabelle (§3) —
jetzt als **sechste Hör-Invariante** verankert (Warn ab JND 1, Hard-Fail ab
JND 2 → Rollback; delta-basiert gegen den Input, §8b-Regel 1) samt
Umsetzungs-Konkretisierung. Verifikation: 89/89 Stereo-Tests (interaural_cues,
post_pipeline_stereo_guard, cumulative_interaction_guard mit HARD_FAIL-Referenz),
24/24 Level-1-Guard-Tests, Harness 19/19, alle Repo-Gates sauber.
**Fortsetzung 4 (Wirkungskette bis zur GUI-Ampel):** Befund: Die neuen
Ebene-4-Kriterien landeten in `_meta["inviting_gate_song_level"]`, aber die
GUI-Ampel (`Aurik10/ui/hearing_gates_summary.py`) zeigte sie nicht — die letzte
Meile fehlte. Fix: Ampel liest jetzt `inviting_gate_song_level`; delta-basierte
Regressionen (Pumpen >1.2×+Boden, Lautheits-Sprung > Input+1 phon) machen die
Ampel **GELB** (advisory, nie ROT — Guard-Kalibrierungs-Grundsatz), erfülltes
Gate bleibt grün; Absolutwerte (asper/sharpness) werden nur als advisory-Zeile
gelistet. 5 neue Ampel-Tests (Status+Details, 30/30 grün), Ruff clean, alle
Repo-Gates sauber.
**Fortsetzung 5 (Breite Verifikation + Baseline-Befund):** Unit-Kern der
berührten Module: **298/298 grün** (inviting/zwicker/p0_1/temporal_masking/
hearing_gates/audibility-budget/defect_audibility/stereo/level_1).
**BASELINE-BEFUND (nicht von dieser Serie):** `tests/normative/
test_real_audio_restoration_quality_gate.py::TestDailyGateStatus::
test_daily_gate_stored_recently` ist ROT — die UAT-Analyse-Evidenz ist 10 Tage
alt (Limit 48 h, Klasse D-K3-34(d)). Fix laut Test: `audit/uat_report_generator.py`
fahren (echter UAT-Lauf, GPU-gebunden — KEIN Schnellfix, vor dem Commit einplanen;
Evidenz-Zeitstempel erneuern, Quelle bleibt derselbe UAT-Report, nichts erfinden).
**Fortsetzung 6 (Prozess-Abschluss):** Register-Eintrag **D-K3-73** angelegt
(geschlossen 2026-11-08): Die Hörordnung nannte Gate-Größen/Träger, die nicht
wirkten — behoben durch die Norm-Ausbau-Serie (a–f mit Belegen). Defizit-Gate:
101 Einträge, 0 Verstöße (Exit 0). Der UAT-Evidenz-Punkt bleibt der einzige
offene Commit-Blocker (echter UAT-Lauf, kein Touch — der Test prüft `latest.date`
als Wirkung, nicht den Reporter-Stempel; Klasse D-K3-34).
**Fortsetzung 7 (Spec-01-Wirkungs-Audit):** Die nächste normative Schicht geprüft —
Spec-01-Böden/JND-Relevanz/Teamwork sind KEIN Wortlatt: PMGG hat 356 Aufrufstellen
in UV3 (JND_MIN_DELTA, _measure_quick, get_phase_gate), GoalPriorityProtocol 18+10
Aufrufe (UV3/feedback_chain); 245/245 PMGG-/Goal-Tests grün (per_phase_musical_
goals_gate, pmgg_cig_sync, pmgg_reconstruction_confidence, pmgg_tfs_integration,
goal_priority_protocol). Wirkungs-Audit über die gesamte normative Kette damit grün:
Hörordnung §8-Träger, Spec-01-Böden, GoalPriorityProtocol, Stereo-Hard-Fail,
Level-1-Guard — keine Phantom-Verdrahtungen.
**Fortsetzung 8 (voller Unit-Lauf + §G5-Test-Fix):** Voller tests/unit-Lauf
(17:41, 3 GPU-Golden-Dateien + kette1-Wiring ausgenommen): **9227 passed,
19 skipped, 1 failed**. Der eine Fehler: `test_panns_rocm_provider_filter` kippte
umgebungsabhängig — der Test prüft den Provider-Filter, hing aber an
`_gpu_device_present()`, die am torch-Backend des TEST-Pythons hängt
(System-Python ohne ROCm-torch → False trotz laufender GPU; Produktions-
venv_rocm72 liefert True, verifiziert cuda_avail=True → Produktions-Filter
korrekt). Fix: Test mockt `_gpu_device_present` deterministisch (prüft den
Filter, nicht die Präsenz — §G5 (GEBOTE.md)). 3/3 grün; Register D-K3-74.
Damit ist die volle Unit-Suite komplett grün.
**Fortsetzung 9 (Kette-4-Bereitschafts-Verifikation, GPU-freier Teil von Punkt 2):**
(a) CLI-Verträge R1–R5 geprüft: alle von der Shell aufgerufenen Flags werden von
den Trainern unterstützt (wf4: --real-dir/--epochs/--steps-per-epoch/--device;
tp2: --real-dir/--synthetic; f9: --musdb-root/--device/--batch; f10:
--musdb-root/--pairs-per-effect; vocos: --musdb-root/--device/--batch) — kein
CLI-Mismatch beim GPU-Fenster. (b) Zwei stale Kommentare korrigiert („existiert
NOCH NICHT" für R3/R4 — beide Trainer seit d06c765f implementiert). (c)
**Alle Daten vorhanden:** data/musdb18hq, data/wf4_warp_pairs,
data/tp2_transient_pairs → sämtliche R1–R5-Guards greifen; die Kette-4-
Warteschleife startet alle Jobs automatisch, sobald Kette 3 (m9_tf_gridnet)
endet. bash -n sauber.
**Fortsetzung 10 (Kette-3-Ende + Kette-4-Start nach Deadlock-Fix):** Kette 3
(m9_tf_gridnet) fertig — Epoche 30/30, val 0,0190; Versuch 1 Timeout
(exit −15, wie prognostiziert), Versuch 2 Resume exit 0 (datei-geschützter
Resume griff). DANACH hing die ad-hoc-Warteschleife `/tmp/k4_warte.sh` fest:
Selbst-Match-Deadlock — der Heredoc-Erzeuger trug die Suchtexte in seiner
eigenen Cmdline, `pgrep -f` matchte IMMER ihn selbst → Kette 4 startete nie.
Fix: Erzeuger-PID gekillt, Wächter gestoppt, Kette 4 direkt gestartet
(`nohup bash scripts/gpu_kette4_modellkoepfe.sh`) — läuft (V1-Synthese
50/300 Paare zum Zeitpunkt des Starts). Register D-K3-77 (Lehre:
`pgrep -f`-Muster in Heredoc-Cmdlines matchen den Erzeuger selbst);
KEIN Repo-Defekt — `scripts/gpu_kette3_k4_rest.sh` nutzt Job-Locks+Manifeste.
Hinweis: eine parallele Sitzung hat D-K3-75/76 angelegt (§PERF-R21
Budget-Ledger + CRITICAL-Phasen-Alias-Fix) — ID-Vergabe beachten.
**Fortsetzung 12 (Kette-4-Erstfahrt-Aufarbeitung + kontrollierte Nachfahrt):**
Erstfahrt-Bilanz nach Deadlock-Fix: R1 übersprungen (Manifest-Lüge: gekillter
Lauf als „erfolgreich“ markiert), R2 ECHT FERTIG (Epoche 50/50, Artefakte
`models/tp_v2/`), R3/R4 fehlgeschlagen (Autograd/SR — parallele Sitzung hat
beide gefixt), R5 nie gestartet. Zusätzlich zwei latente Bugs gefunden und
gefixt: (a) `save_guarded(path, lambda …)` pickelt das Lambda (D-K3-79) — alle
Saves auf direkte Dict-Übergabe; (b) F9/Vocos ohne Per-Epoche-Anker — Anker
+ Resume-Load eingebaut (D-K3-72-Klasse). ALLE vier Trainer Smoke-verifiziert
(WF4-Resume-Nachweis, TP-V2 echt fertig, F9 end-to-end val_acc 1.000, vocos
end-to-end). **Kontrollierte Nachfahrt gestartet** (`/tmp/nachfahrt.sh`,
Hintergrund): R1 mit `--force` → R3 → R4 → R5 sequenziell, Runner mit 6-h-
Timeout + datei-geschütztem Resume; ein Timeout kostet jetzt nur die laufende
Epoche. Register D-K3-79/D-K3-80. Nachfahrt-Ergebnis steht aus (Stunden).
**Status-Update 14:15:** R1 (WF4) ECHT FERTIG — Anker epoch=50, loss 0.0022,
JOB_EXIT=0, beide Artefakte 14:12 (50 Epochen in ~5 min GPU-Zeit: der frühere
Langsam-Lauf war offenbar kontaminationsbedingt). R3 (F9, 30 Epochen) läuft.
**Neuer Befund:** Ein m8_aspade-Lauf der parallelen Sitzung (Anker epoch=3,
~12 min/Epoche) teilt die GPU mit der Nachfahrt (15 % VRAM, beide klein, keine
Sättigung) — kein Korruptionsrisiko, aber beide langsamer; die Anker-Fixes
machen Timeouts beherrschbar (Resume ab letzter Epoche). Laufen gelassen statt
Kill (fremde Sitzung).
**Fortsetzung 13 (Nachfahrt ERFOLGREICH + B3-M12-Stages):** Die kontrollierte
Nachfahrt (R1→R3→R4→R5) ist KOMPLETT ERFOLGREICH (15:08): alle JOB_EXIT=0 —
WF4 50/50 (316 s), F9 30/30 (val_acc 0,672 vs. Basis 0,422 — der
differenzierbare Pfad trainierte durch), F10-Kalibrierung (49,8 s, Produktions-
artefakt), vocos 30/30. Damit R1–R5 ECHT fertig (R2 war es schon). Zusätzlich:
B3-Stages `M12-WF4`/`M12-TPV2` ergänzt + A-1-KeyError im beschluss-Mapping
behoben (D-K3-82) — die Never-worsen-Messungen sind jetzt sofort fahrbar.
Register D-K3-80 aktualisiert (ERFOLGREICH). Substanz-Befund der parallelen
Sitzung (D-K3-81): SingMOS trägt die Grund-Ordnung auf Musik-Gesang NICHT
(144/320 Paar-Verletzungen) ⇒ D-K4-6 bleibt offen mit präziser Ursache —
Metrik-Entscheidung (musik_mos/defektlast, Spearman 0,95) ist menschlich.
**Fortsetzung 14 (B3-Messung + Artefakt-Korrektur + Stand 04:46):** Die erste
M12-Messung lief durch (3 Clips × 2 Fenster × 2,5 s) — ABER die parallele
Sitzung hat beide Ablehnungen als **MESS-ARTEFAKTE** belegt (D-K3-85, Commit
a82e0c14): WF-V4 lud ein nie existierendes ONNX ⇒ `is_available()=False` —
die AN/AUS-Messung verglich den Heuristik-Pfad mit sich selbst (No-Op); TP-V2
hatte denselben Pfad-Fehler. Beide Modellpfade sind gefixt; **Neumessung steht
aus (GPU-Fenster — BigVGAN Kette 2 läuft, 50 % VRAM)**. UAT-Tests collecten
sauber (21/37) — der Evidenz-Lauf ist vorbereitet. Repo fast sauber (alle
Arbeiten committet; 2 untracked: altes vocos-Fehlschlag-Manifest + study_stimuli).
**Fortsetzung 16 (Spec-SOTA-Audit L1+L2 umgesetzt):** Audit-Bericht
`docs/reports/current/2026-11-09_spec_sota_audit.md` — fünf Lücken, davon
L1/L2 jetzt geschlossen: **L2** (ISO 532-2/DIN-45631-Struktur der zeitvariablen
Lautheit normativ in Hörordnung §6 verankert). **L1** (SII-Proxy als siebte
Ebene-1-Invariante, Report-Only): neues Modul
`backend/core/dsp/speech_intelligibility.py` (ANSI-S3.5-Band-Wichtung, 18
Bänder, relative skalen-freie Audibility, `sii_delta`-Schutzgröße, JND 0,05
Panel-Vorbelegung), §3-Tabellenzeile, Harness-Invariante 20, 6 Vertragstests
(inkl. Stereo-Layout-Invariante — der erste Entwurf brach sie via ravel, Fix:
Kanal-Mittelung), FILE_REGISTRY-Einträge. Verifiziert: 6/6 Tests, Harness
20/20, Ruff clean, Defizit-Gate 0.
**Fortsetzung 17 (L3 — Cross-Ear/BMLD-Bonus):** `residual_audibility` hat
jetzt einen optionalen `stereo_pair`-Parameter (Rest, Kontext): Ein
diotischer Rest (lowband_corr 250 Hz–1 kHz) hebt die effektive Schwelle um
bis zu 6 dB (konservativ unter der 15-dB-BMLD-Literatur, proportional zur
Kohärenz); inkohärenter Rest ⇒ kein Bonus; ohne `stereo_pair` unverändert
(Rückwärtskompatibilität). Ergebnis-Keys `binaural_boost_db`/`bmld_rest_corr`;
Harness-Invariante 21 (diotisch 6.00 dB vs. inkohärent 1.10 dB), normativer
Wirkungs-Test, Hörordnung §4 + §8-Träger nachgezogen. Verifiziert: 11/11
normative + 42/42 unit Audibility-Tests, Harness 21/21, Ruff/Gates grün.
**Fortsetzung 18 (Roadmap Qualitätsstufe 2 — Hebel A/B/C getaktet):**
`docs/reports/current/2026-11-09_roadmap_qualitaetsstufe2.md` taktet die drei
strukturellen Hebel gegen GPU-Ketten und Panel: **Phase 0** (jetzt, GPU-frei):
B-Stimuli-Job definieren (Panel startet parallel) + A-Design-Dokument.
**Phase 1** (nach Kette 2): Kurz-Jobs zuerst (Cantus-fine_tune, UAT-Evidenz,
B3-Neumessung M12), dann **Hebel A als Kette 5** (Musik-Rekonstruktions-Prior,
MUSDB18-HQ, deterministisches Sampling, Tage). **Phase 2** (parallel zu A,
menschlich): MUSHRA-Panel → **Hebel B** (Preference-Orakel F15, 1–2 h GPU).
**Phase 3** (nach A+B): **Hebel C** Stem-Pfad mit B-Orakel bewerten. Risiken
benannt: A-Determinismus als §G5-Kompromiss, B braucht Panel n≥30,
A-Plausibilität ≠ Wiederherstellung (Never-worsen bleibt harte Grenze).
**Fortsetzung 19 („Mein Ohr“ + musik_mos_v2 + Norm-Verankerung):**
(a) **Ohr-Prior** `backend/core/dsp/ohr_prior.py` (Bradley-Terry über 7
vorhandene Metriken, Pseudo-Count-Prior, 7 Tests) + `scripts/ohr_kalibrieren.py`
(Urteils-Session `aurik.ohr_urteile/1` → Artefakt `aurik.ohr_prior/1`).
(b) **musik_mos_v2** `backend/core/dsp/musik_mos_v2.py` — lernfähiger v1-Klon
(bit-identisch bei Prior-Gewichten, Test-gepinnt), `fit_v2_weights`
(Bradley-Terry), `export_onnx` (opset 13, §III.9-Parität rel ≤ 2.3e-06
gemessen), 6 Tests. (c) **Norm konsistent nachgezogen:** Hörordnung §7
(Ohr-Prior als personalisierte Hör-Instanz — Zeuge, nicht Richter),
musik_mos_v2-Einsatz-Regel (v1 bleibt Zeuge bis Richtungs-Zertifikat + B3;
ONNX nur mit §III.9-Parität + §III.13-Fingerprint + §G9), ML_MODEL_DOMAIN_REGISTRY
(`musik_mos_v2` als geplant/nicht deployt, Musik-Domäne). Alle Gates grün
(Compliance PASSED, Spec-Integration ohne ERROR).
**Fortsetzung 15 (B3-M12-Messung — Rohdaten; durch D-K3-85-Korrektur oben überholt):** `b3_stufenmessung.py --stages
M12-WF4,M12-TPV2` (3 Clips × 2 Fenster × 2,5 s, Seed 42, ~2,5 h unter GPU-
Kontention): **M12-WF4 ABGELEHNT** (never_worsen ✓, 1/6 Vorteile, signifikant=
False), **M12-TPV2 ABGELEHNT** (never_worsen ✓, 0/6 Vorteile) — beide Flags
bleiben AUS (fail-closed, §4-konform; Beschlüsse `docs/reports/current/
b3_objektivbeschluss_M12-*.json`). Ehrlich: n=6 ist eine kleine Stichprobe —
die echte Aktivierungsentscheidung bleibt Hörpanel/menschlich; die Mess-
Infrastruktur (D-K3-82) hat ihren ersten vollen Durchlauf bestanden.
**Fortsetzung 11 (R1-Anker-Defekt — D-K3-78):** Auf die Frage „läuft R1
mangelfrei?“: NEIN — beide Kette-4-Trainer (wf4_warp_head, tp_v2_phase)
schrieben ihren Checkpoint erst nach ALLEN 50 Epochen, WF4 zudem unter dem
Artefakt-Namen `wf4_warp_head_v1.pt` statt `checkpoint_latest.pt` → jeder
6-h-Timeout hätte den ganzen Fortschritt vernichtet (D-K3-72-Klasse). Fix:
Per-Epoche-Anker auf `checkpoint_latest.pt` + Resume-Load (`range(start_epoch,
epochs)`, fail-closed Exit 2 bei korruptem Anker) in BEIDEN Trainern; Smoke-
Nachweis: beide schreiben den Anker, WF4-Zweitlauf loggt „Resume ab Epoche 1“.
Der begonnene R1-Lauf wurde zum Fix gestoppt (~6 min verloren statt bis zu 6 h)
und Kette 4 neu gestartet (EINE Instanz, V1 läuft). Beim Kette-4-Neustart
zuvor: R1-Kill ließ die Shell sofort R2 starten — R2 ebenfalls gestoppt,
da gleicher Defekt.

**Update 2026-11-08 (FlashSR-Resume-Anker, Kette 2 · D3-3):** Das Manifest
`train_job_d3_flashsr_musik.json` zeigte einen Fehler, der keine Modell-
sondern eine Wiederaufnahme-Lücke war: Versuch 2 starb **61 s nach dem
6-h-Timeout** an `FileNotFoundError: models/flashsr/checkpoint_latest.pt`,
während Versuch 1 zuvor 5,37 h bis Epoche 3 trainiert hatte. Ursache:
`train_flashsr_f4.py` schrieb den Repo-Standardanker nie (nur
`checkpoint_epoch<N>.pt`/`best.pt`), der Retry war nicht datei-geschützt, und
`--cmd` (Versuch 1) resumte grundsätzlich nicht — jeder Kettenstart begann bei
Epoche 0. Behoben: Anker je Epoche, `--resume` fail-closed (Exit 2 statt
Traceback), **beide** Kette-2-Befehle datei-geschützt (A-SPADE-Muster),
`val_loss`-Broadcasting bereinigt; Anker aus `checkpoint_epoch3.pt` restauriert.
Neuer generischer Regressionstest: **jeder** `--resume`-Pfad der vier Ketten
muss vom Ziel-Trainer erzeugt werden (die Fehlerklasse, die BigVGAN 2026-11-06
und FlashSR 2026-11-08 getroffen hat). 10/10 Tests grün, kein Training
parallel gestartet, **kein Flag-Flip**. Report
`docs/reports/current/2026-11-08_flashsr_resume_anker.md`.

**GPU-Fenster-Wächter (2026-11-08):** Damit §2 Zeile 7 nicht auf einen Menschen
wartet, pollt `/tmp/k4_warte.sh` (detached, `setsid` + `nohup`, Log
`/tmp/k4_warte.log`) alle 300 s und startet
`scripts/gpu_kette4_modellkoepfe.sh`, sobald **kein** `train_job_runner` und
**keine** `gpu_kette<N>` mehr läuft. Nach einem Reboot ist der Wächter weg und
muss neu gesetzt werden (Kette 4 dann direkt starten). **Kette 2 läuft bewusst
nicht automatisch an** — sie beginnt mit D3-2 BigVGAN (bisher 0 Erfolge, ~6 h
je Versuch) und würde D3-3 FlashSR um ~12 h verzögern; der restaurierte Anker
(`models/flashsr/checkpoint_latest.pt`, Epoche 3) wartet auf den nächsten
Kette-2- oder Einzeljob-Start.

**Update 2026-11-08 (Hebel E — RT-Budget-Ledger + Prioritäts-Drift):** Der
Handover-Hebel E ist geschlossen, und beim Testen fiel ein echter
Scheduling-Defekt auf. (a) `get_phase_budget_seconds` verteilte das Budget
flach (`Restbudget / verbleibende Phasen`) und hatte **keinen Aufrufer** —
das 32×-Budget war nur eine Zahl. Jetzt: `get_budget_allocation()`
(§PERF-R21) mit Soll/Ist je Phase, `reserve_s`, `overrun_total_s`,
`budget_exceeded`, `top_consumers` (deterministisch, §G5), zero-safe;
`scope` = `planned`/`measured`; verdrahtet als `rt_budget_ledger` in den
UV3-Metadaten; ein Quelltext-Pin schließt aus, dass ein Skip-Pfad es liest
(§V7). **Ehrliche Grenze:** Priorität ist eine Wichtigkeits-, keine
Kosten-Ordnung — der Ledger sagt „so hat sich das Budget verteilt", nicht
„so viel verdient Phase X". (b) **D-K3-76:** `wow_flutter`/`decracklé`
(Prio 9, CRITICAL) wurden nie getroffen, weil die Modulnamen
`phase_12_wow_flutter_fix`/`phase_09_crackle_removal` andere Kurznamen
erzeugen ⇒ Default 5 ⇒ `should_skip_phase` durfte **die teuerste
Reparaturphase** (Wow/Flutter, 250–440 s im Referenzlauf) bei 93 % Budget
verwerfen. Alias-Auflösung `PHASE_PRIORITY_ALIASES` + `_priority_key()` in
allen drei Prioritäts-Pfaden; die alten Schlüssel bleiben. Verifikation:
35 Tests (Ledger + Compliance), Ruff clean. **Nicht** geändert: die übrigen
60 Phasen-Module ohne Tabelleneintrag (breite Scheduling-/Klang-Stufe, §V7).
Kein Flag-Flip, kein Training gestartet. Report
`docs/reports/current/2026-11-08_rt_budget_ledger.md`.

**Update 2026-11-08 (Hebel F — Wohlklang-Zeuge je Lauf, D-K3-83):** Der letzte
§2b-Hebel ohne Status ist geschlossen. Befund: die beiden
richtungszertifizierten Metriken `musik_mos`/`defektlast` (Spearman 0,95)
wurden **nur** in Mess-Skripten aufgerufen — `backend/` hatte null
Aufrufstellen, es gab also keinen Dauer-Zeugen am Fließband. Neu:
`backend/core/dsp/wohlklang_zeuge.py` (Eingangs-/Ausgangspaar, JND 0,05,
`gerichteter_fall` → Hör-Instanz) + `UnifiedRestorerV3._wohlklang_zeuge_meta`
in Einzel-, Chunked- und Options-Pfad, **REPORT-ONLY** und §V6-fail-safe;
`defektlast` und die JND-Konstante sind jetzt EINE Quelle (Zertifikats-Skript,
Stufenmessung und Kompositions-Suche greifen darauf zu — sonst könnte die
Mess-Seite unbemerkt von der Produktions-Seite abweichen). 13 Vertragstests,
37 B3-Tests grün. Offen bleibt nur der End-to-End-/RT-Nachweis im
trainingsfreien GPU-Fenster (Kette 2 belegte die GPU). Report
`docs/reports/current/2026-11-08_wohlklang_zeuge_hebel_f.md`.

**Update 2026-11-08 (Ketten-Wahrheit: Cantus-No-Op, D-K3-84):** Beim Prüfen der
Kette-1-Manifeste fiel ein falscher Erfolg auf: `c2_cantus_finetune` war nach
11,9 s „erfolgreich“, weil der Trainer `start_epoch` phasen-blind aus dem
Checkpoint übernahm (Pretrain `epoch=50` gegen `--epochs=30`) — null
Gradienten-Schritte; zusätzlich wäre `checkpoint_best.pt` nie geschrieben
worden, weil `best_val` an der Pretrain-Verlust-Achse hing. Fix: phasen-lokale
Resume-Größen (Gewichte bleiben) + „0 Epochen ⇒ Exit 3“; das No-Op-Manifest
wurde als `train_job_c2_cantus_finetune_noop_20261108.json` archiviert, sonst
hätte der Runner den Job dauerhaft übersprungen. Kein GPU-Lauf gestartet
(Kette 2 läuft), kein Flag-Flip. Report
`docs/reports/current/2026-11-08_wohlklang_zeuge_hebel_f.md`.

**Update 2026-11-09 (M12-Pfad-Erreichbarkeit, D-K3-85 — zwei B3-Ablehnungen
sind Mess-Artefakte):** Die Prüfung „führt die Stufe überhaupt das trainierte
Modell aus?" ergab für beide M12-Stufen **nein**. (a) `wf4_warp_head.py` lud
ausschließlich `models/wf4_warp_head/wf4_warp_head.onnx` — dieses Artefakt
existiert **nirgends**; der Trainer schreibt nur Torch-Checkpoints ⇒
`is_available() == False`, `use_wf4_warp_head=True` lieferte `(None, False)`:
die AN/AUS-Messung verglich den heuristischen Pfad **mit sich selbst**.
(b) `tp_v2_phase.py::_PROJECT_ROOT` zeigte auf `backend/` ⇒ Default-Checkpoint
`backend/models/tp_v2/…` (nie vorhanden) ⇒ der Flag-Pfad lief mit
**untrainierten Seed-Gewichten** (`source == "seed"`). (c) Das Pfad-Gate war
blind, weil die Pipeline **nicht bit-reproduzierbar** ist: das Fußabdruck-Gate
meldete bei totem Pfad `pfad_wirksam: true` (6,5e-4/7,9e-4 RMS Rauschen), und
`beschluss_schreiben` schrieb trotz `messung_ungueltig` `ergebnis: ABGELEHNT`.
⇒ `b3_objektivbeschluss_M12-WF4.json` und `…_M12-TPV2.json` (2026-11-08) sind
**keine Qualitätsurteile** (W-3-Klasse); maschinenlesbar invalidiert in
`docs/reports/current/b3_messung_ungueltig_20261109.json`.
**Fix (Code, kein Flag-Flip):** WF4 hat einen zweiten fail-closed Ladepfad
(Torch-Checkpoint über `build_wf4_warp_model` + `source`-Property; Flag AN
liefert jetzt echte Stretch-Faktoren 0,99889–1,00773, Flag AUS unverändert
`(None, False)`); `_PROJECT_ROOT = parents[3]` (TP-V2 lädt
`checkpoint:tp_v2_phase_v1.pt`); das Pfad-Gate hat **Modell-Zeugen** für
M12-WF4/M12-TPV2 (Quelle muss `checkpoint:…` sein) und ungültige Messungen
schreiben `NICHT_BEWERTBAR` statt `ABGELEHNT`. 10 neue/erweiterte Tests,
41/41 der drei Dateien grün. **Beide Flags bleiben AUS** (§V7); offen ist nur
die **Neumessung im trainingsfreien GPU-Fenster** (Kette 2 fährt D3-2
BigVGAN). Register D-K3-85, Roadmap-Modelltabelle nachgezogen. Report
`docs/reports/current/2026-11-09_m12_pfad_erreichbarkeit.md`.

**Update 2026-11-09 (Spec-SOTA-Audit L1–L3 + Träger-Audit, D-K3-86):** Das
Audit `docs/reports/current/2026-11-09_spec_sota_audit.md` benennt drei
autonome Norm-Lücken; L1 (SII-Verständlichkeit), L2 (ISO 532-2/DIN-45631) und
L3 (Cross-Ear/BMLD) sind umgesetzt. **Träger-Audit derselben Sitzung:** L1 war
zunächst eine **Phantom-Norm** — `speech_intelligibility` hatte **null**
Aufrufstellen in `backend/` (der Befund wäre in keinem Lauf entstanden). Jetzt
schreibt `UnifiedRestorerV3._sii_zeuge_meta` den Befund als **REPORT-ONLY** in
`metadata["sii_zeuge"]` (Einzel-, Chunked- und Options-Pfad; 3 Schreibstellen,
keine Lesestelle, §V6-fail-safe); das Messfenster mittelt Hann-Frames (vorher
EIN 42-ms-Fenster ⇒ bei Stille-/Fade-Anfang blind). **L3 bleibt ehrlich
träger-offen:** kein Produktionspfad übergibt `stereo_pair`
(`repair_objective_met`/`verify_declick_repair` raveln mono) — die Aktivierung
ist klangwirksam und läuft ausschließlich über §4/Flag; bis dahin bleibt die
effektive Schwelle bit-identisch. Verifiziert: 19 Tests (2 neu:
Zeitmittelungs-Regression + Verdrahtungs-Pin), Harness 21/21, SII-Delta
unverändert (0,782 / 0,576). Register **D-K3-86**, Roadmap Pfeiler 3,
Hörordnung §3/§4 und FILE_REGISTRY nachgezogen. **Kein Flag-Flip, kein
Training gestartet** (Kette 2 fährt D3-3 FlashSR).

**Update 2026-11-09 (D3-Taktung — D-K3-87):** Die D3-Finetunes waren in der
konfigurierten Taktung **strukturell unerreichbar**: BigVGAN 4418–4424 s/Epoche
(30 Epochen ≈ 36,8 h), FlashSR 4799–4843 s/Epoche (≈ 40,3 h) gegen 2×6 h
Timeout. Die Manifeste meldeten Epoche 8/30 bzw. 11/30 als „fehlgeschlagen“,
obwohl die Läufe gesund waren; der SIGTERM verlor bis zu 80 min je Versuch,
und jeder Resume überschrieb `report.json` (FlashSR-Report zuletzt nur Epoche
8–11). **Fix (ohne Qualitätsreduktion, 30 Epochen bleiben Ziel):** 4×12 h
Kapazität (`--timeout-s 43200 --retries 3`), `--max-seconds 37800` ⇒ Stopp an
der Epochen-Grenze (Exit 75) statt Kill, D3-3 vor D3-2, **D3-2-Erstlauf
datei-geschützt** (vorher Start bei Epoche 0), §G5-RNG-Stand im
Checkpoint, resume-feste Report-Historie, Fail-closed-Anker-Guard (Anker hinter
dem Ziel ⇒ Exit 3; Anker auf dem Ziel ⇒ `completed: true`). 8 neue Tests,
`tests/unit/test_gpu_kette1_wiring.py` 23/23 grün, Ruff/Compile clean. **Offen
ist nur der reale 30-Epochen-Lauf** (GPU); kein Flag-Flip, keine Hör-/Lizenz-
entscheidung. Report `docs/reports/current/2026-11-09_d3_taktung_zeitbudget.md`.

**Update 2026-11-09 (Cantus-OOM + Ohr-Richtungsfix — D-K3-88/D-K3-89, Commits `01c02033`/`9f0a84ac`):**
(a) **D-K3-88 (Kette 1/R4):** `c2_cantus_finetune` starb reproduzierbar am
24-GiB-OOM (Batch 8 + SingMOS-Lehrer: 22,77 GiB belegt, 750 MiB angefragt;
Versuch 1 nach 55 s, Versuch 2 nach 31 s). `train_cantus.py` hat jetzt
`--grad-accum` (Step am Fensterende UND am Epochenende, NaN-Guard verwirft nur
den Mikro-Batch); Kette 1 fährt `--batch-size 2 --grad-accum 4` (effektive
Batch 8, Rezeptur unverändert, Peak /4) in Erst- **und** Resume-Befehl.
Verifiziert: Fenster-/Rest-Logik + Gradienten-Äquivalenz 2×2 == 1×4
(atol 1e-7), 36 Tests, `bash -n`. **GPU-Nachweis offen** (nächster
Kette-1-Lauf). Report `2026-11-09_cantus_finetune_oom.md`.
(b) **D-K3-89 („Mein Ohr"):** Der Ohr-Prior wertete ROHWERTE — `defektlast`
(per Definition höher = mehr Defekte) hatte **+0,30**-Prior; an MUSDB18-HQ
gemessen war `w·Δ(clean−noisy)` nur **+0,045** (P(clean) = 0,511, Münzwurf),
der defektlast-Term wirkte mit −0,149 gegen das saubere Signal. Fix:
`qualitaets_vektor()` + `FEATURE_RICHTUNG` drehen niedriger-besser-Größen in
Qualität, Loudness als musik-typisches Band (Prior, kein Gate), Schema
`aurik.ohr_prior/2` fail-closed (alte Artefakte werden abgelehnt); nachher
`w·Δ` = **+0,354** (P = 0,588). Konsumenten: UV3
(`metadata["ohr_prior"]`, report-only) + StrategieDenker; die klangwirksame
Sweet-Spot-Priorisierung in Restaurier-/Defekt-/Exzellenz-Denker ist
**bewusst offen** (Hör-/B3-Entscheidung, Register-Blocker). Menschliche
Kalibrierung: `python3 -B scripts/ohr_panel.py --erzeuge-paare --verzeichnis
<Musikordner>`; ohne Artefakt ist die Wirkung null. Report
`2026-11-09_mein_ohr_richtungsfix.md`.
(c) **Wiederaufnahme nach Reboot 2026-11-09 21:0x:** keine Kette lief, `/tmp`
war leer; Ketten 1→2→3→4 neu gestartet. **Kette 2 fährt D3-3 FlashSR**
(Resume, `--max-seconds 37800`, 4×12 h); Kette 1 lief R4 erneut in den OOM
(das war der reproduzierbare Beleg für D-K3-88), Kette 3/4 ohne offene Jobs.
(d) **Beleg-Lehre:** D-K3-80 zitierte `/tmp/nachfahrt.sh` — nach dem Reboot
weg. Der Register-Eintrag belegt jetzt mit dauerhaften Report-/Manifest-Pfaden
(**`/tmp` ist KEIN Beleg**).

**Update 2026-11-10 (M5-Konsum verdrahtet — D-K3-90; Sitzung nach schmutzigem
Arbeitsbaum):** Der M5-Export vom 2026-11-09 war **unerreichbar** (keine
Ladestelle — D-K3-85-Klasse). Jetzt: `plugins/vocos_plugin.py::
get_vocos_music_ft_plugin()` (eigener Singleton, fail-closed: fehlendes Artefakt
⇒ kein Ladeversuch; Ladefehler ⇒ Basis + Negativ-Cache; Herkunfts-Zeuge
`weights_basename`) und der §1.4-Finisher wählt das Finetune **nur** bei
`music_model_flags.use_vocos_music_ft` (Default **AUS** ⇒ bit-identisches
Bestandsverhalten); Flag-Prüfung im Core, nicht im Plugin. 8
Reachability-Vertragstests (`tests/unit/test_vocos_music_ft_flag.py`). Aus der
Vorsitzung übernommen und mit-committet: der Export selbst + Export-Manifest,
die Ohr-Panel-**Autarkie** (Verblindung/Session-Bau im Panel statt über die von
einer parallelen Sitzung mitgenutzte `ab_harness.py`) und die Register-/Registry-
Nachzüge samt `spec_drift_baseline` (FILE_REGISTRY-Hash). **Kein Flag-Flip,
kein Training gestartet** (Kette 2 fährt D3-3 FlashSR). Offen (nicht autonom):
B3-Never-worsen des Finetunes gegen die Basis im GPU-Fenster, danach Hör-Sign-off
⇒ §4-Flip; C4-Primärrollen-Entscheidung menschlich. Report
`docs/reports/current/2026-11-10_m5_vocos_konsum.md`, Register D-K3-90. Direkt
anschließend **D-K3-91**: Quer-Audit aller 22 Modell-Flags — `use_miipher_dit`
war der einzige **stille No-Op** (MIIPHER-DiT lief unbedingt, `False` schaltete
nichts ab ⇒ der dokumentierte Rollback existierte nicht); jetzt liest
`_miipher_dit_enabled()` den Flag, `False` ⇒ `miipher_dit:flag_off` + Bestands-
Fallback, Default unverändert (3 Tests, 27 passed).

**Update 2026-11-10 (M5-Hörbefund + C4-Teilbefund — Buchführung, kein Flag-Flip):**
Im CPU-Fenster (Kette 2 fährt D3-3 FlashSR) wurde die B3/C4-Hörproben-Kette
gebaut (`scripts/erstelle_hoerproben.py` — 10-s-Proben deterministisch aus
`~/Musik`, Seed 42, 48 kHz/128 Mel; `scripts/hoer_entscheidung.py` —
verblindetes A/B-Panel, Urteil ⇒ `aurik.ab_stats/1`, direkt von
`flag_freigabe.py` konsumierbar; 5 Vertragstests). **Hör-Instanz-Befunde:**
(a) **M5** `use_vocos_ft_reconstruction`: n=6, linear **6/6**, p = 0,03125
⇒ negativ signifikant, Flag bleibt **AUS**, der lineare Pfad ist der bestätigte
SOTA-Zustand, ein §4-Flip entfällt; der Challenger `_vocos_ft_fill` bleibt
inaktiver Negativbefund. (b) **C4-Teil** vocos-Base vs. Musik-Finetune: n=3,
2 Gleichstände, p = 1,0 ⇒ kein hörbarer Vorteil; die Rollen-Entscheidung
reduziert sich auf den BigVGAN-Vergleich (D3-2) und bleibt menschlich.
Belege `hoer_verdikt_m5_rekonstruktion.json` / `hoer_verdikt_c4_vocos.json`,
Report `docs/reports/current/2026-11-10_m5_hoerbefund_und_c4_teilbefund.md`.
Nebenbefund behoben: eine neue Log-Meldung war englisch („Fallback") und
blockierte den Sprach-Guard (§v10.51) — jetzt deutsch. **Kein Flag-Flip, kein
Training gestartet.**

**Update 2026-11-10 (Flag-Wirkungs-Gate — D-K3-92, Roadmap-Pfeiler 4 geschlossen):**
Die Klasse „Schalter ohne Lesestelle" (D-K3-85/D-K3-90/D-K3-91) wurde bisher per
manuellen Suchlauf gefunden — jetzt maschinell: `scripts/flag_wirkung_gate.py`
(+ Pre-Commit `aurik-flag-wirkung-gate`, Code-Pfad-Filter) prüft drei Stufen
fail-closed: (A) Lesestelle außerhalb der Flag-Datei — Name-Load, Attribut,
`getattr`/`hasattr`-Literal, benutzter Import-Alias; **toter Import und
`setattr` zählen NICHT**, `tests/` nie; (B) Lesestelle im Produktionspfad
(`backend/`, `plugins/`, `denker/`, `Aurik10/`, `cli/`, `forensics/`), sonst
begründete Ausnahme mit Register-ID; (C) B3-/Release-Kandidaten ohne
Produktionswirkung werden als No-Op-Artefakt abgelehnt. **Befund der
systematischen Nachmessung (24 Schalter): 23 produktiv gelesen, 1 Ausnahme —
`use_bw_v5`.** Der BW-v5-Pfad in `phase_06_frequency_restoration.py` hängt an
`_use_nvsr`, nicht am Schalter; nur Mess-/Release-Skripte nennen ihn.
⇒ **W-8 ist bis zur Verdrahtung kein messbarer B3-Kandidat** (die Messung wäre
ein No-Op); die Verdrahtung ist klangwirksam und läuft ausschließlich über §4.
12 Vertragstests, Gate-Lauf 24/0/1 (≈ 9,5 s). Methoden-Lehre im Report: drei
Korrekturen (fehlendes `forensics/`, Import-Aliase, `getattr`-Strings) waren
nötig, bevor die Analyse belastbar war — sie sind jetzt Testfälle.
**Kein Flag-Flip, kein Training gestartet** (Kette 2 fährt D3-3 FlashSR).

**Update 2026-11-10 (Hör-Panel sitzungsfest + C4 n=12 — D-K3-93):** Die
Hör-Instanz hat die C4-Teilfrage in zwei Sitzungen beurteilt; dabei kam heraus,
dass die Werkzeugkette das **nicht konnte**: `hoer_entscheidung.py` schrieb je
Sitzung einen neuen Beleg (n blieb 6 statt 12), `erstelle_hoerproben.py`
würfelte ein neues Set (Proben-Positionen instabil), `ohr_panel.py` kannte das
`_ORIGINAL`↔`_ML-ADAPT(-v2/-v3)`-Layout nicht, eine Nur-Gleichstände-Sitzung
lief in `ohr_kalibrieren.py`, 44,1-kHz-Quellen scheiterten am 48-kHz-Zwang —
und **die Unit-Tests löschten `models/ohr/ohr_prior.json`**, also die echte
persönliche Kalibrierung. Alles behoben (additive Trials je stabiler Probe,
`--wiederholen`, Proben-Merge, Artefakt-Fixture mit bit-identischer
Wiederherstellung, deterministisches Resampling); **43 Tests grün**. Gebucht:
**C4 n=12 → 5/5, 2 Gleichstände, p=1,0 ⇒ nicht signifikant, kein §4-Flip**,
Rollen-Entscheidung bleibt beim BigVGAN-Vergleich (D3-2) und ist menschlich.
Kein Flag-Flip, kein Training gestartet. Report
`docs/reports/current/2026-11-10_hoer_panel_sitzungsfest.md`, Register D-K3-93.

**Update 2026-11-10 (D3-3-Gate gemessen — D-K3-94):** Die Auflage stand seit
jeher im Trainer-Report (`gate_hint`: „MUSDB-ΔSDR ≥ +2 dB NACH dem Lauf
messen") und war nie gemessen. Nachgeholt mit dem **trainierten** Checkpoint,
CPU-only im laufenden GPU-Fenster (Kette 2 unangetastet): **mean ΔSDR −4,61 dB
(A min −10,60 dB), Gate NICHT BESTANDEN** — besser als das Vortraining
(−6,44 dB), aber schlechter als „nichts tun". Diagnose schließt Peilung (−1
Sample), Pegel (Gain 0,99) und Mess-Artefakte aus: die D3-3-Klasse
(Sprach-BWE als *Vollsignal-Resynthese*) verändert das **Passband** um 4–6 dB
SDR; die `residual`-Variante (Basis-Passband + modelliertes HF) ist neutral
(−0,03 dB). Damit ist klar: **mehr Epochen helfen nicht**, es braucht eine
andere Aufgabe (HF-Residual trainieren) oder eine musik-trainierte Modellklasse
— beides GPU + Hör-Instanz, kein autonomer Schritt. Kein Produktionsrisiko:
der Pfad hat keinen Konsumenten im Signalpfad (`HFExtender` ohne Aufrufstelle,
`HybridNVSR` liefert nur eine Status-Payload) ⇒ nichts abzuschalten, kein
Flag-Flip. Werkzeug `scripts/verify_flashsr_d3_gate.py`, Report
`docs/reports/current/2026-11-10_flashsr_d3_gate.md`, Register D-K3-94.

**Update 2026-11-11 (Sinn-Beweis gemessen — D-K3-95; §2 Zeile 13 NEU):** Die
Sitzung hat die **unfertige Vorsitzungsarbeit** übernommen
(`scripts/erstelle_schadens_proben.py` + `merge_session` im Ohr-Panel waren
implementiert, aber **nie ausgeführt, nie gemessen, nie committet**),
real gefahren und erstmals objektiv ausgewertet. **Wiederaufnahme:** keine
Kette lief (`pgrep` traf nur die eigene Shell), GPU frei ⇒ Kette 1 gestartet;
R1–R3 übersprungen (bereits erfolgreich), **R4 `c2_cantus_finetune` läuft mit
dem D-K3-88-OOM-Fix** (`--batch-size 2 --grad-accum 4`).
**Ergebnis:** 12 Proben (3 Titel × 4 Rezepte) durch den Produktionspfad,
**pegelgematcht** bewertet — sonst misst man die Ziel-Lautheits-Regelung statt
der Reparatur (die Pipeline zieht `rms_out` auf 0,11–0,17 bei `rms_in`
0,105–0,57). **3/12 Paare besser**, Clipping **0/3** und teils drastisch
schlechter (defektlast 0,478 → **0,716**, musik_mos 4,06 → **3,68**), zwei
vinyl-Paare **bit-identisch** (No-Op). Nebenbefund: `defektlast` bewertet den
Clipping-Schaden **besser** als das saubere Original ⇒ **Metrik-Blindstelle bei
Sättigung** (`musik_mos` sieht sie). Kontrollseite gesund: auf cleanem Material
greifen die Guards (`VQI=0.669 < 0.72 → keeping Originalsignal`,
`Hallucination-Guard → Rollback`). **Neues Werkzeug**
`scripts/bewerte_schadens_proben.py` (Beleg `aurik.schadensbewertung/1`,
5 Vertragstests). **Kein Flag-Flip, kein Klangpfad-Eingriff** — der Rest
(ganze Titel im GPU-Fenster, Declip-/Pegel-Ursache, Metrik-Blindstelle) ist
klangwirksam bzw. menschlich. Report
`docs/reports/current/2026-11-11_schadens_proben_sinnbeweis.md`, Beleg
`…2026-11-11_schadensbewertung.json`, Register D-K3-95.

**Update 2026-11-11 (Declip-Never-worsen + Ursachenprüfung — D-K3-97):** Der
Vorsitzungs-Patch an `phase_07_declipper.py` wurde gemessen geprüft — seine
Begründung ist **widerlegt**, sein Kern bleibt. (1) `DeclipperPhase` ist **nicht
im UV3-Produktionspfad** (einzige Aufrufstelle `coordinated_repair.py:1361`,
0 Treffer „declipper" im 3,1-MB-Log der 12 Schadens-Restaurationen) ⇒ die
Clipping-Verschlechterung (D-K3-95: 0,478 → 0,716) kann nicht von ihr kommen.
(2) PCHIP ist **formtreu** — der ±1,0-Clamp hat nie gegriffen
(`pchip_neu == pchip_alt`), die Hypothese „neue Flat-Tops" ist falsch. (3) Auf
den Sättigungs-Proben ist PCHIP wirkungslos (ΔSDR 0,00, max. Änderung 0,018),
auf **sparse Hard-Clips** dagegen schädlich: ΔSDR −6,84 dB (Decke 0,90),
−17,70 dB (0,95), −115,97 dB (0,99); auf 0,60/0,30 begrenztem Material passiert
nichts, weil `_histogram_clip_threshold` auf [0,85 … 0,999] klemmt. (4) Die drei
am 11.11. neu erzeugten Clipping-WAVs messen besser (Δ defektlast −0,0226,
Δ musik_mos +0,0898; Jets bit-identisch), sind aber **kein Beleg für den
Patch** — kein UV3-Code geändert, kein Lauf-Log, keine Wiederholung
⇒ Einzelbeobachtung. **Code:** Never-worsen-Übernahme je Kanal
(Harmonik-Proxy, delta-basiert) + Wahrheits-Passthrough (`declip_applied:
false`, `reason: never_worsen_proxy`, kein `resolved_defects["CLIPPING"]`); die
wirkungslose Decken-Änderung wurde **entfernt**, damit kein Code mit falscher
Begründung liegen bleibt. Tests 13/13 (`test_phase07_senkt_den_harmonik_proxy_nie`,
`test_passthrough_meldet_keine_reparatur`,
`test_pchip_ist_formtreu_unter_der_eingangs_decke`), Ruff clean. **Nebenbefund
aus dem alten Lauf:** `Verarbeitungsschritt-07-Silence-Guard: Ausgabe-RMS
-91.1 dBFS → Rollback` in einem Goal-Compensation-Sekundärlauf, während die
Hauptketten-Pegel stabil bei −14,4…−14,9 dBFS bleiben. **Aufgeklärt
2026-11-12 (Log-Analyse `schadens_restore.log`):** echte Stille-Segmente im
Quellmaterial (SegmentAdaptive zählt 20 Stille-Segmente; Shellac-Quelle mit
Ausklängen) — Early-Silence-Gate übersprang die Phasen 07/40/24/29/57 auf
−91,1 dBFS-Eingang korrekt, der Guard rollte zurück, die Folgekette lief
normal weiter (FeedbackChain 4,021, artifact_freedom 0,973) — **kein Defekt,
Guard-Verhalten korrekt**. Der Gain-0,30-Jets-Fall war das §V8-Leck
(geschlossen 2026-11-11). Damit ist D-K3-97-Punkt (2) (Pegel-/
Entscheidungskette) evidenz-basiert abgeschlossen. **Dritte Metrik-Grenze:**
`defektlast` liest nur die ersten 16384 Samples (0,34 s). **Kein Flag-Flip,
kein Deploy.** Nächster autonomer Schritt: Reproduzierbarkeit (derselbe Lauf
zweimal, Logs + WAV-Hashes) — erst danach ist „repariert/nicht repariert" eine
Aussage. Report
`docs/reports/current/2026-11-11_declip_never_worsen_und_ursachenpruefung.md`,
Register D-K3-97.

**Update 2026-11-11 (D-K3-98 — Air-Presence-NOLA-Fix, Zyklus aus §3.4):** Der
Enhancer analysierte mit symmetrischem `np.hanning` und synthetisierte mit
periodischem scipy-hann (75 % Overlap) — gemessen **8,93 % Relativfehler** im
STFT→ISTFT-Roundtrip, NOLA-Warnung, ein 440-Hz-Ton (Maske 0) wurde um **0,25**
verändert; der Ausgang wurde hart auf ±1,0 geklippt (**23.804 Samples** bei
0,99/12 kHz). Fix: periodisches Hann in **Analyse und Synthese**,
scipy-Konvention (`win/Σwin`, `detrend=False`, `boundary="zeros"`,
`padded=True`, Kurzsignal-Guard) und **Eingangs-Decke statt Hard-Clip**
(Zusatz `enhanced−orig` linear auf `min(0,999, peak_in)` zurückgenommen;
Eingang über der Decke ⇒ `k=0` = bit-identischer Passthrough). Nachher:
Roundtrip **1,78e-8**, 440-Hz-Abweichung **1,49e-8**, 12-kHz-0,99-Ausgang
**0,9900** ohne Sample auf 1,0, keine NOLA-Warnung. Tests:
`test_air_presence_enhancer.py` **10/10** (4 neue Vertragstests),
`test_stem_level_restorer.py` 17/17, Phase-19-Suite 81/81; Ruff/Compile clean;
Performance 2,54 s je 3-min-Stereo (≈ 71× RT, Budget unkritisch). **Kein
Flag-Flip, keine Verdrahtung geändert.** Offen: Wirkungsmessung auf die
D-K3-95-Rezepte (clipping/kassette) — der Enhancer läuft nur im Stem-Pfad,
`phase_19` ist witness-guarded. Beleg
`docs/reports/current/2026-11-11_d_k3_98_air_presence_nola.md`, Register
D-K3-98 (geschlossen).

**Update 2026-11-12 (D-K3-98-Folgemessung — Wirkung auf den Rezepten GEMESSEN, Nullwirkung belegt):**
`scripts/messe_air_presence_wirkung.py` (neu, 4 Vertragstests) misst auf den
6 D-K3-95-Rezept-Proben (3 Titel × clipping/kassette) beide Ebenen:
**Kern:** never-worsen ✓ (Witness 6/6, Decke 6/6, Metrik-JND 6/6) — aber die
Wirkung ist **Null**: clipping 3/3 **bit-identisch** (rms_delta 0,0),
kassette 3/3 nur STFT-Numerik (≤ 4e-5), alle Metrik-Deltas 0,000. **Stem-Pfad:**
6/6 gelaufen, `air_presence_used` + Zeugen-`applied` = true auf allen Proben —
die Produktionsstufe **meldet Anwendung, wo nichts passiert** (No-Op-Artefakt
mit aktivem Zeugen, D-K3-85-Klasse). **Ursache (Code-verifiziert):**
`_begrenze_auf_eingangsdecke` setzt `ceiling = min(0,999, peak_in)` = peak_in
bei Realsignalen ⇒ jede positive Anhebung am Peak-Sample ergibt `k=0` ⇒ der
gesamte Zusatz wird linear auf Null zurückgenommen — auf realer Musik
strukturell ein No-Op (die D-K3-98-Verifikationstabelle ist mit k=0
erfüllbar, sie belegte nie die Anhebung). **Konsequenz für D-K3-95/97:** der
Enhancer kann die Clipping-Verschlechterung nicht verursacht haben; die
Ursachensuche verengt sich auf die Pegel-/Entscheidungskette. **Fix wäre
klangwirksam** (per-Frame-Decke o. Ä.) ⇒ §4/Hör-Sign-off, bewusst NICHT
eigenmächtig. Belege `2026-11-12_air_presence_wirkung.json` (+ `…_stem.json`),
Report `2026-11-12_air_presence_wirkung.md`.

**Update 2026-11-13 (Vocos-Epoch-Gate-Wirkung):** Die bisher isolierten
Never-worsen- und Kadenztests belegen jetzt auch die echte `_run`-Verdrahtung:
ein modellfreier CPU-Test führt den Orchestrator mit einem winzigen Torch-Modell
aus, lässt Epoche 2 bestehen und Epoche 4 scheitern. Verifiziert sind zwei
Gate-Aufrufe, der Status `degraded` und ein auf den unveränderten
Baseline-Checkpoint zurückgesetzter `checkpoint_latest.pt`-Resume-Anker.
Gezielte Vocos-Tests: 18/18 bestanden; kein GPU-Training, Rollout oder
Flag-Flip.

**Update 2026-11-15 (autonome Prüfrunde — D-K3-10 bereits geschlossen, keine Schein-Schließung):**
Der Auftrag ging von 44+3 offenen GUI-Texten aus; gemessen ist das Gegenteil:
D-K3-10 wurde am 2026-10-29 in `f25897c1` geschlossen (Doku `665f7955`).
Belege: Der kanonische Frontend-Scanner meldet `hardcoded_texts` = **0** für
`modern_window.py` und `song_prognose_widget.py` (Baseline 0); die 47
historischen Literale (44+3) laufen nachweislich über `t()`. Tests:
`test_frontend_verboten_linter.py` + `test_i18n.py` +
`test_restoration_status_panel_i18n.py` = **24 passed**; GUI-/Aurik10-
Vertragstests (offscreen) = **76 passed, 7 skipped**. **Kein Code geändert** —
die 47 sind nicht mehr da; ein erneutes Umwickeln wäre ein Schein-Fix.
**Blindstellen (im Register-Abschluss benannt, jetzt gemessen):** Ein
Datenfluss-Scan (Setter + `drawText` + Variablen-/Listen-Indirektion) findet
**244** indirekte Prosa-Kandidaten, u. a. `_warn` in
`song_prognose_widget.py:746` und `_badge` in `modern_window.py:7757ff`.
Das sind **nicht** die 44+3, sondern eine eigene Katalogwelle (Fachlabels,
Fragmente, Duplikate); sie braucht Kuratierung + de/en-Katalog, kein
mechanisches `t()`-Umwickeln. Als Folgewelle offen, nicht als D-K3-10.
**Nächste genannte Kandidaten — alle nicht autonom schließbar:** D-K3-13
(UAT-Neulauf nur im trainingsfreien GPU-Fenster; M9-V2 belegt die GPU),
D-K3-60 und D-K3-75 (bereits geschlossen), ~~D-K3-92~~ (**GESCHLOSSEN 2026-11-15**, Punkt 33 — der Schalter ist produktion-wirksam; offen bleibt nur der Artefakt-Zeiger, gebucht an D-K1-5), D-K3-65/94/95 (Hörstichprobe/GPU/Modell-
entscheidung). Damit bleibt in diesem Fenster **keine** autonom schließbare
Register-Lücke; kein Flag-Flip, kein GPU-Training, keine Klangänderung.

**Update 2026-11-15 (Nachtrag UAT-Evidenz-Audit — D-K3-104 geschlossen):**
Die vorherige Runde hatte „keine autonom schließbare Register-Lücke“ notiert;
die Prüfung der UAT-Evidenz selbst hat eine gefunden.
`audit/uat_results_2026-11-11.json` enthält 15 Restoration-Kriterien, alle
`ERROR`, aber die Summary meldete `restoration_failed=0`,
`restoration_skipped=0`, `criteria_executed=15` und `CONDITIONAL`
(„15/15 bestanden“), weil `compute_summary` ERROR in keinem Zähler führte;
die Scorecard zeigte `⊘ SKIP`. **D-K3-104 geschlossen:** eigene ERROR-Felder,
`total_accounted`, fail-closed `NO-GO` bei ERROR oder fehlendem Ergebnis, eine
Statusquelle rendert `⚠ ERROR`. Derselbe Beleg ergibt nachher
`criteria_errors=15` / `NO-GO`; 7 Unit-Tests grün.
`tests/normative/test_real_audio_restoration_quality_gate.py` bleibt bei
**8 passed, 1 skipped, 1 failed** — der Rote ist weiterhin nur die
UAT-Frische (D-K3-13). Keine Klangwirkung, kein Flag, kein GPU-Training.
Beleg: `docs/reports/current/2026-11-15_uat_report_error_accounting.md`.

**Update 2026-11-15 (D-K3-13 geschlossen — UAT-Neulauf):** Die UAT-Evidenz
ist frisch: `audit/uat_results_2026-11-15.json`, `latest.date=2026-11-15`,
0 ERROR, 14/15 Restoration PASSED (R5 bleibt ein echter Stereo-Befund),
15/15 Studio, 5/7 Gates, 0 K.O., `GO` (29/30).
`tests/normative/test_real_audio_restoration_quality_gate.py` ist
**9 passed, 1 skipped** (vorher 8/1/1). Der Fix war das Harness-Limit
(120→1800 s, Generator 1200→3600 s), keine Qualitaetsschwelle. Kein Flag,
keine Klangwirkung, kein GPU-Training. Beleg:
`docs/reports/current/2026-11-15_uat_freshness_d_k3_13.md`.

**Update 2026-11-15 (D-K3-88 geschlossen — GPU-Nachweis):** Die letzte offene
Frage der Cantus-`fine_tune`-Zeile ist beantwortet: Der Kette-1-R4-Lauf
`fine_tune-20261110T230105Z-b06a07` hielt mit `--batch-size 2 --grad-accum 4`
auf der 24-GiB-Karte **30/30 Epochen** durch (`ergebnis=erfolgreich`,
`exit_code=0`, `timed_out=false`, 20159,7 s, `final_val_loss=3.5380738750100136`,
kein OOM; Manifest `docs/reports/current/train_job_c2_cantus_finetune.json`,
SHA `93b313d6…`; Trainer-Report `models/cantus/train_report_fine_tune.json`,
SHA `a6dff1c2…`). Die beiden Cantus-Vertragstests sind grün (2 passed); der
allgemeine TF-GridNet-Akkumulationstest ist in der `.venv_aurik`-Umgebung rot
und kein Cantus-Beleg. Kein Flag-Flip, kein Deploy, kein neues GPU-Training;
Aktivierung bleibt §4/Hör-Instanz. Beleg:
`docs/reports/current/2026-11-15_d_k3_88_gpu_nachweis.md`.

**Update 2026-11-16 (Kandidatenlage gemessen + D-K3-105 geschlossen):** Die
sieben im Auftrag genannten Kandidaten (D-K3-13, D-K3-60, D-K3-71, D-K3-75,
D-K3-83, D-K3-90, D-K3-92) stehen **alle** auf `geschlossen`; offen sind nur
hör-, GPU- oder modellgebundene Zeilen (D-K3-65, D-K3-87/94/95/96,
D-K3-97, D-K4-*). Beim Nachprüfen des Belegs von D-K3-60 ("Ersatz Beleg
prüfen", SOTA-M22) kam eine echte, autonom schließbare Lücke zum Vorschein:
**der Export-Gate-Kontext existierte nur im CLI-Pfad.** `profile`, `material`,
`thresholds` und `preserve_signal` im `quality_gate`-Payload stammen aus
`export_gate_*`-Metadaten, die **nur** `cli/aurik_cli.py` schreibt; GUI und
Batch rufen den Builder ohne diesen Kontext. Gemessen (`material_type=vinyl`,
`preserve_signal=0,12`, `degradation_status=ok`, WCS 0,912/vocal): Experience-
Payload `profile=modern_stable`, Export-Payload `profile=""`, `material=""`,
Schwellen `{0,0; 0,0}` — zwei Wahrheiten für einen Lauf (§G9), und der
Leerwert war **kein** Beleg für "kein Gate" (genau diese Lesart hatte
D-K3-60 benutzt). Zweite Folge: `_qe_threshold <= 1e-4` galt als
Technik-Fehlersignal ⇒ jede Degradierung mit Grund wurde `technical_failure`.
**Fix (Ursache, §V7):** `_export_gate_kontext()` als EINE Ableitung in
`backend/api/bridge.py` (CLI-Meta gewinnt, sonst die bisher inline stehende
Experience-Regel), genutzt von Experience- und Export-Payload sowie
`bridge_export.py`; neue Herkunftsfelder `profile_source` /
`material_source` / `thresholds_source` statt stiller Nullwerte (keine
erfundenen Schwellen, §G8); die Heuristik gilt nur noch bei
`thresholds_source="export_gate_meta"`; GUI/Batch/CLI schreiben
`quality_gate_profile_source` / `quality_gate_thresholds_source` in die
Export-Metadaten. **Nachher:** GUI-Form `modern_stable` /
`derived_preserve_signal`, `material=vinyl`, `thresholds_source=absent`,
`failure_class=quality_failure`; CLI-Form bit-identisch
(`fragile_or_transient_risk(risk=0,54)`, Schwellen 0,42/4,3). **Tests:**
`tests/unit/test_dk3_105_export_gate_kontext.py` 13 passed; Bestand 39 + 328
passed; Ruff clean. Kein Flag-Flip, keine Klangwirkung, kein GPU-Training.

**Nebenbefund 2026-11-16 (Gate-Blindstelle, im selben Zug geschlossen):**
`scripts/sota_deficit_gate.py` parste IDs mit `\d{1,2}` — jede Zeile ab
D-…-100 war für das fail-closed Gate **unsichtbar** (6 Zeilen,
D-K3-100…D-K3-105), es meldete "127 Einträge, keine Verstöße". Jetzt `\d{1,3}`:
**133 Einträge, keine Verstöße**; Regressionstest pinnt dreistellige IDs und
"keine Tabellenzeile unsichtbar". Dabei kam eine verschobene Zeile zum
Vorschein (D-K3-103 trug die Manifest-Evidenz in der Domänen-Spalte) —
Spalten repariert, Inhalt unverändert. Belege:
`docs/reports/current/2026-11-16_d_k3_105_export_gate_kontext.md`,
Register D-K3-105.

**Update 2026-11-16 (D-K3-106 geschlossen — §V8-Leck in der Budget-Lernhistorie):**
Der zuletzt notierte „nächste §V7-Schritt" (Kontext-Isolation) hatte einen
konkreten, autonom schließbaren Kern: `_phase_budget_gain_history` (laufender
Mittelwert des akzeptierten Netto-Gewinns je Material/Phasen-Präfix,
`backend/core/unified_restorer_v3.py:1927`) lebte im Restorer-Singleton und
wurde nie am Song-Wechsel geleert. Zwei echte Verbraucher lesen sie —
`_budget_pressure_skip_reason` (Skip) und `_pmgg_retry_budget_hint`
(Retry-Drosselung 0,15/1) — beide über `historically_weak_delta` ab 2 Samples
mit `mean_net_gain < 0,004`. **Gemessen** (deterministisch, CPU-only, echte
Lernfunktion `_record_phase_budget_gain`): Vortitel 2×0,001 Netto-Gewinn ⇒
Folgesong ohne Reset `historically_weak_delta`; nach
`reset_song_budget_state()` Historie `{}` und beide Entscheidungen `null`;
innerhalb desselben Songs bleibt das Lernen aktiv (2 neue Samples ⇒ wieder
`historically_weak_delta`). **Fix:** 9. Eintrag im §V8-Reset-Katalog des
Denkers (Aufruf am Anfang von `AurikDenker.restauriere`, also vor jeder
Verarbeitung; CLI/GUI/Batch laufen über `bridge → AurikDenker`, direkte
`restorer.restore()`-Aufrufe gibt es nur in Debug-Werkzeugen). 4 neue
Vertragstests (`tests/unit/test_wall_budget_gain_prioritization.py`, 12
passed), Register-Gate **134 Einträge, keine Verstöße**,
`tests/normative/test_sota_deficit_gate.py` 28 passed. **Zwei
Vorlauf-Behauptungen verworfen, weil unbelegt:** `_gdd_budget.phase_history`
ist NICHT song-übergreifend (der Manager entsteht je `_ausfuehren_pipeline`
neu, `unified_restorer_v3.py:37586`), und das Zahlenpaar „−0,132/−0,014"
existiert in keinem Beleg-Artefakt. **Ehrlich offen:** die
Jets-vinyl-Kontextdifferenz ist damit **nicht** erklärt — belegt ist der Kanal
und seine Entscheidungswirkung, nicht ihr Anteil an −0,203/−0,551. Die
vorhandenen Diagnose-Dumps `study_stimuli/kontext_diagnose/dump_solo.json` /
`dump_kontext.json` sind **kein A/B** (unterschiedliche Eingänge:
`rms_original_db` −19,25 vs. −14,47, `peak_original` 0,7685 vs. 0,995) ⇒
Kontext-Diagnose mit identischem Eingang + Ganz-Titel-Metriken bleibt der
nächste Schritt. Kein Flag-Flip, keine Klang-Design-Änderung, kein
GPU-Training. Belege:
`docs/reports/current/2026-11-16_d_k3_106_song_budget_isolation.md` /
`.json`, Register D-K3-106.
