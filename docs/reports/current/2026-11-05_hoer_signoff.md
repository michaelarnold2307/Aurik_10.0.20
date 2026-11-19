# Hör-Sign-off 2026-11-05 — Hör-Instanz

> Ausfüllen je Block (Format siehe `docs/reports/current/2026-11-05_hoer_runde_gesamtpaket.md`).
> Dieses Dokument ist die Hör-Instanz-Akte; es wird unverändert zum
> `--signoff`-Eingang des §4-Verfahrens (`scripts/flag_freigabe.py`).

## Block A — W-1 (Höhen-Behandlung heraushören)
Urteil:
Kommentar:

## Block B — W-2 (SCNet-Musik)
Urteil:
Kommentar:

## Block C — EAR-VAE v1 → v2
Urteil: NO-GO (v2 nicht übernehmen; v1 bleibt) — siehe Entblindungs-Sektion
Kommentar: 12:3 für v1 (p=0,035); v1 dumpfer/sauberer, v2 heller/rauschiger;
Wunsch-Richtung: heller UND weniger Rauschen (v2-Helligkeit + v1-Sauberkeit)

## Block D — Ermessen
Status: **VERTAGT (2026-11-05)** — Hör-Instanz kann nicht mehr hören;
Entscheidung W-8 / W-10 / W-11 / Material-CRNN steht aus. Nächste Session fortsetzen.
W-8 (BW-v5): offen
W-10 (musik_mos-Umstellung): offen
W-11 (PANNs-Singing-Offset): offen
Material-CRNN: offen
Kommentar: Sitzungs-Ende 2026-11-05; Blöcke A/B/C abgeschlossen, Block D offen.

---

# Hör-Protokoll (Roh-Eingaben der Hör-Instanz, 2026-11-05)

- pair01: **B besser als A** (Block unklar — beide Blöcke haben ein pair01; Roh-Notiz)
- pair02: **A besser als B** (B: Artefakte hörbar)
- pair03: **B besser als A** (A: Artefakte hörbar)
- pair04: **A besser als B** (B: Artefakte hörbar)
- pair05: **B besser als A** (A: Artefakte hörbar)
- pair06: **A besser als B** (B: Artefakte hörbar)
- pair07: **B besser als A** (A: Artefakte hörbar)
- pair08: **A besser als B** (ohne Artefakt-Notiz)
- pair09 (Fangfrage): **A = B** (kein Unterschied) — korrekt erkannt ✓

## Block A — W-1 (ENTBLINDET, 2026-11-05)

Alle 8 echten Trials: **UNBEHANDELT bevorzugt (0/8 für die Behandlung)**;
in 6/8 Trials wurden die Artefakte auf der BEHANDELTEN Seite gehört
(pair02–07); beide Fangfragen korrekt als „identisch" erkannt.
Schlussfolgerung (Hörordnung §7): Die Höhen-Behandlung ist hörbar, aber als
Verschlechterung (Artefakte) — die musik_mos-Messung ist bestätigt, der
HF-Gain-Beleg („BESTÄTIGT") misst einen als künstlich gehörten Gewinn.

**Urteil Block A (Interpretation der Hör-Instanz, Bestätigung ausstehend):**
W-1 Höhen-Behandlung = **NO-GO** (nicht beibehalten); W-13 folgt dem NO-GO.

## Block B — W-2 SCNet (Bereits entschieden — kein Neuhören)

Hör-Instanz 2026-11-05: Block bereits angehört und entschieden (K0-Runde
2026-10-30). Bestands-Urteil aus `docs/reports/current/2026-10-30_k0_signoff_paket.md`:
**K0-A SCNet = NO-GO** (Strang H komplett: alle drei Musik-Rollouts verworfen).
Der heutige Kurz-Check (pair01 = B bevorzugt) notiert als unverbindliche
Bestätigung; die Entscheidung bleibt NO-GO. Kein Re-Do, kein neues Verfahren.

## Block A — W-1: ENDGÜLTIG BESTÄTIGT (2026-11-05)

**Hör-Instanz-Urteil: W-1 Höhen-Behandlung (HR-V1) = NO-GO** — ausdrücklich
bestätigt („bestätigt"). Begründung: 0/8 Trials bevorzugten die Behandlung,
6× wurden ihre Artefakte gehört (§7-Konflikt zu Gunsten musik_mos entschieden).
**W-13 (BigVGAN-F3-Studio) folgt dem NO-GO** (Kopplung an W-1, KP-5).

## Block C — EAR-VAE v1→v2 (Hör-Protokoll, 17 blinde Paare)

- pair01: **B besser als A**
- pair02: **A besser als B** (B: mehr Rauschen; A: weniger Rauschen)
- pair03: **A besser als B**
- pair04: **A besser als B**
- pair05: **A besser als B**
- pair06: **A besser als B**
- pair07: **B besser als A**
- pair08: **A besser als B**
- pair09: **A besser als B**
- pair10: **klar A besser als B**
- pair11: **gleich — keiner gewinnt** (A dumpfer, B heller aber mehr Rauschen;
  Wunsch-Richtung Hör-Instanz: *heller UND weniger Rauschen* — wird von keiner Fassung erfüllt)
- pair12: **B sauberer als A**
- pair13: **gleich — keiner gewinnt** (A dumpfer, B heller aber mehr Rauschen;
  identisches Profil wie pair11 → gleiche Wunsch-Richtung: *heller UND weniger Rauschen*)
- pair14: **B sauberer als A**
- pair15: **B sauberer als A**
- pair16: **B sauberer als A**
- pair17: **A sauberer als B**

## Block C — EAR-VAE v1→v2: ENTBLINDET (2026-11-05)

Entblindung über `output_audio/hoer_runde_20261105/blockC_ear_vae/key.json`;
Zuordnung Quell-Key → Version über `output/ear_vae_v1_v2/hoer_stimuli/key.json`
(Quell A = v1, Quell B = v2 — per SHA-256 gegen `roh/*_v1.wav`/`roh/*_v2.wav`
verifiziert: pair-A-Dateien = Quell B = v2 usw., konsistent).

Ergebnis (15 entschiedene Paare, 2× „gleich"):
**v1 = 12 : 3 = v2** — exakter zweiseitiger Vorzeichen-Test **p = 0,035** (n = 15).
* v1 gewinnt: pair01–04, 07–10, 12, 15–17
* v2 gewinnt: pair05, 06, 14
* gleich: pair11, 13

Hör-Befund (Roh-Notizen): **v1 = dumpfer, aber sauberer** (weniger Rauschen);
**v2 = heller, aber rauschiger**. Wunsch-Richtung der Hör-Instanz:
**heller UND weniger Rauschen** — keine der beiden Fassungen erfüllt beides.

§7-Konflikt (Hörordnung): Die Metriken hatten v2 bevorzugt (16/17 Trials
vorteilhaft, Δ musik_mos +0,1016, Δ defektlast −0,1165;
`docs/reports/current/2026-11-03_ear_vae_v1_v2_vergleich.json`); die Hör-Instanz
bevorzugt v1 (12:3, p = 0,035). **Die Hör-Instanz entscheidet** — Metriken sind
Zeugen (§7 hoerordnung.instructions.md).

**Urteil Block C (ENDGÜLTIG BESTÄTIGT, 2026-11-05):**
EAR-VAE v2 = **NO-GO** — Ersatz von v1 wird nicht übernommen; v1 bleibt Produktions-
Fassung. Wunsch-Richtung für künftige Iteration: Helligkeit von v2 bei der
Rausch-Sauberkeit von v1.
