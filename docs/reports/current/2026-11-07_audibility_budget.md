# Audibility-Budget — Verdrahtung A-1 (2026-11-07)

**Status:** implementiert, **Flag AUS** (bit-identisches Bestandsverhalten),
Wirkungsmessung im trainingsfreien Fenster offen. Register: **D-K3-71**.
Vorgänger-Plan: `docs/SOTA_HANDOVER.md` §2b Muster A.

## 1. Ziel und normative Basis

Hörordnung §4 (`.github/instructions/hoerordnung.instructions.md`): Ein Defekt
gilt als behoben, sobald sein Rest **unter** der psychoakustischen
Maskierungsschwelle liegt — nicht wenn sein Messwert Null ist. Für jede
subtraktive Stufe folgt daraus eine Obergrenze der Bearbeitungstiefe:

```
budget_db[b,t] = max(0, L_defekt[b,t] − (T_signal[b,t] − marge_db))
```

* `L_defekt` — Pegel des Anteils, den die Stufe entfernen will
  (Rauschschätzung in Phase 03, Sibilantenband in Phase 43).
* `T_signal` — Maskierungsschwelle des **verbleibenden** Materials im selben
  Bark-Band/Frame (kanonisches Johnston-Modell aus
  `backend/core/dsp/masking_model.py`, +6 dB Sicherheitsmarge).
* `budget_db` ist die einzige Rechtfertigung für Dämpfung. Tiefer als
  `−budget_db` zu dämpfen entfernt Material, ohne einen hörbaren Defekt zu
  beseitigen (Überunterdrückung, Lispeln, „tote Stille").

Der Gain-Boden ist `10^(−budget_db/20)`; er **hebt nur an** und bleibt ≤ 1,0.

## 2. Abgrenzung (kein Duplikat, §G9 copilot-instructions.md)

| Modul | Rolle |
|---|---|
| `backend/core/dsp/audibility_gate.py` | entscheidet **vor** der Reparatur („Defekt hörbar?") |
| `backend/core/dsp/audibility_targets.py` | nimmt **nach** der Reparatur ab („Rest unter Ziel?") |
| `backend/core/dsp/audibility_budget.py` | liefert die **Stellgröße** für die Reparatur selbst |

Phase 19 bleibt bewusst außen vor: sie hat bereits
`apply_psychoacoustic_masking_clamp` (§4.5) und das PSY-A1-Audibility-Gate.
Phase 43 (ML-De-Esser) hatte **keine** Maskierungs-Abdeckung — dort ist der
größte Hebel; Phase 03 hat nur einen zeitgemittelten statischen Masken-Boden
(§2.62), keinen rauschbewussten Pro-Frame-Budget.

## 3. Umsetzung

* **Neu:** `backend/core/dsp/audibility_budget.py`
  * `masking_threshold_db` — vektorisierter Johnston-Kern, **bit-identisch**
    zu `masking_model.compute_masking_threshold_db` (gemessener Diff 0,0).
  * `audibility_budget_db`, `gain_floor_from_budget`, `apply_gain_budget`
    (Band-Schleife, speicherarm), `spectrum_budget_db` (STFT, chunked,
    Masker = spektrale Subtraktion `max(|Y|²−|N|²,0)`),
    `band_budget_curve` (Zeitdomäne, Defektband im Masker ausgenullt),
    `audibility_budget_enabled` (fail-safe Flag-Lesen, §V6).
  * Deterministisch (§G5), NaN-sicher, keine Zeit-/Zufallsabhängigkeit.
* **Flag:** `backend/core/music_model_flags.py::use_audibility_budget = False`.
* **Pilot 1 — Phase 03** (`phase_03_denoise.py::_denoise_mono_mrsa`): je Zone
  finale Untergrenze der Zonen-Gains nach `preserve_transients`; Zähler in
  `metadata["audibility_budget"]` (`applied`, `zones`, `max_budget_db`,
  `mean_bound_fraction`, `mean_lift_db`).
* **Pilot 2 — Phase 43** (`phase_43_ml_deesser.py::_deess_channel`): Boden auf
  `gr_smooth`, nach Strength-Cap; Sibilantenband aus dem Masker ausgenommen.
* **Mess-Harness:** `scripts/b3_stufenmessung.py` kennt die Stufe `A-1`.

## 4. Verifikation (Ist-Stand, 2026-11-07)

| Prüfung | Ergebnis |
|---|---|
| `tests/unit/test_audibility_budget.py` | **22/22 grün** |
| `tests/unit/test_audibility_budget_wiring.py` | **8/8 grün** |
| Johnston-Kern vs. `compute_masking_threshold_db` | **Diff 0,0** (bit-identisch) |
| Flag AUS: Budget wird nicht konsultiert | Quell-Spy + deterministischer Doppellauf |
| Flag AUS: Phase-03-Metadaten | `{"applied": False}` |
| Flag AN: Phase 03 | `applied: True`, ≥ 1 Zone, finit, `|x| ≤ 1` |
| Phase 43 Boden 1,0 | transparent (< 1e-6 Rundung), GR = 0 dB |
| Phase 43 Boden 0,5 | Sibilant-RMS höher, GR weniger negativ als ohne Budget |
| Angrenzende Bestandssuiten | 64 passed (Phase 03/43, B3-Komposition, Primary-Paths) |
| `ruff check` / `ruff format --check` | sauber |

Reproduktion:

```bash
python3 -m pytest tests/unit/test_audibility_budget.py \
                 tests/unit/test_audibility_budget_wiring.py -q
ruff check backend/core/dsp/audibility_budget.py
```

## 5. Bewusst NICHT getan

* **Keine Flag-Aktivierung.** `use_audibility_budget` bleibt AUS; eine
  Aktivierung wäre nach §2b A ein B3-/Hör-Urteil und ist ohne Messung +
  Sign-off unzulässig (`scripts/flag_freigabe.py` fail-closed).
* **Keine B3-Messung jetzt.** Die GPU-Kette 2/4 läuft (D3-2 BigVGAN,
  Versuch 1/2); laut Betriebsregel keine parallelen Trainings und keine
  Voll-Suiten. Die Messung braucht das trainingsfreie Fenster.
* **Keine Phase-19-Verdrahtung.** Dort existiert bereits ein post-hoc
  Maskierungs-Clamp + PSY-A1; ein zweiter Mechanismus wäre Duplikat.

## 6. Nächste Schritte (im trainingsfreien Fenster)

1. `nice -n 15 python3 -B scripts/b3_stufenmessung.py --stages A-1 --clips 3`
   — Wirkung je Vertrags-Referenz-Song, never-worsen-Schranke.
2. RT-Kosten je Zone messen (`audibility_budget`-Zähler + Phasen-Zeit).
3. Blindes A/B (Flag AUS/AN) auf ≥ 3 echten Songs, `musik_mos` + `defektlast`
   + Hör-Session; W-1/W-4/W-5 bleiben menschliche Entscheidungen.
4. Nur bei belegtem Vorteil: `python3 scripts/flag_freigabe.py
   --objektivbeschluss ...` (fail-closed) — sonst Flag AUS lassen und D-K3-71
   als „bewusst-akzeptiert" schließen.

## 7. Einordnung in die Rest-Lücken

§2b A war der einzige **autonom abarbeitbare, spezifizierte** Hebel ohne
GPU-/Hör-Bindung. Verbleibend menschlich blockiert: §2 Zeile 6
(W-1/W-4/W-5 Hör-Entscheidungen; W-2/W-3 abgelehnt), Zeile 8
(Hör-/Lizenz-Entscheidungen), §2b D (MUSHRA n ≥ 30). GPU-gebunden: §2 Zeile 7
und 11 (Ketten 1–4).
