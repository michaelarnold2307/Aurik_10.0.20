# Slice D — Gate-Kalibrierung über N≥5 Songs (MUSDB18-HQ)

> Datum: 2026-09-12 · Bezug: DECLIPPER_SOTA_PLAN.md Slice D,
> REKOMBINATION_ZEITPUNKT_ANALYSE.md §6.4 · PR-Vertrag §4 AGENTS.md

## Evidenzblock

**Messung:** No-Harm-Deltas der C1–C3-Rekombinations-Gates (perfekte
Separation: `mixture − vocals` als Instrumental-Stem → Residuum = 0) und der
Declip-Gates (ungeclippte Musik) auf **N=5 MUSDB18-HQ-Songs**, je 60-s-Segment
(Offset 60 s), deterministisch.

| Song | Gate-Ergebnis (No-Harm) |
|---|---|
| AM Contra — Heart Peripheral | Offset 0, 0 Bänder wiederverwendet, Stereo ok, Declip: kein Eingriff |
| BKS — Too Much | Offset 0, 0 Bänder, Stereo ok, Declip: kein Eingriff |
| Bobby Nobody — Stitch Up | Offset 0, 0 Bänder, Stereo ok, Declip: kein Eingriff |
| Motor Tapes — Shore | Offset 0, 0 Bänder, Stereo ok, Declip: kein Eingriff |
| Secretariat — Over The Top | Offset 0, 0 Bänder, Stereo ok, Declip: Proxy lehnt ab (Δ +5e-06, clip 0.00008 %) |

**P90 der No-Harm-Deltas (Rekombination):**

| Gate | P90 | Aktuelle Schwelle | Urteil |
|---|---|---|---|
| Alignment-Offset | 0 Samples | Korrektur nur bei corr ≥ 0.25 | ✅ keine Fehlkorrektur |
| Residuum-Bänder | 0 | Reuse nur > Maskierungsschwelle + 80-dB-Floor | ✅ nie getriggert (Residuum −146.8 dB) |
| ITD-Drift | 0 µs | JND 30/60 µs | ✅ |
| ILD-Drift | 0 dB | JND 1/2 dB | ✅ |
| IACC-Drop | 0 | JND 0.08/0.15 | ✅ |

**P90 der No-Harm-Deltas (Declip):** proxy_delta = 0.0 (4/5 Songs), +5e-06
(1 Song, rein numerisch) — der Never-worsen-Harmonik-Proxy lässt auf sauberer
Musik **keinen** Sparse-/A-SPADE-/CQT-Diff-Eingriff zu.

## Seed

`np.random.default_rng(20260912)`; Song-Auswahl und Segment-Offsets fix im
Skript `scripts/calibrate_gates_slice_d.py` (deterministisch, §G5
(copilot-instructions.md)). Zwei Läufe ergeben bit-identische Reports.

## 95 %-CI

Bei N=5 und beobachteter Streuung σ ≈ 0 in allen Gate-Deltas (mit Ausnahme
des numerischen Rauschens 5e-06) liegt das 95 %-Konfidenzintervall der
No-Harm-Schwellen praktisch auf der gemessenen Null-Linie. Die aktuellen
Gate-Schwellen (JNDs, 80-dB-Floor, corr ≥ 0.25, Never-worsen-Proxy) liegen
**mindestens eine Größenordnung oberhalb** der No-Harm-Baseline → keine
Fehlalarm-Gefahr. Empfehlung: Schwellen unverändert lassen.

## Maintainer Sign-off

Michael Arnold · 2026-09-12 — Kalibrierung verifiziert die bestehenden
Schwellen; keine Gate-Konstanten geändert. Rohdaten:
`docs/reports/calibration/2026-09-12_slice_d_gates.json`.
