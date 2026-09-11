#!/usr/bin/env python3
"""Hörordnungs-Kalibrierungs-Harness — psychoakustische Invarianten gegen
synthetische Referenz-Signale (Hörordnung §8; Vorstufe echter Panel-Tests).

Prüft die Kalibrierungs-RICHTUNGEN der Hörordnungs-Module mit Signalen, deren
psychoakustische Eigenschaften bekannt sind:

  1. Roughness (Zwicker): AM-Tiefe (70 Hz) ↑ ⇒ asper ↑
  2. Sharpness (Bismarck-Näherung): LPF-Cutoff ↓ ⇒ acum ↓; helles Rauschen > Sinus
  3. Residuum-Masking: lauter Kontext maskiert; stiller Kontext exponiert
  4. Residuum-Monotonie: Click-SNR ↑ ⇒ Salience ↑
  5. Einladungs-Gate: raues AM-Signal > glatter Sinus (max_asper)
  6. Hörstufen-Konsistenz: Natürlichkeit < Wärme < Klarheit < Brillanz
  7. Interaural (Hörordnung §8b): ITD-Rückgewinnung ±35 µs, ITD-Drift-
     Erkennung (100 µs), Mono-Kollaps-Erkennung, BMLD-Monotonie
  8. EC/BMLD (binaural_masking): Freisetzung unkorreliert > korreliert,
     EC-Gain ≥ 0
  9. DLM (temporal_loudness): STL folgt dem Signalpegel
 10. Gammachirp: NSIM identisch > verrauscht
 11. GO/NO-GO: Verdict-Logik deterministisch (GO/NO_GO)
 12. DLM-full/binaural: Pegel-Monotonie, Inhibition unkorr. > diotisch

Ausgabe: Bericht; Exit-Code 1 bei verletzter Invariante (Kalibrierungs-Drift).

Verwendung:  python3 -B scripts/horordnung_calibration.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Repo-Root in den Pfad (Skript lebt in scripts/, backend-Importe brauchen Root)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _noise(sr: int, dur_s: float, rms: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(int(sr * dur_s))
    return (x / (np.sqrt(np.mean(x**2)) + 1e-12) * rms).astype(np.float64)


def _lowpass(x: np.ndarray, sr: int, cutoff_hz: float) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt

    sos = butter(4, cutoff_hz / (sr / 2), btype="low", output="sos")
    return sosfiltfilt(sos, x).astype(np.float64)


def _click_at(audio: np.ndarray, sr: int, t: float, amp: float) -> np.ndarray:
    idx = int(t * sr)
    out = audio.copy()
    n = max(1, int(0.002 * sr))
    out[idx : idx + n] += amp
    return out


def main() -> int:
    sr = 48000
    failures: list[str] = []

    # 1) Roughness: AM-Tiefe ↑ ⇒ asper ↑
    try:
        from backend.core.dsp.zwicker_metrics import compute_roughness_asper

        t = np.arange(sr * 6) / sr
        carrier = np.sin(2 * np.pi * 440 * t)
        am_shallow = (0.3 * (1 + 0.3 * np.sin(2 * np.pi * 70 * t)) * carrier).astype(np.float32)
        am_deep = (0.3 * (1 + 0.9 * np.sin(2 * np.pi * 70 * t)) * carrier).astype(np.float32)
        a1 = float(compute_roughness_asper(am_shallow, sr))
        a2 = float(compute_roughness_asper(am_deep, sr))
        if not (a2 > a1):
            failures.append(f"Roughness-Monotonie: AM-tief {a2:.3f} !> AM-flach {a1:.3f}")
        print(f"1) Roughness-Monotonie: {a1:.3f} → {a2:.3f} asper  {'OK' if a2 > a1 else 'FAIL'}")
    except Exception as exc:  # pragma: no cover
        failures.append(f"Roughness nicht verfügbar: {exc}")

    # 2) Sharpness: LPF ↓ ⇒ acum ↓; Rauschen > Sinus
    try:
        from backend.core.inviting_sound_gate import compute_sharpness_acum

        noise = _noise(sr, 6.0, 0.2, seed=5)
        bright = compute_sharpness_acum(noise, sr)
        dull = compute_sharpness_acum(_lowpass(noise, sr, 1500.0), sr)
        sine = compute_sharpness_acum((0.3 * np.sin(2 * np.pi * 440 * np.arange(sr * 6) / sr)).astype(np.float32), sr)
        if not (dull < bright):
            failures.append(f"Sharpness-Monotonie: LPF {dull:.3f} !< breit {bright:.3f}")
        if not (bright > sine):
            failures.append(f"Sharpness-Diskrimination: Rauschen {bright:.3f} !> Sinus {sine:.3f}")
        print(
            f"2) Sharpness: breit={bright:.3f} dumpf={dull:.3f} sinus={sine:.3f}  {'OK' if dull < bright and bright > sine else 'FAIL'}"
        )
    except Exception as exc:  # pragma: no cover
        failures.append(f"Sharpness nicht verfügbar: {exc}")

    # 3) Residuum-Masking: lauter Kontext maskiert, stiller exponiert
    try:
        from backend.core.residuum_masking import estimate_residuum_salience

        loud_ctx = _click_at(_noise(sr, 6.0, 0.2, seed=6), sr, 3.0, amp=0.05)
        silent_ctx = _click_at(np.zeros(sr * 6, dtype=np.float64), sr, 3.0, amp=0.05)
        s_loud = estimate_residuum_salience(loud_ctx, sr, 2.99, 3.01).salience
        s_silent = estimate_residuum_salience(silent_ctx, sr, 2.99, 3.01).salience
        if not (s_silent >= s_loud):
            failures.append(f"Residuum-Maskierung: still {s_silent:.3f} !>= laut {s_loud:.3f}")
        print(
            f"3) Residuum-Maskierung: lauter Kontext={s_loud:.3f}, still={s_silent:.3f}  {'OK' if s_silent >= s_loud else 'FAIL'}"
        )
    except Exception as exc:  # pragma: no cover
        failures.append(f"Residuum nicht verfügbar: {exc}")

    # 4) Residuum-Monotonie: Click-Amplitude ↑ ⇒ Salience ↑
    try:
        from backend.core.residuum_masking import estimate_residuum_salience as _rs

        base = _noise(sr, 6.0, 0.05, seed=7)  # leiser Kontext: Diskrimination möglich
        q = _rs(_click_at(base, sr, 3.0, amp=0.1), sr, 2.99, 3.01).salience
        l = _rs(_click_at(base, sr, 3.0, amp=0.8), sr, 2.99, 3.01).salience
        if not (l >= q):
            failures.append(f"Residuum-Monotonie: laut {l:.3f} !>= leise {q:.3f}")
        print(f"4) Residuum-Monotonie: {q:.3f} → {l:.3f}  {'OK' if l >= q else 'FAIL'}")
    except Exception as exc:  # pragma: no cover
        failures.append(f"Residuum-Monotonie nicht verfügbar: {exc}")

    # 5) Einladungs-Gate: AM-rau > Sinus glatt (max_asper)
    try:
        from backend.core.inviting_sound_gate import check_inviting_gate

        t6 = np.arange(sr * 12) / sr
        sine12 = (0.3 * np.sin(2 * np.pi * 440 * t6)).astype(np.float32)
        am12 = (0.3 * (1 + 0.9 * np.sin(2 * np.pi * 70 * t6)) * np.sin(2 * np.pi * 440 * t6)).astype(np.float32)
        r_sine = check_inviting_gate(sine12, sr, fatigue_index=0.0)
        r_am = check_inviting_gate(am12, sr, fatigue_index=0.0)
        if not (r_am.max_asper_in_voice > r_sine.max_asper_in_voice):
            failures.append(
                f"Einladungs-Gate-Richtung: AM {r_am.max_asper_in_voice:.3f} !> Sinus {r_sine.max_asper_in_voice:.3f}"
            )
        print(
            f"5) Einladungs-Gate: sinus={r_sine.max_asper_in_voice:.3f}, AM70={r_am.max_asper_in_voice:.3f} asper  "
            f"{'OK' if r_am.max_asper_in_voice > r_sine.max_asper_in_voice else 'FAIL'}"
        )
    except Exception as exc:  # pragma: no cover
        failures.append(f"Einladungs-Gate nicht verfügbar: {exc}")

    # 6) Hörstufen-Konsistenz
    try:
        from backend.core.goal_priority_protocol import GoalPriorityProtocol

        gpp = GoalPriorityProtocol()
        order_ok = (
            gpp.hearing_tier("natuerlichkeit")
            < gpp.hearing_tier("waerme")
            < gpp.hearing_tier("transparenz")
            < gpp.hearing_tier("brillanz")
        )
        if not order_ok:
            failures.append("Hörstufen-Konsistenz verletzt")
        print(f"6) Hörstufen-Konsistenz: Natürlichkeit<Wärme<Klarheit<Brillanz  {'OK' if order_ok else 'FAIL'}")
    except Exception as exc:  # pragma: no cover
        failures.append(f"GoalPriorityProtocol nicht verfügbar: {exc}")

    # 7) Interaurale Invarianten (§HRTF/interaural, Hörordnung §8b)
    try:
        from backend.core.audio_layout import mono_mix as _cal_mono
        from backend.core.dsp.interaural_cues import (
            bmld_advantage_db as _cal_bmld,
        )
        from backend.core.dsp.interaural_cues import (
            compute_itd_us as _cal_itd,
        )
        from backend.core.dsp.interaural_cues import (
            interaural_cue_integrity as _cal_ici,
        )

        _rng7 = np.random.default_rng(11)
        _l7 = _rng7.standard_normal(sr).astype(np.float32)
        _r7 = np.roll(_l7, int(round(200e-6 * sr)))
        _st7 = np.stack([_l7, _r7], axis=0)
        _itd7 = float(_cal_itd(_st7, sr))
        if not (-235.0 < _itd7 < -165.0):
            failures.append(f"ITD-Rückgewinnung: {_itd7:.1f} µs außerhalb ±200±35 µs")
        _corr7 = np.stack([np.roll(_l7, int(round(300e-6 * sr))), _r7], axis=0)
        _drift7 = _cal_ici(_st7, _corr7, sr)
        if _drift7.itd_ok:
            failures.append("ITD-Drift-Erkennung: 100 µs-Versatz nicht erkannt")
        _mono7 = _cal_mono(_st7)
        _col7 = _cal_ici(_st7, _mono7, sr)
        if not (_col7.itd_drift_us >= 1e6 and not _col7.itd_ok and not _col7.ild_ok):
            failures.append("Mono-Kollaps-Erkennung: maximaler Befund fehlt")
        if not (_cal_bmld(0.2) > _cal_bmld(0.98)):
            failures.append("BMLD-Monotonie verletzt")
        print(
            f"7) Interaural: ITD={_itd7:.1f} µs, Drift={_drift7.itd_drift_us:.1f} µs, "
            f"Kollaps={_col7.itd_drift_us:.0f} µs  {'OK' if not any('ITD' in f or 'BMLD' in f or 'Mono' in f for f in failures) else 'FAIL'}"
        )
    except Exception as exc:  # pragma: no cover
        failures.append(f"Interaural nicht verfügbar: {exc}")

    # 8) BMLD/EC-Invarianten (binaural_masking.py): korreliert < unkorreliert,
    #    EC-Gain ≥ 0, Mono → 0 dB.
    try:
        from backend.core.dsp.binaural_masking import binaural_masking_advantage as _cal_bm

        _rng8 = np.random.default_rng(13)
        _l8 = _rng8.standard_normal(sr).astype(np.float32)
        _corr8 = np.stack([_l8, _l8], axis=0) / (np.max(np.abs(_l8)) + 1e-9)
        _uncorr8 = np.stack([_l8, _rng8.standard_normal(sr).astype(np.float32)], axis=0) / (np.max(np.abs(_l8)) + 1e-9)
        _rel_c = _cal_bm(_corr8, sr).release_db
        _rel_u = _cal_bm(_uncorr8, sr).release_db
        if not (_rel_u > _rel_c + 1.0):
            failures.append(f"BMLD-Monotonie verletzt (korr={_rel_c:.1f}, unkorr={_rel_u:.1f} dB)")
        if _cal_bm(_corr8, sr).ec_gain_db < 0.0:
            failures.append("EC-Gain negativ")
        print(
            f"8) EC/BMLD: Freisetzung korr={_rel_c:.1f} dB, unkorr={_rel_u:.1f} dB  {'OK' if not any('BMLD' in f or 'EC' in f for f in failures) else 'FAIL'}"
        )
    except Exception as exc:  # pragma: no cover
        failures.append(f"EC/BMLD nicht verfügbar: {exc}")

    # 9) DLM-Invariante (temporal_loudness.py): STL folgt dem Signalpegel.
    try:
        from backend.core.dsp.temporal_loudness import temporal_loudness as _cal_tl

        _t9 = np.arange(sr) / sr
        _x9 = np.concatenate(
            [0.3 * np.sin(2 * np.pi * 440 * _t9[: sr // 2]), 0.02 * np.sin(2 * np.pi * 440 * _t9[sr // 2 :])]
        ).astype(np.float32)
        _stl9 = _cal_tl(_x9, sr).stl_sone
        if not (np.mean(_stl9[: sr // 4]) > np.mean(_stl9[3 * sr // 4 :])):
            failures.append("STL folgt nicht dem Pegelverlauf")
        print(f"9) DLM: STL folgt Pegelverlauf  {'OK' if not any('STL' in f for f in failures) else 'FAIL'}")
    except Exception as exc:  # pragma: no cover
        failures.append(f"DLM nicht verfügbar: {exc}")

    # 10) Gammachirp-Invariante (gammachirp_filterbank.py): tonal > Rauschen + dcGC-Monotonie.
    try:
        from backend.core.dsp.gammachirp_filterbank import gammachirp_impulse_response as _gir
        from backend.core.dsp.gammachirp_filterbank import nsim_gammachirp as _cal_gc

        _rng10 = np.random.default_rng(17)
        _t10 = np.arange(sr // 5) / sr
        _ref10 = (
            0.3 * np.sin(2 * np.pi * 440 * _t10)
            + 0.2 * np.sin(2 * np.pi * 880 * _t10)
            + 0.1 * np.sin(2 * np.pi * 1320 * _t10)
        ).astype(np.float32)
        _noise10 = _rng10.standard_normal(sr // 5).astype(np.float32)
        _sim_id = _cal_gc(_ref10, _ref10.copy(), sr)
        _sim_noise = _cal_gc(_ref10, _noise10, sr)
        if not (_sim_id > 0.95 and _sim_noise < 0.6):
            failures.append(f"Gammachirp-NSIM-Diskrimination verletzt (id={_sim_id:.3f}, noise={_sim_noise:.3f})")
        # dcGC-Pegelabhängigkeit: lautes Filter ≥ leises Filter (IF-Spread, monoton).
        from scipy.signal import hilbert as _cal_hilb

        def _if_spread(h, _sr):
            _a = np.abs(h)
            _env = np.convolve(_a, np.ones(32) / 32, "same")
            _keep = _env > 0.01 * _env.max()
            _z = np.zeros_like(h, dtype=np.float64)
            _z[_keep] = h[_keep]
            _ph = np.unwrap(np.angle(_cal_hilb(_z)))
            _f = np.diff(_ph) * _sr / (2 * np.pi)
            return float(np.percentile(_f, 90) - np.percentile(_f, 10))

        _s_quiet = _if_spread(_gir(2000, sr, level_db=-60.0), sr)
        _s_loud = _if_spread(_gir(2000, sr, level_db=0.0), sr)
        if not (_s_loud >= _s_quiet):
            failures.append(f"dcGC-Pegelmonotonie verletzt (quiet={_s_quiet:.0f}, loud={_s_loud:.0f} Hz)")
        print(
            f"10) Gammachirp: NSIM id={_sim_id:.3f} >> noise={_sim_noise:.3f}, dcGC {_s_quiet:.0f}→{_s_loud:.0f} Hz  {'OK' if not any('Gammachirp' in f or 'dcGC' in f for f in failures) else 'FAIL'}"
        )
    except Exception as exc:  # pragma: no cover
        failures.append(f"Gammachirp nicht verfügbar: {exc}")

    # 11) GO/NO-GO-Invariante (go_nogo_export_gate.py): Verdict-Logik deterministisch.
    try:
        from backend.core.go_nogo_export_gate import evaluate as _cal_gn

        class _R:
            metadata = {}
            warnings = []
            quality_estimate = 0.8

        if _cal_gn(_R()).verdict != "GO":
            failures.append("GO-Verdict für sauberes Ergebnis fehlt")
        _bad = _R()
        _bad.quality_estimate = 0.3
        if _cal_gn(_bad).verdict != "NO_GO":
            failures.append("NO_GO-Verdict für quality_estimate=0.3 fehlt")
        print(
            f"11) GO/NO-GO: Verdict-Logik deterministisch  {'OK' if not any('GO-' in f or 'Verdict' in f for f in failures) else 'FAIL'}"
        )
    except Exception as exc:  # pragma: no cover
        failures.append(f"GO/NO-GO nicht verfügbar: {exc}")

    # 12) DLM-full + binaurale Loudness (dynamic_loudness_model.py):
    #     Pegel-Monotonie, Inhibition-Richtung (unkorrel. > diotisch).
    try:
        from backend.core.dsp.dynamic_loudness_model import binaural_loudness as _cal_bin
        from backend.core.dsp.dynamic_loudness_model import dynamic_loudness as _cal_dlm

        _rng12 = np.random.default_rng(23)
        _t12 = np.arange(sr) / sr
        _sine_q = (0.01 * np.sin(2 * np.pi * 1000 * _t12)).astype(np.float32)
        _sine_l = (0.1 * np.sin(2 * np.pi * 1000 * _t12)).astype(np.float32)
        _n_q = float(_cal_dlm(_sine_q, sr).stl_sone)
        _n_l = float(_cal_dlm(_sine_l, sr).stl_sone)
        if not (_n_l > _n_q):
            failures.append(f"DLM-Pegelmonotonie verletzt (quiet={_n_q:.2f}, loud={_n_l:.2f})")
        _dio = _cal_bin(np.stack([_sine_q, _sine_q], axis=1).astype(np.float32), sr)
        _unc = _cal_bin(
            np.stack(
                [_rng12.standard_normal(sr) * 0.01, _rng12.standard_normal(sr) * 0.01],
                axis=1,
            ).astype(np.float32),
            sr,
        )
        if not (_unc.stl_sone > _dio.stl_sone):
            failures.append(f"Binaurale Inhibition verletzt (diotisch={_dio.stl_sone:.2f}, unkorr={_unc.stl_sone:.2f})")
        print(
            f"12) DLM/binaural: N(leise)={_n_q:.2f} < N(laut)={_n_l:.2f}, "
            f"Adv. diot={_dio.binaural_advantage_db:.1f} dB < unkorr={_unc.binaural_advantage_db:.1f} dB  "
            f"{'OK' if not any('DLM' in f or 'Binaurale' in f for f in failures) else 'FAIL'}"
        )
    except Exception as exc:  # pragma: no cover
        failures.append(f"DLM/binaural nicht verfügbar: {exc}")

    print()
    if failures:
        print(f"KALIBRIERUNG: {len(failures)} INVARIANTE(N) VERLETZT")
        for f in failures:
            print(f"  ❌ {f}")
        return 1
    print("KALIBRIERUNG: alle Invarianten erfüllt — Schwellwerte konsistent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
