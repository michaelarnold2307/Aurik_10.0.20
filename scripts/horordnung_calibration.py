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
 13. MG-ERB-Spreizung: Aufwärts-Masking > Abwärts
 14. MG-ERB-Spreizung: Monotonie — Schwelle fällt beidseitig vom Masker
 15. Fluktuationsstärke (vacil): AM-Tiefe (4 Hz) ↑ ⇒ vacil ↑; unmoduliert < moduliert
 16. Tonality-Gate: Sinus tonal (flatness klein), weißes Rauschen nicht tonal
 17. Forward-Masking: Klick öffnet Boost-Zone direkt danach; ohne Klick keine
 18. Lautheits-Sprung (phon): +12-dB-Stufe zwischen Fenstern >> konstantes Signal
 19. Temporal-Masking in residual_audibility: Zone hebt effektive Schwelle; ohne Klick keine
 20. SII-Proxy (Hörordnung §3, L1): Tiefpass senkt SII deutlich; identisch ⇒ Delta 0
 21. BMLD-Bonus in residual_audibility (L3): diotisch ≈ 6 dB, inkohärent ≪ — Bonus ≤ 6

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
        _low_proxy = _cal_gn(_bad)
        _go_nogo_failed = _low_proxy.verdict != "GO_CAUTION" or bool(_low_proxy.reasons)
        if _go_nogo_failed:
            failures.append("Niedriger quality_estimate-Proxy muss report-only bleiben")
        print(f"11) GO/NO-GO: Verdict-Logik deterministisch  {'FAIL' if _go_nogo_failed else 'OK'}")
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

    # 13) MG-ERB-Spreizung (§2.69d, residuum_masking): Aufwärts-Masking > Abwärts.
    #     Bei realistischem Masker-Pegel (L=20 rel. dB) ist der Abwärts-Hang
    #     (24+0.23·fc+0.2·L) steiler als der Aufwärts-Hang (27 dB/ERB).
    try:
        from backend.core import residuum_masking as _cal_rm

        _masker_db = np.full(len(_cal_rm._BARK_CENTERS), -120.0, dtype=np.float64)
        _low_band = _cal_rm.bark_band_index_of_freq(700.0)
        _high_band = _cal_rm.bark_band_index_of_freq(1000.0)
        _low_masker = _masker_db.copy()
        _low_masker[_low_band] = 20.0
        _high_masker = _masker_db.copy()
        _high_masker[_high_band] = 20.0
        _up = float(_cal_rm._spread_mask_threshold(_low_masker)[_high_band])
        _down = float(_cal_rm._spread_mask_threshold(_high_masker)[_low_band])
        if not (_up > _down):
            failures.append(f"MG-ERB-Asymmetrie verletzt: aufwärts {_up:.1f} !> abwärts {_down:.1f} dB")
        print(
            f"13) MG-ERB-Asymmetrie: aufwärts={_up:.1f} dB > abwärts={_down:.1f} dB  {'OK' if _up > _down else 'FAIL'}"
        )
    except Exception as exc:  # pragma: no cover
        failures.append(f"MG-ERB-Asymmetrie nicht verfügbar: {exc}")

    # 14) MG-ERB-Spreizung: Monotonie — Schwelle fällt beidseitig vom Masker.
    try:
        from backend.core import residuum_masking as _cal_rm2

        _masker_db2 = np.full(len(_cal_rm2._BARK_CENTERS), -120.0, dtype=np.float64)
        _mid = _cal_rm2.bark_band_index_of_freq(1000.0)
        _masker_db2[_mid] = 20.0
        _thr2 = _cal_rm2._spread_mask_threshold(_masker_db2)
        _mono_ok = True
        for _b in range(_mid + 1, len(_thr2)):
            if float(_thr2[_b]) > float(_thr2[_b - 1]) + 1e-9:
                _mono_ok = False
        for _b in range(_mid - 1, -1, -1):
            if float(_thr2[_b]) > float(_thr2[_b + 1]) + 1e-9:
                _mono_ok = False
        if not _mono_ok:
            failures.append("MG-ERB-Monotonie verletzt: Schwelle nicht monoton fallend")
        print(f"14) MG-ERB-Monotonie: Schwelle fällt beidseitig vom Masker  {'OK' if _mono_ok else 'FAIL'}")
    except Exception as exc:  # pragma: no cover
        failures.append(f"MG-ERB-Monotonie nicht verfügbar: {exc}")

    # 15) Fluktuationsstärke (Hörordnung §6, Einladungs-Gate): Rausch-Pumpen
    #     (4 Hz AM-Tiefe) aufwärts ⇒ vacil aufwärts; ungepumpt < gepumpt.
    #     Deterministische Ton-AM sättigt die Welch-PSD (ein Bin) — die
    #     realistische Pumpen-Simulation auf Rauschhintergrund löst auf.
    try:
        from backend.core.dsp.zwicker_metrics import compute_fluctuation_strength_vacil

        _rng15 = np.random.default_rng(7)
        _noise15 = (_rng15.standard_normal(sr * 30) * 0.05).astype(np.float32)
        _t15 = np.arange(sr * 30) / sr
        _pump_shallow = (_noise15 * (1.0 + 0.15 * np.sin(2 * np.pi * 4.0 * _t15)).astype(np.float32)).astype(np.float32)
        _pump_deep = (_noise15 * (1.0 + 0.9 * np.sin(2 * np.pi * 4.0 * _t15)).astype(np.float32)).astype(np.float32)
        v0 = float(compute_fluctuation_strength_vacil(_noise15, sr))
        v1 = float(compute_fluctuation_strength_vacil(_pump_shallow, sr))
        v2 = float(compute_fluctuation_strength_vacil(_pump_deep, sr))
        if not (v2 > v1 > v0):
            failures.append(f"Fluktuations-Monotonie: tief {v2:.4f} !> flach {v1:.4f} !> ungepumpt {v0:.4f} vacil")
        if not (v1 > 0.25 > v0):
            failures.append(f"Fluktuations-Schwelle: gepumpt {v1:.4f} !> 0.25 !> ungepumpt {v0:.4f} vacil")
        print(
            f"15) Fluktuations-Pumpen: ungepumpt={v0:.4f} 15%={v1:.4f} 90%={v2:.4f} vacil  {'OK' if v2 > v1 > v0 and v1 > 0.25 > v0 else 'FAIL'}"
        )
    except Exception as exc:  # pragma: no cover
        failures.append(f"Fluktuationsstärke nicht verfügbar: {exc}")

    # 16) Tonality-Gate (§23-TONALITY): reiner Sinus ist tonal (flatness klein),
    #     weißes Rauschen ist nicht tonal (flatness hoch).
    try:
        from backend.core.dsp.tonality_gate import is_tonal_clean, spectral_flatness

        sine_ton = (0.3 * np.sin(2 * np.pi * 440 * np.arange(sr * 6) / sr)).astype(np.float64)
        noise_ton = _noise(sr, 6.0, 0.2, seed=42)
        flat_sine = float(spectral_flatness(sine_ton, sr))
        flat_noise = float(spectral_flatness(noise_ton, sr))
        clean_sine = bool(is_tonal_clean(sine_ton, sr))
        clean_noise = bool(is_tonal_clean(noise_ton, sr))
        if not (flat_sine < flat_noise):
            failures.append(f"Tonality-Diskrimination: flatness Sinus {flat_sine:.4f} !< Rauschen {flat_noise:.4f}")
        if not (clean_sine and not clean_noise):
            failures.append(f"Tonality-Gate falsch: tonal(Sinus)={clean_sine}, tonal(Rauschen)={clean_noise}")
        print(
            f"16) Tonality-Gate: flat_sinus={flat_sine:.4f} flat_rauschen={flat_noise:.4f} sinus_tonal={clean_sine} rauschen_tonal={clean_noise}  {'OK' if flat_sine < flat_noise and clean_sine and not clean_noise else 'FAIL'}"
        )
    except Exception as exc:  # pragma: no cover
        failures.append(f"Tonality-Gate nicht verfügbar: {exc}")

    # 17) Forward-Masking (Hörordnung §8, Ebene 2): ein Klick öffnet direkt danach
    #     eine Boost-Zone; dasselbe Signal ohne Klick öffnet keine.
    try:
        from backend.core.dsp.temporal_masking import get_forward_masking_guard

        _guard_fm = get_forward_masking_guard()
        _base_fm = _noise(sr, 6.0, 0.02, seed=7)
        _with_click = _click_at(_base_fm, sr, 3.0, amp=0.8)
        _zones_click = _guard_fm.compute_zones(_with_click.astype(np.float32), sr)
        _zones_plain = _guard_fm.compute_zones(_base_fm.astype(np.float32), sr)
        _boost_after_click = any(
            _z.start_sample <= int(sr * 3.05) < _z.end_sample and _z.max_nr_strength_boost > 0.0 for _z in _zones_click
        )
        _plain_boost = any(_z.max_nr_strength_boost > 0.0 for _z in _zones_plain)
        if not (_boost_after_click and not _plain_boost):
            failures.append(
                f"Forward-Masking verletzt: Boost nach Klick={_boost_after_click} (Zonen {len(_zones_click)}), Boost ohne Klick={_plain_boost} (Zonen {len(_zones_plain)})"
            )
        print(
            f"17) Forward-Masking: Zonen_nach_Klick={len(_zones_click)} mit_Boost_50ms={_boost_after_click}, Zonen_ohne_Klick={len(_zones_plain)}  {'OK' if _boost_after_click and not _plain_boost else 'FAIL'}"
        )
    except Exception as exc:  # pragma: no cover
        failures.append(f"Forward-Masking nicht verfügbar: {exc}")

    # 18) Lautheits-Sprung-Diskrimination (Hörordnung §6, Einladungs-Gate):
    #     ein Pegelsprung zwischen Fenstern ⇒ Lautheits-Sprung >> konstant.
    #     Zwei 6-s-Hälften mit 12 dB Pegeldifferenz (Fenster 5 s/Hop 2.5 s ⇒
    #     Sprung liegt zwischen benachbarten Fenstern).
    try:
        from backend.core.dsp.zwicker_loudness import compute_loudness_phon

        _rng18 = np.random.default_rng(7)
        _quiet18 = (_rng18.standard_normal(sr * 6) * 0.01).astype(np.float32)
        _loud18 = (_rng18.standard_normal(sr * 6) * 0.04).astype(np.float32)  # +12 dB
        _const18 = (_rng18.standard_normal(sr * 12) * 0.02).astype(np.float32)
        _stepped18 = np.concatenate([_quiet18, _loud18]).astype(np.float32)
        _p_const = [
            float(compute_loudness_phon(_const18[: sr * 5], sr)),
            float(compute_loudness_phon(_const18[sr * 5 : sr * 10], sr)),
        ]
        _p_step = [
            float(compute_loudness_phon(_stepped18[: sr * 5], sr)),
            float(compute_loudness_phon(_stepped18[sr * 5 : sr * 10], sr)),
        ]
        _jump_const = abs(_p_const[1] - _p_const[0])
        _jump_step = abs(_p_step[1] - _p_step[0])
        if not (_jump_step > _jump_const + 3.0):
            failures.append(
                f"Lautheits-Sprung-Diskrimination: gestuft {_jump_step:.2f} phon !> konstant {_jump_const:.2f} + 3 phon"
            )
        print(
            f"18) Lautheits-Sprung: konstant={_jump_const:.2f} phon, gestuft(+12dB)={_jump_step:.2f} phon  {'OK' if _jump_step > _jump_const + 3.0 else 'FAIL'}"
        )
    except Exception as exc:  # pragma: no cover
        failures.append(f"Lautheits-Sprung-Diskrimination nicht verfügbar: {exc}")

    # 19) Temporal-Masking in residual_audibility (Hörordnung §4): ein Impuls in
    #     einer Forward-Masking-Zone hebt die effektive Schwelle; ohne Klick keine
    #     Zone und keine Anhebung (zeitlich UND spektral lokal).
    try:
        from backend.core.dsp.audibility_targets import residual_audibility

        _rng19 = np.random.default_rng(7)
        _ctx19 = (_rng19.standard_normal(sr * 6) * 0.01).astype(np.float64)
        _ctx19[int(sr * 3.0) : int(sr * 3.0) + 96] += 0.9  # 2-ms-Klick
        _res19 = np.zeros(sr * 6)
        _res19[int(sr * 3.05) : int(sr * 3.05) + 480] = 0.05 * _rng19.standard_normal(480)
        _rep19 = residual_audibility(_res19, _ctx19, sr)
        _ctx19_plain = (_rng19.standard_normal(sr * 6) * 0.01).astype(np.float64)
        _rep19_plain = residual_audibility(_res19, _ctx19_plain, sr)
        if not (_rep19["in_forward_mask_zone"] and _rep19["temporal_boost_db"] > 0.0):
            failures.append(
                f"Temporal-Masking-Zone: in_zone={_rep19['in_forward_mask_zone']}, boost={_rep19['temporal_boost_db']:.2f} dB"
            )
        if not (_rep19["threshold_db_eff"] > _rep19["threshold_db"]):
            failures.append(
                f"Temporal-Masking-Schwelle: eff={_rep19['threshold_db_eff']:.2f} !> basis={_rep19['threshold_db']:.2f} dB"
            )
        if _rep19_plain["in_forward_mask_zone"] or _rep19_plain["temporal_boost_db"] != 0.0:
            failures.append(
                f"Temporal-Masking-Fehlalarm: in_zone={_rep19_plain['in_forward_mask_zone']}, boost={_rep19_plain['temporal_boost_db']:.2f} dB ohne Klick"
            )
        print(
            f"19) Temporal-Masking: boost={_rep19['temporal_boost_db']:.2f} dB (eff={_rep19['threshold_db_eff']:.1f} vs. basis={_rep19['threshold_db']:.1f}), ohne Klick boost={_rep19_plain['temporal_boost_db']:.2f}  {'OK' if _rep19['in_forward_mask_zone'] and _rep19['temporal_boost_db'] > 0.0 and _rep19['threshold_db_eff'] > _rep19['threshold_db'] and not _rep19_plain['in_forward_mask_zone'] else 'FAIL'}"
        )
    except Exception as exc:  # pragma: no cover
        failures.append(f"Temporal-Masking-Integration nicht verfügbar: {exc}")

    # 20) SII-Proxy (Hörordnung §3, L1 aus dem Spec-SOTA-Audit 2026-11-09):
    #     Bandbegrenzung (Tiefpass 2 kHz) senkt die verständlichkeitsrelevante
    #     Band-Energie deutlich; identisches Signal ⇒ Delta 0.
    try:
        from backend.core.dsp.speech_intelligibility import sii_delta, sii_index

        _rng20 = np.random.default_rng(42)
        _n20 = _rng20.standard_normal(sr * 8)
        _sp20 = np.fft.rfft(_n20)
        _f20 = np.fft.rfftfreq(sr * 8, 1 / sr)
        _sp20[_f20 > 8000] = 0
        _full20 = (np.fft.irfft(_sp20, n=sr * 8) / 3).astype(np.float32)
        _sp20b = _sp20.copy()
        _sp20b[_f20 > 2000] *= 0.02
        _tp20 = (np.fft.irfft(_sp20b, n=sr * 8) / 3).astype(np.float32)
        _s_full = float(sii_index(_full20, sr))
        _s_tp = float(sii_index(_tp20, sr))
        _d20 = sii_delta(_full20, _tp20, sr)
        _d_id = sii_delta(_full20, _full20, sr)
        if not (_s_full > _s_tp + 0.15):
            failures.append(f"SII-Richtung: voll {_s_full:.3f} !> tiefpass {_s_tp:.3f} + 0.15")
        if not (_d20["verschlechterung"] and not _d_id["verschlechterung"]):
            failures.append(
                f"SII-Delta: verschlechterung={_d20['verschlechterung']} (erwartet True), identisch={_d_id['verschlechterung']} (erwartet False)"
            )
        print(
            f"20) SII-Proxy: voll={_s_full:.3f} tiefpass={_s_tp:.3f} delta={_d20['delta']:.3f} identisch={_d_id['delta']:.3f}  {'OK' if _s_full > _s_tp + 0.15 and _d20['verschlechterung'] and not _d_id['verschlechterung'] else 'FAIL'}"
        )
    except Exception as exc:  # pragma: no cover
        failures.append(f"SII-Proxy nicht verfügbar: {exc}")

    # 21) Cross-Ear/BMLD-Bonus in residual_audibility (Hörordnung §4, L3,
    #     Spec-SOTA-Audit 2026-11-09): diotischer Rest (beide Ohren identisch)
    #     ⇒ Bonus nahe 6 dB; inkohärenter Rest ⇒ deutlich weniger; Bonus ≤ 6.
    try:
        from backend.core.dsp.audibility_targets import residual_audibility

        _rng21 = np.random.default_rng(7)
        _ctx21 = (_rng21.standard_normal(sr * 6) * 0.01).astype(np.float64)
        _res21 = np.zeros(sr * 6)
        _res21[sr * 3 : sr * 3 + 480] = 0.05 * _rng21.standard_normal(480)
        _dio21 = np.stack([_res21, _res21])
        _rng21b = np.random.default_rng(8)
        _ink21 = np.stack([_res21, np.zeros_like(_res21)])
        _ink21[1, sr * 3 : sr * 3 + 480] = 0.05 * _rng21b.standard_normal(480)
        _ctxst21 = np.stack([_ctx21, _ctx21])
        _a21 = residual_audibility(_res21, _ctx21, sr, temporal_masking=False, stereo_pair=(_dio21, _ctxst21))
        _b21 = residual_audibility(_res21, _ctx21, sr, temporal_masking=False, stereo_pair=(_ink21, _ctxst21))
        _bo_a = float(_a21["binaural_boost_db"])
        _bo_b = float(_b21["binaural_boost_db"])
        if not (_bo_a > 5.0 and _bo_b < 2.0 and _bo_a <= 6.0):
            failures.append(f"BMLD-Bonus: diotisch {_bo_a:.2f} (erwartet >5), inkoherent {_bo_b:.2f} (erwartet <2)")
        print(
            f"21) BMLD-Bonus: diotisch={_bo_a:.2f} dB, inkoherent={_bo_b:.2f} dB  {'OK' if _bo_a > 5.0 and _bo_b < 2.0 and _bo_a <= 6.0 else 'FAIL'}"
        )
    except Exception as exc:  # pragma: no cover
        failures.append(f"BMLD-Bonus nicht verfügbar: {exc}")

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
