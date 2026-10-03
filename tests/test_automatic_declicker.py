"""Regression und Zielabnahme der Declick-Familie (Hörordnung Ebene 2)."""

import logging

import numpy as np
import pytest

from backend.core.dsp.automatic_declicker import AutomaticDeclicker

SR = 44100


def _tone(amp=0.3, freq=330.0, n=SR):
    return (amp * np.sin(2 * np.pi * freq * np.arange(n) / SR)).astype(np.float32)


def test_declicker_removes_clicks():
    audio = np.zeros(SR)
    for pos in (100, 500, 1000):
        audio[pos] = 1.0
    processed = AutomaticDeclicker(threshold=0.5).process(audio, SR)
    for pos in (100, 500, 1000):
        assert abs(processed[pos]) < abs(audio[pos])


@pytest.mark.parametrize("threshold", [0.1, 0.5, 0.9])
def test_declicker_threshold_variation(threshold):
    audio = np.zeros(SR)
    audio[100] = 1.0
    assert AutomaticDeclicker(threshold=threshold).process(audio, SR).shape == audio.shape


def test_declicker_no_clicks_bit_identisch():
    audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, SR))
    assert np.array_equal(AutomaticDeclicker().process(audio, SR), audio)


def test_declicker_quiet_click_removed():
    audio = _tone(amp=0.01, freq=220.0)
    audio[3000] = 0.05
    processed = AutomaticDeclicker(threshold=0.6).process(audio, SR)
    assert abs(processed[3000] - audio[3000]) > 0.04
    assert int(np.sum(processed != audio)) <= 8


def test_declicker_identity_outside_defects():
    audio = (0.5 * np.sin(2 * np.pi * 440 * np.arange(SR) / SR)).astype(np.float32)
    assert np.array_equal(AutomaticDeclicker().process(audio, SR), audio)


def test_declicker_stereo_layout_invariant():
    mono = _tone()
    cf = np.stack([mono, 0.5 * mono], axis=0)
    cf[0, 2000] = 1.0
    out_cf = AutomaticDeclicker().process(cf, SR)
    out_sf = AutomaticDeclicker().process(np.ascontiguousarray(cf.T), SR)
    assert out_cf.shape == cf.shape
    assert np.allclose(out_cf.T, out_sf)
    assert np.array_equal(out_cf[1], cf[1])


def test_dosierung_kleiner_und_grosser_klick_unter_ziel():
    from backend.core.dsp.audibility_targets import verify_declick_repair
    from backend.core.dsp.declick_core import detect_click_mask

    for amp, base in ((0.05, 0.01), (0.9, 0.3)):
        sig = _tone(amp=base, freq=220.0)
        sig[3000] = amp
        mask = detect_click_mask(sig, strictness_k=6.0)
        assert mask[3000], f"Klick amp={amp} nicht detektiert"
        processed = AutomaticDeclicker(threshold=0.6).process(sig, SR)
        rep = verify_declick_repair(sig, processed, mask, SR)
        assert rep["rest_subaudible"] is True, f"Rest über Ziel (amp={amp}): {rep}"
        assert rep["material_preserved"] is True, f"Material verändert (amp={amp}): {rep}"


def test_uebriger_spike_rest_verfehlt_ziel():
    """Mess-Kern: Ein übrig gebliebener Spike-Rest MUSS als 'nicht erreicht' gelten.

    Regression zum Selbstmaskierungs-/Domänen-Befund (2026-10-02): der Rest
    darf nicht durch sich selbst maskiert werden — die Maske kommt aus dem
    Kontext mit ausgenullter Defektstelle (audibility_gate-Muster).
    """
    from backend.core.dsp.audibility_targets import verify_declick_repair

    sig = _tone(amp=0.02, freq=300.0)
    sig[3000] = 5.0  # monströser Spike-Rest bleibt übrig
    mask = np.zeros(SR, dtype=bool)
    mask[3000] = True
    rep = verify_declick_repair(sig, sig, mask, SR)
    assert rep["rest_subaudible"] is False, f"Spike-Rest nicht als Verfehlung erkannt: {rep}"
    assert rep["objective_met"] is False


def test_guard_log_macht_verfehlung_sichtbar():
    """§0c: Ergebnis bleibt unverändert; §V6 (VERBOTEN.md): Verfehlung wird geloggt.

    Propagationsunabhängig gemessen (eigener Handler am Modul-Logger) und
    ohne Messabhängigkeit (verify wird auf 'verfehlt' gesetzt) — der Log-Pfad
    des Guards ist der Testgegenstand, die Messung testet
    test_uebriger_spike_rest_verfehlt_ziel.
    """
    from backend.core.dsp import declick_core as dc

    sig = _tone(amp=0.02, freq=300.0)
    sig[3000] = 5.0
    mask = np.zeros(SR, dtype=bool)
    mask[3000] = True

    records: list[str] = []

    class _H(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = _H()
    dc.logger.addHandler(handler)
    dc.logger.setLevel(logging.DEBUG)
    try:
        dc._log_repair_objective(sig, sig, mask, SR, label="TestGuard")
    finally:
        dc.logger.removeHandler(handler)
    assert any("Zielabnahme" in m for m in records), f"Guard-Log leer: {records!r}"


def test_decrackle_reduces_impulse_energy():
    from backend.core.dsp.automatic_decrackler import AutomaticDecrackler

    rng = np.random.default_rng(7)
    audio = _tone(amp=0.05)
    idx = rng.choice(SR, size=300, replace=False)
    audio[idx] += 0.03 * rng.standard_normal(300).astype(np.float32)
    processed = AutomaticDecrackler(threshold=0.4).decrackle(audio, SR)
    assert float(np.mean(np.abs(np.diff(processed)))) < float(np.mean(np.abs(np.diff(audio))))


def test_shellac_und_riaa_reparieren_mit_defaults():
    from backend.core.dsp.riaa_declicker import AiRiaaDeclicker
    from backend.core.dsp.shellac_declicker import ShellacDeclicker

    audio = np.zeros(SR, dtype=np.float32)
    audio[700] = 0.8
    assert abs(ShellacDeclicker().process(audio, SR, audit_log=False)[700]) < 0.1
    assert abs(AiRiaaDeclicker().declick_riaa(audio, SR, audit_log=False)[700]) < 0.1
