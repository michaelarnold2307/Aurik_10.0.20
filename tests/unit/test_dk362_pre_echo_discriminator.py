"""§D-K3-62 (2026-10-08): Diskriminator „Echo-Duplikat vs. Reparatur-Füllung“.

Befund (Lauf 10.12.17, AURIK_WITNESS_DEBUG-Dumps): Die Reinhör-Witness meldete
auf Reparatur-Phasen pre_echo-Befunde (−2,8/−8,4/−10,5 dB über der −12-dB-
Schwelle), obwohl die Dumps positionsidentische Wellenform-Peaks und
bit-identische Nachbar-Frames zeigten — die „Onsets“ waren unveränderte Musik,
die +Energie war die GEWÜNSCHTE Reparatur-Füllung (D-K3-62-Empfehlung:
Duplikat-/Reparatur-Diskriminator in der Metrik).

Dieser Test pinnt den Diskriminator (`analyze_pre_echo`, report-only):
  - eine CODEC-ARTIGE Kopie des Onsets kurz vor dem Onset → kind "duplicate"
  - eine unkorrelierte lokale Füllung (Dropout-Parität) → kind "repair_fill"
  - kein Delta / kein Onset → kind "none"
  - `pre_echo_ratio_db` bleibt als Wrapper bit-identisch (§G9 copilot-instructions.md)

Die Hörstichprobe vor jeder Verschärfung bleibt ausdrücklich offen (D-K3-62) —
der Test pinnt daher NUR die Klassifikation, keine Veto-Wirkung
(§V7 copilot-instructions.md).
"""

from __future__ import annotations

import numpy as np

from backend.core.dsp.pre_echo_model import (
    analyze_pre_echo,
    pre_echo_ratio_db,
)

SR = 48000


def _onset_base(seed: int = 5, sustain: bool = True) -> tuple[np.ndarray, int]:
    """2-s-Rauschboden mit starkem Onset (Rezept der SUP-F7-Messreihe).

    ``sustain=False`` für die Duplikat-Fälle: das 0,4-Plateau erzeugt einen
    zweiten Onset-Kandidaten, dessen Onset-Fenster flach ist — der Diskriminator
    würde dort korrekt „unkorreliert“ sehen (das Plateau ist kein Transient).
    """
    rng = np.random.default_rng(seed)
    n = int(SR * 2.0)
    base = (rng.standard_normal(n) * 0.01).astype(np.float32)
    onset = int(SR * 1.0)
    base[onset : onset + 200] += 0.8 * np.hanning(200).astype(np.float32)
    if sustain:
        base[onset + 200 : onset + 800] += 0.4
    return base, onset


def _with_duplicate(base: np.ndarray, onset: int, level: float = 0.45) -> np.ndarray:
    """Codec-artige Kopie der Attack-Wellenform 12 ms VOR dem Onset."""
    with_copy = base.copy()
    shift = int(SR * 0.012)  # 12 ms vor dem Onset (Codec-Charakter)
    with_copy[onset - shift : onset - shift + 200] += (level / 0.8) * base[onset : onset + 200]
    return with_copy


def test_kopie_vor_dem_onset_wird_als_duplikat_klassifiziert() -> None:
    """Codec-Pre-Echo ist die rückwärts verschmierte KOPIE des Transienten."""
    base, onset = _onset_base(sustain=False)
    with_copy = _with_duplicate(base, onset)
    res = analyze_pre_echo(base, with_copy, SR)
    assert res.ratio_db > -12.0, f"Befund erwartet, erhalten {res.ratio_db}"
    assert res.kind == "duplicate", f"Kopie muss Duplikat sein, corr={res.corr}"
    assert res.corr >= 0.30
    assert res.onset_sample >= 0


def test_unkorrelierte_fuellung_wird_als_repair_fill_klassifiziert() -> None:
    """Lokale Reparatur-Füllung (Dropout-Parität) ist NEUE, unkorrelierte Energie."""
    base, onset = _onset_base()
    with_fill = base.copy()
    rng = np.random.default_rng(6)
    pre = int(SR * 0.010)
    fill = (rng.standard_normal(pre) * 0.30).astype(np.float32)  # lokale Füllung
    with_fill[onset - pre : onset] += fill
    res = analyze_pre_echo(base, with_fill, SR)
    assert res.ratio_db > -12.0, f"Befund erwartet, erhalten {res.ratio_db}"
    assert res.kind == "repair_fill", f"Füllung darf kein Duplikat sein, corr={res.corr}"
    assert res.corr < 0.30


def test_kein_delta_ist_kein_befund() -> None:
    """Hörordnung Ebene 2: ohne Delta keine Klassifikation — kind "none"."""
    base, _ = _onset_base()
    res = analyze_pre_echo(base, base, SR)
    assert res.ratio_db == -200.0
    assert res.kind == "none"
    assert res.corr == 0.0
    assert res.onset_sample == -1


def test_wrapper_bleibt_bit_identisch() -> None:
    """§G9 (copilot-instructions.md): `pre_echo_ratio_db` = Analyse-Verhältnis."""
    base, onset = _onset_base(sustain=False)
    with_copy = _with_duplicate(base, onset)
    an = analyze_pre_echo(base, with_copy, SR)
    assert pre_echo_ratio_db(base, with_copy, SR) == an.ratio_db


def test_witness_traegt_kind_und_corr_im_report() -> None:
    """Der Reinhör-Witness reicht die Klassifikation ins Ergebnis durch (§G8 copilot-instructions.md)."""
    from backend.core.listening_witness import evaluate_listening_witness

    base, onset = _onset_base(sustain=False)
    with_copy = _with_duplicate(base, onset)
    res = evaluate_listening_witness(base, with_copy, SR, "phase_dk362")
    assert res.pre_echo_kind == "duplicate"
    assert res.pre_echo_corr >= 0.30
    d = res.as_dict()
    assert d["pre_echo_kind"] == "duplicate"
    assert "pre_echo_corr" in d
