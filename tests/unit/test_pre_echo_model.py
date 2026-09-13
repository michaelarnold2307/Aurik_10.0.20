"""§Witness-SOTA P4: Tests für den Pre-Echo-Proxy (pre_echo_model.py).

Belegt die zwei Fixes von 2026-09-13:
1. Lokale Onset-Baseline (10. Perzentil, ±0,5 s) statt globalem Median —
   der globale Median verschluckte Anstiege in lauten Passagen (6,3×-Attack
   wurde als 1,85× gemessen → Onset nie erkannt).
2. Forward-Masking-Audibility-Gate (Zwicker & Fastl §7.2): Nur HINZUGEFÜGTE
   Vor-Fenster-Energie über Onset-Pegel − 18 dB löst einen Befund aus —
   |delta|² war vorzeichenblind: Klick-ENTFERNUNG vor einem Transienten
   zählte wie Pre-Echo-HINZUFÜGUNG (False-Positive bei phase_01_click_removal,
   Produktionsbefund 2026-09-13).
"""

from __future__ import annotations

import numpy as np

from backend.core.dsp.pre_echo_model import pre_echo_ratio_db

SR = 44100


def _attack_signal() -> np.ndarray:
    """2 s Sinus+Rauschen mit steilem 6×-Attack bei 1,0 s."""
    t = np.arange(SR * 2) / SR
    rng = np.random.default_rng(4)
    x = (0.3 * np.sin(2 * np.pi * 220 * t) + 0.02 * rng.standard_normal(SR * 2)).astype(np.float32)
    x[SR:] *= 6.0
    return x


def test_identity_returns_no_finding() -> None:
    x = _attack_signal()
    assert pre_echo_ratio_db(x, x, SR) == -200.0


def test_click_repair_before_transient_is_no_pre_echo() -> None:
    """Declicker-Szenario: Klick-ENTFERNUNG 10 ms vor einem Musikanstieg darf
    keinen Pre-Echo-Befund erzeugen (False-Positive-Fix)."""
    x = _attack_signal()
    click = int(SR * 0.99)
    x_click = x.copy()
    x_click[click : click + 40] = 1.0
    repaired = x_click.copy()
    repaired[click : click + 40] = np.interp(np.arange(40), [0, 39], [x_click[click - 1], x_click[click + 40]])
    assert pre_echo_ratio_db(x, repaired, SR) == -200.0


def test_real_pre_echo_is_detected() -> None:
    """Echtes Pre-Echo (−6 dB Smear, 15 ms vor dem Onset) → Befund (> −12 dB)."""
    x = _attack_signal()
    y = x.copy()
    length = int(0.03 * SR)
    y[SR - length : SR] += 0.5 * x[SR : SR + length]
    assert pre_echo_ratio_db(x, y, SR) > -12.0


def test_inaudible_pre_echo_below_masking_is_ignored() -> None:
    """−30 dB Pre-Energie liegt unter der Forward-Masking-Schwelle → kein Befund."""
    x = _attack_signal()
    y = x.copy()
    length = int(0.03 * SR)
    y[SR - length : SR] += 0.03 * x[SR : SR + length]
    assert pre_echo_ratio_db(x, y, SR) == -200.0


def test_onset_detection_works_in_loud_passages() -> None:
    """Lokale Baseline: Anstieg in lauter Passage wird erkannt (vor dem Fix
    verschluckte der globale Median den 6×-Attack → echter Pre-Echo unentdeckt)."""
    x = _attack_signal()
    y = x.copy()
    length = int(0.03 * SR)
    y[SR - length : SR] += 0.5 * x[SR : SR + length]
    # Ohne funktionierende Onset-Erkennung wäre das Ergebnis -200.
    assert pre_echo_ratio_db(x, y, SR) > -12.0


def test_silent_intro_produces_no_onsets_from_noise_floor() -> None:
    """Realistischer Song-Anfang: digitales Intro (−86 dB Rauschboden) vor dem
    Attack. Die lokale Baseline ist auf −60 dB unter dem Song-Peak gefloort —
    der Rauschboden erzeugt KEINE Onsets (Produktionsbefund auf dem echten
    Import-Song: Baseline ≈ 0 → jeder Hauch Rauschen wirkte als Anstieg).
    Ein Pre-Echo direkt vor dem echten Attack wird weiterhin gemeldet."""
    rng = np.random.default_rng(11)
    x = np.zeros(SR * 3, dtype=np.float32)
    x[:SR] = (5e-6 * rng.standard_normal(SR)).astype(np.float32)  # −86 dB digitales Intro
    t = np.arange(SR * 2) / SR
    x[SR:] = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    length = int(0.03 * SR)
    y = x.copy()
    y[SR - length : SR] += 0.5 * x[SR : SR + length]  # echtes Pre-Echo vor dem Attack
    assert pre_echo_ratio_db(x, y, SR) > -12.0


def test_deterministic() -> None:
    x = _attack_signal()
    y = x.copy()
    length = int(0.03 * SR)
    y[SR - length : SR] += 0.5 * x[SR : SR + length]
    assert pre_echo_ratio_db(x, y, SR) == pre_echo_ratio_db(x, y, SR)
