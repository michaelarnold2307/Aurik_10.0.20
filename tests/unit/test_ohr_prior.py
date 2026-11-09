"""Vertragstests Ohr-Prior („Mein Ohr“, 2026-11-09).

Modell-frei: synthetische Feature-Basis über kurze Signale; der Fit läuft
deterministisch auf CPU (§G5 (GEBOTE.md): Tests dürfen nicht
umgebungsabhängig kippen). Musik-ladende Pfade sind hier nicht nötig —
feature_vector nutzt die vorhandenen Metriken auf synthetischem Material.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.core.dsp.ohr_prior import (
    FEATURE_NAMEN,
    FEATURE_RICHTUNG,
    OHR_PRIOR_WEIGHTS,
    SCHEMA,
    OhrPriorFit,
    band_qualitaet,
    feature_diff,
    feature_vector,
    fit_ohr_prior,
    qualitaets_vektor,
)

SR = 48000


def _tone(freq: float = 440.0, dur_s: float = 2.0, amp: float = 0.3, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(SR * dur_s)) / SR
    noise = rng.standard_normal(len(t)) * 0.001
    return (amp * np.sin(2 * np.pi * freq * t) + noise).astype(np.float32)


def test_feature_vector_endlich_und_deterministisch() -> None:
    a = _tone()
    v1 = feature_vector(a, SR)
    v2 = feature_vector(a, SR)
    assert v1.shape == (7,)
    assert np.all(np.isfinite(v1))
    assert np.all(v1 >= -1.0) and np.all(v1 <= 1.0)
    assert np.array_equal(v1, v2)


def test_feature_diff_antisymmetrisch() -> None:
    a = _tone(440.0)
    b = _tone(880.0, seed=7)
    d_ab = feature_diff(a, b, SR)
    d_ba = feature_diff(b, a, SR)
    assert np.allclose(d_ab, -d_ba, atol=1e-6)


def test_fit_ohne_urteile_ist_prior() -> None:
    """Bei 0 Urteilen gilt der Prior (die zertifizierten Metriken dominieren)."""
    fit = fit_ohr_prior([], SR, pseudo_count=10, seed=42)
    assert fit.n_labels == 0
    assert np.allclose(fit.weights, OHR_PRIOR_WEIGHTS, atol=1e-9)
    assert fit.weights[0] > fit.weights[3], "musik_mos muss den Prior dominieren"


def test_fit_lernt_richtige_richtung_mit_genug_urteilen() -> None:
    """Synthetische Urteile: sauber (a) gewinnt immer gegen rauschig (b).
    Der Fit muss die in den Daten vorhandene Feature-1-Richtung lernen —
    das Gewicht wächst im Betrag über den Prior hinaus (Lernfähigkeit,
    nicht eine Annahme über die defektlast-Semantik).
    """
    rng = np.random.default_rng(42)
    t = np.arange(int(SR * 2)) / SR
    sauber = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    rauschig = (sauber + rng.standard_normal(len(t)).astype(np.float32) * 0.05).astype(np.float32)
    pairs = [(sauber, rauschig, 1)] * 200  # A (sauber) gewinnt 200×
    fit = fit_ohr_prior(pairs, SR, pseudo_count=10, seed=42)
    assert fit.n_labels == 200
    assert abs(fit.weights[1] - OHR_PRIOR_WEIGHTS[1]) > 0.05, (
        f"gelernt {fit.weights[1]:.3f} vs Prior {OHR_PRIOR_WEIGHTS[1]:.3f} — Fit hat nicht gelernt"
    )


def test_qualitaets_vektor_richtungen() -> None:
    """Richtungs-Kontrakt (Fix 2026-11-09): `defektlast`/`roughness`/`sharpness`/
    `fluctuation` sind ROHWERTE mit „höher = schlechter“. Der Qualitäts-Vektor
    muss sie drehen — sonst lernt das Ohr „mehr Defekte sind besser“.
    """
    _schlecht = {
        "musik_mos": 0.2,
        "defektlast": 0.9,
        "roughness_asper": 0.9,
        "sharpness_acum": 0.9,
        "loudness_phon": 0.2,
        "fluctuation_vacil": 0.9,
        "sii": 0.2,
    }
    _gut = {
        "musik_mos": 0.8,
        "defektlast": 0.1,
        "roughness_asper": 0.1,
        "sharpness_acum": 0.1,
        "loudness_phon": 0.7,
        "fluctuation_vacil": 0.1,
        "sii": 0.8,
    }
    _qs = qualitaets_vektor(_schlecht)
    _qg = qualitaets_vektor(_gut)
    for _i, _name in enumerate(FEATURE_NAMEN):
        _richtung = FEATURE_RICHTUNG[_name]
        if _richtung == "niedriger_besser":
            assert _qg[_i] > _qs[_i], f"{_name}: Rohwert-Richtung nicht gedreht"
        elif _richtung == "hoeher_besser":
            assert _qg[_i] > _qs[_i], f"{_name}: höher-besser verletzt"
    assert band_qualitaet(70.0, 70.0, 50.0) == 1.0
    assert band_qualitaet(20.0, 70.0, 50.0) == 0.0
    assert band_qualitaet(95.0, 70.0, 50.0) == 0.5


def test_feature_qualitaets_richtung_auf_breitbandsignal() -> None:
    """End-to-End-Richtung: breitbandiges Musik-Signal (rosa Rauschen +
    Hüllkurve + Transienten) vs. dieselbe Spur mit -30 dB Weißrauschen.
    Erwartet sinken die Qualitäts-Werte von musik_mos, defektlast und
    sharpness; die Differenz zeigt zum sauberen Signal.
    """
    _n = SR * 4
    _rng = np.random.default_rng(42)
    _w = _rng.standard_normal(_n)
    _spec = np.fft.rfft(_w)
    _f = np.fft.rfftfreq(_n, 1.0 / SR)
    _spec[1:] /= np.sqrt(_f[1:])
    _spec[0] = 0.0
    _sig = np.fft.irfft(_spec, n=_n)
    _sig /= np.max(np.abs(_sig)) + 1e-12
    _t = np.arange(_n) / SR
    _sig *= 0.6 + 0.4 * np.sin(2 * np.pi * 2.0 * _t)
    for _k in range(0, _n, SR // 2):
        _m = min(240, _n - _k)
        _sig[_k : _k + _m] += 0.8 * np.exp(-np.arange(_m) / 40.0)
    _clean = (0.3 * _sig / (np.max(np.abs(_sig)) + 1e-12)).astype(np.float32)
    _rausch = np.random.default_rng(7).standard_normal(_n).astype(np.float32) * 0.01
    _noisy = (_clean + _rausch).astype(np.float32)

    _qc = feature_vector(_clean, SR)
    _qn = feature_vector(_noisy, SR)
    _idx = {name: i for i, name in enumerate(FEATURE_NAMEN)}
    for _name in ("musik_mos", "defektlast", "sharpness_acum"):
        assert _qc[_idx[_name]] >= _qn[_idx[_name]], f"{_name}: verrauschtes Signal darf nicht besser sein"
    _d = _qc - _qn
    assert float(np.dot(OHR_PRIOR_WEIGHTS, _d)) > 0.0, "Prior muss das saubere Signal vorziehen"


def test_alte_schema_version_wird_abgelehnt(tmp_path) -> None:
    """Schema `/1` hatte ROHWERTE (invertierte defektlast-Richtung) — es darf
    NICHT interpretiert werden, sondern fail-closed abgelehnt (§V6 (copilot-instructions.md))."""
    import json as _json

    from backend.core.dsp.ohr_brief import lade_ohr_prior
    from backend.core.dsp.ohr_prior import OhrPriorFit as _Fit

    _alt = _Fit().to_dict()
    _alt["schema"] = "aurik.ohr_prior/1"
    _p = tmp_path / "alt.json"
    _p.write_text(_json.dumps(_alt), encoding="utf-8")
    assert SCHEMA == "aurik.ohr_prior/2"
    _fit, _grund = lade_ohr_prior(_p)
    assert _fit is None and _grund == "ValueError", (_fit, _grund)


def test_predict_symmetrie() -> None:
    a = _tone(440.0)
    b = _tone(880.0, seed=7)
    fit = fit_ohr_prior([], SR, pseudo_count=10, seed=42)
    p_ab = float(fit.predict(a, b, SR)["p_a_gewinnt"])
    p_ba = float(fit.predict(b, a, SR)["p_a_gewinnt"])
    assert p_ab == pytest.approx(1.0 - p_ba, abs=1e-6)


def test_to_from_dict_roundtrip() -> None:
    fit = fit_ohr_prior([(_tone(440.0), _tone(880.0, seed=7), 1)] * 30, SR, pseudo_count=10, seed=42)
    data = fit.to_dict()
    assert data["schema"] == "aurik.ohr_prior/2"
    fit2 = OhrPriorFit.from_dict(data)
    assert np.allclose(fit.weights, fit2.weights, atol=1e-9)
    assert fit2.n_labels == fit.n_labels


def test_fit_determinismus() -> None:
    pairs = [(_tone(440.0, seed=3), _tone(880.0, seed=5), 1.0)] * 40
    f1 = fit_ohr_prior(pairs, SR, pseudo_count=10, seed=42)
    f2 = fit_ohr_prior(pairs, SR, pseudo_count=10, seed=42)
    assert np.allclose(f1.weights, f2.weights, atol=1e-12)
    assert f1.log_loss == f2.log_loss


def test_uv3_ohr_prior_konsultation_verdrahtet() -> None:
    """Wiring-Guard (Muster C, §7): Der Ohr-Prior MUSS im Produktionspfad
    konsultiert werden (Report-Only, artefakt-geführt) — sonst wäre die
    §7-Verankerung eine Phantom-Norm.
    """
    from pathlib import Path as _P

    _root = _P(__file__).resolve().parents[2]
    uv3 = (_root / "backend" / "core" / "unified_restorer_v3.py").read_text(encoding="utf-8")
    assert '_flow_meta["ohr_prior"]' in uv3, "Ohr-Prior-Report fehlt in UV3"
    assert "erstelle_ohr_brief" in uv3, "UV3 konsultiert nicht die zentrale Ohr-Brief-Quelle"

    brief = (_root / "backend" / "core" / "dsp" / "ohr_brief.py").read_text(encoding="utf-8")
    assert "models/ohr/ohr_prior.json" in brief, "Ohr-Prior-Artefaktpfad fehlt in ohr_brief"
    assert "OhrPriorFit.from_dict" in brief, "Ohr-Prior-Ladung fehlt in ohr_brief"
    assert "ohr_prior_verfuegbar=False" in brief, "Fail-safe-Zweig fehlt in ohr_brief"


def test_denker_ohr_brief_verdrahtet() -> None:
    """Wiring-Guard (Muster C, §7): Das Ohr MUSS im Produktionspfad ankommen.

    Aktuell verbraucht: UV3 (Report `metadata["ohr_prior"]`) und der
    StrategieDenker (Ohr-Brief im Strategieplan). Die weitergehende
    Priorisierung im Sweet-Spot-Loop (Restaurier-/Defekt-/Exzellenz-Denker)
    ist bewusst noch NICHT verdrahtet — sie wäre klangwirksam und braucht
    eine eigene Hör-/B3-Entscheidung (Blocker in D-K3-89).
    """
    from pathlib import Path as _P

    _root = _P(__file__).resolve().parents[2]
    for name, erwartet in [
        ("denker/strategie_denker.py", ["erstelle_ohr_brief", "ohr_brief"]),
        ("backend/core/unified_restorer_v3.py", ["erstelle_ohr_brief", '_flow_meta["ohr_prior"]']),
    ]:
        src = (_root / name).read_text(encoding="utf-8")
        for token in erwartet:
            assert token in src, f"{token} fehlt in {name} — Ohr nicht verdrahtet"


def test_ohr_brief_ziel_gewichte_invarianten() -> None:
    """Ohr-Brief: ohne Artefakt None (Default-Pfad); mit Artefakt deterministisch,
    Summe exakt 1, nur die 9 Sweet-Spot-Metriken.
    """
    import json
    import tempfile
    from pathlib import Path as _P_tmp

    from backend.core.dsp.ohr_brief import erstelle_ohr_brief
    from backend.core.dsp.ohr_prior import OhrPriorFit

    _brief_ohne = erstelle_ohr_brief()
    assert _brief_ohne.ohr_prior_verfuegbar is False
    assert _brief_ohne.ziel_gewichte() is None

    with tempfile.TemporaryDirectory() as _td:
        _p = _P_tmp(_td) / "ohr_prior.json"
        _fit = OhrPriorFit(weights=[0.40, 0.30, 0.05, 0.04, 0.08, 0.03, 0.10], n_labels=18, pseudo_count=6)
        _p.write_text(json.dumps(_fit.to_dict()), encoding="utf-8")
        _brief = erstelle_ohr_brief(artefakt=_p)
        assert _brief.ohr_prior_verfuegbar is True
        _w = _brief.ziel_gewichte()
        assert _w is not None
        assert abs(sum(_w.values()) - 1.0) < 1e-12
        assert set(_w) == {
            "hpe",
            "inviting",
            "transparency",
            "goosebumps",
            "comb_filter",
            "musical_compression",
            "masking_health",
            "spectral_color",
            "microdynamics",
        }
        assert _w == _brief.ziel_gewichte(), "ziel_gewichte nicht deterministisch"
        assert _w["goosebumps"] == 0.0, "goosebumps hat kein Ohr-Feature-Mapping (neutral)"


def test_ohr_stufen_und_zonen_kalibrierung() -> None:
    """§7 M3 (2026-11-09): Gestufte Ohr-Reifung — Stufe 0/1 neutral, Stufe 2
    Zonen-Kalibrierung ±5 %, Stufe 3 ±8 %; deterministisch und summen-neutral.
    """
    import json
    import tempfile
    from pathlib import Path as _P

    from backend.core.dsp.ohr_brief import erstelle_ohr_brief
    from backend.core.dsp.ohr_prior import OhrPriorFit

    # Stufe 0: ohne Artefakt — alles neutral
    _b0 = erstelle_ohr_brief()
    assert _b0.ohr_stufe() == 0
    assert _b0.zonen_kalibrierung() is None
    assert _b0.as_dict()["ohr_stufe"] == 0

    def _brief_mit(n_labels: int, weights: list[float]) -> object:
        _dir = tempfile.TemporaryDirectory()
        _p = _P(_dir.name) / "ohr_prior.json"
        _fit = OhrPriorFit(weights=weights, n_labels=n_labels, pseudo_count=6)
        _p.write_text(json.dumps(_fit.to_dict()), encoding="utf-8")
        return erstelle_ohr_brief(artefakt=_p)

    # Stufe 1: getragen < 0.5 (n=4/10=0.4) — Priorisierung, KEINE Zonen
    _b1 = _brief_mit(4, [0.40, 0.30, 0.05, 0.04, 0.08, 0.03, 0.10])
    assert _b1.ohr_stufe() == 1
    assert _b1.zonen_kalibrierung() is None

    # Stufe 2: getragen >= 0.5 (n=18/24=0.75) — Zonen ±5 %, deterministisch
    _b2 = _brief_mit(18, [0.25, 0.45, 0.06, 0.05, 0.08, 0.04, 0.07])
    assert _b2.ohr_stufe() == 2
    _z2 = _b2.zonen_kalibrierung()
    assert _z2 is not None
    assert set(_z2) == {
        "hpe",
        "inviting",
        "transparency",
        "goosebumps",
        "comb_filter",
        "musical_compression",
        "masking_health",
        "spectral_color",
        "microdynamics",
    }
    assert all(abs(v) <= 0.05 + 1e-9 for v in _z2.values())
    assert abs(sum(_z2.values())) < 1e-5, "Zonen-Deltas müssen summen-neutral sein"
    assert _z2 == _b2.zonen_kalibrierung(), "zonen_kalibrierung nicht deterministisch"

    # Stufe 3: getragen >= 0.8 (n=60/66≈0.91) — Zonen ±8 %
    _b3 = _brief_mit(60, [0.25, 0.45, 0.06, 0.05, 0.08, 0.04, 0.07])
    assert _b3.ohr_stufe() == 3
    _z3 = _b3.zonen_kalibrierung()
    assert _z3 is not None
    assert all(abs(v) <= 0.08 + 1e-9 for v in _z3.values())
    assert max(abs(_z3[k]) for k in _z3) >= max(abs(_z2[k]) for k in _z2) - 1e-9, "Stufe 3 ≥ Stufe 2"

    # Persönliche Abweichung wirkt: defektlast stärker gewichtet ⇒
    # comb_filter/masking_health strenger (positives Delta)
    assert _z2["comb_filter"] > 0.0 and _z2["masking_health"] > 0.0


def test_ohr_zonen_loop_cap() -> None:
    """§7 M3: Die Loop-Anwendung der Zonen-Deltas ist kumulativ auf
    [0.85, 1.10] der Basis-GREEN_ZONE begrenzt — auch bei Extrem-Deltas.
    """
    import numpy as np

    from backend.core.sweet_spot_optimizer import GREEN_ZONE

    _extreme = dict.fromkeys(GREEN_ZONE, 0.50)  # +50 % — weit über jedem Cap
    _t = {
        k: float(
            np.clip(
                GREEN_ZONE[k] * (1.0 + _extreme.get(k, 0.0)),
                GREEN_ZONE[k] * 0.85,
                GREEN_ZONE[k] * 1.10,
            )
        )
        for k in GREEN_ZONE
    }
    for _k in GREEN_ZONE:
        assert GREEN_ZONE[_k] * 0.85 <= _t[_k] <= GREEN_ZONE[_k] * 1.10, _k
    assert all(abs(_t[_k] - GREEN_ZONE[_k] * 1.10) < 1e-9 for _k in GREEN_ZONE)
