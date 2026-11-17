"""Vertragstests: referenzfreier Wohlklang-Zeuge je Lauf (docs/SOTA_HANDOVER.md §2b Hebel F).

Geprüft wird die Wirkungs-Semantik, nicht die Implementierung:

* Richtung — eine verschlechterte Ausgabe wird als gerichteter Fall markiert,
  eine identische nicht (Hörordnung §7/§8a: Zeuge, keine Entscheidung).
* REPORT-ONLY — kein Sample-Eingriff, keine Ausnahme im Produktionspfad (§V6 (copilot-instructions.md)).
* Determinismus (§G5 (copilot-instructions.md)) und Layout-Invarianz (AGENTS.md §3).
* EINE Wahrheit — die Metrik-Definition ist bit-identisch zur Mess-Referenz,
  die JND-Konstante identisch in Stufenmessung/Kompositions-Suche/Zeuge.
* Verdrahtung — UV3 hängt den Befund in BEIDEN Produktionspfaden an die Metadaten.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from backend.core.dsp import wohlklang_zeuge as wz

PROJEKT = Path(__file__).resolve().parents[2]
SR = 48000


def _musik(seconds: float = 1.0, seed: int = 42) -> np.ndarray:
    """Deterministisches musikalisches Testsignal (Dreiklang + Rauschteppich)."""
    t = np.arange(int(seconds * SR), dtype=np.float64) / SR
    rng = np.random.default_rng(seed)
    x = (
        0.30 * np.sin(2.0 * np.pi * 220.0 * t)
        + 0.20 * np.sin(2.0 * np.pi * 330.0 * t)
        + 0.15 * np.sin(2.0 * np.pi * 440.0 * t)
        + 0.005 * rng.standard_normal(t.size)
    )
    return np.clip(x, -1.0, 1.0).astype(np.float32)


def _verdefektet(x: np.ndarray) -> np.ndarray:
    """Deutlich verschlechterte Fassung: Clipping + hochgezogenes Rauschen."""
    y = np.clip(x * 3.0, -1.0, 1.0)
    rng = np.random.default_rng(7)
    y = y + 0.08 * rng.standard_normal(y.size).astype(np.float32)
    return np.clip(y, -1.0, 1.0).astype(np.float32)


def test_identisches_signal_ist_kein_gerichteter_fall() -> None:
    x = _musik()
    befund = wz.messe_wohlklang_zeuge(x, x, SR)

    assert befund["verfuegbar"] is True
    assert befund["report_only"] is True
    assert befund["delta"] == {"musik_mos": 0.0, "defektlast": 0.0, "saettigung_kanal": 0.0, "jitter_kanal": 0.0}
    assert befund["gerichteter_fall"] is False
    assert befund["hoer_instanz"] == "keine_vorlage"
    assert befund["grund"] == []
    assert befund["metriken"] == ["musik_mos", "defektlast", "saettigung_kanal", "jitter_kanal"]


def test_verschlechterung_wird_gerichteter_fall() -> None:
    x = _musik()
    befund = wz.messe_wohlklang_zeuge(x, _verdefektet(x), SR)

    assert befund["delta"]["defektlast"] > 0.0
    assert befund["regression"]["defektlast"] is True
    assert befund["gerichteter_fall"] is True
    assert befund["hoer_instanz"] == "vorzulegen"
    assert any("defektlast" in g for g in befund["grund"])


def test_verbesserung_wird_nicht_gemeldet() -> None:
    """Nie-worsen ist asymmetrisch: besser = kein Befund (kein Belohnungs-Gate)."""
    x = _musik()
    befund = wz.messe_wohlklang_zeuge(_verdefektet(x), x, SR)

    assert befund["delta"]["defektlast"] < 0.0
    assert befund["gerichteter_fall"] is False


def test_determinismus_gleicher_eingang_gleicher_befund() -> None:
    x = _musik()
    y = _verdefektet(x)
    assert wz.messe_wohlklang_zeuge(x, y, SR) == wz.messe_wohlklang_zeuge(x, y, SR)


def test_stereo_layout_invarianz() -> None:
    """(C, N) und (N, C) müssen denselben Befund liefern (AGENTS.md §3)."""
    mono = _musik()
    stereo_cf = np.vstack([mono, 0.5 * mono])  # (2, N) channels-first
    stereo_nc = stereo_cf.T  # (N, 2) samples-first

    assert wz.defektlast(stereo_cf, SR) == wz.defektlast(stereo_nc, SR)
    a = wz.messe_wohlklang_zeuge(stereo_cf, stereo_cf, SR)
    b = wz.messe_wohlklang_zeuge(stereo_nc, stereo_nc, SR)
    assert (
        a["delta"] == b["delta"] == {"musik_mos": 0.0, "defektlast": 0.0, "saettigung_kanal": 0.0, "jitter_kanal": 0.0}
    )


def test_kein_eingriff_in_die_audio_daten() -> None:
    """REPORT-ONLY: der Zeuge darf weder Eingang noch Ausgang verändern."""
    x = _musik()
    y = _verdefektet(x)
    x_kopie, y_kopie = x.copy(), y.copy()

    wz.messe_wohlklang_zeuge(x, y, SR)

    assert np.array_equal(x, x_kopie)
    assert np.array_equal(y, y_kopie)


def test_jnd_grenze_steuert_die_meldung() -> None:
    """Unterhalb der JND kein Befund; oberhalb schon — ohne Sonderpfade."""
    x = _musik()
    y = _verdefektet(x)

    streng = wz.messe_wohlklang_zeuge(x, y, SR, jnd=1e-9)
    assert streng["gerichteter_fall"] is True

    tolerant = wz.messe_wohlklang_zeuge(x, y, SR, jnd=10.0)
    assert tolerant["gerichteter_fall"] is False
    assert tolerant["hoer_instanz"] == "keine_vorlage"


def test_zu_kurzes_signal_wirft_im_kern_und_meldet_sich_produktionsnah() -> None:
    kurz = np.zeros(1024, dtype=np.float32)

    with pytest.raises(ValueError, match="zu kurz"):
        wz.messe_wohlklang_zeuge(kurz, kurz, SR)

    meta = wz.zeuge_meta(kurz, kurz, SR)  # §V6 (copilot-instructions.md): nie werfend
    assert meta["verfuegbar"] is False
    assert meta["report_only"] is True
    assert meta["gerichteter_fall"] is False
    assert "zu kurz" in meta["grund"]


def test_ungueltige_sample_rate_meldet_sich_produktionsnah() -> None:
    x = _musik()
    meta = wz.zeuge_meta(x, x, 0)
    assert meta["verfuegbar"] is False
    assert "ungültige Sample-Rate" in meta["grund"]


def test_defektlast_bit_identisch_zur_messreferenz() -> None:
    """EINE Wahrheit: Zertifikats-Skript und Produktions-Zeuge rechnen gleich."""
    from scripts.quality_metric_direction_gate import _defektlast

    for signal in (_musik(), _verdefektet(_musik()), _musik(seconds=0.25, seed=3)):
        assert wz.defektlast(signal, SR) == _defektlast(signal, SR)


def test_jnd_konstante_ist_eine_quelle() -> None:
    """Stufenmessung, Kompositions-Suche und Zeuge dürfen nicht auseinanderlaufen."""
    import scripts.b3_komposition_suche as komposition
    import scripts.b3_stufenmessung as stufen

    assert stufen.JND_TOLERANZ == wz.JND_TOLERANZ
    assert komposition.JND_TOLERANZ == wz.JND_TOLERANZ


def test_befund_ist_json_faehig_und_nennt_pegel_delta() -> None:
    import json

    befund = wz.messe_wohlklang_zeuge(_musik(), _verdefektet(_musik()), SR)
    json.dumps(befund)  # metadaten-tauglich (keine numpy-Skalare)
    assert befund["schema"] == wz.SCHEMA
    # Diagnose, keine Metrik: die Test-Degradation ist ~+9 dB lauter.
    assert befund["pegel_delta_db"] > 3.0
    assert befund["laenge_delta_samples"] == 0


def test_uv3_verdrahtung_beide_produktionspfade() -> None:
    """Wirkungs-Pin (Muster C): der Befund landet im Einzel- und im Chunked-Pfad."""
    quelle = (PROJEKT / "backend/core/unified_restorer_v3.py").read_text(encoding="utf-8")

    assert "def _wohlklang_zeuge_meta(" in quelle
    # 3 Aufrufe (Einzelpfad, Chunked-Pfad, Options-Pfad) — Definition separat geprüft
    assert quelle.count("self._wohlklang_zeuge_meta(") == 3
    assert 'result.metadata["wohlklang_zeuge"]' in quelle
    assert '_meta["wohlklang_zeuge"]' in quelle
    # REPORT-ONLY: genau DREI Schreibstellen, keine Lesestelle (kein Gate/Pfad
    # entscheidet über den Zeugen — die Hör-Instanz entscheidet).
    assert quelle.count('"wohlklang_zeuge"') == 3


# ---------------------------------------------------------------------------
# Jitter-Kanal (Q0-Witness-Forschung 2026-11-14)
# ---------------------------------------------------------------------------


def test_jitter_kanal_ist_gain_invariant() -> None:
    """Skalierung kürzt sich im Seitenband-Verhältnis (gain-invariant)."""
    x = _musik()
    assert wz.jitter_kanal(x, SR) == pytest.approx(wz.jitter_kanal(0.2 * x, SR), abs=1e-3)


def test_jitter_kanal_sieht_fm_jitter() -> None:
    """FM-Jitter (8 Hz Phasenmodulation) erhöht den Seitenband-Anteil messbar.
    Akkord statt Einzelton: bei einer einzelnen Sinus-Linie liegen alle
    Top-Bins im selben Hauptkeulen (degeneriert) — gemessen: Akkord
    clean 1,18 → FM 1,25 (Δ +0,068 > JND 0,02)."""
    t = np.arange(8192) / SR
    clean = (
        0.25
        * (
            np.sin(2 * np.pi * 440 * t)
            + np.sin(2 * np.pi * 880 * t)
            + np.sin(2 * np.pi * 1320 * t)
            + np.sin(2 * np.pi * 1760 * t)
        )
    ).astype(np.float32)
    fm = 0.3 * np.sin(2 * np.pi * 8 * t)
    jittert = (
        0.25
        * (
            np.sin(2 * np.pi * 440 * t + fm)
            + np.sin(2 * np.pi * 880 * t + fm)
            + np.sin(2 * np.pi * 1320 * t + fm)
            + np.sin(2 * np.pi * 1760 * t + fm)
        )
    ).astype(np.float32)
    assert wz.jitter_kanal(jittert, SR) > wz.jitter_kanal(clean, SR) + 0.02


def test_jitter_kanal_stille_ist_null() -> None:
    """Stille-Guard: ohne Signal-Energie ist Jitter nicht definiert ⇒ 0,0."""
    assert wz.jitter_kanal(np.zeros(8192, dtype=np.float32), SR) == 0.0


# ---------------------------------------------------------------------------
# DRT-10-Produktions-Verdikt (Spec 26 DRT-10, 2026-11-17)
# ---------------------------------------------------------------------------


def test_ueberrestauration_flag_drt10(monkeypatch) -> None:
    """Überrestauration = defektlast-Gewinn ≥ JND bei musik_mos-Verlust ≥ JND."""
    x = _musik()
    y = _musik(seed=7)
    zaehler: dict[str, int] = {"last": 0, "mos": 0}

    def _fake_last(a: np.ndarray, sr: int) -> float:
        zaehler["last"] += 1
        return 1.0 if zaehler["last"] == 1 else 0.8  # Δ −0,2 ≤ −JND (Gewinn)

    def _fake_mos(a: np.ndarray, sr: int) -> float:
        zaehler["mos"] += 1
        return 3.0 if zaehler["mos"] == 1 else 2.8  # Δ −0,2 ≤ −JND (Verlust)

    monkeypatch.setattr(wz, "defektlast", _fake_last)
    monkeypatch.setattr(wz, "_musik_mos", _fake_mos)

    befund = wz.messe_wohlklang_zeuge(x, y, SR)

    assert befund["ueberrestauration"] is True
    assert befund["gerichteter_fall"] is True
    assert any("UEBERRESTAURATION" in g for g in befund["grund"])


def _befund_ueberrestauration() -> dict:
    return {"verfuegbar": True, "ueberrestauration": True}


def test_drt10_verdict_degradiert_bei_ueberrestauration() -> None:
    """M1: Produktions-Invariante — Überrestauration degradiert JEDEN Lauf."""
    meta: dict = {"wohlklang_zeuge": _befund_ueberrestauration()}

    assert wz.drt10_verdict_in_metadata(meta) is True
    assert meta["degradation_status"] == "degraded"
    assert meta["fail_reason"] == wz.DRT10_FAIL_REASON
    assert any(r["error_code"] == wz.DRT10_FAIL_REASON for r in meta["fail_reasons"])


def test_drt10_verdict_ohne_ueberrestauration_kein_eingriff() -> None:
    meta: dict = {"wohlklang_zeuge": {"verfuegbar": True, "ueberrestauration": False}}

    assert wz.drt10_verdict_in_metadata(meta) is False
    assert "degradation_status" not in meta


def test_drt10_verdict_unverfuegbarer_befund_kein_eingriff() -> None:
    meta: dict = {"wohlklang_zeuge": {"verfuegbar": False}}

    assert wz.drt10_verdict_in_metadata(meta) is False
    assert "degradation_status" not in meta


def test_drt10_verdict_idempotent() -> None:
    meta: dict = {"wohlklang_zeuge": _befund_ueberrestauration()}
    wz.drt10_verdict_in_metadata(meta)
    assert wz.drt10_verdict_in_metadata(meta) is True
    codes = [r["error_code"] for r in meta["fail_reasons"] if isinstance(r, dict)]
    assert codes.count(wz.DRT10_FAIL_REASON) == 1


def test_drt10_verdrahtung_drei_pfade() -> None:
    """Wirkungs-Pin: Verdikt + Identitäts-Rollback in Einzel-, Options- und Chunked-Pfad."""
    quelle = (PROJEKT / "backend/core/unified_restorer_v3.py").read_text(encoding="utf-8")

    assert quelle.count("self._drt10_verdict_in_metadata(") == 3
    assert quelle.count("self._drt10_rollback_zulaessig(") == 3
    assert quelle.count('"drt10_rollback"') >= 3
    assert "_drt10_kandidat_aktiv" not in quelle  # M1: Kandidaten-Gate entfernt
