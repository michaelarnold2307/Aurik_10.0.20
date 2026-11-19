"""Vertragstests für scripts/verify_m9_gate.py (SOTA-M9-Gate, 2026-11-14).

Modell-frei: die Gate-Logik (Never-worsen, Vorteil, Determinismus, fail-closed)
wird ohne ONNX-Inferenz gepinnt. Kein Flag-Flip — das Skript darf die
Flag-Datei weder importieren noch schreiben (Quelltext-Pin, §V7 (copilot-instructions.md)).
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts import verify_m9_gate as gate

# ---------------------------------------------------------------------------
# 1) Determinismus (§G5 (copilot-instructions.md))
# ---------------------------------------------------------------------------


def test_track_auswahl_deterministisch(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Gleicher Seed ⇒ identische, sortierte Track-Auswahl."""
    monkeypatch.setattr(gate, "MUSDB_TEST", tmp_path)
    for i in range(6):
        (tmp_path / f"t{i:02d}").mkdir()
        (tmp_path / f"t{i:02d}" / "mixture.wav").write_bytes(b"x")
    a = gate._waehle_tracks(3, 42)
    b = gate._waehle_tracks(3, 42)
    assert a == b
    assert a == sorted(a)
    assert len(a) == 3


def test_rausch_arm_deterministisch() -> None:
    """Gleicher Seed ⇒ bit-identisches Rausch-Signal; anderer Seed ⇒ anders."""
    x = np.zeros(44100, dtype=np.float32)
    a = gate._rausch_arm(x, 42)
    b = gate._rausch_arm(x, 42)
    c = gate._rausch_arm(x, 43)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


# ---------------------------------------------------------------------------
# 2) Gate-Logik (Never-worsen + Vorteil, fail-closed auf applied=False)
# ---------------------------------------------------------------------------


def _messung(**kwargs) -> gate.Messung:
    werte = {
        "track": "t",
        "arm": "clean",
        "applied": True,
        "mos_basis": 4.0,
        "mos_m9": 4.0,
        "defekt_basis": 0.3,
        "defekt_m9": 0.3,
        "hnr_basis": 20.0,
        "hnr_m9": 20.0,
    }
    werte.update(kwargs)
    return gate.Messung(**werte)


def test_never_worsen_grenzen() -> None:
    assert gate._never_worsen_ok(_messung())  # identisch ⇒ OK
    assert gate._never_worsen_ok(_messung(mos_m9=4.0 - gate.JND))  # exakt JND ⇒ OK
    assert not gate._never_worsen_ok(_messung(mos_m9=4.0 - gate.JND - 1e-4))
    assert not gate._never_worsen_ok(_messung(defekt_m9=0.3 + gate.JND + 1e-4))
    assert not gate._never_worsen_ok(_messung(hnr_m9=20.0 - gate.HNR_TOLERANZ_DB - 1e-4))


def test_vorteil_schwellen() -> None:
    assert not gate._vorteil(_messung())  # identisch ⇒ kein Vorteil
    assert gate._vorteil(_messung(hnr_m9=20.0 + gate.HNR_TOLERANZ_DB + 1e-3))
    assert gate._vorteil(_messung(mos_m9=4.0 + gate.JND + 1e-4))
    assert gate._vorteil(_messung(defekt_m9=0.3 - gate.JND - 1e-4))


def test_applied_false_ist_immer_fail() -> None:
    """Ein nicht angewandter Kandidat kann weder schützen noch gewinnen."""
    assert not gate._never_worsen_ok(_messung(applied=False))
    assert not gate._vorteil(_messung(applied=False))


# ---------------------------------------------------------------------------
# 3) Fail-closed-Pfade + Vertrag
# ---------------------------------------------------------------------------


def test_schema_pin() -> None:
    assert gate.SCHEMA == "aurik.m9_gate/1"


def test_main_ohne_daten_fail_closed(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Ohne MUSDB-Testdaten ⇒ Exit 2 (kein stiller Erfolg)."""
    monkeypatch.setattr(gate, "MUSDB_TEST", tmp_path / "gibt_es_nicht")
    assert gate.main(["--tracks", "3"]) == 2


def test_skript_fasst_flags_nicht_an() -> None:
    """Quelltext-Pin: das Gate-Skript importiert/schreibt keine Musik-Flags —
    ein bestandenes Gate ist Voraussetzung für das §4-Verfahren, kein Flip."""
    quelltext = (gate._ROOT / "scripts" / "verify_m9_gate.py").read_text(encoding="utf-8")
    assert "music_model_flags" not in quelltext
    assert "flippe_flag" not in quelltext


def test_denoiser_ist_cpu_only_per_konstruktion() -> None:
    """Quelltext-Pin: die ONNX-Session des M9-Denoisers erzeugt hartkodiert
    den CPUExecutionProvider — das Gate kann die GPU nicht anfassen."""
    quelltext = (gate._ROOT / "backend" / "core" / "dsp" / "tf_gridnet_denoiser.py").read_text(encoding="utf-8")
    assert 'providers=["CPUExecutionProvider"]' in quelltext
