"""Vertragstests für scripts/verify_cantus_gate.py (SOTA-M14-Gate, 2026-11-19).

Modell-frei: die Gate-Logik (Never-worsen, Vorteil, Determinismus, fail-closed)
wird ohne ONNX-Inferenz gepinnt. Kein Flag-Flip — das Skript darf die
Flag-Datei weder importieren noch schreiben (Quelltext-Pin, §V7 (copilot-instructions.md)).
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts import verify_cantus_gate as gate


def _messung(**over: object) -> gate.CantusMessung:
    basis = {
        "arm": "clean",
        "track": "t",
        "hnr_eingang": 20.0,
        "hnr_basis": 20.0,
        "hnr_kandidat": 20.0,
        "mos_eingang": 4.0,
        "mos_basis": 4.0,
        "mos_kandidat": 4.0,
        "defekt_eingang": 0.3,
        "defekt_basis": 0.3,
        "defekt_kandidat": 0.3,
        "basis_applied": True,
        "kandidat_applied": True,
    }
    basis.update(over)
    return gate.CantusMessung(**basis)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 1) Determinismus (§G5 (copilot-instructions.md))
# ---------------------------------------------------------------------------


def test_vocal_und_paar_auswahl_deterministisch(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Gleicher Seed ⇒ identische, sortierte Auswahl (Vocals + degradierte Paare)."""
    musdb = tmp_path / "musdb"
    musdb.mkdir()
    for i in range(6):
        (musdb / f"t{i:02d}").mkdir()
        (musdb / f"t{i:02d}" / "vocals.wav").write_bytes(b"x")
    monkeypatch.setattr(gate, "MUSDB_TEST", musdb)
    real = tmp_path / "real"
    real.mkdir()
    for i in range(8):
        (real / f"song{i}_vinyl_degraded.wav").write_bytes(b"x")
    monkeypatch.setattr(gate, "CANTUS_REAL", real)

    a = gate._waehle_vocal_tracks(3, 42)
    b = gate._waehle_vocal_tracks(3, 42)
    assert a == b and a == sorted(a) and len(a) == 3

    c = gate._waehle_degradierte(3, 42)
    d = gate._waehle_degradierte(3, 42)
    assert c == d and c == sorted(c) and len(c) == 3


# ---------------------------------------------------------------------------
# 2) Gate-Logik (fail-closed)
# ---------------------------------------------------------------------------


def test_never_worsen_grenzen() -> None:
    """JND-Grenzen: mos −JND ok, darunter fail; defektlast +JND ok, darüber fail."""
    ok = _messung()
    assert gate._never_worsen_ok(ok)

    mos_knapp = _messung(mos_kandidat=4.0 - gate.JND)
    assert gate._never_worsen_ok(mos_knapp)
    mos_zu_tief = _messung(mos_kandidat=4.0 - gate.JND - 1e-3)
    assert not gate._never_worsen_ok(mos_zu_tief)

    defekt_knapp = _messung(defekt_kandidat=0.3 + gate.JND)
    assert gate._never_worsen_ok(defekt_knapp)
    defekt_zu_hoch = _messung(defekt_kandidat=0.3 + gate.JND + 1e-3)
    assert not gate._never_worsen_ok(defekt_zu_hoch)

    hnr_knapp = _messung(hnr_kandidat=20.0 - gate.HNR_TOLERANZ_DB)
    assert gate._never_worsen_ok(hnr_knapp)
    hnr_zu_tief = _messung(hnr_kandidat=20.0 - gate.HNR_TOLERANZ_DB - 1e-3)
    assert not gate._never_worsen_ok(hnr_zu_tief)


def _neutral() -> gate.CantusMessung:
    """Neutraler Kandidat: identische Werte wie die Wiener-Basis (kein Vorteil)."""
    return _messung(
        arm="degradiert",
        hnr_eingang=10.0,
        hnr_basis=10.0,
        hnr_kandidat=10.0,
        mos_eingang=3.0,
        mos_basis=3.0,
        mos_kandidat=3.0,
        defekt_eingang=0.5,
        defekt_basis=0.5,
        defekt_kandidat=0.5,
    )


def test_vorteil_schwellen() -> None:
    """Vorteil = Kandidat schlägt die Wiener-Basis in EINER Metrik über die Schwelle."""
    assert not gate._vorteil(_neutral())

    assert gate._vorteil(_messung(arm="degradiert", hnr_basis=10.0, hnr_kandidat=10.0 + gate.HNR_TOLERANZ_DB + 1e-3))
    assert gate._vorteil(_messung(arm="degradiert", mos_basis=3.0, mos_kandidat=3.0 + gate.JND + 1e-3))
    assert gate._vorteil(_messung(arm="degradiert", defekt_basis=0.5, defekt_kandidat=0.5 - gate.JND - 1e-3))
    # Unter der Schwelle zählt kein Vorteil.
    assert not gate._vorteil(_messung(arm="degradiert", hnr_basis=10.0, hnr_kandidat=10.0 + gate.JND))


def test_applied_false_ist_immer_fail() -> None:
    """Fail-closed: ein nicht angewandter Kandidat besteht weder Never-worsen noch Vorteil."""
    assert not gate._never_worsen_ok(_messung(kandidat_applied=False))
    assert not gate._vorteil(_messung(kandidat_applied=False))


def test_schema_pin() -> None:
    assert gate.SCHEMA == "aurik.cantus_gate/1"


def test_main_ohne_kandidat_fail_closed(tmp_path) -> None:
    """Fehlender Kandidaten-ONNX ⇒ Exit 2 (Messung nicht möglich, kein Gate-Urteil)."""
    assert gate.main(["--modell", str(tmp_path / "fehlt.onnx")]) == 2


def test_skript_fasst_flags_nicht_an() -> None:
    """Kein Flag-Flip: das Skript importiert music_model_flags nie und schreibt keine Flag-Datei."""
    quelle = (gate._ROOT / "scripts" / "verify_cantus_gate.py").read_text(encoding="utf-8")
    assert "music_model_flags" not in quelle
    assert "flippe_flag" not in quelle


def test_kandidat_pfad_ist_cpu_only_per_konstruktion() -> None:
    """CPU-only-Vertrag: expliziter model_path ⇒ kein Torch-ROCm-Kern, kein CUDA-Init im Skript."""
    quelle = (gate._ROOT / "scripts" / "verify_cantus_gate.py").read_text(encoding="utf-8")
    assert "CantusPlugin(model_path=" in quelle
    assert 'to("cuda' not in quelle and ".cuda(" not in quelle and "cuda()" not in quelle
