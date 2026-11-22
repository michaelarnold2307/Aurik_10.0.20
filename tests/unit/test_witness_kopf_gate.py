"""Vertragstests für das Witness-Kopf-Gate (D-K3-95, 2026-11-15). Modell-frei."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import scripts.verify_witness_kopf_gate as gate


def test_schwellen_pin() -> None:
    assert gate.Q_MIN == 0.5
    assert gate.NW_TOL == 0.02
    assert gate.NW_ANTEIL_MIN == 0.95
    assert gate.SCHEMA == "aurik.witness_kopf_gate/2"


def test_gate_logik_beide_arme() -> None:
    """Gate = Median-Q ≥ 0,5 UND Never-worsen-Anteil ≥ 0,95 — beide Bedingungen."""
    assert gate.messe is not None
    gut = {"q_median": 0.6, "never_worsen_anteil": 0.98}
    assert gut["q_median"] >= gate.Q_MIN and gut["never_worsen_anteil"] >= gate.NW_ANTEIL_MIN
    schlecht = {"q_median": 0.6, "never_worsen_anteil": 0.9}
    assert not (schlecht["q_median"] >= gate.Q_MIN and schlecht["never_worsen_anteil"] >= gate.NW_ANTEIL_MIN)


def test_messe_berichtet_rohwerte_fuer_reproduzierbare_aggregate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gate, "_lade_mono", lambda pfad: np.ones(600, dtype=np.float32))
    # Reihenfolge der _p-Aufrufe: p_s, p_c, p_out_s, p_c_ref (clean-eigene RMS),
    # p_out_c (Kopf auf clean-eigener RMS) — DRT-4-Parität des Schutz-Arms.
    messwerte = iter([0.9, 0.1, 0.4, 0.1, 0.1])
    monkeypatch.setattr(gate, "_p", lambda _witness, _signal: next(messwerte))
    result = gate.messe(
        lambda audio: audio,
        object(),
        [("title", Path("damaged.wav"), Path("clean.wav"))],
    )
    assert result["q_values"] == pytest.approx([0.625])
    assert result["never_worsen_flags"] == [True]
    assert result["n_q_bewertbar"] == 1
    assert result["n_never_worsen_bewertbar"] == 1


def test_produktions_semantik_mapping_liefert_exakte_dsp_funktionen() -> None:
    for klasse, (modul_name, funktions_name) in gate.PRODUKTIONS_SEMANTIK.items():
        modul = importlib.import_module(f"backend.core.dsp.{modul_name}")
        name, modul_fn, status = gate._lade_produktions_semantik(
            klasse,
            Path(modul._KOPF_CKPT),
            Path(modul._WITNESS_CKPT),
        )
        assert name == modul_name
        assert modul_fn is getattr(modul, funktions_name)
        assert status == "verifiziert"


def test_produktions_semantik_lehnt_fremdes_checkpoint_ab(tmp_path: Path) -> None:
    modul_name, modul_fn, status = gate._lade_produktions_semantik(
        "dropout_head_contact",
        tmp_path / "anderer-kopf.pt",
        tmp_path / "anderer-witness.pt",
    )
    assert modul_name == "witness_kopf_dropout_head"
    assert modul_fn is None
    assert status == "kopf_checkpoint_abweichend"


def test_rohkopf_diagnostik_kann_produktions_gate_nicht_bestehen() -> None:
    result = gate._belege_produktions_semantik({"gate_bestanden": True}, None, None, "nicht_zugeordnet")
    assert result["gate_qualitaet_bestanden"] is True
    assert result["produktions_semantik_verifiziert"] is False
    assert result["schutz_arm_semantik"] == "roher_kopf_diagnostisch"
    assert result["gate_bestanden"] is False


def test_fail_closed_ohne_checkpoint(tmp_path: Path) -> None:
    """Fehlendes Kopf-Checkpoint ⇒ Exit 2 vor jeder Messung."""
    rc = gate.main(["--klasse", "groove_echo", "--checkpoint", str(tmp_path / "fehlt.pt")])
    assert rc == 2


def test_quelltext_pin_kein_flag() -> None:
    quelltext = Path(gate.__file__).read_text(encoding="utf-8")
    assert "music_model_flags" not in quelltext
    assert "flippe_flag" not in quelltext
    assert "cuda()" not in quelltext


# ---------------------------------------------------------------------------
# rms_skip-Parität (D-K3-95-Nacharbeit, 2026-11-22) — Produktions-Konvention
# ---------------------------------------------------------------------------


class _FakeWitnessSkip(torch.nn.Module):
    rms_skip = True


class _FakeWitnessAlt(torch.nn.Module):
    rms_skip = False


def test_p_fuehrt_wahren_pegel_an_rms_skip_witness() -> None:
    """Bei rms_skip-Witness muss das Gate den wahren RAW-Pegel übergeben — nicht
    den konstanten Normalisierungs-Pegel (Befund 2026-11-22)."""
    import torch

    gesehen: dict[str, object] = {}

    class Aufzeichner(_FakeWitnessSkip):
        def forward(self, x, rms_global=None):
            gesehen["rms"] = rms_global.detach().cpu().numpy().tolist()
            return torch.zeros(x.shape[0])

    w = Aufzeichner()
    x = torch.zeros(1, 1, 44100)
    p = gate._p(w, x, raw_db=-5.25)
    assert p == pytest.approx(0.5)  # sigmoid(0)
    assert gesehen["rms"] == pytest.approx([-5.25])


def test_p_alt_ohne_rms_skip_uebergibt_keinen_pegel() -> None:
    """Ohne rms_skip-Witness bleibt der Aufruf ohne rms_global (bit-identisch)."""
    import torch

    gesehen: dict[str, object] = {}

    class Aufzeichner(_FakeWitnessAlt):
        def forward(self, x, rms_global=None):
            gesehen["rms"] = rms_global
            return torch.zeros(x.shape[0])

    w = Aufzeichner()
    x = torch.zeros(1, 1, 44100)
    assert gate._p(w, x) == pytest.approx(0.5)
    assert gesehen["rms"] is None


def test_messe_rms_skip_misst_produktions_konvention(monkeypatch: pytest.MonkeyPatch) -> None:
    """Im rms_skip-Pfad misst das Gate exakt wie die Produktions-Inferenz
    `_witness_p`: clean-Referenz mit EIGENER RMS + wahrem Pegel, Kopf-Ausgang
    denormalisiert und mit eigener RMS renormiert."""
    import torch

    aufrufe: list[tuple[float | None, np.ndarray]] = []
    werte = iter([0.9, 0.1, 0.4, 0.1, 0.1])

    def fake_p(_witness, x, raw_db=None):
        arr = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
        aufrufe.append((raw_db, np.asarray(arr, dtype=np.float64)))
        return next(werte)

    monkeypatch.setattr(gate, "_lade_mono", lambda pfad: np.ones(600, dtype=np.float32))
    monkeypatch.setattr(gate, "_p", fake_p)
    result = gate.messe(
        lambda audio: audio,
        _FakeWitnessSkip(),
        [("title", Path("damaged.wav"), Path("clean.wav"))],
    )
    assert result["q_values"] == pytest.approx([0.625])
    assert result["never_worsen_flags"] == [True]
    assert len(aufrufe) == 5  # p_s, p_c, p_out_s, p_c_ref, p_out_c
    s_w = gate._fenster(np.ones(600, dtype=np.float32))
    db_s = gate._roh_db(s_w)
    # p_s: wahrer Pegel des Schaden-Fensters
    assert aufrufe[0][0] == pytest.approx(db_s, rel=1e-5)
    # p_c: clean mit EIGENER RMS (≈ RMS_ZIEL) + wahrem Pegel
    assert aufrufe[1][0] == pytest.approx(db_s, rel=1e-5)
    assert float(np.sqrt(np.mean(aufrufe[1][1] ** 2))) == pytest.approx(gate.RMS_ZIEL, rel=1e-5)
    # p_out_s: denormalisiert → renormiert (RMS ≈ RMS_ZIEL)
    assert float(np.sqrt(np.mean(aufrufe[2][1] ** 2))) == pytest.approx(gate.RMS_ZIEL, rel=1e-5)
    # p_c_ref (Schutz-Arm): clean-eigene RMS + wahrer Pegel
    assert aufrufe[3][0] == pytest.approx(db_s, rel=1e-5)
    assert float(np.sqrt(np.mean(aufrufe[3][1] ** 2))) == pytest.approx(gate.RMS_ZIEL, rel=1e-5)
