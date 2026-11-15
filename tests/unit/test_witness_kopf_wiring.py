"""Vertragstests für die Witness-Kopf-Verdrahtung (D-K3-95-Folge, 2026-11-15).

Geprüft: Flag-Semantik (Default AUS = bit-identisch), Produktions-Lesestelle,
fail-closed-Verhalten des DSP-Moduls, Never-worsen-Konstruktion. Modell-frei.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.core import music_model_flags


def test_flag_default_aus() -> None:
    assert music_model_flags.use_witness_kopf_groove_echo is False
    assert music_model_flags.use_witness_kopf_tape_head_level_dip is False
    assert music_model_flags.use_witness_kopf_clicks is False
    assert music_model_flags.use_witness_kopf_compression_artifacts is False
    assert music_model_flags.use_witness_kopf_tape_head_clog is False
    assert music_model_flags.use_witness_kopf_phase_rotation is False
    assert music_model_flags.use_witness_kopf_sibilance is False
    assert music_model_flags.use_witness_kopf_sticky_shed_residue is False
    assert music_model_flags.use_witness_kopf_jitter_artifacts is False
    assert music_model_flags.use_witness_kopf_transport_bump is False


def test_lesestelle_phase_07_rest_koepfe_verdrahtet() -> None:
    """Ausrollung: Phase 07 konsumiert auch die clicks- und compression-Flags."""
    phase = (Path(__file__).resolve().parents[2] / "backend" / "core" / "phases" / "phase_07_declipper.py").read_text(
        encoding="utf-8"
    )
    assert "use_witness_kopf_clicks" in phase
    assert "use_witness_kopf_compression_artifacts" in phase
    assert '"witness_kopf_clicks"' in phase
    assert '"witness_kopf_compression"' in phase
    assert "witness_kopf_clicks_used" in phase
    assert "witness_kopf_compression_used" in phase
    assert "witness_kopf_tape_head_clog_used" in phase
    assert "witness_kopf_phase_rotation_used" in phase
    assert "witness_kopf_sibilance_used" in phase
    assert "witness_kopf_sticky_shed_used" in phase
    assert "witness_kopf_jitter_used" in phase
    assert "witness_kopf_transport_used" in phase


def test_lesestelle_phase_07_tape_head_verdrahtet() -> None:
    """Zweiter Kanal: Phase 07 konsumiert auch das tape_head-Flag und ruft das Modul."""
    phase = (Path(__file__).resolve().parents[2] / "backend" / "core" / "phases" / "phase_07_declipper.py").read_text(
        encoding="utf-8"
    )
    assert "use_witness_kopf_tape_head_level_dip" in phase
    assert "witness_kopf_tape_head_level_dip(" in phase
    assert "witness_kopf_tape_used" in phase
    i_flag = phase.find("if use_witness_kopf_tape_head_level_dip:")
    i_import = phase.find("from backend.core.dsp.witness_kopf_tape_head import")
    assert i_flag > 0 and i_import > i_flag


def test_lesestelle_phase_07_und_modul_verdrahtet() -> None:
    """Produktions-Lesestelle: Phase 07 konsumiert das Flag und ruft das Modul."""
    phase = (Path(__file__).resolve().parents[2] / "backend" / "core" / "phases" / "phase_07_declipper.py").read_text(
        encoding="utf-8"
    )
    assert "use_witness_kopf_groove_echo" in phase
    assert "witness_kopf_groove_echo(" in phase
    assert "witness_kopf_used" in phase
    # Flag-Prüfung steht VOR dem Modul-Import (kein Laden bei AUS).
    i_flag = phase.find("if use_witness_kopf_groove_echo:")
    i_import = phase.find("from backend.core.dsp.witness_kopf_groove_echo import")
    assert i_flag > 0 and i_import > i_flag


def test_modul_fail_closed_ohne_checkpoint(monkeypatch) -> None:
    """Fehlender Checkpoint ⇒ None + Warnung (kein stiller Fehler, §V6 (copilot-instructions.md))."""
    import backend.core.dsp.witness_kopf_groove_echo as wk

    monkeypatch.setattr(wk, "_KOPF_CKPT", Path("/fehlt/nirgendwo.pt"))
    import numpy as np

    audio = np.zeros(44100, dtype=np.float32)
    assert wk.witness_kopf_groove_echo(audio, 44100) is None


def test_modul_nan_inf_schutz_und_never_worsen_pin() -> None:
    """NaN/Inf-Schutz + Never-worsen-Kandidat + 200-ms-Fades sind im Quelltext verankert."""
    import backend.core.dsp.witness_kopf_groove_echo as wk

    quelltext = Path(wk.__file__).read_text(encoding="utf-8")
    assert "nan_to_num" in quelltext
    assert "kandidat" in quelltext
    assert "_FADE" in quelltext  # 200-ms-Soft-Knee-Fade (§III (copilot-instructions.md))


def test_kein_cuda_zwang_und_kein_flag_import_im_modul() -> None:
    import backend.core.dsp.witness_kopf_groove_echo as wk

    quelltext = Path(wk.__file__).read_text(encoding="utf-8")
    assert "cuda()" not in quelltext
    # Flag-Konsum bleibt bei der Phase: kein Import des Flag-Moduls im DSP-Modul
    # (Kommentar-Verweise sind erlaubt, Import wäre ein Flag-Flip-Kanal).
    assert "from backend.core.music_model_flags import" not in quelltext
    assert "import music_model_flags" not in quelltext
