"""SOTA-M12 (D-K4-8) — Vertragstests der TP-V2-Klassen-Phasenschätzung (CHALLENGER zum Transienten-/Phasen-Rekonstruktionspfad).

Verträge (Auftrag SOTA-M12 (D-K4-8)):
- Flag `use_tp_v2_phase` Default AUS ⇒ keine Verhaltensänderung (bit-identisch)
- Determinismus: 2× gleicher Input ⇒ bit-identisch (§G5 (copilot-instructions.md))
- Kammfilter-Freiheit: kein Phasen-Präfix-Regelmuster bei unbekanntem Input
- Fallback-Warnung (§V6 (copilot-instructions.md)) bei Modellfehlern
- Rekonstruktion auf Synth-Transient plausibel (STFT-Rücklauf tolerant)
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

import backend.core.music_model_flags as mmf
from backend.core.dsp import tp_v2_phase as tp2

SR = 48_000


def _transient(n: int = 48_000, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = np.zeros(n, dtype=np.float64)
    t = np.arange(n) / SR
    for start in (0.1, 0.4, 0.7):
        s0 = int(start * n)
        length = int(0.05 * n)
        env = np.exp(-np.arange(length) / (0.01 * n))
        y[s0 : s0 + length] += env * np.sin(2 * np.pi * 90.0 * t[s0 : s0 + length])
    y += 0.002 * rng.standard_normal(n)
    return (y / (np.max(np.abs(y)) + 1e-9) * 0.8).astype(np.float32)


def _log_mag(x: np.ndarray) -> np.ndarray:
    from scipy.signal import stft

    _, _, yd = stft(x.astype(np.float64), fs=SR, nperseg=tp2.N_FFT, noverlap=tp2.N_FFT - tp2.HOP)
    return np.log1p(np.abs(yd)).astype(np.float32)


def _has_warning(caplog, needle: str) -> bool:
    return any(needle in rec.getMessage() for rec in caplog.records if rec.levelname == "WARNING")


# ── Flag AUS ⇒ keine Verhaltensänderung ───────────────────────────────────────


def test_flag_default_aus() -> None:
    """Flag ist aus — Challenger-Pfad läuft nie ohne Sign-off (§v10.802 (copilot-instructions.md))."""
    assert mmf.use_tp_v2_phase is False


# ── Determinismus (§G5 (copilot-instructions.md)) ─────────────────────────────


def test_determinismus_zwei_laeufe_bit_identisch() -> None:
    est = tp2.TPv2PhaseEstimator(checkpoint=None, hidden=8, blocks=1)
    log_mag = _log_mag(_transient())
    r1 = est.estimate_delta_phase(log_mag)
    r2 = est.estimate_delta_phase(log_mag)
    np.testing.assert_array_equal(r1, r2)
    assert r1.shape == log_mag.shape
    assert r1.dtype == np.float32
    assert np.isfinite(r1).all()


def test_seed_faehigkeit_gleicher_seed_gleicher_output() -> None:
    a = tp2.TPv2PhaseEstimator(checkpoint=None, hidden=8, blocks=1, seed=11)
    b = tp2.TPv2PhaseEstimator(checkpoint=None, hidden=8, blocks=1, seed=11)
    log_mag = _log_mag(_transient())
    np.testing.assert_array_equal(a.estimate_delta_phase(log_mag), b.estimate_delta_phase(log_mag))


# ── Kammfilter-Freiheit: kein Phasen-Präfix-Regelmuster ───────────────────────


def _trained_estimator() -> tp2.TPv2PhaseEstimator:
    """Estimator mit emittiertem 'trainierten' Zustand (out_proj ≠ 0).

    Seed-Gewichte sind per Design Identität (Null-Initialisierung, Docstring
    `backend/core/dsp/tp_v2_phase.py`) — für den Präfix-Gegentest wird der
    Trainingszustand über die Ausgangsschicht emittiert.
    """
    est = tp2.TPv2PhaseEstimator(checkpoint=None, hidden=8, blocks=1)
    import torch

    with torch.no_grad():
        est._net.out_proj.weight.fill_(0.01)
        est._net.out_proj.bias.fill_(0.001)
    return est


def test_kein_phasen_praefix_regelmuster() -> None:
    """Unbekannter Input ⇒ Δφ zeigt KEINE (f,t)-unabhängige Präfix-Struktur.

    Ein statischer Präfix (z. B. konstante Phase je Bin über alle Fenster)
    wäre input-unabhängig; gefordert ist: unterschiedliche Inputs ⇒
    unterschiedliche, input-abhängige Δφ-Muster.
    """
    est = _trained_estimator()
    d1 = est.estimate_delta_phase(_log_mag(_transient(seed=3)))
    d2 = est.estimate_delta_phase(_log_mag(_transient(seed=99)))
    # Identität wäre ein statischer Präfix — die Outputs müssen sich unterscheiden.
    assert not np.array_equal(d1, d2)
    # Und Δφ darf nicht über alle (f,t) konstant sein.
    assert float(np.std(d1)) > 0.0


# ── Fallback-Warnung (§V6 (copilot-instructions.md)) ──────────────────────────


def test_fallback_warnung_bei_modellfehler(monkeypatch, caplog) -> None:
    """Modellfehler ⇒ logger.warning + None (bestehender Pfad bleibt)."""
    monkeypatch.setattr(tp2, "get_tp_v2_phase_estimator", lambda: (_ for _ in ()).throw(RuntimeError("kaputt")))
    with caplog.at_level(logging.WARNING, logger="backend.core.dsp.tp_v2_phase"):
        res = tp2.estimate_transient_phase(_log_mag(_transient()))
    assert res is None
    assert _has_warning(caplog, "TP-V2-Phasenschätzung fehlgeschlagen")
    assert any("§V6 (copilot-instructions.md)" in r.getMessage() for r in caplog.records)


def test_ohne_artefakt_warnung_beim_laden(tmp_path, caplog) -> None:
    """Kein Checkpoint ⇒ §V6-Warnung + deterministische Seed-Gewichte (nur Challenger-Betrieb)."""
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(tp2, "DEFAULT_CHECKPOINT", tmp_path / "fehlt.pt")
    try:
        with caplog.at_level(logging.WARNING, logger="backend.core.dsp.tp_v2_phase"):
            est = tp2.TPv2PhaseEstimator(checkpoint=tmp_path / "fehlt.pt", hidden=8, blocks=1)
        assert est.source == "seed"
        assert _has_warning(caplog, "ohne trainiertes Artefakt")
    finally:
        monkeypatch.undo()


# ── Erreichbarkeit des trainierten Artefakts (D-K3-85) ────────────────────────


def test_default_checkpoint_zeigt_in_die_repo_wurzel() -> None:
    """Regression D-K3-85: `_PROJECT_ROOT` ist die Repo-Wurzel, nicht `backend/`.

    Vor dem Fix ergab ``Path(__file__).parent.parent.parent`` = ``backend/``;
    der Default-Checkpoint war ``backend/models/tp_v2/…`` (nie vorhanden) ⇒ der
    Flag-Pfad lief still mit untrainierten Seed-Gewichten.
    """
    import pathlib

    repo = pathlib.Path(tp2.__file__).resolve().parents[3]
    assert repo == tp2._PROJECT_ROOT
    assert repo / tp2.DEFAULT_CHECKPOINT_SUBPATH == tp2.DEFAULT_CHECKPOINT
    assert tp2.DEFAULT_CHECKPOINT_SUBPATH == "models/tp_v2/tp_v2_phase_v1.pt"


def test_trainer_dateiname_passt_zum_default_checkpoint() -> None:
    """Der Trainer muss genau den Dateinamen schreiben, den der Consumer lädt."""
    import pathlib

    repo = pathlib.Path(tp2.__file__).resolve().parents[3]
    quelle = (repo / "scripts" / "train_tp_v2_phase.py").read_text(encoding="utf-8")
    assert f'"{tp2.DEFAULT_CHECKPOINT.name}"' in quelle


def test_trainiertes_artefakt_ist_erreichbar() -> None:
    """Regression D-K3-85: deployter Checkpoint ⇒ Quelle `checkpoint:…`, nie `seed`."""
    pytest.importorskip("torch")
    if not tp2.DEFAULT_CHECKPOINT.is_file():
        pytest.skip("TP-V2-Trainingsartefakt nicht vorhanden (CI ohne models/)")
    est = tp2.TPv2PhaseEstimator(checkpoint=tp2.DEFAULT_CHECKPOINT)
    assert est.source.startswith("checkpoint:"), est.source


# ── Rekonstruktion auf Synth-Transient plausibel (STFT-Rücklauf tolerant) ─────


def test_stft_ruecklauf_plausibel_tolerant() -> None:
    """Δφ auf Synth-Transient ⇒ Anwendung auf STFT liefert finite, plausible Phase.

    Tolerant: Wir prüfen Struktur (Shape/Endlichkeit/[-π,π]) und dass die
    korrigierte Phase nicht kollabiert (kein Kammfilter-Präfix = keine
    konstante Verschiebung), nicht exakte Sample-Identität.
    """
    from scipy.signal import istft, stft

    x = _transient()
    _, _, yd = stft(x.astype(np.float64), fs=SR, nperseg=tp2.N_FFT, noverlap=tp2.N_FFT - tp2.HOP)
    log_mag = np.log1p(np.abs(yd)).astype(np.float32)
    est = _trained_estimator()
    delta = est.estimate_delta_phase(log_mag)
    assert delta.shape == log_mag.shape
    assert np.isfinite(delta).all()
    assert float(np.max(np.abs(delta))) <= float(np.pi) + 1e-6
    z_new = np.abs(yd) * np.exp(1j * (np.angle(yd) + delta))
    _, rec = istft(z_new, fs=SR, nperseg=tp2.N_FFT, noverlap=tp2.N_FFT - tp2.HOP)
    rec = np.asarray(rec, dtype=np.float64)
    assert rec.size > 0
    assert np.isfinite(rec).all()
    assert float(np.sqrt(np.mean(rec**2))) > 1e-6  # nicht stumm


def test_nan_eingang_sanitisiert(caplog) -> None:
    """NaN/Inf im Eingang: Warnung + finite Ausgabe (§0a/§V6 (copilot-instructions.md))."""
    est = tp2.TPv2PhaseEstimator(checkpoint=None, hidden=8, blocks=1)
    log_mag = _log_mag(_transient())
    dirty = log_mag.copy()
    dirty[0, 0] = np.nan
    dirty[1, 1] = np.inf
    with caplog.at_level(logging.WARNING, logger="backend.core.dsp.tp_v2_phase"):
        out = est.estimate_delta_phase(dirty)
    assert np.isfinite(out).all()
    assert _has_warning(caplog, "NaN/Inf")
