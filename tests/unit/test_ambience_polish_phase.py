"""Tests für die Pipeline-Verdrahtung der Ambience-Politur (Spec 25, Punkt 4).

Abgedeckt:
- Phase-Metadaten + Intervention-Registry (Familie gesetzt)
- Materialprofil-Schalter: AN für Vintage-Klassen (Schellack/Kassette/Tape,
  auch als MaterialType-Enum), AUS für „lebendiges“ Material (CD) —
  dann bit-identischer Passthrough
- Strength-Contract: strength=0 => Passthrough (disabled)
- Deterministische Phase (zweimal gleicher Aufruf => identisches Audio)
- DAG-Ordnung: phase_47 → phase_ambience_polish → phase_glue_stage

Testbasis: echtes Audio (tests/real_world_validation/test_library).

Spec: .github/specs/25_ambience_match_plugin.md
"""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from backend.core.defect_scanner import MaterialType
from backend.core.phase_dag import validate_phase_order
from backend.core.phases.phase_ambience_polish import (
    AmbiencePolishPhase,
    _is_vintage_material,
)
from backend.core.unified_restorer_v3 import UnifiedRestorerV3

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REAL_AUDIO = _REPO_ROOT / "tests/real_world_validation/test_library/digital/digital_test_01.wav"
SR = 44100
N_SECONDS = 3  # ein Synthese-Chunk — hält die Tests schnell


@pytest.fixture(scope="module")
def signal() -> np.ndarray:
    """Echtes Audio: digital_test_01.wav (erste 3 s, 44,1 kHz, mono)."""
    if not _REAL_AUDIO.exists():
        pytest.skip("tests/real_world_validation/test_library fehlt (nicht im CI-Checkout)")
    x_raw, sr = sf.read(str(_REAL_AUDIO), dtype="float32")
    assert sr == SR
    x = np.asarray(x_raw, dtype=np.float32)
    return x[: N_SECONDS * SR]


def test_registry_covers_ambience_polish() -> None:
    registry = UnifiedRestorerV3.get_phase_intervention_registry()
    assert "phase_ambience_polish" in registry
    assert registry["phase_ambience_polish"] == "stereo_enhancement"


def test_phase_metadata() -> None:
    meta = AmbiencePolishPhase().get_metadata()
    assert meta.phase_id == "phase_ambience_polish"


def test_vintage_material_keys() -> None:
    for key in ("shellac", "cassette", "tape", "reel_tape", "wire_recording"):
        assert _is_vintage_material(key)
    for key in ("vinyl", "cd_digital", "mp3_low", "streaming", "unknown"):
        assert not _is_vintage_material(key)
    # Legacy-Schreibweisen
    assert _is_vintage_material("Schellack")
    assert _is_vintage_material("Kassette")
    assert _is_vintage_material("Tonband")


@pytest.mark.parametrize(
    "material",
    ["shellac", "cassette", "tape", MaterialType.CASSETTE, MaterialType.SHELLAC],
)
def test_enabled_for_vintage_material(signal: np.ndarray, material) -> None:
    phase = AmbiencePolishPhase()
    result = phase.process(signal, SR, material_type=material)
    assert result.metadata["ambience_polish"] == "applied"
    out = np.asarray(result.audio, dtype=np.float32)
    assert out.shape == signal.shape
    assert np.all(np.isfinite(out))
    assert float(np.max(np.abs(out - signal))) > 1e-6  # Hülle wurde addiert


def test_disabled_for_living_material_passthrough(signal: np.ndarray) -> None:
    phase = AmbiencePolishPhase()
    result = phase.process(signal, SR, material_type="cd_digital")
    assert result.metadata["ambience_polish"] == "disabled"
    out = np.asarray(result.audio, dtype=np.float32)
    assert out is not None
    assert np.array_equal(out, signal)  # bit-identischer Passthrough


def test_explicit_enable_overrides_material(signal: np.ndarray) -> None:
    phase = AmbiencePolishPhase()
    result = phase.process(signal, SR, material_type="cd_digital", ambience_polish_enabled=True)
    assert result.metadata["ambience_polish"] == "applied"


def test_strength_zero_disables(signal: np.ndarray) -> None:
    phase = AmbiencePolishPhase()
    result = phase.process(signal, SR, material_type="shellac", strength=0.0)
    assert result.metadata["ambience_polish"] == "disabled"
    assert np.array_equal(np.asarray(result.audio), signal)


def test_phase_determinism(signal: np.ndarray) -> None:
    phase = AmbiencePolishPhase()
    r1 = phase.process(signal, SR, material_type="shellac", strength=0.4)
    r2 = phase.process(signal, SR, material_type="shellac", strength=0.4)
    a1 = np.asarray(r1.audio, dtype=np.float32)
    a2 = np.asarray(r2.audio, dtype=np.float32)
    assert np.array_equal(a1, a2)  # bit-identisch (§G5 (copilot-instructions.md))


def test_dag_orders_ambience_before_glue() -> None:
    ordered = [
        "phase_40_loudness_normalization",
        "phase_47_truepeak_limiter",
        "phase_ambience_polish",
        "phase_glue_stage",
        "phase_41_output_format_optimization",
    ]
    assert validate_phase_order(ordered) == []

    # Studio-Modus: Mastering-Polish vor der Politur, nie dazwischen.
    ordered_studio = [
        "phase_17_mastering_polish",
        "phase_40_loudness_normalization",
        "phase_47_truepeak_limiter",
        "phase_ambience_polish",
        "phase_glue_stage",
    ]
    assert validate_phase_order(ordered_studio) == []

    wrong = [
        "phase_40_loudness_normalization",
        "phase_47_truepeak_limiter",
        "phase_glue_stage",
        "phase_ambience_polish",
        "phase_41_output_format_optimization",
    ]
    assert validate_phase_order(wrong)  # Verletzung gemeldet
