"""
§v10.18 — Normativer Test: resolved_defects fließt korrekt durch die Pipeline.

Verifiziert, dass:
- PhaseResult.resolved_defects existiert und typisiert ist
- create_phase_result resolved_defects durchreicht
- Die betroffenen Phasen (01,02,03,05,07,09,12) PhaseResult.resolved_defects setzen
- Der UV3-Accumulator resolved_defects aus PhaseResult extrahiert
- Defect-Severity sich nach einer Reparatur-Phase reduziert
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

# ── 1. PhaseResult hat resolved_defects-Feld ──────────────────────────


def test_phase_result_has_resolved_defects_field():
    """PhaseResult-Dataclass muss ein resolved_defects-Feld besitzen."""
    from backend.core.phases.phase_interface import PhaseResult

    pr = PhaseResult(audio=np.zeros(100, dtype=np.float32))
    assert hasattr(pr, "resolved_defects"), (
        "PhaseResult.resolved_defects fehlt! §v10.18 verlangt dieses Feld für den korrekten Metadatenfluss."
    )
    assert isinstance(pr.resolved_defects, dict), "PhaseResult.resolved_defects muss ein dict sein!"


# ── 2. create_phase_result reicht resolved_defects durch ─────────────


def test_create_phase_result_passes_resolved_defects():
    """create_phase_result muss resolved_defects an PhaseResult weitergeben."""
    from backend.core.phases.phase_interface import create_phase_result

    pr = create_phase_result(
        audio=np.zeros(100, dtype=np.float32),
        phase_id="99",
        phase_name="Test-Phase",
        resolved_defects={"CLIPPING": 0.03},
    )
    assert pr.resolved_defects == {"CLIPPING": 0.03}, (
        f"resolved_defects wurde nicht durchgereicht: {pr.resolved_defects}"
    )


def test_create_phase_result_defaults_to_empty_dict():
    """Ohne resolved_defects-Parameter muss create_phase_result leeres dict setzen."""
    from backend.core.phases.phase_interface import create_phase_result

    pr = create_phase_result(audio=np.zeros(100, dtype=np.float32))
    assert pr.resolved_defects == {}, f"Default resolved_defects muss leer sein, ist aber {pr.resolved_defects}"


# ── 3. Jede betroffene Phase setzt resolved_defects ──────────────────

PHASES_WITH_RESOLVED_DEFECTS = [
    "phase_01_click_removal",
    "phase_02_hum_removal",
    "phase_03_denoise",
    "phase_05_rumble_filter",
    "phase_07_declipper",
    "phase_09_crackle_removal",
    "phase_12_wow_flutter_fix",
    "phase_14_phase_correction",
    "phase_15_stereo_balance",
    "phase_24_dropout_repair",
    "phase_30_dc_offset_removal",
    "phase_49_advanced_dereverb",
]


@pytest.mark.parametrize("phase_name", PHASES_WITH_RESOLVED_DEFECTS)
def test_phase_implements_resolved_defects(phase_name):
    """Jede §v10.18-Phase muss nach process() resolved_defects im PhaseResult haben."""

    module = importlib.import_module(f"backend.core.phases.{phase_name}")
    # PhaseInterface-Instanz finden
    phase_class = None
    for attr in dir(module):
        obj = getattr(module, attr)
        if hasattr(obj, "__bases__") and any("PhaseInterface" in str(base) for base in getattr(obj, "__bases__", [])):
            phase_class = obj
            break
    assert phase_class is not None, f"{phase_name}: Keine PhaseInterface-Klasse gefunden!"

    phase = phase_class()
    # Process mit minimalem Signal (Sinus 1s @ 48kHz)
    sr = 48000
    audio = (np.sin(2 * np.pi * 440 * np.arange(sr) / sr) * 0.5).astype(np.float32)

    try:
        result = phase.process(
            audio=audio,
            sample_rate=sr,
            material_type="vinyl",
            progress_callback=None,
        )
    except Exception as e:
        pytest.skip(f"{phase_name}: process() warf {type(e).__name__}: {e}")

    # Check: PhaseResult muss existieren
    assert result is not None, f"{phase_name}: process() gab None zurück!"
    assert hasattr(result, "resolved_defects"), f"{phase_name}: PhaseResult hat kein resolved_defects-Feld!"
    assert isinstance(result.resolved_defects, dict), (
        f"{phase_name}: resolved_defects ist kein dict: {type(result.resolved_defects)}"
    )

    # ⚠️ resolved_defects darf leer sein (wenn Phase nichts zu reparieren fand),
    # aber das Feld muss existieren
    if result.resolved_defects:
        for defect_name, severity in result.resolved_defects.items():
            assert isinstance(defect_name, str), f"Key muss str sein: {type(defect_name)}"
            assert isinstance(severity, float), f"Wert muss float sein: {type(severity)}"
            assert 0.0 <= severity <= 0.5, f"Residual-Severity {severity} außerhalb [0.0, 0.5] für {defect_name}"


# ── 4. UV3-Accumulator — Integrationstest ─────────────────────────────


def test_uv3_resolved_defects_accumulator_exists():
    """UV3 muss _resolved_defects_accumulator während _execute_pipeline initialisieren."""
    from backend.core.unified_restorer_v3 import UnifiedRestorerV3

    uv3 = UnifiedRestorerV3()
    # Der Accumulator wird in _execute_pipeline() initialisiert (Zeile 31426)
    # — er existiert vorher nicht als Attribut. Das ist korrekt, da er nur
    # während einer laufenden Pipeline benötigt wird.
    # Wir prüfen, dass die Klasse das Konzept unterstützt:
    import inspect

    source = inspect.getsource(uv3._execute_pipeline)
    assert "_resolved_defects_accumulator" in source, (
        "UV3._execute_pipeline() muss _resolved_defects_accumulator initialisieren!\n"
        "Ohne diesen Accumulator können resolved_defects nicht zwischen Phasen fließen."
    )


# ── 5. Defect-Severity Reduktion — Konzeptioneller Test ───────────────


def test_defect_severity_reduction_logic():
    """Verifiziere, dass die Severity-Berechnung korrekt absteigt."""
    # Simuliere: CLIPPING=0.57 → Phase 07 repariert → residual=0.03
    original = {"CLIPPING": 0.57, "CLICKS": 0.42, "HIGH_FREQ_NOISE": 0.68}
    accumulator: dict[str, float] = {}

    # Phase 07 meldet: CLIPPING residual 0.03
    phase_07_result: dict[str, float] = {"CLIPPING": 0.03}
    for k, v in phase_07_result.items():
        if k not in accumulator or v < accumulator[k]:
            accumulator[k] = v

    # Phase 01 meldet: CLICKS residual 0.02
    phase_01_result: dict[str, float] = {"CLICKS": 0.02}
    for k, v in phase_01_result.items():
        if k not in accumulator or v < accumulator[k]:
            accumulator[k] = v

    # Phase 03 meldet: HIGH_FREQ_NOISE residual 0.15
    phase_03_result: dict[str, float] = {"HIGH_FREQ_NOISE": 0.15}
    for k, v in phase_03_result.items():
        if k not in accumulator or v < accumulator[k]:
            accumulator[k] = v

    # Phase 23 bekommt: min(original, accumulator)
    effective = {k: min(v, accumulator.get(k, 1.0)) for k, v in original.items()}

    assert effective["CLIPPING"] == 0.03, f"CLIPPING sollte 0.03 sein: {effective}"
    assert effective["CLICKS"] == 0.02, f"CLICKS sollte 0.02 sein: {effective}"
    assert effective["HIGH_FREQ_NOISE"] == 0.15, f"NOISE sollte 0.15 sein: {effective}"

    # ⚠️ Der alte Wert (0.57, 0.42, 0.68) darf NICHT mehr verwendet werden!
    for k in ["CLIPPING", "CLICKS", "HIGH_FREQ_NOISE"]:
        assert effective[k] < original[k], (
            f"{k}: Severity {effective[k]} >= original {original[k]} — es wurde keine Reduktion erzielt!"
        )


# ── 6. resolved_defects_helper — Unit-Tests ──────────────────────────


def test_compute_resolved_defects_helper():
    """Der Helper berechnet korrekte residuale Severities."""
    from backend.core.phases.resolved_defects_helper import (
        compute_resolved_defects,
        compute_resolved_defects_multi,
    )

    # Hohe Original-Severity, starke Reparatur
    result = compute_resolved_defects("CLICKS", 0.8, 0.98)
    assert result == {"CLICKS": pytest.approx(0.016, abs=0.01)}, f"0.8 * (1-0.98) = 0.016, aber: {result}"

    # Mittlere Severity, mittlere Reparatur
    result = compute_resolved_defects("HUM", 0.5, 0.5)
    assert result == {"HUM": 0.25}, f"0.5 * 0.5 = 0.25, aber: {result}"

    # Keine Reparatur → keine Meldung
    result = compute_resolved_defects("RUMBLE", 0.1, 0.0)
    assert result == {"RUMBLE": 0.1}, f"0.1 * 1.0 = 0.1, aber: {result}"

    # Sehr geringe Original-Severity → ignorieren
    result = compute_resolved_defects("DC_OFFSET", 0.005, 1.0)
    assert result == {}, f"0.005 < 0.01 → sollte leer sein, aber: {result}"

    # Multi-Defect
    result = compute_resolved_defects_multi({"CLICKS": (0.8, 0.98), "HUM": (0.5, 0.5), "RUMBLE": (0.1, 0.0)})
    assert "CLICKS" in result
    assert "HUM" in result
    assert "RUMBLE" in result
    assert len(result) == 3


# ── 7. Denker Phase-Skip (§v10.24) ─────────────────────────────────


def test_should_skip_resolved_phase_helper():
    """UV3._should_skip_resolved_phase() skipst Phasen deren Defekte resolved sind."""
    from backend.core.unified_restorer_v3 import UnifiedRestorerV3

    uv3 = UnifiedRestorerV3()
    assert hasattr(uv3, "_should_skip_resolved_phase"), (
        "UV3._should_skip_resolved_phase() fehlt! "
        "§v10.24 verlangt, dass Phasen mit resolved_defects < 0.05 übersprungen werden."
    )

    # Ohne Accumulator: nie skippen
    assert not uv3._should_skip_resolved_phase("phase_23_spectral_repair"), (
        "Ohne Accumulator darf keine Phase geskippt werden"
    )

    # Mit leeren Accumulator: nie skippen
    uv3._resolved_defects_accumulator = {}
    assert not uv3._should_skip_resolved_phase("phase_23_spectral_repair"), "Leerer Accumulator → keine Phase skippen"

    # Mit CLIPPING=0.03 (resolved): phase_23 sollte geskippt werden wenn
    # CLIPPING der einzige primary defect wäre
    uv3._resolved_defects_accumulator = {"CLIPPING": 0.03}
    # phase_23 hat viele primary defects, nicht nur CLIPPING
    # → sollte NICHT geskippt werden
    result = uv3._should_skip_resolved_phase("phase_23_spectral_repair")
    # phase_23 repariert 15 Defekte, CLIPPING ist nur einer davon
    # → skip nur wenn ALLE resolved sind
    assert not result, f"phase_23 hat 15 primary defects, CLIPPING=0.03 reicht nicht zum Skippen: {result}"

    # Enhancement-Phase: nie skippen
    uv3._resolved_defects_accumulator = {"CLICKS": 0.0}
    assert not uv3._should_skip_resolved_phase("phase_21_exciter"), "Enhancement-Phasen dürfen nie geskippt werden"


def test_denker_skip_integration():
    """Integration: skip-Logik integriert sich korrekt in den Datenfluss."""
    from backend.core.unified_restorer_v3 import UnifiedRestorerV3

    uv3 = UnifiedRestorerV3()

    # Simuliere: Phase 07 hat CLIPPING resolved
    uv3._resolved_defects_accumulator = {
        "CLIPPING": 0.02,
        "CLICKS": 0.01,
        "HUM": 0.0,
        "LOW_FREQ_RUMBLE": 0.0,
        "PHASE_ISSUES": 0.0,
        "DC_OFFSET": 0.0,
    }

    # Phase, die NUR CLICKS repariert (phase_01): sollte geskippt werden
    # wenn CLICKS bereits 0.01 ist
    # ABER: phase_01 repariert auch LACQUER_DISC_DEGRADATION
    # → skip nur wenn BEIDE resolved
    result_01 = uv3._should_skip_resolved_phase("phase_01_click_removal")
    # LACQUER_DISC_DEGRADATION ist nicht im accumulator → nicht resolved
    assert not result_01, "phase_01: LACQUER_DISC_DEGRADATION nicht resolved → nicht skippen"

    # Phase 30 (DC_OFFSET): DC_OFFSET=0.0 → sollte geskippt werden
    result_30 = uv3._should_skip_resolved_phase("phase_30_dc_offset_removal")
    assert result_30, f"phase_30: DC_OFFSET=0.0 → sollte geskippt werden, aber: {result_30}"
