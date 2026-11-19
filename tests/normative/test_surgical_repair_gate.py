import pytest

"""§2.59.11: Garantiert dass chirurgische Zonen NIEMALS den gesamten Song umfassen."""
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))


@pytest.mark.unit
def test_surgical_zones_are_localized():
    """Stellt sicher dass Zonen aus echten Locations kommen, nicht aus Placeholdern."""
    from backend.core.surgical_defect_analyzer import SurgicalDefectAnalyzer

    analyzer = SurgicalDefectAnalyzer()

    # Test 1: Mit echten Locations → kleine Zonen
    zones = analyzer.analyze(
        defect_scores={"clicks": 0.8, "crackle": 0.5, "transport_bump": 0.6},
        audio_duration_s=225.0,
        defect_locations={
            "clicks": [(1.2, 1.21), (5.6, 5.61)],
            "transport_bump": [(30.0, 30.15)],
        },
    )
    assert len(zones) == 3, f"Erwartet 3 Zonen, bekam {len(zones)}"
    for z in zones:
        dur_s = z.end_s - z.start_s
        assert dur_s < 1.0, f"Zone {z.defect_type} ist {dur_s * 1000:.0f}ms — maximal 1s erlaubt. Placeholder-Verdacht!"
        assert z.start_s > 0 or z.end_s < 225.0, (
            f"Zone {z.defect_type} spannt gesamten Song ({z.start_s}–{z.end_s}) — das ist ein Placeholder!"
        )

    # Test 2: Ohne Locations → KEINE Zonen
    zones_no_loc = analyzer.analyze(
        defect_scores={"clicks": 0.8, "wow": 0.9},
        audio_duration_s=225.0,
        defect_locations={},
    )
    assert len(zones_no_loc) == 0, f"Ohne Locations dürfen KEINE Zonen erstellt werden, bekam {len(zones_no_loc)}"

    # Test 3: Placeholder-Locations (>50% des Songs) → ignoriert
    zones_placeholder = analyzer.analyze(
        defect_scores={"modulation_noise": 0.7},
        audio_duration_s=225.0,
        defect_locations={"modulation_noise": [(0.0, 225.0)]},
    )
    assert len(zones_placeholder) == 0, (
        f"Placeholder-Zonen (0–225s) müssen ignoriert werden, bekam {len(zones_placeholder)}"
    )


def test_surgical_repair_accepts_short_zones():
    """Stellt sicher dass der SurgicalRepair kurze Events nicht ablehnt."""
    import numpy as np

    from backend.core.surgical_repair import DefectInstance, SurgicalRepair, _repair_clicks

    surgeon = SurgicalRepair(sr=48000)
    audio = np.random.randn(2, 48000).astype(np.float32) * 0.1  # 1s Stereo

    # Simuliere einen 2ms Click — muss reparierbar sein
    instances = [DefectInstance(0.5, 0.502, "clicks", 0.8)]
    result = surgeon.repair(audio, instances, phase_fn=_repair_clicks)

    assert result.zones_repaired == 1, (
        f"2ms Click muss repariert werden, "
        f"wurde aber übersprungen (repaired={result.zones_repaired}, "
        f"skipped={result.zones_skipped})"
    )
    assert result.zones_skipped == 0
    # Audio muss gleiche Shape haben
    assert result.audio.shape == audio.shape
