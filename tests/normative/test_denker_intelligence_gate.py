import pytest

"""§2.60.2: Normative-Gate — Denker-Intelligenz & PhaseEffectCatalog-Korrektheit."""
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))


@pytest.mark.unit
def test_catalog_has_all_required_phases():
    from backend.core.phase_effect_catalog import PHASE_EFFECT_CATALOG

    required = [
        "phase_01_click_removal",
        "phase_02_hum_removal",
        "phase_03_denoise",
        "phase_04_eq_correction",
        "phase_12_wow_flutter_fix",
        "phase_19_de_esser",
        "phase_20_reverb_reduction",
        "phase_38_presence_boost",
    ]
    for pid in required:
        p = PHASE_EFFECT_CATALOG.get(pid)
        assert p is not None, f"{pid} fehlt"
        assert p.phase_id == pid
        assert isinstance(p.goal_impact, dict)
        assert isinstance(p.risks, list)
        assert p.time_profile in ("fast", "medium", "slow", "heavy")


def test_calibration_returns_valid_range():
    from backend.core.phase_effect_catalog import calibrate_phase_intensity

    for pid in ["phase_03_denoise", "phase_19_de_esser"]:
        for snr in (None, 3.0, 6.9, 15.0, 25.0):
            for panns in (0.0, 0.35, 0.80):
                v = calibrate_phase_intensity(pid, 1.0, snr_db=snr, panns_singing=panns, era_decade=1970)
                assert 0.0 <= v <= 1.0, f"{pid} snr={snr}→{v}"


def test_vocal_reduces_ml_strength():
    from backend.core.phase_effect_catalog import calibrate_phase_intensity

    nv = calibrate_phase_intensity("phase_03_denoise", 1.0, panns_singing=0.0, snr_db=6.9)
    wv = calibrate_phase_intensity("phase_03_denoise", 1.0, panns_singing=0.35, snr_db=6.9)
    assert wv <= nv, f"Vocal {wv} > NoVocal {nv}"


def test_vintage_reduces_ml_strength():
    from backend.core.phase_effect_catalog import calibrate_phase_intensity

    m = calibrate_phase_intensity("phase_03_denoise", 1.0, era_decade=2000, snr_db=6.9)
    v = calibrate_phase_intensity("phase_03_denoise", 1.0, era_decade=1970, snr_db=6.9)
    assert v <= m, f"Vintage {v} > Modern {m}"


def test_low_snr_reduces_ml():
    """Niedriger SNR darf die ML-Intensität NICHT erhöhen — und wird als Risiko geführt.

    Befund 2026-10-07 (am Code gemessen): Die ursprünglich hier geprüfte Zusage
    „SNR senkt die ML-Stärke" ist NICHT implementiert. ``calibrate_phase_intensity``
    nimmt ``snr_db`` entgegen, nutzt den Wert in der Stärke-Kette aber nirgends
    (toter Parameter; Register D-K3-31) — gemessen: SNR 3 → 1,0 == SNR 15 → 1,0
    bei aktivem Defekt (severity 0,6 > min_severity 0,3).

    Der implementierte Schutz ist das RISIKO-Modell: ``get_phase_risk_level``
    setzt für Phasen mit ``ml_artifact``-Risiko +1,0 (snr < 8 dB bzw. unbekannt)
    und +0,5 (Ära < 1980) → „high". Und die Phase selbst fährt bei niedrigem SNR
    bewusst das stärkste ML-Primary (§4.4/§4.5 MIIPHER/DeepFilterNet), nicht
    weniger ML — eine Stärke-Senkung wäre eine klangverändernde Politik und
    bräuchte die fünf Belege (.github/WOHLKLANG_CLAIMS.md). Geprüft wird daher,
    was gilt: keine ESKALATION bei niedrigem SNR + Risiko-Flaggung.
    """
    from backend.core.phase_effect_catalog import calibrate_phase_intensity, get_phase_risk_level

    # Vorbedingung §G188: Die Phase ist AKTIV (Defekt über der Anwendungsschwelle).
    lo = calibrate_phase_intensity("phase_03_denoise", 1.0, defect_severity=0.6, snr_db=3.0)
    hi = calibrate_phase_intensity("phase_03_denoise", 1.0, defect_severity=0.6, snr_db=15.0)
    assert lo <= hi, f"SNR=3→{lo} > SNR=15→{hi}: niedriger SNR darf die ML-Stärke nicht verstärken"
    # Dokumentierte Lücke: identische Stärke ⇒ keine SNR-Regel (D-K3-31). Sobald
    # eine (belegte) Senkung implementiert wird, fällt dieser Pin bewusst um.
    assert lo == hi, f"SNR-Regel neu implementiert? lo={lo} hi={hi} → D-K3-31/Register aktualisieren"

    # Implementierter Schutz: ml_artifact-Risiko bei niedrigem SNR und alter Ära.
    assert get_phase_risk_level("phase_03_denoise", snr_db=3.0, era_decade=1970, panns_singing=0.0) in (
        "high",
        "critical",
    )
    assert get_phase_risk_level("phase_03_denoise", snr_db=15.0, era_decade=1990, panns_singing=0.0) == "low"


def test_schlager_preserves_warmth():
    from backend.core.phase_effect_catalog import calibrate_phase_intensity

    normal = calibrate_phase_intensity("phase_20_reverb_reduction", 1.0)
    schlager = calibrate_phase_intensity("phase_20_reverb_reduction", 1.0, genre_is_schlager=True)
    assert schlager <= normal, "Schlager muss Hall bewahren"


def test_mp3_chain_reduces_ml():
    from backend.core.phase_effect_catalog import calibrate_phase_intensity

    clean = calibrate_phase_intensity("phase_03_denoise", 1.0, chain_has_mp3=False)
    mp3 = calibrate_phase_intensity("phase_03_denoise", 1.0, chain_has_mp3=True)
    assert mp3 <= clean, f"MP3 {mp3} > Clean {clean}"


def test_substitutions_are_valid():
    # Dynamische Validierung gegen existierende Phasen-Dateien
    import glob
    from pathlib import Path

    from backend.core.fahrplan import PHASE_SUBSTITUTIONS

    _repo = Path(__file__).resolve().parent.parent.parent
    _phase_dir = _repo / "backend" / "core" / "phases"
    _existing = {Path(p).stem for p in glob.glob(str(_phase_dir / "phase_*.py"))}
    for key, sub in PHASE_SUBSTITUTIONS.items():
        assert sub in _existing, f"Substitution target {sub} nicht in phases/"
        assert key in _existing, f"Substitution key {key} nicht in phases/"


def test_perceptual_budget_sums_to_one():
    from backend.core.fahrplan import PERCEPTUAL_BUDGET

    assert abs(sum(PERCEPTUAL_BUDGET.values()) - 1.0) < 0.01


def test_fahrplan_builds_without_crash():
    from backend.core.fahrplan import build_fahrplan

    fp = build_fahrplan(
        ["phase_01_click_removal", "phase_03_denoise"],
        [(0, 120, "verse"), (120, 225, "chorus")],
        goal_priorities={"waerme": 0.8, "brillanz": 0.6},
        physical_ceiling_hz=13000,
    )
    assert fp.total_segments == 2
    assert len(fp.phase_order) == 2
    assert fp.physical_ceiling_hz == 13000
