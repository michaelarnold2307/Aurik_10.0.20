from __future__ import annotations

"""
tests/unit/test_musikalischer_globalplan.py
==========================================

Unit-Tests für den MusikalischerGlobalplan-Dienst (§Dach-Layer).

Prüft:
- Singleton-Verhalten (Thread-sicher)
- Rückgabe-Datenklassen (Portrait + Plan)
- DSP-Fallback ohne EraClassifier / GermanSchlagerClassifier
- Phase-Adjustments für alle bekannten Phasen
- NaN/Inf-Robustheit
- Mono- und Stereo-Input
- Era-Profil-Mapping (Dekaden-Snap)
- Genre-Modifikatoren
- Plausibilitätsbereiche aller float-Felder
- Sinnhaftigkeit der Cross-Phase-Koordination (NR vs. Multiband)
"""


import threading

import numpy as np
import pytest

from backend.core.musikalischer_globalplan import (
    _ERA_PROFILES,
    _GENRE_MODIFIERS,
    MusikalischerGlobalplanDienst,
    MusikalischesPortrait,
    StilbewussterRestaurierungsplan,
    _build_semantic_description,
    _estimate_brightness,
    _estimate_dynamic_range,
    _estimate_mood,
    _estimate_warmth,
    _nearest_era_profile,
    erstelle_globalplan,
    get_musikalischer_globalplan_dienst,
)

SR = 48000

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mono_sine():
    """1 Hz Sinus × 3 s bei 48 kHz, Mono."""
    t = np.linspace(0, 3.0, SR * 3, endpoint=False, dtype=np.float32)
    return (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


@pytest.fixture
def stereo_sine(mono_sine):
    """Stereo-Version des Sinus-Signals."""
    return np.stack([mono_sine, mono_sine * 0.9], axis=0)


@pytest.fixture
def noisy_signal():
    """Bandrauschen — simuliert alte Aufnahme."""
    rng = np.random.default_rng(42)
    return rng.normal(0, 0.1, SR * 2).astype(np.float32)


@pytest.fixture
def dienst():
    """Frische Instanz (kein Singleton) für Isolation der Tests."""
    return MusikalischerGlobalplanDienst()


# ---------------------------------------------------------------------------
# 1. Singleton
# ---------------------------------------------------------------------------


class TestSingleton:
    def test_singleton_gleiche_instanz(self):
        a = get_musikalischer_globalplan_dienst()
        b = get_musikalischer_globalplan_dienst()
        assert a is b, "Singleton liefert nicht dieselbe Instanz"

    def test_singleton_thread_sicher(self):
        """Parallele Zugriffe aus 10 Threads müssen dieselbe Instanz erhalten."""
        instances = []
        errors = []

        def _get():
            try:
                instances.append(get_musikalischer_globalplan_dienst())
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_get) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread-Fehler: {errors}"
        assert all(i is instances[0] for i in instances), "Nicht alle Instanzen identisch"

    def test_convenience_funktion_liefert_plan(self, mono_sine):
        plan = erstelle_globalplan(mono_sine, SR)
        assert isinstance(plan, StilbewussterRestaurierungsplan)


# ---------------------------------------------------------------------------
# 2. Rückgabe-Typen und Felder
# ---------------------------------------------------------------------------


class TestRueckgabeTypen:
    def test_plan_ist_dataclass(self, dienst, mono_sine):
        plan = dienst.erstelle_plan(mono_sine, SR)
        assert isinstance(plan, StilbewussterRestaurierungsplan)

    def test_portrait_ist_dataclass(self, dienst, mono_sine):
        plan = dienst.erstelle_plan(mono_sine, SR)
        assert isinstance(plan.portrait, MusikalischesPortrait)

    def test_alle_float_felder_im_bereich(self, dienst, mono_sine):
        plan = dienst.erstelle_plan(mono_sine, SR)
        assert 0.0 <= plan.authenticity_target <= 1.0
        assert 0.0 <= plan.warmth_target <= 1.0
        assert 0.0 <= plan.presence_target <= 1.0
        assert 0.0 <= plan.stereo_width_target <= 1.0
        assert 3.0 <= plan.hf_ceiling_khz <= 25.0

    def test_portrait_felder_im_bereich(self, dienst, mono_sine):
        plan = dienst.erstelle_plan(mono_sine, SR)
        p = plan.portrait
        assert 1890 <= p.decade <= 2030
        assert 0.0 <= p.era_confidence <= 1.0
        assert 0.0 <= p.warmth_score <= 1.0
        assert 0.0 <= p.brightness_score <= 1.0
        assert 0.0 <= p.dynamic_range_estimate <= 1.0
        assert p.bpm > 0.0

    def test_reasoning_trace_nicht_leer(self, dienst, mono_sine):
        plan = dienst.erstelle_plan(mono_sine, SR)
        assert len(plan.reasoning_trace) >= 2

    def test_emotional_intention_nicht_leer(self, dienst, mono_sine):
        plan = dienst.erstelle_plan(mono_sine, SR)
        assert isinstance(plan.emotional_intention, str)
        assert len(plan.emotional_intention) > 0

    def test_semantic_description_nicht_leer(self, dienst, mono_sine):
        plan = dienst.erstelle_plan(mono_sine, SR)
        assert isinstance(plan.portrait.semantic_description, str)
        assert len(plan.portrait.semantic_description) > 5

    def test_as_dict_serialiserbar(self, dienst, mono_sine):
        plan = dienst.erstelle_plan(mono_sine, SR)
        d = plan.as_dict()
        assert isinstance(d, dict)
        assert "portrait" in d
        assert "phase_adjustments" in d
        assert "reasoning_trace" in d

    def test_portrait_as_dict(self, dienst, mono_sine):
        plan = dienst.erstelle_plan(mono_sine, SR)
        pd = plan.portrait.as_dict()
        assert "decade" in pd
        assert "genre" in pd
        assert "bpm" in pd


# ---------------------------------------------------------------------------
# 3. Phase-Adjustments
# ---------------------------------------------------------------------------


class TestPhaseAdjustments:
    EXPECTED_PHASES = [
        "phase_01_click_removal",
        "phase_02_hum_removal",
        "phase_03_denoise",
        "phase_04_eq_correction",
        "phase_06_frequency_restoration",
        "phase_07_harmonic_restoration",
        "phase_13_stereo_enhancement",
        "phase_14_phase_correction",
        "phase_17_mastering_polish",
        "phase_21_exciter",
        "phase_22_tape_saturation",
        "phase_35_multiband_compression",
        "phase_37_bass_enhancement",
        "phase_38_presence_boost",
        "phase_39_air_band_enhancement",
        "phase_46_spatial_enhancement",
        "phase_48_stereo_width_enhancer",
    ]

    def test_alle_erwarteten_phasen_vorhanden(self, dienst, mono_sine):
        plan = dienst.erstelle_plan(mono_sine, SR)
        for ph in self.EXPECTED_PHASES:
            assert ph in plan.phase_adjustments, f"Phase {ph} fehlt in plan.phase_adjustments"

    def test_get_phase_params_liefert_dict(self, dienst, mono_sine):
        plan = dienst.erstelle_plan(mono_sine, SR)
        params = plan.get_phase_params("phase_03_denoise")
        assert isinstance(params, dict)
        assert "aggressiveness" in params

    def test_get_phase_params_unbekannte_phase_liefert_defaults(self, dienst, mono_sine):
        plan = dienst.erstelle_plan(mono_sine, SR)
        params = plan.get_phase_params("phase_99_unbekannt")
        assert isinstance(params, dict)
        assert "authenticity_weight" in params

    def test_nr_aggressiveness_im_bereich(self, dienst, mono_sine):
        plan = dienst.erstelle_plan(mono_sine, SR)
        nr = plan.get_nr_aggressiveness()
        assert 0.1 <= nr <= 1.0

    def test_phase_03_aggressivness_float(self, dienst, mono_sine):
        plan = dienst.erstelle_plan(mono_sine, SR)
        ag = plan.phase_adjustments["phase_03_denoise"]["aggressiveness"]
        assert isinstance(ag, float)
        assert 0.1 <= ag <= 1.0

    def test_cross_phase_koordination(self, dienst, mono_sine):
        """NR-Aggressivität muss mit Multiband-Ratio koordiniert sein:
        Alte Aufnahmen (pre-1950): niedrige NR + moderate Kompression.
        """
        plan = dienst.erstelle_plan(mono_sine, SR, hint_decade=1930)
        nr = plan.phase_adjustments["phase_03_denoise"]["aggressiveness"]
        mb_ratio = plan.phase_adjustments["phase_35_multiband_compression"]["ratio"]
        # Pre-1950: NR sanft (< 0.75), Multiband-Ratio niedrig (näher 1.0)
        assert nr < 0.75, f"Erwarte sanfte NR für 1930er, got {nr}"
        assert mb_ratio <= 2.0, f"Erwarte moderate Kompression für 1930er, got {mb_ratio}"

    def test_stereo_width_0_fuer_pre_1940(self, dienst, mono_sine):
        """Vor 1940 war nur Mono üblich — stereo_width sollte minimal sein."""
        plan = dienst.erstelle_plan(mono_sine, SR, hint_decade=1920)
        sw = plan.phase_adjustments["phase_13_stereo_enhancement"]["target_width"]
        assert sw < 0.05, f"Erwarte Mono-Steuerung für 1920er, got {sw}"


# ---------------------------------------------------------------------------
# 4. Hint-Parameter (Überschreiben von Klassifikatoren)
# ---------------------------------------------------------------------------


class TestHints:
    def test_hint_decade_ueberschreibt_erkennung(self, dienst, mono_sine):
        plan = dienst.erstelle_plan(mono_sine, SR, hint_decade=1940)
        assert plan.portrait.decade == 1940

    def test_hint_genre_schlager(self, dienst, mono_sine):
        plan = dienst.erstelle_plan(mono_sine, SR, hint_genre="schlager")
        assert plan.portrait.genre == "schlager"
        # Schlager erhöht Wärme
        assert plan.warmth_target >= _nearest_era_profile(plan.portrait.decade)["warmth_target"]

    def test_hint_genre_jazz(self, dienst, mono_sine):
        plan = dienst.erstelle_plan(mono_sine, SR, hint_genre="jazz")
        assert plan.portrait.genre == "jazz"

    def test_hint_material_in_portrait(self, dienst, mono_sine):
        plan = dienst.erstelle_plan(mono_sine, SR, material="shellac")
        assert plan.portrait.material == "shellac"

    def test_kombination_hint_decade_und_genre(self, dienst, mono_sine):
        plan = dienst.erstelle_plan(mono_sine, SR, hint_decade=1935, hint_genre="schlager")
        assert plan.portrait.decade == 1935
        assert plan.portrait.genre == "schlager"


# ---------------------------------------------------------------------------
# 5. NaN/Inf-Robustheit
# ---------------------------------------------------------------------------


class TestNaNRobustheit:
    def test_nan_input_kein_crash(self, dienst):
        audio = np.full(SR * 2, np.nan, dtype=np.float32)
        plan = dienst.erstelle_plan(audio, SR)
        assert isinstance(plan, StilbewussterRestaurierungsplan)

    def test_inf_input_kein_crash(self, dienst):
        audio = np.full(SR, np.inf, dtype=np.float32)
        plan = dienst.erstelle_plan(audio, SR)
        assert isinstance(plan, StilbewussterRestaurierungsplan)

    def test_leeres_array_kein_crash(self, dienst):
        audio = np.zeros(64, dtype=np.float32)
        plan = dienst.erstelle_plan(audio, SR)
        assert isinstance(plan, StilbewussterRestaurierungsplan)

    def test_nan_portrait_felder_endlich(self, dienst):
        audio = np.full(SR, np.nan, dtype=np.float32)
        plan = dienst.erstelle_plan(audio, SR)
        p = plan.portrait
        assert np.isfinite(p.warmth_score)
        assert np.isfinite(p.brightness_score)
        assert np.isfinite(p.dynamic_range_estimate)
        assert np.isfinite(plan.authenticity_target)


# ---------------------------------------------------------------------------
# 6. Mono / Stereo
# ---------------------------------------------------------------------------


class TestMonoStereo:
    def test_mono_2d_shape(self, dienst, mono_sine):
        audio_2d = mono_sine[np.newaxis, :]
        plan = dienst.erstelle_plan(audio_2d, SR)
        assert isinstance(plan, StilbewussterRestaurierungsplan)

    def test_stereo_input(self, dienst, stereo_sine):
        plan = dienst.erstelle_plan(stereo_sine, SR)
        assert isinstance(plan, StilbewussterRestaurierungsplan)

    def test_mono_stereo_gleiche_era(self, dienst, mono_sine, stereo_sine):
        plan_m = dienst.erstelle_plan(mono_sine, SR, hint_decade=1960)
        plan_s = dienst.erstelle_plan(stereo_sine, SR, hint_decade=1960)
        assert plan_m.portrait.decade == plan_s.portrait.decade == 1960


# ---------------------------------------------------------------------------
# 7. Ära-Profil-Mapping
# ---------------------------------------------------------------------------


class TestEraProfil:
    @pytest.mark.parametrize(
        "decade,expected_nr_max",
        [
            (1890, 0.60),
            (1920, 0.65),
            (1940, 0.75),
            (1960, 0.85),
            (1990, 1.00),
        ],
    )
    def test_nr_sinkt_fuer_aeltere_aufnahmen(self, dienst, mono_sine, decade, expected_nr_max):
        plan = dienst.erstelle_plan(mono_sine, SR, hint_decade=decade)
        nr = plan.phase_adjustments["phase_03_denoise"]["aggressiveness"]
        assert nr <= expected_nr_max, f"NR für {decade}er zu aggressiv: {nr:.3f} > {expected_nr_max}"

    @pytest.mark.parametrize(
        "decade,expected_hf_min",
        [
            (1890, 3.0),
            (1930, 6.0),
            (1960, 12.0),
            (1990, 18.0),
        ],
    )
    def test_hf_ceiling_waechst_mit_aeea(self, dienst, mono_sine, decade, expected_hf_min):
        plan = dienst.erstelle_plan(mono_sine, SR, hint_decade=decade)
        assert plan.hf_ceiling_khz >= expected_hf_min, f"HF-Deckel für {decade}er zu niedrig: {plan.hf_ceiling_khz}"

    def test_preserve_grain_aktiv_fuer_pre_1930(self, dienst, mono_sine):
        plan = dienst.erstelle_plan(mono_sine, SR, hint_decade=1920)
        assert plan.preserve_grain is True

    def test_preserve_grain_inaktiv_fuer_post_1940(self, dienst, mono_sine):
        plan = dienst.erstelle_plan(mono_sine, SR, hint_decade=1960)
        assert plan.preserve_grain is False


# ---------------------------------------------------------------------------
# 8. Genre-Modifikatoren
# ---------------------------------------------------------------------------


class TestGenreModifikatoren:
    def test_schlager_erhoeht_warme(self, dienst, mono_sine):
        base = dienst.erstelle_plan(mono_sine, SR, hint_decade=1960, hint_genre="unknown")
        schlager = dienst.erstelle_plan(mono_sine, SR, hint_decade=1960, hint_genre="schlager")
        assert schlager.warmth_target >= base.warmth_target

    def test_jazz_sanftere_nr(self, dienst, mono_sine):
        base = dienst.erstelle_plan(mono_sine, SR, hint_decade=1960, hint_genre="unknown")
        jazz = dienst.erstelle_plan(mono_sine, SR, hint_decade=1960, hint_genre="jazz")
        nr_base = base.phase_adjustments["phase_03_denoise"]["aggressiveness"]
        nr_jazz = jazz.phase_adjustments["phase_03_denoise"]["aggressiveness"]
        assert nr_jazz <= nr_base + 0.05, "Jazz soll NR nicht verschärfen"

    def test_klassik_hoechste_authentizitaet(self, dienst, mono_sine):
        klassik = dienst.erstelle_plan(mono_sine, SR, hint_decade=1960, hint_genre="klassik")
        nr_klassik = klassik.phase_adjustments["phase_03_denoise"]["aggressiveness"]
        assert nr_klassik < 0.90, "Klassik: NR sehr sanft für Raumakustik-Erhalt"


# ---------------------------------------------------------------------------
# 9. DSP-Hilfsfunktionen
# ---------------------------------------------------------------------------


class TestDSPHilfsfunktionen:
    def test_warmth_im_bereich(self, mono_sine):
        w = _estimate_warmth(mono_sine, SR)
        assert 0.0 <= w <= 1.0

    def test_brightness_im_bereich(self, mono_sine):
        b = _estimate_brightness(mono_sine, SR)
        assert 0.0 <= b <= 1.0

    def test_dynamic_range_im_bereich(self, mono_sine):
        dr = _estimate_dynamic_range(mono_sine)
        assert 0.0 <= dr <= 1.0

    def test_warmth_nan_kein_crash(self):
        w = _estimate_warmth(np.full(100, np.nan, dtype=np.float32), SR)
        assert np.isfinite(w)

    def test_brightness_leeres_array(self):
        b = _estimate_brightness(np.zeros(32, dtype=np.float32), SR)
        assert np.isfinite(b)

    @pytest.mark.parametrize(
        "genre,bpm,expected_keyword",
        [
            ("schlager", 90, "schwungvoll"),
            ("schlager", 60, "melancholisch"),
            ("jazz", 120, "improvisi"),
            ("klassik", 80, "romantisch"),
        ],
    )
    def test_mood_enthaelt_keyword(self, mono_sine, genre, bpm, expected_keyword):
        # Warmth und brightness aus echtem Signal ableiten
        mono = mono_sine
        warmth = _estimate_warmth(mono, SR)
        brightness = _estimate_brightness(mono, SR)
        mood = _estimate_mood(warmth, brightness, float(bpm), genre)
        assert expected_keyword in mood.lower(), f"Erwarte '{expected_keyword}' in Stimmung '{mood}'"

    def test_nearest_era_profile_snap(self):
        p = _nearest_era_profile(1945)
        # 1945 ist näher zu 1940 als zu 1950
        assert p is _nearest_era_profile(1940)

    def test_build_semantic_description_nicht_leer(self):
        desc = _build_semantic_description(1940, "schlager", "wiener", "shellac", "nostalgisch", 80.0)
        assert len(desc) > 10
        assert "1940" in desc or "40er" in desc

    def test_era_profiles_vollstaendig(self):
        """Alle definierten ERA-Profile haben die Pflichtfelder."""
        required = {
            "nr_aggressiveness",
            "harmonic_restore",
            "hf_ceiling_khz",
            "presence_boost",
            "stereo_width",
            "warmth_target",
            "authenticity_weight",
            "nr_preserves_grain",
        }
        for decade, profile in _ERA_PROFILES.items():
            missing = required - set(profile.keys())
            assert not missing, f"Era {decade}: fehlende Felder {missing}"

    def test_alle_genre_modifier_keys_numerisch(self):
        """Alle Werte in Genre-Modifikatoren müssen float sein."""
        for genre, mods in _GENRE_MODIFIERS.items():
            for key, val in mods.items():
                assert isinstance(val, (int, float)), f"Genre {genre}.{key} kein float: {type(val)}"


# ---------------------------------------------------------------------------
# 10. SR-Invariante
# ---------------------------------------------------------------------------


class TestSRInvariante:
    def test_falscher_sr_wirft_assertion(self, dienst, mono_sine):
        with pytest.raises(AssertionError, match="48000"):
            dienst.erstelle_plan(mono_sine, sr=44100)


# ---------------------------------------------------------------------------
# 11. plan_version
# ---------------------------------------------------------------------------


class TestPlanVersion:
    def test_plan_version_vorhanden(self, dienst, mono_sine):
        plan = dienst.erstelle_plan(mono_sine, SR)
        assert isinstance(plan.plan_version, str)
        assert len(plan.plan_version) > 0


# ---------------------------------------------------------------------------
# 12. Chain-aware NR-Cap
# ---------------------------------------------------------------------------


class TestChainAwareNRCap:
    def test_tape_mp3_chain_caps_nr_aggressiveness(self, dienst, mono_sine):
        plan = dienst.erstelle_plan(
            mono_sine,
            SR,
            material="tape",
            hint_decade=1970,
            chain_info={"chain": ["tape", "mp3_low"], "primary": "tape"},
        )
        nr = plan.phase_adjustments["phase_03_denoise"]["aggressiveness"]
        assert nr <= 0.20, f"Erwarte NR-Cap <= 0.20 für tape→mp3_low, got {nr:.3f}"

    def test_convenience_function_forwarded_chain_info(self, mono_sine):
        plan = erstelle_globalplan(
            mono_sine,
            SR,
            material="tape",
            hint_decade=1970,
            chain_info={"chain": ["tape", "mp3_low"], "primary": "tape"},
        )
        nr = plan.phase_adjustments["phase_03_denoise"]["aggressiveness"]
        assert nr <= 0.20, f"Erwarte NR-Cap <= 0.20 via Convenience-Funktion, got {nr:.3f}"


# ---------------------------------------------------------------------------
# 13. Material-Decade-Floor
# ---------------------------------------------------------------------------


class TestMaterialDecadeFloor:
    def test_vinyl_chain_forces_min_1950(self, dienst, mono_sine):
        plan = dienst.erstelle_plan(
            mono_sine,
            SR,
            hint_decade=1930,
            chain_info={"primary": "vinyl", "chain": ["vinyl", "tape", "mp3_low"]},
        )
        assert plan.portrait.decade >= 1950, f"Vinyl-Floor verletzt: {plan.portrait.decade}"

    def test_primary_material_alias_is_honored(self, dienst, mono_sine):
        plan = dienst.erstelle_plan(
            mono_sine,
            SR,
            hint_decade=1930,
            chain_info={"primary_material": "cassette", "transfer_chain": ["cassette", "mp3_low"]},
        )
        assert plan.portrait.decade >= 1965, f"Cassette-Floor verletzt: {plan.portrait.decade}"
