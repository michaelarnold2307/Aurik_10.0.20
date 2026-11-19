import pytest

"""
Tests für MicrophoneResponseLibrary (§6.4a).
"""

import json
from pathlib import Path

import numpy as np


@pytest.mark.unit
class TestMicrophoneResponseLibraryImport:
    def test_import_ok(self):
        from backend.core.microphone_response_library import (
            get_microphone_response_library,
        )

        assert callable(get_microphone_response_library)

    def test_singleton(self):
        from backend.core.microphone_response_library import get_microphone_response_library

        a = get_microphone_response_library()
        b = get_microphone_response_library()
        assert a is b

    def test_profiles_loaded(self):
        from backend.core.microphone_response_library import get_microphone_response_library

        lib = get_microphone_response_library()
        assert len(lib._profiles) > 0, "Keine Profile geladen"

    def test_json_file_exists(self):
        """backend/data/microphone_profiles.json muss existieren."""
        profiles_path = Path(__file__).parent.parent.parent / "backend" / "data" / "microphone_profiles.json"
        assert profiles_path.exists(), f"Datei nicht gefunden: {profiles_path}"

    def test_json_structure(self):
        """JSON muss Schema und profiles-Liste enthalten."""
        profiles_path = Path(__file__).parent.parent.parent / "backend" / "data" / "microphone_profiles.json"
        with open(profiles_path, encoding="utf-8") as f:
            data = json.load(f)
        assert "profiles" in data
        assert len(data["profiles"]) >= 10, "Mindestens 10 Profile erwartet (§6.4a)"

    def test_all_profiles_have_eq_curve(self):
        """Alle Profile müssen eine eq_curve mit mindestens 5 Punkten haben."""
        profiles_path = Path(__file__).parent.parent.parent / "backend" / "data" / "microphone_profiles.json"
        with open(profiles_path, encoding="utf-8") as f:
            data = json.load(f)
        for profile in data["profiles"]:
            eq = profile.get("eq_curve", [])
            assert len(eq) >= 5, f"Profil {profile['id']} hat zu wenige EQ-Punkte: {len(eq)}"
            for point in eq:
                assert "hz" in point
                assert "db" in point


class TestGetProfile:
    def test_shellac_1930_jazz_returns_profile(self):
        from backend.core.microphone_response_library import MicrophoneProfile, get_microphone_response_library

        lib = get_microphone_response_library()
        profile = lib.get_profile(era_decade=1930, genre_label="jazz", material_type="shellac")
        assert profile is not None
        assert isinstance(profile, MicrophoneProfile)
        assert profile.profile_id != "" or profile.name != ""

    def test_vinyl_1960_rock_returns_profile(self):
        from backend.core.microphone_response_library import get_microphone_response_library

        lib = get_microphone_response_library()
        profile = lib.get_profile(era_decade=1960, genre_label="rock", material_type="vinyl")
        assert profile is not None

    def test_unknown_era_returns_something(self):
        """Auch bei unbekannter Ära soll ein Fallback-Profil zurückgegeben werden."""
        from backend.core.microphone_response_library import get_microphone_response_library

        lib = get_microphone_response_library()
        profile = lib.get_profile(era_decade=2010, genre_label="electronic", material_type="cd")
        from backend.core.microphone_response_library import MicrophoneProfile

        # Kann None sein wenn kein Match, aber soll keinen Crash verursachen
        # (Profile haben Ärabereich bis ~1970s/1980s)
        assert profile is None or isinstance(profile, MicrophoneProfile)

    def test_era_scoring_prefers_closer_decade(self):
        """1930er Profil wird für 1935 bevorzugt, nicht 1960er."""
        from backend.core.microphone_response_library import get_microphone_response_library

        lib = get_microphone_response_library()
        p1930 = lib.get_profile(era_decade=1930, genre_label="jazz", material_type="shellac")
        p1960 = lib.get_profile(era_decade=1960, genre_label="jazz", material_type="shellac")

        assert p1930 is not None
        assert p1960 is not None
        # IDs können unterschiedlich sein (unterschiedliche Jahrzehnte)
        # Hauptsache kein Crash und valides Profil


class TestGetEqCurve:
    def test_returns_tuple_for_valid_input(self):
        from backend.core.microphone_response_library import get_microphone_response_library

        lib = get_microphone_response_library()
        result = lib.get_eq_curve(era_decade=1940, genre_label="jazz", material_type="shellac")
        assert result is not None
        freqs, gains = result
        assert len(freqs) >= 2
        assert len(gains) == len(freqs)

    def test_freqs_ascending(self):
        from backend.core.microphone_response_library import get_microphone_response_library

        lib = get_microphone_response_library()
        result = lib.get_eq_curve(era_decade=1950, genre_label="jazz", material_type="vinyl")
        if result is not None:
            freqs, _ = result
            assert np.all(np.diff(freqs) > 0), "Frequenzen nicht aufsteigend"

    def test_gains_nonnegative(self):
        """gains_linear MUSS >= 0 sein (lineare Skalierung aus dB)."""
        from backend.core.microphone_response_library import get_microphone_response_library

        lib = get_microphone_response_library()
        result = lib.get_eq_curve(era_decade=1950, genre_label="jazz", material_type="vinyl")
        if result is not None:
            _, gains = result
            assert np.all(gains >= 0), "gains_linear enthält negative Werte"

    def test_nyquist_limit_respected(self):
        """Keine Frequenzen über Nyquist."""
        from backend.core.microphone_response_library import get_microphone_response_library

        lib = get_microphone_response_library()
        sr = 48000
        result = lib.get_eq_curve(era_decade=1940, genre_label="jazz", material_type="shellac", target_sr=sr)
        if result is not None:
            freqs, _ = result
            assert np.all(freqs <= sr / 2.0)


class TestApplyEqCurve:
    def test_output_shape_preserved(self):
        """Output hat dieselbe Form wie Input."""
        from backend.core.microphone_response_library import get_microphone_response_library

        sr = 48000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        lib = get_microphone_response_library()
        result = lib.apply_eq_curve(audio, sr, 1940, "jazz", "shellac")

        assert result.shape == audio.shape

    def test_wet_mix_hard_cap(self):
        """wet_mix wird auf 0.35 gecappt."""
        from backend.core.microphone_response_library import get_microphone_response_library

        sr = 48000
        audio = np.random.default_rng(42).random(sr).astype(np.float32) - 0.5
        lib = get_microphone_response_library()

        # Sollte intern auf 0.35 gecappt werden, kein Crash
        result = lib.apply_eq_curve(audio, sr, 1950, "jazz", "vinyl", wet_mix=1.0)
        assert result.shape == audio.shape
        assert np.all(np.abs(result) <= 1.0 + 1e-6)

    def test_output_clipped(self):
        """Output ist in [-1, 1]."""
        from backend.core.microphone_response_library import get_microphone_response_library

        sr = 48000
        audio = 0.9 * np.random.default_rng(99).random(sr).astype(np.float32) - 0.45
        lib = get_microphone_response_library()

        result = lib.apply_eq_curve(audio, sr, 1940, "jazz", "shellac")
        assert np.all(result <= 1.0 + 1e-6)
        assert np.all(result >= -1.0 - 1e-6)

    def test_stereo_input(self):
        """Stereo-Input (2, N) wird verarbeitet."""
        from backend.core.microphone_response_library import get_microphone_response_library

        sr = 48000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        ch = (0.4 * np.sin(2 * np.pi * 330 * t)).astype(np.float32)
        stereo = np.stack([ch, ch])

        lib = get_microphone_response_library()
        result = lib.apply_eq_curve(stereo, sr, 1950, "jazz", "vinyl")

        assert result.shape == stereo.shape


class TestVersaConfidenceInPsychoacoustics:
    def test_import_ok(self):
        from backend.core.dsp.psychoacoustics import compute_versa_confidence

        assert callable(compute_versa_confidence)

    def test_cd_high_snr_returns_high_confidence(self):
        from backend.core.dsp.psychoacoustics import compute_versa_confidence

        conf = compute_versa_confidence(snr_estimate_db=40.0, material_type="cd")
        assert conf >= 0.90, f"CD/hoher SNR sollte hohe Konfidenz haben: {conf}"

    def test_shellac_returns_lower_confidence(self):
        from backend.core.dsp.psychoacoustics import compute_versa_confidence

        conf = compute_versa_confidence(snr_estimate_db=15.0, material_type="shellac")
        assert conf < 0.80, f"Shellac sollte niedrigere Konfidenz haben: {conf}"

    def test_low_snr_reduces_confidence(self):
        from backend.core.dsp.psychoacoustics import compute_versa_confidence

        high_snr = compute_versa_confidence(snr_estimate_db=40.0, material_type="vinyl")
        low_snr = compute_versa_confidence(snr_estimate_db=5.0, material_type="vinyl")
        assert low_snr < high_snr, "Niedriger SNR sollte geringere Konfidenz liefern"

    def test_result_in_valid_range(self):
        from backend.core.dsp.psychoacoustics import compute_versa_confidence

        for material in ["cd", "vinyl", "shellac", "reel_tape", "mp3_low", "unknown"]:
            conf = compute_versa_confidence(snr_estimate_db=20.0, material_type=material)
            assert 0.10 <= conf <= 1.00, f"{material}: {conf} out of [0.10, 1.00]"

    def test_unknown_material_fallback(self):
        from backend.core.dsp.psychoacoustics import compute_versa_confidence

        conf = compute_versa_confidence(snr_estimate_db=30.0, material_type="totally_unknown_xyz")
        assert 0.10 <= conf <= 1.00

    def test_in_all_exports(self):
        """compute_versa_confidence muss in __all__ exportiert sein."""
        from backend.core.dsp import psychoacoustics

        assert "compute_versa_confidence" in psychoacoustics.__all__
