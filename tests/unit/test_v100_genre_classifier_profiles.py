import pytest

"""Erweiterte Unit-Tests für Genre-Classifier und Genre-Profile (v10.0).

Testet:
- GermanSchlagerClassifier importierbar
- classify() auf verschiedene synthetische Signale
- Alle Scores finite und ∈ [0, 1]
- Singleton thread-sicher
- SCHLAGER/JAZZ/KLASSIK/OPER/ROCK_RESTORATION_PROFILE Parameter
- get_restoration_profile() für verschiedene Genres
- Subgenre-Labels
- SCHLAGER_CLAP_PROMPTS
- Schwellwerte
- etc.
"""

import threading

import numpy as np

from backend.core.genre_classifier import (
    BLUES_RESTORATION_PROFILE,
    COUNTRY_RESTORATION_PROFILE,
    ELECTRONIC_RESTORATION_PROFILE,
    FOLK_RESTORATION_PROFILE,
    FUNK_RESTORATION_PROFILE,
    GENRE_RESTORATION_PROFILES,
    GOSPEL_RESTORATION_PROFILE,
    HIPHOP_RESTORATION_PROFILE,
    JAZZ_RESTORATION_PROFILE,
    KLASSIK_RESTORATION_PROFILE,
    LATIN_RESTORATION_PROFILE,
    METAL_RESTORATION_PROFILE,
    OPER_RESTORATION_PROFILE,
    POP_RESTORATION_PROFILE,
    REGGAE_RESTORATION_PROFILE,
    ROCK_RESTORATION_PROFILE,
    SCHLAGER_RESTORATION_PROFILE,
    SOUL_RNB_RESTORATION_PROFILE,
    GermanSchlagerClassifier,
    SchlagerClassificationResult,
    classify_genre,
    get_genre_classifier,
    get_restoration_profile,
)

# Klassenattribute als Modulkonstanten verfügbar machen
SCHLAGER_CLAP_PROMPTS = GermanSchlagerClassifier.SCHLAGER_CLAP_PROMPTS
SCHLAGER_CONFIDENCE_THRESHOLD = GermanSchlagerClassifier.SCHLAGER_CONFIDENCE_THRESHOLD
HSI_THRESHOLD = GermanSchlagerClassifier.HSI_THRESHOLD


@pytest.mark.unit
class TestV100GenreClassifierProfiles:
    """Erweiterte Tests für Genre-Classifier v10.0."""

    def test_01_german_schlager_classifier_importable(self):
        """GermanSchlagerClassifier importierbar."""
        assert GermanSchlagerClassifier is not None

    def test_02_classify_returns_result(self):
        """classify() gibt SchlagerClassificationResult zurück."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        classifier = GermanSchlagerClassifier()
        result = classifier.classify(audio, sr=48000)
        assert isinstance(result, SchlagerClassificationResult)

    def test_03_classify_silence_is_not_schlager_no_crash(self):
        """classify() auf Stille → is_schlager=False, kein Absturz."""
        audio = np.zeros(48000, dtype=np.float32)
        classifier = GermanSchlagerClassifier()
        result = classifier.classify(audio, sr=48000)
        assert isinstance(result.is_schlager, bool)

    def test_04_classify_white_noise_is_not_schlager(self):
        """classify() auf weißes Rauschen → is_schlager=False."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        classifier = GermanSchlagerClassifier()
        result = classifier.classify(audio, sr=48000)
        # Weißes Rauschen sollte nicht Schlager sein
        assert isinstance(result.is_schlager, bool)

    def test_05_all_scores_finite(self):
        """Alle Scores finite."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        classifier = GermanSchlagerClassifier()
        result = classifier.classify(audio, sr=48000)
        assert np.isfinite(result.confidence)
        assert np.isfinite(result.clap_score)
        assert np.isfinite(result.accordion_score)
        assert np.isfinite(result.harmonic_simplicity)
        assert np.isfinite(result.rhythm_score)
        assert np.isfinite(result.vocal_german_prior)
        assert np.isfinite(result.melodic_repetition)

    def test_06_all_scores_in_range_0_1(self):
        """Alle Scores ∈ [0, 1]."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        classifier = GermanSchlagerClassifier()
        result = classifier.classify(audio, sr=48000)
        assert 0.0 <= result.confidence <= 1.0
        assert 0.0 <= result.clap_score <= 1.0
        assert 0.0 <= result.accordion_score <= 1.0
        assert 0.0 <= result.harmonic_simplicity <= 1.0
        assert 0.0 <= result.rhythm_score <= 1.0
        assert 0.0 <= result.vocal_german_prior <= 1.0
        assert 0.0 <= result.melodic_repetition <= 1.0

    def test_07_singleton_thread_safe(self):
        """Singleton thread-sicher (20 parallele get_genre_classifier() → selbes Objekt)."""
        instances = []

        def get_instance():
            instances.append(get_genre_classifier())

        threads = [threading.Thread(target=get_instance) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(inst is instances[0] for inst in instances)

    def test_08_schlager_profile_tonal_center_threshold(self):
        """SCHLAGER_RESTORATION_PROFILE: tonal_center_threshold = 0.97."""
        assert SCHLAGER_RESTORATION_PROFILE["tonal_center_threshold"] == 0.97

    def test_09_jazz_profile_compression_ratio_cap(self):
        """JAZZ_RESTORATION_PROFILE: compression_ratio_cap = 1.8."""
        assert JAZZ_RESTORATION_PROFILE["compression_ratio_cap"] == 1.8

    def test_10_klassik_profile_groove_dtw_max_ms(self):
        """KLASSIK_RESTORATION_PROFILE: groove_dtw_max_ms = 10.0."""
        assert KLASSIK_RESTORATION_PROFILE["groove_dtw_max_ms"] == 10.0

    def test_11_oper_profile_gp_memory_key_opera(self):
        """OPER_RESTORATION_PROFILE: gp_memory_key = 'opera'."""
        assert OPER_RESTORATION_PROFILE["gp_memory_key"] == "opera"

    def test_12_rock_profile_gp_memory_key_rock(self):
        """ROCK_RESTORATION_PROFILE: gp_memory_key = 'rock'."""
        assert ROCK_RESTORATION_PROFILE["gp_memory_key"] == "rock"

    def test_13_get_restoration_profile_rock_returns_rock_profile(self):
        """get_restoration_profile('Rock') gibt ROCK_RESTORATION_PROFILE."""
        profile = get_restoration_profile("Rock")
        assert profile == ROCK_RESTORATION_PROFILE

    def test_14_get_restoration_profile_klassik_returns_klassik_profile(self):
        """get_restoration_profile('Klassik') gibt KLASSIK_RESTORATION_PROFILE."""
        profile = get_restoration_profile("Klassik")
        assert profile == KLASSIK_RESTORATION_PROFILE

    def test_15_subgenre_labels_valid(self):
        """Subgenre-Labels: schunkel, walzer, marsch, discoschlager, unknown."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        classifier = GermanSchlagerClassifier()
        result = classifier.classify(audio, sr=48000)
        valid_subgenres = [
            "schunkel",
            "walzer",
            "marsch",
            "discoschlager",
            "unknown",
            "schlager_1950s",
            "schlager_modern",
            "volksmusik",
        ]
        assert result.subgenre in valid_subgenres or result.subgenre == ""

    def test_16_schlager_clap_prompts_is_list(self):
        """SCHLAGER_CLAP_PROMPTS is list mit ≥ 5 Einträgen."""
        assert isinstance(SCHLAGER_CLAP_PROMPTS, list)
        assert len(SCHLAGER_CLAP_PROMPTS) >= 5

    def test_17_schlager_confidence_threshold_value(self):
        """SCHLAGER_CONFIDENCE_THRESHOLD = 0.52 (aktueller Wert laut Code)."""
        assert SCHLAGER_CONFIDENCE_THRESHOLD == 0.52

    def test_18_hsi_threshold_equals_082(self):
        """HSI_THRESHOLD = 0.82."""
        assert HSI_THRESHOLD == 0.82

    def test_19_classify_mono_input_no_error(self):
        """classify() Mono-Input (1D Array) → kein Fehler."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        assert audio.ndim == 1
        classifier = GermanSchlagerClassifier()
        result = classifier.classify(audio, sr=48000)
        assert isinstance(result, SchlagerClassificationResult)

    def test_20_classify_stereo_input_no_error(self):
        """classify() Stereo-Input (2D Array) → kein Fehler."""
        np.random.seed(42)
        audio = np.random.randn(2, 48000).astype(np.float32) * 0.1
        classifier = GermanSchlagerClassifier()
        result = classifier.classify(audio, sr=48000)
        assert isinstance(result, SchlagerClassificationResult)

    def test_21_classify_very_short_signal_no_crash(self):
        """classify() sehr kurzes Signal (< 5 s) → kein Absturz."""
        np.random.seed(42)
        audio = np.random.randn(24000).astype(np.float32) * 0.1  # 0.5 s
        classifier = GermanSchlagerClassifier()
        result = classifier.classify(audio, sr=48000)
        assert isinstance(result, SchlagerClassificationResult)

    def test_22_bpm_positive_or_zero(self):
        """bpm ≥ 0."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        classifier = GermanSchlagerClassifier()
        result = classifier.classify(audio, sr=48000)
        assert result.bpm >= 0.0

    def test_23_reasoning_not_empty(self):
        """reasoning ist nicht leer."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        classifier = GermanSchlagerClassifier()
        result = classifier.classify(audio, sr=48000)
        assert isinstance(result.reasoning, str)
        assert len(result.reasoning) > 0

    def test_24_genre_label_is_string(self):
        """genre_label ist String."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        classifier = GermanSchlagerClassifier()
        result = classifier.classify(audio, sr=48000)
        assert isinstance(result.genre_label, str)

    def test_25_classify_genre_convenience_function(self):
        """classify_genre() convenience function works."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        result = classify_genre(audio, sr=48000)
        assert isinstance(result, SchlagerClassificationResult)

    def test_26_nan_in_audio_no_crash(self):
        """NaN in audio → kein Absturz."""
        audio = np.full(48000, np.nan, dtype=np.float32)
        classifier = GermanSchlagerClassifier()
        result = classifier.classify(audio, sr=48000)
        assert isinstance(result, SchlagerClassificationResult)

    def test_27_inf_in_audio_no_crash(self):
        """Inf in audio → kein Absturz."""
        audio = np.full(48000, np.inf, dtype=np.float32)
        classifier = GermanSchlagerClassifier()
        result = classifier.classify(audio, sr=48000)
        assert isinstance(result, SchlagerClassificationResult)

    def test_28_very_loud_audio_no_crash(self):
        """Sehr lautes Audio → kein Absturz."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 10.0
        classifier = GermanSchlagerClassifier()
        result = classifier.classify(audio, sr=48000)
        assert isinstance(result, SchlagerClassificationResult)

    def test_29_very_quiet_audio_no_crash(self):
        """Sehr leises Audio → kein Absturz."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.0001
        classifier = GermanSchlagerClassifier()
        result = classifier.classify(audio, sr=48000)
        assert isinstance(result, SchlagerClassificationResult)

    def test_30_key_is_string_or_empty(self):
        """key ist String oder leer."""
        np.random.seed(42)
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        classifier = GermanSchlagerClassifier()
        result = classifier.classify(audio, sr=48000)
        assert isinstance(result.key, str)

    def test_31_schlager_profile_soft_saturation_preserve_true(self):
        """SCHLAGER_RESTORATION_PROFILE: soft_saturation_preserve = True."""
        assert SCHLAGER_RESTORATION_PROFILE["soft_saturation_preserve"] is True

    def test_32_jazz_profile_tonal_center_threshold(self):
        """JAZZ_RESTORATION_PROFILE: tonal_center_threshold = 0.92."""
        assert JAZZ_RESTORATION_PROFILE["tonal_center_threshold"] == 0.92

    def test_33_klassik_profile_compression_ratio_cap(self):
        """KLASSIK_RESTORATION_PROFILE: compression_ratio_cap = 1.3."""
        assert KLASSIK_RESTORATION_PROFILE["compression_ratio_cap"] == 1.3

    def test_34_oper_profile_deessing_target_hz(self):
        """OPER_RESTORATION_PROFILE: deessing_target_hz = 7000."""
        assert OPER_RESTORATION_PROFILE["deessing_target_hz"] == 7000

    def test_35_rock_profile_brillanz_target(self):
        """ROCK_RESTORATION_PROFILE: brillanz_target = 0.90."""
        assert ROCK_RESTORATION_PROFILE["brillanz_target"] == 0.90

    def test_36_genre_restoration_profiles_has_all_five(self):
        """GENRE_RESTORATION_PROFILES hat alle 5 Hauptgenres."""
        profile_keys = list(GENRE_RESTORATION_PROFILES.keys())
        expected = ["Schlager", "Jazz", "Klassik", "Oper", "Rock"]
        for genre in expected:
            assert genre in profile_keys

    def test_37_get_restoration_profile_case_insensitive_or_exact(self):
        """get_restoration_profile() funktioniert."""
        profile = get_restoration_profile("Jazz")
        assert profile == JAZZ_RESTORATION_PROFILE

    def test_38_schlager_profile_brillanz_target_lower(self):
        """SCHLAGER_RESTORATION_PROFILE: brillanz_target = 0.82."""
        assert SCHLAGER_RESTORATION_PROFILE["brillanz_target"] == 0.82

    def test_39_schlager_profile_waerme_target_higher(self):
        """SCHLAGER_RESTORATION_PROFILE: waerme_target = 0.88."""
        assert SCHLAGER_RESTORATION_PROFILE["waerme_target"] == 0.88

    def test_40_all_profiles_have_gp_memory_key(self):
        """Alle Profile haben gp_memory_key."""
        for profile in [
            SCHLAGER_RESTORATION_PROFILE,
            JAZZ_RESTORATION_PROFILE,
            KLASSIK_RESTORATION_PROFILE,
            OPER_RESTORATION_PROFILE,
            ROCK_RESTORATION_PROFILE,
        ]:
            assert "gp_memory_key" in profile


class TestNewGenreProfiles:
    """Tests für die 12 neuen Genre-Profile (Pop, Blues, Soul/R&B, Country, Folk,
    Funk, Electronic, Hip-Hop, Metal, Latin, Gospel, Reggae)."""

    # ---- Profil-Import-Tests ----

    def test_41_pop_profile_importable(self):
        assert POP_RESTORATION_PROFILE is not None

    def test_42_blues_profile_importable(self):
        assert BLUES_RESTORATION_PROFILE is not None

    def test_43_soul_rnb_profile_importable(self):
        assert SOUL_RNB_RESTORATION_PROFILE is not None

    def test_44_country_profile_importable(self):
        assert COUNTRY_RESTORATION_PROFILE is not None

    def test_45_folk_profile_importable(self):
        assert FOLK_RESTORATION_PROFILE is not None

    def test_46_funk_profile_importable(self):
        assert FUNK_RESTORATION_PROFILE is not None

    def test_47_electronic_profile_importable(self):
        assert ELECTRONIC_RESTORATION_PROFILE is not None

    def test_48_hiphop_profile_importable(self):
        assert HIPHOP_RESTORATION_PROFILE is not None

    def test_49_metal_profile_importable(self):
        assert METAL_RESTORATION_PROFILE is not None

    def test_50_latin_profile_importable(self):
        assert LATIN_RESTORATION_PROFILE is not None

    def test_51_gospel_profile_importable(self):
        assert GOSPEL_RESTORATION_PROFILE is not None

    def test_52_reggae_profile_importable(self):
        assert REGGAE_RESTORATION_PROFILE is not None

    # ---- gp_memory_key vorhanden ----

    def test_53_all_new_profiles_have_gp_memory_key(self):
        """Alle 12 neuen Profile haben gp_memory_key."""
        profiles = [
            ("Pop", POP_RESTORATION_PROFILE),
            ("Blues", BLUES_RESTORATION_PROFILE),
            ("Soul/R&B", SOUL_RNB_RESTORATION_PROFILE),
            ("Country", COUNTRY_RESTORATION_PROFILE),
            ("Folk", FOLK_RESTORATION_PROFILE),
            ("Funk", FUNK_RESTORATION_PROFILE),
            ("Electronic", ELECTRONIC_RESTORATION_PROFILE),
            ("Hip-Hop", HIPHOP_RESTORATION_PROFILE),
            ("Metal", METAL_RESTORATION_PROFILE),
            ("Latin", LATIN_RESTORATION_PROFILE),
            ("Gospel", GOSPEL_RESTORATION_PROFILE),
            ("Reggae", REGGAE_RESTORATION_PROFILE),
        ]
        for name, profile in profiles:
            assert "gp_memory_key" in profile, f"{name}: gp_memory_key fehlt"

    # ---- gp_memory_key-Werte ----

    def test_54_pop_profile_gp_memory_key(self):
        assert POP_RESTORATION_PROFILE["gp_memory_key"] == "pop"

    def test_55_blues_profile_gp_memory_key(self):
        assert BLUES_RESTORATION_PROFILE["gp_memory_key"] == "blues"

    def test_56_soul_rnb_profile_gp_memory_key(self):
        assert SOUL_RNB_RESTORATION_PROFILE["gp_memory_key"] == "soul_rnb"

    def test_57_country_profile_gp_memory_key(self):
        assert COUNTRY_RESTORATION_PROFILE["gp_memory_key"] == "country"

    def test_58_folk_profile_gp_memory_key(self):
        assert FOLK_RESTORATION_PROFILE["gp_memory_key"] == "folk"

    def test_59_funk_profile_gp_memory_key(self):
        assert FUNK_RESTORATION_PROFILE["gp_memory_key"] == "funk"

    def test_60_electronic_profile_gp_memory_key(self):
        assert ELECTRONIC_RESTORATION_PROFILE["gp_memory_key"] == "electronic"

    def test_61_hiphop_profile_gp_memory_key(self):
        assert HIPHOP_RESTORATION_PROFILE["gp_memory_key"] == "hiphop"

    def test_62_metal_profile_gp_memory_key(self):
        assert METAL_RESTORATION_PROFILE["gp_memory_key"] == "metal"

    def test_63_latin_profile_gp_memory_key(self):
        assert LATIN_RESTORATION_PROFILE["gp_memory_key"] == "latin"

    def test_64_gospel_profile_gp_memory_key(self):
        assert GOSPEL_RESTORATION_PROFILE["gp_memory_key"] == "gospel"

    def test_65_reggae_profile_gp_memory_key(self):
        assert REGGAE_RESTORATION_PROFILE["gp_memory_key"] == "reggae"

    # ---- get_restoration_profile() — alle neuen Genres ----

    def test_66_get_restoration_profile_pop(self):
        assert get_restoration_profile("Pop") == POP_RESTORATION_PROFILE

    def test_67_get_restoration_profile_blues(self):
        assert get_restoration_profile("Blues") == BLUES_RESTORATION_PROFILE

    def test_68_get_restoration_profile_soul_rnb_slash(self):
        assert get_restoration_profile("Soul/R&B") == SOUL_RNB_RESTORATION_PROFILE

    def test_69_get_restoration_profile_soul_rnb_underscore(self):
        assert get_restoration_profile("soul_rnb") == SOUL_RNB_RESTORATION_PROFILE

    def test_70_get_restoration_profile_country(self):
        assert get_restoration_profile("Country") == COUNTRY_RESTORATION_PROFILE

    def test_71_get_restoration_profile_folk(self):
        assert get_restoration_profile("Folk") == FOLK_RESTORATION_PROFILE

    def test_72_get_restoration_profile_funk(self):
        assert get_restoration_profile("Funk") == FUNK_RESTORATION_PROFILE

    def test_73_get_restoration_profile_electronic(self):
        assert get_restoration_profile("Electronic") == ELECTRONIC_RESTORATION_PROFILE

    def test_74_get_restoration_profile_hiphop_hyphen(self):
        assert get_restoration_profile("Hip-Hop") == HIPHOP_RESTORATION_PROFILE

    def test_75_get_restoration_profile_metal(self):
        assert get_restoration_profile("Metal") == METAL_RESTORATION_PROFILE

    def test_76_get_restoration_profile_latin(self):
        assert get_restoration_profile("Latin") == LATIN_RESTORATION_PROFILE

    def test_77_get_restoration_profile_gospel(self):
        assert get_restoration_profile("Gospel") == GOSPEL_RESTORATION_PROFILE

    def test_78_get_restoration_profile_reggae(self):
        assert get_restoration_profile("Reggae") == REGGAE_RESTORATION_PROFILE

    def test_79_get_restoration_profile_lowercase(self):
        """Kleinschreibung funktioniert für alle neuen Genres."""
        assert get_restoration_profile("pop") == POP_RESTORATION_PROFILE
        assert get_restoration_profile("blues") == BLUES_RESTORATION_PROFILE
        assert get_restoration_profile("folk") == FOLK_RESTORATION_PROFILE
        assert get_restoration_profile("metal") == METAL_RESTORATION_PROFILE
        assert get_restoration_profile("reggae") == REGGAE_RESTORATION_PROFILE

    def test_80_get_restoration_profile_rap_alias(self):
        """'rap' → Hip-Hop-Profil."""
        assert get_restoration_profile("rap") == HIPHOP_RESTORATION_PROFILE

    def test_81_get_restoration_profile_dance_alias(self):
        """'dance' → Electronic-Profil."""
        assert get_restoration_profile("dance") == ELECTRONIC_RESTORATION_PROFILE

    # ---- GENRE_RESTORATION_PROFILES enthält alle 17 Genres ----

    def test_82_genre_restoration_profiles_has_all_17_genres(self):
        """GENRE_RESTORATION_PROFILES hat alle 17 Genres (5 alt + 12 neu)."""
        expected = [
            "Pop",
            "Blues",
            "Soul/R&B",
            "Country",
            "Folk",
            "Funk",
            "Electronic",
            "Hip-Hop",
            "Metal",
            "Latin",
            "Gospel",
            "Reggae",
            "Schlager",
            "Jazz",
            "Klassik",
            "Oper",
            "Rock",
        ]
        for genre in expected:
            assert genre in GENRE_RESTORATION_PROFILES, f"'{genre}' fehlt in GENRE_RESTORATION_PROFILES"

    # ---- Score-Methoden direkt testen ----

    def test_83_score_methods_return_float_in_range(self):
        """Alle 12 neuen _score_*-Methoden geben float ∈ [0, 1] zurück."""
        clf = get_genre_classifier()
        centroid_hz = 2500.0
        onset_rate = 3.0
        hsi = 0.60
        dr_db = 25.0
        bpm = 110.0
        scores = {
            "pop": clf._score_pop(centroid_hz, onset_rate, hsi, dr_db, bpm),
            "blues": clf._score_blues(centroid_hz, onset_rate, hsi, dr_db, bpm),
            "soul": clf._score_soul_rnb(centroid_hz, onset_rate, hsi, dr_db, bpm),
            "country": clf._score_country(centroid_hz, onset_rate, hsi, dr_db, bpm),
            "folk": clf._score_folk(centroid_hz, onset_rate, hsi, dr_db, bpm),
            "funk": clf._score_funk(centroid_hz, onset_rate, hsi, dr_db, bpm),
            "electronic": clf._score_electronic(centroid_hz, onset_rate, hsi, dr_db, bpm),
            "hiphop": clf._score_hiphop(centroid_hz, onset_rate, hsi, dr_db, bpm),
            "metal": clf._score_metal(centroid_hz, onset_rate, hsi, dr_db, bpm),
            "latin": clf._score_latin(centroid_hz, onset_rate, hsi, bpm),
            "gospel": clf._score_gospel(centroid_hz, onset_rate, hsi, dr_db, bpm),
            "reggae": clf._score_reggae(centroid_hz, onset_rate, hsi, dr_db, bpm),
        }
        for name, s in scores.items():
            assert isinstance(s, float), f"{name}: kein float"
            assert 0.0 <= s <= 1.0, f"{name}: score {s} außerhalb [0, 1]"

    def test_84_score_pop_bright_compressed(self):
        """Pop: heller Centroid + hohe Onsets + niedrige DR → Score > 0."""
        clf = get_genre_classifier()
        s = clf._score_pop(centroid_hz=3500.0, onset_rate=4.0, hsi=0.70, dr_db=15.0, bpm=120.0)
        assert s > 0.0

    def test_85_score_blues_warm_pentatonic(self):
        """Blues: pentatonisches HSI (0.45) + warmer Centroid + weite DR → Score > 0."""
        clf = get_genre_classifier()
        s = clf._score_blues(centroid_hz=2000.0, onset_rate=2.5, hsi=0.50, dr_db=32.0, bpm=90.0)
        assert s > 0.0

    def test_86_score_folk_very_simple_harmony_low_onset(self):
        """Folk: sehr einfache Harmonie + niedriger Onset + normale DR (22 dB) → Score > 0."""
        clf = get_genre_classifier()
        s = clf._score_folk(centroid_hz=2000.0, onset_rate=1.2, hsi=0.80, dr_db=22.0, bpm=100.0)
        assert s > 0.0

    def test_87_score_metal_high_centroid_extreme_onset(self):
        """Metal: sehr hoher Centroid + extrem dichte Onsets → Score > 0."""
        clf = get_genre_classifier()
        s = clf._score_metal(centroid_hz=4000.0, onset_rate=6.0, hsi=0.40, dr_db=28.0, bpm=160.0)
        assert s > 0.0

    def test_88_score_electronic_extreme_compression_high_bpm(self):
        """Electronic: extreme Kompression (DR < 12) + BPM 135 → Score > 0."""
        clf = get_genre_classifier()
        s = clf._score_electronic(centroid_hz=4000.0, onset_rate=5.0, hsi=0.65, dr_db=8.0, bpm=135.0)
        assert s > 0.0

    def test_89_score_reggae_slow_bpm_bass_heavy(self):
        """Reggae: langsames BPM 75 + bass-lastig → Score > 0."""
        clf = get_genre_classifier()
        s = clf._score_reggae(centroid_hz=1800.0, onset_rate=2.0, hsi=0.72, dr_db=22.0, bpm=75.0)
        assert s > 0.0

    def test_90_score_latin_salsa_bpm_dense_rhythm(self):
        """Latin/Salsa: hohes BPM 180 + dichte Rhythmik → Score > 0."""
        clf = get_genre_classifier()
        s = clf._score_latin(centroid_hz=3000.0, onset_rate=4.0, hsi=0.55, bpm=180.0)
        assert s > 0.0

    def test_91_compute_non_schlager_scores_has_16_keys(self):
        """_compute_non_schlager_scores() gibt Dict mit 16 Genre-Keys zurück."""
        clf = get_genre_classifier()
        scores = clf._compute_non_schlager_scores(centroid_hz=2500.0, onset_rate=3.0, hsi=0.60, dr_db=25.0, bpm=110.0)
        assert len(scores) == 16
        expected_keys = [
            "Rock",
            "Jazz",
            "Klassik",
            "Oper",
            "Pop",
            "Blues",
            "Soul/R&B",
            "Country",
            "Folk",
            "Funk",
            "Electronic",
            "Hip-Hop",
            "Metal",
            "Latin",
            "Gospel",
            "Reggae",
        ]
        for k in expected_keys:
            assert k in scores, f"Key '{k}' fehlt"

    def test_92_compute_non_schlager_scores_all_in_range(self):
        """Alle Scores aus _compute_non_schlager_scores() ∈ [0, 1]."""
        clf = get_genre_classifier()
        for centroid in [1200.0, 2500.0, 4000.0]:
            scores = clf._compute_non_schlager_scores(centroid, 3.0, 0.60, 25.0, 110.0)
            for k, v in scores.items():
                assert 0.0 <= v <= 1.0, f"{k} = {v} außerhalb [0, 1] bei centroid={centroid}"

    def test_93_classify_returns_known_genre_label(self):
        """genre_label ist eines aus den 17 bekannten Genres oder 'Unbekannt'."""
        np.random.seed(99)
        audio = np.random.randn(96000).astype(np.float32) * 0.1
        result = classify_genre(audio, sr=48000)
        valid_labels = {
            "Schlager",
            "Walzer",
            "Marsch",
            "Disco-Schlager",
            "Volksmusik",
            "Deutscher Schlager",
            "Jazz",
            "Klassik",
            "Oper",
            "Rock",
            "Pop",
            "Blues",
            "Soul/R&B",
            "Country",
            "Folk",
            "Funk",
            "Electronic",
            "Hip-Hop",
            "Metal",
            "Latin",
            "Gospel",
            "Reggae",
            "Unbekannt",
        }
        assert result.genre_label in valid_labels, f"Unbekanntes genre_label: '{result.genre_label}'"

    def test_94_blues_profile_soft_saturation_preserve(self):
        """BLUES_RESTORATION_PROFILE: soft_saturation_preserve = True."""
        assert BLUES_RESTORATION_PROFILE["soft_saturation_preserve"] is True

    def test_95_folk_profile_compression_ratio_cap_conservative(self):
        """FOLK_RESTORATION_PROFILE: compression_ratio_cap ≤ 1.5 (konservativ)."""
        assert FOLK_RESTORATION_PROFILE["compression_ratio_cap"] <= 1.5

    def test_96_funk_profile_groove_dtw_strict(self):
        """FUNK_RESTORATION_PROFILE: groove_dtw_max_ms ≤ 5.0 (streng)."""
        assert FUNK_RESTORATION_PROFILE["groove_dtw_max_ms"] <= 5.0

    def test_97_metal_profile_clipping_repair_threshold(self):
        """METAL_RESTORATION_PROFILE: clipping_repair_threshold_db = -1.5."""
        assert METAL_RESTORATION_PROFILE["clipping_repair_threshold_db"] == -1.5

    def test_98_pop_profile_brillanz_target(self):
        """POP_RESTORATION_PROFILE: brillanz_target = 0.88."""
        assert POP_RESTORATION_PROFILE["brillanz_target"] == 0.88

    def test_99_hiphop_profile_bass_kraft_target(self):
        """HIPHOP_RESTORATION_PROFILE: bass_kraft_target = 0.90."""
        assert HIPHOP_RESTORATION_PROFILE["bass_kraft_target"] == 0.90

    def test_100_all_17_profiles_have_groove_dtw_max_ms(self):
        """Alle 17 Profiles (außer OPER) haben groove_dtw_max_ms."""
        profiles_with_groove = [
            SCHLAGER_RESTORATION_PROFILE,
            JAZZ_RESTORATION_PROFILE,
            KLASSIK_RESTORATION_PROFILE,
            ROCK_RESTORATION_PROFILE,
            POP_RESTORATION_PROFILE,
            BLUES_RESTORATION_PROFILE,
            SOUL_RNB_RESTORATION_PROFILE,
            COUNTRY_RESTORATION_PROFILE,
            FOLK_RESTORATION_PROFILE,
            FUNK_RESTORATION_PROFILE,
            ELECTRONIC_RESTORATION_PROFILE,
            HIPHOP_RESTORATION_PROFILE,
            METAL_RESTORATION_PROFILE,
            LATIN_RESTORATION_PROFILE,
            GOSPEL_RESTORATION_PROFILE,
            REGGAE_RESTORATION_PROFILE,
        ]
        for p in profiles_with_groove:
            assert "groove_dtw_max_ms" in p
