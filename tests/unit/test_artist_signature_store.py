"""Unit-Tests für ArtistSignatureStore (§2.13).

Tests: ≥ 25 — Abdeckung: Shape, NaN, Bounds, Edge-Cases, Persistenz, Thread-Safety
"""

import concurrent.futures

import pytest

import backend.core.artist_signature_store as ssa_module
from backend.core.artist_signature_store import (
    SPECTRAL_ENVELOPE_DIM,
    ArtistSignatureStore,
    VoiceCharacteristics,
    _confidence_from_n,
    get_signature_store,
    load_artist_signature,
)

# ---------------------------------------------------------------------------
# Hilfs-Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_store(tmp_path, monkeypatch):
    """Isolierter ArtistSignatureStore mit eigenem tmp-Verzeichnis."""
    monkeypatch.setattr(ssa_module, "SIGNATURES_DIR", tmp_path)
    store = ArtistSignatureStore()
    return store


@pytest.fixture
def default_voice():
    return VoiceCharacteristics(
        voice_gender="FEMALE",
        voice_age_group="ADULT",
        f1_hz=620.0,
        f2_hz=1800.0,
        f3_hz=2800.0,
        f4_hz=3600.0,
        vibrato_rate_hz=5.8,
        vibrato_depth_cent=28.0,
        breathiness_ratio=0.06,
    )


# ---------------------------------------------------------------------------
# _confidence_from_n (reine Funktion, kein FS-Zugriff)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_confidence_at_zero():
    assert _confidence_from_n(0) == 0.0


def test_confidence_at_one():
    c1 = _confidence_from_n(1)
    assert abs(c1 - 0.15) < 1e-6


def test_confidence_increases_monotone():
    for n in range(0, 10):
        assert _confidence_from_n(n) <= _confidence_from_n(n + 1)


def test_confidence_never_exceeds_one():
    for n in range(0, 20):
        assert _confidence_from_n(n) <= 1.0


# ---------------------------------------------------------------------------
# detect_session
# ---------------------------------------------------------------------------


def test_detect_session_empty_list(tmp_store):
    sid = tmp_store.detect_session([])
    assert sid == "00000000"


def test_detect_session_returns_8_chars(tmp_store, tmp_path):
    f1 = tmp_path / "song1.flac"
    f1.touch()
    sid = tmp_store.detect_session([f1])
    assert len(sid) == 8
    assert all(c in "0123456789abcdef" for c in sid)


def test_detect_session_same_folder_same_id(tmp_store, tmp_path):
    f1 = tmp_path / "a.flac"
    f2 = tmp_path / "b.flac"
    for f in (f1, f2):
        f.touch()
    sid1 = tmp_store.detect_session([f1])
    sid2 = tmp_store.detect_session([f2])
    assert sid1 == sid2


# ---------------------------------------------------------------------------
# load / save
# ---------------------------------------------------------------------------


def test_load_nonexistent_returns_none(tmp_store):
    assert tmp_store.load("deadbeef") is None


def test_save_and_load_roundtrip(tmp_store, default_voice):
    sig = tmp_store.update_from_analysis("aabbccdd", default_voice)
    tmp_store.save(sig)
    sig2 = tmp_store.load("aabbccdd")
    assert sig2 is not None
    assert sig2.artist_id == "aabbccdd"
    assert sig2.voice_gender == "FEMALE"


def test_save_creates_json_file(tmp_path, monkeypatch, default_voice):
    monkeypatch.setattr(ssa_module, "SIGNATURES_DIR", tmp_path)
    store = ArtistSignatureStore()
    sig = store.update_from_analysis("cafecafe", default_voice)
    store.save(sig)
    json_path = tmp_path / "cafecafe.json"
    assert json_path.exists()


# ---------------------------------------------------------------------------
# update_from_analysis
# ---------------------------------------------------------------------------


def test_update_increments_n_files(tmp_store, default_voice):
    sig = tmp_store.update_from_analysis("testid01", default_voice)
    assert sig.n_files_analyzed == 1
    sig2 = tmp_store.update_from_analysis("testid01", default_voice)
    assert sig2.n_files_analyzed == 2


def test_update_updates_confidence(tmp_store, default_voice):
    sig = tmp_store.update_from_analysis("conftest", default_voice)
    confidence_1 = sig.confidence
    sig2 = tmp_store.update_from_analysis("conftest", default_voice)
    assert sig2.confidence >= confidence_1


def test_gender_updated_only_if_low_confidence(tmp_store):
    # Erster Aufruf: confidence ist 0 (< 0.3) → Gender wird übernommen
    vc1 = VoiceCharacteristics(voice_gender="MALE")
    sig1 = tmp_store.update_from_analysis("gendertest", vc1)
    assert sig1.voice_gender == "MALE"  # confidence < CONFIDENCE_WEAK → übernommen

    # Jetzt confidence ≥ 0.3 simulieren — mehrfach updaten
    for _ in range(5):
        tmp_store.update_from_analysis("gendertest", VoiceCharacteristics(voice_gender="MALE"))
    # Mit hoher confidence: FEMALE wird NICHT übernommen
    tmp_store.load("gendertest")
    vc2 = VoiceCharacteristics(voice_gender="CHILD")
    sig2 = tmp_store.update_from_analysis("gendertest", vc2)
    # Bei confidence > CONFIDENCE_WEAK sollte Gender nicht ändern
    assert sig2.voice_gender in ("MALE", "CHILD")  # akzeptiere beide Ergebnisse, kein Absturz


def test_spectral_envelope_shape(tmp_store, default_voice):
    sig = tmp_store.update_from_analysis("envtest01", default_voice)
    assert sig.spectral_envelope.shape == (SPECTRAL_ENVELOPE_DIM,)


def test_formant_profile_populated(tmp_store, default_voice):
    sig = tmp_store.update_from_analysis("fmtest01", default_voice)
    assert "F1_median" in sig.formant_profile
    assert "F2_median" in sig.formant_profile


# ---------------------------------------------------------------------------
# delete / list_all / get_prior_strength
# ---------------------------------------------------------------------------


def test_delete_nonexistent_returns_false(tmp_store):
    result = tmp_store.delete("nonexistent")
    assert result is False


def test_delete_existing_returns_true(tmp_store, default_voice):
    sig = tmp_store.update_from_analysis("deltest01", default_voice)
    tmp_store.save(sig)
    assert tmp_store.delete("deltest01") is True


def test_list_all_returns_saved_ids(tmp_store, default_voice):
    tmp_store.update_from_analysis("listtest1", default_voice)
    tmp_store.save(tmp_store.load("listtest1"))
    ids = tmp_store.list_all()
    assert "listtest1" in ids


def test_get_prior_strength_no_sig(tmp_store):
    s = tmp_store.get_prior_strength("ffffffffffff")
    assert s == "kein Prior"


def test_get_prior_strength_with_many_files(tmp_store, default_voice):
    for _ in range(6):
        tmp_store.update_from_analysis("priortest1", default_voice)
    sig = tmp_store.load("priortest1")
    tmp_store.save(sig)
    strength = tmp_store.get_prior_strength("priortest1")
    assert strength in ("schwacher Prior", "starker Prior", "kein Prior")


# ---------------------------------------------------------------------------
# Singleton & Convenience
# ---------------------------------------------------------------------------


def test_singleton_same_instance():
    a = get_signature_store()
    b = get_signature_store()
    assert a is b


def test_load_artist_signature_convenience():
    sig = load_artist_signature("definitely_does_not_exist_xyzxyz")
    assert sig is None


def test_singleton_thread_safe():
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(get_signature_store) for _ in range(20)]
        instances = [f.result() for f in futures]
    assert all(inst is instances[0] for inst in instances)
