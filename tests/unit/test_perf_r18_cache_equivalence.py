"""§PERF-R18 (2026-10-28): Äquivalenz-Pinning der F-Reihe-Laufzeitoptimierungen.

Zwei bit-exakte Optimierungen werden gegen ihre Referenzen gepinnt:

1. ``_pyin_candidate_taus`` (phase_12_wow_flutter_fix) ≡ der alte
   Multi-Threshold-Doppel-Loop (Schleifen-Original als Referenz im Test).
2. Wow/Flutter-Feature-Cache (defect_scanner) ≡ Uncached-Pfad — auch bei
   geänderten Schwellwerten zwischen den Aufrufen (genau der Fall, der das
   Verdict-Memo in c4307207 verworfen hat), inkl. Scan-Ebenen-Gegenprobe.

Determinismus §G5 (copilot-instructions.md): gleicher Input ⇒ bit-identischer
Output, unabhängig vom Cache-Zustand. Kein librosa/GPU nötig.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.core import defect_scanner
from backend.core.defect_scanner import DefectScanner, DefectType, MaterialType
from backend.core.phases.phase_12_wow_flutter_fix import _pyin_candidate_taus

# ---------------------------------------------------------------------------
# 1) pYIN Multi-Threshold-Tau-Suche: Vektorisierung ≡ Schleifen-Original
# ---------------------------------------------------------------------------


def _reference_pyin_candidate_taus(
    cmnd: np.ndarray, min_period: int, max_period: int, thresholds: np.ndarray
) -> np.ndarray:
    """Referenz: der vor §PERF-R18 produktive Doppel-Loop (unverändert kopiert)."""
    found: list[int] = []
    for thr in thresholds:
        tau_est = 0
        for tau in range(min_period, max_period):
            if cmnd[tau] < thr and 0 < tau < max_period - 1:
                if cmnd[tau] <= cmnd[tau - 1] and cmnd[tau] <= cmnd[tau + 1]:
                    tau_est = tau
                    break
        found.append(tau_est)
    return np.array(found, dtype=np.int64)


def _pyin_case_arrays() -> list[tuple[str, np.ndarray, int, int]]:
    """Strukturierte cmnd-Fälle inkl. Randfälle der Schleifen-Bedingungen."""
    rng = np.random.default_rng(20261028)
    window = 1024
    cases: list[tuple[str, np.ndarray, int, int]] = []

    # Zufälliges cmnd (YIN-CMNDF-Bereich [0, 2])
    cases.append(("random", rng.uniform(0.0, 2.0, size=window), 48, min(960, window // 2)))

    # Einzelnes tiefes Minimum bei tau=100
    single = np.full(window, 1.5)
    single[100] = 0.2
    cases.append(("single_min", single, 48, min(960, window // 2)))

    # Plateau unter der Schwelle: <= macht jeden tau zum lokalen Minimum,
    # Break == erstes True == argmax ⇒ beide liefern tau=min_period
    cases.append(("plateau", np.full(window, 0.2), 48, min(960, window // 2)))

    # Zwei gleiche Minima unter der Schwelle ⇒ erstes gewinnt in beiden Pfaden
    two_min = np.full(window, 1.5)
    two_min[80] = 0.1
    two_min[300] = 0.1
    cases.append(("two_equal_minima", two_min, 48, min(960, window // 2)))

    # Nichts unter der Schwelle → Marker 0 (Fallback-Pfad)
    cases.append(("all_above", np.full(window, 5.0), 48, min(960, window // 2)))

    # Minimum auf max_period-2 (gefunden) ...
    edge_found = np.full(window, 1.5)
    edge_found[window // 2 - 2] = 0.1
    cases.append(("min_at_max_minus_2", edge_found, 48, window // 2))

    # ... vs. auf max_period-1 (von der Loop ausgeschlossen → Marker 0)
    edge_excl = np.full(window, 1.5)
    edge_excl[window // 2 - 1] = 0.1
    cases.append(("min_at_max_minus_1_excluded", edge_excl, 48, window // 2))

    # Minimum auf min_period (erstes gültiges tau)
    edge_lo = np.full(window, 1.5)
    edge_lo[48] = 0.1
    cases.append(("min_at_min_period", edge_lo, 48, window // 2))

    # min_period=0: alte Loop-Bedingung „0 < tau" schließt tau=0 aus
    edge_zero = np.full(window, 1.5)
    edge_zero[0] = 0.05
    edge_zero[7] = 0.1
    cases.append(("min_period_zero", edge_zero, 0, window // 2))

    # Degenerierter Bereich (leere Suche → Marker 0)
    cases.append(("empty_range", np.full(window, 1.5), 100, 50))

    return cases


@pytest.mark.parametrize("name,cmnd,min_period,max_period", _pyin_case_arrays())
def test_pyin_candidate_taus_bit_identical_to_reference_loop(
    name: str, cmnd: np.ndarray, min_period: int, max_period: int
) -> None:
    thresholds = np.linspace(0.05, 1.0, 20)
    ref = _reference_pyin_candidate_taus(cmnd, min_period, max_period, thresholds)
    fast = _pyin_candidate_taus(cmnd, min_period, max_period, thresholds)
    assert fast.tolist() == ref.tolist(), f"Vektorisierung weicht ab (Fall: {name})"


def test_pyin_candidate_taus_is_deterministic() -> None:
    """§G5 (copilot-instructions.md): gleicher Input ⇒ identischer Output."""
    _, cmnd, min_period, max_period = _pyin_case_arrays()[0]
    thresholds = np.linspace(0.05, 1.0, 20)
    a = _pyin_candidate_taus(cmnd, min_period, max_period, thresholds)
    b = _pyin_candidate_taus(cmnd, min_period, max_period, thresholds)
    assert a.tolist() == b.tolist()


# ---------------------------------------------------------------------------
# 2) Wow/Flutter-Feature-Cache: Treffer ≡ Neuberechnung (bit-exakt)
# ---------------------------------------------------------------------------

SR = 48000  # Pipeline-Rate: der volle scan() hat HF-Filter, die bei 16000 Hz fs/2 überschreiten


def _flut_signal(kind: str, seconds: float = 3.0) -> np.ndarray:
    """Deterministische Synthese-Signale (FM-Ton bzw. Rauschen)."""
    rng = np.random.default_rng(4711)
    t = np.arange(int(seconds * SR)) / SR
    if kind == "fm_tone":
        carrier = 220.0 * (1.0 + 0.005 * np.sin(2.0 * np.pi * 6.0 * t))
        sig = 0.5 * np.sin(2.0 * np.pi * np.cumsum(carrier) / SR)
        sig += 0.01 * rng.standard_normal(len(t))
    else:
        sig = 0.3 * rng.standard_normal(len(t))
    return np.asarray(sig, dtype=np.float64)


def _assert_score_equal(a, b, label: str) -> None:
    """Feldweise Bit-Identität eines DefectScore."""
    assert a.defect_type == b.defect_type, label
    assert a.severity == b.severity, f"{label}: severity {a.severity!r} != {b.severity!r}"
    assert a.confidence == b.confidence, f"{label}: confidence {a.confidence!r} != {b.confidence!r}"
    assert a.locations == b.locations, label
    assert a.metadata == b.metadata, f"{label}: metadata {a.metadata!r} != {b.metadata!r}"


@pytest.mark.parametrize("kind", ["fm_tone", "noise"])
def test_flutter_feature_cache_hit_is_bit_identical_across_thresholds(
    monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    """Cache-Treffer liefert dieselben Scores wie Neuberechnung — auch wenn die
    Schwellwerte zwischen den Aufrufen wechseln (D-K3-40-Szenario)."""
    audio = _flut_signal(kind)
    scanner = DefectScanner(sample_rate=SR)

    calls = {"n": 0}
    orig = DefectScanner._flutter_base_features

    def _counting(audio_arg, sample_rate_arg):
        calls["n"] += 1
        return orig(audio_arg, sample_rate_arg)

    monkeypatch.setattr(DefectScanner, "_flutter_base_features", staticmethod(_counting))

    # Referenz: uncached, je Schwelle neu gerechnet
    scanner._wf_feature_cache_enabled = False
    scanner.thresholds = {DefectType.FLUTTER: 0.3}
    ref_low = scanner._detect_flutter(audio)
    scanner.thresholds = {DefectType.FLUTTER: 0.9}
    ref_high = scanner._detect_flutter(audio)
    assert calls["n"] == 2

    # Cache-Pfad: erster Aufruf rechnet, zweiter Aufruf trifft
    scanner._wf_feature_cache_enabled = True
    scanner.thresholds = {DefectType.FLUTTER: 0.3}
    cached_low = scanner._detect_flutter(audio)
    scanner.thresholds = {DefectType.FLUTTER: 0.9}
    cached_high = scanner._detect_flutter(audio)
    assert calls["n"] == 3, "Feature-Schicht wurde trotz Cache neu gerechnet"

    _assert_score_equal(ref_low, cached_low, f"flutter/{kind}/thr=0.3")
    _assert_score_equal(ref_high, cached_high, f"flutter/{kind}/thr=0.9")


@pytest.mark.parametrize("kind", ["fm_tone", "noise"])
def test_wow_feature_cache_hit_is_bit_identical_across_thresholds(monkeypatch: pytest.MonkeyPatch, kind: str) -> None:
    audio = _flut_signal(kind)
    scanner = DefectScanner(sample_rate=SR)

    calls = {"n": 0}
    orig = DefectScanner._wow_track_features

    def _counting(audio_arg, sample_rate_arg):
        calls["n"] += 1
        return orig(audio_arg, sample_rate_arg)

    monkeypatch.setattr(DefectScanner, "_wow_track_features", staticmethod(_counting))

    scanner._wf_feature_cache_enabled = False
    scanner.thresholds = {DefectType.WOW: 0.2}
    ref_low = scanner._detect_wow(audio)
    scanner.thresholds = {DefectType.WOW: 0.8}
    ref_high = scanner._detect_wow(audio)
    assert calls["n"] == 2

    scanner._wf_feature_cache_enabled = True
    scanner.thresholds = {DefectType.WOW: 0.2}
    cached_low = scanner._detect_wow(audio)
    scanner.thresholds = {DefectType.WOW: 0.8}
    cached_high = scanner._detect_wow(audio)
    assert calls["n"] == 3, "Feature-Schicht wurde trotz Cache neu gerechnet"

    _assert_score_equal(ref_low, cached_low, f"wow/{kind}/thr=0.2")
    _assert_score_equal(ref_high, cached_high, f"wow/{kind}/thr=0.8")


def test_feature_cache_does_not_change_repeated_calls() -> None:
    """§G5 (copilot-instructions.md): warmer Cache ⇒ identischer Score."""
    audio = _flut_signal("fm_tone")
    scanner = DefectScanner(sample_rate=SR)
    scanner.thresholds = {DefectType.FLUTTER: 0.5, DefectType.WOW: 0.5}
    first_fl = scanner._detect_flutter(audio)
    first_wo = scanner._detect_wow(audio)
    second_fl = scanner._detect_flutter(audio)
    second_wo = scanner._detect_wow(audio)
    _assert_score_equal(first_fl, second_fl, "flutter/warm-cache")
    _assert_score_equal(first_wo, second_wo, "wow/warm-cache")


def test_scan_scores_bit_identical_with_and_without_feature_cache() -> None:
    """Scan-Ebenen-Gegenprobe: alle Scores identisch, Cache an vs. aus."""
    audio = _flut_signal("fm_tone")

    defect_scanner._scan_cache.clear()
    scanner_on = DefectScanner(sample_rate=SR, material_type=MaterialType.VINYL)
    scanner_on._wf_feature_cache_enabled = True
    result_on = scanner_on.scan(audio, SR)

    defect_scanner._scan_cache.clear()
    scanner_off = DefectScanner(sample_rate=SR, material_type=MaterialType.VINYL)
    scanner_off._wf_feature_cache_enabled = False
    result_off = scanner_off.scan(audio, SR)

    assert result_on.material_type == result_off.material_type
    assert set(result_on.scores.keys()) == set(result_off.scores.keys())
    for defect_type, score_on in result_on.scores.items():
        _assert_score_equal(result_off.scores[defect_type], score_on, f"scan/{defect_type}")


# ---------------------------------------------------------------------------
# 3) §PERF-R21d (2026-11-14): Material-Erkennungs-Wow/Flutter auf dem
#    geteilten 60-s-Center-Crop — ein FM-Lauf pro Scan statt zwei
# ---------------------------------------------------------------------------


def _long_signal(seconds: float = 65.0) -> np.ndarray:
    """Deterministisches Lang-Signal (> 60 s → Crop greift in beiden Pfaden)."""
    rng = np.random.default_rng(20261114)
    t = np.arange(int(seconds * SR)) / SR
    sig = 0.4 * np.sin(2.0 * np.pi * 440.0 * t)
    sig += 0.02 * rng.standard_normal(len(t))
    return np.asarray(sig, dtype=np.float64)


def test_wf_stationary_crop_matches_scan_crop_arithmetic() -> None:
    """Der Crop der Material-Erkennung muss exakt dieselben Samples liefern
    wie die §9.7.5-Center-Crop-Arithmetik des scan()-Hauptpfads — sonst greift
    der content-keyed FM-Cache (§PERF-R18) nicht."""
    scanner = DefectScanner(sample_rate=SR)
    mono = _long_signal()
    cap = scanner.WF_CROP_CAP_S * SR
    mid, half = len(mono) // 2, cap // 2
    expected = mono[mid - half : mid + half]
    got = scanner._wf_stationary_crop(mono)
    assert got.shape == expected.shape
    assert np.array_equal(got, expected)


def test_wf_stationary_crop_short_signal_passthrough() -> None:
    """Signale ≤ 60 s werden unverändert (identische Samples) durchgereicht."""
    scanner = DefectScanner(sample_rate=SR)
    mono = _long_signal(seconds=5.0)
    got = scanner._wf_stationary_crop(mono)
    assert len(got) == len(mono)
    assert np.array_equal(got, mono)


def test_material_detection_wow_flutter_uses_60s_crop(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mono- UND Stereo-Material-Erkennung rechnen Wow/Flutter auf dem
    60-s-Crop — damit ist der Cache-Schlüssel identisch zum Haupt-Scan."""
    seen: dict[str, list[int]] = {"mono_wow": [], "mono_flut": [], "stereo_wow": [], "stereo_flut": []}
    mono = _long_signal()

    class _LenScore:
        defect_type = DefectType.WOW

        def __init__(self, sev: float):
            self.severity = sev
            self.confidence = 0.5
            self.locations = []
            self.metadata = {}

    def _fake_det(which: str):
        def _fn(self, audio):
            seen[which].append(len(audio))
            return _LenScore(0.0)

        return _fn

    monkeypatch.setattr(DefectScanner, "_detect_wow", _fake_det("mono_wow"))
    monkeypatch.setattr(DefectScanner, "_detect_flutter", _fake_det("mono_flut"))
    scanner = DefectScanner(sample_rate=SR)
    scanner._detect_mono_material(mono)
    assert seen["mono_wow"] == [SR * scanner.WF_CROP_CAP_S]
    assert seen["mono_flut"] == [SR * scanner.WF_CROP_CAP_S]

    seen["stereo_wow"] = []
    seen["stereo_flut"] = []
    monkeypatch.setattr(DefectScanner, "_detect_wow", _fake_det("stereo_wow"))
    monkeypatch.setattr(DefectScanner, "_detect_flutter", _fake_det("stereo_flut"))
    stereo = np.stack([mono, mono[::-1]], axis=1)
    scanner._detect_stereo_material(stereo)
    assert seen["stereo_wow"] == [SR * scanner.WF_CROP_CAP_S]
    assert seen["stereo_flut"] == [SR * scanner.WF_CROP_CAP_S]


def test_material_detection_fills_fm_cache_for_scan_crop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Der Kern des Rätsels: Material-Erkennung rechnet die Wow/Flutter-
    Feature-Schicht auf dem Crop vor — der Haupt-Scan trifft danach den
    content-keyed Cache (Feature-Schicht genau EINMAL gerechnet statt zweimal).
    Ein ANDERES Array (anderer Inhalt) muss neu rechnen (kein Stale-Hit)."""
    mono = _long_signal()
    calls = {"wow": 0, "flutter": 0}

    _wow_orig = DefectScanner._wow_track_features
    _flut_orig = DefectScanner._flutter_base_features

    def _counting_wow(audio, sr):
        calls["wow"] += 1
        return _wow_orig(audio, sr)

    def _counting_flut(audio, sr):
        calls["flutter"] += 1
        return _flut_orig(audio, sr)

    monkeypatch.setattr(DefectScanner, "_wow_track_features", staticmethod(_counting_wow))
    monkeypatch.setattr(DefectScanner, "_flutter_base_features", staticmethod(_counting_flut))

    scanner = DefectScanner(sample_rate=SR)
    scanner._wf_feature_cache_enabled = True
    # Material-Erkennung: Feature-Schicht auf dem 60-s-Crop (je 1×)
    scanner._detect_mono_material(mono)
    assert calls["wow"] == 1
    assert calls["flutter"] == 1

    # Haupt-Scan-Aufrufe auf dem identischen Crop → Cache-Treffer, kein Nachrechnen
    crop = scanner._wf_stationary_crop(mono)
    scanner._detect_wow(crop)
    scanner._detect_flutter(crop)
    assert calls["wow"] == 1, "WOW-Feature-Schicht wurde trotz identischem Crop neu gerechnet"
    assert calls["flutter"] == 1, "FLUTTER-Feature-Schicht wurde trotz identischem Crop neu gerechnet"

    # Anderes Array (anderer Inhalt) → content-keyed: neuer Eintrag, Neuberechnung
    intro = mono[: SR * 20]
    scanner._detect_flutter(intro)
    assert calls["flutter"] == 2, "content-keyed Cache traf trotz anderem Array (Stale-Hit)"
