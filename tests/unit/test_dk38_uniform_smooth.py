"""§PERF-D-K3-38-Vertrag: ``_uniform_smooth`` gegen die NumPy-Referenz.

Teilbehebung 4 des Defekt-Scanner-Budgets (D-K3-38/D-K3-40): die gleichmäßige
Hüllkurven-Glättung ``np.convolve(x, ones(size)/size)`` (O(N·size)) wird durch
den O(N)-Leitungsfad ``scipy.ndimage.uniform_filter1d`` ersetzt. Der Ersatz ist
NICHT bit-identisch (laufende Summe statt Direkt-Faltung), aber detektor-
äquivalent — dieses Modul pinnt genau das:

* ``same`` und ``valid`` liefern innerhalb ``1e-12`` das Ergebnis der
  NumPy-Referenz (gemessene Produktionsabweichung 3,3e-15, ≈ 10 Größen-
  ordnungen unter dem CD-Rauschboden −96 dBFS);
* die crackle-Maske kippt auf dem deterministischen Prüfsignal bei keiner
  Schwellenspanne (0,5×/0,7×/1,0× des 95. Perzentils) — die Detektor-
  Entscheidung bleibt identisch;
* Randfälle (``size == 1``, Fenster länger als das Signal) gehen durch die
  NumPy-Referenz und sind bit-identisch;
* Determinismus (§G5 (copilot-instructions.md)): gleicher Input ⇒ bit-
  identisches Ergebnis über Aufrufe hinweg.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.core.defect_scanner import _uniform_smooth


def _ref(x: np.ndarray, size: int, mode: str) -> np.ndarray:
    return np.convolve(np.asarray(x, dtype=np.float64), np.ones(size) / size, mode=mode)


@pytest.mark.parametrize("mode", ["same", "valid"])
@pytest.mark.parametrize(
    ("n", "w"),
    [
        (1000, 5),  # Signal länger, ungerades Fenster
        (1000, 50),  # Signal länger, gerades Fenster
        (64, 4096),  # Fenster länger als Signal (Randfall → Referenz)
        (4096, 1),  # Fenster Länge 1 (Randfall → Referenz)
        (65, 5),  # ungerade Signallänge
        (64, 6),  # gerade Kombination (zentrierungs-heikel)
        (33, 33),  # Fenster == Signal
        (200, 199),  # fast gleich lang
    ],
)
def test_uniform_smooth_ist_numerisch_gleich_zu_numpy(mode: str, n: int, w: int) -> None:
    x = np.random.default_rng(42).normal(0.0, 0.3, size=n).astype(np.float64)
    got = _uniform_smooth(x, w, mode=mode)
    want = _ref(x, w, mode)
    assert got.shape == want.shape, f"Längen-Divergenz {got.shape} != {want.shape} (n={n}, w={w})"
    if want.size == 0:
        assert got.size == 0
        return
    assert np.allclose(got, want, rtol=1e-12, atol=1e-12), (
        f"max|Δ|={np.max(np.abs(got - want)):.3e} überschreitet 1e-12 (n={n}, w={w}, mode={mode})"
    )


def test_uniform_smooth_produktionsparameter_crackle_huelle() -> None:
    """30-s-Produktionsfixture-Größen: 44,1 kHz, 50-ms-Fenster (2205)."""
    sr = 44100
    w = int(0.05 * sr)
    n = 30 * sr
    rng = np.random.default_rng(42)
    # Hüllkurven-ähnlich: halbwellen-rectifizierte Sinusmischung + Rauschen
    t = np.arange(n) / sr
    env = np.abs(0.4 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 97 * t)) + 0.05 * rng.normal(size=n)
    got = _uniform_smooth(env, w, mode="same")
    want = _ref(env, w, mode="same")
    assert np.max(np.abs(got - want)) < 1e-12

    # Detektor-Maske (Muster _detect_crackle) über die Schwellenspanne
    for scale in (0.5, 0.7, 1.0):
        thr_got = scale * np.percentile(got, 95)
        thr_want = scale * np.percentile(want, 95)
        m_got = got > thr_got
        m_want = want > thr_want
        assert int(np.sum(m_got != m_want)) == 0, f"Maske kippt bei scale={scale}"


def test_uniform_smooth_valid_liefert_volle_ueberlappungslaenge() -> None:
    x = np.random.default_rng(7).normal(size=500)
    for w in (5, 6, 51, 2205):
        if w > x.size:
            continue
        out = _uniform_smooth(x, w, mode="valid")
        assert out.shape == (x.size - w + 1,)


def test_uniform_smooth_deterministisch() -> None:
    x = np.random.default_rng(1).normal(size=4096)
    a = _uniform_smooth(x, 64)
    b = _uniform_smooth(x.copy(), 64)
    assert np.array_equal(a, b)


def test_uniform_smooth_randfaelle_sind_bit_identisch() -> None:
    x = np.random.default_rng(3).normal(size=100)
    # size == 1 und size > len(x) laufen über die NumPy-Referenz
    assert np.array_equal(_uniform_smooth(x, 1), _ref(x, 1, "same"))
    assert np.array_equal(_uniform_smooth(x, 1, mode="valid"), _ref(x, 1, "valid"))
    assert np.array_equal(_uniform_smooth(x, 4096), _ref(x, 4096, "same"))
    assert np.array_equal(_uniform_smooth(x, 4096, mode="valid"), _ref(x, 4096, "valid"))


def test_uniform_smooth_valid_schicht_um_gleich_zu_numpy() -> None:
    """Das Slice-Muster (off = (w-1)//2) reproduziert np.convolve(valid)."""
    rng = np.random.default_rng(11)
    for n, w in ((300, 5), (300, 6), (301, 7), (2048, 256), (2048, 257)):
        x = rng.normal(size=n)
        got = _uniform_smooth(x, w, mode="valid")
        want = _ref(x, w, mode="valid")
        assert np.allclose(got, want, rtol=1e-12, atol=1e-12), f"n={n} w={w}: max|Δ|={np.max(np.abs(got - want)):.3e}"


def test_uniform_smooth_valid_schleift_fehler_bei_ungueltigem_mode() -> None:
    with pytest.raises(ValueError, match="mode"):
        _uniform_smooth(np.zeros(16), 4, mode="mirror")
    with pytest.raises(ValueError, match="size"):
        _uniform_smooth(np.zeros(16), 0)


def test_uniform_smooth_erhaelt_nicht_negativitaet() -> None:
    """Invariante: nicht-negativer Input ⇒ nicht-negatives Ergebnis.

    Die laufende Summe der O(N)-Leitung driftet bei Null-Läufen (Dropouts)
    um ~1e-15 unter Null, die Direkt-Faltung liefert exakt ≥ 0. Ohne Clamp
    kippt die Drift in eine NaN-Kaskade der Folgestufen (Produktionsbefund:
    ``sqrt(_npow_s + 1e-20)`` in ``_detect_modulation_noise`` — Severity NaN,
    gefunden von ``test_no_nan_severities``).
    """
    rng = np.random.default_rng(42)
    for w in (12, 13, 64, 2205):
        x = np.abs(rng.normal(size=8192))
        x[1000:1400] = 0.0  # Dropout-Lauf wie im Gold-Standard-Fixture
        for mode in ("same", "valid"):
            got = _uniform_smooth(x, w, mode=mode)
            assert bool(np.all(got >= 0.0)), f"negativer Drift (w={w}, mode={mode}): min={float(got.min()):.3e}"
            want = _ref(x, w, mode)
            assert np.allclose(got, want, rtol=1e-12, atol=1e-12), (
                f"Äquivalenz verloren (w={w}, mode={mode}): max|Δ|={np.max(np.abs(got - want)):.3e}"
            )


def test_uniform_smooth_nicht_negativitaet_auch_im_randfall() -> None:
    """Auch der NumPy-Referenz-Pfad (size==1 bzw. Fenster > Signal) bleibt ≥ 0."""
    rng = np.random.default_rng(7)
    x = np.abs(rng.normal(size=100))
    for w in (1, 4096):
        for mode in ("same", "valid"):
            got = _uniform_smooth(x, w, mode=mode)
            assert bool(np.all(got >= 0.0))


def test_uniform_smooth_regression_modulation_noise_kein_nan() -> None:
    """Ende-zu-Ende: Dropout-Signal ⇒ modulation_noise-Score endlich (kein NaN)."""
    import math

    from backend.core.defect_scanner import DefectScanner, DefectType, MaterialType

    sr = 44100
    t = np.arange(3 * sr) / sr
    audio = 0.4 * np.sin(2 * np.pi * 440.0 * t) + 0.2 * np.sin(2 * np.pi * 97.0 * t)
    audio[5 * sr // 3 : 5 * sr // 3 + 2000] = 0.0  # Dropout-Läufe
    audio[7 * sr // 4 : 7 * sr // 4 + 3000] = 0.0
    scanner = DefectScanner(sample_rate=sr)
    score = scanner._detect_modulation_noise(audio)
    assert math.isfinite(score.severity), f"Severity ist NaN/Inf: {score.severity}"
    assert math.isfinite(score.confidence), f"Confidence ist NaN/Inf: {score.confidence}"
    assert score.defect_type == DefectType.MODULATION_NOISE
    assert 0.0 <= score.severity <= 1.0
    # Der Fix wirkt an der Ursache — der Score bleibt auch ohne Clamp-Drift
    # über den vollen Scan endlich (Material-Adaption TAPE wie im Gold-Standard).
    result = scanner.scan(audio, sr, MaterialType.TAPE)
    for dt, sc in result.scores.items():
        assert math.isfinite(sc.severity), f"{dt.value}: Severity ist NaN/Inf"
