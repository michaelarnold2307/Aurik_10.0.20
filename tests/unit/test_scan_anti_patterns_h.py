"""Tests für die H-Serie des Anti-Pattern-Scanners (Hörordnungs-/Exportqualität)."""

from __future__ import annotations

import sys
from pathlib import Path

# Scanner liegt unter .agents/skills/bug-prevention/ — für Import verfügbar machen
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".agents" / "skills" / "bug-prevention"))

from scan_anti_patterns import check_hoerordnung_export_patterns as _check


def _run(source: str, path: str = "backend/core/phases/test_phase.py") -> list[str]:
    return _check(path, source)


def test_h01_naked_int16_flagged_dithered_not() -> None:
    bad = "audio = signal.astype(np.int16)\n"
    assert any("H01" in m for m in _run(bad))
    good = "audio = dither_powr3(signal).astype(np.int16)\n"
    assert not any("H01" in m for m in _run(good))


def test_h02_griffinlim_flagged() -> None:
    assert any("H02" in m for m in _run("out = griffinlim(mag)\n"))
    assert not any("H02" in m for m in _run("out = pghi_reconstruct(mag, sr)\n"))


def test_h03_sosfilt_additiv_flagged_analyse_nicht() -> None:
    """Norm (.github/VERBOTEN.md): zero-phase ist Pflicht, wo das Bandfilter-Ergebnis
    addiert wird; `sosfilt` bleibt für Analyse/Sidechain ausdrücklich zulässig."""
    # Additionspfad über eine Zwischenvariable → Meldung
    assert any("H03" in m for m in _run("band = sosfilt(sos, x)\nout = x + band\n", "backend/core/phases/p.py"))
    # Direkte Addition in derselben Zeile → Meldung
    assert any("H03" in m for m in _run("out = x + sosfilt(sos, x)\n", "backend/core/phases/p.py"))
    # Reiner Analyse-Pfad (Envelope/Energie) → keine Meldung
    assert not any("H03" in m for m in _run("band = sosfilt(sos, x)\nenv = np.abs(band)\n", "backend/core/phases/p.py"))
    # Crossover-Split-Sum (alle Additionspartner filter-abgeleitet) → normerlaubt
    _cross = "low = sosfilt(sos_l, x)\nhigh = sosfilt(sos_h, x)\nrec = low + high\n"
    assert not any("H03" in m for m in _run(_cross, "backend/core/dsp/crossover.py"))
    # Epsilon-/Messkontext (Energie + 1e-10, np.stack) → keine Meldung
    _meas = (
        "band = sosfilt(sos, x)\nnrg = np.sqrt(np.mean(band**2))\n"
        "db = 20 * np.log10(nrg + 1e-10)\nch = np.stack([band, band], axis=-1)\n"
    )
    assert not any("H03" in m for m in _run(_meas, "backend/core/dsp/measure.py"))
    # Add auf einen nicht-gefilterten Akkumulator bleibt ein Verstoß
    assert any("H03" in m for m in _run("band = sosfilt(sos, x)\nacc += band\n", "backend/core/dsp/crossover.py"))
    # Serielle Filterkette (kein Add aufs Original) → keine Meldung
    assert not any("H03" in m for m in _run("y = sosfilt(sos, y)\n", "backend/core/dsp/d.py"))
    # Analyse-Datei (kein Phasen/DSP-Pfad) → keine Meldung
    assert not any("H03" in m for m in _run("y = sosfilt(sos, x)\n", "backend/core/metrics.py"))
    # sosfiltfilt selbst ist kein Treffer
    assert not any("H03" in m for m in _run("y = sosfiltfilt(sos, x)\n", "backend/core/phases/p.py"))
    # Umgebrochene Zeile (ruff-format) mit Marker auf der Folgezeile → ausgenommen
    _wrapped = "y = sosfilt(sos, x).astype(\n    float\n)  # H-SCAN-EXEMPT: Allpass\n"
    assert not any("H03" in m for m in _run(_wrapped, "backend/core/dsp/a.py"))


def test_h04_ttl_lifecycle_not_flagged() -> None:
    """TTL-/Datei-Lebenszyklus-Entscheidungen brauchen Wall-Clock (§G5-Grenze).

    Persistierte Zeitstempel (Prüfpunkt-/Cache-Ablauf) sind nur mit Wall-Clock über
    Prozess-/Boot-Grenzen vergleichbar; dieselbe Abgrenzung führt der
    Code-Weakness-Scanner als `wallclock_ttl_housekeeping`. Der Befund 2026-10-05
    war ein Widerspruch zwischen beiden Gates.
    """
    _ttl = (
        "import os, time\nMAX_CHECKPOINT_AGE_S = 604800\n"
        "def cleanup_expired():\n    now = time.time()\n"
        "    for name in os.listdir('.'):\n"
        "        if time.time() - now > MAX_CHECKPOINT_AGE_S:\n            os.remove(name)\n"
    )
    assert not any("H04" in m for m in _run(_ttl, "backend/core/recovery.py"))


def test_h04_time_in_decision_flagged_profiling_not() -> None:
    assert any("H04" in m for m in _run("if time.time() > deadline:\n    break\n"))
    assert not any("H04" in m for m in _run("t0 = time.time()\n"))


def test_h05_resample_without_guard_flagged() -> None:
    bad = "from scipy import signal\ny = signal.resample(x, 48000)\n"
    assert any("H05" in m for m in _run(bad, "backend/core/dsp/p.py"))
    guarded = "if abs(len(x) - target) / target > 0.001:\n    pass\ny = signal.resample(x, target)\n"
    assert not any("H05" in m for m in _run(guarded, "backend/core/dsp/p.py"))
    # Ratio-/SR-basierte Konvertierungen sind strukturell zeitachsen-treu → kein Befund
    assert not any("H05" in m for m in _run("y = resample_poly(x, 4, 1)\n", "backend/core/dsp/p.py"))
    assert not any(
        "H05" in m for m in _run("y = librosa.resample(x, orig_sr=48000, target_sr=16000)\n", "backend/core/dsp/p.py")
    )
    # Ziel = Originallänge → kein Befund
    assert not any("H05" in m for m in _run("y = signal.resample(x, len(x))\n", "backend/core/dsp/p.py"))


def test_h06_hard_clamp_flagged_soft_knee_not() -> None:
    bad = "out = np.clip(audio, -1.0, 1.0)\n"
    assert any("H06" in m for m in _run(bad, "backend/core/phases/p.py"))
    soft = "out = soft_knee_limit(audio)\n"
    assert not any("H06" in m for m in _run(soft, "backend/core/phases/p.py"))


def test_h07_silent_except_flagged_logged_not() -> None:
    bad = "try:\n    x = ml()\nexcept Exception:\n    return 0.5\n"
    assert any("H07" in m for m in _run(bad))
    good = "try:\n    x = ml()\nexcept Exception:\n    logger.warning('fallback')\n    return 0.5\n"
    assert not any("H07" in m for m in _run(good))
    # Lokaler Log-Wrapper statt direktem logger-Aufruf (shellac_mono_strategy._audit_log)
    wrapper = (
        "def _audit_log(level, message):\n"
        "    {'error': logger.error, 'warn': logger.warning}.get(level, logger.info)(message)\n"
        "try:\n    x = ml()\nexcept Exception as e:\n    _audit_log('error', str(e))\n    return 0.0\n"
    )
    assert not any("H07" in m for m in _run(wrapper))


def test_p3_stft_noverlap_crash_boundary() -> None:
    """P3 prüft die echte scipy-Bedingung `noverlap >= nperseg`.

    Die alte Textsuche nach `noverlap=n_fft - hop` erzeugte 8 Fehlalarme in
    bandwidth_extension, hybrid_ml_denoiser und spectral_subtractor (alle drei
    belegt strukturell bzw. explizit abgesichert).
    """
    from scan_anti_patterns import check_stft_without_clamp as _p3

    # Crash-Form: noverlap ohne Subtraktion
    assert any("P3" in m for m in _p3("a.py", "X = stft(x, nperseg=n_fft, noverlap=n_fft)\n"))
    # Unguardiertes `- hop` bleibt ein Befund (Forensik-Muster Juli 2026)
    assert any("P3" in m for m in _p3("a.py", "X = stft(x, nperseg=n_fft, noverlap=n_fft - hop)\n"))
    # Expliziter Guard (hybrid_ml_denoiser-Muster) → kein Befund
    _guarded = "if n_fft <= hop:\n    n_fft = hop + 1\nX = stft(x, nperseg=n_fft, noverlap=n_fft - hop)\n"
    assert not any("P3" in m for m in _p3("a.py", _guarded))
    # Bewusster Retry-Pfad (spectral_subtractor-Muster) → kein Befund
    _retry = (
        "try:\n    X = stft(x, nperseg=n_fft, noverlap=n_fft - hop)\n"
        'except ValueError as e:\n    if "noverlap must be less than nperseg" in str(e):\n        pass\n'
    )
    assert not any("P3" in m for m in _p3("a.py", _retry))
    # Literal-Subtrahend ≥ 1 und Inline-Clamp → kein Befund
    assert not any("P3" in m for m in _p3("a.py", "X = stft(x, nperseg=n_fft, noverlap=n_fft - 512)\n"))
    assert not any("P3" in m for m in _p3("a.py", "X = stft(x, nperseg=n_fft, noverlap=min(n_fft - hop, n_fft - 1))\n"))
