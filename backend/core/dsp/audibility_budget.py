"""Audibility-Budget — Hörbarkeits-Überschuss als Obergrenze der Bearbeitungstiefe.

Hörordnung §4 (`.github/instructions/hoerordnung.instructions.md`): Ein Defekt
gilt als behoben, sobald sein Rest UNTER der psychoakustischen
Maskierungsschwelle liegt — nicht, wenn sein Messwert Null ist. Für jede
subtraktive Stufe folgt daraus ein Budget:

    budget_db[b, t] = max(0, L_defekt[b, t] - (T_signal[b, t] - marge_db))

* ``L_defekt`` ist der Pegel des Anteils, den die Stufe entfernen will
  (Rauschschätzung bzw. Sibilantenband).
* ``T_signal`` ist die Maskierungsschwelle des VERBLEIBENDEN Materials im
  selben Bark-Band/Frame (kanonisches Johnston-Modell aus
  ``backend/core/dsp/masking_model.py``, inkl. +6 dB Sicherheitsmarge).
* Das Budget ist die einzige Rechtfertigung für Dämpfung: tiefer als
  ``-budget_db`` zu dämpfen entfernt Material, ohne einen hörbaren Defekt zu
  beseitigen (Überunterdrückung, Lispeln, „tote Stille").

Anwendung: ``apply_gain_budget`` bzw. die beiden Fassaden
``spectrum_budget_db`` (STFT-Domäne, z. B. Phase 03) und
``band_budget_curve`` (Zeitdomäne, z. B. Phase 43) heben einen Gain nie über
1.0 an, sondern begrenzen ausschließlich die Dämpfung nach unten
(``G >= 10 ** (-budget/20)``). Bei ``budget_db == 0`` bleibt der Gain 1.0 —
der Anteil ist bereits maskiert und jede weitere Absenkung wäre unhörbar
teuer.

Abgrenzung (keine Doppelung, §G9 copilot-instructions.md):

* ``audibility_gate.py`` entscheidet VOR der Reparatur („Defekt hörbar?").
* ``audibility_targets.py`` nimmt NACH der Reparatur ab („Rest unter Ziel?").
* dieses Modul liefert die Stellgröße für die Reparatur selbst.

Flag: ``use_audibility_budget`` (Default AUS, bit-identisches
Bestandsverhalten). Aktivierung ausschließlich über §4-Verfahren
(``scripts/flag_freigabe.py``) nach B3-/Kompositions-Messung und Hör-Sign-off.

Determinismus §G5 (copilot-instructions.md): rein numpy/scipy, keine
Zufallszahlen, keine Zeitabhängigkeit. NaN-/Inf-sicher (§0a
copilot-instructions.md). Kein Kanal-Layout — alle Funktionen arbeiten auf
Monodaten bzw. auf einer bereits kanalreduzierten STFT-Matrix (F, T).

Speicher: Bandpegel und Schwellen entstehen frame-gechunkt; die
(F, T)-Gain-Anwendung läuft bandweise. Damit bleibt der Zusatzbedarf auch in
den hochauflösenden MRSA-Zonen (≈40k Frames) im zweistelligen MB-Bereich.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

#: Kanonische Sicherheitsmarge des Johnston-Modells (masking_model.py).
SAFETY_MARGIN_DB = 6.0
#: Flankensteilheit der Maskierungs-Ausbreitung (konservativ, Johnston 1988).
SPREAD_DB_PER_BARK = 10.0
#: Harte Obergrenze des Budgets — entspricht −60 dB erlaubter Dämpfung.
MAX_BUDGET_DB = 60.0

_LEVEL_FLOOR_DB = -200.0
_POWER_EPS = 1e-12
_FRAME_CHUNK = 2048  # Frames je Rechen-Chunk: (B, B, chunk) bleibt speicherarm


@dataclass(frozen=True)
class BudgetStats:
    """Aggregierte Wirkung eines Budget-Laufs (nur Beobachtung, kein Zustand)."""

    applied: bool
    mean_budget_db: float
    max_budget_db: float
    #: Anteil der (Band, Frame)-Paare mit budget > 0 (eine Obergrenze existiert).
    limited_fraction: float
    #: Anteil der Gain-Einträge, die tatsächlich angehoben wurden (nur Anwendung).
    bound_fraction: float
    #: Mittlere Anhebung in dB über die gebundenen Einträge (nur Anwendung).
    mean_lift_db: float

    def as_dict(self) -> dict[str, float | bool]:
        return {
            "applied": self.applied,
            "mean_budget_db": self.mean_budget_db,
            "max_budget_db": self.max_budget_db,
            "limited_fraction": self.limited_fraction,
            "bound_fraction": self.bound_fraction,
            "mean_lift_db": self.mean_lift_db,
        }


def _disabled_stats() -> BudgetStats:
    return BudgetStats(False, 0.0, 0.0, 0.0, 0.0, 0.0)


def audibility_budget_enabled() -> bool:
    """Liest ``use_audibility_budget`` defensiv; jeder Fehler ⇒ AUS (§V6 (copilot-instructions.md))."""
    try:
        from backend.core import music_model_flags as _flags

        return bool(getattr(_flags, "use_audibility_budget", False))
    except Exception as _exc:  # §V6 (copilot-instructions.md): nie blockierend
        logger.warning(
            "use_audibility_budget nicht lesbar (%s) — Audibility-Budget bleibt AUS (§V6, copilot-instructions.md)",
            _exc,
        )
        return False


def band_index_for_freqs(freqs_hz: np.ndarray, edges_hz: np.ndarray) -> np.ndarray:
    """Ordnet FFT-Bins den Bark-Bändern zu (letzte Kante inklusive Nyquist)."""
    f = np.asarray(freqs_hz, dtype=np.float64)
    e = np.asarray(edges_hz, dtype=np.float64)
    if e.size < 2:
        raise ValueError("bark_band_edges braucht mindestens zwei Kanten")
    idx = np.searchsorted(e, f, side="right") - 1
    clipped: np.ndarray = np.clip(idx, 0, e.size - 2).astype(np.int32)
    return clipped


def bark_band_centres_z(edges_hz: np.ndarray) -> np.ndarray:
    """Bark-Zentren (Traunmüller) der Bänder — identisch zu masking_model."""
    e = np.asarray(edges_hz, dtype=np.float64)
    centres = 0.5 * (e[:-1] + e[1:])
    z: np.ndarray = 13.0 * np.arctan(0.00076 * centres) + 3.5 * np.arctan((centres / 7500.0) ** 2)
    return z


def band_power_db(power: np.ndarray, band_idx: np.ndarray, n_bands: int) -> np.ndarray:
    """Summiert Leistung je Band/Frame und liefert Pegel in dB (B, T)."""
    p = np.asarray(power, dtype=np.float64)
    if p.ndim == 1:
        p = p[:, np.newaxis]
    if p.size:
        p = np.nan_to_num(p, copy=True, nan=0.0, posinf=0.0, neginf=0.0)
        np.maximum(p, 0.0, out=p)
    idx = np.asarray(band_idx, dtype=np.int32)
    out = np.zeros((int(n_bands), p.shape[1]), dtype=np.float64)
    for b in range(int(n_bands)):
        sel = idx == b
        if np.any(sel):
            out[b, :] = np.sum(p[sel, :], axis=0)
    levels: np.ndarray = 10.0 * np.log10(out + _POWER_EPS)
    return levels


def masking_threshold_db(
    band_level_db: np.ndarray,
    centres_z: np.ndarray,
    *,
    safety_margin_db: float = SAFETY_MARGIN_DB,
    spread_db_per_bark: float = SPREAD_DB_PER_BARK,
) -> np.ndarray:
    """Johnston-Maskierungsschwelle je Band/Frame (dB) — vektorisierter Kern.

    Gleiche Mathematik wie ``masking_model.compute_masking_threshold_db``
    (SFM-Tonalität, Offset ``alpha*(14.5+z) + (1-alpha)*5.5``, +6 dB Marge,
    10 dB/Bark beidseitige Ausbreitung), aber auf einer bereits vorliegenden
    Bandpegel-Matrix statt eines eigenen Zeitdomänen-STFT.

    Args:
        band_level_db: (B, T) Bandpegel in dB.
        centres_z: (B,) Bark-Zentren.
        safety_margin_db: Zusätzliche Marge (Default +6 dB, kanonisch).
        spread_db_per_bark: Flankensteilheit.

    Returns:
        (B, T) Schwelle in dB, NaN-frei.
    """
    levels = np.asarray(band_level_db, dtype=np.float64)
    if levels.ndim == 1:
        levels = levels[:, np.newaxis]
    levels = np.nan_to_num(levels, nan=_LEVEL_FLOOR_DB, posinf=_LEVEL_FLOOR_DB, neginf=_LEVEL_FLOOR_DB)
    n_bands, n_frames = levels.shape
    z = np.asarray(centres_z, dtype=np.float64)
    if z.size != n_bands:
        raise ValueError("centres_z und band_level_db müssen dieselbe Bandzahl haben")

    # Tonalität je Band aus der Zeitfolge (kanonische Konvention aus
    # masking_model: lineare Amplituden aus dB/20, SFM über die Zeit).
    alpha = np.zeros(n_bands, dtype=np.float64)
    if n_frames > 2:
        lin = 10.0 ** (levels / 20.0)
        gm = np.exp(np.mean(np.log(lin + 1e-12), axis=1))
        am = np.mean(lin, axis=1)
        sfm = 10.0 * np.log10(np.clip(gm / (am + 1e-12), 1e-12, 1.0))
        alpha = np.clip(sfm / -60.0, 0.0, 1.0)

    offset = alpha * (14.5 + z) + (1.0 - alpha) * 5.5 + float(safety_margin_db)
    dz = np.abs(np.arange(n_bands)[:, np.newaxis] - np.arange(n_bands)[np.newaxis, :]).astype(np.float64)

    threshold = np.empty((n_bands, n_frames), dtype=np.float64)
    for start in range(0, n_frames, _FRAME_CHUNK):
        end = min(n_frames, start + _FRAME_CHUNK)
        chunk = levels[:, start:end]
        # (Zielband, Maskerband, Frames): Masker j hebt Ziel b um E_j - 10*|b-j| an.
        spread = chunk[np.newaxis, :, :] - spread_db_per_bark * dz[:, :, np.newaxis]
        threshold[:, start:end] = np.max(spread, axis=1) - offset[:, np.newaxis]

    result: np.ndarray = np.nan_to_num(threshold, nan=_LEVEL_FLOOR_DB, posinf=_LEVEL_FLOOR_DB, neginf=_LEVEL_FLOOR_DB)
    return result


def audibility_budget_db(
    defect_band_db: np.ndarray,
    threshold_band_db: np.ndarray,
    *,
    margin_db: float = 3.0,
    max_budget_db: float = MAX_BUDGET_DB,
) -> np.ndarray:
    """Erlaubte Dämpfung je Band/Frame (dB) — nie negativ, nie unbegrenzt.

    ``budget = max(0, L_defekt - (T_signal - margin_db))``: genau so viel
    Dämpfung, dass der Defekt ``margin_db`` unter der Maske landet.
    """
    defect = np.nan_to_num(
        np.asarray(defect_band_db, dtype=np.float64),
        nan=_LEVEL_FLOOR_DB,
        posinf=_LEVEL_FLOOR_DB,
        neginf=_LEVEL_FLOOR_DB,
    )
    threshold = np.nan_to_num(
        np.asarray(threshold_band_db, dtype=np.float64),
        nan=_LEVEL_FLOOR_DB,
        posinf=_LEVEL_FLOOR_DB,
        neginf=_LEVEL_FLOOR_DB,
    )
    budget = defect - (threshold - float(margin_db))
    clipped: np.ndarray = np.clip(budget, 0.0, float(max_budget_db))
    return clipped


def gain_floor_from_budget(budget_db: np.ndarray, *, max_budget_db: float = MAX_BUDGET_DB) -> np.ndarray:
    """Linearer Gain-Boden ``10 ** (-budget/20)`` zu einem dB-Budget."""
    b = np.nan_to_num(np.asarray(budget_db, dtype=np.float64), nan=0.0, posinf=float(max_budget_db), neginf=0.0)
    b = np.clip(b, 0.0, float(max_budget_db))
    floor: np.ndarray = 10.0 ** (-b / 20.0)
    return floor


def _resample_time(matrix: np.ndarray, n_frames: int) -> np.ndarray:
    """Lineare Zeit-Interpolation einer (B, T)-Matrix auf ``n_frames``."""
    mat: np.ndarray = np.asarray(matrix, dtype=np.float64)
    if mat.shape[1] == n_frames:
        return mat
    if mat.shape[1] == 0:
        ones: np.ndarray = np.ones((mat.shape[0], n_frames), dtype=np.float64)
        return ones
    src = np.linspace(0.0, 1.0, mat.shape[1])
    dst = np.linspace(0.0, 1.0, n_frames)
    out: np.ndarray = np.empty((mat.shape[0], n_frames), dtype=np.float64)
    for i in range(mat.shape[0]):
        out[i, :] = np.interp(dst, src, mat[i, :])
    return out


def apply_gain_budget(
    gain: np.ndarray,
    band_idx: np.ndarray,
    budget_db: np.ndarray,
) -> tuple[np.ndarray, BudgetStats]:
    """Begrenzt eine (F, T)-Gain-Matrix auf ``G >= 10 ** (-budget/20)``.

    Hebt ausschließlich an (nie über 1.0); bei Fehlern bleibt der Gain
    unverändert und ``applied=False`` (§V6 (copilot-instructions.md)). Die Anwendung läuft bandweise,
    damit kein zusätzliches (F, T)-Bodenarray entsteht.
    """
    try:
        g = np.asarray(gain, dtype=np.float64)
        if g.ndim != 2 or g.size == 0:
            return np.asarray(gain), _disabled_stats()
        budget = np.atleast_2d(np.asarray(budget_db, dtype=np.float64))
        if budget.shape[1] != g.shape[1]:
            budget = _resample_time(budget, g.shape[1])
        idx = np.asarray(band_idx, dtype=np.int32)
        if idx.size != g.shape[0]:
            logger.warning(
                "Gain-Begrenzung: Band-Index (%d) passt nicht zu Gain (%d) — unverändert (§V6 (copilot-instructions.md))",
                idx.size,
                g.shape[0],
            )
            return np.asarray(gain), _disabled_stats()

        out = np.array(g, dtype=np.float64, copy=True)
        floor_db = gain_floor_from_budget(budget)
        bound = np.zeros(g.shape, dtype=bool)
        for b in range(floor_db.shape[0]):
            sel = idx == b
            if not np.any(sel):
                continue
            floor_b = floor_db[b, :][np.newaxis, :]
            raised = out[sel, :] < floor_b
            if np.any(raised):
                out[sel, :] = np.where(raised, floor_b, out[sel, :])
                bound[sel, :] = raised

        n_bound = int(np.count_nonzero(bound))
        if n_bound:
            mean_lift = float(np.mean(20.0 * np.log10(np.maximum(out[bound], 1e-12) / np.maximum(g[bound], 1e-12))))
        else:
            mean_lift = 0.0
        stats = BudgetStats(
            applied=True,
            mean_budget_db=round(float(np.mean(budget)), 3),
            max_budget_db=round(float(np.max(budget)), 3),
            limited_fraction=round(float(np.mean(budget > 0.0)), 4),
            bound_fraction=round(n_bound / float(g.size), 4),
            mean_lift_db=round(mean_lift, 3),
        )
        dtype = np.float32 if np.asarray(gain).dtype == np.float32 else np.float64
        return np.asarray(out, dtype=dtype), stats
    except Exception as _exc:  # §V6 (copilot-instructions.md): nie blockierend
        logger.warning("Gain-Begrenzung fehlgeschlagen (%s) — Gain unverändert (§V6 (copilot-instructions.md))", _exc)
        return np.asarray(gain), _disabled_stats()


def _bark_setup(sr: int, freqs_hz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    from backend.core.dsp.masking_model import bark_band_edges

    edges = bark_band_edges(sr)
    band_idx = band_index_for_freqs(freqs_hz, edges)
    centres_z = bark_band_centres_z(edges)
    return edges, band_idx, centres_z, int(edges.size - 1)


def spectrum_budget_db(
    magnitude: np.ndarray,
    noise_magnitude: np.ndarray,
    freqs_hz: np.ndarray,
    sr: int,
    *,
    margin_db: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, BudgetStats]:
    """dB-Budget (B, T) aus STFT-Magnitude und Rauschschätzung.

    Masker ist die per spektraler Subtraktion geschätzte Signalschätzung
    ``max(|Y|² - |N|², 0)`` — NICHT das Gemisch, sonst maskiert das Rauschen
    sich selbst und eine rauschdominierte Stille dürfte nicht ausreichend
    abgesenkt werden. Defekt ist die Rauschschätzung ``|N|²``.

    Returns:
        (budget (B, T), band_idx (F,), Stats). Fehler ⇒ Budget 0 (kein Eingriff).
    """
    mag = np.asarray(magnitude, dtype=np.float64)
    noise = np.asarray(noise_magnitude, dtype=np.float64)
    if mag.ndim != 2 or noise.ndim != 2:
        raise ValueError("magnitude/noise_magnitude müssen (F, T) sein")
    n_f = min(mag.shape[0], noise.shape[0], np.asarray(freqs_hz).size)
    n_t = min(mag.shape[1], noise.shape[1])
    _edges, band_idx, centres_z, n_bands = _bark_setup(sr, np.asarray(freqs_hz)[:n_f])
    defect_db = np.empty((n_bands, n_t), dtype=np.float64)
    signal_db = np.empty((n_bands, n_t), dtype=np.float64)
    for start in range(0, n_t, _FRAME_CHUNK):
        end = min(n_t, start + _FRAME_CHUNK)
        m = np.nan_to_num(mag[:n_f, start:end], copy=True, nan=0.0, posinf=0.0, neginf=0.0)
        nz = np.nan_to_num(noise[:n_f, start:end], copy=True, nan=0.0, posinf=0.0, neginf=0.0)
        np.maximum(m, 0.0, out=m)
        np.maximum(nz, 0.0, out=nz)
        power = m * m
        noise_power = nz * nz
        defect_db[:, start:end] = band_power_db(noise_power, band_idx, n_bands)
        signal_power = power - noise_power
        np.maximum(signal_power, 0.0, out=signal_power)
        signal_db[:, start:end] = band_power_db(signal_power, band_idx, n_bands)
    threshold_db = masking_threshold_db(signal_db, centres_z)
    budget = audibility_budget_db(defect_db, threshold_db, margin_db=margin_db)
    stats = BudgetStats(
        applied=True,
        mean_budget_db=round(float(np.mean(budget)), 3),
        max_budget_db=round(float(np.max(budget)), 3),
        limited_fraction=round(float(np.mean(budget > 0.0)), 4),
        bound_fraction=0.0,
        mean_lift_db=0.0,
    )
    return budget, band_idx, stats


def spectrum_gain_floor(
    magnitude: np.ndarray,
    noise_magnitude: np.ndarray,
    freqs_hz: np.ndarray,
    sr: int,
    *,
    margin_db: float = 3.0,
) -> tuple[np.ndarray, BudgetStats]:
    """Bequemlichkeits-Fassade: (F, T)-Gain-Boden aus :func:`spectrum_budget_db`.

    Für große Zonen ist ``spectrum_budget_db`` + ``apply_gain_budget``
    speicherschonender; diese Fassade dient Tests und kleinen Eingaben.
    Fehler ⇒ floor = 1 (kein Eingriff).
    """
    try:
        budget, band_idx, stats = spectrum_budget_db(magnitude, noise_magnitude, freqs_hz, sr, margin_db=margin_db)
        ones = np.ones((band_idx.size, budget.shape[1]), dtype=np.float32)
        floor, applied = apply_gain_budget(ones, band_idx, budget)
        return np.asarray(floor, dtype=np.float64), BudgetStats(
            True,
            stats.mean_budget_db,
            stats.max_budget_db,
            stats.limited_fraction,
            applied.bound_fraction,
            applied.mean_lift_db,
        )
    except Exception as _exc:  # §V6 (copilot-instructions.md): nie blockierend
        logger.warning(
            "spectrum_gain_floor fehlgeschlagen (%s) — kein Budget-Eingriff (§V6, copilot-instructions.md)", _exc
        )
        return np.ones_like(np.asarray(magnitude), dtype=np.float64), _disabled_stats()


def _to_mono(x: np.ndarray) -> np.ndarray:
    """Layout-agnostische Monosumme (Stereo-Invariante des Repos)."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr.mean(axis=0) if arr.shape[0] <= 2 else arr.mean(axis=1)
    clean: np.ndarray = np.nan_to_num(arr.ravel(), nan=0.0, posinf=0.0, neginf=0.0)
    return clean


def band_budget_curve(
    signal: np.ndarray,
    sr: int,
    band_hz: tuple[float, float],
    *,
    margin_db: float = 3.0,
    n_fft: int = 2048,
) -> tuple[np.ndarray, BudgetStats]:
    """Sample-genauer Gain-Boden für eine Zeitdomänen-Stufe (z. B. De-Esser).

    Masker ist das Spektrum AUSSERHALB des Defektbands (das Sibilantenband
    wird für die Schwellenberechnung ausgenullt — kein Selbstmaskieren).
    Defekt ist die Energie IM Band. Der Boden ist das Maximum des Budgets
    über alle überlappenden Bark-Bänder (der hörbarste Überschuss bestimmt
    die Rechtfertigung der Banddämpfung).

    Returns:
        (floor (N,) in [0, 1] bei ``sr``, Stats). Fehler ⇒ floor = 1.
    """
    try:
        from backend.core.audio_utils import safe_stft

        mono = _to_mono(signal)
        n = mono.size
        if n < max(16, n_fft):
            return np.ones(n, dtype=np.float64), _disabled_stats()

        hop = max(1, n_fft // 4)
        freqs, times, z = safe_stft(mono, fs=sr, nperseg=n_fft, noverlap=n_fft - hop, boundary="even")
        mag = np.abs(np.asarray(z, dtype=np.complex128))
        power = mag**2

        lo, hi = float(min(band_hz)), float(max(band_hz))
        lo = max(0.0, lo)
        hi = min(float(sr) / 2.0, hi)
        defect_bins = (freqs >= lo) & (freqs <= hi)
        if not np.any(defect_bins):
            return np.ones(n, dtype=np.float64), _disabled_stats()

        _edges, band_idx, centres_z, n_bands = _bark_setup(sr, freqs)
        masker_power = power * (~defect_bins)[:, np.newaxis]
        signal_db = band_power_db(masker_power, band_idx, n_bands)
        defect_db = band_power_db(power * defect_bins[:, np.newaxis], band_idx, n_bands)
        threshold_db = masking_threshold_db(signal_db, centres_z)
        budget = audibility_budget_db(defect_db, threshold_db, margin_db=margin_db)

        band_has_defect = np.bincount(band_idx[defect_bins], minlength=n_bands) > 0
        if not np.any(band_has_defect):
            return np.ones(n, dtype=np.float64), _disabled_stats()
        budget_curve = np.max(budget[band_has_defect, :], axis=0)
        floor_frames = gain_floor_from_budget(budget_curve)

        centres_samples = np.asarray(times, dtype=np.float64) * float(sr)
        floor = np.interp(np.arange(n, dtype=np.float64), centres_samples, floor_frames)
        floor = np.clip(np.nan_to_num(floor, nan=1.0, posinf=1.0, neginf=1.0), 0.0, 1.0)
        stats = BudgetStats(
            applied=True,
            mean_budget_db=round(float(np.mean(budget_curve)), 3),
            max_budget_db=round(float(np.max(budget_curve)), 3),
            limited_fraction=round(float(np.mean(budget_curve > 0.0)), 4),
            bound_fraction=round(float(np.mean(floor < 0.999)), 4),
            mean_lift_db=0.0,
        )
        return np.asarray(floor), stats
    except Exception as _exc:  # §V6 (copilot-instructions.md): nie blockierend
        logger.warning(
            "band_budget_curve fehlgeschlagen (%s) — kein Budget-Eingriff (§V6, copilot-instructions.md)", _exc
        )
        return np.ones(np.asarray(signal).size, dtype=np.float64), _disabled_stats()


__all__ = [
    "MAX_BUDGET_DB",
    "SAFETY_MARGIN_DB",
    "SPREAD_DB_PER_BARK",
    "BudgetStats",
    "apply_gain_budget",
    "audibility_budget_db",
    "audibility_budget_enabled",
    "band_budget_curve",
    "band_index_for_freqs",
    "band_power_db",
    "bark_band_centres_z",
    "gain_floor_from_budget",
    "masking_threshold_db",
    "spectrum_budget_db",
    "spectrum_gain_floor",
]
