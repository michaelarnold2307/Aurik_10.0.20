"""§SR-CK3 (Nutzerbefund 2026-09-24): SOTA-Zeitbereich-Fein-Declicker für
kontinuierliches Knistern — iZotope-RX-De-crackle-Klasse, Godsill & Rayner 1998.

Hörbefund: Kontinuierliches feines Knistern ist KEIN Frequenz-Floor, sondern
eine dichte Textur aus Mikro-Impulsen (≤2 ms) in der Zeit. Spektrale
Floor-Ansätze (MCRA/OMLSA) und blinde Breitband-Enhancer (BANQUET-Klasse)
schätzen den stationären Floor — die Impuls-Bursts sind Zeit-Ausreißer und
überleben jede Floor-Subtraktion (Produktionsbefund: OMLSA −1,3 dB
Breitband, Knistern-Severity unverändert). Die SOTA-korrekte Klasse ist der
Fein-Declicker im Zeitbereich mit signalmodell-basierter Restaurierung.

Pipeline (deterministisch, kein ML, kein RNG):
1. Betonung: 4.-Ordnung-Hochpass (2,2 kHz, zero-phase) — Knistern ist
   HF-gewichtet; tonale LF-Träger werden unterdrückt.
2. Robuste Lokal-Statistik: laufender Median der HF-Hüllkurve (12 ms) als
   Baseline + lokale MAD-Skala (1,4826) — die Schwelle folgt dem lokalen
   Pegel, ohne dass das dichte Knistern die Baseline anhebt (Median ist bis
   50 % Ausreißer-Dichte robust).
3. Primär-Detektion: Hüllkurve > Median + 4,5·MAD → Impuls-Kandidat.
4. Musik-Onset-Veto (SOTA-Kern, RX-äquivalent): Kandidat nur, wenn
   (a) Dauer ≤ 2 ms (Impuls-Kompaktheit),
   (b) keine nachfolgende HF-Energie (Sustain-Test — Musik-Transienten
       klingen nach, Knistern nicht),
   (c) kein Vollband-Energiesprung vor/nach dem Impuls (Drum-/Sibilant-Veto
       — Knistern liegt AUF dem laufenden Musikpegel und ändert ihn nicht).
5. Kontext-Sensitivität (RX „adaptive sensitivity"): sekundäre Kandidaten
   (Median + 3,0·MAD) werden nur im Knistern-Kontext akzeptiert (Primär-
   Ereignis im ±30-ms-Fenster oder Primär-Dichte > 20/s) — dichtes feines
   Knistern wird erreicht, isolierte Musik-HF bleibt geschützt.
6. Span-Verfeinerung: Impuls-Randbereiche werden bis zum Abklingen
   (0,5 ms je Rand, 0,75·MAD) mitgenommen — kein Rest-Tick bleibt stehen.
7. Restaurierung nach Godsill & Rayner (1998): AR(p)-Interpolation —
   Vorwärts-Prädiktor aus dem sauberen Fenster links, Rückwärts-Prädiktor
   aus dem Fenster rechts, Least-Squares über den Impuls-Span (kein
   Mittelwert-Klötzchen, stetige Anschlüsse an die sauberen Nachbarn);
   kurze Spans (≤4 Samples) über Catmull-Rom-Interpolation.
8. Stereo: Detektion auf Mid (Mittelwert aller Kanäle) — Knistern ist
   kohärent; die Restaurierung läuft pro Kanal auf den eigenen Samples —
   Stereobreite und L/R-Timing bleiben erhalten.

Deterministisch (§G5 (copilot-instructions.md)), NaN-sicher, fail-open:
jeder Abbruchgrund wird geloggt (§V6 (copilot-instructions.md)).
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import signal as _signal
from scipy.ndimage import median_filter as _median_filter

logger = logging.getLogger(__name__)

# ── Konstanten (psychoakustisch kalibriert, §G6 (copilot-instructions.md)) ──
_HP_FREQ_HZ = 2200.0  # Betonungs-Hochpass: Knistern ist HF-gewichtet
_MEDIAN_WIN_S = 0.012  # robuste Lokal-Baseline der HF-Hüllkurve
_MAD_K = 1.4826  # MAD→σ-Konsistenzfaktor (Normalverteilung)
_PRIMARY_K = 4.5  # Primär-Schwelle (σ)
_SECONDARY_K = 3.0  # Sekundär-Schwelle (σ, nur im Knistern-Kontext)
_REFINE_K = 0.75  # Abkling-Schwelle der Span-Verfeinerung (σ)
_REFINE_HALF_S = 0.0005  # max. Verfeinerung je Rand
_MAX_SPAN_S = 0.002  # Impuls-Kompaktheit (Musik-Transienten sind länger)
_SUSTAIN_S = 0.008  # Sustain-Test-Fenster (Nachklang-Veto)
_CONTEXT_S = 0.030  # Knistern-Kontext für Sekundär-Kandidaten
_DENSE_EVENTS_PER_S = 20.0  # Dichte-Regime: Kontext entbehrlich
_SUSTAIN_RATIO = 2.5  # HF-Nachklang > 2,5× Baseline ⇒ Musik-Transient
_ONSET_RATIO = 2.0  # Vollband-Sprung > 2× ⇒ Musik-Onset
_ONSET_ABS = 0.01  # absoluter Vollband-Floor des Onset-Vetos
_ONSET_WINDOW_S = 0.001  # Vollband-Fenster (1 ms) für den Onset-Vergleich
_MIN_LEN_S = 0.1  # kürzere Eingaben: fail-open
_MAX_AR_ORDER = 32
_MIN_AR_ORDER = 8


def declick_fine_crackle(
    audio: np.ndarray,
    sample_rate: int,
    strength: float = 1.0,
) -> np.ndarray:
    """Entfernt kontinuierliches feines Knistern (Mikro-Impuls-Textur).

    Args:
        audio: mono (N,) oder stereo — (2, N) channels-first ODER (N, 2)
               channels-last; beide Layouts werden normalisiert
               (§Stereo-Layout-Invariante).
        sample_rate: Abtastrate in Hz (≥ 8000).
        strength: 0..1 — skaliert die Reparatur sekundärer (kontext-
                  bestätigter) Kandidaten; primäre Impulse werden immer
                  voll restauriert.

    Returns:
        Audio mit restaurierten Knistern-Impulsen; gleiche Form und gleiches
        Layout wie die Eingabe. Bit-identisch deterministisch
        (§G5 (copilot-instructions.md)).
    """
    _src = np.asarray(audio, dtype=np.float32)
    _sr = int(sample_rate)
    _strength = float(np.clip(strength, 0.0, 1.0))

    # ── Layout-Normalisierung (§Stereo-Layout-Invariante) ──
    if _src.ndim == 2:
        _channels_first = _src.shape[0] == 2 and _src.shape[1] > 2
        _work = _src if _channels_first else _src.T
        _channels = [_work[i] for i in range(_work.shape[0])]
        _mid = np.nan_to_num(_work[0], nan=0.0, posinf=0.0, neginf=0.0).copy()
        if _work.shape[0] > 1:
            for _i in range(1, _work.shape[0]):
                _mid += np.nan_to_num(_work[_i], nan=0.0, posinf=0.0, neginf=0.0)
            _mid /= float(_work.shape[0])
    else:
        _channels_first = False
        _channels = None
        _mid = np.nan_to_num(_src, nan=0.0, posinf=0.0, neginf=0.0).copy()

    _n = _mid.shape[0]
    _win = int(round(_MEDIAN_WIN_S * _sr)) | 1
    if _n < max(int(_MIN_LEN_S * _sr), _win * 4) or _sr < 8000:
        logger.warning(
            "§SR-CK3 Fein-Declicker übersprungen (Eingang zu kurz: %d Samples @ %d Hz)",
            _n,
            _sr,
        )
        return _src.copy()  # type: ignore[no-any-return]

    # ── 1. Betonung ──
    _hp = float(min(_HP_FREQ_HZ, 0.4 * _sr))
    _sos = _signal.butter(4, _hp, btype="highpass", fs=_sr, output="sos")
    _x_hp = _signal.sosfiltfilt(_sos, _mid).astype(np.float32)
    _env = np.abs(_x_hp).astype(np.float64)

    # ── 2. Robuste Lokal-Statistik ──
    _med = _median_filter(_env, size=_win, mode="nearest").astype(np.float64)
    _scale = _median_filter(np.abs(_env - _med), size=_win, mode="nearest").astype(np.float64) * _MAD_K
    _scale = np.maximum(_scale, 1e-9)
    _thr_p = _med + _PRIMARY_K * _scale
    _thr_s = _med + _SECONDARY_K * _scale
    _thr_r = _med + _REFINE_K * _scale

    _mask_p = _env > _thr_p
    _mask_s = _env > _thr_s
    if not _mask_p.any() and _strength <= 0.0:
        logger.info("§SR-CK3 Fein-Declicker: keine Primär-Impulse, strength=0 — Passthrough")
        return _src.copy()  # type: ignore[no-any-return]

    # ── Hilfsgrößen für die Vetos ──
    _sus = max(1, int(_SUSTAIN_S * _sr))
    _env_s = np.convolve(_env, np.ones(_sus, dtype=np.float64) / _sus, mode="same")
    _rms_win = max(1, int(_ONSET_WINDOW_S * _sr))
    _rms = np.sqrt(
        np.maximum(np.convolve(_mid.astype(np.float64) ** 2, np.ones(_rms_win) / _rms_win, mode="same"), 0.0)
    )

    _max_span = max(2, int(_MAX_SPAN_S * _sr))
    _ctx = int(_CONTEXT_S * _sr)
    _refine_half = max(1, int(_REFINE_HALF_S * _sr))
    _edge = int(0.05 * _sr)  # Filter-Einschwingzone des Hochpasses ausschließen

    def _passes_vetos(_a: int, _b: int) -> bool:
        """Musik-Onset-Veto: nur echte Mikro-Impulse überleben (SOTA-Kern)."""
        # (b) Sustain: keine nachfolgende HF-Energie — Musik klingt nach.
        #     Skalenfrei (Post/Pre-Verhältnis) + absoluter Anteil am eigenen
        #     Peak: baseline-unabhängig, auch in HF-armen Passagen gültig.
        _guard = max(1, int(0.001 * _sr))
        _post = min(_n, _b + _guard + _sus)
        _post_mean = float(np.mean(_env_s[_b + _guard : _post])) if _post > _b + _guard else 0.0
        _pre_a0, _pre_b0 = max(0, _a - 2 * _sus), max(0, _a - _guard)
        _pre_mean = float(np.mean(_env_s[_pre_a0:_pre_b0])) if _pre_b0 > _pre_a0 else 0.0
        _peak = float(_env[_a:_b].max()) if _b > _a else 0.0
        if _post_mean > max(2.5 * _pre_mean + 1e-9, 0.25 * _peak):
            return False
        # (c) Vollband-Onset: kein Energie-Sprung im Originalband (Drum-/Sibilant-Veto).
        #     Lange Vorher-Fenster (30 ms) machen das Veto phasenrobust: bei
        #     periodischer Musik (Sinus an beliebiger Phase) ist Vorher ≈ Nachher;
        #     nur ein echter Musik-Onset (Ruhe → Schlag) springt.
        _pa, _pb = max(0, _a - int(0.030 * _sr)), max(0, _a - _rms_win)
        _qa, _qb = min(_n, _b + _rms_win), min(_n, _b + int(0.010 * _sr))
        _pre = float(np.mean(_rms[_pa:_pb])) if _pb > _pa else 0.0
        _post2 = float(np.mean(_rms[_qa:_qb])) if _qb > _qa else 0.0
        if _post2 > _ONSET_ABS and _post2 > _ONSET_RATIO * _pre:
            return False
        return True

    def _runs(_mask: np.ndarray) -> list[tuple[int, int]]:
        if not _mask.any():
            return []
        _d = np.diff(np.concatenate(([0], _mask.view(np.int8), [0])))
        _starts = np.where(_d == 1)[0]
        _ends = np.where(_d == -1)[0]
        return list(zip(_starts.tolist(), _ends.tolist()))

    _primary: list[tuple[int, int]] = []
    for _a, _b in _runs(_mask_p):
        # (a) Effektive Impuls-Dauer statt Roh-Länge: Musik-Transienten halten
        #     > 15 % ihres Peaks lange; Knistern-Wavelets klingen sofort ab.
        #     (In HF-leisen Passagen liegt die Schwelle am numerischen Floor
        #     und die Wavelet klingelt > 2 ms — Roh-Länge würde sie verwerfen.)
        _peak0 = float(_env[_a:_b].max())
        _eff = int(np.sum(_env[_a:_b] > np.maximum(_thr_p[_a:_b], 0.15 * _peak0)))
        if _eff > _max_span:
            continue
        if _a < _edge or _b > _n - _edge:  # Einschwingzone
            continue
        if not _passes_vetos(_a, _b):
            continue
        _primary.append((_a, _b))

    _secondary: list[tuple[int, int]] = []
    if _primary:
        _prim_times = np.asarray([(_a + _b) / 2.0 for _a, _b in _primary])
        _dense = len(_primary) / max(_n / _sr, 1e-9) > _DENSE_EVENTS_PER_S
        for _a, _b in _runs(_mask_s & ~_mask_p):
            _peak0 = float(_env[_a:_b].max())
            if int(np.sum(_env[_a:_b] > np.maximum(_thr_s[_a:_b], 0.15 * _peak0))) > _max_span:
                continue
            if _a < _edge or _b > _n - _edge:
                continue
            _c = (_a + _b) / 2.0
            if not _dense and not np.any(np.abs(_prim_times - _c) <= _ctx):
                continue
            if not _passes_vetos(_a, _b):
                continue
            _secondary.append((_a, _b))

    if not _primary and not _secondary:
        logger.info("§SR-CK3 Fein-Declicker: keine Knistern-Impulse bestätigt — Passthrough")
        return _src.copy()  # type: ignore[no-any-return]

    # ── 6. Span-Verfeinerung + Merge ──
    def _refine(_a: int, _b: int) -> tuple[int, int]:
        # Impuls-relativ: nur bis zum Abklingen der eigenen Wavelet (20 % des
        # Run-Peaks) — ein reiner HF-Floor würde in Musik-HF hineinlaufen.
        _peak = float(_env[_a:_b].max()) if _b > _a else 0.0
        _thr_span = np.maximum(_thr_r, 0.20 * _peak)
        _lo, _hi = _a, _b
        while _lo > 0 and (_a - _lo) < _refine_half and _env[_lo - 1] > _thr_span[_lo - 1]:
            _lo -= 1
        while _hi < _n - 1 and (_hi - _b) < _refine_half and _env[_hi] > _thr_span[_hi]:
            _hi += 1
        if _hi - _lo > _max_span:
            return _a, _b
        return _lo, _hi

    def _finalize(_a: int, _b: int) -> tuple[int, int] | None:
        """Verfeinern + ggf. auf den Ring-Kern schrumpfen (>5 % des Run-Peaks)."""
        _lo, _hi = _refine(_a, _b)
        if _hi - _lo > _max_span:
            _peak = float(_env[_a:_b].max())
            _thr_c = np.maximum(_thr_r, 0.05 * _peak)
            _core = np.where(_env[_lo:_hi] > _thr_c[_lo:_hi])[0]
            if _core.size == 0:
                return None
            _old_lo = _lo
            _lo = _old_lo + int(_core[0])
            _hi = _old_lo + int(_core[-1]) + 1
            if _hi - _lo > _max_span:
                return None  # auch der Kern ist zu lang — Musik-Transient
        return _lo, _hi

    _tagged = []
    for _a, _b in _primary:
        _f = _finalize(_a, _b)
        if _f is not None:
            _tagged.append((*_f, False))
    for _a, _b in _secondary:
        _f = _finalize(_a, _b)
        if _f is not None:
            _tagged.append((*_f, True))
    _tagged.sort(key=lambda _t: (_t[0], _t[1]))
    # Merge nur bei ÜBERLAPPUNG und nur bis zur Impuls-Kompaktheitsgrenze:
    # Kettenschaltungen benachbarter Cluster-Spans würden 100–200-Sample-Lücken
    # erzeugen, über die AR extrapolieren müsste (Phasendrift, Produktionsbefund).
    _merged: list[tuple[int, int, bool]] = []
    for _a, _b, _sec in _tagged:
        if _merged and _a <= _merged[-1][1] and (_b - _merged[-1][0]) <= _max_span:
            _pa, _pb, _psec = _merged[-1]
            _merged[-1] = (_pa, max(_pb, _b), _psec and _sec)
        else:
            _merged.append((_a, _b, _sec))

    # ── 7. Restaurierung (Godsill & Rayner 1998) ──
    _span_mask = np.zeros(_n, dtype=bool)
    for _a, _b, _ in _merged:
        _span_mask[_a:_b] = True

    def _restore(_x: np.ndarray) -> np.ndarray:
        _out = _x.astype(np.float64).copy()
        for _a, _b, _sec in _merged:
            _L = _b - _a
            if _L <= 0:
                continue
            _seg: np.ndarray | None
            if _L <= 4:
                _seg = _interp_catmull_rom(_x, _a, _b, _n, _span_mask)
            else:
                _seg = _interp_ar_ls(_x, _a, _b, _n, _span_mask)
            if _seg is None:
                continue  # konservativ: nicht rekonstruierbar — Original behalten
            if not _splice_ok(_seg, _x, _a, _b, _n, _span_mask):
                continue  # Diskontinuität ⇒ unsichere Interpolation — Original behalten
            if _sec:
                _seg = (1.0 - _strength) * _x[_a:_b].astype(np.float64) + _strength * _seg
            _out[_a:_b] = _seg
        return _out  # type: ignore[no-any-return]

    if _channels is None:
        _result = _restore(_mid)
    else:
        _result = np.stack([_restore(_ch) for _ch in _channels], axis=0)

    _result = np.clip(_result, -1.0, 1.0).astype(np.float32)
    if _channels is not None and not _channels_first:
        _result = _result.T
    _total_ms = float(sum(_b - _a for _a, _b, _ in _merged)) / _sr * 1000.0
    logger.info(
        "§SR-CK3 Fein-Declicker: %d primär + %d sekundär → %d Spans (%.2f ms) restauriert",
        len(_primary),
        len(_secondary),
        len(_merged),
        _total_ms,
    )
    return _result  # type: ignore[no-any-return]


def _interp_catmull_rom(x: np.ndarray, a: int, b: int, n: int, span_mask: np.ndarray | None = None) -> np.ndarray:
    """Catmull-Rom durch vier SAUBERE Anker (wandert an Spans vorbei, deterministisch)."""

    def _anchor(_i0: int, _step: int) -> float:
        _i = _i0
        while 0 <= _i < n:
            if span_mask is None or not span_mask[_i]:
                return float(x[_i])
            _i += _step
        return float(x[max(0, min(n - 1, _i - _step))])

    _p1 = _anchor(a - 1, -1)
    _p0 = _anchor(a - 2, -1)
    _p2 = _anchor(b, +1)
    _p3 = _anchor(b + 1, +1)
    _L = b - a
    _t = np.linspace(0.0, 1.0, _L + 2, endpoint=True)[1:-1]
    _t2 = _t * _t
    _t3 = _t2 * _t
    return 0.5 * (  # type: ignore[no-any-return]
        2.0 * _p1
        + (-_p0 + _p2) * _t
        + (2.0 * _p0 - 5.0 * _p1 + 4.0 * _p2 - _p3) * _t2
        + (-_p0 + 3.0 * _p1 - 3.0 * _p2 + _p3) * _t3
    )


def _splice_ok(seg: np.ndarray, x: np.ndarray, a: int, b: int, n: int, span_mask: np.ndarray, k: float = 3.0) -> bool:
    """Splice-Check: Interpolation muss stetig an die sauberen Nachbarn anschließen.

    Eine korrekte AR-/CR-Interpolation ist an den Rändern nahezu stetig (die
    LS-Anker erzwingen das); Fehl-Interpolationen springen. Referenz ist die
    lokale Sample-zu-Sample-Skala der sauberen Nachbarschaft (p90).
    """
    _l_lo = max(0, a - 200)
    _left = x[_l_lo:a][~span_mask[_l_lo:a]]
    _r_hi = min(n, b + 200)
    _right = x[b:_r_hi][~span_mask[b:_r_hi]]
    _ctx = np.concatenate([_left, _right])
    if _ctx.size < 8:
        return True  # kein Referenz-Kontext — die AR-Gates haben bereits entschieden
    _d = np.abs(np.diff(_ctx))
    _thr = k * float(np.percentile(_d, 90)) + 1e-9
    if a >= 1 and abs(float(seg[0]) - float(x[a - 1])) > _thr:
        return False
    if b < n and abs(float(seg[-1]) - float(x[b])) > _thr:
        return False
    return True


def _interp_ar_ls(x: np.ndarray, a: int, b: int, n: int, span_mask: np.ndarray) -> np.ndarray | None:
    """AR(p)-Least-Squares-Interpolation nach Godsill & Rayner (1998).

    Vorwärts-Prädiktor af (x_t = Σ af_k · x_{t-k}) aus dem sauberen Fenster
    links, Rückwärts-Prädiktor ab (x_t = Σ ab_k · x_{t+k}) aus dem sauberen
    Fenster rechts; dann Least-Squares über den Span mit beiden Prädiktoren
    als Anker (stetige Anschlüsse an die sauberen Nachbarn).
    Ridge-stabilisiert; fällt deterministisch auf Catmull-Rom zurück.
    """
    _L = b - a
    _p_target = int(np.clip(_L + 4, _MIN_AR_ORDER, _MAX_AR_ORDER))
    # Saubere Schätzfenster: andere Spans ausschließen (dichtes Knistern würde
    # die AR-Koeffizienten sonst mit Nachbar-Impulsen kontaminieren). AR
    # braucht ZUSAMMENHÄNGENDE Samples (Lag-Struktur) — deshalb die
    # nächstgelegene zusammenhängende saubere Strecke je Seite.
    _want = 4 * _p_target

    def _contig(_i0: int, _step: int) -> np.ndarray:
        _vals = np.empty(_want, dtype=np.float64)
        _cnt = 0
        _i = _i0
        while 0 <= _i < n and _cnt < _want:
            if span_mask[_i]:
                break
            _vals[_cnt] = x[_i]
            _cnt += 1
            _i += _step
        return _vals[:_cnt]  # type: ignore[no-any-return]

    def _solve(_af: np.ndarray, _ab: np.ndarray, _p: int) -> np.ndarray | None:
        _M = np.zeros((2 * _L, _L), dtype=np.float64)
        _c = np.zeros(2 * _L, dtype=np.float64)
        for _i in range(_L):
            _M[_i, _i] += 1.0
            for _k in range(1, _p + 1):
                if _i - _k >= 0:
                    _M[_i, _i - _k] -= _af[_k - 1]
                else:
                    _c[_i] += _af[_k - 1] * float(x[a + _i - _k])
            _M[_L + _i, _i] += 1.0
            for _k in range(1, _p + 1):
                if _i + _k < _L:
                    _M[_L + _i, _i + _k] -= _ab[_k - 1]
                else:
                    _c[_L + _i] += _ab[_k - 1] * float(x[a + _i + _k])
        _G = _M.T @ _M
        _lam = 1e-6 * (np.trace(_G) / max(_L, 1) + 1e-12)
        try:
            _u = np.linalg.solve(_G + _lam * np.eye(_L), _M.T @ _c)
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(_u)):
            return None
        return _u  # type: ignore[no-any-return]

    _left_c = _contig(a - 1, -1)[::-1]  # älteste → neueste
    _right_c = _contig(b, +1)
    _w = min(_left_c.size, _right_c.size)
    if _w >= 8:
        # Saubere Fenster: adaptive Ordnung — kurze Lücken in dichtem Knistern
        # bekommen AR niedriger Ordnung statt CR-Fallback.
        _p = int(np.clip(min(_p_target, _w // 2), 2, _MAX_AR_ORDER))
        if 4 * _p >= _L:
            _af = _estimate_ar(_left_c[-2 * _p :], _p)
            _ab = _estimate_ar(_right_c[: 2 * _p][::-1], _p)
            if _af is not None and _ab is not None:
                _u = _solve(_af, _ab, _p)
                if _u is not None:
                    return _u
    # Zweite Stufe: Roh-Fenster (zusammenhängend) — nur wenn sie überwiegend
    # sauber sind (dichte Cluster würden die AR-Koeffizienten korrumpieren).
    _P = max(4 * _p_target, 2 * _L)
    _l0 = max(0, a - _P)
    _r1 = min(n, b + _P)
    if int(span_mask[_l0:a].sum()) > 0.2 * (a - _l0) or int(span_mask[b:_r1].sum()) > 0.2 * (_r1 - b):
        if _L <= 24:
            return _interp_catmull_rom(x, a, b, n, span_mask)
        return None
    _left = x[_l0:a]
    _right = x[b:_r1]
    if _left.size < 2 * _p_target or _right.size < 2 * _p_target:
        # Nicht rekonstruierbar — kurze Spans über Catmull-Rom, lange Spans
        # konservativ ÜBERSPRINGEN (nie verschlechtern statt phaseninvertierter
        # Kurven, §G7 (copilot-instructions.md)).
        if _L <= 24:
            return _interp_catmull_rom(x, a, b, n, span_mask)
        return None  # type: ignore[return-value]
    _af = _estimate_ar(_left.astype(np.float64), _p_target)
    _ab = _estimate_ar(_right[::-1].astype(np.float64), _p_target)
    if _af is None or _ab is None:
        if _L <= 24:
            return _interp_catmull_rom(x, a, b, n, span_mask)
        return None  # type: ignore[return-value]
    _u = _solve(_af, _ab, _p_target)
    if _u is None:
        if _L <= 24:
            return _interp_catmull_rom(x, a, b, n, span_mask)
        return None  # type: ignore[return-value]
    return _u  # type: ignore[no-any-return]


def _estimate_ar(seq: np.ndarray, p: int) -> np.ndarray | None:
    """AR(p)-Koeffizienten per Least-Squares (Yule-Walker-ähnlich, ridge-stabilisiert).

    Returns:
        Koeffizientenvektor (p,) oder None, wenn das System singulär ist
        (Aufrufer fällt dann deterministisch auf Catmull-Rom zurück).
    """
    _m = seq.shape[0]
    _X = np.empty((_m - p, p), dtype=np.float64)
    for _k in range(p):
        _X[:, _k] = seq[p - 1 - _k : _m - 1 - _k]
    _y = seq[p:]
    _A = _X.T @ _X
    _lam = 1e-6 * (np.trace(_A) / max(p, 1) + 1e-12)
    try:
        return np.linalg.solve(_A + _lam * np.eye(p), _X.T @ _y)  # type: ignore[no-any-return]
    except np.linalg.LinAlgError:
        return None
