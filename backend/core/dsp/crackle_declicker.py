"""§SR-CK4 (Hör-Urteil 2026-09-24): SOTA-Zeitbereich-Fein-Declicker v4 —
subtraktive Impuls-Modellierung mit parametrischer Matched-Filter-Bank.

Hörbefund v1: Interpolations-Reparatur ersetzte in den behandelten Regionen
~50 % der Musik-HF-Textur (HP-RMS 0,0099 → 0,0049) → hörbare Lücken; das
Original klang besser. Konsequenz: Reparatur nie durch Synthese von Musik,
sondern durch SUBTRAKTION der modellierten Klick-Komponente — der
Musik-Untergrund bleibt stehen, nichts wird erfunden.

Architektur v4 (deterministisch, kein ML, kein RNG):
1. Detektion: HP-Betonung (2,2 kHz), robuste Lokal-Statistik (laufender
   Median + MAD, adaptive Schwelle), Musik-Onset-Vetos (Sustain-Test,
   Vollband-Onset-Test), effektive Impuls-Dauer — ohne Dense-Regime-
   Freischaltung (die griff Musik-HF).
2. Parametrisches Klick-Modell (Vaseghi-Klasse): Doppel-Exponential-Klick
   c(t) = A·(e^(−t/τ1) − e^(−t/τ2))·u(t) mit 18 (τ1, τ2)-Gittern — deckt
   Staub-Klicks (kurz) bis Rillen-Schäden (lang) ab. Je Template wird die
   HP-Antwort des Beobachtungsfilters einmal analytisch erzeugt und
   peak-ausgerichtet gespeichert (keine gemittelten Prototypen — die
   deckten die Klick-Familie nicht ab, Produktionsbefund).
3. Ereignis-Klassifikation: jeder Kandidat (primär ≥ 4,5σ, sekundär ≥ 3σ)
   wird gegen die Matched-Filter-Bank gefittet (LS-Amplitude mit Vorzeichen,
   bestes Template) — Musik-HF-Ausreißer passen auf kein Template und werden
   verworfen; feine Knistern-Textur wird erreichbar.
4. Cluster-Reparatur IM HP-BAND: Ereignisse ≤ 20 ms bilden Cluster; je
   Cluster ein gemeinsamer Least-Squares-Fit (überlappende Templates,
   Amplituden-Deckel je Ereignis) → Subtraktion der gefitteten Klick-
   Komponente im HP-Band. Das LP-Band (< 2,2 kHz, der Musik-Körper) bleibt
   bis auf float32-Rundung unangetastet. Der LF-Anteil des Klicks ist durch
   den Beobachtungs-Hochpass prinzipiell unbeobachtbar — eine Inversion
   würde dort Rauschen um Größenordnungen verstärken (Produktionsbefund:
   Korrelation 0,64) und ist deshalb ausgeschlossen; der verbleibende
   LF-Schmier-Rest (~<1 % der Klick-Energie) ist weich und maskiert.
5. Validierung: Fit-Residuum > 50 % ⇒ Cluster überspringen (letzte
   Sicherheitsstufe, geloggt).

Deterministisch (§G5 (copilot-instructions.md)), NaN-sicher, fail-open
(§V6 (copilot-instructions.md)), Stereo-Layout-Invariante (Detektion auf
Mid, Subtraktion pro Kanal mit gemeinsamer Maske).
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
_SECONDARY_K = 3.0  # Sekundär-Schwelle (σ, nur mit Template-Bestätigung)
_MAX_SPAN_S = 0.002  # effektive Impuls-Dauer (Musik-Transienten sind länger)
_SUSTAIN_S = 0.008  # Sustain-Test-Fenster (Nachklang-Veto)
_ONSET_RATIO = 2.0  # Vollband-Sprung > 2× ⇒ Musik-Onset
_ONSET_ABS = 0.01  # absoluter Vollband-Floor des Onset-Vetos
_ONSET_WINDOW_S = 0.001  # Vollband-Fenster (1 ms) für den Onset-Vergleich
_MIN_LEN_S = 0.1  # kürzere Eingaben: fail-open
_TEMPLATE_WIN = 36  # ±Samples des Klick-Templates
_CLUSTER_GAP_S = 0.020  # Ereignis-Abstand für Cluster-Bildung
_FIT_ACCEPT_PRIMARY = 0.5  # max. Template-Residuum (primär)
_FIT_ACCEPT_SECONDARY = 0.4  # max. Template-Residuum (sekundär)
_SUB_FACTOR = 0.85  # partielle Subtraktion (Sicherheitsmarge)
_AMP_CAP = 1.0  # max. |Amplitude| relativ zum Ereignis-HP-Peak (kein Überziehen)
_RESID_ACCEPT = 0.5  # max. Cluster-Fit-Residuum (sonst überspringen)
_TAU1_GRID = (1.0, 2.0, 4.0, 8.0)  # Anstiegs-Zeitkonstanten des Klick-Modells
_TAU2_GRID = (2.0, 4.0, 8.0, 16.0, 32.0, 64.0)  # Abkling-Zeitkonstanten


def _make_templates(sos: np.ndarray, sr: int) -> list[np.ndarray]:
    """Matched-Filter-Bank: HP-Antworten des Doppel-Exponential-Klick-Modells.

    c(t) = (e^(−t/τ1) − e^(−t/τ2))·u(t) für alle (τ1, τ2) mit τ2 > τ1,
    je Template peak-ausgerichtet und auf Peak 1 normalisiert (Länge
    2·_TEMPLATE_WIN+1). Deterministisch — wird einmal pro Filter erzeugt.
    """
    _L = 2 * _TEMPLATE_WIN + 1
    _pad = 64
    _N = _L + 2 * _pad
    _t = np.arange(_N, dtype=np.float64)  # in SAMPLES (τ-Grid ist in Samples kalibriert)
    _templates: list[np.ndarray] = []
    for _t1 in _TAU1_GRID:
        for _t2 in _TAU2_GRID:
            if _t2 <= _t1:
                continue
            _click = np.zeros(_N, dtype=np.float64)
            _click[2 * _pad :] = np.exp(-_t[: _N - 2 * _pad] / _t2) - np.exp(-_t[: _N - 2 * _pad] / _t1)
            _tpl = _finalize_template(_click, sos, _L, _pad)
            if _tpl is not None:
                _templates.append(_tpl)
    # Rechteck-Klicks (scharfe Impulse): Stufenantwort-Differenzen — die
    # Familie der Doppel-Exponentiale enthält keine scharfen Rechteck-Impulse
    # (Produktionsbefund: Residuum 0,71 für 1-Sample-Bumps). Feines Breiten-
    # Grid, damit jede Klick-Breite ein exaktes Template hat.
    for _w_rect in range(1, 13):
        _click = np.zeros(_N, dtype=np.float64)
        _click[2 * _pad : 2 * _pad + _w_rect] = 1.0
        _tpl = _finalize_template(_click, sos, _L, _pad)
        if _tpl is not None:
            _templates.append(_tpl)
    if not _templates:
        raise RuntimeError("§SR-CK4: Template-Erzeugung fehlgeschlagen")
    return _templates


def _finalize_template(_click: np.ndarray, _sos: np.ndarray, _L: int, _pad: int) -> np.ndarray | None:
    """HP-Antwort eines Klick-Modells → peak-ausgerichtet, auf Peak 1 normiert."""
    _hp = _signal.sosfiltfilt(_sos, _click).astype(np.float64)
    _hp = _hp[2 * _pad - _TEMPLATE_WIN : 2 * _pad + _TEMPLATE_WIN + 1]
    _pk = int(np.argmax(np.abs(_hp)))
    _shift = _pk - _TEMPLATE_WIN  # Sample _pk soll auf Position _TEMPLATE_WIN
    _aligned = np.zeros(_L, dtype=np.float64)
    if _shift >= 0:
        _aligned[: _L - _shift] = _hp[_shift:]
    else:
        _aligned[-_shift:] = _hp[: _L + _shift]
    _amp = float(np.abs(_aligned).max())
    if _amp < 1e-12:
        return None
    return (_aligned / _amp).astype(np.float64)  # type: ignore[no-any-return]


def declick_fine_crackle(
    audio: np.ndarray,
    sample_rate: int,
    strength: float = 1.0,
) -> np.ndarray:
    """Entfernt kontinuierliches feines Knistern durch subtraktive Impuls-Modellierung.

    Args:
        audio: mono (N,) oder stereo — (2, N) channels-first ODER (N, 2)
               channels-last; beide Layouts werden normalisiert
               (§Stereo-Layout-Invariante).
        sample_rate: Abtastrate in Hz (≥ 8000).
        strength: 0..1 — skaliert den Subtraktions-Faktor sekundärer
                  (template-bestätigter) Ereignisse; primäre Ereignisse
                  werden immer mit vollem Faktor behandelt.

    Returns:
        Audio mit subtrahierter Knistern-Komponente; gleiche Form und
        gleiches Layout wie die Eingabe. Bit-identisch deterministisch
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
            "§SR-CK4 Fein-Declicker übersprungen (Eingang zu kurz: %d Samples @ %d Hz)",
            _n,
            _sr,
        )
        return _src.copy()  # type: ignore[no-any-return]

    # ── 1. Betonung + robuste Lokal-Statistik ──
    _hp_freq = float(min(_HP_FREQ_HZ, 0.4 * _sr))
    _sos = _signal.butter(4, _hp_freq, btype="highpass", fs=_sr, output="sos")
    _templates = _make_templates(_sos, _sr)
    _W = _TEMPLATE_WIN
    _x_hp = _signal.sosfiltfilt(_sos, _mid).astype(np.float32)
    _env = np.abs(_x_hp).astype(np.float64)

    _med = _median_filter(_env, size=_win, mode="nearest").astype(np.float64)
    _scale = _median_filter(np.abs(_env - _med), size=_win, mode="nearest").astype(np.float64) * _MAD_K
    _scale = np.maximum(_scale, 1e-9)
    _thr_p = _med + _PRIMARY_K * _scale
    _thr_s = _med + _SECONDARY_K * _scale

    _max_span = max(2, int(_MAX_SPAN_S * _sr))
    _sus = max(1, int(_SUSTAIN_S * _sr))
    _env_s = np.convolve(_env, np.ones(_sus, dtype=np.float64) / _sus, mode="same")
    _rms_win = max(1, int(_ONSET_WINDOW_S * _sr))
    _rms = np.sqrt(
        np.maximum(np.convolve(_mid.astype(np.float64) ** 2, np.ones(_rms_win) / _rms_win, mode="same"), 0.0)
    )
    _edge = int(0.05 * _sr)  # Filter-Einschwingzone des Hochpasses ausschließen

    def _passes_vetos(_a: int, _b: int) -> bool:
        """Musik-Onset-Veto: nur echte Mikro-Impulse überleben."""
        # (b) Sustain: keine nachfolgende HF-Energie — Musik klingt nach.
        _guard = max(1, int(0.001 * _sr))
        _post = min(_n, _b + _guard + _sus)
        _post_mean = float(np.mean(_env_s[_b + _guard : _post])) if _post > _b + _guard else 0.0
        _pre_a0, _pre_b0 = max(0, _a - 2 * _sus), max(0, _a - _guard)
        _pre_mean = float(np.mean(_env_s[_pre_a0:_pre_b0])) if _pre_b0 > _pre_a0 else 0.0
        _peak = float(_env[_a:_b].max()) if _b > _a else 0.0
        if _post_mean > max(2.5 * _pre_mean + 1e-9, 0.25 * _peak):
            return False
        # (c) Vollband-Onset: kein Energie-Sprung im Originalband.
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

    def _effective_len(_a: int, _b: int, _thr: np.ndarray) -> int:
        _peak0 = float(_env[_a:_b].max())
        return int(np.sum(_env[_a:_b] > np.maximum(_thr[_a:_b], 0.15 * _peak0)))

    # ── Detektion: primäre (Saat) und sekundäre Kandidaten ──
    _mask_p = _env > _thr_p
    _mask_s = _env > _thr_s
    if not _mask_p.any() and _strength <= 0.0:
        logger.info("§SR-CK4 Fein-Declicker: keine Primär-Impulse, strength=0 — Passthrough")
        return _src.copy()  # type: ignore[no-any-return]

    _primary: list[tuple[int, int]] = []
    for _a, _b in _runs(_mask_p):
        if _effective_len(_a, _b, _thr_p) > _max_span:
            continue
        if _a < _edge or _b > _n - _edge:
            continue
        if not _passes_vetos(_a, _b):
            continue
        _primary.append((_a, _b))
    _secondary: list[tuple[int, int]] = []
    for _a, _b in _runs(_mask_s & ~_mask_p):
        if _effective_len(_a, _b, _thr_s) > _max_span:
            continue
        if _a < _edge or _b > _n - _edge:
            continue
        if not _passes_vetos(_a, _b):
            continue
        _secondary.append((_a, _b))

    if not _primary:
        logger.info("§SR-CK4 Fein-Declicker: keine Knistern-Impulse bestätigt — Passthrough")
        return _src.copy()  # type: ignore[no-any-return]

    # ── 3. Ereignis-Klassifikation über die Matched-Filter-Bank ──
    def _classify(_a: int, _b: int) -> tuple[float, int, float, float] | None:
        """(Residuum, Template-Index, Amplitude mit Vorzeichen, HP-Peak)."""
        _pk = _a + int(np.argmax(np.abs(_x_hp[_a:_b])))
        _lo = max(0, _pk - _W)
        _hi = min(_n, _pk + _W + 1)
        if _hi - _lo < 8:
            return None
        _seg = _x_hp[_lo:_hi].astype(np.float64)
        _amp0 = float(np.abs(_seg).max())
        if _amp0 < 1e-6:
            return None
        _best: tuple[float, int, float, float] | None = None
        for _k, _tpl in enumerate(_templates):
            _t = _tpl[_W - (_pk - _lo) : _W - (_pk - _lo) + (_hi - _lo)]
            if _t.shape[0] != _seg.shape[0]:
                continue
            _den = float(np.dot(_t, _t))
            if _den < 1e-12:
                continue
            _a_fit = float(np.dot(_seg, _t) / _den)  # vorzeichenbehaftet
            _res = float(np.sum((_seg - _a_fit * _t) ** 2) / max(np.sum(_seg**2), 1e-12))
            if _best is None or _res < _best[0]:
                _best = (_res, _k, _a_fit, _amp0)
        return _best

    _events: list[tuple[int, float, int, float, float, bool]] = []  # (peak, res, tpl, amp, hp_amp, primary)
    for _a, _b in _primary:
        _c = _classify(_a, _b)
        if _c is not None and _c[0] <= _FIT_ACCEPT_PRIMARY:
            _events.append((_a + int(np.argmax(np.abs(_x_hp[_a:_b]))), _c[0], _c[1], _c[2], _c[3], True))
    for _a, _b in _secondary:
        _c = _classify(_a, _b)
        if _c is not None and _c[0] <= _FIT_ACCEPT_SECONDARY:
            _events.append((_a + int(np.argmax(np.abs(_x_hp[_a:_b]))), _c[0], _c[1], _c[2], _c[3], False))

    if not _events:
        logger.info("§SR-CK4 Fein-Declicker: keine Ereignisse template-bestätigt — Passthrough")
        return _src.copy()  # type: ignore[no-any-return]

    # ── 4. Cluster-Bildung ──
    _events.sort(key=lambda _e: _e[0])
    _gap = int(_CLUSTER_GAP_S * _sr)
    _clusters: list[list[tuple[int, float, int, float, float, bool]]] = []
    for _e in _events:
        if _clusters and _e[0] - _clusters[-1][-1][0] <= _gap:
            _clusters[-1].append(_e)
        else:
            _clusters.append([_e])

    # ── 5. Subtraktive Cluster-Reparatur im HP-Band ──
    def _restore(_x: np.ndarray) -> np.ndarray:
        _hp_x = _signal.sosfiltfilt(_sos, _x).astype(np.float64)
        _lp_x = _x.astype(np.float64) - _hp_x  # Musik-Körper bleibt (bis auf float32-Rundung) unangetastet
        for _cl in _clusters:
            _c0 = min(_e[0] for _e in _cl) - _W - 2
            _c1 = max(_e[0] for _e in _cl) + _W + 3
            _lo = max(0, _c0)
            _hi = min(_n, _c1)
            _m = _hi - _lo
            if _m < 16:
                continue
            _A = np.zeros((_m, len(_cl)), dtype=np.float64)
            for _j, _e in enumerate(_cl):
                _p = _templates[_e[2]]
                for _k in range(len(_p)):
                    _idx = _e[0] - _W + _k - _lo
                    if 0 <= _idx < _m:
                        _A[_idx, _j] = _p[_k]
            _y = _x_hp[_lo:_hi].astype(np.float64)
            _coef, *_ = np.linalg.lstsq(_A, _y, rcond=None)
            _amps = np.array(
                [float(np.clip(_coef[_j], -_AMP_CAP * _cl[_j][4], _AMP_CAP * _cl[_j][4])) for _j in range(len(_cl))]
            )
            _pred = _A @ _amps
            _rel = float(np.sum((_y - _pred) ** 2) / max(np.sum(_y**2), 1e-12))
            if _rel > _RESID_ACCEPT:
                continue  # Fit unzureichend — letzte Sicherheitsstufe
            for _j, _e in enumerate(_cl):
                _factor = _SUB_FACTOR * (1.0 if _e[5] else _strength)
                if _factor <= 0.0:
                    continue
                _p = _templates[_e[2]]
                for _k in range(len(_p)):
                    _idx = _e[0] - _W + _k
                    if _lo <= _idx < _hi:
                        _hp_x[_idx] -= _factor * _amps[_j] * _p[_k]
        return np.clip(_lp_x + _hp_x, -1.0, 1.0)  # type: ignore[no-any-return]

    if _channels is None:
        _result = _restore(_mid)
    else:
        _result = np.stack([_restore(_ch) for _ch in _channels], axis=0)

    _result = np.clip(_result, -1.0, 1.0).astype(np.float32)
    if _channels is not None and not _channels_first:
        _result = _result.T
    _n_prim = sum(1 for _e in _events if _e[5])
    _n_sec = len(_events) - _n_prim
    logger.info(
        "§SR-CK4 Fein-Declicker: %d primär + %d sekundär template-bestätigt → %d Cluster subtrahiert (%d Templates)",
        _n_prim,
        _n_sec,
        len(_clusters),
        len(_templates),
    )
    return _result  # type: ignore[no-any-return]
