"""Vertragstests für das stereo_imbalance-Rezept des Sinn-Beweises (2026-11-18).

Paritäts-Register „Stereo-Imaging/Azimuth“: der Sinn-Beweis-Generator
(`scripts/erstelle_schadens_proben.py`) arbeitete mono — eine Stereo-Kette
fehlte. Diese Tests pinnen das neue Rezept:

1. Deterministisch: zwei Läufe mit identischem Seed ⇒ identischer Hash
   (§G5 (copilot-instructions.md) — gleicher Input + gleiche Version ⇒
   bit-identischer Output).
2. Schaden im ±1,5–6-dB-Bereich, Ausgabe bleibt Stereo (2, N) @48 kHz —
   KEIN Mono-Kollaps (§III (copilot-instructions.md) Stereo-Layout-Invariante:
   channels-first (C, N)).
3. Balance-Metrik-Richtung: Schaden entfernt sich vom Quell-Wert,
   Restauration nähert sich an (JND 0,5 dB, eine zentrale Konstante —
   §G5 (GEBOTE.md) Konsistenz-Mandat).
4. Backward-Compat-Pins: Bestands-Rezepte bleiben mono und bit-identisch.

Modell-frei, CPU-only, deterministisch: kein Backend-Import nötig —
`schaden_rezept` (Stereo-Zweig) und `bewerte_paar` mit gefälschten Metriken
sind reine NumPy-Logik (Muster: tests/unit/test_bewerte_schadens_proben.py).
"""

from __future__ import annotations

import hashlib
import inspect

import numpy as np
import pytest

SR = 48000


def _stereo_sinus(n: int = 48000, freq: float = 440.0) -> np.ndarray:
    """Ausgeglichenes Stereo-Signal (2, N) mit L == R — Balance 0 dB."""
    t = np.arange(n, dtype=np.float64) / SR
    kanal = (0.4 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return np.stack([kanal, kanal.copy()])


def _neutrale_metriken(monkeypatch) -> None:
    """Ersetzt `_metriken` durch identische Werte für clean/schaden/restauriert.

    Die bestehenden defektlast/musik_mos-Kriterien sind damit neutral — die
    Balance-Richtung entscheidet allein (modellfrei: Logik statt Netz).
    """
    import scripts.bewerte_schadens_proben as mod

    aufrufe = iter([{"defektlast": 0.5, "musik_mos": 4.0}] * 3)

    def _f(audio):
        return dict(next(aufrufe))

    monkeypatch.setattr(mod, "_metriken", _f)


# ─── 1. Determinismus (§G5 (copilot-instructions.md)) ────────────────────────


def test_stereo_imbalance_ist_deterministisch_gleicher_hash() -> None:
    """Zwei Läufe mit identischem Seed liefern identische Arrays und Hashes."""
    from scripts.erstelle_schadens_proben import schaden_rezept

    x = _stereo_sinus()

    def lauf() -> np.ndarray:
        return schaden_rezept(x, "stereo_imbalance", np.random.default_rng(20261118))

    a, b = lauf(), lauf()
    assert np.array_equal(a, b)
    assert hashlib.sha256(a.tobytes()).hexdigest() == hashlib.sha256(b.tobytes()).hexdigest()


# ─── 2. Schaden im dB-Bereich + Stereo-Form (2, N) ───────────────────────────


def test_stereo_imbalance_schaden_db_bereich_und_stereo_form() -> None:
    """Ausgabe bleibt Stereo (2, N), endlich — und der Schaden liegt im
    ±1,5–6-dB-Bereich (Stereo-Layout-Invariante §III (copilot-instructions.md),
    kein Mono-Kollaps). Bei ausgeglichener Quelle ist der Balance-Abstand zum
    Quell-Wert exakt die gezogene Gain-Stärke (Richtung: Schaden entfernt sich)."""
    from scripts.bewerte_schadens_proben import _stereo_balance_ratio
    from scripts.erstelle_schadens_proben import (
        STEREO_IMBALANCE_MAX_DB,
        STEREO_IMBALANCE_MIN_DB,
        schaden_rezept,
    )

    assert STEREO_IMBALANCE_MIN_DB == 1.5 and STEREO_IMBALANCE_MAX_DB == 6.0
    x = _stereo_sinus()  # Quelle ausgeglichen ⇒ Balance 0 dB
    for seed in range(24):
        y = schaden_rezept(x, "stereo_imbalance", np.random.default_rng(seed))
        assert y.shape == x.shape
        assert y.shape[0] == 2 and y.shape[-1] == x.shape[-1]
        assert np.isfinite(y).all()
        betrag_db = abs(_stereo_balance_ratio(y))
        assert 1.5 - 1e-3 <= betrag_db <= 6.0 + 1e-3
        assert not np.array_equal(y, x)  # der Schaden ist hörbar da


def test_stereo_imbalance_mono_quelle_wird_auf_stereo_hochgeruestet() -> None:
    """Mono-Eingang wird defensiv auf (2, N) hochgerüstet — nie umgekehrt
    (KEIN Mono-Kollaps in der Schadens-Erzeugung)."""
    from scripts.bewerte_schadens_proben import _stereo_balance_ratio
    from scripts.erstelle_schadens_proben import schaden_rezept

    x = _stereo_sinus()[0]  # 1-D
    y = schaden_rezept(x, "stereo_imbalance", np.random.default_rng(3))
    assert y.ndim == 2 and y.shape[0] == 2 and y.shape[-1] == x.shape[-1]
    assert 1.5 - 1e-3 <= abs(_stereo_balance_ratio(y)) <= 6.0 + 1e-3


# ─── 3. Balance-Metrik-Richtung (JND 0,5 dB, §G5 (GEBOTE.md)) ────────────────


def test_balance_richtung_imbalance_zu_ausgeglichen_ist_besser(monkeypatch) -> None:
    """Künstlicher Fall: Quelle ausgeglichen, Schaden +4 dB links, Restauration
    exakt ausgeglichen ⇒ die Balance-Richtung macht das Paar BESSER — zusätzlich
    zu defektlast/musik_mos, die hier neutral bleiben (Paritäts-Register
    „Stereo-Imaging/Azimuth“)."""
    import scripts.bewerte_schadens_proben as mod

    _neutrale_metriken(monkeypatch)
    clean = _stereo_sinus()
    schaden = clean.copy()
    schaden[0] *= 10.0 ** (4.0 / 20.0)  # links +4 dB — weg vom Quell-Wert
    restauriert = clean.copy()  # perfekt ausgeglichen

    r = mod.bewerte_paar(clean, schaden, restauriert, rezept="stereo_imbalance")
    assert r["besser"] is True
    b = r["stereo_balance"]
    assert b["clean_db"] == pytest.approx(0.0, abs=1e-3)
    assert b["schaden_db"] == pytest.approx(4.0, abs=1e-3)
    assert b["restauriert_db"] == pytest.approx(0.0, abs=1e-3)
    assert b["abstand_schaden_db"] == pytest.approx(4.0, abs=1e-3)
    assert b["abstand_restauriert_db"] == pytest.approx(0.0, abs=1e-3)
    assert b["besser"] is True


def test_balance_richtung_verschlechterung_zaehlt_nicht(monkeypatch) -> None:
    """Restauration entfernt sich weiter vom Quell-Wert ⇒ Balance zählt NICHT
    als besser (die Richtung ist Teil des Kriteriums, nicht der Betrag)."""
    import scripts.bewerte_schadens_proben as mod

    _neutrale_metriken(monkeypatch)
    clean = _stereo_sinus()
    schaden = clean.copy()
    schaden[0] *= 10.0 ** (3.0 / 20.0)  # links +3 dB
    restauriert = clean.copy()
    restauriert[1] *= 10.0 ** (4.0 / 20.0)  # rechts −4 dB — weiter weg

    r = mod.bewerte_paar(clean, schaden, restauriert, rezept="stereo_imbalance")
    assert r["stereo_balance"]["besser"] is False
    assert r["besser"] is False  # Metriken neutral ⇒ Balance entscheidet allein


def test_balance_jnd_grenze_0_5_db(monkeypatch) -> None:
    """JND 0,5 dB: eine 0,4-dB-Restdifferenz verfehlt das Kriterium, 0,6 dB
    erfüllt es — die Schwelle ist zentral definiert (§G5 (GEBOTE.md))."""
    import scripts.bewerte_schadens_proben as mod

    assert mod.JND_BALANCE_DB == 0.5
    for gain_db, erwartet in ((0.4, False), (0.6, True)):
        _neutrale_metriken(monkeypatch)
        clean = _stereo_sinus()
        schaden = clean.copy()
        schaden[0] *= 10.0 ** (gain_db / 20.0)
        r = mod.bewerte_paar(clean, schaden, clean.copy(), rezept="stereo_imbalance")
        assert r["stereo_balance"]["besser"] is erwartet
        assert r["besser"] is erwartet


def test_stereo_rezept_bestehende_kriterien_bleiben_wirksam(monkeypatch) -> None:
    """defektlast/musik_mos (Mono-Mixdown) bleiben für das Stereo-Rezept
    unverändert wirksam — die Balance-Richtung kommt ZUSÄTZLICH dazu (OR)."""
    import scripts.bewerte_schadens_proben as mod

    werte = iter(
        [
            {"defektlast": 0.50, "musik_mos": 4.0},  # clean
            {"defektlast": 0.50, "musik_mos": 4.0},  # schaden
            {"defektlast": 0.30, "musik_mos": 4.4},  # restauriert
        ]
    )

    def _f(audio):
        return dict(next(werte))

    monkeypatch.setattr(mod, "_metriken", _f)
    clean = _stereo_sinus()
    schaden = clean.copy()
    schaden[0] *= 10.0 ** (4.0 / 20.0)
    restauriert = schaden.copy()  # Balance NICHT repariert, Klang schon

    r = mod.bewerte_paar(clean, schaden, restauriert, rezept="stereo_imbalance")
    assert r["stereo_balance"]["besser"] is False
    assert r["besser"] is True  # die bestehenden Metriken entscheiden positiv


def test_stereo_balance_ratio_grundwerte() -> None:
    """Grundwerte der Metrik: mono ⇒ 0 dB, ±6-dB-Gain ⇒ ±6 dB, Stille ⇒ 0 dB."""
    from scripts.bewerte_schadens_proben import _stereo_balance_ratio

    x = _stereo_sinus()
    assert _stereo_balance_ratio(x) == pytest.approx(0.0, abs=1e-6)
    assert _stereo_balance_ratio(x[0]) == pytest.approx(0.0, abs=1e-6)
    links_laut = x.copy()
    links_laut[0] *= 10.0 ** (6.0 / 20.0)
    assert _stereo_balance_ratio(links_laut) == pytest.approx(6.0, abs=1e-3)
    assert _stereo_balance_ratio(np.zeros((2, 1000), dtype=np.float32)) == pytest.approx(0.0, abs=1e-9)


# ─── 4. Backward-Compat-Pins (Bestands-Rezepte / Mono-Pfad) ──────────────────


def test_rezepte_reihenfolge_pinnt_mono_idx_seeds() -> None:
    """stereo_imbalance hängt HINTEN an beiden REZEPTE-Tupeln — die
    Rezept-Indizes der Bestands-Rezepte (und damit die rng_item-Seeds
    [seed, song, idx]) bleiben identisch (§G5 (copilot-instructions.md))."""
    from scripts.bewerte_schadens_proben import REZEPTE as B_REZEPTE
    from scripts.erstelle_schadens_proben import REZEPTE as E_REZEPTE

    erwartet = (
        "vinyl",
        "kassette",
        "clipping",
        "mp3_128",
        "hum",
        "wowflutter",
        "kompression",
        "kammfilter",
        "verzerrung",
        "clicks",
        "pitch_drift",
        "dropout",
        "raum",
        "lautheit",
        "amp_drift",
        "stereo_imbalance",
    )
    assert E_REZEPTE == erwartet
    assert B_REZEPTE == erwartet


def test_bestands_rezepte_bleiben_mono_und_deterministisch() -> None:
    """Backward-Compat-Pin: vinyl/kassette/clipping liefern für Mono-Eingang
    weiter 1-D-Ausgaben und sind deterministisch (gleicher Seed ⇒ identisch)."""
    from scripts.erstelle_schadens_proben import schaden_rezept

    t = np.arange(48000, dtype=np.float64) / 48000.0
    x = (0.4 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    for rezept in ("vinyl", "kassette", "clipping"):
        a = schaden_rezept(x, rezept, np.random.default_rng(7))
        b = schaden_rezept(x, rezept, np.random.default_rng(7))
        assert a.ndim == 1, f"{rezept} darf nicht auf Stereo umgestellt werden"
        assert a.shape == x.shape
        assert np.array_equal(a, b)


def test_mono_decode_pfad_quelltext_pin() -> None:
    """Quelltext-Pins: der Mono-Decodierpfad (`-ac 1`) und die Mono-Zweige der
    Bestands-Rezepte sind unverändert — NUR stereo_imbalance dekodiert stereo
    (`-ac 2`) und der Pipeline-Aufruf reicht das Stereo-Array unverändert durch
    (§III (copilot-instructions.md), kein stiller Mono-Kollaps)."""
    from scripts.erstelle_schadens_proben import (
        _decode_48k,
        _decode_48k_stereo,
        _mp3_128,
        restauriere_schnipsel,
        schaden_rezept,
    )

    assert '"-ac"' in inspect.getsource(_decode_48k)
    assert '"1"' in inspect.getsource(_decode_48k)
    assert '"-ac", "1"' in inspect.getsource(_mp3_128)
    assert '"-ac", "2"' in inspect.getsource(_decode_48k_stereo)
    src_rezept = inspect.getsource(schaden_rezept)
    for token in (
        'defekt_variante(x, "bandlimit", rng)',
        'defekt_variante(y, "rausch", rng)',
        "klicker(y, rng, dichte=4.0)",
        "np.tanh(8.0 * x) / np.tanh(8.0)",
        "return _mp3_128(x)",
    ):
        assert token in src_rezept
    # Pipeline-Aufruf: Stereo-Array durchreichen, kein Downmix davor.
    src_pipe = inspect.getsource(restauriere_schnipsel)
    assert "audio_48k=audio" in src_pipe
    assert "denker.denke(" in src_pipe
    assert "mean(" not in src_pipe


def test_mono_pfad_unveraendert_kein_balance_schluessel(monkeypatch) -> None:
    """Mono-Rezepte bleiben exakt beim alten Verhalten: kein stereo_balance-
    Schlüssel, besser-Logik unverändert (Backward-Compat-Pin)."""
    import scripts.bewerte_schadens_proben as mod

    werte = iter(
        [
            {"defektlast": 0.20, "musik_mos": 4.8},  # clean
            {"defektlast": 0.50, "musik_mos": 4.0},  # schaden
            {"defektlast": 0.30, "musik_mos": 4.4},  # restauriert
        ]
    )

    def _f(audio):
        return dict(next(werte))

    monkeypatch.setattr(mod, "_metriken", _f)
    r = mod.bewerte_paar(np.zeros(10), np.zeros(10), np.zeros(10))
    assert r["besser"] is True
    assert "stereo_balance" not in r


# ─── Layout-Helfer (Stereo-Layout-Invariante, §III (copilot-instructions.md)) ─


def test_schreib_layout_normalisiert_channels_first_fuer_sf_write() -> None:
    """(C, N) wird für sf.write zu (Frames, Kanäle); Mono und (N, C) bleiben
    unverändert (bit-identisch für die Bestands-Rezepte)."""
    from scripts.erstelle_schadens_proben import _schreib_layout

    assert _schreib_layout(np.zeros((2, 4800), np.float32)).shape == (4800, 2)
    assert _schreib_layout(np.zeros((4800, 2), np.float32)).shape == (4800, 2)
    assert _schreib_layout(np.zeros(4800, np.float32)).shape == (4800,)


def test_mittel_ausschnitt_erhaelt_stereo_layout() -> None:
    """Der Mittel-Ausschnitt schneidet die Zeitachse — das (2, N)-Layout bleibt."""
    from scripts.erstelle_schadens_proben import _mittel_ausschnitt

    x = np.zeros((2, 48000), dtype=np.float32)
    assert _mittel_ausschnitt(x, 0.5).shape == (2, 24000)
