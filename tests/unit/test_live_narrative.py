"""Tests für das Live-Narrativ (Aurik10/ui/live_narrative.py).

Hintergrund (§v10.305 copilot-instructions.md, Befund 2026-10-07): Vorher wählte
die Oberfläche bei langen Pausen aus **drei** festen Beruhigungssätzen und
wiederholte denselben Satz im 8-Sekunden-Takt — bei einer 30-Minuten-Phase rund
200-mal, ohne eine einzige neue Information. Der Nutzereindruck war „monoton".

Diese Tests sichern die drei Eigenschaften, die das verhindern:

1. **Rotation** — jede Fassung eines Bandes kommt innerhalb eines Zyklus vor.
2. **Keine Wiederholung in Folge** — zwei aufeinander folgende Meldungen sind nie
   identisch (das ist die eigentliche Zusage gegen die Monotonie).
3. **Fortschritt im Text** — die Zeile trägt Störungs-Fortschritt und Restzeit,
   statt nur den Zustand zu wiederholen.

Dazu die Determinismus-Zusage nach §G5 (copilot-instructions.md): Die Auswahl ist
eine reine Funktion aus Band und Meldungszähler — kein Zufall, keine Uhrzeit (ein
Test hält genau das fest).
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

_MODULE_PATH = pathlib.Path(__file__).resolve().parents[2] / "Aurik10" / "ui" / "live_narrative.py"


def _nav():
    """Das Narrativ-Modul laden — ohne `Aurik10.ui`-Paketimport (kein Qt im Test)."""
    spec = importlib.util.spec_from_file_location("_live_narrative_under_test", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
def test_band_boundaries_match_previous_thresholds() -> None:
    """Bandgrenzen bleiben 12 % / 92 % — der Tonwechsel verschiebt sich nicht."""
    nav = _nav()
    assert nav.band_for(0.0) == "analysis"
    assert nav.band_for(11.99) == "analysis"
    assert nav.band_for(12.0) == "working"
    assert nav.band_for(91.99) == "working"
    assert nav.band_for(92.0) == "finalize"
    assert nav.band_for(100.0) == "finalize"


@pytest.mark.unit
@pytest.mark.parametrize("band", ["analysis", "working", "finalize"])
def test_rotation_uses_every_variant_within_one_cycle(band: str) -> None:
    """Jede Fassung eines Bandes erscheint innerhalb eines Durchlaufs."""
    nav = _nav()
    variants = nav.BAND_VARIANTS[band]
    seen = {nav.variant_key(band, i) for i in range(len(variants))}
    assert seen == set(variants), f"Band {band}: nicht alle Fassungen erreichbar"


@pytest.mark.unit
@pytest.mark.parametrize("band", ["analysis", "working", "finalize"])
def test_no_two_consecutive_messages_are_identical(band: str) -> None:
    """Die Kernzusage: keine Meldung wiederholt die unmittelbar vorangehende.

    200 Meldungen entsprechen ~13 Minuten im 4-Sekunden-Takt — die Größenordnung
    einer echten langen Phase. Vorher war hier jede zweite Meldung identisch.
    """
    nav = _nav()
    keys = [nav.variant_key(band, i) for i in range(200)]
    for i in range(1, len(keys)):
        assert keys[i] != keys[i - 1], f"Wiederholung an Stelle {i} im Band {band}"


@pytest.mark.unit
@pytest.mark.parametrize("band", ["analysis", "working", "finalize"])
def test_second_cycle_is_not_identical_to_the_first(band: str) -> None:
    """Auch die Reihenfolge wiederholt sich nicht — sonst klänge der Zyklus gleich."""
    nav = _nav()
    n = len(nav.BAND_VARIANTS[band])
    first = tuple(nav.variant_key(band, i) for i in range(n))
    second = tuple(nav.variant_key(band, i + n) for i in range(n))
    assert first != second, f"Band {band}: zweiter Durchlauf identisch mit dem ersten"


@pytest.mark.unit
def test_selection_is_deterministic_and_clock_free() -> None:
    """§G5 (copilot-instructions.md): gleicher Zähler ⇒ gleiche Fassung.

    Kein Zufall, keine Uhrzeit — sonst wäre die Zusage „gleicher Input, gleicher
    Output" für die Oberfläche nicht mehr prüfbar.

    Geprüft wird der **Syntaxbaum**, nicht der Quelltext: Die erste Fassung dieses
    Tests suchte Zeichenketten im Quelltext und fiel über den eigenen Docstring —
    das Wort „``time.time()``" steht dort als *Verbot* und wurde als Verstoß
    gelesen (§G9 copilot-instructions.md: dieselbe Prosa-Falle wie in D-K3-5 und
    D-K3-9). Ein Bau-Modul ohne AST sieht diesen Unterschied nicht.
    """
    nav = _nav()
    for band in ("analysis", "working", "finalize"):
        for i in (0, 3, 7, 42, 199):
            assert nav.variant_key(band, i) == nav.variant_key(band, i)

    import ast
    import inspect

    tree = ast.parse(inspect.getsource(nav))

    # (a) Reinheit als zwei Bedingungen: erlaubt sind ausschließlich reine
    #     Module (Typ-Hinweise, Mathematik) und die eigene Sprachschicht —
    #     verboten sind Zufall, Uhrzeit, Nebenläufigkeit, Qt und E/A. Die
    #     Positivliste nennt bewusst Modulwurzeln: `collections.abc.Callable`
    #     ist ein Typ-Hinweis und kein Seiteneffekt (§G5 copilot-instructions.md
    #     verbietet allein die Entscheidung aus Zufall oder Uhrzeit).
    pure_roots = {"__future__", "typing", "collections", "dataclasses", "enum", "math", "re"}
    impure_roots = {
        "random",
        "time",
        "datetime",
        "uuid",
        "secrets",
        "os",
        "subprocess",
        "socket",
        "threading",
        "multiprocessing",
        "asyncio",
        "PySide6",
        "PyQt5",
        "PyQt6",
    }
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    verboten = imported & impure_roots
    assert not verboten, f"Nichtdeterministische oder nebenwirkende Quelle im Narrativ: {sorted(verboten)}"
    unbekannt = imported - pure_roots - {"Aurik10"}
    assert not unbekannt, f"Unerlaubter Import im Narrativ (Reinheit): {sorted(unbekannt)}"

    # (b) Kein Zugriff auf Zufalls-/Zeitquellen als Modulname (`time.time`,
    #     `random.choice`, `datetime.now`, `uuid.uuid4`).
    used_modules = {
        node.value.id for node in ast.walk(tree) if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
    }
    forbidden = used_modules & {"random", "time", "datetime", "uuid"}
    assert not forbidden, f"Nichtdeterministische Quelle im Narrativ: {sorted(forbidden)}"


@pytest.mark.unit
def test_milestones_fire_once_each_in_order() -> None:
    """Meilensteine genau einmal: 25 %, 50 %, 75 %, 100 %."""
    nav = _nav()
    seen: set[str] = set()
    fired: list[str] = []
    for done in (1, 5, 10, 25, 26, 50, 75, 99, 100, 100):
        key = nav.milestone_key(done, 100, seen)
        if key is not None:
            seen.add(key)
            fired.append(key)
    assert fired == list(nav.MILESTONE_KEYS), fired

    # Nach dem letzten Meilenstein kommt nichts mehr — kein Dauer-Toast.
    assert nav.milestone_key(100, 100, seen) is None


@pytest.mark.unit
def test_milestones_need_a_known_total() -> None:
    """Ohne bekannte Gesamtzahl gibt es keine Meilensteinmeldung (kein Raten)."""
    nav = _nav()
    assert nav.milestone_key(0, 0, set()) is None
    assert nav.milestone_key(5, 0, set()) is None
    assert nav.milestone_key(0, 10, set()) is None


@pytest.mark.unit
def test_compose_carries_progress_and_remaining_time() -> None:
    """Die Zeile trägt Fortschritt und Restzeit — nicht nur den Zustand."""
    nav = _nav()

    def fake_msg(key: str, **kwargs: object) -> str:
        return f"{key}|" + ",".join(f"{k}={v}" for k, v in sorted(kwargs.items()))

    text = nav.compose_reassure(
        band="working",
        index=0,
        defects_done=12,
        defects_total=31,
        eta_text="ungefähr 7 Min.",
        msg=fake_msg,
    )
    assert nav.BAND_VARIANTS["working"][0] in text
    assert "narrative.clause.defects" in text
    assert "done=12" in text and "total=31" in text
    assert "narrative.clause.eta" in text
    assert "eta=ungefähr 7 Min." in text


@pytest.mark.unit
def test_compose_without_data_stays_a_single_clause() -> None:
    """Ohne Fortschrittsdaten bleibt nur der Situationssatz — kein leerer Rest."""
    nav = _nav()
    text = nav.compose_reassure(band="analysis", index=1, msg=lambda key, **kwargs: key)
    assert text == nav.BAND_VARIANTS["analysis"][1]
    assert "  ·  " not in text


@pytest.mark.unit
def test_all_narrative_keys_exist_in_both_languages() -> None:
    """i18n-Pflicht (§VI.5 copilot-instructions.md): de und en symmetrisch."""
    nav = _nav()
    import Aurik10.i18n as i18n_module

    translations = getattr(i18n_module, "_TRANSLATIONS", {})
    de = translations.get("de", {})
    en = translations.get("en", {})

    needed = set(nav.MILESTONE_KEYS) | {"narrative.clause.defects", "narrative.clause.eta"}
    for keys in nav.BAND_VARIANTS.values():
        needed |= set(keys)

    missing_de = sorted(k for k in needed if k not in de)
    missing_en = sorted(k for k in needed if k not in en)
    assert not missing_de, f"Fehlt im deutschen Katalog: {missing_de}"
    assert not missing_en, f"Fehlt im englischen Katalog: {missing_en}"


@pytest.mark.unit
def test_narrative_texts_are_free_of_internal_jargon() -> None:
    """Kein interner Jargon im Nutzertext (Befund 2026-10-07).

    Die ersetzten Texte trugen „Aurik Denker", „Phase(n) injiziert" und
    „Off-Track" — Begriffe aus der Architektur, nicht aus der Welt des Nutzers.
    Der Test ist die Sperre dagegen, dass sie über neue Texte zurückkehren.
    """
    nav = _nav()
    import Aurik10.i18n as i18n_module

    de = getattr(i18n_module, "_TRANSLATIONS", {}).get("de", {})
    keys = {k for keys in nav.BAND_VARIANTS.values() for k in keys} | set(nav.MILESTONE_KEYS)
    keys |= {
        "narrative.clause.defects",
        "narrative.clause.eta",
        "narrative.thinker_skipped",
        "narrative.thinker_injected",
        "narrative.thinker_risk",
        "narrative.batch_celebrate",
        "status.time_limit_securing",
        "status.off_track_correcting",
        "status.processing_wait_short",
    }

    jargon = ("Denker", "Off-Track", "Rechenintensive", "Phase(n)", "injiziert")
    for key in sorted(keys):
        text = de.get(key, "")
        for word in jargon:
            assert word not in text, f"{key} enthält Jargon „{word}“: {text!r}"
        # Typografie: kein ASCII-Bindestrich als Gedankenstrich in Fließtexten.
        assert " - " not in text, f"{key} nutzt „ - “ statt Gedankenstrich: {text!r}"
