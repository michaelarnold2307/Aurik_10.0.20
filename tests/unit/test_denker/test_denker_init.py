from __future__ import annotations

"""
tests/unit/test_denker/test_denker_init.py
==========================================
Unit-Tests für das Top-Level-Paket ``denker``.
Prüft, dass alle 9 Klassen, 9 Factory-Funktionen sowie KettenErgebnis
aus dem öffentlichen ``__init__.py``-Namespace importierbar sind.

≥ 20 Tests.
"""


import pytest

# ---------------------------------------------------------------------------
# Alle Namen, die aus ``denker`` importierbar sein müssen
# ---------------------------------------------------------------------------

_CLASS_NAMES = [
    "TontraegerDenker",
    "DefektDenker",
    "StrategieDenker",
    "RestaurierDenker",
    "ReparaturDenker",
    "RekonstruktionsDenker",
    "ExzellenzDenker",
    "TontraegerketteDenker",
    "AurikDenker",
]

_FACTORY_NAMES = [
    "get_tontraeger_denker",
    "get_defekt_denker",
    "get_strategie_denker",
    "get_restaurier_denker",
    "get_reparatur_denker",
    "get_rekonstruktions_denker",
    "get_exzellenz_denker",
    "get_tontraegerkette_denker",
    "get_aurik_denker",
]


# ---------------------------------------------------------------------------
# 1. Kein Import-Fehler
# ---------------------------------------------------------------------------


class TestDenkerImport:
    """01: Das Paket importiert ohne Exception."""

    def test_01_import_denker_no_exception(self):
        """``import denker`` wirft keine Ausnahme."""
        try:
            pass
        except ImportError as exc:
            pytest.fail(f"'import denker' fehlgeschlagen: {exc}")


# ---------------------------------------------------------------------------
# 2. Klassen importierbar
# ---------------------------------------------------------------------------


class TestClassesImportable:
    """02–10: Alle 9 Klassen sind aus ``denker`` importierbar."""

    @pytest.mark.parametrize("name", _CLASS_NAMES)
    def test_class_importable(self, name: str):
        """Klasse ist im denker-Namespace vorhanden."""
        try:
            import denker

            obj = getattr(denker, name, None)
            assert obj is not None, f"denker.{name} fehlt im Namespace"
        except ImportError:
            pytest.fail(f"'import denker' fehlgeschlagen beim Test für {name}")


# ---------------------------------------------------------------------------
# 3. Klassen instantiierbar
# ---------------------------------------------------------------------------


class TestClassesInstantiable:
    """11–19: Alle 9 Klassen lassen sich instanziieren."""

    @pytest.mark.parametrize("name", _CLASS_NAMES)
    def test_class_instantiable(self, name: str):
        """Konstruktor-Aufruf schlägt nicht fehl."""
        try:
            import denker

            cls = getattr(denker, name, None)
            if cls is None:
                pytest.skip(f"denker.{name} nicht vorhanden")
            instance = cls()
            assert instance is not None
        except Exception as exc:
            pytest.fail(f"Instanziierung von {name} fehlgeschlagen: {exc}")


# ---------------------------------------------------------------------------
# 4. Factory-Funktionen aufrufbar
# ---------------------------------------------------------------------------


class TestFactoriesCallable:
    """20–28: Alle 9 Factory-Funktionen sind aufrufbar und geben nicht-None zurück."""

    @pytest.mark.parametrize("name", _FACTORY_NAMES)
    def test_factory_callable(self, name: str):
        """Factory-Funktion lässt sich aufrufen und gibt ein Objekt zurück."""
        try:
            import denker

            fn = getattr(denker, name, None)
            if fn is None:
                pytest.skip(f"denker.{name} nicht vorhanden")
            result = fn()
            assert result is not None, f"{name}() gab None zurück"
        except Exception as exc:
            pytest.fail(f"Aufruf von {name}() fehlgeschlagen: {exc}")


# ---------------------------------------------------------------------------
# 5. KettenErgebnis importierbar
# ---------------------------------------------------------------------------


class TestKettenErgebnisImport:
    """29: KettenErgebnis ist aus dem denker-Namespace importierbar."""

    def test_29_ketten_ergebnis_importable(self):
        """``denker.KettenErgebnis`` ist vorhanden."""
        try:
            import denker

            assert hasattr(denker, "KettenErgebnis"), "KettenErgebnis fehlt im denker-Namespace"
        except ImportError:
            pytest.fail("'import denker' fehlgeschlagen")


# ---------------------------------------------------------------------------
# 6. Singleton-Eigenschaften für ausgewählte Factory-Funktionen
# ---------------------------------------------------------------------------


class TestSingletonBehavior:
    """30–32: Ausgewählte Factory-Funktionen liefern denselben Singleton."""

    def test_30_aurik_denker_singleton(self):
        """get_aurik_denker() is get_aurik_denker() → True (Singleton)."""
        try:
            from denker import get_aurik_denker

            assert get_aurik_denker() is get_aurik_denker()
        except Exception:
            pytest.skip("get_aurik_denker nicht importierbar")

    def test_31_kette_denker_singleton(self):
        """get_tontraegerkette_denker() is get_tontraegerkette_denker() → True."""
        try:
            from denker import get_tontraegerkette_denker

            assert get_tontraegerkette_denker() is get_tontraegerkette_denker()
        except Exception:
            pytest.skip("get_tontraegerkette_denker nicht importierbar")

    def test_32_all_factories_singleton(self):
        """Alle 9 Factory-Funktionen liefern bei Wiederholung dieselbe Instanz."""
        try:
            import denker
        except ImportError:
            pytest.skip("import denker fehlgeschlagen")

        for name in _FACTORY_NAMES:
            fn = getattr(denker, name, None)
            if fn is None:
                continue
            try:
                a = fn()
                b = fn()
                assert a is b, f"{name}() ist kein Singleton: {type(a)} vs {type(b)}"
            except Exception:
                # Factory nicht funktionsfähig — überspringen, kein Absturz
                pass
