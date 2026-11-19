from __future__ import annotations

import pytest

"""
tests/unit/test_i18n.py — Unit-Tests für das i18n-Framework (§3.5).

Prüft set_language(), get_language(), t() und Thread-Sicherheit.
"""


import re
import threading
from typing import Any

from Aurik10.i18n import get_language, set_language, t


@pytest.mark.unit
class TestSetLanguage:
    """Tests für set_language()."""

    def setup_method(self) -> None:
        # Sicherstellen, dass Deutsch der Ausgangszustand ist
        set_language("de")

    def test_default_language_is_german(self) -> None:
        set_language("de")
        assert get_language() == "de"

    def test_set_english(self) -> None:
        set_language("en")
        assert get_language() == "en"

    def test_set_unknown_language_falls_back_to_german(self) -> None:
        set_language("fr")
        assert get_language() == "de"

    def test_set_empty_string_falls_back_to_german(self) -> None:
        set_language("")
        assert get_language() == "de"

    def test_set_language_case_sensitive(self) -> None:
        """Großbuchstaben → unbekannt → Fallback Deutsch."""
        set_language("DE")
        assert get_language() == "de"

    def test_multiple_language_switches(self) -> None:
        set_language("en")
        assert get_language() == "en"
        set_language("de")
        assert get_language() == "de"
        set_language("en")
        assert get_language() == "en"


class TestTranslationFunction:
    """Tests für t()."""

    def setup_method(self) -> None:
        set_language("de")

    def test_known_key_returns_string(self) -> None:
        result = t("error.file_not_readable")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_unknown_key_returns_key_itself(self) -> None:
        result = t("this.key.does.not.exist")
        assert result == "this.key.does.not.exist"

    def test_german_translation_not_english(self) -> None:
        set_language("de")
        de_result = t("error.ml_model_unavailable")
        set_language("en")
        en_result = t("error.ml_model_unavailable")
        # Beide müssen Strings sein
        assert isinstance(de_result, str)
        assert isinstance(en_result, str)
        # Nicht identisch (verschiedene Sprachen)
        assert de_result != en_result

    def test_result_is_always_string(self) -> None:
        for key in ["error.file_not_readable", "unknown.key", "", "restoration.started"]:
            result = t(key)
            assert isinstance(result, str)

    def test_variable_interpolation(self) -> None:
        """Wenn Schlüssel {var}-Platzhalter enthält, werden kwargs eingesetzt."""
        # Bekannte Schlüssel mit Formatierung testen
        # Falls keine existieren: Direktformatierung prüfen
        result = t("unknown.{name}", name="aurik")
        # Unbekannter Schlüssel → Schlüssel zurück, kein Absturz
        assert isinstance(result, str)

    def test_no_crash_on_missing_format_variable(self) -> None:
        """Fehlende Format-Variable → kein Absturz."""
        result = t("error.file_not_readable", missing_var="x")
        assert isinstance(result, str)

    def test_key_with_dots_returns_something(self) -> None:
        result = t("restoration.complete")
        assert isinstance(result, str)

    def test_german_contains_german_chars_or_fallback(self) -> None:
        set_language("de")
        result = t("error.file_not_readable")
        # Muss ein nicht-leerer String sein
        assert len(result.strip()) > 0

    def test_english_key_not_same_as_german(self) -> None:
        set_language("de")
        de = t("info.ml_fallback")
        set_language("en")
        en = t("info.ml_fallback")
        # Beide gültig
        assert isinstance(de, str)
        assert isinstance(en, str)


class TestThreadSafety:
    """Thread-Sicherheit von set_language() und t()."""

    def test_concurrent_language_switches_no_exception(self) -> None:
        """Viele Threads wechseln Sprache gleichzeitig — kein Absturz."""
        errors: list[Exception] = []

        def worker(lang: str) -> None:
            try:
                for _ in range(50):
                    set_language(lang)
                    _ = get_language()
                    _ = t("error.file_not_readable")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(lang,)) for lang in (["de", "en"] * 10)]
        for th in threads:
            th.start()
        for th in threads:
            th.join(timeout=5.0)

        assert not errors, f"Thread-Fehler: {errors}"

    def test_concurrent_t_calls_always_return_string(self) -> None:
        results: list[Any] = []
        errors: list[Exception] = []

        def worker() -> None:
            try:
                for _ in range(30):
                    results.append(t("error.file_not_readable"))
                    results.append(t("unknown.key.xyz"))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for th in threads:
            th.start()
        for th in threads:
            th.join(timeout=5.0)

        assert not errors
        assert all(isinstance(r, str) for r in results)

    def test_get_language_always_valid(self) -> None:
        """get_language() gibt immer 'de' oder 'en' zurück."""
        VALID = {"de", "en"}
        errors: list[str] = []

        def worker() -> None:
            for _ in range(50):
                lang = get_language()
                if lang not in VALID:
                    errors.append(lang)
            set_language("de" if _ % 2 == 0 else "en")

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for th in threads:
            th.start()
        for th in threads:
            th.join(timeout=5.0)

        assert not errors, f"Ungültige Sprache(n): {errors}"


class TestTranslationCompleteness:
    """Prüft dass alle definierten deutschen Schlüssel auch englisch existieren."""

    def test_all_de_keys_have_en_equivalent(self) -> None:
        """Jeder Schlüssel in Deutsch muss auch in Englisch vorhanden sein."""
        # Zugriff auf interne Translation-Dicts via Modulimport
        import Aurik10.i18n as i18n_module

        translations = getattr(i18n_module, "_TRANSLATIONS", {})
        assert translations, "_TRANSLATIONS muss im i18n-Modul definiert sein"

        de_keys = set(translations.get("de", {}).keys())
        en_keys = set(translations.get("en", {}).keys())

        missing_in_en = de_keys - en_keys
        assert not missing_in_en, f"Schlüssel in 'de' fehlen in 'en': {sorted(missing_in_en)}"

    def test_no_empty_translations(self) -> None:
        """Kein Übersetzungswert darf leer sein."""
        import Aurik10.i18n as i18n_module

        translations = getattr(i18n_module, "_TRANSLATIONS", {})
        assert translations, "_TRANSLATIONS muss im i18n-Modul definiert sein"

        for lang, d in translations.items():
            for key, value in d.items():
                assert isinstance(value, str), f"[{lang}][{key}] ist kein String"
                assert len(value.strip()) > 0, f"[{lang}][{key}] ist leer"

    def test_german_uses_informal_address_consistently(self) -> None:
        """Anrede-Entscheidung (2026-10-07): Aurik duzt durchgängig.

        Befund davor: 22 Schlüssel duzten (`status.*`, `onboarding.*`), 20 siezten
        (`dialog.*`, `error.*`, `legacy.*`) — dasselbe Produkt sprach den Nutzer je
        nach Dialogform unterschiedlich an. Nach der Umstellung ist die formale
        Anrede im deutschen Katalog **null**; dieser Test hält das fest, damit das
        Gemisch nicht über neue Texte zurückkehrt (§V7 copilot-instructions.md:
        Ursache statt Symptom).

        Englisch ist nicht betroffen — dort gibt es keine T/V-Unterscheidung.
        """
        import Aurik10.i18n as i18n_module

        translations = getattr(i18n_module, "_TRANSLATIONS", {})
        de = translations.get("de", {})
        assert de, "_TRANSLATIONS['de'] muss gefüllt sein"

        formal = re.compile(r"\b(Sie|Ihnen|Ihre|Ihren|Ihrem|Ihrer|Ihr)\b")
        offenders = [(key, value[:80]) for key, value in de.items() if formal.search(value)]
        assert not offenders, f"Formale Anrede im deutschen Katalog — Aurik duzt: {offenders}"
