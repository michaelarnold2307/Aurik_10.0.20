"""§D-K3-61 (2026-10-08): P3/P4-Triage des Anti-Pattern-Scanners — Wirkungs-Pins.

Bughunt-Befund: Der Pre-Commit-Scanner (`scan_anti_patterns.py`) meldete 198
„potentielle Bugs", davon nach Stichprobe (18 Stellen geprüft) **17 False
Positives**: bare `noverlap=<variable>`/`<literal>` wurde pauschal als
Crash-Risiko gewertet, obwohl die Werte im Umfeld belegbar sicher sind
(Tuple-/Signatur-Literale mit ``noverlap < nperseg``, ``min(...)``-Clamps,
``nperseg * 3 // 4``-Arithmetik, ``_safe_stft_params``-Helper), ebenso
funktionslokale ``import os``-Importe (P4). Fix: belegte Triage-Klassen in
``_p3_bare_variable_is_safe`` / ``_p3_subtrahend_is_safe`` / P4-Vorlauf-Import;
Wirkung: 198 → 67 Findings (dieselbe Codebasis).

Gepinnt — jeweils mit POSITIV-KONTROLLE, damit die Triage echte
Crash-Kandidaten (bare Variable/npseg ohne jeden Beleg, ``- 0``) nie schluckt.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SCANNER = Path(__file__).resolve().parents[2] / ".agents" / "skills" / "bug-prevention" / "scan_anti_patterns.py"

_spec = importlib.util.spec_from_file_location("scan_anti_patterns_under_test", _SCANNER)
assert _spec is not None and _spec.loader is not None
_scan = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_scan)


def _p3(src: str) -> list[str]:
    return list(_scan.check_stft_without_clamp("test_modul.py", src))


def _p4(src: str) -> list[str]:
    return list(_scan.check_os_without_import("test_modul.py", src))


# ── P3: belegte Safe-Klassen (False-Positive-Fälle von 2026-10-08) ──────────


def test_p3_tuple_literal_kleinwert_ist_safe() -> None:
    src = "nperseg, noverlap = 2048, 1536\n_, _, Z = _stft(x, nperseg=nperseg, noverlap=noverlap)\n"
    assert _p3(src) == []


def test_p3_literal_zahlen_ist_safe() -> None:
    src = "_, _, Z = stft(x, fs=sr, nperseg=2048, noverlap=1536)\n"
    assert _p3(src) == []


def test_p3_clamp_variable_ist_safe() -> None:
    src = "_nperseg = min(nperseg, n)\n_noverlap = min(noverlap, max(0, _nperseg - 1))\nZ = _stft(x, nperseg=_nperseg, noverlap=_noverlap)\n"
    assert _p3(src) == []


def test_p3_faktor_arithmetik_ist_safe() -> None:
    src = "noverlap = self._NPERSEG * 3 // 4\nZ = _stft(x, nperseg=self._NPERSEG, noverlap=noverlap)\n"
    assert _p3(src) == []


def test_p3_subtrahend_mit_belegtem_hop_ist_safe() -> None:
    src = "nperseg = min(4096, n)\nhop = nperseg // 4\nZ = _istft(z, fs=sr, nperseg=nperseg, noverlap=nperseg - hop)\n"
    assert _p3(src) == []


def test_p3_safe_helper_quelle_ist_safe() -> None:
    src = "nperseg, noverlap, hop = self._safe_stft_params(sizes, default_nperseg=512)\nZ = _stft(x, nperseg=nperseg, noverlap=noverlap)\n"
    assert _p3(src) == []


# ── P3: Positiv-Kontrollen — echte Crash-Kandidaten bleiben sichtbar ────────


def test_p3_bare_variable_ohne_beleg_bleibt_finding() -> None:
    src = "Z = _stft(x, nperseg=n_fft, noverlap=overlap_var)\n"
    assert _p3(src)


def test_p3_direkte_nperseg_variable_bleibt_finding() -> None:
    src = "Z = _stft(x, nperseg=n_fft, noverlap=n_fft)\n"
    assert _p3(src)


def test_p3_minus_null_bleibt_finding() -> None:
    src = "Z = _stft(x, nperseg=n_fft, noverlap=n_fft - 0)\n"
    assert _p3(src)


def test_p3_literal_groesser_gleich_nperseg_bleibt_finding() -> None:
    src = "Z = _stft(x, nperseg=2048, noverlap=2048)\n"
    assert _p3(src)


# ── P4: funktionslokaler Import ─────────────────────────────────────────────


def test_p4_funktionslokaler_import_vor_nutzung_ist_safe() -> None:
    src = 'if __name__ == "__main__":\n    import os\n\n    os.makedirs("x", exist_ok=True)\n'
    assert _p4(src) == []


def test_p4_nutzung_ohne_import_bleibt_finding() -> None:
    src = 'p = os.path.join("a", "b")\n'
    assert _p4(src)
