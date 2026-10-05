#!/usr/bin/env python3
"""Aurik SOTA Bug-Prevention Hook (§v10.105)

Fängt die 6 Bug-Klassen aus der Exception-Forensik (Juli 2026) PROAKTIV ab,
BEVOR sie in die Pipeline gelangen.

Gefundene Anti-Patterns (aus 460 analysierten Exceptions):
  P1: shape[0] <= shape[1] — falsche Kanal-Detection (→ Broadcast-Crash)
  P2: filtfilt( ohne Längen-Guard     (→ padlen-Crash)
  P3: stft( ohne noverlap-Clamp       (→ noverlap-Crash)
  P4: os.* ohne import os            (→ UnboundLocalError)
  P5: np.asarray(Tuple) in __post_init__ (→ inhomogeneous-Crash)
  P6: MaterialType-Enum als String    (→ KeyError in Dict-Lookups)
"""

import ast
import logging
import os
import re
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Konfiguration ──────────────────────────────────────────────────────────

EXCLUDE_DIRS = {
    "__pycache__",
    ".git",
    ".venv",
    ".venv_aurik",
    "build",
    "dist",
    "models",
    "output_audio",
    "sessions",
    "logs",
    "data",
    "golden_samples",
    "chain_templates",
    "configs",
    ".eggs",
    "tests",  # Tests dürfen Anti-Patterns für negative Tests enthalten
}

EXCLUDE_FILES = {
    "setup.py",
    "conftest.py",
    # Fixer-Scripts beschreiben Anti-Patterns (nicht nutzen sie)
    "fix_p6_material_lookups.py",
    "fix_p6_v2.py",
}

MIN_SEVERITY = "warning"  # "error" stoppt Commit, "warning" warnt nur

# §v10.115: Continuous Analysis — Scanner lädt neue Patterns aus Exception-Forensik
_PATTERN_FEED_PATH = Path(__file__).resolve().parents[3] / "logs" / "discovered_patterns.json"

# ── P1: shape[0] <= shape[1] Anti-Pattern ─────────────────────────────────


def check_shape_anti_pattern(filepath: str, source: str) -> list[str]:
    """Findet `audio.shape[0] <= audio.shape[1]` ohne `shape[1] > 2`-Check."""
    issues = []
    # Regex: shape[0] <= shape[1] aber NICHT gefolgt von "and shape[1] > 2" auf gleicher Zeile
    pattern = re.compile(r"\.shape\[0\]\s*<=\s*\.shape\[1\]")
    lines = source.split("\n")
    for i, line in enumerate(lines, 1):
        if pattern.search(line):
            # Erlaubt wenn "shape[1] > 2" oder "shape[0] <= 2 and" auf gleicher Zeile
            if "shape[0] <= 2 and" in line or "shape[1] > 2" in line:
                continue
            # Erlaubt wenn in Kommentar
            if line.strip().startswith("#"):
                continue
            issues.append(
                f"{filepath}:{i}: P1 shape[0]<=shape[1] ohne channels-first-Guard "
                f"(→ Broadcast-Crash bei channels-last mit N≤2). "
                f"FIX: `shape[0] <= 2 and shape[1] > 2`"
            )
    return issues


# ── P2: filtfilt ohne Längen-Guard ────────────────────────────────────────


def check_filtfilt_without_guard(filepath: str, source: str) -> list[str]:
    """Findet bare `filtfilt(` oder `signal.filtfilt(` (nicht safe_filtfilt)."""
    issues = []
    lines = source.split("\n")
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        # Bare filtfilt( calls (nicht safe_filtfilt, nicht sosfiltfilt)
        if re.search(r"(?<!safe_)(?<!sos)(?<!_)(?<!\.)\bfiltfilt\(", stripped):
            # Skip spec_constitution.py — filtfilt inside ForbiddenPattern strings
            if "spec_constitution.py" in filepath:
                continue
            # Skip files that DEFINE safe_filtfilt (audio_utils.py)
            if "def safe_filtfilt" in source:
                continue
            # Prüfe ob safe_filtfilt importiert oder im File definiert ist
            if "from backend.core.audio_utils import safe_filtfilt" not in source and "safe_filtfilt" not in source:
                issues.append(
                    f"{filepath}:{i}: P2 filtfilt() ohne Längen-Guard "
                    f"(→ padlen-Crash bei kurzem Audio). "
                    f"FIX: `from backend.core.audio_utils import safe_filtfilt` + Ersetzung"
                )
    return issues


# ── P3: stft ohne noverlap-Clamp ──────────────────────────────────────────


_P3_GUARD_MARKERS = (
    "noverlap must be less than nperseg",  # bewusster Retry-Pfad (spectral_subtractor)
    "if n_fft <= hop",  # expliziter Guard (hybrid_ml_denoiser)
    "n_fft = hop + 1",
    "if hop <= 0",
    "max(1,",
    "min(n_fft - hop,",
    "min(nperseg - hop,",
)


def check_stft_without_clamp(filepath: str, source: str) -> list[str]:
    """Findet `stft(...)`-Aufrufe, bei denen `noverlap >= nperseg` eintreten kann.

    Crash-Bedingung bei scipy ist ausschließlich `noverlap >= nperseg`.
      - `noverlap=<expr>` OHNE Subtraktion (z. B. `noverlap=n_fft`) → Crash möglich.
      - `noverlap=<a> - <b>`: sicher, sobald `<b> >= 1` feststeht — über ein Literal,
        `max(1, …)`, einen Guard im Umfeld (`if n_fft <= hop:`, `n_fft = hop + 1`,
        `if nperseg < …: return`) oder einen bewussten Retry-Pfad
        („noverlap must be less than nperseg").
      - `noverlap=<a> - 0` → Crash möglich.

    Belegte Gegenproben (2026-10-05): bandwidth_extension (`nperseg = min(4096, n)`
    + Early-Return für `nperseg < 256` → `hop = nperseg // 4 >= 64`),
    hybrid_ml_denoiser (expliziter Guard) und spectral_subtractor (Retry-Pfad) —
    alle drei waren Fehlalarme der reinen Textsuche nach `noverlap=n_fft - hop`.
    """
    issues = []
    lines = source.split("\n")
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if "stft(" not in stripped or "noverlap=" not in stripped:
            continue
        if "min(" in stripped:  # Inline-Clamp
            continue
        match = re.search(r"noverlap\s*=\s*([^,\)]+)", stripped)
        expr = (match.group(1) if match else "").replace(" ", "")
        subtrahend = expr.rsplit("-", 1)[1] if "-" in expr else None
        body = "\n".join(lines[max(0, i - 30) : i + 5])
        guarded = any(marker in body for marker in _P3_GUARD_MARKERS)
        if subtrahend is None:
            crash_possible = True  # noverlap = nperseg/n_fft direkt
        elif subtrahend.isdigit():
            crash_possible = int(subtrahend) == 0
        else:
            crash_possible = not guarded  # ungeprüfte Variable als Subtrahend
        if crash_possible:
            issues.append(
                f"{filepath}:{i}: P3 stft() noverlap kann nperseg erreichen "
                f"(→ noverlap-Crash bei kurzem Audio). "
                f"FIX: `_noverlap = min(n_fft - hop, max(0, n_fft - 1))`"
            )
    return issues


# ── P4: os.* ohne import os ───────────────────────────────────────────────


def check_os_without_import(filepath: str, source: str) -> list[str]:
    """Findet `os.`-Nutzung ohne `import os` auf Module-Ebene."""
    issues = []
    if "os." not in source:
        return issues
    # Prüfe ob import os existiert (nicht in Funktionen, sondern auf Module-Ebene)
    has_module_import = bool(re.search(r"^(import os|from os import)", source, re.MULTILINE))
    if not has_module_import:
        # Prüfe ob os.* in einer Funktion verwendet wird (wo import fehlen könnte)
        try:
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute):
                    if isinstance(node.value, ast.Name) and node.value.id == "os":
                        issues.append(
                            f"{filepath}:{node.lineno}: P4 os.{node.attr} ohne `import os` "
                            f"(→ UnboundLocalError in bestimmten Umgebungen). "
                            f"FIX: `import os` am Modul-Anfang"
                        )
        except SyntaxError:
            logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
    return issues


# ── P5: np.asarray(Tuple) in PhaseResult.__post_init__ ─────────────────────


def check_asarray_tuple(filepath: str, source: str) -> list[str]:
    """Findet `np.asarray(self.audio)` ohne Tuple-Check in __post_init__."""
    issues = []
    if "def __post_init__" not in source:
        return issues
    if "np.asarray(self.audio" not in source and "np.asarray(self.audio" not in source:
        return issues
    # Prüfe ob Tuple-Check VOR asarray existiert
    if "isinstance(self.audio, (tuple, list))" not in source:
        issues.append(
            f"{filepath}: P5 np.asarray(self.audio) ohne Tuple→ndarray-Guard "
            f"(→ inhomogeneous-Crash bei Tuple-Rückgaben). "
            f"FIX: isinstance-Check vor np.asarray()"
        )
    return issues


# ── P6: MaterialType-Enum als String in Dict-Lookup ─────────────────────────


def check_enum_as_dict_key(filepath: str, source: str) -> list[str]:
    """Findet Dict-Lookups mit material/mat wo Keys MaterialType-Enums sind."""
    issues = []
    # Nur in Dateien die MaterialType importieren
    if "MaterialType" not in source:
        return issues
    lines = source.split("\n")
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        # Skip docstring/formula lines (inside triple-quoted strings)
        if "·" in stripped or "log10" in stripped:
            continue
        # Pattern: DICT[material] oder DICT.get(material) wo MaterialType-Enum-Keys
        if re.search(r"\[material\]", stripped) or re.search(r"\.get\(material[,\)]", stripped):
            # Prüfe ob Normalisierung existiert
            context_start = max(0, i - 3)
            context = "\n".join(lines[context_start:i])
            if (
                "isinstance(material, MaterialType)" not in context
                and 'hasattr(material, "value")' not in context
                and "_mat_enum_" not in context
                and ".get(_mat_" not in context
            ):
                issues.append(
                    f"{filepath}:{i}: P6 Dict-Lookup [material] ohne Enum-Normalisierung "
                    f"(→ KeyError wenn material String statt Enum). "
                    f"FIX: isinstance(material, MaterialType) + .get()-Fallback"
                )
    return issues


# ── H-Serie: Hörordnungs-/Exportqualitäts-Anti-Patterns ─────────────────────
# Muster, die Aurik daran hindern, die hochwertigsten Exportergebnisse für das
# menschliche Gehör zu liefern. Quellen: Hörordnung (hoerordnung.instructions.md),
# dsp.instructions.md, copilot-instructions (§V5/§G5), Befunde 2026-08-23.


def _inside_string(line: str, match: "re.Match[str]") -> bool:
    """True, wenn der Treffer innerhalb eines String-Literals der Zeile liegt.

    Verhindert False-Positives in Regel-Katalogen (z. B. griffinlim-Pattern
    in Linter-Skripten) und Docstring-Text.
    """
    _before = line[: match.start()]
    _in_str: str | None = None
    _esc = False
    for _ch in _before:
        if _in_str is not None:
            if _esc:
                _esc = False
            elif _ch == "\\":
                _esc = True
            elif _ch == _in_str:
                _in_str = None
        elif _ch in "\"'":
            _in_str = _ch
    return _in_str is not None


_H03_ANALYSIS_SKIPPED: list[str] = []


def _sosfilt_assignment_target(stripped: str) -> str | None:
    m = re.match(r"\s*([A-Za-z_]\w*)\s*=\s*(?!=)", stripped)
    return m.group(1) if m else None


def _filter_derived_names(source: str) -> set[str]:
    """Namen, die (auch abgeleitet, 2 Stufen) aus sosfilt/sosfiltfilt stammen.

    Dient der Unterscheidung der beiden normrelevanten Fälle:
      a) Bandfilter-Ergebnis auf das ORIGINAL addieren → zero-phase-Pflicht
         (Pre-Ringing/Pegelexplosion, .github/VERBOTEN.md),
      b) Crossover-Split-Sum: komplementäre Bänder werden untereinander summiert;
         dort bleibt `sosfilt` zulässig, sofern alle parallelen Bänder denselben
         Filtertyp nutzen (§v10.1013 / audio_utils.safe_sosfiltfilt).
    """
    derived: set[str] = set(re.findall(r"^\s*([A-Za-z_]\w*)\s*=\s*[^\n=]*?sosfilt(?:filt)?\(", source, re.MULTILINE))
    # Tupel-Unpacking mitnehmen: `audio_mid_proc, _ = wiener_mid.process(audio_mid, sr)`
    # leitet den Bandnamen weiter — sonst meldet die Regel den Crossover-Split-Sum
    # in advanced_dereverb als Verstoß (Befund 2026-10-05).
    derived |= set(
        re.findall(r"^\s*([A-Za-z_]\w*)\s*,\s*[A-Za-z_]\w*\s*=\s*[^\n=]*?sosfilt(?:filt)?\(", source, re.MULTILINE)
    )
    # Nullinitialisierte Akkumulatoren sind kein Dry-Pfad: `result = np.zeros_like(audio)`
    # und danach `result += band` ist eine Crossover-Rekonstruktion (Befund
    # 2026-10-05, _declip_core.multiband_ar_declip), keine Interferenz mit dem Original.
    derived |= set(re.findall(r"^\s*([A-Za-z_]\w*)\s*=\s*np\.(?:zeros|empty)\w*\(", source, re.MULTILINE))
    if not derived:
        return derived
    for _ in range(2):
        added = False
        for m in re.finditer(r"^\s*([A-Za-z_]\w*)\s*=\s*(.+)$", source, re.MULTILINE):
            name, rhs = m.group(1), m.group(2)
            if name in derived:
                continue
            if set(re.findall(r"[A-Za-z_]\w*", rhs)) & derived:
                derived.add(name)
                added = True
        # Tupel-Ziele der zweiten Ableitungsstufe (z. B. `x, _ = f(band_var)`)
        for m in re.finditer(r"^\s*([A-Za-z_]\w*)\s*,\s*[A-Za-z_]\w*\s*=\s*(.+)$", source, re.MULTILINE):
            name, rhs = m.group(1), m.group(2)
            if name in derived:
                continue
            if set(re.findall(r"[A-Za-z_]\w*", rhs)) & derived:
                derived.add(name)
                added = True
        if not added:
            break
    return derived


_MEASUREMENT_HELPERS = frozenset(
    {"np", "numpy", "math", "min", "max", "abs", "stack", "mean", "sqrt", "log10", "log", "float", "int", "clip"}
)


def _addition_partner(rhs: str, target: str) -> str | None:
    """Identifier direkt neben dem Ziel-Term in einer Summe (oder None).

    Nur der unmittelbare Additionspartner entscheidet: `low + band` → Partner
    `low`; `np.stack([a, b])` liefert keinen Partner (Mess-/Styling-Kontext);
    `x + 1e-10` liefert keinen Partner (Epsilon-Schutz, kein Signal-Add).
    """
    terms = re.split(r"(?<![eE])[+\-]", rhs)
    for idx, term in enumerate(terms):
        if not re.search(rf"\b{re.escape(target)}\b", term):
            continue
        for other in (
            terms[idx + 1] if idx + 1 < len(terms) else "",
            terms[idx - 1] if idx > 0 else "",
        ):
            ids = [
                i
                for i in re.findall(r"[A-Za-z_]\w*", re.sub(r"\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", " ", other))
                if i not in _MEASUREMENT_HELPERS
            ]
            if ids:
                return ids[0]
    return None


def _sosfilt_used_additively(stripped: str, lines: list[str], lineno: int, derived: set[str]) -> bool:
    """True, wenn das sosfilt-Ergebnis auf das Original addiert/gemischt wird.

    Norm (.github/VERBOTEN.md, Anti-Pattern-Tabelle): zero-phase ist Pflicht, wo
    das Bandfilter-Ergebnis auf das Originalsignal addiert wird; `sosfilt` bleibt
    für Analyse/Sidechain zulässig. Nicht gemeldet werden: Analyse-Envelopes
    (abs/hilbert/Energie), serielle Filterketten und Crossover-Split-Sums, deren
    Additionspartner selbst aus demselben Filtertyp stammen (`derived`).
    """
    if re.search(r"(\+\s*[\w.]*sosfilt\(|sosfilt\([^)]*\)\s*[+\-])", stripped):
        return True
    target = _sosfilt_assignment_target(stripped)
    if not target:
        return False
    names = rf"{re.escape(target)}\w*"
    add_rx = re.compile(rf"(\+\s*{names}\b|\b{names}\b\s*\+|\b{names}\b\s*\+=|\+=.*\b{names}\b)")
    mix_rx = re.compile(r"\b(mix|blend|crossfade|wet)\w*\b", re.IGNORECASE)
    for candidate in lines[lineno : lineno + 30]:
        match = add_rx.search(candidate)
        if match:
            variant = re.search(rf"\b({re.escape(target)}\w*)", match.group(0))
            name = variant.group(1) if variant else target
            rhs = candidate.split("=", 1)[-1]
            partner = _addition_partner(rhs, name)
            if partner is None and "+=" in candidate:
                lhs = candidate.split("+=", 1)[0]
                lhs_ids = [i for i in re.findall(r"[A-Za-z_]\w*", lhs) if i not in _MEASUREMENT_HELPERS]
                if lhs_ids:
                    partner = lhs_ids[-1]
            if partner is None:
                continue  # Epsilon-/Messkontext (z. B. `x + 1e-10`, `np.stack([...])`)
            if partner in derived:
                continue  # Crossover-Split-Sum (normerlaubt)
            return True
        if mix_rx.search(candidate) and re.search(names, candidate) and re.search(r"[+\-*]", candidate):
            return True
    return False


_H04_TTL_SKIPPED: list[str] = []

_TTL_MARKERS = (
    "os.remove",
    "os.unlink",
    "os.listdir",
    "os.scandir",
    "os.path.getmtime",
    "os.path.getctime",
    ".glob(",
    "rmtree",
    "_cleanup_checkpoint_files",
    "MAX_CHECKPOINT_AGE",
    "MAX_FINGERPRINT_AGE",
)


_RESAMPLE_LEN_RX = re.compile(r"(?:\bsignal\.resample|librosa\.resample)\(\s*[^,]+,\s*([^)]+)")


def _resample_target_length(stripped: str) -> bool:
    """True, wenn hier eine freie ZIEL-LÄNGE gesetzt wird — dann ist H05 relevant.

    Zeitachsen-Stabilität (Hörordnung §Zeitachse; Produktionsbefund 2026-08-23
    „224 s vs 30 s -> FATAL-Trim"):
      - `resample_poly(x, up, down)` ist ratio-basiert und damit strukturell
        zeitachsen-treu (Ausgabelänge ≈ len(x)·up/down) -> kein Befund.
      - `librosa.resample(x, orig_sr=…, target_sr=…)` ist eine Sample-Rate-
        Konvertierung; die Dauer bleibt erhalten -> kein Befund.
      - `signal.resample(x, N)` ist nur ungefährlich, wenn `N` aus der
        Originallänge abgeleitet ist (`len(...)`); ein freier Ziel-Längenwert
        (z. B. feste 48000 oder ein Fremd-Ausdruck) braucht den Guard.
    """
    if "resample_poly(" in stripped:
        return False
    if "librosa.resample(" in stripped and ("target_sr" in stripped or "orig_sr" in stripped):
        return False
    match = _RESAMPLE_LEN_RX.search(stripped)
    if not match:
        return False
    return "len(" not in match.group(1)


def _has_exempt_marker(lines: list[str], lineno: int) -> bool:
    """True, wenn die Zeile oder die beiden Folgezeilen den Exemptions-Marker tragen.

    `ruff-format` bricht lange Zeilen um; der Marker `# H-SCAN-EXEMPT:` rutscht
    dadurch auf die Folgezeile (Befund 2026-10-05, multi_track_specialist-Allpass)
    und die Ausnahme würde unbemerkt wirkungslos.
    """
    return any("# H-SCAN-EXEMPT:" in candidate for candidate in lines[lineno - 1 : lineno + 3])


def _ttl_lifecycle_context(lines: list[str], lineno: int) -> bool:
    """True, wenn die Zeile in einem Datei-Lebenszyklus/TTL-Kontext liegt.

    §G5 (copilot-instructions.md) verbietet Wall-Clock in der
    ENTSCHEIDUNGSLOGIK des Restaurierungs-Audios. TTL-Politik (Prüfpunkt-/Cache-
    Ablauf) braucht Wall-Clock, weil die verglichenen Zeitstempel persistiert
    und damit über Prozess-/Boot-Grenzen vergleichbar sein müssen — dieselbe
    Abgrenzung führt der Code-Weakness-Scanner als `wallclock_ttl_housekeeping`.
    Ohne diese Abgrenzung widersprechen sich beide Gates (Befund 2026-10-05).
    """
    start = 0
    for idx in range(lineno - 1, -1, -1):
        if lines[idx].lstrip().startswith("def "):
            start = idx
            break
    end = len(lines)
    for idx in range(lineno, len(lines)):
        if lines[idx].startswith("def "):
            end = idx
            break
    block = "\n".join(lines[start:end])
    return any(marker in block for marker in _TTL_MARKERS)


def check_hoerordnung_export_patterns(filepath: str, source: str) -> list[str]:
    """H-Serie: Psychoakustik-/Exportqualitäts-Schwachstellen im Code."""
    issues = []
    lines = source.split("\n")
    _derived = _filter_derived_names(source)
    _in_docstring = False
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        # Docstring-Zeilen (Triple-Quotes) sind Text, kein Code.
        if _in_docstring:
            if '"""' in line or "'''" in line:
                _in_docstring = False
            continue
        if '"""' in stripped or "'''" in stripped:
            _in_docstring = (line.count('"""') % 2 == 1) or (line.count("'''") % 2 == 1)
            if _in_docstring:
                continue

        # H01: nacktes astype(np.int16) ohne Dither (POW-r/TPDF) — §V5 (copilot-instructions.md)
        if re.search(r"astype\(np\.int16\)", stripped):
            if not re.search(r"powr|tpdf|dither|noise_shape", source, re.IGNORECASE):
                issues.append(
                    f"{filepath}:{i}: H01 nacktes astype(np.int16) ohne Dither "
                    f"(→ Quantisierungsrauschen, §V5 (copilot-instructions.md)). "
                    f"FIX: POW-r Type 3 / TPDF vor Int16-Export"
                )

        # H02: griffinlim() als Endschritt — VERBOTEN V05 (PGHI/Vocos-Pflicht)
        _m02 = re.search(r"\bgriffinlim\(", stripped)
        if _m02 and not _inside_string(stripped, _m02):
            issues.append(
                f"{filepath}:{i}: H02 griffinlim() in Produktionscode "
                f"(→ nicht-deterministisch, V05). "
                f"FIX: PGHI (pghi_reconstruct) oder Vocos"
            )

        # H03: sosfilt() ohne zero-phase (sosfiltfilt) im Master-Audio-Pfad —
        # dsp.instructions „Bandfilter — Zero-Phase“. Nur Phasen/DSP melden:
        # dort ist Filterung Signal-Verarbeitung, wo Phase nicht zum Original
        # addiert werden darf. Analyse-/Realtime-Kontexte sind ausgenommen.
        if (
            re.search(r"\bsosfilt\(", stripped)
            and ("/phases/" in filepath.replace("\\", "/") or "/dsp/" in filepath.replace("\\", "/"))
            and not _has_exempt_marker(lines, i)
        ):
            if _sosfilt_used_additively(stripped, lines, i, _derived):
                issues.append(
                    f"{filepath}:{i}: H03 sosfilt() statt sosfiltfilt() "
                    f"(→ Phase addiert zu Original, Zero-Phase-Verstoß). "
                    f"FIX: sosfiltfilt(sos, audio)"
                )
            else:
                _H03_ANALYSIS_SKIPPED.append(f"{filepath}:{i}")

        # H04: time.time() IN Entscheidungslogik (if/compare) — §G5 (copilot-instructions.md) Determinismus.
        # Reines Profiling (Zuweisung/Subtraktion) ist zulässig, ebenso TTL-/
        # Datei-Lebenszyklus-Entscheidungen (persistierte Zeitstempel brauchen
        # Wall-Clock) — letztere werden transparent mitgezählt.
        if re.search(r"\btime\.time\(\)", stripped) and re.search(
            r"\bif\b.*time\.time\(\)|time\.time\(\).*(?:<|>|==|!=|<=|>=)", stripped
        ):
            if _ttl_lifecycle_context(lines, i):
                _H04_TTL_SKIPPED.append(f"{filepath}:{i}")
            else:
                issues.append(
                    f"{filepath}:{i}: H04 time.time() in Entscheidungslogik "
                    f"(→ nicht-deterministisch, §G5 (copilot-instructions.md)). "
                    f"FIX: Session-Seed / monotonic statt wall-clock"
                )

        # H05: resample() ohne Längen-Guard im Master-Audio-Pfad (Zeitachsen-
        # Zerstörung — Befund 2026-08-23: 224s vs 30s → FATAL-Trim; Hörordnung:
        # Sample-Exaktheit der Zeitachse). Nur Phasen/DSP — SR-Konvertierung an
        # zentralen, bewusst guardierten Stellen ist ausgenommen.
        if (
            re.search(r"\b(librosa\.resample|signal\.resample|resample_poly)\(", stripped)
            and _resample_target_length(stripped)
            and ("/phases/" in filepath.replace("\\", "/") or "/dsp/" in filepath.replace("\\", "/"))
        ):
            if not re.search(
                r"_len_diff|shape\[[^]]*\].*==|Längen|laenge|len_mismatch|abs\(len\(",
                source,
                re.IGNORECASE,
            ):
                issues.append(
                    f"{filepath}:{i}: H05 resample() ohne Längen-Differenz-Guard "
                    f"(→ Zeitkompression bei Längen-Mismatch, Hörordnung §Zeitachse). "
                    f"FIX: >0.1% Differenz → trim/pad statt resample"
                )

        # H06: Hard-Clamp (-1,1) in Phasen ohne Soft-Knee — §III (copilot-instructions.md)
        if "backend/core/phases" in filepath.replace("\\", "/") and re.search(
            r"np\.clip\([^)]*(?:-1\.0?,\s*1\.0?|1\.0?,\s*-1\.0?)", stripped
        ):
            if not re.search(r"soft_knee|soft-knee|knee|hanning|hann", source, re.IGNORECASE):
                issues.append(
                    f"{filepath}:{i}: H06 Hard-Clamp (-1,1) ohne Soft-Knee "
                    f"(→ hörbare Clipping-Artefakte, §III). "
                    f"FIX: Soft-Knee (6 dB, 200 ms Hanning)"
                )

        # H07: Silent-Except mit neutralem Return ohne logger — §V6 (copilot-instructions.md) Silent-Failure-Verbot
        if re.search(r"except\s+Exception", stripped) or stripped == "except Exception:":
            _window = "\n".join(_ln for _ln in lines[i : min(i + 3, len(lines))] if not _ln.strip().startswith("#"))
            # Lokale Log-Wrapper erfüllen §V6 (copilot-instructions.md) semantisch:
            # shellac_mono_strategy._audit_log routet 'error' → logger.error. Die reine
            # Textsuche nach `logger.` sah das nicht (Produktionsbefund 2026-10-05, H07-Fehlalarm).
            _wrapper_names = {
                _m.group(1)
                for _m in re.finditer(r"(?m)^def\s+([A-Za-z_]\w*)\s*\(", source)
                if re.search(r"logger\.(?:warning|error|critical)", source[_m.end() : _m.end() + 600])
            }
            if (
                re.search(r"return\s+[01]\.\d*", _window)
                and not re.search(r"logger\.|log\.warning", _window)
                and not any(f"{_name}(" in _window for _name in _wrapper_names)
            ):
                issues.append(
                    f"{filepath}:{i}: H07 Silent-Except → neutraler Return ohne logger.warning "
                    f"(→ ML→DSP-Fallback unsichtbar, §V6 (copilot-instructions.md))"
                )

    return issues


# ── §v10.115 Continuous Analysis: Dynamische Pattern-Erkennung ────────────────


_H_PRIORITY = {"H02": 0, "H01": 1, "H06": 2, "H03": 3, "H05": 4, "H04": 5, "H07": 6}
_H_TITLES = {
    "H01": "H01 — Dither-Pflicht (Int16-Export)",
    "H02": "H02 — PGHI statt Griffin-Lim",
    "H03": "H03 — Zero-Phase-Filter (sosfiltfilt)",
    "H04": "H04 — Determinismus (time.time in Entscheidungen)",
    "H05": "H05 — Resample ohne Längen-Guard",
    "H06": "H06 — Soft-Knee statt Hard-Clamp",
    "H07": "H07 — Silent-Failure-Verbot (logger im Except)",
}


def _write_hoerordnung_todo(all_issues: list[str], todo_path: str) -> None:
    """Persistiert H-Serie-Funde als priorisierte, abarbeitbare To-Do-Liste.

    Format: Markdown-Checklisten mit stabilen IDs (H02-001, …), sortiert nach
    Schwere (H02 > H01 > H06 > H03 > H05 > H04 > H07). Beim Neuschreiben
    werden bereits abgehakte Einträge ([x]) übernommen — erledigte Punkte
    gehen über Sessions nicht verloren.
    """
    import hashlib as _hl
    import re as _re
    from datetime import date as _date

    _h_issues = [s for s in all_issues if _re.search(r":\s*(H0[1-7]) ", s)]
    if not _h_issues:
        _h_issues = []
    # Bereits erledigte IDs aus der bestehenden Datei laden
    _done: set[str] = set()
    try:
        with open(todo_path, encoding="utf-8") as f:
            _old = f.read()
        _done = set(_re.findall(r"^- \[x\] (H0[1-7]-[0-9a-f]{6}) ", _old, _re.MULTILINE))
    except (OSError, UnicodeDecodeError):
        # Expected: erste Ausführung oder Datei unlesbar → frische To-do-Liste.
        pass

    # Gruppieren nach ID-Präfix + Datei (stabile ID pro Fundstelle)
    _entries: list[tuple[int, str, str, str]] = []
    _seen: dict[str, int] = {}
    for _issue in sorted(_h_issues):
        _m = _re.match(r"^(.+?):(\d+): (H0[1-7]) (.*)$", _issue)
        if not _m:
            continue
        _path, _line, _hid, _desc = _m.group(1), int(_m.group(2)), _m.group(3), _m.group(4)
        _key = f"{_path}:{_line}:{_hid}"
        if _key in _seen:
            continue
        _seen[_key] = 1
        # Stabile ID pro Fundstelle (Hash aus Pfad:Zeile:Muster) — unabhängig
        # von der Gesamtmenge, damit der Status-Erhalt über Sessions hält.
        _eid = f"{_hid}-{_hl.sha256(_key.encode('utf-8')).hexdigest()[:6]}"
        _prio = _H_PRIORITY.get(_hid, 9)
        _entries.append((_prio, _eid, f"{_path}:{_line}", _desc))

    _entries.sort(key=lambda e: (e[0], e[2]))

    _out_lines = [
        "# Hörordnungs-Schwachstellen — To-Do für die nächste Programmier-Session",
        "",
        f"> Generiert von `scan_anti_patterns.py --write-todo` ({_date.today().isoformat()}).",
        "> Abgehakte Einträge bleiben erledigt; neue Funde werden angehängt.",
        "> Priorität: H02 > H01 > H06 > H03 > H05 > H04 > H07.",
        "",
    ]
    _prev_prio: int | None = None
    for _prio, _eid, _loc, _desc in _entries:
        _hid = _eid[:3]
        if _prev_prio != _prio:
            _out_lines.append(f"## {_H_TITLES.get(_hid, _hid)}")
            _prev_prio = _prio
        _mark = "x" if _eid in _done else " "
        _out_lines.append(f"- [{_mark}] {_eid} | {_loc} | {_desc}")
    _out_lines.append("")

    _dest = Path(todo_path)
    _dest.parent.mkdir(parents=True, exist_ok=True)
    _dest.write_text("\n".join(_out_lines), encoding="utf-8")
    logger.info(
        "§V01 Hörordnungs-To-Do geschrieben: %s (%d Einträge, %d bereits erledigt)", _dest, len(_entries), len(_done)
    )


def _load_discovered_patterns() -> list[str]:
    """Lädt vom Pattern-Miner entdeckte Patterns aus logs/discovered_patterns.json."""
    import json

    if not _PATTERN_FEED_PATH.exists():
        return []
    try:
        with open(_PATTERN_FEED_PATH) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []

    issues = []
    for pattern in data.get("patterns", []):
        if pattern.get("status") != "active":
            continue
        regex = pattern.get("regex")
        if not regex:
            continue
        message = pattern.get("message", "P7 Dynamisch entdecktes Anti-Pattern")
        for root in data.get("scan_roots", ["backend/core"]):
            repo_root = _PATTERN_FEED_PATH.parents[1]
            scan_dir = repo_root / root
            if not scan_dir.exists():
                continue
            for dirpath, _dirnames, filenames in os.walk(scan_dir):
                for fn in filenames:
                    if not fn.endswith(".py"):
                        continue
                    fp = os.path.join(dirpath, fn)
                    try:
                        with open(fp, encoding="utf-8") as fh:
                            src = fh.read()
                    except (UnicodeDecodeError, IsADirectoryError):
                        continue
                    for i, line in enumerate(src.split("\n"), 1):
                        s = line.strip()
                        if s.startswith("#"):
                            continue
                        if re.search(regex, s):
                            issues.append(f"{fp}:{i}: {message}")
    return issues


# ── Main ───────────────────────────────────────────────────────────────────


def main() -> int:
    """Scannt alle Python-Dateien auf bekannte Bug-Patterns.

    §v10.114: Scanner auf alle Layer ausgeweitet (backend/core, plugins,
    Aurik10, denker, scripts).

    --write-todo: persistiert die H-Serie-Funde als priorisierte To-Do-Liste
    (reports/hoerordnung_schwachstellen.md) für die nächste Programmier-Session.
    Status-Erhalt: bereits abgehakte Einträge bleiben erledigt.
    """
    import argparse

    _parser = argparse.ArgumentParser()
    _parser.add_argument("--write-todo", action="store_true", help="H-Serie als To-Do-Liste persistieren")
    _parser.add_argument(
        "--todo-path",
        default=str((Path(__file__).resolve().parents[3]) / "reports" / "hoerordnung_schwachstellen.md"),
        help="Zielpfad der To-Do-Liste",
    )
    _args, _ = _parser.parse_known_args()

    root = Path(__file__).resolve().parents[3]  # .agents/skills/bug-prevention/ → repo root

    SCAN_ROOTS = [
        root / "backend" / "core",
        root / "plugins",
        root / "Aurik10",
        root / "denker",
        root / "scripts",
    ]

    all_issues: list[str] = []
    files_scanned = 0

    for scan_root in SCAN_ROOTS:
        if not scan_root.exists():
            continue

        for dirpath, dirnames, filenames in os.walk(scan_root):
            # Filtere Verzeichnisse
            dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]

            for filename in filenames:
                if not filename.endswith(".py"):
                    continue
                if filename in EXCLUDE_FILES:
                    continue

                filepath = os.path.join(dirpath, filename)
                files_scanned += 1

                try:
                    with open(filepath, encoding="utf-8") as f:
                        source = f.read()
                except (UnicodeDecodeError, IsADirectoryError):
                    continue

                # Alle Checks
                all_issues.extend(check_shape_anti_pattern(filepath, source))
                all_issues.extend(check_filtfilt_without_guard(filepath, source))
                all_issues.extend(check_stft_without_clamp(filepath, source))
                all_issues.extend(check_os_without_import(filepath, source))
                all_issues.extend(check_asarray_tuple(filepath, source))
                all_issues.extend(check_enum_as_dict_key(filepath, source))
                all_issues.extend(check_hoerordnung_export_patterns(filepath, source))

    # §v10.115: Lade dynamisch entdeckte Patterns aus Exception-Forensik
    discovered = _load_discovered_patterns()
    all_issues.extend(discovered)

    # ── H-Serie-To-Do persistieren (Session-Übergabe) ──
    if _args.write_todo:
        _write_hoerordnung_todo(all_issues, _args.todo_path)

    # Ausgabe (§V01: Logger statt print)
    if all_issues:
        logger.warning(
            "§V01 Aurik SOTA Bug-Scan: %d potentielle Bugs gefunden (%d Dateien gescannt)",
            len(all_issues),
            files_scanned,
        )
        _by_class: dict[str, int] = {}
        for issue in all_issues:
            _m = re.search(r"\b(H\d{2})\b", issue)
            if _m:
                _by_class[_m.group(1)] = _by_class.get(_m.group(1), 0) + 1
        _skip = f" | H03-Analyse-Skip={len(_H03_ANALYSIS_SKIPPED)}" if _H03_ANALYSIS_SKIPPED else ""
        _skip += f" | H04-TTL-Skip={len(_H04_TTL_SKIPPED)}" if _H04_TTL_SKIPPED else ""
        logger.warning(
            "§V01 Klassen: %s%s",
            ", ".join(f"{k}={v}" for k, v in sorted(_by_class.items())) or "—",
            _skip,
        )
        _H03_ANALYSIS_SKIPPED.clear()
        _H04_TTL_SKIPPED.clear()
        for issue in sorted(all_issues):
            logger.info("  %s", issue)

        if MIN_SEVERITY == "error":
            logger.error(
                "§V01 %d Fehler — Commit blockiert. Behebe Anti-Patterns oder füge begründete Ausnahmen hinzu.",
                len(all_issues),
            )
            return 1
        else:
            logger.warning("§V01 %d Warnungen — bitte vor Commit prüfen.", len(all_issues))
            return 0
    else:
        logger.info("§V01 Aurik SOTA Bug-Scan: Keine Anti-Patterns gefunden (%d Dateien gescannt)", files_scanned)
        return 0


if __name__ == "__main__":
    sys.exit(main())
