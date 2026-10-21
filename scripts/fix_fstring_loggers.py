"""fix_fstring_loggers.py
=========================
Converts f-string logger calls to %-format (lazy evaluation, per coding standard).

    logger.info(f"text {var}")  →  logger.info("text %s", var)
    logger.debug(f"val={x:.2f}")  →  logger.debug("val=%.2f", x)

Rules:
- Only single-line calls are transformed (multi-line: skipped, too risky).
- Each {expr} is one %-argument.
- Format specifiers (.2f, d, 05d, etc.) are preserved; !r/!s/!a converted to %r/%s/%s.
- Escaped {{ }} become literal { }.
- Lines with nested f-string calls or complex expressions are skipped.
- Dry-run by default; pass --apply to write changes.

Usage:
    python scripts/fix_fstring_loggers.py [--apply] [path ...]

    Default path: backend/ dsp/ plugins/ denker/
"""

import argparse
import logging
import re
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# logger.LEVEL(f"...", ...) — capture the full argument list as one group
_LOG_RE = re.compile(
    r"""(logger\s*\.\s*(?:debug|info|warning|error|critical|exception)\s*\()""" r"""(f['"])(.*?['"])(\s*\))""",
    re.DOTALL,
)

# One {expr} or {expr:fmt} or {expr!conv} or {expr!conv:fmt}
_FEXPR_RE = re.compile(
    r"""\{"""
    r"""([^{}!:]+?)"""  # the expression (no nested braces)
    r"""(?:!([rsa]))?"""  # optional conversion
    r"""(?::([^{}]*))?"""  # optional format spec
    r"""\}""",
)


def _fspec_to_percent(spec: str, conv: str | None) -> str:
    """Convert a Python format-spec to a %-format placeholder."""
    if conv == "r":
        return "%r"
    if conv in ("s", "a", None) and not spec:
        return "%s"
    if not spec:
        return "%s"
    # Simple numeric specs: d, f, .2f, 05d, +.3e, etc.
    # We only handle the common subset; exotic specs fall back to %s.
    # Remove fill/align (not supported by %-format in most cases).
    _numeric = re.fullmatch(r"""([+-]?)(\d*)(\.(\d+))?([dioxXeEfFgG%])""", spec.strip())
    if _numeric:
        sign, width, _prec_full, prec, typ = _numeric.groups()
        parts = "%" + sign
        if width:
            parts += width
        if prec is not None:
            parts += "." + prec
        parts += typ
        return parts
    # Integer zero-padded: 05d
    _zero = re.fullmatch(r"""0(\d+)([dioxX])""", spec.strip())
    if _zero:
        return f"%0{_zero.group(1)}{_zero.group(2)}"
    # String width: 10s or <10s (left-align → %s, truncation ignored)
    if spec.strip().endswith("s"):
        return "%s"
    return "%s"


def _transform_fstring(fstr: str) -> tuple[str, list[str]] | None:
    """
    Convert an f-string body (without surrounding quotes) to a
    (format_string, [args]) pair.

    Returns None if transformation is not safe (nested braces, etc.).
    """
    # Reject: nested braces inside expressions (dict literals, nested f-strings)
    depth = 0
    i = 0
    positions: list[tuple[int, int]] = []  # (start, end) of each {} group
    while i < len(fstr):
        if fstr[i] == "{":
            if i + 1 < len(fstr) and fstr[i + 1] == "{":
                i += 2  # escaped {{ → literal {
                continue
            depth += 1
            if depth > 1:
                return None  # nested braces — too complex
            start = i
        elif fstr[i] == "}":
            if i + 1 < len(fstr) and fstr[i + 1] == "}":
                i += 2  # escaped }} → literal }
                continue
            depth -= 1
            if depth == 0 and "start" in dir():
                positions.append((start, i + 1))
        i += 1

    fmt_parts: list[str] = []
    args: list[str] = []
    prev = 0

    for start, end in positions:
        # Literal text before this brace group
        literal = fstr[prev:start]
        # Unescape {{ → { and }} → }
        literal = literal.replace("{{", "{").replace("}}", "}")
        # Escape % in literal text (avoid treating literal % as format specifier)
        literal = literal.replace("%", "%%")
        fmt_parts.append(literal)

        inner = fstr[start + 1 : end - 1]
        # Parse inner: expr [!conv] [:spec]
        m = re.fullmatch(r"""([^!:{}]+?)(?:!([rsa]))?(?::([^{}]*))?""", inner.strip())
        if m is None:
            return None  # complex inner expression
        expr = m.group(1).strip()
        conv = m.group(2)
        spec = m.group(3) or ""

        # Reject expressions with call chains that contain quotes (could be nested f-strings)
        if "'" in expr or '"' in expr:
            return None

        placeholder = _fspec_to_percent(spec, conv)
        fmt_parts.append(placeholder)
        args.append(expr)
        prev = end

    # Trailing literal
    trailing = fstr[prev:]
    trailing = trailing.replace("{{", "{").replace("}}", "}")
    trailing = trailing.replace("%", "%%")
    fmt_parts.append(trailing)

    return "".join(fmt_parts), args


def _process_line(line: str) -> str | None:
    """
    Transform a single source line.  Returns the transformed line, or None
    if no change was made / transformation not safe.
    """
    # Quick pre-filter: must contain f" or f' after logger.
    if not re.search(r"""logger\s*\.\s*\w+\s*\(f['"]""", line):
        return None
    # Skip multi-line (continuation backslash or unbalanced parens handled elsewhere)
    stripped = line.rstrip("\n").rstrip()

    # Find logger call with f-string first argument
    m = re.search(
        r"""(logger\s*\.\s*(?:debug|info|warning|error|critical|exception)\s*\()(f)(["'])(.*?)\3((?:\s*,\s*.+?)?\s*\))$""",
        stripped,
    )
    if m is None:
        return None

    prefix = m.group(1)  # "logger.info("
    # group 2: "f"
    quote = m.group(3)  # " or '
    fstr_body = m.group(4)  # content of the f-string
    suffix = m.group(5)  # possible ", extra, args)" or ")"

    result = _transform_fstring(fstr_body)
    if result is None:
        return None

    fmt_str, new_args = result

    # Re-quote fmt_str; prefer double-quotes unless original was single-quote
    q = '"'
    if quote == "'" and '"' not in fmt_str:
        q = "'"
    # If fmt_str itself would need quote escaping, use single
    if q == '"' and '"' in fmt_str:
        if "'" not in fmt_str:
            q = "'"
        else:
            # Cannot safely re-quote without escaping; skip
            return None

    # Reassemble suffix args (anything AFTER the f-string)
    # suffix could be ");" or ", extra_arg)" etc.
    inner_suffix = suffix.strip()
    if inner_suffix.startswith(","):
        extra_args = inner_suffix[1:].rstrip(")")
    else:
        extra_args = ""

    all_args = new_args + ([a.strip() for a in extra_args.split(",") if a.strip()] if extra_args else [])

    if all_args:
        args_str = ", ".join(all_args)
        new_call = f"{prefix}{q}{fmt_str}{q}, {args_str})"
    else:
        new_call = f"{prefix}{q}{fmt_str}{q})"

    # Preserve original indentation
    indent = len(line) - len(line.lstrip())
    return line[:indent] + new_call + "\n"


def _process_file(path: Path, apply: bool) -> int:
    """Process one file.  Returns number of lines changed."""
    try:
        original = path.read_text(encoding="utf-8")
    except Exception:
        logger.warning("fix_fstring_loggers.py::_process_file fallback", exc_info=True)
        return 0

    lines = original.splitlines(keepends=True)
    changed = 0
    new_lines: list[str] = []

    for i, line in enumerate(lines):
        transformed = _process_line(line)
        if transformed is not None and transformed != line:
            new_lines.append(transformed)
            changed += 1
        else:
            new_lines.append(line)

    if changed and apply:
        path.write_text("".join(new_lines), encoding="utf-8")

    return changed


def _collect_files(roots: list[Path]) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if root.is_file() and root.suffix == ".py":
            files.append(root)
        elif root.is_dir():
            for p in sorted(root.rglob("*.py")):
                if "__pycache__" not in p.parts:
                    files.append(p)
    return files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Write changes (default: dry-run)")
    parser.add_argument(
        "paths",
        nargs="*",
        default=["backend", "dsp", "plugins", "denker"],
        help="Directories or files to process",
    )
    args = parser.parse_args(argv)

    base = Path(__file__).parent.parent
    roots = [base / p for p in args.paths]
    files = _collect_files(roots)

    total_files = 0
    total_lines = 0

    for f in files:
        n = _process_file(f, apply=args.apply)
        if n:
            total_files += 1
            total_lines += n
            rel = f.relative_to(base)
            print(f"  {'CHANGED' if args.apply else 'WOULD CHANGE'} {rel}: {n} lines")

    action = "changed" if args.apply else "would change"
    print(f"\nTotal: {total_lines} lines {action} in {total_files} files.")
    if not args.apply:
        print("Run with --apply to write changes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
