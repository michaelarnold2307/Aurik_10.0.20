"""Safe rendering for CSV files that may be opened in spreadsheet software."""

from __future__ import annotations

import csv
import io
from collections.abc import Iterable, Sequence


def _validate_row(row: Sequence[str]) -> None:
    for value in row:
        if value.lstrip().startswith(("=", "+", "-", "@")):
            raise ValueError("CSV export refused a formula-like cell; rename the source value and retry.")


def render_spreadsheet_safe_csv(headers: Sequence[object], rows: Iterable[Sequence[object]]) -> str:
    """Render fully quoted CSV and reject cells that could be evaluated as formulas."""
    normalized_headers = tuple(str(value) for value in headers)
    normalized_rows = [tuple(str(value) for value in row) for row in rows]

    _validate_row(normalized_headers)
    for row in normalized_rows:
        _validate_row(row)

    output = io.StringIO(newline="")
    writer = csv.writer(output, quoting=csv.QUOTE_ALL)
    writer.writerow(normalized_headers)
    writer.writerows(normalized_rows)
    return output.getvalue()
