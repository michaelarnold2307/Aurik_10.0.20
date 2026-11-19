"""
backend/core/restoration_report.py — Restoration-Report HTML (§v10.9)
======================================================================

Erzeugt einen detaillierten HTML-Restaurierungsbericht für Archivare/Toningenieure.
Enthält: Era, Material, alle Phasen mit Stärke/Wet-Dry, Defekte vorher/nachher.

Usage:
    from backend.core.restoration_report import generate_report
    html = generate_report(result_dict, output_path="restoration_report.html")
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_REPORT_TEMPLATE = """<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="UTF-8">
<title>Aurik Restoration Report</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 900px; margin: 40px auto; padding: 20px; background: #0a0e1a; color: #d0dcff; }}
  h1 {{ color: #667eea; border-bottom: 2px solid rgba(102,126,234,0.3); padding-bottom: 8px; }}
  h2 {{ color: #82B89A; margin-top: 30px; }}
  .meta {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin: 16px 0; }}
  .meta dt {{ color: #8898BB; font-size: 0.85em; }}
  .meta dd {{ margin: 0 0 8px 0; font-weight: 600; }}
  table {{ width: 100%; border-collapse: collapse; margin: 12px 0; }}
  th {{ background: rgba(102,126,234,0.15); padding: 8px 12px; text-align: left; font-size: 0.85em; color: #8898BB; }}
  td {{ padding: 6px 12px; border-bottom: 1px solid rgba(102,126,234,0.1); font-size: 0.9em; }}
  .good {{ color: #82B89A; }}
  .warn {{ color: #C8A84B; }}
  .bad {{ color: #B87A7A; }}
  .footer {{ margin-top: 40px; font-size: 0.75em; color: #555; text-align: center; }}
</style>
</head>
<body>
<h1>🎵 Aurik Restoration Report</h1>
<p>Erstellt: {timestamp}</p>

<h2>📋 Material &amp; Era</h2>
<dl class="meta">
  <dt>Datei</dt><dd>{file_name}</dd>
  <dt>Dauer</dt><dd>{duration}</dd>
  <dt>Trägermedium</dt><dd>{material}</dd>
  <dt>Ära</dt><dd>{era}</dd>
  <dt>Genre</dt><dd>{genre}</dd>
  <dt>Modus</dt><dd>{mode}</dd>
</dl>

<h2>📊 Qualität</h2>
<dl class="meta">
  <dt>Qualität vorher</dt><dd class="warn">{quality_before}%</dd>
  <dt>Qualität nachher</dt><dd class="good">{quality_after}%</dd>
  <dt>Artifact Freedom</dt><dd>{artifact_freedom}</dd>
  <dt>Hörgenuss (Joy)</dt><dd>{joy_pct}%</dd>
  <dt>Hörermüdung (Fatigue)</dt><dd>{fatigue_pct}%</dd>
</dl>

<h2>🔧 Phasen</h2>
<table>
<tr><th>Phase</th><th>Stärke</th><th>Wet/Dry</th><th>Zeit (s)</th><th>Status</th></tr>
{phase_rows}
</table>

<h2>🐛 Defekte</h2>
<table>
<tr><th>Defekt</th><th>Severity vorher</th><th>Status</th></tr>
{defect_rows}
</table>

<div class="footer">
  Aurik Professional — Version 10.0.9<br>
  Report generiert am {timestamp}
</div>
</body>
</html>"""


def generate_report(
    result_data: dict[str, Any],
    output_path: str | None = None,
) -> str:
    """Generiert einen HTML-Restaurierungsbericht.

    Args:
        result_data: Dict mit Restaurierungs-Ergebnisdaten.
        output_path: Optionaler Pfad zum Speichern der HTML-Datei.

    Returns:
        HTML-String.
    """
    _ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Phasen-Zeilen
    _phase_rows = ""
    _phases = result_data.get("phases", [])
    if isinstance(_phases, list):
        for p in _phases:
            if not isinstance(p, dict):
                continue
            _name = p.get("name", "?")
            _strength = p.get("strength", 0)
            _wet_dry = p.get("wet_dry", 1.0)
            _time = p.get("time_s", 0)
            _ok = p.get("success", True)
            _status = '<span class="good">✅</span>' if _ok else '<span class="bad">❌</span>'
            _phase_rows += f"<tr><td>{_name}</td><td>{_strength:.2f}</td><td>{_wet_dry:.2f}</td><td>{_time:.1f}</td><td>{_status}</td></tr>\n"
    if not _phase_rows:
        _phase_rows = '<tr><td colspan="5"><em>Keine Phasen-Daten verfügbar</em></td></tr>'

    # Defekt-Zeilen
    _defect_rows = ""
    _defects = result_data.get("defects", {})
    if isinstance(_defects, dict):
        for _name, _sev in sorted(
            _defects.items(), key=lambda x: -float(x[1]) if isinstance(x[1], (int, float)) else 0
        ):
            _s = float(_sev) if isinstance(_sev, (int, float)) else 0.0
            _cls = "good" if _s < 0.3 else ("warn" if _s < 0.6 else "bad")
            _fixed = "✅ Behoben" if _s < 0.1 else ("⚠️ Teilweise" if _s < 0.4 else "❌ Nicht behoben")
            _defect_rows += f'<tr><td>{_name}</td><td class="{_cls}">{_s:.2f}</td><td>{_fixed}</td></tr>\n'
    if not _defect_rows:
        _defect_rows = '<tr><td colspan="3"><em>Keine Defekt-Daten</em></td></tr>'

    _joy = float(result_data.get("joy_index", 0.0) or 0.0)
    _fatigue = float(result_data.get("fatigue_index", 0.0) or 0.0)

    html = _REPORT_TEMPLATE.format(
        timestamp=_ts,
        file_name=result_data.get("file_name", "?"),
        duration=result_data.get("duration_seconds", 0),
        material=result_data.get("material_detected", "?"),
        era=result_data.get("era_detected", "?"),
        genre=result_data.get("genre", "?"),
        mode=result_data.get("mode", "?"),
        quality_before=int(result_data.get("quality_before", 50)),
        quality_after=int(result_data.get("quality_after", 85)),
        artifact_freedom=f"{result_data.get('artifact_freedom', 0.0):.3f}",
        joy_pct=int(_joy * 100),
        fatigue_pct=int(_fatigue * 100),
        phase_rows=_phase_rows,
        defect_rows=_defect_rows,
    )

    if output_path:
        try:
            Path(output_path).write_text(html, encoding="utf-8")
            logger.info("📄 Restoration Report gespeichert: %s", output_path)
        except Exception as exc:
            logger.warning("⚠️ Report speichern fehlgeschlagen: %s", exc)

    return html
