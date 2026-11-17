"""hearing_gates_summary — GUI-Aufbereitung der Hör-Gates (T1, headless-sicher).

Reine Funktionen über result.metadata (kein PySide-Import → unit-testbar).
Konsumierte Keys (aus unified_restorer_v3.py):
  * metadata["audibility_gate"] = {gate_passed, threshold, n_audible_unmasked,
    n_masked, n_resolved, improvable_types, …}
  * metadata["einladungs_gate_passed"/"_corrected"/"_sharpness_jump"/…]
  * metadata["wohlklang_ordnung"] = {status, violated_goals, detail, violations}
  * metadata["vocal_drive_*"] (vocal_drive_telemetry-Felder)
"""

from __future__ import annotations

from typing import Any

from Aurik10.i18n import t


def _ag(meta: dict) -> dict:
    v = meta.get("audibility_gate") or {}
    return v if isinstance(v, dict) else {}


def _wo(meta: dict) -> dict:
    v = meta.get("wohlklang_ordnung") or {}
    return v if isinstance(v, dict) else {}


def _inv(meta: dict) -> dict:
    """Hörordnungs-Einladungs-Gate (Ebene 4) — _meta["inviting_gate_song_level"]."""
    v = meta.get("inviting_gate_song_level") or {}
    return v if isinstance(v, dict) else {}


def _num(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def hearing_gate_status(meta: dict) -> str:
    """green | yellow | red — Ampel über alle Hör-Gates."""
    ag = _ag(meta)
    if ag.get("gate_passed") is False:
        return "red"
    if _wo(meta).get("status") == "VIOLATION":
        return "red"
    if meta.get("einladungs_gate_passed") is False and not meta.get("einladungs_gate_corrected"):
        return "red"
    if meta.get("vocal_drive_hard_revert") is True:
        return "red"
    # gelb: korrigiert, maskiert oder Nachbehandlung gequeued
    if (
        meta.get("einladungs_gate_corrected") is True
        or _num(ag.get("n_masked")) > 0
        or (bool(ag.get("improvable_types")) and meta.get("m1b_retry_applied") is not True)
        or meta.get("vocal_drive_blend", 1.0) < 1.0
        or bool(_inv(meta).get("fluctuation_regression"))
        or bool(_inv(meta).get("loudness_jump_regression"))
    ):
        return "yellow"
    return "green"


def hearing_gates_line(meta: dict) -> str:
    """Kurzzeile für den Qualitäts-Score-Text."""
    icon = {"green": "\U0001f7e2", "yellow": "\U0001f7e1", "red": "\U0001f534"}[hearing_gate_status(meta)]
    return t("hearing.gates_line", icon=icon, status=hearing_gate_status(meta).upper())


def hearing_gates_details(meta: dict) -> list[str]:
    """Detailzeilen für den Ergebnis-Banner."""
    out: list[str] = []
    ag = _ag(meta)
    wo = _wo(meta)
    if ag:
        out.append(
            t("hearing.audibility_passed") if ag.get("gate_passed") is not False else t("hearing.audibility_residual")
        )
        if _num(ag.get("n_audible_unmasked")) > 0:
            out.append(t("hearing.audible_n", n=int(_num(ag.get("n_audible_unmasked")))))
        if ag.get("improvable_types"):
            if meta.get("m1b_retry_applied") is True:
                # §P1-7: Backend hat die Stufe-2-Nachbehandlung intern ausgeführt
                out.append(t("hearing.stufe2_executed", types=", ".join(str(x) for x in ag["improvable_types"][:6])))
            else:
                out.append(t("hearing.stufe2_queue", types=", ".join(str(x) for x in ag["improvable_types"][:6])))
        if _num(ag.get("n_masked")) > 0:
            out.append(t("hearing.masked_n", n=int(_num(ag.get("n_masked")))))
    if wo.get("status") == "VIOLATION":
        goals = wo.get("violated_goals") or []
        names = ", ".join(str(g) for g in goals)
        out.append(t("hearing.wohlklang_ordnung_violated", names=names))
    elif wo.get("status") == "PASS":
        out.append(t("hearing.wohlklang_ordnung_ok"))
    eg_passed = meta.get("einladungs_gate_passed")
    if eg_passed is False:
        if meta.get("einladungs_gate_corrected"):
            out.append(t("hearing.wohlklang_corrected"))
        else:
            out.append(t("hearing.wohlklang_violated"))
    elif eg_passed is True:
        out.append(t("hearing.wohlklang_ok"))
    if meta.get("vocal_drive_hard_revert") is True:
        out.append(t("hearing.vocal_revert"))
    elif meta.get("vocal_drive_blend", 1.0) < 1.0:
        out.append(t("hearing.vocal_blend", blend=f"{_num(meta.get('vocal_drive_blend')):.2f}"))
    inv = _inv(meta)
    if inv:
        if inv.get("fluctuation_regression"):
            out.append(
                t(
                    "hearing.invite_pumping",
                    max_vacil=f"{_num(inv.get('max_vacil')):.2f}",
                    max_vacil_input=f"{_num(inv.get('max_vacil_input')):.2f}",
                )
            )
        if inv.get("loudness_jump_regression"):
            out.append(
                t(
                    "hearing.invite_loudness_jump",
                    loudness=f"{_num(inv.get('loudness_jump_max')):.2f}",
                    loudness_input=f"{_num(inv.get('loudness_jump_max_input')):.2f}",
                )
            )
        if not inv.get("fluctuation_regression") and not inv.get("loudness_jump_regression"):
            if inv.get("passed") is False:
                out.append(
                    t(
                        "hearing.invite_advisory",
                        asper=f"{_num(inv.get('max_asper_in_voice')):.2f}",
                        sharpness=f"{_num(inv.get('sharpness_jump_max')):.2f}",
                    )
                )
            else:
                out.append(t("hearing.invite_ok"))
    return out or [t("hearing.no_metadata")]


def apply_resolved_defects(counts: dict[str, int], resolved: list[str] | None) -> tuple[dict[str, int], int, list[str]]:
    """Defekt-Chip-Subtraktion (Echtzeit, pure Logik).

    Args:
        counts: Chip-Zähler je Defekttyp (aktuelle Restanzahl).
        resolved: Defekttypen, die seit dem letzten Callback behoben wurden
            (Engine-Payload live_metrics["resolved"]).

    Returns:
        (neue Zähler, verbleibende Gesamtzahl, Liste der nun vollständig
        behobenen Typen) — nie negativ, unbekannte Typen werden ignoriert.
    """
    out = dict(counts or {})
    done: list[str] = []
    for dt in resolved or []:
        key = str(dt)
        if key not in out or out[key] <= 0:
            continue
        out[key] = out[key] - 1
        if out[key] == 0:
            done.append(key)
    return out, int(sum(out.values())), done


def apply_resolved_event_counts(init_counts: dict[str, int], resolved_counts: dict[str, int] | None) -> dict[str, int]:
    """§v10.802 GUI-Sync: Echte Instanzen-Subtraktion für den Echtzeit-Defektzähler.

    Aus den initialen Event-Zählungen je Defekttyp und den kumulierten
    Behebungs-Zählungen des Backends (live_metrics["defect_progress"]
    ["resolved_counts"]) die verbleibenden Zählungen berechnen — exakt,
    nie negativ, Schlüssel case-insensitiv normalisiert (UPPERCASE).

    Args:
        init_counts:      {DefectType: initiale Anzahl} aus dem Scan-Payload.
        resolved_counts:  {DefectType: kumulierte Behebungen} aus dem Backend.

    Returns:
        {DefectType: verbleibende Anzahl} — fehlende Typen bleiben unverändert.
    """
    remaining = dict(init_counts or {})
    resolved = resolved_counts or {}
    if not resolved:
        return remaining
    for key, val in remaining.items():
        k = str(key).upper()
        done = 0
        for rk, rv in resolved.items():
            if str(rk).upper() == k:
                try:
                    done = int(rv)
                except (TypeError, ValueError):
                    done = 1 if rv else 0
                break
        if done > 0:
            remaining[key] = max(0, int(val) - done)
    return remaining
