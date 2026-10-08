"""D-K3-68: §G10-Garantiekette im Chunked-Pfad — Song-Audibility-Nachzug.

Befund (Lauf 10.12.17, Chunk-0-Tail): `§Hörbarkeits-Gate UNGEPRÜFT … kein
§B2-Post-Scan-Ergebnis (0 Typ-Einträge) — gate_passed=True ist damit KEIN
Nachweis „Residuum unter der Maskierungsschwelle"`. Root-Cause: Im Chunked-Pfad
überspringen Nicht-letzte-Chunks den §B2-Post-Scan (§P0-1 (d)) → ihr
`audibility_gate`-Report ist `scan_missing`; `_restore_chunked` erbt aber genau
diesen Chunk-0-Report nach oben (`_meta = dict(_first_result.metadata)`) — es
gab keinen song-globalen Nachzug. Damit konnte die §G10-Qualitäts-Zusage
(gate_verified) im Export nie vergeben werden.

Kontrakte:
1. Wirkungs-Semantik: `post_scan_ran=True` ⇒ `evaluated`/`no_residual_defects`
   + `gate_verified=True`; `post_scan_ran=False` ⇒ `scan_missing` +
   `gate_verified=False` (fail-closed).
2. Source-Pin: `_restore_chunked` enthält den §G10-Song-Audibility-Nachzug.
3. Source-Pin: Der Nachzug schreibt in `_first_result.metadata` (damit der
   spätere `_meta`-Erb-Bau den Song-Stand trägt) UND setzt das §G10-Feld
   `defect_residual_audibility`.
4. Source-Pin: `post_scan_ran` wird aus der Datenlage abgeleitet (`bool(_song_aud_data)`),
   nicht hart auf True gesetzt (fail-closed bei fehlendem Scan).
"""

from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "backend" / "core" / "unified_restorer_v3.py"


def _src() -> str:
    return SRC.read_text(encoding="utf-8")


def test_semantics_gate_verified_requires_scan() -> None:
    """Kontrakt 1: Wirkungs-Semantik der Kette (nicht nur Prosa)."""
    from backend.core.defect_audibility_gate import evaluate_defect_audibility

    rep_ok = evaluate_defect_audibility({}, material_key="vinyl", post_scan_ran=True)
    assert rep_ok.evidence_state == "no_residual_defects", rep_ok.evidence_state
    assert rep_ok.gate_passed is True
    assert rep_ok.gate_verified is True

    rep_missing = evaluate_defect_audibility({}, material_key="vinyl", post_scan_ran=False)
    assert rep_missing.evidence_state == "scan_missing"
    assert rep_missing.gate_verified is False


def test_song_audibility_after_assembly_present() -> None:
    """Kontrakt 2: Der Nachzug existiert im Assembly-Pfad."""
    src = _src()
    assert "§G10 (copilot-instructions.md) Song-Audibility nach Assembly" in src
    assert "D-K3-68" in src


def test_song_audibility_writes_into_first_result_metadata() -> None:
    """Kontrakt 3: Erb-Träger + §G10-Feldname."""
    src = _src()
    assert '_fd_meta["audibility_gate"] = _song_aud_meta' in src
    assert '_fd_meta["defect_residual_audibility"] = _song_aud_meta' in src


def test_post_scan_ran_derived_fail_closed() -> None:
    """Kontrakt 4: post_scan_ran aus Datenlage, nicht hart True."""
    src = _src()
    assert "post_scan_ran=bool(_song_aud_data)" in src
    assert "post_scan_ran=True,\n" not in src
