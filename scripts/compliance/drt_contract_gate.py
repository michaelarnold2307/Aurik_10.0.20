"""scripts/compliance/drt_contract_gate.py — Pre-Commit-Gate für Spec 26 (DRT).

Fail-closed: erzwingt die Leitplanken des Defekt-Reparatur-Trainingsvertrags
strukturell, bevor ein Commit mit Reparatur-Trainer/Gate/DSP durchgeht.
Verhindert die vier dokumentierten Verlustrunden (Spec 26, „Warum diese Spec").

Checks:
  A) Trainer (scripts/train_witness_kopf.py):
     - Identitäts-Anker im Loss (DRT-2): `lam_id`, `l1_id`
     - Gate-Score-Auswahl mit beiden Armen (DRT-1/5): `score_h = ... lam_id`
     - Witness eingefroren (DRT-1): `requires_grad_(False)`
     - Mel-Parität (DRT-1/4): slaney + `10.0 * torch.log10(torch.clamp(`
     - Pre-flight-Kontrast (DRT-3): Kontrast-Prüfung vor dem Trainings-Loop
     - kein music_model_flags-Import (Flag-Konsum bleibt der Phase, DRT-6)
  B) Gate-Skript (scripts/verify_witness_kopf_gate.py):
     - Schwellen-Pins (DRT-5): Q_MIN == 0.5, NW_ANTEIL_MIN == 0.95
  C) Produktions-DSP (backend/core/dsp/witness_kopf_*.py):
     - Training-Parität (DRT-4): `np.pad(x, (0, _FENSTER - len(x)))`,
       `laenge_in`, `kandidat[:laenge_in]`
     - RMS_ZIEL als Import statt lokaler Definition (§V25 (copilot-instructions.md))
     - Never-worsen-Striktvergleich und Trigger-Fail-closed (DRT-6)

Exit 1 bei jedem Verstoß. Kein Mutator — nur Prüfung.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent

TRAINER = _ROOT / "scripts" / "train_witness_kopf.py"
GATE_SKRIPT = _ROOT / "scripts" / "verify_witness_kopf_gate.py"
DSP_GLOB = "backend/core/dsp/witness_kopf_*.py"


def _fehler(meldung: str, fehler: list[str]) -> None:
    fehler.append(meldung)


def _pruefe_trainer(fehler: list[str]) -> None:
    if not TRAINER.is_file():
        _fehler(f"{TRAINER.name} fehlt", fehler)
        return
    q = TRAINER.read_text(encoding="utf-8")
    # DRT-2: Identitäts-Anker
    if "lam_id" not in q or "l1_id" not in q:
        _fehler("DRT-2: Trainer ohne Identitäts-Anker (lam_id/l1_id) — Schutz-Arm fehlt im Loss", fehler)
    if "loss = l1 + lam * pen + lam_id * l1_id" not in q:
        _fehler("DRT-2: Loss ohne Schutz-Arm-Term (l1 + lam*pen + lam_id*l1_id)", fehler)
    # DRT-1/5: Gate-Score-Auswahl mit beiden Armen
    if "lam_id * l1_id_h" not in q:
        _fehler("DRT-5: Holdout-Score ohne Schutz-Arm (Modellauswahl bewertet das Gate nicht)", fehler)
    # DRT-1: Witness eingefroren
    if "requires_grad_(False)" not in q:
        _fehler("DRT-1: Witness-Parameter nicht eingefroren", fehler)
    # DRT-1/4: Mel-Parität (slaney + 10*log10 + amin)
    if 'norm="slaney"' not in q or "10.0 * torch.log10(torch.clamp(" not in q:
        _fehler("DRT-1/4: Mel-Repräsentation ohne slaney + 10·log10 + amin (Witness sättigt sonst)", fehler)
    # DRT-3: Pre-flight-Kontrast vor dem Trainings-Loop
    if "Kontrast" not in q:
        _fehler("DRT-3: Pre-flight-Kontrastprüfung fehlt", fehler)
    # DRT-6: kein Flag-Import im Trainer
    if re.search(r"from backend\.core\.music_model_flags import|import music_model_flags", q):
        _fehler("DRT-6: Trainer importiert music_model_flags (Flag-Konsum gehört in die Phase)", fehler)


def _pruefe_gate(fehler: list[str]) -> None:
    if not GATE_SKRIPT.is_file():
        _fehler(f"{GATE_SKRIPT.name} fehlt", fehler)
        return
    q = GATE_SKRIPT.read_text(encoding="utf-8")
    m_q = re.search(r"Q_MIN\s*=\s*([0-9.]+)", q)
    m_nw = re.search(r"NW_ANTEIL_MIN\s*=\s*([0-9.]+)", q)
    if not m_q or abs(float(m_q.group(1)) - 0.5) > 1e-9:
        _fehler("DRT-5: Q_MIN ≠ 0,5 (Schwellen-Senkung verboten)", fehler)
    if not m_nw or abs(float(m_nw.group(1)) - 0.95) > 1e-9:
        _fehler("DRT-5: NW_ANTEIL_MIN ≠ 0,95 (Schwellen-Senkung verboten)", fehler)


def _pruefe_dsp(fehler: list[str]) -> None:
    module = sorted((_ROOT / "backend" / "core" / "dsp").glob("witness_kopf_*.py"))
    if not module:
        _fehler("DRT: keine Produktions-DSP-Module gefunden", fehler)
        return
    for m in module:
        q = m.read_text(encoding="utf-8")
        # DRT-4: Training-Parität (0,1-s-Regionen-Pad) + Längen-Vertrag
        if "np.pad(x, (0, _FENSTER - len(x)))" not in q:
            _fehler(f"DRT-4: {m.name} ohne 1-s-Pad-Parität (0,1-s-Regionen ⇒ No-Op, Befund 2026-11-15)", fehler)
        if "laenge_in" not in q or "kandidat[:laenge_in]" not in q:
            _fehler(f"DRT-4: {m.name} ohne Längen-Schnitt (Phase erwartet Eingangs-Shape)", fehler)
        # §V25 (copilot-instructions.md): RMS_ZIEL als Import, keine lokale Definition
        if re.search(r"^_RMS_ZIEL\s*=\s*[0-9.]+", q, re.MULTILINE):
            _fehler(
                f"§V25 (copilot-instructions.md): {m.name} definiert _RMS_ZIEL lokal (Import aus Trainer Pflicht)",
                fehler,
            )
        # DRT-6: Trigger fail-closed + Never-worsen strikt
        if "_DEFAULT_THRESHOLD" not in q or "kandidat" not in q:
            _fehler(f"DRT-6: {m.name} ohne Trigger/Never-worsen-Konstruktion", fehler)


def _pruefe_drt_789(fehler: list[str]) -> None:
    """DRT-7/8/9 (2026-11-15-Verschärfung): Ganz-Titel-Sinn-Beweis + No-Op-Erkennung."""
    for skript in ("scripts/erstelle_schadens_proben.py", "scripts/bewerte_schadens_proben.py"):
        if not (_ROOT / skript).is_file():
            _fehler(f"DRT-8: {skript} fehlt (Ganz-Titel-Sinn-Beweis-Infrastruktur)", fehler)
    bewerte = _ROOT / "scripts" / "bewerte_schadens_proben.py"
    if bewerte.is_file():
        q = bewerte.read_text(encoding="utf-8")
        if "n_besser" not in q or "besser" not in q:
            _fehler("DRT-9: Bewertungs-Skript ohne besser/n_besser-No-Op-Erkennung", fehler)
        if "ueberrestauration" not in q or "JND_MOS" not in q:
            _fehler("DRT-10: Bewertungs-Skript ohne Überrestaurations-Erkennung (JND_MOS/ueberrestauration)", fehler)
    # DRT-7/8: der Ganz-Titel-Beleg muss existieren — ohne ihn ist keine Aktivierung erlaubt
    belege = list((_ROOT / "docs" / "reports" / "current").glob("*sinnbeweis*.json"))
    if not belege:
        _fehler("DRT-8: kein Sinn-Beweis-Beleg (Ganz-Titel-Ebene unbelegt — Aktivierung nicht erlaubt)", fehler)
    # DRT-7 (flächendeckend, 2026-11-17): auch BESTEHENDE Kandidaten brauchen ihr
    # Produktions-Semantik-Gate (Never-worsen vor Übernahme) — sonst bliebe die
    # Spec-Aussage „für ALLE Reparatur-Kandidaten“ Prosa.
    phase07 = _ROOT / "backend" / "core" / "phases" / "phase_07_declipper.py"
    if phase07.is_file() and "_harmonic_distortion_proxy" not in phase07.read_text(encoding="utf-8"):
        _fehler("DRT-7: Declip-Kandidaten ohne Harmonik-Never-worsen-Witness", fehler)
    phase09 = _ROOT / "backend" / "core" / "phases" / "phase_09_crackle_removal.py"
    if phase09.is_file() and "Spitzenwachstum" not in phase09.read_text(encoding="utf-8"):
        _fehler("DRT-7: de_crackle (VinylNet) ohne Spitzenwachstum-Never-worsen", fehler)
    hybrid = _ROOT / "backend" / "core" / "dsp" / "hybrid_denoise_fusion.py"
    if hybrid.is_file() and "never_worsen" not in hybrid.read_text(encoding="utf-8"):
        _fehler("DRT-7: Dritt-Denoiser-Fusion ohne Never-worsen", fehler)


def _pruefe_drt_131415(fehler: list[str]) -> None:
    """DRT-13/14/15 (2026-11-17): Teil-Rollback, Wirksamkeits-Gebot, B3-Komposition."""
    # DRT-13: Snapshot-Modul + Verdrahtung (Phase 07 → UV3-Verdikt)
    mod = _ROOT / "backend" / "core" / "dsp" / "reparatur_phasen_attribution.py"
    if not mod.is_file():
        _fehler("DRT-13: backend/core/dsp/reparatur_phasen_attribution.py fehlt (Snapshot/Teil-Rollback)", fehler)
    else:
        q = mod.read_text(encoding="utf-8")
        if "def phase_aus_variante" not in q or "def snapshot" not in q:
            _fehler("DRT-13: Modul ohne snapshot/phase_aus_variante (Teil-Rollback unmöglich)", fehler)
    phase07 = _ROOT / "backend" / "core" / "phases" / "phase_07_declipper.py"
    if phase07.is_file():
        q7 = phase07.read_text(encoding="utf-8")
        if "reparatur_phasen_attribution" not in q7:
            _fehler("DRT-13: Phase 07 ohne Reparatur-Snapshot (Attribution nicht verdrahtet)", fehler)
    uv3 = _ROOT / "backend" / "core" / "unified_restorer_v3.py"
    if uv3.is_file():
        qu = uv3.read_text(encoding="utf-8")
        if "drt10_teil_rollback" not in qu or "phase_aus_variante" not in qu:
            _fehler("DRT-13: UV3-Verdikt ohne Teil-Rollback-Pfad (M1-Fallback allein ist Normverstoß)", fehler)
    # DRT-14: Wirksamkeits-Bericht (Zeuge-Helper + UV3-Verdrahtung)
    zeuge = _ROOT / "backend" / "core" / "dsp" / "wohlklang_zeuge.py"
    if zeuge.is_file():
        qz = zeuge.read_text(encoding="utf-8")
        if "def reparatur_nachweis_bericht" not in qz:
            _fehler("DRT-14: wohlklang_zeuge ohne reparatur_nachweis_bericht", fehler)
        if "reparaturquote_median" not in qz:
            _fehler(
                "DRT-14: Bericht liest nicht das reale defekt_delta-Schema (aggregate/reparaturquote_median)", fehler
            )
    if uv3.is_file():
        if "reparatur_nachweis" not in uv3.read_text(encoding="utf-8"):
            _fehler("DRT-14: UV3 ohne reparatur_nachweis-Metadaten (Wirksamkeits-Gebot unbelegt)", fehler)
    # DRT-14-Coverage (2026-11-17): der Sinn-Beweis muss ≥ 15 Rezept-Klassen abdecken
    proben = _ROOT / "scripts" / "erstelle_schadens_proben.py"
    if proben.is_file():
        qp = proben.read_text(encoding="utf-8")
        for r in (
            "vinyl",
            "kassette",
            "clipping",
            "mp3_128",
            "hum",
            "wowflutter",
            "kompression",
            "kammfilter",
            "verzerrung",
            "clicks",
            "pitch_drift",
            "dropout",
            "raum",
            "lautheit",
            "amp_drift",
        ):
            if f'"{r}"' not in qp:
                _fehler(f"DRT-14-Coverage: Rezept {r!r} fehlt in erstelle_schadens_proben.py", fehler)
    # DRT-15: B3-Beschluss trägt kompositions_status
    b3 = _ROOT / "scripts" / "b3_stufenmessung.py"
    if b3.is_file():
        if "kompositions_status" not in b3.read_text(encoding="utf-8"):
            _fehler("DRT-15: B3-Beschluss ohne kompositions_status (Ablehnung ohne Kompositions-Beleg)", fehler)


def _pruefe_drt_12(fehler: list[str]) -> None:
    """DRT-12 (2026-11-19): identische Kontrolle und Kanalsatz-A/B."""
    skript = _ROOT / "scripts" / "messe_kanalsatz_ab.py"
    if not skript.is_file():
        _fehler("DRT-12: scripts/messe_kanalsatz_ab.py fehlt (Kanalsatz-A/B-Infrastruktur)", fehler)
        return
    q = skript.read_text(encoding="utf-8")
    if "KANALSAETZ_FLAGS" not in q or "_aktiviere_kanalsatz" not in q:
        _fehler("DRT-12: Kanalsatz-Aktivierung fehlt", fehler)
    if "flag_freigabe" in q:
        _fehler("DRT-12: A/B-Skript darf keine Flag-Freigabe berühren (mess-only)", fehler)
    if 'os.environ.setdefault("AURIK_FORCE_CPU", "1")' not in q:
        _fehler("DRT-12: A/B-Skript ohne CPU-only-Vertrag (stört die GPU-Kette)", fehler)
    spec26 = (_ROOT / ".github" / "specs" / "26_defekt_reparatur_trainingsvertrag.md").read_text(encoding="utf-8")
    if "## DRT-12 — Kontroll- und Kanalsatz-A/B" not in spec26:
        _fehler("DRT-12: Spec-26-Abschnitt fehlt", fehler)


def _pruefe_drt_16(fehler: list[str]) -> None:
    """DRT-16: externe RX-12-Parität darf nicht aus internen Scores folgen."""
    spec26 = _ROOT / ".github" / "specs" / "26_defekt_reparatur_trainingsvertrag.md"
    register = _ROOT / ".github" / "WETTBEWERBS_PARITAET.md"
    if not spec26.is_file():
        _fehler("DRT-16: Spec 26 fehlt", fehler)
        return
    spec_text = spec26.read_text(encoding="utf-8")
    for begriff in (
        "## DRT-16 — Externer RX-12-Paritätsvergleich",
        "aurik.rx12_parity/1",
        "musik_mos_delta_ci95",
        "defektlast_delta_ci95",
        "source_id",
        "musik_mos_delta",
        "defektlast_delta",
        "Student-t-95-%-Konfidenzintervalls",
        "1e-6",
        "mindestens",
        "30 Hörer:innen",
        "Holdout",
    ):
        if begriff not in spec_text:
            _fehler(f"DRT-16: verpflichtender Belegbestandteil fehlt: {begriff}", fehler)
    parity_gate = _ROOT / "scripts" / "compliance" / "wettbewerbs_paritaet_gate.py"
    if not parity_gate.is_file():
        _fehler("DRT-16: Wettbewerbs-Paritäts-Gate fehlt", fehler)
    else:
        parity_code = parity_gate.read_text(encoding="utf-8")
        if "_ci95_aus_deltas" not in parity_code:
            _fehler("DRT-16: Wettbewerbs-Paritäts-Gate berechnet CI nicht aus Einzeltitel-Deltas neu", fehler)
        if "_paar_deltas_gebunden" not in parity_code:
            _fehler("DRT-16: Einzeltitel-Deltas sind nicht an gehashte Quellenpaare gebunden", fehler)
    if not register.is_file():
        _fehler("DRT-16: Wettbewerbs-Paritätsregister fehlt", fehler)
        return
    register_text = register.read_text(encoding="utf-8")
    if "## RX-12-Paritätsvertrag" not in register_text:
        _fehler("DRT-16: Register ohne RX-12-Paritätsvertrag", fehler)
    if "`GESICHERT`" not in register_text or "`PARITAET`" not in register_text:
        _fehler("DRT-16: Register ohne getrennte interne/externe Statusgrenze", fehler)


def main() -> int:
    fehler: list[str] = []
    _pruefe_trainer(fehler)
    _pruefe_gate(fehler)
    _pruefe_dsp(fehler)
    _pruefe_drt_789(fehler)
    _pruefe_drt_131415(fehler)
    _pruefe_drt_12(fehler)
    _pruefe_drt_16(fehler)
    if fehler:
        print("🔌 DRT-VERTRAG (Spec 26) — VERSTOSS:")
        for f in fehler:
            print(f"  ❌ {f}")
        return 1
    print("🔌 DRT-VERTRAG (Spec 26) — ✅ alle Leitplanken eingehalten")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
