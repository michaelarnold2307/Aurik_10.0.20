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


def main() -> int:
    fehler: list[str] = []
    _pruefe_trainer(fehler)
    _pruefe_gate(fehler)
    _pruefe_dsp(fehler)
    _pruefe_drt_789(fehler)
    if fehler:
        print("🔌 DRT-VERTRAG (Spec 26) — VERSTOSS:")
        for f in fehler:
            print(f"  ❌ {f}")
        return 1
    print("🔌 DRT-VERTRAG (Spec 26) — ✅ alle Leitplanken eingehalten")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
