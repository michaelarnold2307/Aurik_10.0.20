"""Unit-Tests: musik_mos/Wohlklang-Term im Witness-Kopf-Loss + Zwei-Metrik-Modellauswahl.

Belegt die SOTA-HANDOVER-Punkt-34f-Umsetzung (DRT-8/10, D-K3-95-Folge,
2026-11-18) modell-frei, CPU-only und deterministisch:
- Loss-Term-Mathematik: Gewicht 0 ⇒ bit-identisch zum alten Loss;
  Gewicht > 0 senkt den mos-Proxy-Verlust (Gradienten-Schritt).
- Zwei-Metrik-Modellauswahl: defektlast UND musik_mos mit mos-Mindest-Schwelle
  (JND 0,05) statt nur val_loss — an zwei fiktiven Checkpoint-Metriken.
- Rückwärtskompatibilität: Default AUS ohne Flag (§G5 (GEBOTE.md)).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import scripts.train_witness_kopf as wk
import scripts.verify_witness_kopf_gate as gate

_SRC_TRAIN = Path(wk.__file__).read_text(encoding="utf-8")
_SRC_GATE = Path(gate.__file__).read_text(encoding="utf-8")


# ─── Loss-Term-Mathematik ─────────────────────────────────────────────────────


def test_gewicht_null_bitidentisch_zum_alten_loss() -> None:
    """lam_mos = 0 ⇒ Gesamt-Loss ist bit-identisch zum alten Loss (torch.equal)."""
    torch.manual_seed(42)
    l1, pen, l1_id, spec, mos_term = (torch.rand(()) for _ in range(5))
    lam, lam_id, lam_spec = 0.1, 0.5, 0.3
    alt = l1 + lam * pen + lam_id * l1_id + lam_spec * spec
    assert torch.equal(
        wk._kombinierter_loss(l1, pen, l1_id, spec, mos_term, lam, lam_id, lam_spec, 0.0),
        alt,
    )
    # mos_term None (Default-Pfad ohne Proxy) ebenfalls bit-identisch
    assert torch.equal(
        wk._kombinierter_loss(l1, pen, l1_id, spec, None, lam, lam_id, lam_spec, 0.0),
        alt,
    )


def test_gewicht_positiv_addiert_mos_term_exakt() -> None:
    """lam_mos > 0 ⇒ der mos-Term wird exakt dazuaddiert (keine andere Änderung)."""
    torch.manual_seed(42)
    l1, pen, l1_id, spec, mos_term = (torch.rand(()) for _ in range(5))
    lam, lam_id, lam_spec, lam_mos = 0.1, 0.5, 0.3, 2.0
    alt = l1 + lam * pen + lam_id * l1_id + lam_spec * spec
    assert torch.equal(
        wk._kombinierter_loss(l1, pen, l1_id, spec, mos_term, lam, lam_id, lam_spec, lam_mos),
        alt + lam_mos * mos_term,
    )


def test_gewicht_positiv_senkt_mos_proxy_verlust() -> None:
    """Gradienten-Schritte auf dem mos-Term senken den Hinge-Verlust Richtung
    ziel-mos; Gewicht 0 lässt Parameter und Proxy-Verlust unverändert."""
    ziel = 4.0

    def proxy(w: torch.Tensor) -> torch.Tensor:
        # Modell-freies "Proxy": mos = 1 + 4·sigmoid(w) ∈ (1, 5), differenzierbar.
        return 1.0 + 4.0 * torch.sigmoid(w)

    verlust_start = float(wk._mos_proxy_verlust(proxy(torch.tensor(0.0)), ziel))
    assert verlust_start == pytest.approx(1.0)  # sigmoid(0) = 0,5 ⇒ mos 3,0 ⇒ hinge 1,0

    # Gewicht 0: Gradient des Terms ist 0 — nichts ändert sich.
    w_null = torch.tensor(0.0, requires_grad=True)
    (0.0 * wk._mos_proxy_verlust(proxy(w_null), ziel)).backward()
    assert w_null.grad is not None and float(w_null.grad) == 0.0

    # Gewicht 1,0: 200 Gradienten-Schritte senken den mos-Proxy-Verlust.
    w = torch.tensor(0.0, requires_grad=True)
    for _ in range(200):
        verlust = 1.0 * wk._mos_proxy_verlust(proxy(w), ziel)
        w.grad = None
        verlust.backward()
        with torch.no_grad():
            w.data -= 0.5 * w.grad
    mos_ende = float(proxy(w.detach()))
    verlust_ende = float(wk._mos_proxy_verlust(proxy(w.detach()), ziel))
    assert verlust_ende < verlust_start
    assert mos_ende > 3.5


def test_proxy_klasse_deterministisch_und_differenzierbar() -> None:
    """Das Proxy-Netz ist deterministisch (gleicher Seed ⇒ gleiche Werte, §G5 (GEBOTE.md))
    und differenzierbar in der Mel-Eingabe."""
    torch.manual_seed(42)
    proxy_a = wk.MusikMOSProxy(n_mels=8, frames=16)
    torch.manual_seed(42)
    proxy_b = wk.MusikMOSProxy(n_mels=8, frames=16)
    mel = torch.randn(2, 1, 8, 16)
    mos = proxy_a(mel)
    assert tuple(mos.shape) == (2,)
    assert bool((mos > 1.0).all() and (mos < 5.0).all())
    assert torch.equal(proxy_a(mel), proxy_b(mel))
    mel_g = torch.randn(2, 1, 8, 16, requires_grad=True)
    proxy_a(mel_g).sum().backward()
    assert mel_g.grad is not None and bool(torch.isfinite(mel_g.grad).all())


# ─── Zwei-Metrik-Modellauswahl (fiktive Checkpoint-Metriken) ─────────────────


def test_zwei_metrik_auswahl_mos_schwelle_schuetzt_wohlklang() -> None:
    """DRT-8/10: die defektlast-optimale Epoche mit mos-Einbruch (> JND) verliert
    gegen die mos-stabile — Mindest-Schwelle statt nur val_loss."""
    k1 = {"ep": 0, "score": 0.10, "mos": 3.00}  # die alte Auswahl hätte k1 gewählt
    k2 = {"ep": 1, "score": 0.12, "mos": 4.50}
    assert wk._zwei_metrik_auswahl([k1, k2]) is k2


def test_zwei_metrik_auswahl_innerhalb_jnd_gewinnt_defektlast() -> None:
    """Innerhalb der mos-Schwelle (JND 0,05) gewinnt die kleinere defektlast."""
    k1 = {"ep": 0, "score": 0.10, "mos": 4.48}
    k2 = {"ep": 1, "score": 0.12, "mos": 4.50}
    assert wk._zwei_metrik_auswahl([k1, k2]) is k1


def test_zwei_metrik_auswahl_ohne_mos_faellt_auf_defektlast_zurueck() -> None:
    """Ohne messbaren mos gilt der alte Weg: kleinster Score (§V6-Fallback)."""
    k1 = {"ep": 0, "score": 0.10, "mos": None}
    k2 = {"ep": 1, "score": 0.12, "mos": None}
    assert wk._zwei_metrik_auswahl([k1, k2]) is k1
    assert wk._zwei_metrik_auswahl([]) is None


def test_zwei_metrik_auswahl_unbelegte_kandidaten_gewinnen_nicht() -> None:
    """Ein Kandidat ohne mos-Beleg konkurriert nicht gegen belegte (DRT-8/10)."""
    k1 = {"ep": 0, "score": 0.05, "mos": None}
    k2 = {"ep": 1, "score": 0.10, "mos": 4.0}
    assert wk._zwei_metrik_auswahl([k1, k2]) is k2


def test_auswahl_nw_schutzarm_gewinnt_vor_schlechtem_nw() -> None:
    """Exit-1-Fix (2026-11-19): mit --nw-min gewinnen NUR Epochen mit
    Holdout-nw ≥ Schwelle — die Runde 2 wählte eine Epoche mit kollabiertem
    never-worsen (wow nw 0,06), obwohl eine nw-sichere existierte."""
    k_gut = {"ep": 1, "score": 0.11, "mos": 4.50, "nw": 0.93}
    k_schlecht = {"ep": 2, "score": 0.10, "mos": 4.52, "nw": 0.06}
    assert wk._zwei_metrik_auswahl([k_gut, k_schlecht], nw_min=0.9) is k_gut


def test_auswahl_nw_fallback_ohne_genuegende_epoche() -> None:
    """Keine nw-genügende Epoche ⇒ alte Zwei-Metrik-Auswahl (nie schlechter als bisher)."""
    k1 = {"ep": 0, "score": 0.10, "mos": 4.48, "nw": 0.5}
    k2 = {"ep": 1, "score": 0.12, "mos": 4.50, "nw": 0.4}
    assert wk._zwei_metrik_auswahl([k1, k2], nw_min=0.9) is k1


def test_auswahl_nw_default_none_bit_identisch() -> None:
    """nw_min=None lässt das alte Verhalten unverändert (§G5 (GEBOTE.md))."""
    k1 = {"ep": 0, "score": 0.10, "mos": 4.48, "nw": 0.2}
    k2 = {"ep": 1, "score": 0.12, "mos": 4.50, "nw": 0.9}
    assert wk._zwei_metrik_auswahl([k1, k2]) is wk._zwei_metrik_auswahl([k1, k2], nw_min=None) is k1


# ─── Rückwärtskompatibilität / Spec-Referenzen ────────────────────────────────


def test_backward_compat_default_aus() -> None:
    """Default AUS: --lambda-mos 0,0; ohne Flag entsteht kein Proxy und der alte
    Auswahl-Pfad (kleinster Holdout-Score) bleibt der Default (§G5 (GEBOTE.md))."""
    assert '"--lambda-mos"' in _SRC_TRAIN
    assert "default=0.0" in _SRC_TRAIN
    assert "if lam_mos > 0.0" in _SRC_TRAIN
    assert "if mos_proxy is None:" in _SRC_TRAIN
    assert "_zwei_metrik_auswahl" in _SRC_TRAIN
    assert "MOS_JND" in _SRC_TRAIN


def test_spec_referenzen_dokumentiert() -> None:
    """Die geforderten Spec-Referenzen stehen im Quelltext (DRT-8/10, D-K3-95,
    §G5 (GEBOTE.md), singmos-Proxy-Muster)."""
    assert "DRT-8/10" in _SRC_TRAIN
    assert "D-K3-95" in _SRC_TRAIN
    assert "§G5 (GEBOTE.md)" in _SRC_TRAIN
    assert "scripts/train_cantus.py" in _SRC_TRAIN


def test_gate_mos_messung_default_aus() -> None:
    """Das Gate bleibt ohne --mos bit-identisch: die Zwei-Metrik-Felder entstehen
    nur im Flag-Pfad (kein Verhalten ohne Flag geändert, §G5 (GEBOTE.md))."""
    assert '"--mos"' in _SRC_GATE
    assert "mos_messen" in _SRC_GATE
    assert "mos_messen: bool = False" in _SRC_GATE
    assert 'action="store_true"' in _SRC_GATE
    assert "if mos_messen:" in _SRC_GATE
    assert "musik_mos_delta_mittel" in _SRC_GATE
