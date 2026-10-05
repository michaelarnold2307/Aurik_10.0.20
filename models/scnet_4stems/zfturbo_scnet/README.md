# zfturbo_scnet — Vendored SCNet-Architektur (unverändert)

- **Quelle:** https://github.com/ZFTurbo/Music-Source-Separation-Training → `models/scnet/`
  (`scnet.py`, `separation.py`, `__init__.py`), MIT-Lizenz (LICENSE beiliegend).
- **Entnommen:** 2026-10-05, byte-genau kopiert (keine Änderungen).
- **Zweck:** Architektur-Definition zum Kandidaten-Checkpoint
  `models/scnet_4stems/huge_scnet_4stems_v1.2.ckpt` (Aname-Tommy/Huge-SCNet-4stems,
  Apache-2.0) für das TODO-P1-2-A/B (`scripts/eval_scnet_vs_mdx23c.py`).
  Autoren-Hinweis der Model-Card: „If UVR not works, try ZFTurbo's msst."
- **Rolle:** Bewertungs-Kopie, **kein Produktionspfad** — es wird kein
  Produktions-Flag umgestellt; Integrationsentscheidung bleibt menschlich (C4).
