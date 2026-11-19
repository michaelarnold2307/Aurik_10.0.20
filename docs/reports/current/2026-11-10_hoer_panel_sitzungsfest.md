# Hör-Entscheidungs-Infrastruktur sitzungsfest — C4-Befund n=12 (2026-11-10)

**Klasse:** K3 (Werkzeug/Prozess, kein Klangpfad) · **Register:** D-K3-93 ·
**Kein Flag-Flip, kein Training, keine Hör-Entscheidung.**

## 1. Anlass

Die Hör-Instanz hat am 2026-11-10 die C4-Teilfrage (vocos-Base vs.
48-kHz-Musik-Finetune) **in zwei Sitzungen** beurteilt. Beim Zusammenführen fiel
auf: die Werkzeugkette konnte mehrere Sitzungen **nicht** addieren, und die
Unit-Tests löschten dabei das **echte** Ohr-Artefakt des Nutzers. Beides ist
repariert, das Ergebnis (n=12) ist gebucht.

## 2. Befunde (alle mit Beleg im Code/Test)

| # | Befund | Wirkung vorher | Fix |
| --- | --- | --- | --- |
| 1 | `tests/unit/test_ohr_panel.py` löschte `models/ohr/ohr_prior.json` (E2E-Tests, `unlink()` ohne Sicherung) | **Jeder Testlauf zerstörte die menschliche Ohr-Kalibrierung** — die persönliche Hör-Instanz ist ein Produktionsartefakt und war nicht reproduzierbar | Fixture `ohr_artefakt_gesichert`: sichert die Bytes, löscht für den Test, stellt **bit-identisch** wieder her (zwei E2E-Tests) |
| 2 | `scripts/hoer_entscheidung.py` schrieb je Sitzung einen **neuen** Beleg | n blieb bei 6 statt 12; frühere Urteile waren weg ⇒ Signifikanz strukturell unerreichbar | Trials werden **additiv** je Probe (stabiler Schlüssel `stamm-slug__position`) gespeichert; `--wiederholen` ersetzt gezielt; Kennzahlen werden über **alle** Trials aggregiert |
| 3 | `scripts/erstelle_hoerproben.py` erzeugte bei erneutem Lauf ein **neues** Set | Proben-Positionen wanderten ⇒ gespeicherte Urteile hätten auf andere Paare gezeigt | `_merge_set`: bestehende Proben bleiben, neue werden nur angehängt (Stämme eindeutig); Manifest-Ziel `--anzahl` ist eine Obergrenze |
| 4 | `scripts/ohr_panel.py` kannte nur das `<stamm>_restauriert`-Layout | die stärksten Paare (`<stamm>_ORIGINAL` ↔ `_ML-ADAPT`, `-v2`, `-v3`) wurden nicht gefunden | ORIGINAL-Varianten-Layout ergänzt (`a` = Original, je Variante ein Paar), sortiert/deterministisch |
| 5 | Nur-Gleichstände-Sitzung lief in `ohr_kalibrieren.py` | Crash ohne Urteil, Gefahr eines halb geschriebenen Artefakts | `ohr_panel.py` bricht vor dem Fit sauber ab (`return 1`, „bestehendes Ohr-Artefakt bleibt unverändert") |
| 6 | `scripts/ohr_kalibrieren.py` verlangte exakt 48 kHz | 44,1-kHz-Exporte (Produktionsbefund) konnten nicht kalibriert werden | deterministisches `resample_poly` auf die Zielrate (§G5), INFO-Log statt Hard-Fail |

## 3. Verifikation

- `python3 -m pytest tests/unit/test_ohr_panel.py tests/unit/test_ohr_prior.py tests/unit/test_hoerproben_b3c4.py -q`
  → **43 passed** (18,6 s), inkl. neuer Fälle: ORIGINAL-Layout, Nur-Gleichstände,
  Artefakt-Rundreise, additive Trials.
- Das Nutzer-Artefakt `models/ohr/ohr_prior.json` (602 B, Stand 16:51) ist nach
  dem Lauf **unverändert** vorhanden.

## 4. C4-Teilbefund (gebucht, kein Flag-Flip)

| Kandidat | Siege | n entscheidend | Gleichstände | p (Vorzeichen) | signifikant |
| --- | --- | --- | --- | --- | --- |
| vocos-Base (A) vs. Musik-Finetune (B) | 5 / 5 | 10 | 2 | 1,0 | nein |

Beleg: `docs/reports/current/hoer_verdikt_c4_vocos.json` (12 Trials, Schema
`aurik.ab_stats/1`, `flag_freigabe.py`-kompatibel). **Lesart:** das
30-Epochen-Finetune ist für dieses Ohr von der Basis nicht unterscheidbar ⇒
kein hörbarer Vorteil, kein §4-Flip. Die C4-Primärrollen-Entscheidung reduziert
sich auf den BigVGAN-Vergleich (D3-2) und bleibt **menschlich**.

## 5. Grenzen (ehrlich)

- Die B3-W-Stufen (W-1/W-4/W-5 …) brauchen die objektive Messung im
  **trainingsfreien GPU-Fenster**; das Panel zeigt sie deshalb nur als
  „wartet auf Messung" an. Kette 2 belegte die GPU (D3-2 BigVGAN).
- `n=12` ist eine kleine Stichprobe; die Aussage gilt für dieses Material
  (`~/Musik`, 10-s-Proben, Seed 42) und ist **kein** MUSHRA-Ersatz (n≥30,
  menschlich).
- Die persönliche Kalibrierung wirkt nur, wenn `models/ohr/ohr_prior.json`
  existiert; ohne Artefakt ist die Wirkung null (unverändert).
