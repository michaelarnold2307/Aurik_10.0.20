# CI-Determinismus-Checkliste

Version: 1.0
Stand: 2026-04-08
Geltung: Benchmark-, Gate- und Release-Jobs

## 1. Ziel

Sicherstellen, dass identische Eingaben reproduzierbare Ergebnisse erzeugen und qualitative Gates nicht zufallsgetrieben entscheiden.

## 2. Pflichtmetadaten pro Run

- `git_commit`
- `python_version`
- `platform`
- `seed_global`
- `model_versions`
- `input_hashes`
- `config_hash`

## 3. Determinismus-Checks (Pflicht)

1. Wiederholungslauf mit identischem Seed
2. Vergleich aller Kernergebnisse innerhalb enger Toleranzbaender
3. Keine driftenden Gate-Entscheidungen zwischen Re-Runs

Empfohlene Toleranzen:

- `quality_estimate`: abs_delta <= 0.005
- Goal-Scores: abs_delta <= 0.005
- OQS/PQS: abs_delta <= 0.01
- Export-Gate-Entscheidung: exakt identisch

## 4. Pflichtbefehle (Beispiel)

```bash
# Run A
.venv_aurik/bin/python -m pytest tests -p no:xdist --override-ini="addopts=--strict-markers --import-mode=importlib" -q

# Run B (identische Parameter)
.venv_aurik/bin/python -m pytest tests -p no:xdist --override-ini="addopts=--strict-markers --import-mode=importlib" -q
```

## 5. Failure-Policy

Ein CI-Job faellt hart, wenn:

- Seeds fehlen,
- Toleranzbaender verletzt werden,
- Gate-Entscheidungen zwischen Run A/B differieren,
- Model- oder Config-Hashes nicht geloggt sind.

## 6. Ergebnisprotokoll (Pflichtfelder)

- Determinismus: PASS/FAIL
- Max-Delta je Kernmetrik
- Betroffene Szenarien
- Root-Cause-Hinweis
- Reproduktionskommando

## 7. Integration in Release

Kein Release ohne gruenen Determinismus-Block fuer:

- Unit/Integration-Qualitaetsgates
- AMRB-Lauf
- Export-Gate-Szenarien
