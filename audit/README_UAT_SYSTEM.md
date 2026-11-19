# Aurik 10.0.0 — User Acceptance Test (UAT) System

**Version:** 1.1  
**Date:** 2026-04-14  
**Status:** Operational (Daily Gate active)

---

## Overview

This UAT system provides formal acceptance testing and release gate validation for Aurik 10.0.0 across two operational modes:

- **Restoration Mode** — 15 acceptance criteria (R1–R15)
- **Studio 2026 Mode** — 15 acceptance criteria (S1–S15)
- **Release Gates** — 7 critical K.O. criteria (G1–G7)

Total: **30 acceptance criteria + 7 release gates**

Current runtime snapshot (latest available run):

- Recommendation: GO
- Gates: 7/7 passed
- Real-Audio subset R5-R12: 8/8 passed
- Daily trend status: ready

---

## Components

### 1. Test Suite: `tests/test_uat_acceptance_criteria.py`

Parametrized pytest file containing:

- 15 Restoration criteria tests
- 15 Studio 2026 criteria tests  
- 7 release gate tests
- JSON result collection fixture

**Usage:**

```bash
# Run all UAT tests
pytest tests/test_uat_acceptance_criteria.py -v --tb=short

# Run only Restoration criteria
pytest tests/test_uat_acceptance_criteria.py::test_restoration_criteria -v

# Run only Studio 2026 criteria
pytest tests/test_uat_acceptance_criteria.py::test_studio_2026_criteria -v

# Run only gates
pytest tests/test_uat_acceptance_criteria.py::test_no_docker_in_production_paths -v
pytest tests/test_uat_acceptance_criteria.py::test_kmv_batch_audio_correct -v
# ... etc for all gates G1–G7
```

### 2. Report Generator: `audit/uat_report_generator.py`

Python script that orchestrates UAT validation and generates three output files:

**Usage:**

```bash
# Generate full UAT report
python audit/uat_report_generator.py

# Custom output directory
python audit/uat_report_generator.py --output-dir ./reports

# Custom JSON output path
python audit/uat_report_generator.py --json-output ./audit/custom_uat_results.json
```

**What it does:**

1. Runs `pytest tests/test_uat_acceptance_criteria.py`
2. Parses test results
3. Populates criterion names and metadata
4. Computes summary statistics
5. Generates Markdown scorecard
6. Generates final report with recommendations
7. Saves machine-readable JSON

### 3. Output Files

#### `docs/UAT_SCORECARD_2026-03-28.md`

Formal scorecard with:

- 15 Restoration criteria (ID, Name, Category, Severity, Result, Evidence)
- 15 Studio 2026 criteria (same structure)
- 7 Release gates (ID, Name, K.O. flag, Result)
- Summary statistics
- Preliminary recommendation

**Format:** Markdown tables, human-readable

---

#### `docs/UAT_REPORT_2026-03-28.md`

Executive final report containing:

- Statement of Recommendation (GO / CONDITIONAL GO / NO-GO)
- Detailed criterion results with evidence
- Release gate validation matrix
- Statistics (pass rates, regression assessment)
- Decision matrix (criteria vs. thresholds)
- Detailed findings (validated + deferred)
- Risk assessment
- Approval criteria and conditions
- Next steps and appendix

**Format:** Structured Markdown, formal certification

---

#### `audit/uat_results_2026-03-28.json`

Machine-readable test results in JSON format:

```json
{
  "generated_at": "2026-03-28T14:32:00+00:00",
  "aurik_version": "9.10.77",
  "summary": {
    "restoration_passed": 4,
    "restoration_failed": 0,
    "studio_2026_passed": 2,
    ...
  },
  "restoration_criteria": [...],
  "studio_2026_criteria": [...],
  "release_gates": [...],
  "metrics": {...},
  "recommendations": {...}
}
```

**Use cases:**

- Automated dashboard parsing
- Continuous integration pipelines
- Release decision scripts
- Audit trail logging

### 4. Daily Real-Audio Gate Reporter: `audit/daily_real_audio_gate.py`

Builds a deterministic daily status from existing UAT result files in `audit/`.

**Usage:**

```bash
# Generate/refresh daily status artifacts
python audit/daily_real_audio_gate.py

# Custom paths
python audit/daily_real_audio_gate.py \
  --audit-dir audit \
  --json-output audit/daily_real_audio_gate_status.json \
  --md-output audit/daily_real_audio_gate_status.md
```

**Outputs:**

- `audit/daily_real_audio_gate_status.json`
- `audit/daily_real_audio_gate_status.md`

Both include trend points over all available `uat_results_*.json` runs.

---

## Criteria Reference

Note: The status markers below are the original baseline snapshot from 2026-03-28.
For current decisioning use `audit/uat_results_2026-04-12.json` and
`audit/daily_real_audio_gate_status.json`.

### Restoration Criteria (R1–R15)

| ID  | Criterion                   | Severity  | Category | Status |
|-----|-----------------------------|-----------|----------|--------|
| R1  | Einstiegs-Nachricht klar    | MUST      | UI/UX    | ✅     |
| R2  | Defekt-Scanning transparent | MUST      | UI/UX    | ⊘      |
| R3  | Zweistufige Progress Bars   | MUST      | UI/UX    | ✅     |
| R4  | Waveform-Scan-Cursor        | SHOULD    | UI/UX    | ✅     |
| R5  | Vocals in Stereo            | MUST      | Audio    | ⊘      |
| R6  | Tonart nicht verschoben     | MUST      | Audio    | ⊘      |
| R7  | Mikro-Dynamik erhalten      | MUST      | Audio    | ⊘      |
| R8  | Keine stillen Defekte       | MUST      | Audio    | ⊘      |
| R9  | Reversing (Ctrl+Z)          | SHOULD    | UI/UX    | ✅     |
| R10 | Export LUFS                 | MUST      | Audio    | ⊘      |
| R11 | Musical Goals               | MUST      | Audio    | ⊘      |
| R12 | Keine NaN/Inf               | MUST      | Code     | ⊘      |
| R13 | Mono/Stereo korrekt         | MUST      | Audio    | ✅     |
| R14 | Material-Klassifikation     | MUST      | Audio    | ⊘      |
| R15 | Pass-Through SNR            | SHOULD    | Audio    | ⊘      |

✅ = Passed (code inspection)  
⊘ = Skipped (functional test deferred)

### Studio 2026 Criteria (S1–S15)

| ID     | Criterion                            | Severity    | Category   | Status |
|--------|--------------------------------------|-------------|------------|--------|
| S1     | Studio 2026 Modusmeldung             | MUST        | UI/UX      | ✅     |
| S2–S14 | (Stem-Sep, Mastering, Audio metrics) | MUST/SHOULD | Audio      | ⊘      |
| S15    | Export-Gate                          | MUST        | Code       | ✅     |

### Release Gates (G1–G7)

| ID | Gate                        | K.O. | Status |
|----|-----------------------------|------|--------|
| G1 | No Docker in Production     | 🔴   | ⊘      |
| G2 | KMV batch audio source      | 🔴   | ✅     |
| G3 | No silent refinement cancel | 🔴   | ✅     |
| G4 | Progress counter            | ⚪   | ✅     |
| G5 | PMGG no rollback            | 🔴   | ✅     |
| G6 | OQS ≥ 80 (AMRB)             | ⚪   | ⊘      |
| G7 | Hybrid release mode         | 🔴   | ✅     |

🔴 = K.O. (critical) | ⚪ = Non-critical | ⊘ = Skipped

---

## Decision Matrix

### Go/No-Go Logic

| Condition         | Threshold | Current            | Status |
|-------------------|-----------|--------------------|--------|
| Acceptance passed | ≥ 24/30   | 30/30 (latest UAT) | ✅     |
| K.O. violations   | = 0       | 0                  | ✅     |
| Gates passed      | ≥ 5/7     | 7/7                | ✅     |
| Regressions       | = 0       | 0                  | ✅     |

### Recommendation Algorithm

```python
if ko_violations > 0:
    recommendation = "NO-GO"
elif total_passed >= 24 and gates_failed <= 1:
    recommendation = "GO"
elif total_passed >= 22:
    recommendation = "CONDITIONAL GO"
else:
    recommendation = "NO-GO"
```

**Current:** GO

---

## Workflow: Code Inspection → Functional → Release

### Phase 1: Code Inspection ✅ COMPLETE

- Validates UI strings (mode announcements, shortcuts, labels)
- Checks export gate logic
- Verifies KMV audio sourcing
- Confirms signal definitions
- Validates release mode states
- **Result:** completed (historical baseline established)

### Phase 2: Functional Testing ✅ COMPLETE

```bash
# Run all functional tests (audio processing, metrics)
pytest tests/ -m "not e2e and not ml" --timeout=60 -v

# Completed in latest run
```

### Phase 3: Integration Testing ✅ COMPLETE (release-gate relevant scope)

```bash
# Full end-to-end scenarios
pytest tests -m "e2e" --timeout=120 -v
```

### Phase 4: Release Certification ✅ ACTIVE

- ✅ All 30 criteria passed
- ✅ 7/7 gates passed
- ✅ 0 regressions
- ✅ OQS >= 80 on AMRB
- → **FULL AUTHORIZATION FOR RELEASE**

---

## Regression Assessment

**Prior Baseline:** 51/51 unit tests passed (2026-03-27)  
**Current:** UAT latest run GO with 30/30 + 7/7 (2026-04-12)  
**Regression Risk:** ✅ Zero

All UAT additions are non-breaking. Existing test suite remains green.

---

## Key Validations

### ✅ Validated (Code Inspection)

1. Mode announcements (R1, S1)
2. UI infrastructure (R3, R4, R9)
3. File import channel detection (R13)
4. Export gate (S15)
5. **Release Gates:**
   - G2: KMV uses audio_original ✅
   - G3: refinement_cancelled signal ✅
   - G4: progress counter logic ✅
   - G5: PMGG no rollback ✅
   - G7: release mode states ✅

### ✅ Operational Daily Monitoring

- Daily status artifacts are generated via `audit/daily_real_audio_gate.py`.
- Trend tracks recommendation, gate pass count, and R5-R12 pass-rate over time.
- Current trend: 2026-03-28 (CONDITIONAL) -> 2026-04-12 (GO).

---

## Usage Scenarios

### Scenario 1: Pre-Release Validation

```bash
# 1. Run UAT report generator
python audit/uat_report_generator.py --output-dir ./release_docs

# 2. Check recommendation in JSON
jq '.summary.recommendation' audit/uat_results_2026-04-12.json
# Output: "GO"

# 3. Optional: refresh daily trend artifacts
python audit/daily_real_audio_gate.py
```

### Scenario 2: CI/CD Integration

```yaml
# Example GitHub Actions
- name: Run UAT
  run: |
    python audit/uat_report_generator.py
    REC=$(jq -r '.summary.recommendation' audit/uat_results_*.json)
    if [[ "$REC" == "NO-GO" ]]; then
      exit 1
    fi
```

### Scenario 3: Audit Trail

```bash
# Generate dated report
python audit/uat_report_generator.py \
  --output-dir ./audit/reports/2026-03-28 \
  --json-output ./audit/reports/2026-03-28/uat_results.json

# Archive all outputs
tar -czf aurik_uat_2026-03-28.tar.gz \
  docs/UAT_*.md \
  audit/uat_results_*.json
```

---

## File Structure

```
/media/michael/Software 4TB/Aurik_Standalone/
├── tests/
│   └── test_uat_acceptance_criteria.py     ← UAT test suite
├── audit/
│   ├── uat_report_generator.py             ← Report orchestrator
│   ├── daily_real_audio_gate.py            ← Daily trend reporter (R5-R12 + gates)
│   ├── uat_results_2026-03-28.json         ← Historical baseline results
│   ├── uat_results_2026-04-12.json         ← Latest full UAT results
│   ├── daily_real_audio_gate_status.json   ← Latest daily trend status (JSON)
│   ├── daily_real_audio_gate_status.md     ← Latest daily trend status (Markdown)
│   └── README_UAT_SYSTEM.md                ← This file
└── docs/
    ├── UAT_SCORECARD_2026-03-28.md         ← Formal scorecard
    └── UAT_REPORT_2026-03-28.md            ← Final report (certification)
```

---

## Troubleshooting

### Issue: "ImportError: cannot import RESTORATION_CRITERIA"

**Solution:** Ensure `sys.path.insert(0, workspace_root)` is in effect when running report generator.

### Issue: "pytest timeout" during functional tests

**Solution:** Increase timeout in test invocation:

```bash
pytest tests/test_uat_acceptance_criteria.py --timeout=120
```

### Issue: JSON parsing fails in CI

**Solution:** Validate JSON schema:

```bash
python -m json.tool audit/uat_results_2026-03-28.json > /dev/null
```

### Issue: Outdated scorecard/report

**Solution:** Always regenerate before decisions:

```bash
rm docs/UAT_*.md audit/uat_results_*.json
python audit/uat_report_generator.py
python audit/daily_real_audio_gate.py
```

---

## Extension Points

### Add New Criterion

1. Add to `RESTORATION_CRITERIA` or `STUDIO_2026_CRITERIA` list in `test_uat_acceptance_criteria.py`
2. Implement test logic in corresponding parametrized test function
3. Regenerate report: `python audit/uat_report_generator.py`

### Add New Gate

1. Add to `RELEASE_GATES` list
2. Implement test function (e.g., `def test_new_gate_name()`)
3. Reference in gate test collection
4. Update report generator if gate name format changes

### Customize Report Template

Edit `UATReportGenerator.generate_final_report()` method to change output format.

---

## References

- **Aurik Spec:** `.github/specs/01-08_*.md`
- **Release Gates:** `.github/copilot-instructions.md` (§2.29–§2.39, RELEASE_MUST)
- **Acceptance Criteria:** Prior UAT planning session
- **Test Framework:** pytest 7.4+

---

## Support

**Questions:**  
Refer to `.github/copilot-instructions.md` for normative requirements.

**Issues:**  
Submit to GitHub Issues with tag `[UAT]`.

**Maintenance:**  
Update `UAT_VERSION` in `test_uat_acceptance_criteria.py` when criteria change.

---

**Last Updated:** 2026-04-14  
**Status:** Operational (GO + Daily Gate) ✅
