# Aurik 10.0.0 — UAT Quick Reference

- **Last Updated:** 2026-11-22
- **Status:** NOT CERTIFIED — UAT analysis stale
- **Version:** 10.14.0
- **Test Framework:** pytest

> The metrics and test counts below are a historical 2026-03-28 snapshot, not
> a current release decision. The latest available analysis is dated
> 2026-11-15 (7 days old): 29/30 criteria, R5 failed, 5/7 release gates,
> G1 skipped and G6 missing. Its recorded `GO` is invalid. The freshness gate
> failed on 2026-11-22; a fresh UAT run is pending a training-free window.

---

## 30-Second Summary

| Metric | Value | Status |
|--------|-------|--------|
| **Acceptance Criteria Passed** | 6/30 (20%) | ⚠️ Code baseline |
| **Release Gates Passed** | 5/7 (71%) | ✅ Critical |
| **K.O. Gate Violations** | 0 | ✅ Clear |
| **Test Regressions** | 0/51 prior | ✅ Clean |
| **Recommendation** | CONDITIONAL GO | ⚠️ Proceed with conditions |

---

## Criteria Breakdown

### 🟢 Passed (6 Tests)

- **R1:** Mode announcement clear
- **R3:** Dual progress bars
- **R4:** Waveform cursor
- **R9:** Ctrl+Z reversing
- **R13:** Mono/stereo detection
- **S1:** Studio 2026 announcement
- **S15:** Export gate quality check

### 🟡 Skipped (24 Tests)

- **R2, R5–R8, R10–R12, R14–R15:** Restoration audio tests (functional, deferred)
- **S2–S14:** Studio 2026 audio tests (functional, deferred)

### 🔴 Failed (0 Tests)

- None

---

## Release Gates Status

| ID | Gate | K.O. | Status |
|----|------|------|--------|
| G1 | No Docker | 🔴 | ⊘ PENDING |
| G2 | KMV audio source | 🔴 | ✅ PASS |
| G3 | No silent cancellation | 🔴 | ✅ PASS |
| G4 | Progress counter | ⚪ | ✅ PASS |
| G5 | PMGG no rollback | 🔴 | ✅ PASS |
| G6 | OQS ≥ 80 (AMRB) | ⚪ | ⊘ PENDING |
| G7 | Release mode | 🔴 | ✅ PASS |

**K.O. Status:** 0 violations ✅

---

## Current Decision Path

```
                    START
                      ↓
    ┌─────────────────────────────────┐
    │ pytest exit 0; no failed/error/ │
    │ missing result; no K.O.?       │
    └─────────────────────────────────┘
                      │
                  YES │ NO
                      ↓
    ┌─────────────────────────────────┐
    │ 30/30 criteria PASS and 7/7     │
    │ gates PASS with no skips?       │
    └─────────────────────────────────┘
                      │
           YES        │        NO
            ↓         ↓         ↓
           GO    CONDITIONAL  NO-GO
           ✅        ⚠️        ❌
```

**Current Status:** NOT CERTIFIED. A failed, skipped, errored, or missing
mandatory check cannot be counted as a pass. Skips can result in at most
`CONDITIONAL`; only complete 30/30 + 7/7 all-pass evidence results in `GO`.

---

## What This Means

### Current Status: Not certified

- The latest analysis is stale, so it cannot establish current readiness.
- The recorded run failed R5 and did not produce all seven gate results.
- Zero K.O. violations alone do not override failures or missing evidence.

### ⚠️ Conditions for Final Release

**Required before release:**

1. Run the full UAT in a training-free window.
2. Resolve every failed criterion and gate; investigate missing results.
3. Regenerate the daily status from the new UAT JSON and rerun the freshness test.
4. Validate audio quality with paired full-title measurements and human
   listening sign-off; UAT alone is not an audio-quality or RX-12 claim.

**Next quality step:** Recheck R5's stereo-correlation regression in the fresh
run before changing its guard. If it reproduces, trace the production path and
repair the cause; do not loosen the threshold to make the test pass.

---

## Key Validations ✅

| Component | Finding |
|-----------|---------|
| **Mode Announcements** | Both strings present |
| **UI Progress Bars** | Dual bars configured |
| **KMV Batch Processing** | Uses original audio (not tube3) |
| **Refinement Cancellation** | Signal defined & fired |
| **Export Guard** | Quality gate implemented |
| **Release Mode States** | primary/fallback/blocked defined |
| **PMGG Best-Effort** | Rollback prohibited, best-effort used |
| **Test Regression** | 51/51 prior tests still pass |

---

## Next Steps (Priority Order)

### Immediate (Phase 2 — Functional)

```bash
# ~2 hours runtime
pytest tests/ -m "not e2e and not ml" --timeout=60 -v

Expected: R2, R5–R8, R10–R12, R14–R15, S2–S14 → PASS
```

### Soon (Phase 3 — Integration)

```bash
# ~1 hour runtime
pytest tests -m "e2e" --timeout=120 -v
```

### Optional (Phase 4 — Benchmark)

```bash
# ~4 hours runtime
pytest tests -m "ml" --run-heavy-tests --timeout=120
```

---

## Current Risks (as of 2026-11-22)

| Risk | Evidence | Required action |
|---|---|---|
| Stale UAT | Latest analysis 2026-11-15; freshness test fails at 7 days | Run full UAT in a training-free window |
| Stereo criterion R5 | Latest analysis reports correlation 0.595 → 0.759, above its 0.695 ceiling | Reproduce on fresh run, then fix the production cause |
| Incomplete release gates | G1 skipped; G6 missing; 5/7 passed | Produce all seven gate results |
| False-positive release label | Old summary said GO despite 29/30 and incomplete gates | Generator now fails closed; rerun after fix |
| Regression claim | UAT does not compare a regression baseline | Do not claim zero regressions from UAT alone |

## Release Recommendation

No release approval is asserted. `GO` requires a successful pytest exit, all
30 acceptance criteria passing, all 7 release gates passing, no skips/errors,
and complete result accounting. Any failure, error, or missing result is
`NO-GO`; explicit skips can yield at most `CONDITIONAL`.

---

## Artifacts Produced

| Document | Path | Purpose |
|----------|------|---------|
| Scorecard | `docs/UAT_SCORECARD_2026-03-28.md` | Formal test matrix |
| Report | `docs/UAT_REPORT_2026-03-28.md` | Executive certification |
| JSON Results | `audit/uat_results_2026-03-28.json` | Machine parsing |
| This Card | `audit/UAT_QUICK_REFERENCE.md` | Quick lookup |
| System Docs | `audit/README_UAT_SYSTEM.md` | How-to guide |
| Test Suite | `tests/test_uat_acceptance_criteria.py` | Executable tests |
| Generator | `audit/uat_report_generator.py` | Report orchestrator |

---

## Frequently Asked Questions

**Q: Can we release now?**  
A: No current release certification is available. The latest analysis is stale, failed R5, and omitted a gate result.

**Q: What are K.O. violations?**  
A: Critical gates that block release if failed. Zero K.O. violations do not override failures, skips, errors, or missing results.

**Q: Why are so many tests skipped?**  
A: A skip is not a pass. Skips can result in at most a conditional recommendation.

**Q: What if functional tests fail?**  
A: The report must be NO-GO; investigate and fix the production cause, then rerun the full suite.

**Q: What is needed for an actual audio-quality breakthrough?**

A: Reproducible, paired full-title gains without over-restoration, protected hearing invariants, and blinded human listening. RX-12 parity additionally requires the external DRT-16 holdout and panel evidence.

**Q: How is UAT freshness renewed?**

A: Run `audit/uat_report_generator.py` in a training-free window, regenerate `audit/daily_real_audio_gate.py` from that result, then rerun `test_daily_gate_stored_recently`. Do not refresh only the report timestamp.

---

## Legend

| Symbol | Meaning |
|--------|---------|
| ✅ | Passed / Validated / Clear |
| ❌ | Failed / Blocked |
| ⚠️ | Conditional / Pending |
| ⊘ | Skipped / Deferred |
| 🔴 | K.O. Critical Gate |
| ⚪ | Non-Critical Gate |

---

## Report Generation

- **Historical Snapshot:** 2026-03-28
- **Latest Available UAT Analysis:** 2026-11-15 (stale)
- **Current Version:** 10.14.0
- **Python:** Repository `.venv_aurik` environment

To regenerate:

```bash
python audit/uat_report_generator.py
```

---

**CURRENT STATUS: NOT CERTIFIED — fresh UAT and audio-quality evidence pending**

---

_Aurik 10 — User Acceptance Testing Framework_
_Historical snapshot retained; current status updated 2026-11-22_
