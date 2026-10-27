# GO/NO-GO Decision Protocol — Hearing Test Validation (v10.0.0)

- Purpose: Structured decision framework for PR reviewers evaluating hearing-test results

- Spec Reference: .github/specs/07_quality_and_tests.md §5.7a [RELEASE_MUST] §8.1.1a

- Version: 10.0.0 | Date: 2026-05-19 | Mandatory for Aurik 10.0.0 Release

---

## I. Overview & Purpose

This protocol guides PR reviewers through **hearing-test results evaluation** for Aurik 10.0.0.x restoration and studio2026 enhancement pipelines. It ensures:

1. **Mode-Separated Scoring**: Restoration and studio2026 use incompatible quality dimensions; scores must NOT be mixed
2. **Objective GO/NO-GO Decision**: Quantitative gates + subjective listener agreement consensus
3. **Artifact-Detection Rigor**: Any reported artifact triggers per-listener deep-dive investigation
4. **OQS/Hörurteil Consistency**: Studio2026 OQS ≥ 88 must align with listener MOS ≥ 4.3–4.5; inconsistency = investigation blocker
5. **Traceability**: Full audit trail of decisions, dissenting opinions, and remediation notes

---

## II. Pre-Review Checklist (Immediate, Before Scoring)

### A. Scenario Metadata Validation

- [ ] All 6 scenarios present: 3 restoration (RESTORATION_SCENARIO_1–3), 3 studio2026 (STUDIO2026_SCENARIO_1–3)
- [ ] File timestamps consistent (all created ≤ 24 hours apart)
- [ ] Scenario YAML/JSON intact and parseable (no truncation, encoding issues)
- [ ] **Rejection Criteria**: Any missing scenario or corrupted metadata → NO-GO (ask researcher to re-run hearing test)

### B. Listener Demographics & Methodology

- [ ] Listener count: 8 per scenario (N=8 minimum)
- [ ] Listener pool: "Trained" / "Untrained" documented; recommend ≥ 4 trained (golden ears)
- [ ] Listening environment: Headphones or calibrated room documented (loudness ≤ 85 dBSPL)
- [ ] **Confirmation**: Each listener completed all 6 scenarios or document dropout reason
- [ ] **Rejection Criteria**: N < 6, no environment documentation, or listener dropout without reason → NO-GO

### C. Scoring Format & Completeness

- [ ] MOS (1–5 ACR scale) or equivalent 5-point scale documented
- [ ] Per-scenario aggregate data: Mean, Std Dev, Min, Max reported for each listener group
- [ ] Open-ended comments/artifacts reported: Any listener mentions "artifact," "unnatural," "synthetic" recorded verbatim
- [ ] **Rejection Criteria**: Incomplete data (e.g., MOS missing for scenario) → Request remediation

### D. Required External Evidence Pack (Head-to-Head)

- [ ] Externes Blindtest-Artefakt vorhanden (MUSHRA/ABX), zugeordnet zu aktuellem Commit/PR
- [ ] Head-to-Head gegen mindestens eine externe Referenz dokumentiert
- [ ] Statistik vorhanden: Signifikanztest, Effektstaerke, 95 %-Konfidenzintervall
- [ ] Pre-Registration/Analyseplan verlinkt (z. B. Studienordner unter docs/reports/studies/)
- [ ] **Rejection Criteria**: Fehlendes Evidence-Pack bei kernaendernden Audio-PRs → NO-GO

---

## III. Restoration Mode Decision Flow

### Phase 1: Aggregate Gate Check (All 3 Restoration Scenarios)

**Step 1.1: Natürlichkeit & Authentizität Consensus**

```
For each RESTORATION_SCENARIO_1–3:
  ✓ PASS if: ≥ 6/8 listeners rate "Naturalness" ≥ 3.5/5 (Good or better)
            AND ≥ 6/8 perceive restoration as "original" or "minimally processed" (blind detection)
  ✗ FAIL if: < 6/8 hit threshold
  → Action on FAIL: Review listener comments; identify specific defect (e.g., "click removal too aggressive")
```

**Step 1.2: Artifact Freedom Absolute Veto**

```
For each scenario:
  ✗ VETO (Automatic NO-GO) if: ANY listener reports new artifact unprompted (e.g., "robotic," "phase-weird," "echo")
                                AND artifact appears in ≥ 2/8 listener comments
  ✓ PASS if: ≤ 1 listener reports possible artifact (outlier), majority (≥ 6/8) perceive "natural"
  → Action on VETO: Stop here. Root-cause investigation required (§2.49 Delta/Baseline check; likely AFG calibration error)
```

**Step 1.3: P1/P2 Goals Consistency (Authenticity, Artikulation)**

```
For each scenario:
  Cross-check: Listeners rate "Authenticity" AND "Articulation" (Lyric Clarity)
  ✓ PASS if: Both ≥ 3.5/5 (Good); no P1/P2 regression
  ⚠ WARN if: One dimension dips to 3.0–3.5 (Acceptable, but marginal)
  ✗ FAIL if: Either < 3.0 (Poor); indicates defect in phase execution or RIAA/Material-Adaptive gate
  → Action on WARN: Flag in PR; acceptable if trade-off is justified and artifact-free
  → Action on FAIL: Escalate to phase debugging
```

### Phase 2: Per-Scenario Thresholds (All 3 Must Pass)

**Restoration Scenario 1 (Vinyl Crackle)**: FocusPoints = "Crackle removal quality" + "Vocal clarity"

```
✓ PASS if:
  • Crackle removal rated ≥ 3.5/5 by ≥ 7/8 listeners (e.g., "unnoticeable" or "minimal clickiness")
  • Vocal intelligibility maintained/improved by ≥ 6/8 listeners
  • No phase-cancellation reported; stereo balance stable (<6 dB L/R imbalance in restoration)
  • Artifact veto NOT triggered (Step 1.2)

⚠ WARN if:
  • Crackle rating 3.0–3.5/5 (marginal quality)
  • 1–2 listeners report minor artifacts but ≥ 6 say "natural overall"

✗ FAIL if: Crackle removal < 3.0/5 or artifact veto triggered
```

**Restoration Scenario 2 (Tape Hiss + Dropout)**: FocusPoints = "Hiss reduction" + "Dropout repair" + "Warmth"

```
✓ PASS if:
  • Hiss reduction rated ≥ 3.5/5 by ≥ 6/8 listeners (tape texture preserved, not "dead")
  • Dropout repair rated ≥ 3.7/5 by ≥ 7/8 (seamless blend, no speech gaps)
  • Wärme (tape saturation) maintained ≥ 3.5/5 by ≥ 6/8 listeners
  • Artifact veto NOT triggered

⚠ WARN if:
  • Hiss reduction < 3.5/5 OR warmth < 3.5/5 (acceptable tradeoff, but needs justification)
  • 1 listener perceives dropout repair as "audible patch" but 7/8 do not

✗ FAIL if: Hiss < 3.0/5 or Dropout repair < 3.0/5 or artifact veto triggered
```

**Restoration Scenario 3 (Shellac Click Storm)**: FocusPoints = "Click removal" + "Frequency extension" + "Brittleness"

```
✓ PASS if:
  • Click removal rated ≥ 3.5/5 by ≥ 7/8 listeners (audibly effective, no residual clicks)
  • Frequency restoration rated ≥ 3.5/5 by ≥ 6/8 (presence without harshness)
  • Overall "Brillanz" (not defined as "brightness," but as "intelligible presence") ≥ 3.3/5 by ≥ 6/8
  • Artifact veto NOT triggered

⚠ WARN if:
  • Click removal 3.0–3.5/5 (marginal; may need phase tuning)
  • Frequency extension perceived as "slightly bright" by 3–4/8 listeners but acceptable by majority

✗ FAIL if: Click removal < 3.0/5 or Brillanz < 3.0/5 or artifact veto triggered
```

### Phase 3: Restoration Mode GO/NO-GO Summary

**GO (Release Blocker Clear)** if:

- Phase 1 all ✓ PASS (Natürlichkeit, Authentizität, Artifact-Veto clear, P1/P2 intact)
- Phase 2 all ✓ PASS (All 3 scenarios pass per-scenario thresholds)
- Optional ⚠ WARN flags documented in PR but do not block merge

**NO-GO (Release Blocker Active)** if:

- Artifact veto triggered in Step 1.2 (any scenario)
- Phase 2 any ✗ FAIL (defect scores < thresholds)
- Unexplained listener disagreement (e.g., 4/8 "natural" vs. 4/8 "robotic") → requires investigation before merge

**Conditional GO (Requires Justification)** if:

- Phase 2 has ⚠ WARN flags + no failures
- PR author must document: "Tradeoff rationale: [specific reason], listener impact acceptable"
- Reviewer must approve justification explicit in PR

---

## IV. Studio2026 Mode Decision Flow

### Phase 1: Objective Quality Gate Checks (All 3 Scenarios)

**Step 1.1: OQS Compliance Gate** (§8.1.1a [RELEASE_MUST])

```
For each STUDIO2026_SCENARIO_1–3:
  Measure: OQS (iZotope or equivalent; must be ≥ 86 minimum per scenario, ≥ 88 ideal)
  ✓ PASS if: OQS ≥ 88 in all 3 scenarios (Excellent compliance)
  ⚠ WARN if: OQS 86–87 in 1–2 scenarios (Acceptable for lo-fi/special cases; requires documentation)
  ✗ FAIL if: OQS < 86 in any scenario (Below compliance; NO-GO)

  → Action on WARN: Document edge-case justification (e.g., "STUDIO2026_SCENARIO_2 is lo-fi folk; OQS=86.5 justified")
  → Action on FAIL: NO-GO automatically; pipeline defect (likely ML fallback triggered, defect in phases)
```

**Step 1.2: PQS MOS Gate** (Studio2026 enhancement-quality benchmark)

```
For each scenario, measure PQS (SOTA perceptual quality score, humanized as MOS 1–5 proxy):
  STUDIO2026_SCENARIO_1: Target PQS_MOS ≥ 4.5
  STUDIO2026_SCENARIO_2: Target PQS_MOS ≥ 4.3 (lo-fi grace)
  STUDIO2026_SCENARIO_3: Target PQS_MOS ≥ 4.4

  ✓ PASS if: All 3 scenarios ≥ target
  ⚠ WARN if: 1 scenario 0.2 points below target (e.g., S1 = 4.3 vs. 4.5)
  ✗ FAIL if: Any scenario ≥ 0.3 points below target (SOTA gap too large)

  → Action on PASS: Green light for enhancement quality
  → Action on WARN: Investigate whether acceptable (e.g., listener MOS corroborates; see Step 2.2)
  → Action on FAIL: NO-GO; enhancement chain not SOTA-competitive; suspect ML model degradation or phase misconfiguration
```

**Step 1.3: Artifact Freedom Veto**

```
For each scenario:
  ✗ VETO (Automatic NO-GO) if: ANY listener reports new artifact (e.g., "pumping," "distortion," "phase weirdness")
                                AND artifact ≥ 2/8 listener reports
  ✓ PASS if: ≤ 1 listener possible artifact (outlier background noise, not pipeline artifact)

  → Note: Studio2026 uses more aggressive phases (Stem Sep, Dereverb, Multiband Comp). Artifact detection must be strict.
```

### Phase 2: Listener MOS & Mode-Specific Goals

**Step 2.1: Absolute MOS Consensus (All Scenarios)**

```
For each scenario, measure listener MOS or ACR (1–5 scale):
  ✓ PASS if: Mean MOS ≥ 4.0 (Good quality) AND ≥ 6/8 individual listeners rate ≥ 3.5/5
  ⚠ WARN if: Mean MOS 3.7–3.9 (Acceptable range, but marginal)
  ✗ FAIL if: Mean MOS < 3.7 or < 6/8 at ≥ 3.5/5 (Poor acceptance, likely artifact or over-processing)

  → Action on PASS: Enhancement quality validated by humans; matches automated OQS/PQS
  → Action on WARN: Investigate listener comments for specific defect; may accept if artifact-free majority
  → Action on FAIL: NO-GO; human listeners do not accept quality level
```

**Step 2.2: Mode-Specific Studio Goals (Restoration vs. Studio2026 Scoring Diff)**

```
Studio2026 uses different scoring matrix than restoration. Listeners must rate:
- Frische / Presence (Vocal/Instruments "present, clear, not recessed")
- Punch / Bass-Kraft (Drums/Bass "defined, punchy, not loose or boomy")
- Klarheit (Spectral transparency, no holes or mud)
- Artefaktfreiheit (No musical noise, pumping, phase issues)
- Original-Preference (Would you prefer enhanced or original? Binary Y/N + confidence)

For each scenario:
  ✓ PASS if:
    • Frische ≥ 3.8/5 by ≥ 6/8 listeners (Vocal/instruments enhanced presence)
    • Punch ≥ 3.9/5 by ≥ 6/8 listeners (Dynamics restored/enhanced)
    • Klarheit ≥ 3.9/5 by ≥ 6/8 listeners (No spectral problems)
    • ≥ 6/8 listeners prefer enhanced over original (Enhancement justified)

  ⚠ WARN if: One dimension 3.5–3.8/5 (acceptable but marginal; needs rationale)
  ✗ FAIL if: Any dimension < 3.5/5 (Goal not achieved; likely phase underconfigured)
```

**Step 2.3: OQS / Listening Consistency Check** (§8.1.1a Invariant)

```
Compare machine metrics (OQS, PQS) vs. listener MOS:
  ✓ PASS if: OQS ≥ 88 AND listener MOS ≥ 4.0 (Consistent: both indicate high quality)
  ⚠ WARN if: OQS ≥ 88 BUT listener MOS 3.7–3.9 (Soft inconsistency: automated metrics optimistic; investigate listener comments for specific complaint)
  ✗ FAIL if: OQS ≥ 88 BUT listener MOS < 3.7 (Hard inconsistency: major gap; likely artifact not caught by OQS model or listener bias; BLOCKER for studio2026)
            OR OQS < 88 AND listener MOS ≥ 4.0 (OQS model defect; investigate metrics calibration)

  → Action on WARN: Acceptable with investigation; may be listener pool bias or specific artifact class
  → Action on FAIL: Release BLOCKING issue for studio2026. Stop here. Require metrics audit or re-test with different listeners.
```

### Phase 3: Per-Scenario Fine-Grained Thresholds

**Studio2026 Scenario 1 (Pop/Dance Compressed Mix)**:

```
✓ PASS if:
  • OQS ≥ 88
  • PQS_MOS ≥ 4.5
  • Frische/Presence ≥ 3.8/5 by ≥ 6/8 (Vocal sparkle enhanced)
  • Punch/Bass ≥ 4.0/5 by ≥ 6/8 (Kick: punchy, drums: lively)
  • ≥ 6/8 prefer enhanced (confidence: "definitely" or "yes")
  • Artifact veto clear

⚠ WARN if:
  • OQS 86–87 (acceptable grace if pop/dance mastered for competitive loudness)
  • Frische 3.5–3.8/5 (enhancement present but subtle; acceptable if no artifacts)
  • 5/8 prefer enhanced (marginal consensus; acceptable if strong listener agreement on quality)

✗ FAIL if:
  • PQS_MOS < 4.3 (not SOTA-competitive)
  • Punch < 3.5/5 (kick not enhanced; phase failure)
  • Artifact veto triggered
  • < 5/8 prefer enhanced (no consensus improvement)
```

**Studio2026 Scenario 2 (Lo-Fi Hip-Hop Muddy Mix)**:

```
✓ PASS if:
  • OQS ≥ 86 (lower gate for lo-fi; 86–87 acceptable)
  • PQS_MOS ≥ 4.3 (lo-fi grace: lower target)
  • Klarheit ≥ 3.8/5 by ≥ 6/8 (Mud reduced, beat clear)
  • Vocal intelligibility improved by ≥ 1.0 point on 1–5 clarity scale by ≥ 6/8
  • ≥ 5/8 prefer enhanced (lo-fi: lower preference threshold acceptable)
  • Lo-fi aesthetic preserved: "organic" not "over-processed" by ≥ 6/8
  • Artifact veto clear

⚠ WARN if:
  • OQS 85–85.9 (marginal; acceptable only if listener MOS ≥ 4.3 + artifact-free)
  • Vocal clarity improved but listeners note 1–2 artifacts (acceptable if majority: "natural overall")

✗ FAIL if:
  • PQS_MOS < 4.0 (lo-fi quality too low)
  • Klarheit < 3.5/5 (enhancement not effective; mud still present)
  • > 4/8 listeners perceive "artificial" or "over-processed" (lo-fi aesthetic violated)
  • Artifact veto triggered
```

**Studio2026 Scenario 3 (Folk Acoustic Warmth + Space)**:

```
✓ PASS if:
  • OQS ≥ 87 (acoustic recordings require high transparency)
  • PQS_MOS ≥ 4.4
  • Wärme/Tone ≥ 3.9/5 by ≥ 6/8 (Vocal warmth enhanced, not bright)
  • Raumtiefe/Space ≥ 3.8/5 by ≥ 6/8 (Spatial presence added, not artificial reverb)
  • Vocal intimacy maintained ≥ 4.0/5 by ≥ 6/8 (emotional connection intact)
  • ≥ 6/8 prefer enhanced (confidence)
  • Artifact veto clear

⚠ WARN if:
  • Wärme 3.5–3.9/5 (enhancement subtle; acceptable if acoustic transparency maintained)
  • Raumtiefe 3.5–3.8/5 (space added but cautious; acceptable if no reverb artifacts)
  • 5/8 prefer enhanced (slight consensus, but acceptable if quality metrics high)

✗ FAIL if:
  • PQS_MOS < 4.2 (quality insufficient)
  • Wärme < 3.5/5 or Raumtiefe < 3.3/5 (enhancement goals not met)
  • > 3/8 listeners perceive "too much reverb" or "robotic room" (over-processing)
  • Vocal intimacy dips below 3.5/5 (emotional disconnection; phase misconfiguration)
  • Artifact veto triggered
```

### Phase 4: Studio2026 Mode GO/NO-GO Summary

**GO (Release Blocker Clear)** if:

- Phase 1 all ✓ PASS (OQS ≥ 88, PQS gates met, Artifact veto clear)
- Phase 2 Step 2.1 ✓ PASS (Listener MOS consensus ≥ 4.0)
- Phase 2 Step 2.3 ✓ PASS (OQS/Listener consistency verified)
- Phase 3 all ✓ PASS (All per-scenario thresholds met)
- Optional ⚠ WARN on specific dimensions acceptable if documented

**NO-GO (Release Blocker Active)** if:

- Artifact veto triggered in Step 1.3 (ANY scenario)
- Phase 1 Step 1.1 ✗ FAIL (OQS < 86 any scenario)
- Phase 1 Step 1.2 ✗ FAIL (PQS_MOS gap ≥ 0.3 points any scenario)
- Phase 2 Step 2.3 ✗ FAIL (Hard OQS/Listener inconsistency) — **BLOCKING; requires metrics audit**
- Phase 3 any ✗ FAIL (Per-scenario threshold violation)

**Conditional GO (Requires Tech Lead Approval)** if:

- Phase 1 Step 1.1 ⚠ WARN (OQS 86–87 in lo-fi scenario)
- Phase 2 Step 2.1 ⚠ WARN (Listener MOS 3.7–3.9 range)
- Phase 3 ⚠ WARN flags (marginal thresholds, no failures)
- Tech Lead must document: "Tradeoff approved: [reason], quality acceptable for release"
- Requires explicit approval in PR thread before merge

---

## V. Cross-Mode Consistency Check (Both Restoration & Studio2026)

After completing both decision flows:

### Combined Gate

```
✓ RELEASE GO if:
  • Restoration mode: GO (no blockers, ≤ warnings)
  • Studio2026 mode: GO (no blockers, ≤ warnings)
  • Combined listener N ≥ 48 (8 listeners × 6 scenarios = sufficient statistical power)
  • No systemic issues spanning both modes (e.g., "all scenarios report phase artifacts")

✗ RELEASE NO-GO if:
  • Either mode: NO-GO (blocking issue)
  • Cross-scenario artifact pattern (e.g., "dropout repair fails in Restoration; Stem Sep fails in Studio2026") suggests phase execution bug
  • Listener pool bias evident (e.g., "untrained listeners rate lower; re-test with trained pool required")
```

---

## VI. Remediation & Re-Test Protocol

### If NO-GO Decision Reached

**Immediate Actions:**

1. **Root-Cause Analysis** (48 hours):
   - If artifact veto: Run `backend/core/artifact_freedom_gate._detect_*()` diagnostics on failing scenario audio
   - If OQS gate fail: Check OQS model calibration; compare to reference benchmark
   - If listener MOS low: Review listener comments verbatim; identify specific complaint (e.g., "too bright," "pumping")

2. **Phase Engineering Fix**:
   - If specific phase identified (e.g., Phase 49 Dereverb causing artifacts): Adjust `strength` parameter or dry/wet balance
   - If ML fallback triggered: Verify fallback DSP correct (e.g., OMLSA not HPSS for DeepFilterNet OOM)
   - If OQS/PQS mismatch: Check metric calibration against ground-truth SOTA samples

3. **Targeted Re-Test** (1 scenario minimum):
   - Re-run problematic scenario only with same listener pool (or 4-listener subset if time-limited)
   - Verify fix resolves specific issue without regression on other dimensions

4. **Full 6-Scenario Re-Test** (if multiple scenarios affected):
   - Execute complete hearing test suite again
   - Budget: 1–2 weeks; listener coordination required

---

## VII. Sign-Off & Documentation

### PR Reviewer Responsibility

**Template:** [PR Number] — Hearing Test Validation ([Mode]: [GO/NO-GO])

```
## Hearing Test Decision: [Mode] [GO/NO-GO]

### Objective Metrics
- OQS: [RESTORATION: N/A | STUDIO2026: Mean ≥ 88, range X–Y]
- PQS_MOS: [RESTORATION: N/A | STUDIO2026: Mean ≥ 4.3, range X–Y]
- Listener N: [8 per scenario × 6 scenarios = 48 total]

### Decision
- [GO | NO-GO | Conditional GO]
- Blocking issue(s) if NO-GO: [List]
- Conditional approval terms if Conditional GO: [List]

### Required Evidence Pack
- Blindtest-Artefakt: [Pfad]
- Head-to-Head Referenz: [Tool/System]
- Statistik: [p-Wert, Effektstaerke, 95 %-CI]
- Commit/PR-Zuordnung: [SHA/PR]

### Listener Consensus (Highlights)
- Restoration Mode: [Natürlichkeit ≥ 3.5/5: ✓; Artifact veto: ✓]
- Studio2026 Mode: [MOS ≥ 4.0: ✓; OQS/Listener consistency: ✓]

### Remediation Required (if applicable)
- [ ] Phase engineering fix: [Phase X, parameter Y change]
- [ ] Metric calibration audit: [OQS model, PQS reference pool]
- [ ] Re-test schedule: [Date, scenario(s), listener pool size]

### Approver
- Reviewer: [@github_handle]
- Date: [YYYY-MM-DD]
- Signature: [Approval]
```

---

## VIII. Appendix: Scenario Selection Rationale

**Restoration Scenarios (Carrier-Chain Inversion):**

1. **Vinyl Wear + Surface Noise (Rock Vocal)**: Most common restoration input (40% of industry volume); tests crackle removal, wow/flutter, vinyl-character preservation
2. **Tape Hiss + Oxide Dropout (Jazz Vocal)**: Tests subtle defect repair (hiss removal without "detuning") and dropout coherence; vocal-pivotal for intimacy assessment
3. **Shellac Brittleness + Click Storm (Classical Vocal)**: Tests severe degradation recovery; bandwidth extension without over-brightening; classical vocal demands authenticity

**Studio2026 Scenarios (Modern Enhancement):**

1. **Compressed Pop Mix + Thin Vocal (Pop/Dance Vocal)**: Most common modern mix defect; tests dynamic restoration, vocal presence, modern studio polish
2. **Lo-Fi Hip-Hop Muddy Mix + Weak Vocal (Hip-Hop Vocal)**: Tests clarity + artistic-aesthetic preservation (lo-fi character must survive); genre-specific goals
3. **Acoustic Folk Thin + Narrow Stereo (Folk Vocal)**: Tests subtle enhancement (warmth, space) without destroying intimacy; vocal-dependent emotional arc

**Mandatory Vocal-Inclusive Requirement (§2.36 Lyrics-Guided Enhancement):**

- All 6 scenarios include human vocal track ≥ 30% of mix duration
- Lyrics-intelligibility and vocal-presence are primary focus areas
- Emotional-arc preservation mandatory for both modes

---

## X. Release-Checkliste — Vier Bedingungen (Pfeiler 8, SOTA-Roadmap)

Aurik ist genau dann „durchgehend SOTA", wenn für den Release-Kandidaten alle
vier Bedingungen gelten (Nordstern & SOTA-Definition in
`docs/TODOS_SOTA_ROADMAP.md`). Fehlt eine Bedingung: **NO-GO**.

- [ ] **1. Never-worsen-Invarianten erfüllt** — gemessen auf dem
  eingefrorenen Referenz-Korpus (`scripts/release_candidate_pipeline.py`,
  Metrik-Report je Kandidat); jede Gate-Zahl mit Metrik-Identität
  (Wohlklang-Vertrag, Beleg 6).
- [ ] **2. Hör-Gate bestanden** — Release-Hörstichprobe (3×10 s
  Golden-Sample) **und** Kandidaten-Abnahme (2AFC,
  `scripts/mushra_harness.py thresholds-fit`); Verdikte im
  `.github/SOTA_DEFICIT_REGISTER.md` gebucht (Beleg 4 Wohlklang-Vertrag).
- [ ] **3. Fallback-Quote 0** — im Referenz-Korpus kein ML→DSP-Fallback
  (§V6 (copilot-instructions.md)-Warnungen werden gezählt, nicht nur
  geloggt; Zähler-Lane Pfeiler 5 der SOTA-Roadmap).
- [ ] **4. Provenienz vollständig** — jedes Artefakt mit (Befehl, Seed,
  Phase, Checkpoint-Hash); Trainingsläufe mit Run-ID/Lock (Pfeiler 4).

**Freigabe:** `GO` nur bei 4/4 — andernfalls `NO-GO` mit der fehlenden
Bedingung als Begründung. Keine „Beobachtung überwacht"-Ausnahme.

---

## IX. Version History

- 9.10.130 | 2026-04-07 | Initial release; 6-scenario template + GO/NO-GO decision logic

---

**Document Status**: FINAL (v10.0.0)
**Last Updated**: 2026-04-07
**Mandatory For**: Aurik 10.0.0 Release (v10.0.0.x or later)
**Review Cycle**: Quarterly (Q1–Q4 2026)
