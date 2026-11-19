# GitHub Labels Configuration for Aurik 10.0.0

This file documents the label structure used in the Aurik project for issue and PR management.

## How to Apply Labels

### Automatic (via GitHub CLI)

```bash
# Install GitHub CLI if not already installed
# https://cli.github.com/

# Navigate to repository
cd Aurik_Standalone

# Apply all labels (requires authentication)
gh label create "bug" --color "d73a4a" --description "Something isn't working"
gh label create "enhancement" --color "a2eeef" --description "New feature or request"
gh label create "performance" --color "fbca04" --description "Performance optimization or issue"
gh label create "documentation" --color "0075ca" --description "Documentation improvement"
gh label create "triage" --color "ffffff" --description "Needs initial review and classification"
gh label create "good first issue" --color "7057ff" --description "Good for newcomers"
gh label create "help wanted" --color "008672" --description "Extra attention is needed"
gh label create "priority: critical" --color "b60205" --description "Must fix immediately"
gh label create "priority: high" --color "d93f0b" --description "Should fix soon"
gh label create "priority: medium" --color "fbca04" --description "Fix when possible"
gh label create "priority: low" --color "0e8a16" --description "Nice to have"
gh label create "area: dsp" --color "c5def5" --description "DSP algorithms and processing"
gh label create "area: ml" --color "c5def5" --description "Machine Learning models"
gh label create "area: gui" --color "c5def5" --description "Graphical User Interface"
gh label create "area: cli" --color "c5def5" --description "Command Line Interface"
gh label create "area: api" --color "c5def5" --description "API and core architecture"
gh label create "area: testing" --color "c5def5" --description "Testing and QA"
gh label create "area: ci-cd" --color "c5def5" --description "CI/CD and automation"
gh label create "status: in-progress" --color "0052cc" --description "Currently being worked on"
gh label create "status: blocked" --color "e99695" --description "Blocked by dependency or decision"
gh label create "status: needs-discussion" --color "d4c5f9" --description "Requires community discussion"
gh label create "quality: regression" --color "e11d21" --description "Broken functionality that worked before"
gh label create "quality: crash" --color "e11d21" --description "Application crashes"
gh label create "quality: audio" --color "1d76db" --description "Audio quality or artifact issues"
gh label create "duplicate" --color "cfd3d7" --description "This issue or PR already exists"
gh label create "wontfix" --color "ffffff" --description "This will not be worked on"
gh label create "invalid" --color "e4e669" --description "Invalid issue or PR"
```

### Manual (via GitHub Web UI)

1. Go to: Repository → Issues → Labels
2. Create each label with the specified color and description
3. Apply to issues/PRs as needed

## Label Categories

### Type Labels (Primary Classification)

| Label | Color | Description | Usage |
|-------|-------|-------------|-------|
| `bug` | `#d73a4a` | Something isn't working | Any unexpected behavior, crashes, errors |
| `enhancement` | `#a2eeef` | New feature or request | New features, improvements to existing features |
| `performance` | `#fbca04` | Performance optimization or issue | Slow processing, high memory usage, RT factor issues |
| `documentation` | `#0075ca` | Documentation improvement | Missing, incorrect, or unclear documentation |

### Priority Labels

| Label | Color | Description | When to Use |
|-------|-------|-------------|-------------|
| `priority: critical` | `#b60205` | Must fix immediately | Crashes, data loss, security issues |
| `priority: high` | `#d93f0b` | Should fix soon | Significant impact on workflow |
| `priority: medium` | `#fbca04` | Fix when possible | Moderate impact, workarounds available |
| `priority: low` | `#0e8a16` | Nice to have | Minor improvements, cosmetic issues |

### Area Labels (Component-Based)

| Label | Color | Description | Component |
|-------|-------|-------------|-----------|
| `area: dsp` | `#c5def5` | DSP algorithms and processing | core/, dsp/, processing/ |
| `area: ml` | `#c5def5` | Machine Learning models | ML models, UnifiedRestorerV3 |
| `area: gui` | `#c5def5` | Graphical User Interface | aurik_90/ui/ |
| `area: cli` | `#c5def5` | Command Line Interface | aurik_cli.py, batch_processor.py |
| `area: api` | `#c5def5` | API and core architecture | Architecture, RestorationConfig |
| `area: testing` | `#c5def5` | Testing and QA | tests/, benchmarks/ |
| `area: ci-cd` | `#c5def5` | CI/CD and automation | .github/workflows/ |

### Status Labels (Workflow)

| Label | Color | Description | When to Apply |
|-------|-------|-------------|---------------|
| `triage` | `#ffffff` | Needs initial review and classification | Automatically applied to new issues |
| `status: in-progress` | `#0052cc` | Currently being worked on | When work begins |
| `status: blocked` | `#e99695` | Blocked by dependency or decision | Waiting on external factor |
| `status: needs-discussion` | `#d4c5f9` | Requires community discussion | Complex decisions, design questions |

### Quality Labels (Specific Issue Types)

| Label | Color | Description | Usage |
|-------|-------|-------------|-------|
| `quality: regression` | `#e11d21` | Broken functionality that worked before | Features that stopped working |
| `quality: crash` | `#e11d21` | Application crashes | Crashes, segfaults, fatal errors |
| `quality: audio` | `#1d76db` | Audio quality or artifact issues | Distortion, artifacts, quality degradation |

### Community Labels

| Label | Color | Description | Usage |
|-------|-------|-------------|-------|
| `good first issue` | `#7057ff` | Good for newcomers | Easy issues for new contributors |
| `help wanted` | `#008672` | Extra attention is needed | Need community help or expertise |

### Resolution Labels

| Label | Color | Description | Usage |
|-------|-------|-------------|-------|
| `duplicate` | `#cfd3d7` | This issue or PR already exists | Link to original issue, close |
| `wontfix` | `#ffffff` | This will not be worked on | Out of scope, by design |
| `invalid` | `#e4e669` | Invalid issue or PR | Not a bug, user error, unclear |

## Label Application Guidelines

### For Bug Reports

```
Required: bug, triage
Add: priority: [critical/high/medium/low]
Add: area: [dsp/ml/gui/cli/api]
Optional: quality: [regression/crash/audio]
```

**Example:** Bug in ML model inference

```
Labels: bug, triage, priority: high, area: ml
```

### For Feature Requests

```
Required: enhancement, triage
Add: priority: [high/medium/low]
Add: area: [dsp/ml/gui/cli/api]
Optional: status: needs-discussion
```

**Example:** Request for new DSP algorithm

```
Labels: enhancement, triage, priority: medium, area: dsp
```

### For Performance Issues

```
Required: performance, triage
Add: priority: [high/medium/low]
Add: area: [dsp/ml/gui/cli] (where slowness occurs)
```

**Example:** Slow GUI responsiveness

```
Labels: performance, triage, priority: high, area: gui
```

### For Documentation Issues

```
Required: documentation, triage
Add: priority: [medium/low]
Optional: good first issue (if simple fix)
```

**Example:** Missing API documentation

```
Labels: documentation, triage, priority: medium, area: api
```

## Label Workflow

### New Issue Workflow

1. **Auto-apply:** `triage` (via template)
2. **Maintainer reviews:**
   - Verify issue type label (bug/enhancement/performance/documentation)
   - Add priority label
   - Add area label(s)
   - Add specific labels (quality, status) if needed
   - Remove `triage` label
3. **If assigned:** Add `status: in-progress`
4. **If resolved:** Close issue, optionally add `duplicate`, `wontfix`, or `invalid`

### Pull Request Workflow

1. **PR creator:** Add relevant area labels
2. **Maintainer:** Add priority if urgent
3. **During review:** Add `status: needs-discussion` if needed
4. **After merge:** Close, auto-links to issues via keywords

## Color Scheme

**Color Palette:**

- Red tones (`#b60205`, `#d73a4a`, `#e11d21`): Urgent, critical, bugs
- Orange/Yellow (`#d93f0b`, `#fbca04`): Warnings, performance
- Blue tones (`#0075ca`, `#1d76db`, `#c5def5`): Information, areas
- Green (`#0e8a16`, `#008672`): Low priority, help wanted
- Purple (`#7057ff`, `#d4c5f9`): Community, discussion
- Gray (`#cfd3d7`, `#ffffff`): Resolution, triage

## Best Practices

1. **One Type Label:** Each issue should have exactly one primary type label (bug/enhancement/performance/documentation)
2. **One Priority Label:** Assign one priority label after triage
3. **Multiple Area Labels:** OK if issue spans multiple components
4. **Update Status:** Keep status labels current as work progresses
5. **Remove Triage:** Always remove `triage` after initial classification
6. **Good First Issue:** Mark beginner-friendly issues to encourage contributions

## Automation

### GitHub Actions (Future Enhancement)

```yaml
# Auto-label based on file changes
- uses: actions/labeler@v4
  with:
    repo-token: "${{ secrets.GITHUB_TOKEN }}"
    configuration-path: .github/labeler.yml

# Auto-add triage to new issues
- uses: actions/github-script@v6
  with:
    script: |
      github.rest.issues.addLabels({
        issue_number: context.issue.number,
        owner: context.repo.owner,
        repo: context.repo.repo,
        labels: ['triage']
      })
```

## Statistics & Metrics

Track label usage to understand project health:

- **Bug density:** Count of `bug` labels vs total issues
- **Performance issues:** Track `performance` labels over time
- **Community engagement:** `help wanted` + `good first issue` activity
- **Backlog health:** Count of `triage` labels (should be low)

---

**Last Updated:** 16. Februar 2026  
**Maintainers:** See CONTRIBUTING.md  
**Feedback:** Open an issue with `documentation` label
