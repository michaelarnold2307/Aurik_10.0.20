# Issue Management Guide

**Aurik 10.0.0 - Issue Tracking & Community Feedback**  
Documentation for maintainers and contributors

---

## 📋 Overview

Aurik uses GitHub Issues for bug tracking, feature requests, performance issues, and documentation improvements. This guide covers:

- How to create effective issues
- Issue triage process
- Label management
- Resolution workflows

**For Users:** How to report bugs and request features  
**For Contributors:** How to find and work on issues  
**For Maintainers:** How to manage and triage issues

---

## 🐛 Reporting Issues

### Before Creating an Issue

1. **Search existing issues** to avoid duplicates
2. **Check documentation** (README, docs/, CHANGELOG)
3. **Update to latest version** if possible
4. **Gather information:**
   - Aurik version
   - Operating system
   - Audio file format/specs
   - Error messages or logs
   - Steps to reproduce

### Issue Types

#### Bug Report 🐛

Use when Aurik is not working as expected.

**Good Bug Reports:**

- Clear title: `[Bug]: Crash when processing 32-bit WAV files`
- Reproducible steps
- Expected vs actual behavior
- System information
- Log output

**Template:** [bug_report.yml](ISSUE_TEMPLATE/bug_report.yml)

#### Feature Request ✨

Use to suggest new features or enhancements.

**Good Feature Requests:**

- Clear problem statement
- Proposed solution
- Use cases and examples
- Priority indication

**Template:** [feature_request.yml](ISSUE_TEMPLATE/feature_request.yml)

#### Performance Issue ⚡

Use when processing is slower than expected.

**Good Performance Reports:**

- Specific metrics (RT factor, memory usage)
- Audio file specifications
- System specifications
- Processing mode used

**Template:** [performance_issue.yml](ISSUE_TEMPLATE/performance_issue.yml)

#### Documentation Issue 📚

Use for documentation problems.

**Good Documentation Reports:**

- Specific file/page location
- What's wrong or missing
- Suggested improvement

**Template:** [documentation.yml](ISSUE_TEMPLATE/documentation.yml)

---

## 🏷️ Label System

See [LABELS.md](LABELS.md) for complete label reference.

### Quick Label Guide

**Type Labels (choose one):**

- `bug` - Something isn't working
- `enhancement` - New feature or request
- `performance` - Performance optimization
- `documentation` - Documentation issue

**Priority Labels (choose one):**

- `priority: critical` - Must fix immediately
- `priority: high` - Should fix soon
- `priority: medium` - Fix when possible
- `priority: low` - Nice to have

**Area Labels (can have multiple):**

- `area: dsp` - DSP algorithms
- `area: ml` - Machine Learning
- `area: gui` - Graphical Interface
- `area: cli` - Command Line
- `area: api` - Core architecture
- `area: testing` - Tests and QA
- `area: ci-cd` - CI/CD automation

---

## 🔄 Issue Lifecycle

### 1. **Triage** (Initial Review)

New issues get `triage` label automatically.

**Maintainer Actions:**

1. Verify issue is valid (not duplicate, has enough info)
2. Add appropriate labels:
   - Type label (bug/enhancement/performance/documentation)
   - Priority label
   - Area label(s)
3. Remove `triage` label
4. Request more info if needed (keep `triage` if waiting)

**Timelines:**

- Critical bugs: Triage within 24 hours
- Other issues: Triage within 1 week

### 2. **Investigation** (For Bugs)

**Maintainer/Contributor Actions:**

1. Reproduce the issue
2. Identify root cause
3. Comment findings on issue
4. Add `status: in-progress` if working on fix
5. Add `status: blocked` if waiting on dependency

### 3. **Discussion** (For Features/Enhancements)

**Community Process:**

1. Add `status: needs-discussion` label
2. Community discusses in comments
3. Maintainer makes decision:
   - Accept → Keep open, plan implementation
   - Decline → Close with `wontfix`, explain why
4. If accepted, add to roadmap or milestone

### 4. **Implementation**

**Workflow:**

1. Assign issue to contributor
2. Add `status: in-progress` label
3. Contributor creates PR referencing issue (`Fixes #123`)
4. PR review process
5. Merge PR → Issue auto-closes

### 5. **Resolution**

**Close With:**

- **Fixed** - PR merged, issue resolved
- `duplicate` - Link to original issue
- `wontfix` - Out of scope, won't implement
- `invalid` - Not a bug, user error, unclear

**Always:**

- Comment why issue is being closed
- Link to related issues/PRs
- Thank reporter for contribution

---

## 👥 Roles & Permissions

### Users (Everyone)

✅ Create issues  
✅ Comment on issues  
✅ Vote (👍 reactions)  
❌ Edit labels  
❌ Close issues

### Contributors (With merged PRs)

✅ All user permissions  
✅ Self-assign issues marked `good first issue`  
✅ Request reviews  
❌ Edit labels (request via comment)  
❌ Close others' issues

### Maintainers (Core Team)

✅ All contributor permissions  
✅ Edit labels  
✅ Close any issue  
✅ Assign issues to others  
✅ Manage milestones

---

## 🎯 Finding Issues to Work On

### For New Contributors

**Look for:**

- `good first issue` - Beginner-friendly
- `help wanted` - Need assistance
- `documentation` - Often easier to start

**Filter Example:**

```
is:open is:issue label:"good first issue" no:assignee
```

### For Experienced Contributors

**Look for:**

- `priority: high` - Important issues
- Your expertise area: `area: ml`, `area: dsp`, etc.
- Performance optimizations

**Filter Example:**

```
is:open is:issue label:"priority: high" label:"area: ml" no:assignee
```

### Claiming an Issue

1. Comment: "I'd like to work on this issue"
2. Wait for maintainer confirmation
3. Self-assign (if you have permission) or ask maintainer
4. Update within 2 weeks or issue may be reassigned

---

## 📊 Issue Metrics

### Health Indicators

**Good:**

- Low `triage` count (<5)
- `priority: critical` addressed quickly (<48 hours)
- High percentage of closed issues
- Active community discussion

**Needs Attention:**

- Many old `triage` issues (>1 month)
- Critical bugs open >1 week
- Low contributor activity
- Duplicate issues (search not working?)

### Regular Reviews

**Weekly:**

- Triage new issues
- Check `priority: critical` and `priority: high`
- Respond to questions/comments

**Monthly:**

- Review all open issues
- Close stale issues (inactive >3 months)
- Update priorities
- Plan next release based on issue trends

---

## 🔧 Maintainer Tools

### GitHub CLI Commands

**List issues needing triage:**

```bash
gh issue list --label "triage" --limit 50
```

**Assign issue to yourself:**

```bash
gh issue edit 123 --add-assignee "@me"
```

**Add labels:**

```bash
gh issue edit 123 --add-label "bug,priority: high,area: ml"
```

**Close issue:**

```bash
gh issue close 123 --reason "completed" -c "Fixed in PR #456"
```

### Automation Ideas

**Stale Issue Bot:**

```yaml
- name: Close stale issues
  uses: actions/stale@v8
  with:
    days-before-stale: 90
    days-before-close: 14
    stale-issue-message: 'This issue is stale. Please confirm if still relevant.'
```

**Auto-label by file:**

```yaml
# .github/labeler.yml
'area: ml':
  - core/ml_hybrid/**
  - core/unified_restorer_v3.py

'area: gui':
  - aurik_90/ui/**
```

---

## 💡 Best Practices

### For Issue Reporters

**Do:**
✅ Search before creating  
✅ Use templates  
✅ Provide complete information  
✅ Be respectful  
✅ Follow up if more info is requested  

**Don't:**
❌ Demand immediate fixes  
❌ Bump issues repeatedly  
❌ Post "+1" comments (use 👍 reaction instead)  
❌ Hijack unrelated issues  

### For Maintainers

**Do:**
✅ Respond promptly (at least acknowledge)  
✅ Be kind and professional  
✅ Close duplicates with links  
✅ Keep labels current  
✅ Thank contributors  

**Don't:**
❌ Let `triage` pile up  
❌ Leave critical bugs open  
❌ Ghost reporters  
❌ Close without explanation  

### For Contributors

**Do:**
✅ Ask before starting work  
✅ Link issue in PR  
✅ Update if you can't finish  
✅ Be patient during review  

**Don't:**
❌ Work on claimed issues  
❌ Submit PRs without related issue  
❌ Abandon work without notice  

---

## 📞 Getting Help

**For users:**

- 🐛 Bug? Create issue with `bug` template
- 💬 Question? Use GitHub Discussions
- 📖 Documentation issue? Use `documentation` template

**For contributors:**

- Need help? Comment on issue with questions
- Want to take on issue? Comment to claim
- PR help? Tag maintainers in review

**For maintainers:**

- Escalation? Use `status: needs-discussion` label
- Complex decision? Discuss in team channel
- Need more maintainers? Post in Discussions

---

## 📈 Success Metrics

Track these to measure issue management health:

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Triage Time | <1 week | Time from creation to triage complete |
| Critical Bug Resolution | <48 hours | Time from creation to fix |
| Issue Close Rate | >70% | Closed / (Closed + Open) |
| Stale Issues | <10% | Issues inactive >3 months |
| Community Response | <24 hours | Time to first response |
| Duplicate Rate | <5% | Duplicates / Total issues |

---

## 🎓 Resources

**Templates:**

- [bug_report.yml](ISSUE_TEMPLATE/bug_report.yml)
- [feature_request.yml](ISSUE_TEMPLATE/feature_request.yml)
- [performance_issue.yml](ISSUE_TEMPLATE/performance_issue.yml)
- [documentation.yml](ISSUE_TEMPLATE/documentation.yml)

**Guides:**

- [LABELS.md](LABELS.md) - Complete label reference
- [CONTRIBUTING.md](../CONTRIBUTING.md) - Contribution guidelines
- [docs/aurik9_roadmap.md](../docs/aurik9_roadmap.md) - Project roadmap

**GitHub Docs:**

- [About Issues](https://docs.github.com/en/issues/tracking-your-work-with-issues/about-issues)
- [Issue Templates](https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests)
- [Labels](https://docs.github.com/en/issues/using-labels-and-milestones-to-track-work/managing-labels)

---

**Last Updated:** 16. Februar 2026  
**Version:** 1.0  
**Maintainers:** See CONTRIBUTING.md  
**Feedback:** Open an issue with `documentation` label
