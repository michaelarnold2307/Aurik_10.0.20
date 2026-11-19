# Decision: No VST3/DAW Plugin Development

**Date:** 9. Februar 2026  
**Status:** ❌ VST3/AU Plugin development cancelled  
**Alternative:** ✅ Standalone Desktop Application (Phase 2.4)

---

## Executive Summary

Nach Analyse der VST3-Lizenzierung (Steinberg) und der damit verbundenen Abhängigkeitsrisiken wurde entschieden, **KEINE Plugin-Entwicklung** zu verfolgen. AURIK bleibt vollständig unabhängig als eigenständige Desktop-Anwendung.

---

## Reasons for Decision

### 1. Licensing Dependency Risk

```
VST3 SDK:
  - Owned by: Steinberg (Yamaha)
  - License: Dual (GPLv3 OR Proprietary Agreement)
  - Risk: License terms can change at any time
  - Control: Outside our control
```

**Problem:** Steinberg könnte jederzeit:
- Lizenzgebühren einführen
- Nutzungsbedingungen ändern
- SDK-Zugang einschränken
- AURIK-spezifische Restriktionen auferlegen

### 2. Independence > Features

```
Core Value: Full Control über die Software
  ✅ No external dependencies
  ✅ No licensing negotiations
  ✅ No approval processes
  ✅ No trademark restrictions
```

**Principle:** Software-Unabhängigkeit ist wichtiger als Plugin-Integration

### 3. Desktop App Sufficiency

```
Current Status:
  ✅ Professional Desktop Application (2,482+ lines)
  ✅ Batch Processing for studio workflows
  ✅ Complete AURIK feature set
  ✅ Standalone executable
  ✅ Cross-platform (Linux, planned: Windows/macOS)
```

**Conclusion:** Desktop App erfüllt alle Use Cases

---

## Alternative Considered: CLAP

**CLAP (Clever Audio Plugin):**
- ✅ Fully Open Source (MIT License)
- ✅ No licensing dependencies
- ✅ Modern architecture
- ⚠️ Limited DAW support (noch nicht weit verbreitet)

**Decision:** Even CLAP not pursued - Standalone app preferred

---

## Impact on Roadmap

### Removed:

- ❌ VST3 Plugin Development (6-8 weeks, +2-3 points)
- ❌ AU Plugin Development
- ❌ AAX Plugin Development
- ❌ JUCE Framework integration
- ❌ pybind11 Python bridge

### Kept/Enhanced:

- ✅ Standalone Desktop Application (already complete)
- ✅ Batch Processing (already implemented)
- ✅ Professional UI (already complete)
- ✅ Preset Management (10 factory + user presets)

**Net Impact:** 0 points lost (Desktop App replaces all plugin functionality)

---

## User Workflow Comparison

### With Plugin (cancelled):

```
DAW (Logic Pro)
  → Add AURIK Plugin to track
  → Real-time processing
  → Parameter automation
```

### With Desktop App (current):

```
AURIK Professional
  → Load audio file(s)
  → Batch processing queue
  → Export processed files
  → Import back to DAW (if needed)
```

**Trade-off:**
- 🔴 No real-time in-DAW processing
- 🟢 Full independence and control
- 🟢 No licensing risks
- 🟢 No external approvals needed

---

## Technical Details

### VST3 Licensing Terms (Reference)

```
Option 1: GPLv3
  - Free for open source projects
  - Requires GPL-compatible license for AURIK plugin
  - Source code must be published

Option 2: Steinberg Proprietary License Agreement
  - Free (no royalties)
  - Requires agreement signature
  - Approval process (2-7 days)
  - Terms can change
```

**Risk Assessment:**
- Current: Free
- Future: Uncertain (Steinberg can change terms)
- Mitigation: None (outside our control)

---

## Alternatives Evaluated

### Option 1: VST3 + AU Plugins ❌

- **Pros:** Professional integration, real-time processing
- **Cons:** Licensing dependency, approval processes
- **Decision:** Rejected (independence priority)

### Option 2: CLAP Plugin ❌

- **Pros:** Open source, no licensing issues
- **Cons:** Limited DAW support, still external dependency
- **Decision:** Rejected (unnecessary complexity)

### Option 3: Standalone Desktop App ✅

- **Pros:** Full control, no dependencies, already complete
- **Cons:** No in-DAW integration
- **Decision:** SELECTED (matches project values)

---

## Documentation Updates

### Files Modified:

1. `Finalisierungs_Roadmap.md`
   - VST3/Plugin references removed
   - Section 10 marked as "REMOVED FROM ROADMAP"
   - Timeline adjusted

2. `docs/DAW_Plugin_Architecture.md`
   - Marked as DEPRECATED
   - Kept for reference only

3. `docs/DECISION_NO_VST3_PLUGIN.md` (this file)
   - Decision documentation
   - Rationale and alternatives

### Desktop App References:

- ✅ `docs/Desktop_App_Final_Status.md` - Complete implementation
- ✅ `docs/Desktop_App_Plan.md` - Original strategy
- ✅ `docs/Packaging_Documentation.md` - Executable distribution

---

## Future Considerations

### If Conditions Change:

```
IF:
  - Fully open-source plugin standard emerges
  - No licensing dependencies
  - Wide DAW adoption

THEN:
  - Reconsider plugin development
  - Evaluate CLAP maturity
```

**Current Status:** No plans to revisit

### Enhancement Priorities:

1. **Package Optimization** (7 GB → 1.5 GB)
2. **ML Models** (MERT, AST, madmom)
3. **Extended Testing** (A/B vs iZotope RX)
4. **Multi-platform Builds** (Windows, macOS)

---

## Conclusion

**AURIK bleibt unabhängig.**

Die Entscheidung gegen Plugin-Entwicklung schützt die Unabhängigkeit des Projekts und vermeidet Risiken durch externe Lizenz-Kontrolle. Die Desktop Application bietet bereits alle benötigten Funktionen ohne Kompromisse.

**Values > Features**

---

**Status:** ✅ Decision final  
**Alternative:** Desktop App (Phase 2.4) fully replaces plugin functionality  
**Points:** 190/100 (unaffected by this decision)
