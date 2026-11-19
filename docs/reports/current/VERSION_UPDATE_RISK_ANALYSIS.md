# Risiko-Analyse: Major Version Updates (13. Februar 2026)

> Legacy-Hinweis: Diese Analyse bezieht sich auf historische Docker-basierte Evaluationspfade.
> Der produktive Releasepfad bleibt Desktop-only und folgt dem kanonischen Vertrag
> Bridge -> `AurikDenker.denke(...)` -> `export_guard()`.

## Zusammenfassung

**EMPFEHLUNG: ⚠️ SELEKTIVES ROLLBACK ERFORDERLICH**

Von den 5 identifizierten Major Version Changes sind **2 HOCHRISKANT** und erfordern Rollback.

---

## Kritische Updates im Detail

### 🔴 **HOHE GEFAHR: deepfilternet_v3_ii**

#### Version Changes:

```
torch:       1.11.0  → 2.10.0  (99 Minor-Versionen Sprung!)
torchaudio:  0.11.0  → 2.5.1   (214 Minor-Versionen Sprung!)
numpy:       1.26.4  → 2.2.6   (Breaking Changes)
```

#### Risiko-Faktoren:

1. **PyTorch 1.11 → 2.10: KRITISCH**
   - DeepFilterNet nutzt Rust-Bindings (libDF, pyDF)
   - Rust-Code wurde gegen torch 1.11 C++ API kompiliert
   - Torch 2.x hat fundamentale C++ API Changes
   - **Wahrscheinlichkeit von Segfaults: SEHR HOCH**

2. **torchaudio 0.11 → 2.5: KRITISCH**
   - API-Breaking Changes in torchaudio.transforms
   - Resampler API komplett geändert
   - STFT/iSTFT Parameter-Änderungen

3. **NumPy 1.x → 2.x: HOCH**
   - Viele deprecated APIs entfernt
   - dtype handling geändert
   - Array creation functions geändert

#### Verwendung im System:

- ✅ **AKTIV IN PRODUKTION**
- `backend/adaptive_pipeline.py` (Hauptverwendung)
- `dsp/sota_denoiser.py`
- `backend/aurik_restore.py`
- Tests: 4 E2E-Tests
- Fallback-Chain Position: #3

#### **EMPFEHLUNG: ROLLBACK ERFORDERLICH** ⚠️

```bash
cd models/deepfilternet_v3_ii
cp requirements.txt.backup_20260213 requirements.txt
```

---

### 🟡 **MITTLERES RISIKO: flashsr**

#### Version Changes:

```
torch:         2.1.0  → 2.10.0  (9 Minor-Versionen)
accelerate:    0.21.0 → 1.12.0  (Major Version Change)
transformers:  4.30.2 → 5.1.0   (Major Version Change)
```

#### Risiko-Faktoren:

1. **Transformers 4.x → 5.x: MITTEL-HOCH**
   - Breaking Changes in AutoModel API
   - Config handling geändert
   - Tokenizer interface geändert
   - **ABER**: FlashSR nutzt relativ einfache APIs

2. **Accelerate 0.21 → 1.12: MITTEL**
   - API-Änderungen in Trainer integration
   - Mixed precision handling geändert
   - **ABER**: FlashSR nutzt basic features

3. **Torch 2.1 → 2.10: NIEDRIG**
   - Kleinerer Sprung als deepfilternet
   - Torch 2.x intern bereits stabil

#### Verwendung im System:

- ✅ **AKTIV IN PRODUKTION**
- `enhancement/hf_extender.py` (HF Extension)
- `plugins/flashsr_plugin.py`
- `src/ensemble_processor.py`
- Tests: ML Policy Engine

#### **EMPFEHLUNG: TESTEN, DANN ENTSCHEIDEN** ⚠️

**Test-Strategie:**

1. Docker Image neu bauen
2. Test mit Beispiel-Audio durchführen
3. Bei Fehler: Rollback
4. Bei Erfolg: Version beibehalten

```bash
# Test durchführen
cd models/flashsr
docker build -t flashsr:test -f Dockerfile.flashsr .
# Wenn erfolgreich: OK
# Wenn fehlgeschlagen: Rollback mit cp requirements.txt.backup_20260213 requirements.txt
```

---

## Nicht-kritische Updates

### ✅ **AKZEPTABEL: Andere Modelle**

Die meisten anderen Modelle hatten **nur Minor Updates** (z.B. torch 2.8→2.10) und sind unkritisch:

- audioldm2: torch 2.8.0 → 2.10.0 ✅
- vampnet: torch 2.0.1 → 2.10.0 ✅  
- sgmse_plus: torch 2.0.1 → 2.10.0 ✅
- resemble_enhance: torch 2.1.0 → 2.10.0 ✅

---

## Rollback-Plan

### Sofortmaßnahme: deepfilternet_v3_ii

```bash
cd /mnt/1846D15B46D139E8/Aurik_Standalone/models/deepfilternet_v3_ii
cp requirements.txt.backup_20260213 requirements.txt
echo "✅ deepfilternet_v3_ii: Rollback auf torch 1.11.0 abgeschlossen"
```

### Optional: flashsr (nur bei Problemen)

```bash
cd /mnt/1846D15B46D139E8/Aurik_Standalone/models/flashsr
cp requirements.txt.backup_20260213 requirements.txt
echo "✅ flashsr: Rollback auf torch 2.1.0 abgeschlossen"
```

---

## Test-Plan für flashsr

### Quick-Test (5 Minuten)

```bash
cd /mnt/1846D15B46D139E8/Aurik_Standalone

# Docker Image bauen
cd models/flashsr
docker build -t flashsr:test -f Dockerfile.flashsr . 2>&1 | tee build.log

# Bei Build-Fehler → ROLLBACK
if [ $? -ne 0 ]; then
    echo "❌ Build fehlgeschlagen - Rollback erforderlich"
    cp requirements.txt.backup_20260213 requirements.txt
    exit 1
fi

# Test-Inferenz
python3 << 'PYTHON'
from plugins.flashsr_plugin import FlashSRPlugin
import tempfile
import numpy as np
import soundfile as sf

# Dummy-Audio erstellen (1 Sekunde, 16kHz)
dummy = np.random.randn(16000).astype(np.float32) * 0.01
with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
    sf.write(f.name, dummy, 16000)
    input_path = f.name

with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
    output_path = f.name

try:
    plugin = FlashSRPlugin(docker_image="flashsr:test")
    plugin.process(input_path, output_path)
    print("✅ FlashSR Test erfolgreich!")
except Exception as e:
    print(f"❌ FlashSR Test fehlgeschlagen: {e}")
    print("⚠️ Rollback empfohlen!")
PYTHON
```

### Erweiteter Test (15 Minuten)

```bash
# E2E Test mit echtem Audio
cd /mnt/1846D15B46D139E8/Aurik_Standalone
pytest tests/test_ml_policy_engine.py::TestMLPolicyEngine::test_super_resolution_selects_flashsr -v
```

---

## Zusammenfassung & Aktionsplan

### ✅ **SOFORT DURCHFÜHREN:**

1. **deepfilternet_v3_ii Rollback** (KRITISCH)

   ```bash
   cd models/deepfilternet_v3_ii
   cp requirements.txt.backup_20260213 requirements.txt
   ```

### ⚠️ **TESTEN & ENTSCHEIDEN:**

2. **flashsr Test durchführen**
   - Docker Build testen
   - Bei Fehler: Rollback
   - Bei Erfolg: Version beibehalten

### ✅ **BEHALTEN:**

3. **Alle anderen Modelle** (audioldm2, vampnet, sgmse_plus, ...) können die neuen Versionen behalten

---

## Langfristige Strategie

### DeepFilterNet Modernisierung (Optional)

Falls du perspektivisch torch 2.x für deepfilternet nutzen möchtest:

1. **Alternative:** DeepFilterNet neu kompilieren

   ```bash
   # Im Dockerfile:
   # rustup install 1.75.0  # Neuere Rust-Version
   # Rebuild mit torch 2.10 headers
   ```

2. **Alternative:** Auf neuere DeepFilterNet-Version wechseln

   - Check GitHub: [Rikorose/DeepFilterNet](https://github.com/Rikorose/DeepFilterNet)
   - Neuere Releases unterstützen möglicherweise torch 2.x

3. **Alternative:** Modell in isolation lassen
   - Eigenes Docker Image mit frozen dependencies
   - Separate virtual environment

---

## Fazit

**ANTWORT AUF DEINE FRAGE:**  
> "Lassen sich diese Updates ohne negative Folgen ausführen?"

**NEIN** - Nicht alle Updates sind sicher:

- ❌ **deepfilternet_v3_ii**: torch 1.11→2.10 ist **HOCHRISKANT** → **ROLLBACK ERFORDERLICH**
- ⚠️ **flashsr**: transformers 4→5 ist **MÄSZIG RISKANT** → **TESTEN EMPFOHLEN**
- ✅ **Andere Modelle**: Minor Updates sind **SICHER** → **BEHALTEN**

**Nächste Schritte:**

1. Rollback für deepfilternet_v3_ii durchführen
2. flashsr testen (optional)
3. Dokumentation aktualisieren
