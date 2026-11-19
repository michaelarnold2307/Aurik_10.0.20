# Aurik — Wissenschaftlicher Literatur-Index

> **Zitierte Werke aus dem Aurik-Codebase.**
> Jeder Eintrag referenziert die Code-Stelle, die ihn zitiert.
> Dieses Dokument ist der wissenschaftliche Unterbau von Auriks
> Restaurations-Entscheidungen — warum ein Threshold 0.22 und nicht 0.30 ist.

---

## Akustik & Psychoakustik

| Referenz | Thema | Code-Referenz |
|---|---|---|
| Zwicker & Fastl (1990) *Psychoacoustics: Facts and Models* | Bark-Skala, Maskierung, Lautheit | `per_phase_musical_goals_gate.py`, `psychoacoustic_artifact_detector.py` |
| Aures (1985) | Rauigkeit (Roughness) | `musical_goals_metrics.py` |
| Moore & Glasberg (1996) | ERB-Skala, Frequenzgruppen | `perceptual_salience.py` |
| Ando (1998) *Architectural Acoustics* | IACC, Räumlichkeit | `musical_goals_metrics.py` |
| Beranek (2004) *Concert Halls and Opera Houses* | RT60, DRR | `unified_restorer_v3.py §2.46f` |
| Begault (1994) *3-D Sound for Virtual Reality* | HRTF, Spatial Depth | `musical_goals_metrics.py` |

## Audio-Restauration (Analog)

| Referenz | Thema | Code-Referenz |
|---|---|---|
| **IEC 60386:1987** | Wow/Flutter-Messnorm (WOW < 0.5 Hz, FLUTTER 0.5–200 Hz) | `defect_scanner.py:29`, `phase_12_wow_flutter_fix.py` |
| Janssen, Veldhuis & Vries (1986) IEEE TASLP 34:203 | Click-Detektion (autoregressives Modell) | `defect_scanner.py` |
| Bailey, Casebeer & Fazekas (2019) AES 147th Conv. | Vinyl-Crackle-Klassifikation (Deep Learning) | `defect_scanner.py` |
| Maher (2010) J. Audio Eng. Soc. 58:702 | Survey analoger Artefakt-Erkennung | `medium_detector.py` |
| Dahimene et al. (2008) | Dropout-Detektion in Bandaufnahmen | `phase_24_dropout_repair.py` |
| Cartwright, Pardo & Wallis (2016) DAFX-16 | Vinyl-ID via spektrale Features | `medium_detector.py` |
| Declercq, De Backer & Zhu (2007) ICASSP | Bayesian-Trägerklassifikation (Gaussian-Mixture) | `medium_detector.py` |
| Hess (1988) | Acetat-Zersetzung | `defect_scanner.py` |
| Dolby (1967) | Dolby-Rauschunterdrückung (A/B/C/S) | `phase_29_tape_hiss_reduction.py` |
| Bitto (2000) AES Conv. 109 | Jitter-Artefakte in D/A-Wandlung | `defect_scanner.py` |

## Audio-Restauration (Digital/Codec)

| Referenz | Thema | Code-Referenz |
|---|---|---|
| Brandenburg & Bosi (1994) J. AES 42:381 | MP3/MPEG-1 Layer III Codec-Artefakte | `medium_detector.py` |
| Pan (1995) J. AES 43:529 | AAC/MPEG-2 Codec-Charakteristika | `medium_detector.py` |
| Herre & Johnston (1996) AES Conv. 101 | MP3/AAC Pre-Echo (Temporal Masking) | `defect_scanner.py` |
| Müller & Ewert (2011) IEEE Signal Proc. Mag. 28:42 | MDCT-Codec-Fingerprinting | `medium_detector.py` |
| Spijkervet & Haasdijk (2020) ISMIR | ML-basierte MP3/AAC-Unterscheidung | `medium_detector.py` |

## Maschinelles Lernen & Signalverarbeitung

| Referenz | Thema | Code-Referenz |
|---|---|---|
| Défossez et al. (2022) | Demucs (Hybrid Transformer + Waveform) | `phase_32_mono_to_stereo.py` |
| Défossez et al. (2020) | Denoiser (DEMUCS/DeepNoise) | `phase_03_denoise.py` |
| Schroff et al. (2015) | FaceNet/Triplet-Loss (d-Vektor Sprecher-Embedding) | `resemblyzer_plugin.py` |
| Kong et al. (2020) | PANNs (Large-Scale Pretrained Audio Neural Networks) | `panns_plugin.py` |
| Radford et al. (2023) | Whisper (Robust Speech Recognition) | `phase_58_lyrics_guided_enhancement.py` |

## Musik-Informatik

| Referenz | Thema | Code-Referenz |
|---|---|---|
| Bello et al. (2005) | Onset Detection | `musical_goals_metrics.py` |
| McFee et al. (2015) *librosa* | Audio-Analyse-Toolkit | Diverse Phasen |
| Müller (2015) *Fundamentals of Music Processing* | Chroma, DTW, Tempogram | `groove_metric.py` |
| Raffel et al. (2014) | Onset/Beat-Tracking | `musical_goals_metrics.py` |

## Normen & Standards

| Referenz | Thema | Code-Referenz |
|---|---|---|
| **IEC 60386:1987** | Wow/Flutter-Messung | `defect_scanner.py` |
| **ISO 11172-3** | MPEG-1 Layer III (vereinfachtes Maskierungsmodell) | `defect_precision_enhancer.py` |
| **ITU-R BS.1770-4** | LUFS-Lautheitsmessung | `phase_40_loudness_normalization.py` |
| **EBU R128** | Loudness-Normalisierung | `phase_40_loudness_normalization.py` |
| **AES17-2015** | Dynamikbereich-Messung | `phase_26_dynamic_range_expansion.py` |
| **IEC 61672-1** | Schallpegelmesser (SPL-Proxy) | `musical_goals_metrics.py` |

---

## Wie neue Literatur hinzufügen

1. Im Code: `# Müller & Ewert (2011) IEEE Signal Proc. Mag. 28:42 — Thema`
2. In diesem Dokument: Eintrag unter der passenden Kategorie
3. Format: `Autor (Jahr) Publikation — Thema`

**Prinzip:** Jede numerische Konstante mit wissenschaftlicher Herkunft MUSS
mit einem Literatureintrag verknüpft sein. Weltklasse-Restauration braucht
zitierbare Grundlagen.
