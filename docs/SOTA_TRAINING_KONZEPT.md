Everything needed for the "Hörprobe (audition sample)" section is in place:

The guide docs/reports/current/2026-10-07_hoerprobe_anleitung.md covers both SCNet C4 and HR-V1 W-1.
Packages: output_audio/mushra/thresholds_scnet_c4_fair_r2 (SCNet round 2, 12 trials + 2 traps), output_audio/mushra/thresholds_wohlklang_w1_hrv1/ (HR-V1).
SGMSE A/B: no .wav output found — only a metric JSON report; the audio is generated on-the-fly by the benchmark script. In other words, for the SGMSE 3-way hearing acceptance test (Hörabnahme), the package needs to be built (that's part of KP-1).
Now I have everything I need. Time to write docs/SOTA_TRAINING_KONZEPT.md. Let me carefully draft it — comprehensive yet concise, in German, with § citations that include their sources, and a live progress section (§0) for resuming after an interruption.

Main content plan (final version):

# SOTA Implementation / Training / Fine-tuning Concept — Aurik 10
> Purpose + normative anchors + source of truth (verification report 2026-10-09)

## 0. Status and how to resume work (live section — update on every change)
- Table: KP | name | status | next step | gate/evidence
- Overall % per wave; "If interrupted here, start from…" pointer.

## 1. Decision principles (normative chain)
1. Aurik criterion §III.11: speech must be singing/music (a speech-trained model in the signal path = conceptual deficit).
2. Metrics are witnesses — the auditory instance decides (hoerordnung §1); never-worsen at the IN-V1/V2 seam; flag flips only after ONNX → parity → A/B → witness → hearing acceptance.
3. Determinism §G5: seed per session, no time.time() in decision logic.
4. Silent-failure prohibition §V6: every ML→DSP fallback needs logger.warning + reason.
5. No overwriting without evidence: existing models only replaced after A/B proves superiority (user's goal).
6. Real music / real singing over synthetic — as much as the use case allows; hybrid rule: real content + synthetic defect overlay = OK for denoise fine-tuning; for material/era classification, a real carrier surface is required (§7 nuance).

## 2. Inventory: what's already done (no training needed) — silent capital rollout queue
Table model | state | remaining action | gate/evidence
- DFN Musik ✅ enabled (use_df_musik=True) — nothing left to do
- MIIPHER-DiT ✅ enabled (use_miipher_dit=True)
- Harmonic-Inpainting-DiT ✅ enabled (use_harmonic_inpainting=True)
- EAR-VAE v1-MUSDB ✅ deployed; v2 upstream acquired → comparison benchmark pending (D-K0-9)
- FlashSR F4 ✅ music, gate +18–19.5 dB passed — done
- APPLADE/A-SPADE ✅ production phase_07 de-clicker
- MP-SENet Musik ✅ enabled (use_mp_senet_musik=True)
- BigVGAN-F3: trained + exported + parity; C4 audition pending → rollout decision (D-K0-1, W-1)
- SGMSE+ Musik-Core: ep80/6.1525, ONNX parity 6.7e-5; metric A/B ~1dB better; hearing acceptance test pending (D-K0-3)
- SCNet 4stems: wired behind never-worsen; C4 r2 package built → audition + sign-off → use_scnet_music=True (+ RT budget D-K3-37/D-K0-5)
- Whisper-Denoiser: deprecated — re-evaluate or archive (D-K0-4)
- AudioLDM2: plugin exists and is wired into phase_24 tier 3; role decision after S1 pending

## 3. P0 — maximum effect at minimum effort (substitution + calibration, no training)
KP-1 Rollout of silent capital (audition round): C4 SCNet r2 package + W-1 HR-V1 + SGMSE 3-way hearing acceptance test → up to 3 flag flips without GPU. Human gate: MUSHRA / audition panel.
KP-2 Music MOS oracle for F8 (substitute UTMOS with MuQ+A1 head and SingMOS Pro): path A no training (witness pair, threshold calibration on a MUSHRA subset; gate = directional stability across all degradation classes + non-inferiority against the UTMOS report-only period); path B training only if A is insufficient (≥50 pairs of MUSHRA music ratings).
KP-3 Vocal identity F9: substitute Resemblyzer with MERT/MuQ embeddings for level 1 guard + phase_65; gate = A/B of witness decisions on the golden set (same decision → cos invariant) + hearing acceptance test.
KP-4 VAD: Silero → PANNs singing class / AST as a music-adapted gate offset (pattern in phase_19, D-K3-52); use_silero_vad_music stays False until A/B is done.
KP-5 Studio Finisher vocoder decision (D-K2-3): Vocos 48k primary → BigVGAN-F3 (already music-trained) as stage 1 of the studio path (after C4 audition), Vocos kept as §V6 fallback; if RT budget fails → vocal fine-tune of Vocos on MUSDB18-HQ. Highest-impact speech-in-signal-path fix without new training.
KP-6c? No — BANQUET is KP-7 in P1 (GPU). Hmm, per the user's draft: P0 = KP-1..KP-5; P1 = KP-6 BANQUET… but I was planning to include the optimality assessment for BANQUET. Keep the numbering as planned:

## 4. P1 — GPU training runs (7900 XTX, ROCm)
KP-6 BANQUET real vinyl fine-tune F12 (+ upstream optimality assessment):
   - Upstream: kwatcharasupat/query-bandit (ISMIR 2024, MoisesDB = music ✓ MIT; Zenodo ev-pre-aug.ckpt)
   - Aurik use case: phase_09 vinyl crackle, blind spectral mask enhancer [1,128,128,128]→same shape; DSP fallback is median de-clicker + HPF (§V6); gate QUALITY_MODE and _is_vinyl and is_phase_ml_enabled(9) (root fix applied 2026-09-12)
   - Optimality verdict: architecture fits (bandsplit RNN 24-cell, 23.69M params — both low-band rumble + high-band crackle; CPU parallel windows 1.58–1.61×); query/PaSST part not needed for blind enhancement (already baked into vinyl_final)
   - Required adjustments:
     a) Data: real vinyl↔reissue pairs (≥30, covering crackle/hiss/wow/surface-noise); locally only 6 synthetic CC0 pairs; hybrid approach acceptable (real PD content + defect overlay), but for the material classifier a real carrier surface is required (KP-12)
     b) Provenance hygiene: ev-pre-aug.ckpt referenced by upload_models_to_hf.py doesn't exist in models/banquet/; no SHA link between upstream Zenodo weights and deployed ONNX → per §III.13, document the provenance chain or re-derive from ev-pre-aug.ckpt
     c) Gate: ΔSDR ≥ +2 dB vs zero-shot on destructive cases + never-worsen ≥ −0.1 dB on clean refs; then full contract (export → parity rel ≤ 1e-3 §III.9 → A/B → witness → hearing acceptance test → flag flip); script ends with evidence report, no automatic flag flip
   - Estimate: data procurement is the bottleneck; run itself a few hours (23.7M params)
KP-7 BW-Reconstructor v5 retrain F11: masking-aware A1 loss; target HF gain gate ≥ 1.02 (failed at 0.73); pass → use_bw_v5=True as shellac/lacquer high-frequency path
KP-8 Cantus = vocal Inpainting DiT lane (§C): model exists (full preset 204.27M, ONNX export parity rel ≤ 2.2e-5), already wired in StemLevelRestorer §SLR-1e2 behind zoo status gate + hallucination guard; pretrain tiny smoke run done (CPU); data cantus_pairs 3600 pairs real music + synthetic degradation available locally
   - Remaining: full-preset training on GPU (pretrain → fine_tune → domain_adapt, 3 phases), re-export, parity check, A/B against MIIPHER-DiT+HNR baseline, hearing acceptance test, zoo status training→active
   - Hygiene items: README claims checkpoint_best.pt = "204.27M params, epoch 10" but actually tiny (0.37M params, ep1/step2); parity.json references overwritten checkpoints; placeholder files coexist with real artifacts → update
KP-9 GaCELA retrain F2 (optional): only if KP-8 is insufficient for 375–1500ms gaps; task reformulation (lesson from GaCELA: silence-gap + energy gating, not full waveform match)
KP-10 RT60 regressor: GPU training on RIR pairs (local DFN asset rir_sim_* + real recordings); replaces the conservative T60 witness cap in phase_20/49
KP-11 Symphonia = instrumental core (mirror image of Cantus): status pending_training, only tiny ONNX; registry path symphonia.onnx missing (hygiene: point to actual export or create it); data/symphonia_pairs(+_full/_batches) available locally; after training → same contract as KP-8
KP-12 SingMOS Pro calibration F10: light fine-tune/calibration of singing MOS on processed music vocals — data from golden listening set + MUSHRA

## 5. P2 — data-dependent / conditional
KP-13 Material classifier D-K0-8: WP-7 status (machine-readable wp7_coverage.json: total labels 143, real carriers shellac 47/digital 10/cassette 4/reel 1/tape 1/vinyl 0; target ≥50 per class reached for shellac in total labels = 58 — the "shellac 47 vs 58" discrepancy is resolved: two different counters from the same JSON); training on 6-class CLAP/shallow head once real carriers ≥10 per class are met; vinyl/Tape requires license policy (Unlocked Recordings)
KP-14 MERT Genre head: GTZAN/MagnaTagATune or MUSDB-based genre labels; gate = accuracy against human labels + A/B goal adaptation
KP-15 Gender head F13 + calibration F14: label corpus ≥50 per class (data/gender_labels.csv missing)
KP-16 DDSP C4 retrain on DAC encoder: only if val-MAE is clearly below mean-value baseline; otherwise accept D-K0-6

## 6. P3 — new models (no local base)
KP-17 Neural warp estimator WF-V4: train a warp head on FCPE/CRePE-F0 features (base available locally); gate = residual wow/flutter below hearing threshold + never-worsen against phase_12 warp
KP-18 Neural phase estimation for transients TP-V2: vocoder as phase synthesizer / small net; data = transient-rich music segments with phase ground truth from clean references
KP-19 AudioLDM2 latent inpainting F6 (optional): RePaint-style CLAP-audio conditioning — only if KP-8 doesn't cover extreme cases; plugin exists and is wired into phase_24

## 7. Synthetic → real replacement register
Table: synthetic test/training | location | substitute with real music/singing | status
- BANQUET corpus/vinyl (16 damaged/6 clean all CC0 synthetic) → real vinyl↔reissue pairs — pending (KP-6a)
- Material classifier carriers 63 real vs previously 12 (WP-7 in progress; vinyl structurally 0, tape/reel minimal) → license policy decision (D-K1-2)
- Gender labels: data missing (KP-15)
- SGMSE A/B n=3 @10dB SNR + synthetic reverb → extend to more songs with real room reverb before hearing acceptance test (part of KP-1)
- DFN paired retrain candidate (§v10.25 P2 3–4): corpus = real PD music content + applied defects = OK for denoise fine-tuning (hybrid rule); documented in §1.6
- Cantus pairs: 3600 MUSDB-HQ vocals with synthetic degradation — hybrid, acceptable for flow-matching pretraining; domain_adapt phase should use real degraded recordings where available

## 8. Sequencing and GPU budget
Wave A (now, no training): KP-5 decision + C4 audition round (SCNet r2 ready) + SGMSE hearing acceptance test → up to 3 flag flips without GPU
Wave B (GPU in parallel within VRAM limits): KP-6 BANQUET (data first), KP-7 BW-v5 (~hours), KP-10 RT60, KP-8 Cantus full pretrain
Wave C: KP-8 fine_tune/domain_adapt + export/parity/A/B; then conditional KP-9/KP-12/KP-11 Symphonia
MUSHRA study WP-8 (n≥30) in parallel as human bottleneck — calibrates F8, supports all hearing acceptance tests
VRAM: 23.9 GiB free measured — Cantus full (204M params fp32 ≈ ~1GB weights + activations at chunk 4s/batch 8) fits; runs are sequential unless VRAM allows

## 9. Open decisions (user/maintainer sign-off)
Table: # | decision | recommendation | owner | context
1 SCNet enablement after C4 r2 (+ RT budget acceptance D-K3-37/D-K0-5) — recommend enablement if hearing acceptance test passes and performance debt accepted
2 BigVGAN-F3 rollout to studio finisher cascade instead of Vocos 48k primary (KP-5)
3 SGMSE flag flip after hearing acceptance test (or WPE-only decision)
4 EAR-VAE v1↔v2 comparison benchmark (D-K0-9)
5 Whisper-Denoiser: archive or re-evaluate (D-K0-4)
6 AERO: MUSHRA challenger round vs task reformulation vs archive (B.5)
7 Vinyl license policy for real carrier corpus (Unlocked Recordings; D-K1-2) — blocks KP-6a + KP-13 vinyl class
8 AudioLDM2 role as generative baseline after S1

## 10. Hörproben (collected, per task requirements at the end of the concept)
Table: package | path | decision it enables | manifest/guide | status
- SCNet C4 round 2 | output_audio/mushra/thresholds_scnet_c4_fair_r2 | use_scnet_music flag flip | docs/reports/current/2026-10-08_hoerprobe_scnet_c4_r2.json | ready (12 trials + 2 traps)
- HR-V1 W-1 timbre consistency | output_audio/mushra/thresholds_wohlklang_w1_hrv1/ | BigVGAN-F3 rollout / C4 (D-K0-1) | guide docs/reports/current/2026-10-07_hoerprobe_anleitung.md | ready
- SGMSE 3-way hearing acceptance test | to be built from scripts/benchmark_sgmse_ab.py assets | use_sgmse_musik flag flip (D-K0-3) | — | pending build (KP-1)
- MUSHRA smoke package | output_audio/mushra/smoke_player, smoke_thresholds | tooling verification | — | done
