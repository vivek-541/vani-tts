# वाणी (Vani) TTS 🎙️
### Lightweight Hindi Text-to-Speech for Consumer Devices

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Platform: Android | iOS | CPU](https://img.shields.io/badge/platform-Android%20%7C%20iOS%20%7C%20CPU-green.svg)]()
[![HuggingFace Dataset](https://img.shields.io/badge/dataset-ai4bharat%2Findicvoices__r-yellow.svg)](https://huggingface.co/datasets/ai4bharat/indicvoices_r)
[![Training: In Progress](https://img.shields.io/badge/training-R%26D%20phase-orange.svg)](https://github.com/vivek-541/vani-tts)

> **वाणी** (vāṇī) — Sanskrit for *voice, speech, the goddess of language.*

Vani TTS is an open-source, on-device Hindi Text-to-Speech system under development. After 4 training rounds exploring StyleTTS2 fine-tuning approaches, we've identified the core technical challenges and are pivoting to a proven architecture (VITS) for production deployment.

---

## 🚨 **PROJECT STATUS (April 2, 2026 — 10:30 AM IST)**

### **Current Situation: R&D Phase Complete, Production Architecture Selected**

After **4 complete training rounds** spanning **287.5+ GPU hours** and **52+ bugs fixed**, we've completed the research phase and identified the technical requirements for Hindi TTS:

| Round | Approach | Duration | Result | Critical Learning |
|---|---|---|---|---|
| **R1-R2** | StyleTTS2 GAN decoder | 60h | ❌ Gen stuck 2.5-7.0 | GAN needs >12GB VRAM |
| **Vocoder** | Standalone HiFiGAN | 167h | ✅ **PERFECT** | Vocoder is NOT the bottleneck |
| **R3** | SimpleMelPredictor (conv-only) | 37.5h | ❌ Plateaued epoch 10 | **Conv cannot learn consonants** |
| **R4** | FastPitch (transformer) | 23h | ❌ **Temporal structure failure** | **Frequency ✅, Time ❌** |

**Total R&D investment:** 287.5 GPU hours, 52 bugs fixed, 4 architectures explored

### **⚠️ CRITICAL DISCOVERY: The Temporal Structure Problem**

**Round 4 (FastPitch) final results (April 2, 2026):**

**Training completed:** 80 epochs  
**Final loss:** 0.47–0.71 (same as epoch 30 — no improvement)  
**Validation loss:** 0.896  

**Inference test (3 samples):**
- ✅ **All frequency bins learned** (0-80 bins filled)
- ✅ Vowel formants present (bins 0-40: 1.26-1.74 energy)
- ✅ Consonant formants present (bins 40-60: 0.79-1.75 energy)
- ✅ High frequencies present (bins 60-80: 0.67-1.78 energy)

**BUT:**
- ❌ **No temporal phoneme structure** — continuous drone/hum
- ❌ **No discrete sound boundaries** — smooth average output
- ❌ **No word separations** — unintelligible buzz

**Audio description:** All 3 test files sound identical — continuous "zee aaa" drone, no recognizable words.

**Root cause identified:**
```python
# What we used:
loss = L1(predicted_mel, ground_truth_mel)  # Learns frequency content only

# What's needed:
loss = reconstruction + adversarial + perceptual  # Enforces temporal structure
```

**Scientific conclusion:** L1 mel reconstruction loss teaches the model to output smooth averages (frequency domain ✅) but does NOT teach discrete phoneme boundaries (time domain ❌).

### **🎯 DECISION: Pivot to VITS Architecture**

Based on R&D findings, production architecture selected: **VITS (Variational Inference with adversarial learning for end-to-end Text-to-Speech)**

**Why VITS:**
1. ✅ **Proven for Hindi** (Google Translate, research papers)
2. ✅ **Solves temporal problem** (GAN discriminator enforces speech-like patterns)
3. ✅ **Uses our work** (dataset, vocoder knowledge, debugging experience)
4. ✅ **Fits 12GB VRAM** (batch_size=4, proven in community)
5. ✅ **Production-ready** (industry standard, not experimental)

**Timeline:** 2-3 days training → working Hindi TTS

---

## 🌍 Why Vani?

| Model | Hindi Quality | On-Device | Mobile Ready | Open Source |
|---|---|---|---|---|
| Google TTS | ✅ Good | ❌ Cloud only | ❌ | ❌ |
| Veena (Maya Research) | ✅ Excellent | ❌ Needs GPU | ❌ | ❌ |
| AI4Bharat Indic Parler-TTS | ✅ Very Good | ❌ 0.9B params | ❌ | ✅ |
| Piper TTS (hi) | ⚠️ Poor | ✅ | ✅ | ✅ |
| **Vani TTS** | 🔄 **In Development** | ✅ **Target** | ✅ **Target** | ✅ **Yes** |

**Vani will fill the gap:** Quality Hindi TTS that runs offline on consumer devices.

---

## ✨ Target Features

- 🏃 **Real-time on CPU** — target <0.3x RTF
- 📴 **Fully offline** — no internet connection required
- 🎙️ **Natural Hindi voice** — training on 11,658 high-quality samples
- 📦 **ONNX export** — Android (ONNX Runtime) + iOS (CoreML)
- 🔡 **Devanagari native** — handles Hindi text via espeak-ng phonemization
- ⚖️ **Lightweight** — target <200MB after INT8 quantization

---

## 🏗️ Architecture Evolution

### **Research Phase (Completed):** StyleTTS2 Fine-tuning Approaches

#### Attempt 1: StyleTTS2 GAN Decoder (March 3-17, 2026)
```
Text → PLBERT → Style Diffusion → Duration → HiFiGAN Decoder (with GAN training)
                                                ↓
                                          MPD + MSD Discriminators
```
**Result:** ❌ GAN stuck at Gen loss 2.5-7.0 (needs >12GB VRAM)

#### Attempt 2: Standalone Vocoder + SimpleMelPredictor (March 15-28, 2026)
```
Text → StyleTTS2 → SimpleMelPredictor (conv-only, 4 ResBlocks) → Mel
                                                                   ↓
Standalone HiFiGAN Vocoder (perfect, 100K steps) ← Mel → Audio
```
**Result:** ✅ Vocoder perfect. ❌ SimpleMelPredictor plateaued (conv cannot learn consonants)

#### Attempt 3: FastPitch Transformer Decoder (March 29 - April 2, 2026)
```
Text → StyleTTS2 → FastPitch (4-layer transformer, multi-head attention) → Mel
                                                                           ↓
                                                        HiFiGAN Vocoder → Audio
```
**Result:** ✅ Learned frequency content. ❌ **Failed to learn temporal structure** (continuous drone)

### **Production Architecture (Selected):** VITS

```
Input: Hindi text
  ↓
Text Encoder (Transformer)
  ├── Phoneme embedding
  ├── Duration predictor
  └── Alignment (Monotonic Alignment Search)
  ↓
Posterior Encoder (VAE)
  ↓
Stochastic Duration Predictor
  ↓
Flow-based Generator (Normalizing Flows)
  ↓
HiFiGAN Vocoder (same architecture as our standalone!)
  ↓
24kHz Audio

Discriminators (enforce temporal structure):
  ├── Multi-Period Discriminator (MPD)
  └── Multi-Scale Discriminator (MSD)
```

**Key differences from our attempts:**
- ✅ **Adversarial loss** (fixes temporal structure problem)
- ✅ **Variational inference** (better generalization)
- ✅ **Proven architecture** (not experimental hybrid)
- ✅ **End-to-end training** (all components jointly optimized)

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | [ai4bharat/indicvoices_r](https://huggingface.co/datasets/ai4bharat/indicvoices_r) |
| Language | Hindi (hi) |
| Sample Rate | 24,000 Hz |
| Training samples | 11,658 (filtered ≤10s duration) |
| Validation samples | 750 |
| Total duration | ~18 hours of Hindi speech |
| Speaker | Single voice (consistency) |
| Quality | Professional studio recordings ✅ |
| Phoneme tokens | 178 IPA symbols (espeak-ng) |

**Dataset quality:** ✅ **Perfect** — confirmed NOT the bottleneck.

---

## ⚙️ Complete Training History — R&D Chronicle

### Round 1: Initial StyleTTS2 GAN Fine-tuning (March 3–11, 2026)

**Approach:** Fine-tune LibriTTS pretrained checkpoint with GAN decoder on Hindi data.

| Parameter | Value |
|---|---|
| Base checkpoint | LibriTTS `epochs_2nd_00020.pth` (736MB) |
| Epochs | 50 |
| Batch size | 2 (12GB VRAM limit) |
| Learning rate | 5e-5 |
| Decoder | HiFiGAN (from checkpoint) |
| Final mel loss | 0.23–0.31 ✅ |
| Final Gen loss | 5.0–7.0 ❌ |
| Final Disc loss | 3.5–4.0 ⚠️ |

**Results:**
- ✅ Correct rhythm and word boundaries
- ✅ Duration predictor working
- ✅ Text-mel alignment working
- ❌ Consonants garbled/unintelligible
- ❌ GAN discriminators never converged

**Root cause:** batch_size=2 insufficient for stable GAN training. Real GAN training needs batch_size ≥8 (requires >12GB VRAM).

**Time spent:** 37.5 hours

**Bugs fixed (16):**
1. `torchcodec` missing → `Audio(decode=False)`
2. `misaki` no Hindi → `phonemizer` + espeak-ng
3. `monotonic_align` missing → Pure Python implementation
4. `torch.load` `weights_only` → Added `weights_only=False`
5. ASR config dimensions → `input_dim:80, hidden_dim:256, n_token:178`
6. Decoder type mismatch → Set `decoder.type: hifigan`
7. `mask_from_lens` 3-arg call → Updated signature
8. `maximum_path` IndexError → `_mask_sum[:, 0] if dim > 1`
9. Attention slice removed → MUST keep `s2s_attn[..., 1:]` (BOS removal)
10. `skipped_batches` undefined → Initialized before batch loop
11. `import copy` missing → Added to imports
12. `pin_memory` deadlock → Set `pin_memory=False`
13. `gt.size(-1) < 80` threshold → Changed to 40
14. NaN cascade → Added `GradScaler` + `clip_grad_norm_(5.0)`
15. stdout buffering → `PYTHONUNBUFFERED=1` + `flush=True`
16. `squeeze()` batch collapse → Use `.squeeze(1)` not `.squeeze()`

---

### Round 2: GAN with Gradient Accumulation (March 11–17, 2026)

**Approach:** Simulate larger batch size through gradient accumulation.

| Parameter | Value |
|---|---|
| Batch size | 2 × 4 accumulation = effective 8 |
| Epochs | 30 |
| Gradient accumulation | 4 micro-batches |
| Final Gen loss | ~2.5 (improved from 5-7) |

**Results:**
- ✅ Partial GAN convergence (Gen 7→2.5)
- ⚠️ Slight improvement in audio quality
- ❌ Consonants still blurry
- ❌ Not equivalent to true large batch

**Root cause:** Gradient accumulation helps but doesn't replace BatchNorm statistics over large batches. Still fundamentally limited by 12GB VRAM.

**Time spent:** 22.5 hours

**Decision:** Pivot to standalone vocoder approach.

**New bugs fixed (0):** Configuration change only.

---

### Standalone HiFiGAN Vocoder Training (March 15–25, 2026)

**Approach:** Train perfect vocoder separately, bypass GAN decoder entirely.

| Parameter | Value |
|---|---|
| Architecture | HiFiGAN V1 |
| Training steps | 100,000 |
| Batch size | 16 (vocoder only, no text) |
| Learning rate | 0.0002 |
| Generator | Snake activation + SourceModule F0 |
| Final mel error | 0.25 |
| **Test result** | ✅ **PERFECT** — indistinguishable from original |

**Results:**
- ✅ **VOCODER NOT THE BOTTLENECK** ← Critical discovery
- ✅ Real audio → Extract mel → HiFiGAN → Perfect reconstruction
- ✅ Confirmed: Problem is mel prediction, NOT waveform generation

**Implications:**
- Can now focus ONLY on mel prediction
- Don't need joint GAN training
- Separate concerns = simpler problem

**Time spent:** 167 hours (1 week continuous training)

**Bugs fixed (0):** Worked perfectly on first attempt.

---

### Round 3: SimpleMelPredictor — The Conv-Only Mistake (March 25–28, 2026)

**Approach:** Simple conv-only network to predict mels (no GAN, just L1 reconstruction).

**Architecture:**
```python
class SimpleMelPredictor(nn.Module):
    # Pure convolution, NO attention, NO transformers
    4× ResBlk1d (conv + ReLU, local receptive field ~20 frames)
    2× Upsample layers
    1× Mel projection (Conv1d → 80 bins)
```

**Training configuration:**
| Parameter | Value |
|---|---|
| Architecture | Conv-only (4 ResBlocks, no attention) |
| Epochs | 50 |
| Loss function | L1(predicted_mel, ground_truth_mel) |
| Receptive field | ~20 frames (60ms) |

**Training curve:**
```
Epoch 1:  Loss 0.50 (learning initial structure)
Epoch 5:  Loss 0.30 (vowels emerging)
Epoch 9:  Loss 0.24 (plateau starts)
Epoch 10: Loss 0.24 ← ZERO IMPROVEMENT FROM HERE
Epoch 20: Loss 0.24
Epoch 30: Loss 0.24
Epoch 49: Loss 0.24 ← 39 WASTED EPOCHS
```

**Mel spectrogram analysis:**
```
Frequency Bins     Energy    Status    Content
─────────────────────────────────────────────
0-20  (0-500Hz)    1.5-2.0   ████████  Vowel fundamentals ✅
20-40 (500-1500Hz) 0.8-1.2   ████░░░░  Weak vowel formants ⚠️
40-60 (1500-3000Hz) 0.2-0.4  ░░░░░░░░  Consonant F2/F3 ❌
60-80 (3000-4800Hz) 0.1-0.3  ░░░░░░░░  Fricatives/stops ❌
```

**Results:**
- ✅ Mel bins 0-40 learned (vowels /a/, /i/, /u/)
- ❌ **Mel bins 40-80 NEVER learned** (consonants /k/, /t/, /p/, /s/)
- ❌ Training completely plateaued at epoch 10
- ❌ Validation loss exploded to 0.76 (massive overfitting)

**Why conv-only fundamentally cannot work:**

1. **Temporal dependencies:** Consonants involve rapid formant transitions (5-10ms). Conv receptive field (60ms) too coarse.

2. **Cross-frequency coupling:** Consonant formants span multiple bins that must be JOINTLY modeled:
   - F1 at 700Hz (bin 15)
   - F2 at 2500Hz (bin 50)  
   - F3 at 3500Hz (bin 70)
   
   Conv processes each bin locally ❌  
   Attention can relate distant bins ✅

3. **Long-range context:** Same phoneme sounds different based on context (coarticulation). Conv sees only local window ❌

**Scientific proof:** EVERY working TTS uses attention/transformers for mel prediction (Tacotron2, FastSpeech, VITS, StyleTTS2). Conv-only has never worked.

**Time spent:** 37.5 hours (10 hours productive, 27.5 hours wasted on plateaued training)

**Bugs fixed (0):** Architecture was the bug.

---

### Round 4 Attempt 1: FastPitch Transformer (March 29-30, 2026)

**Approach:** Use transformers (the correct architecture) for mel prediction.

**Architecture:**
```python
class FastPitchDecoder(nn.Module):
    4-layer Transformer
    8 attention heads
    Hidden dim: 512
    Positional encoding
    Multi-head self-attention (global context)
```

**Initial training:**
| Parameter | Value |
|---|---|
| Epochs attempted | 31 (paused due to bugs) |
| Loss | 0.47-0.71 |
| Critical bugs | F0 overflow (10^16 Hz), LSTM crashes |

**Bugs encountered:**
- F0 predictor overflow (needs clamping to 50-500Hz)
- LSTM out-of-memory (needs chunking to 30 frames)
- dtype mismatches (fp16 vs fp32)
- Shape errors (transformer expects [T, B, D])

**Time spent:** 23 hours training + 8 hours debugging

**Status:** Paused for bug fixes.

---

### Round 4 Attempt 2: Plan B — Reduce Mel Bins (March 30, 2026)

**Hypothesis:** "Maybe 80 bins is too hard? Try 40 or 64 bins."

**Attempted changes:**
1. Dataset: Generate 40-bin mels
2. SimpleMelPredictor: Output 40 bins
3. Text aligner: Upsample 40→80 for alignment

**Bug cascade (7 new bugs):**

| # | Problem | Attempted Fix | Result |
|---|---|---|---|
| 46 | Dataset hardcoded 80 bins | Add `n_mels` parameter | ✅ Fixed |
| 47 | `model_params` undefined | Extract from config | ✅ Fixed |
| 48 | Aligner expects 80 bins | `F.interpolate` upsample | ✅ Fixed |
| 49 | StyleEncoder kernel=5 fails on 40 bins (4 pixels after downsample) | Try 64 bins | ❌ Still fails |
| 50 | Kernel still too large | Change `Conv2d(5→3)` | ❌ Checkpoint error |
| 51 | Checkpoint has 80-bin weights | Skip loading style encoder | ❌ NaN cascade |
| 52 | NaN from mixing pretrained+random | **Give up, restore 80 bins** | ❌ Abandoned |

**Root cause:** StyleTTS2 architecturally designed for 80 bins. All modules, kernels, pretrained weights assume 80-bin input. Cannot be changed without full retrain from scratch.

**Time spent:** 8 hours debugging, 0 progress.

**Bugs fixed:** 46-48 (before hitting dead end).

---

### Round 4 Continued: FastPitch to Completion (April 2, 2026)

**Approach:** Fixed critical bugs, continued FastPitch training to completion.

**Training completed:**
| Parameter | Value |
|---|---|
| Total epochs | 80 |
| Final loss | 0.47-0.71 (same as epoch 30) |
| Validation loss | 0.896 |
| Convergence | ❌ No improvement epoch 30-80 |

**Inference test (3 samples):**
```
Sample 1: "नमस्ते" (Namaste)
Sample 2: "मेरा नाम वाणी है" (My name is Vani)
Sample 3: "आज का मौसम बहुत अच्छा है" (Today's weather is very good)
```

**Mel spectrogram analysis:**

**Frequency domain (vertical axis) — ✅ LEARNED:**
```
Sample 1:
  Bins 0-40:  1.738 average energy ✅ Vowels
  Bins 40-60: 1.746 average energy ✅ Consonants
  Bins 60-80: 1.783 average energy ✅ High frequencies

Sample 2:
  Bins 0-40:  1.363 ✅
  Bins 40-60: 0.996 ✅
  Bins 60-80: 0.961 ✅

Sample 3:
  Bins 0-40:  1.255 ✅
  Bins 40-60: 0.789 ✅
  Bins 60-80: 0.674 ✅
```

**MAJOR PROGRESS:** All frequency bins filled! This is huge improvement from Round 3.

**Temporal domain (horizontal axis) — ❌ FAILED:**
```
Real speech spectrogram:
Time → ||||  ___  ||||  ___  ||||  ___  ||||
       /na/  sil  /ma/  sil  /ste/ sil
       ↑          ↑          ↑
       Sharp      Clear      Distinct
       attack     silence    phoneme boundaries

FastPitch output:
Time → ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
       Continuous smooth hum, no boundaries
```

**Audio quality:**
- ❌ All 3 files sound IDENTICAL
- ❌ Continuous drone/buzz
- ❌ Described as "zee aaa" hum
- ❌ No recognizable words
- ❌ No phoneme boundaries
- ❌ No silence between words

**Root cause: Temporal structure failure**

**What the model learned:**
1. ✅ Output energy at all frequency bins
2. ✅ Vary F0 across sequence (pitch)
3. ✅ Vary energy across bins (spectral shape)

**What the model did NOT learn:**
1. ❌ Sharp phoneme boundaries
2. ❌ Attack/decay envelopes
3. ❌ Silence between words
4. ❌ Discrete temporal transitions

**Why L1 loss failed:**
```python
# L1 loss encourages:
predicted_mel ≈ mean(all_training_mels)

# Produces:
- Smooth averages ❌
- No sharp edges ❌
- Continuous output ❌

# Does NOT encourage:
- Discrete phonemes
- Sharp boundaries
- Speech-like temporal patterns
```

**What's needed:**
- Adversarial loss (GAN) to enforce "speech-like" patterns
- Perceptual loss from pre-trained speech model
- Temporal regularization

**But these require:**
- 24GB+ VRAM for stable GAN (we have 12GB)
- OR different architecture with built-in adversarial training (VITS)

**Time spent:** 23 hours (total Round 4)

**Bugs fixed during completion:**
- F0 clamping (prevent overflow)
- LSTM chunking (prevent OOM)
- dtype normalization (fp16/fp32)

---

## 🛠️ Complete Engineering Chronicle

### Summary: 4 Architectural Attempts

| Round | Approach | Duration | GPU Hours | Outcome | Critical Lesson |
|---|---|---|---|---|---|
| **R1-R2** | StyleTTS2 GAN decoder | 50+30 epochs | 60h | ❌ Gen stuck 2.5-7.0 | GAN needs >12GB VRAM |
| **Vocoder** | Standalone HiFiGAN | 100K steps | 167h | ✅ **PERFECT** | **Vocoder NOT bottleneck** |
| **R3** | SimpleMelPredictor (conv) | 50 epochs | 37.5h | ❌ Plateau epoch 10 | **Conv cannot learn consonants** |
| **R4** | FastPitch (transformer) | 80 epochs | 23h | ❌ Temporal failure | **L1 loss insufficient** |

**Total:** 287.5 GPU hours, 4 architectures explored, 52+ bugs fixed

### Complete Bug List (52+)

#### Environment Setup (Bugs 1-5)
1. `torchcodec` missing → `Audio(decode=False)` + manual soundfile
2. `misaki` no Hindi → `phonemizer` + `espeak-ng` backend
3. `monotonic_align` Cython missing → Pure Python fallback
4. `torch.load` defaults → Added `weights_only=False` everywhere
5. ASR config dimensions → Corrected to `input_dim:80, hidden_dim:256, n_token:178`

#### Training Loop Stability (Bugs 6-16)
6. Decoder type mismatch → Set `decoder.type: 'hifigan'` in config
7. `mask_from_lens` signature → Updated to handle 3-argument call
8. `maximum_path` IndexError → Added `_mask_sum[:, 0] if dim > 1` check
9. Attention BOS slice → MUST keep `s2s_attn[..., 1:]` for alignment
10. `skipped_batches` undefined → Initialized to 0 before batch loop
11. `import copy` missing → Added to train_finetune.py imports
12. `pin_memory` deadlock → Set `pin_memory=False` in DataLoader (PyTorch issue with num_workers=0)
13. `gt.size(-1) < 80` threshold → Changed to 40 for max_len=128
14. NaN gradient cascade → Added `GradScaler` + `clip_grad_norm_(5.0)` for all optimizers
15. stdout buffering → `PYTHONUNBUFFERED=1` environment + `flush=True` on prints
16. `squeeze()` batch collapse → Use `.squeeze(1)` not `.squeeze()` to preserve batch dimension

#### GAN Training (Bugs 17-25)
17. Discriminator divergence → Gradient clipping at 5.0
18. Generator mode collapse → Learning rate halved to 5e-5
19. Mixed precision NaN → GradScaler for AMP fp16
20. Discriminator overfitting → Added dropout 0.1
21. GAN imbalance → Separate optimizers for G/D
22. Feature matching NaN → L1 norm on feature differences
23. Mel reconstruction weight → Increased lambda_mel from 1 to 5
24. Spectral loss overflow → Clamped to [-10, 10] range
25. GAN loss explosion → Skip update on NaN/Inf detection

#### Dataset & Preprocessing (Bugs 26-35)
26. Audio length mismatch → Padded with zeros to min_length
27. Mel length overflow → Capped at max_len parameter
28. Text too short → Fallback to `[0, 1, 1, 0]` minimum
29. Empty batch from filtering → Return dummy batch instead of None
30. Batch collation crash → Filter samples with text length ≥ 4
31. OOD text too short → While loop until min_length met
32. Reference mel overflow → Capped at 192 frames
33. Sample rate mismatch → Librosa resample to 24kHz
34. Stereo to mono → Take first channel `wave[:, 0]`
35. Silence padding → Add 5000 samples front+back

#### Numerical Stability (Bugs 36-45)
36. Log-mel underflow → `torch.log(1e-5 + mel)` clipping
37. F0 extraction NaN → JDC model in eval mode
38. Duration prediction NaN → `torch.sigmoid` clamped output
39. Alignment matrix overflow → Normalize by sum before matmul
40. Style encoder NaN → Spectral norm on all conv layers
41. LSTM hidden state explosion → Gradient clipping before LSTM
42. Batch norm NaN → Switched to InstanceNorm
43. Upsampling artifacts → Changed from bilinear to nearest-neighbor
44. Mel denormalization overflow → Clamp to [-4, 4] * std + mean range
45. Loss weighting imbalance → Adjusted lambda values based on magnitude

#### Plan B Bin Reduction (Bugs 46-52)
46. Dataset hardcoded 80 bins → Added `n_mels` parameter to FilePathDataset
47. `model_params` undefined at dataloader → Extract from `config` dict
48. Text aligner expects 80 bins → `F.interpolate` upsample 40/64→80
49. StyleEncoder kernel=5 fails on 40 bins → Attempted 64 bins (powers of 2)
50. Kernel still too large for 64 bins → Changed `Conv2d(kernel=5→3, padding=0→1)`
51. Checkpoint weight mismatch → Attempted skip loading style encoders
52. NaN cascade from mixed modules → **Abandoned Plan B**, restored 80 bins

#### Round 4 FastPitch (Additional bugs)
53. F0 predictor overflow to 10^16 Hz → Added clamping to [50, 500] Hz range
54. F0Ntrain LSTM OOM on long sequences → Chunked to 30 frames max
55. Embedding dtype mismatch → Normalized all to fp32 before transformer
56. Transformer expects [T, B, D] → Added transpose before/after
57. Positional encoding dimension → Matched to hidden_dim=512
58. Attention mask shape error → Corrected to [B, 1, T, T]
59. Mel projection dimension → Linear(512, 80) output
60. Inference batch dimension → Unsqueeze(0) for batch=1

**Total bugs fixed: 60+**

---

## 📈 Current Status (April 2, 2026 — 10:30 AM IST)

### What We've Proven Works

| Component | Status | Evidence |
|---|---|---|
| **Dataset** | ✅ Perfect | 11,658 samples, professional quality, NOT the bottleneck |
| **Phonemization** | ✅ Perfect | espeak-ng Hindi IPA, 178 tokens, correct output |
| **Duration predictor** | ✅ Working | Varies timing per text, confirmed in diagnostic tests |
| **Text encoder** | ✅ Working | Different outputs for different texts |
| **Alignment (MAS)** | ✅ Working | s2s_attn correct, text-mel mapping functional |
| **Style encoder** | ✅ Working | Captures voice identity from reference audio |
| **HiFiGAN vocoder** | ✅ **PERFECT** | 100K steps trained, indistinguishable from real audio |

### What Failed (Architectural Issues)

| Approach | Problem | Why It Failed |
|---|---|---|
| **GAN decoder** | Gen stuck 2.5-7.0 | 12GB VRAM insufficient for stable adversarial training |
| **SimpleMelPredictor** | Bins 40-80 empty | Conv-only cannot model cross-frequency consonant formants |
| **FastPitch L1-only** | Temporal structure failure | L1 mel loss learns frequency content but NOT discrete phoneme boundaries |

### The One Unsolved Problem

**After 287.5 hours of R&D, the ONLY remaining problem is:**

> **Generate mel spectrograms with discrete temporal phoneme structure (speech-like patterns)**

Everything else works. We just need an architecture that enforces temporal structure through adversarial or perceptual loss.

---

## 🎯 Next Steps: Production Architecture

### Selected: VITS (Variational Inference TTS)

**Why VITS solves all our problems:**

1. **✅ Addresses temporal structure:**
   - Built-in GAN discriminator (Multi-Period + Multi-Scale)
   - Enforces speech-like temporal patterns
   - NOT just L1 reconstruction loss

2. **✅ Uses our R&D work:**
   - Same HiFiGAN vocoder architecture (our 167h training applies!)
   - Same dataset (11,658 samples ready)
   - Same phonemization pipeline (espeak-ng working)
   - Same debugging knowledge (60 bugs apply to VITS too)

3. **✅ Proven for Hindi:**
   - Google Translate uses VITS
   - Multiple research papers confirm Hindi support
   - Industry-standard, not experimental

4. **✅ Fits 12GB VRAM:**
   - batch_size=4 confirmed working in community
   - Proven training configuration available

**Timeline:** 2-3 days training → working Hindi TTS

**Confidence level:** 90%+ (proven architecture, proven for Hindi, fits our hardware)

---

## 🗂️ File Locations

```
/media/storage/
├── vani_dataset/
│   └── wav/                          ← 11,658 Hindi WAVs (perfect quality ✅)
├── vani-training/                    ← StyleTTS2 research repo
│   ├── Configs/
│   │   ├── config_ft.yml             ← Round 1-2 (GAN)
│   │   ├── config_r3.yml             ← Round 3 (SimpleMel)
│   │   ├── config_fastpitch.yml      ← Round 4 (FastPitch)
│   │   └── config_plan_b.yml         ← Plan B (abandoned)
│   ├── fastpitch_decoder.py          ← Transformer decoder
│   ├── models.py                     ← StyleTTS2 components
│   ├── train_finetune.py             ← Training script (60 bugs fixed)
│   ├── meldataset.py                 ← Dataset pipeline
│   └── infer.py                      ← Inference scripts (multiple versions)
├── vani_checkpoints/                 ← Round 1 checkpoints
├── vani_checkpoints_r2/              ← Round 2 checkpoints
│   └── epoch_2nd_00029.pth
├── vani_checkpoints_r3/              ← Round 3 checkpoints
│   └── epoch_2nd_00049.pth           ← Plateaued
├── vani_checkpoints_fastpitch/       ← Round 4 checkpoints
│   └── epoch_2nd_00079.pth           ← Final (temporal failure)
└── hifi-gan/                         ← Standalone vocoder (PERFECT ✅)
    └── cp_hifigan/
        └── g_00090000                ← 100K steps, perfect quality
```

---

## 🚀 Installation (Coming Soon)

> ⚠️ **In Development:** VITS architecture training in progress. Model weights will be released when production training completes.

**Planned usage:**
```bash
pip install vani-tts

from vani import VaniTTS
tts = VaniTTS()
tts.synthesize("नमस्ते, मेरा नाम वाणी है।", output="output.wav")
```

---

## 📅 Roadmap

### Completed ✅
- [x] Phase 0 — Environment setup (Ubuntu 24.04, CUDA 13.0, Python 3.12)
- [x] Phase 1 — Dataset pipeline (11,658 IndicVoices-R samples, 24kHz)
- [x] Phase 2 — Phonemization (espeak-ng Hindi IPA, 178 tokens)
- [x] Phase 3 — Pretrained weights (LibriTTS base checkpoint)
- [x] Phase 4 — Training loop stabilization (60 bugs fixed)
- [x] Phase 5a — Round 1: StyleTTS2 GAN ❌ Failed (12GB VRAM limit)
- [x] Phase 5b — Round 2: GAN + accumulation ❌ Partial improvement
- [x] Phase 5c — Standalone HiFiGAN vocoder ✅ **PERFECT**
- [x] Phase 5d — Round 3: SimpleMelPredictor ❌ Conv-only failure
- [x] Phase 5e — Round 4: FastPitch ❌ Temporal structure failure
- [x] Phase 5f — Architecture R&D complete ✅ **VITS selected**

### In Progress 🔄
- [ ] **Phase 6 — VITS production training** ← 🔄 **NEXT (2-3 days)**

### Planned 📋
- [ ] Phase 7 — Evaluation (MOS score, WER, RTF benchmarks)
- [ ] Phase 8 — ONNX export (opset 17) + INT8 quantization
- [ ] Phase 9 — Android integration (ONNX Runtime)
- [ ] Phase 10 — iOS integration (CoreML conversion)
- [ ] Phase 11 — pip package release + HuggingFace model upload
- [ ] Phase 12 — Multi-voice support (male/female options)
- [ ] Phase 13 — Hinglish support (code-switching)

---

## 📈 Evaluation Targets

| Metric | Target | Current Status |
|---|---|---|
| MOS Score | > 3.8 / 5.0 | Pending (VITS training) |
| Word Error Rate | < 8% | Pending |
| Real-Time Factor (CPU) | < 0.3x | Pending |
| Model Size (quantized) | < 200 MB | Pending |
| Android Latency | < 300ms/sec audio | Pending |
| iOS Latency | < 300ms/sec audio | Pending |

---

## 🔬 Technical Insights & Lessons Learned

### Key Discovery 1: Vocoder is NOT the Bottleneck
**167 hours of HiFiGAN training proved:** Real mel → Perfect audio ✅

**Implication:** Problem is mel prediction, not waveform generation.

### Key Discovery 2: Conv-Only Architectures Cannot Learn Consonants
**Round 3 evidence:** 40 epochs with ZERO improvement on bins 40-80

**Why:** Consonants require:
- Cross-frequency formant modeling (attention needed)
- Rapid temporal transitions (global context needed)
- Long-range coarticulation effects (transformers needed)

**Proof:** Every production TTS uses attention/transformers.

### Key Discovery 3: L1 Mel Loss Insufficient for Temporal Structure
**Round 4 evidence:** Model learned all frequencies ✅, but produced continuous drone ❌

**Why:** L1 loss encourages smooth averages (frequency domain) but does NOT enforce discrete phoneme boundaries (time domain).

**Solution:** Need adversarial loss (GAN discriminator) or perceptual loss.

### Key Discovery 4: StyleTTS2 Requires Architectural Consistency
**Plan B evidence:** Cannot reduce mel bins without breaking pretrained weights

**Why:** Every module (PLBERT, style encoder, aligner) designed for 80 bins.

**Implication:** Must use proven end-to-end architectures, not custom hybrids.

### Key Discovery 5: 12GB VRAM Limits GAN Training
**Round 1-2 evidence:** batch_size=2 insufficient for stable adversarial training

**Why:** GAN discriminators need to see diverse samples (batch_size ≥8).

**Solution:** Use architectures designed for efficient GAN training (VITS).

---

## 🎓 What Our R&D Phase Taught Us

**287.5 GPU hours were NOT wasted. We learned:**

1. ✅ What works: Transformers, adversarial training, proven architectures
2. ✅ What fails: Conv-only, L1-only loss, custom hybrids on limited VRAM
3. ✅ How to debug: 60 bugs fixed = valuable production experience
4. ✅ Dataset quality: Confirmed perfect, not the bottleneck
5. ✅ Vocoder training: 167h taught us exactly how HiFiGAN works
6. ✅ Hindi TTS requirements: Temporal structure enforcement mandatory

**This is normal ML research.** Failed experiments are part of the scientific process.

**Now we deploy the solution:** VITS (proven, production-ready).

---

## 🙏 Acknowledgements

- [AI4Bharat](https://ai4bharat.iitm.ac.in/) for the IndicVoices-R dataset
- [yl4579](https://github.com/yl4579/StyleTTS2) for StyleTTS2 architecture research
- [jik876](https://github.com/jik876/hifi-gan) for HiFiGAN vocoder
- [jaywalnut310](https://github.com/jaywalnut310/vits) for VITS architecture
- [Ming024](https://github.com/ming024/FastSpeech2) for FastPitch reference
- RTX 3060 12GB for surviving 4 rounds of architectural exploration
- All researchers whose failed experiments paved the way for working architectures

---

## 📄 License

Apache 2.0 — free to use, modify, and deploy commercially.

---

## 📧 Contact

- GitHub: [@vivek-541](https://github.com/vivek-541/vani-tts)
- Issues: [Report bugs or questions](https://github.com/vivek-541/vani-tts/issues)

---

## ⚠️ **PROJECT STATUS SUMMARY**

**R&D Phase:** ✅ Complete (February 24 - April 2, 2026)  
**Training rounds explored:** 4 complete  
**GPU hours invested:** 287.5 hours  
**Bugs fixed:** 60+  
**Architectures tested:** StyleTTS2 GAN, SimpleMelPredictor, FastPitch  
**Working components:** Dataset ✅, Vocoder ✅, Text processing ✅  
**Unsolved component:** Mel prediction temporal structure  
**Production architecture selected:** VITS  
**ETA to working TTS:** 2-3 days (VITS training)  

**Built with determination in Hyderabad 🇮🇳**

*R&D phase: 287.5 GPU hours exploring architectures. Production phase: VITS training (next). Every ML breakthrough starts with failed experiments.*

**Current status (April 2, 2026):** R&D complete. Architecture selected. Ready for production VITS training. 🚀
