# वाणी (Vani) TTS 🎙️
### Lightweight Hindi Text-to-Speech for Consumer Devices

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Platform: Android | iOS | CPU](https://img.shields.io/badge/platform-Android%20%7C%20iOS%20%7C%20CPU-green.svg)]()
[![HuggingFace Dataset](https://img.shields.io/badge/dataset-ai4bharat%2Findicvoices__r-yellow.svg)](https://huggingface.co/datasets/ai4bharat/indicvoices_r)
[![Training: StyleTTS2](https://img.shields.io/badge/training-StyleTTS2-orange.svg)](https://github.com/yl4579/StyleTTS2)

> **वाणी** (vāṇī) — Sanskrit for *voice, speech, the goddess of language.*

Vani TTS is an open-source, on-device Hindi Text-to-Speech system built on the **StyleTTS2** acoustic model, fine-tuned on [AI4Bharat IndicVoices-R](https://huggingface.co/datasets/ai4bharat/indicvoices_r). Designed to run **real-time on CPU** — no internet, no GPU, no cloud — suitable for Android, iOS, and low-end laptops.

---

## 🚨 **PROJECT STATUS (March 30, 2026 — 12:30 PM IST)**

### **Current Situation: Architecture Crisis**

After **4 training rounds** and **150+ hours of GPU time**, we're stuck in a loop:

| Round | Approach | Outcome | Root Cause |
|---|---|---|---|
| **R1-R2** | StyleTTS2 GAN decoder | ❌ Gen stuck at 2.5-7.0 | 12GB VRAM insufficient for stable GAN |
| **R3** | SimpleMelPredictor (conv-only) | ❌ Plateaued epoch 10, bins 40-80 empty | **Conv-only cannot learn consonants** |
| **R4** | FastPitch (transformer) | ❌ F0 overflow (10^16 Hz), integration bugs | Complex architecture, debugging ongoing |
| **R3 retry** | SimpleMelPredictor + 80 bins | 🔄 **Currently running** | **Expected to plateau again** |

### **⚠️ CRITICAL INSIGHT: We're Repeating Round 3**

**What's running now:**
```yaml
Decoder: SimpleMelPredictor (same conv-only architecture that failed in R3)
Mel bins: 80 (full resolution)
Config: config_plan_b.yml
Status: Epoch 1, Step 200/2914, Loss 0.72
```

**Why this will likely fail:**
- ✅ StyleTTS2 components working (duration, alignment, style)
- ✅ 80 bins confirmed compatible with all modules
- ❌ **SimpleMelPredictor architecture unchanged** (still conv-only)
- ❌ Round 3 showed plateauing at epoch 10 regardless of data/config
- ❌ Mel bins 40-80 will stay empty (consonants unlearnable)

**Evidence from Round 3 (identical architecture):**
```
Epoch 9:  Mel loss 0.24, bins 40-80 empty
Epoch 49: Mel loss 0.24, bins 40-80 empty  ← ZERO improvement
```

The problem was **NOT** "80 bins too hard" — it was **"conv-only architecture fundamentally wrong"**.

---

## 🌍 Why Vani?

| Model | Hindi Quality | On-Device | Mobile Ready | Open Source |
|---|---|---|---|---|
| Google TTS | ✅ Good | ❌ Cloud only | ❌ | ❌ |
| Veena (Maya Research) | ✅ Excellent | ❌ Needs GPU | ❌ | ❌ |
| AI4Bharat Indic Parler-TTS | ✅ Very Good | ❌ 0.9B params | ❌ | ✅ |
| Piper TTS (hi) | ⚠️ Poor | ✅ | ✅ | ✅ |
| **Vani TTS** | ⚠️ **Training struggles** | ✅ **Target** | ✅ **Target** | ✅ **Yes** |

---

## ✨ Target Features (Not Yet Achieved)

- 🏃 **Real-time on CPU** — goal, not yet validated
- 📴 **Fully offline** — architecture complete, model quality pending
- 🎙️ **Natural Hindi voice** — training dataset ready (11,658 samples)
- 📦 **ONNX export** — planned after successful training
- 🔡 **Devanagari native** — phonemization pipeline working
- ⚖️ **Lightweight** — target <200MB (untested)

---

## 🏗️ Architecture

### Current Training (SimpleMelPredictor Retry — March 30, 2026)

```
Devanagari Text
      ↓
espeak-ng (IPA phonemes, 178 symbols) ✅ WORKING
      ↓
StyleTTS2 Acoustic Model ✅ WORKING
  ├── PLBERT (prosody) ✅
  ├── Style Diffusion ✅
  ├── Duration Predictor ✅
  └── Text-to-mel alignment ✅
      ↓
SimpleMelPredictor (conv-only) ⚠️ LIKELY TO FAIL AGAIN
  ├── 4× ResBlocks (no attention)
  ├── 2× Upsample layers
  └── Mel projection
      ↓
Loss: L1(predicted_mel, ground_truth_mel)
```

**Why this is a retry of Round 3:**
- Same SimpleMelPredictor architecture (conv-only, no transformers)
- Same training loss (mel reconstruction)
- Only difference: Full 80-bin resolution (vs mixed in R3)
- Expected outcome: **Plateau at epoch 10, bins 40-80 empty**

### What We Need (FastPitch) — Not Currently Running

```
FastPitch Transformer Decoder ✅ CODE EXISTS, ❌ NOT INTEGRATED
  ├── 4-layer Transformer
  ├── Multi-head self-attention (global context)
  └── Position encoding
```

**Status:** Code written, bugs encountered (F0 overflow, shape mismatches), integration paused.

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | [ai4bharat/indicvoices_r](https://huggingface.co/datasets/ai4bharat/indicvoices_r) |
| Language | Hindi (hi) |
| Sample Rate | 24,000 Hz |
| Training samples | 11,658 (filtered ≤10s) |
| Validation samples | 750 |
| Phoneme tokens | 178 IPA symbols |
| **Dataset quality** | ✅ **Perfect** — not the bottleneck |

---

## ⚙️ Training History — Complete Failure Chronicle

### Round 1: Initial StyleTTS2 Fine-tuning (✅ Complete — March 3–11, 2026)

| Parameter | Value |
|---|---|
| Epochs | 50 |
| Batch size | 2 |
| Final mel loss | 0.23–0.31 ✅ |
| Final Gen loss | 5–7 ❌ (GAN failed) |
| **Result** | ✅ Rhythm correct. ❌ Consonants garbled |
| **Root cause** | batch_size=2 insufficient for GAN stability |
| **Time spent** | 37.5 hours |

---

### Round 2: GAN with Gradient Accumulation (✅ Complete — March 11–17, 2026)

| Parameter | Value |
|---|---|
| Batch size | 2 × 4 accumulation = effective 8 |
| Epochs | 30 |
| Final Gen loss | ~2.5 (partial convergence) |
| **Result** | ✅ Slight improvement. ❌ Consonants still blurry |
| **Root cause** | GAN never fully converged on 12GB VRAM |
| **Time spent** | 22.5 hours |
| **Decision** | Pivot to standalone vocoder |

---

### Standalone HiFiGAN Vocoder Training (✅ Complete — March 15–25, 2026)

| Parameter | Value |
|---|---|
| Architecture | HiFiGAN V1 |
| Steps | 100,000 |
| Final mel error | 0.25 |
| **Test result** | ✅ **Perfect** — vocoder NOT the bottleneck |
| **Time spent** | 167 hours (1 week) |

**Confirmed:** Real audio → HiFiGAN → indistinguishable from original. The vocoder works. Problem is mel prediction.

---

### Round 3: SimpleMelPredictor (❌ FAILED — March 25–28, 2026)

**The critical architectural mistake.**

| Parameter | Value |
|---|---|
| Architecture | Conv-only (4 ResBlocks, no attention) |
| Epochs trained | 50 |
| Mel loss | 0.24 (epoch 9) → **0.24 (epoch 49)** ← ZERO improvement |
| Val loss | 0.76 (massive overfitting) |
| **Mel bins 0-40** | ✅ Yellow/green (vowels learned) |
| **Mel bins 40-80** | ❌ **Dark purple (empty — consonants NOT learned)** |
| **Time wasted** | 37.5 hours (training stopped learning at epoch 10) |

#### 🔬 Post-Mortem Analysis

**Why SimpleMelPredictor cannot work:**

Consonants require modeling:
- Short-range dependencies (formant transitions 5-10ms)
- Spectral precision (F2/F3 formants at 2000-3500 Hz = bins 40-80)
- Cross-frequency coupling (harmonics)

**Conv-only networks cannot do this.** Fixed receptive field (~20 frames) cannot capture:
- Temporal context needed for /ka/ vs /ta/ distinction
- Cross-bin correlations (F1, F2, F3 formants spread across bins)

**Transformers with attention are mandatory** for mel prediction. This is why FastSpeech2, Tacotron2, all production TTS use attention.

**Evidence:** 40 epochs (epoch 10-49) produced IDENTICAL mel spectrograms. Training stopped dead.

---

### Round 4 Attempt 1: FastPitch Integration (❌ FAILED — March 29–30, 2026)

**Tried to do the right thing (transformers), hit implementation bugs.**

| Parameter | Value |
|---|---|
| Architecture | 4-layer Transformer, 2 heads, hidden_dim=256 |
| Epochs attempted | 80 planned |
| Actual progress | Epoch 31 |
| **Fatal bug** | F0 predictor overflow (10^16 Hz instead of 80-300 Hz) |
| **Secondary bugs** | Shape mismatches, integration complexity |
| **Time spent** | 23 hours training + 8 hours debugging |
| **Decision** | Paused for debugging |

**Root cause:** F0Ntrain LSTM numerical instability. Needs clamping fix.

---

### Round 4 Attempt 2: Plan B — Reduce Mel Bins (❌ FAILED — March 30, 2026)

**Hypothesis:** "Maybe 80 bins is too hard. Try 40/64 bins to simplify."

| Change Attempted | Result |
|---|---|
| Dataset: 80 → 40 bins | ✅ Fixed meldataset.py |
| Text aligner: Upsample 40→80 | ✅ Interpolation working |
| StyleEncoder: 40 bins | ❌ Conv2d kernel=5 fails on 4-pixel height |
| StyleEncoder: Try 64 bins instead | ❌ Still fails (kernel too large) |
| StyleEncoder: Change kernel 5×5 → 3×3 | ❌ Checkpoint loading error (80-bin weights incompatible) |
| Skip loading style encoders | ❌ NaN cascade (pretrained + random modules) |
| **Final decision** | ❌ **Give up, restore 80 bins** |

**Time wasted:** 8 hours  
**Lesson:** StyleTTS2 architecture designed for 80 bins. Cannot be reduced without full retrain from scratch.

---

### Round 3 Retry: SimpleMelPredictor + 80 Bins (🔄 IN PROGRESS — March 30, 2026)

**Current training (started 12:00 PM IST):**

```yaml
Decoder: SimpleMelPredictor (same failed architecture)
Mel bins: 80 (full resolution)
Epoch: 1/50
Step: 200/2914
Loss: 0.72
```

**Expected outcome based on Round 3 evidence:**
- ✅ Epoch 1-10: Loss drops to ~0.24
- ❌ Epoch 10-50: **Loss plateaus at 0.24** (no improvement)
- ❌ Mel bins 40-80: **Stay empty** (consonants unlearnable)
- ❌ Audio quality: Vowels only, consonants garbled

**Why we're doing this:** Out of options. FastPitch has bugs, SimpleMelPredictor is known-bad, but 80 bins might behave differently than Round 3's mixed setup.

**Realistic probability of success:** **~10%**  
SimpleMelPredictor failed because of architecture, not mel bin count. Expecting different results from same architecture is wishful thinking.

---

## 🛠️ Engineering — Complete Bug Chronicle (52 bugs + 4 architectural dead ends)

### Critical Architectural Failures (4 rounds, 175+ hours wasted)

| Round | Approach | Duration | GPU Hours | Outcome | Lesson |
|---|---|---|---|---|---|
| **R1-R2** | StyleTTS2 GAN decoder | 80 epochs | 60 hours | ❌ Gen stuck 2.5-7.0 | 12GB VRAM insufficient |
| **R3** | SimpleMelPredictor (conv-only) | 50 epochs | 37.5 hours | ❌ Plateaued epoch 10 | **Conv-only cannot learn consonants** |
| **R4-A** | FastPitch (transformers) | 31 epochs | 23 hours | ❌ F0 overflow bugs | Needs F0 clamping fix |
| **R4-B** | Reduce mel bins 80→40/64 | 0 epochs | 8 hours debugging | ❌ Architecture incompatible | StyleTTS2 designed for 80 bins |
| **R3 retry** | SimpleMelPredictor + 80 bins | 🔄 In progress | TBD | ⚠️ **Expected to fail** | Same architecture as R3 |

**Total wasted:** 128.5 hours training + 8 hours debugging = **136.5 hours**

### Individual Bugs Fixed (45)

*[Previous bug list remains unchanged — environment, training loop, numerical stability, system, GAN, inference bugs]*

### Plan B Failure Details (New Section)

| # | Problem | Attempted Fix | Result |
|---|---|---|---|
| 46 | Dataset loads 80-bin mels | Add `n_mels` parameter to FilePathDataset | ✅ Dataset configurable |
| 47 | `model_params` undefined at dataloader creation | Extract from `config` dict instead | ✅ Fixed |
| 48 | Text aligner expects 80 bins | Upsample 40/64→80 with F.interpolate | ✅ Alignment works |
| 49 | StyleEncoder kernel=5 fails on 40-bin (4 pixels after downsample) | Try 64 bins (powers of 2) | ❌ Still fails |
| 50 | StyleEncoder kernel still too large | Change Conv2d(5→3) + padding(0→1) | ❌ Checkpoint loading error |
| 51 | Checkpoint has 80-bin weights | Skip loading style encoders | ❌ NaN cascade |
| 52 | NaN losses from mixing pretrained + random | Give up Plan B, restore 80 bins | ✅ Training started |

**Total Plan B bugs:** 7 new bugs, 8 hours wasted, 0 progress

---

## 📈 Current Status (March 30, 2026 — 12:30 PM IST)

| Component | Status | Quality |
|---|---|---|
| Phonemization | ✅ Working | Correct Hindi IPA |
| Duration / Alignment | ✅ Working | Correct timing |
| Style encoder | ✅ Working | Voice identity from reference |
| HiFiGAN vocoder (separate) | ✅ Working | **Perfect** on real audio |
| **SimpleMelPredictor** | 🔄 **Training** | **Expected to plateau epoch 10** |
| **Overall TTS** | ❌ **Pending architecture fix** | **No working solution yet** |

### Honest Assessment

**What works:**
- ✅ Dataset pipeline (11,658 samples, perfect quality)
- ✅ Phonemization (espeak-ng IPA, 178 tokens)
- ✅ StyleTTS2 pretrained components (BERT, duration, style)
- ✅ HiFiGAN vocoder (confirmed perfect in isolation)

**What doesn't work:**
- ❌ **Mel prediction** — the critical unsolved problem
- ❌ SimpleMelPredictor: Conv-only, plateaus epoch 10
- ❌ FastPitch: Has bugs (F0 overflow), needs debugging
- ❌ Plan B (reduce bins): Incompatible with pretrained weights

**Next steps (realistic):**
1. Let SimpleMelPredictor retry run to epoch 10
2. Confirm plateau (mel bins 40-80 empty, same as R3)
3. **Decision point:** Fix FastPitch bugs OR retrain StyleTTS2 from scratch with different decoder

---

## 🗂️ File Locations

```
/media/storage/
├── vani_dataset/wav/                    ← 11,658 Hindi WAVs (perfect quality)
├── vani-training/                       ← StyleTTS2 repo
│   ├── Configs/
│   │   ├── config_fastpitch.yml         ← Round 4 FastPitch (paused)
│   │   └── config_plan_b.yml            ← Current (SimpleMelPredictor retry)
│   ├── fastpitch_decoder.py             ← FastPitch code (has bugs)
│   ├── models.py                        ← StyleTTS2 components
│   ├── train_finetune.py                ← Training script (52 bugs fixed)
│   └── meldataset.py                    ← Dataset pipeline
├── vani_checkpoints/                    ← Round 1 (GAN failed)
├── vani_checkpoints_r2/                 ← Round 2 (GAN partial)
│   └── epoch_2nd_00029.pth              ← Base for R3/R4
├── vani_checkpoints_r3/                 ← Round 3 (SimpleMelPredictor FAILED)
│   └── epoch_2nd_00049.pth              ← Plateaued, bins 40-80 empty
├── vani_checkpoints_fastpitch/          ← Round 4 FastPitch (paused at epoch 31)
└── vani_checkpoints_plan_b/             ← Current retry (SimpleMelPredictor + 80 bins)
    └── epoch_2nd_000XX.pth              ← Saving every 2 epochs
```

---

## 🚀 Installation (NOT READY)

> ⚠️ **Model weights not available.** Training stuck after 4 failed rounds. No working TTS output yet.

```bash
# When/if model is released:
pip install vani-tts

from vani import VaniTTS
tts = VaniTTS()
tts.synthesize("नमस्ते", output="output.wav")
```

---

## 📅 Roadmap (Updated)

- [x] Phase 0-4 — Environment, dataset, phonemization, pretrained weights
- [x] Phase 5a — Round 1: GAN attempt ❌ Failed
- [x] Phase 5b — Round 2: GAN with accumulation ❌ Partial
- [x] Phase 5c — HiFiGAN vocoder ✅ **Perfect**
- [x] Phase 5d — Round 3: SimpleMelPredictor ❌ **Plateaued**
- [x] Phase 5e — Round 4 FastPitch attempt ❌ **Bugs**
- [x] Phase 5f — Plan B (reduce bins) ❌ **Incompatible**
- [ ] **Phase 5g — SimpleMelPredictor retry** ← 🔄 **IN PROGRESS (likely to fail)**
- [ ] **Phase 5h — Fix FastPitch bugs** ← **Critical path** (if R3 retry fails)
- [ ] Phase 6 — Evaluation (blocked)
- [ ] Phase 7-10 — Export, deployment (blocked)

---

## 📈 Evaluation Targets (Not Yet Met)

| Metric | Target | Current Status |
|---|---|---|
| MOS Score | > 3.8 / 5.0 | **Not measurable** (no working model) |
| Word Error Rate | < 8% | **Not measurable** |
| Real-Time Factor (CPU) | < 0.3x | Untested |
| Model Size (quantized) | < 200 MB | N/A |

---

## 🔬 Technical Insights

### Why We're Stuck

**Proven fact:** HiFiGAN vocoder produces perfect audio from real mels.

**Unsolved problem:** Generate high-quality predicted mels.

**Failed approaches:**
1. **GAN decoder:** Needs >12GB VRAM for stable training
2. **SimpleMelPredictor:** Conv-only architecture fundamentally cannot learn consonants
3. **FastPitch:** Right architecture, has integration bugs

**The path forward:**
- ✅ FastPitch is the correct architecture (transformers mandatory for mel prediction)
- ❌ Current implementation has F0 predictor overflow bugs
- ⚠️ SimpleMelPredictor retry is a Hail Mary (10% success probability)

### Evidence SimpleMelPredictor Will Fail Again

**Round 3 mel spectrogram evidence:**
```
Mel bins 0-20:  ████████ Yellow (strong vowels)
Mel bins 20-40: ████░░░░ Green  (weak vowels)
Mel bins 40-80: ░░░░░░░░ Purple (EMPTY — no learning)
```

**Training curve:**
```
Epoch 1:  0.50
Epoch 5:  0.30
Epoch 9:  0.24
Epoch 10: 0.24  ← PLATEAU STARTS
Epoch 20: 0.24
Epoch 49: 0.24  ← 39 WASTED EPOCHS
```

**This happened with:**
- 11,658 training samples ✅
- Perfect dataset quality ✅
- Correct phonemization ✅
- Working StyleTTS2 components ✅
- **Conv-only decoder architecture ❌** ← The bottleneck

Changing from "mixed 80-bin config" to "clean 80-bin config" will NOT fix an architectural problem.

---

## 🙏 Acknowledgements

- [AI4Bharat](https://ai4bharat.iitm.ac.in/) for IndicVoices-R
- [yl4579](https://github.com/yl4579/StyleTTS2) for StyleTTS2
- [jik876](https://github.com/jik876/hifi-gan) for HiFiGAN
- [Ming024](https://github.com/ming024/FastSpeech2) for FastSpeech2 reference
- RTX 3060 12GB for surviving 4 rounds of failed experiments

---

## 📄 License

Apache 2.0 — free to use, modify, deploy commercially.

---

## 📧 Contact

- GitHub: [@vivek-541](https://github.com/vivek-541/vani-tts)
- Issues: [Report bugs or questions](https://github.com/vivek-541/vani-tts/issues)

---

## ⚠️ **HONEST PROJECT STATUS**

**Training rounds:** 4 (+ 1 retry in progress)  
**GPU hours spent:** 136.5 hours  
**Working TTS output:** ❌ **None**  
**Architectural dead ends:** 3 (GAN, SimpleMelPredictor, reduce-bins)  
**Bugs fixed:** 52  
**Current approach:** SimpleMelPredictor retry (same architecture that failed in R3)  
**Realistic ETA:** Unknown — depends on FastPitch bug fixes

**Built with determination in Hyderabad 🇮🇳 — 175+ hours of GPU time, 4 failed rounds, still searching for a solution.**

**Current status:** SimpleMelPredictor retry training (Epoch 1/50, Step 200/2914, Loss 0.72). Expected to plateau at epoch 10 based on Round 3 evidence. If plateau confirmed, will return to debugging FastPitch. 🔄
