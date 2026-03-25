# वाणी (Vani) TTS 🎙️
### Lightweight Hindi Text-to-Speech for Consumer Devices

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Platform: Android | iOS | CPU](https://img.shields.io/badge/platform-Android%20%7C%20iOS%20%7C%20CPU-green.svg)]()
[![HuggingFace Dataset](https://img.shields.io/badge/dataset-ai4bharat%2Findicvoices__r-yellow.svg)](https://huggingface.co/datasets/ai4bharat/indicvoices_r)
[![Training: StyleTTS2](https://img.shields.io/badge/training-StyleTTS2-orange.svg)](https://github.com/yl4579/StyleTTS2)

> **वाणी** (vāṇī) — Sanskrit for *voice, speech, the goddess of language.*

Vani TTS is an open-source, on-device Hindi Text-to-Speech system built on the **StyleTTS2** acoustic model, fine-tuned on [AI4Bharat IndicVoices-R](https://huggingface.co/datasets/ai4bharat/indicvoices_r), with a standalone **HiFiGAN vocoder** trained separately on Hindi audio. Designed to run **real-time on CPU** — no internet, no GPU, no cloud — suitable for Android, iOS, and low-end laptops.

---

## 🌍 Why Vani?

| Model | Hindi Quality | On-Device | Mobile Ready | Open Source |
|---|---|---|---|---|
| Google TTS | ✅ Good | ❌ Cloud only | ❌ | ❌ |
| Veena (Maya Research) | ✅ Excellent | ❌ Needs GPU | ❌ | ❌ |
| AI4Bharat Indic Parler-TTS | ✅ Very Good | ❌ 0.9B params | ❌ | ✅ |
| Piper TTS (hi) | ⚠️ Poor | ✅ | ✅ | ✅ |
| **Vani TTS** | 🔄 **In Training** | ✅ **Yes** | ✅ **Yes** | ✅ **Yes** |

---

## ✨ Features

- 🏃 **Real-time on CPU** — runs on any Android/iOS device
- 📴 **Fully offline** — no internet connection required
- 🎙️ **Natural Hindi voice** — trained on 15,000 Hindi speech samples from IndicVoices-R
- 📦 **ONNX export** — deploy on Android (ONNX Runtime) or iOS (CoreML)
- 🔡 **Devanagari native** — handles Hindi script via espeak-ng IPA phonemization
- ⚖️ **Lightweight** — target model size under 200MB after quantization

---

## 🏗️ Architecture

```
Devanagari Text
      ↓
espeak-ng (IPA phonemes, 178-symbol vocabulary)
      ↓
StyleTTS2 Acoustic Model (fine-tuned on Hindi)
  ├── PLBERT (prosody-aware BERT)
  ├── Style Diffusion (voice identity from reference audio)
  ├── Text Encoder + Duration Predictor
  ├── Prosody Predictor — F0Ntrain (pitch + energy, chunked 30 frames)
  └── HiFiGAN Decoder → rough waveform
      ↓
  mel spectrogram (HiFiGAN's exact mel_spectrogram() function)
      ↓
Standalone HiFiGAN Vocoder (trained 100k steps on Hindi audio)
      ↓
24kHz Clean Waveform
```

**Two-stage design:** StyleTTS2 handles text → mel structure (timing, rhythm, phoneme identity). Standalone HiFiGAN converts that mel to clean waveform. The vocoder was trained separately on raw Hindi audio — no text alignment needed — making it dramatically more stable than joint GAN training.

**Critical implementation notes:**
- HiFiGAN MUST use its own `mel_spectrogram()` from `hifi-gan/meldataset.py` — NOT torchaudio (different filter banks → garbage output)
- StyleTTS2 decoder must run in fp32 — `torch.amp.autocast` causes NaN/Inf for sequences >60 frames
- F0Ntrain LSTM overflows to 10^16 Hz for long sequences — must chunk to 30 frames max
- Style split: `s_pred[:, :128]` = acoustic → decoder, `s_pred[:, 128:]` = prosodic → F0Ntrain

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | [ai4bharat/indicvoices_r](https://huggingface.co/datasets/ai4bharat/indicvoices_r) |
| Language | Hindi (hi) |
| Sample Rate | 24,000 Hz |
| Training samples | 11,658 (filtered ≤10s) |
| Validation samples | 750 |
| Vocoder training | All 15,000 WAVs (no transcripts needed) |
| Phoneme tokens | 178 IPA symbols |

---

## ⚙️ Training History

### StyleTTS2 Acoustic Model

#### Round 1 (Complete ✅ — Mar 3–11, 2026)

| Parameter | Value |
|---|---|
| Base checkpoint | LibriTTS `epochs_2nd_00020.pth` (736MB) |
| Epochs | 50 |
| Batch size | 2 |
| Learning rate | 5e-5 |
| Final mel loss | 0.23–0.31 |
| Final Gen loss | 5–7 (GAN never converged) |
| Result | Correct Hindi rhythm + word boundaries. Consonants not intelligible. |

**Root cause:** batch_size=2 is too small for the GAN discriminator. Acoustic structure (mel) was confirmed correct via Griffin-Lim test — produced speech-shaped audio. Only the vocoder was broken.

#### Round 2 (Complete ✅ — Mar 11–17, 2026)

| Parameter | Value |
|---|---|
| Base | epoch_2nd_00049.pth from Round 1 |
| Effective batch | 8 (accum_steps=4) |
| Learning rate | 2e-5 |
| Result | GAN partially converged (Gen ~2.5). Word structure improved. Still blurry consonants. |

**Decision:** Switch to standalone HiFiGAN vocoder instead of continuing to fight StyleTTS2's internal GAN on 12GB VRAM.

#### Round 3 (🔄 Next step — config_ft_r3.yml ready)

| Parameter | Value |
|---|---|
| Base | epoch_2nd_00029.pth from Round 2 |
| lambda_gen | 0.5 (gentle GAN pressure) |
| Learning rate | 1e-5 |
| Gradient accum | None |
| Goal | Converge decoder GAN → sharp consonants |

### Standalone HiFiGAN Vocoder (Complete ✅ — Mar 15–25, 2026)

| Parameter | Value |
|---|---|
| Architecture | HiFiGAN V1 |
| Training data | 15,000 Hindi WAVs at 24kHz |
| Sample rate | 24,000 Hz |
| hop_size | 300 (matches StyleTTS2 exactly) |
| n_fft | 2,048 |
| segment_size | 9,600 |
| upsample_rates | [10, 5, 3, 2] |
| Steps completed | 100,000 |
| Final mel error | 0.25 (plateaued — LR decayed to ~0) |

**Vocoder is confirmed working** — `test_vocoder_real.wav` (real audio → vocoder) sounds perfect. The blur in TTS output comes from StyleTTS2's smeared mel, not from the vocoder.

| Step | Mel Error | Status |
|---|---|---|
| 0 | 2.318 | Random noise |
| 25,000 | 0.291 | Word structure visible |
| 65,000 | 0.257 | Vocoder confirmed working on real audio |
| 100,000 | 0.250 | Plateaued — LR exhausted |

---

## 📈 Current Status

| Component | Status | Quality |
|---|---|---|
| Phonemization | ✅ Working | Correct Hindi IPA |
| Duration predictor | ✅ Working | Correct word timing and pauses |
| Text encoder / alignment | ✅ Working | Correct phoneme sequence |
| Style encoder | ✅ Working | Voice identity from reference |
| HiFiGAN vocoder | ✅ Working | Perfect on real audio |
| StyleTTS2 decoder GAN | ❌ Not converged | Vowels clear, consonants blurry |
| **Overall TTS output** | 🔄 Partial | **Vowels + pauses clear, words not intelligible** |

**What you hear now:** A human-sounding voice with correct rhythm and pauses, but consonants are smeared — you can tell someone is speaking but can't make out individual words.

**What Round 3 training will fix:** The decoder GAN sharpening consonants. Once Gen loss drops below ~1.5, words will become individually clear.

---

## 🛠️ Engineering — Bug Chronicle (40 bugs fixed)

### Environment & Dependencies (5)

| # | Problem | Fix |
|---|---|---|
| 1 | `torchcodec` import error | `Audio(decode=False)` + soundfile |
| 2 | `misaki` has no Hindi support | `phonemizer` with `backend='espeak', language='hi'` |
| 3 | `monotonic_align` Cython fails | Pure Python fallback |
| 4 | `torch.load` weights_only error (PyTorch 2.6) | `weights_only=False` everywhere |
| 5 | Decoder type mismatch (iSTFT vs HiFiGAN) | `decoder.type: hifigan` in config |

### Training Loop (7)

| # | Problem | Fix |
|---|---|---|
| 6 | `mask_from_lens` 3-arg signature | Updated pure-Python implementation |
| 7 | `maximum_path` IndexError | Guard `[:, 0] if dim > 1` |
| 8 | **Every batch silently skipped** | Restored BOS attention slice `s2s_attn[..., 1:]` |
| 9 | `skipped_batches` NameError | Initialize to 0 before batch loop |
| 10 | **`gt.size(-1) < 80` skipped 100% of batches** | Raised `max_len`, lowered guard to `< 40` |
| 11 | `import copy` missing | Added import |
| 12 | `pin_memory=True` + `num_workers=0` deadlock | `pin_memory=False` |

### Numerical Stability (3)

| # | Problem | Fix |
|---|---|---|
| 13 | NaN cascade mid-training | `GradScaler` + `clip_grad_norm_(5.0)` |
| 14 | Discriminator fp16 overflow | Run `dl()` in fp32 |
| 15 | Generator grads contaminating discriminator | `requires_grad_(False)` on MPD/MSD during gen backward |

### System (2)

| # | Problem | Fix |
|---|---|---|
| 16 | Cascade OOM | `expandable_segments:True` + `empty_cache()` |
| 17 | No stdout when piped through `tee` | `PYTHONUNBUFFERED=1` + `flush=True` |

### Inference (3)

| # | Problem | Fix |
|---|---|---|
| 18 | Only vowels, no consonants | Style split `[:128]`/`[128:]` reversed |
| 19 | Duration explosion | `clamp(min=1, max=8)` + hard frame cap |
| 20 | Buzzy audio | `ref_s` vs `s_acoustic` confusion in decoder call |

### Round 2 GAN Stabilization (10)

| # | Problem | Fix |
|---|---|---|
| 21 | OOM cascade | `del` activations + `empty_cache()` every 20 steps |
| 22 | Validation OOM | `empty_cache()` before each val batch |
| 23 | **Broken gradient accumulation** | Zero_grad only after full accumulation window |
| 24 | `d_loss` NaN | Remove autocast from `dl()` |
| 25 | `g_loss` NaN | Remove autocast from `gl()` |
| 26 | fp16/fp32 type mismatch | `gl(wav.float(), y_rec.float())` |
| 27 | Discriminator frozen at 4.44 | Freeze disc during gen backward |
| 28 | Disc stayed frozen after NaN skip | Safety unfreeze at start of every batch |
| 29 | `accum_count` not reset on skip | Reset on every `continue` path |
| 30 | GAN stuck in lazy equilibrium | `lambda_gen: 0.5` + `lr: 1e-5` |

### HiFiGAN Compatibility (3)

| # | Problem | Fix |
|---|---|---|
| 31 | `librosa.filters.mel()` positional args | Switch to keyword args |
| 32 | `torch.stft()` missing `return_complex` | `return_complex=True` |
| 33 | Feature matching size mismatch | Clip to `min_len` before L1 loss |

### Inference Pipeline (7 new)

| # | Problem | Fix |
|---|---|---|
| 34 | torchaudio mel ≠ HiFiGAN mel → buzz output | Use HiFiGAN's own `mel_spectrogram()` function |
| 35 | `autocast` NaN on decoder for mel_len > 100 | Remove autocast, run decoder in fp32 |
| 36 | Diffusion `s_prosodic` → F0 = 10^16 Hz | F0Ntrain chunked to 30 frames, concatenated |
| 37 | `embedding_scale=0.0` disabled text guidance | Restored to 1.5 |
| 38 | F0/N interpolate shape error (2D vs 3D) | `reshape(1, 1, -1)` before interpolate |
| 39 | F0/N on CPU, decoder on CUDA | `.to(DEVICE)` on F0_pred and N_pred |
| 40 | `json` not imported before use | Moved to top-level imports |

---

### 🏆 Most Painful Bug — #10 (8 hours of wasted compute)

Every batch silently discarded. Weights never changed. `if gt.size(-1) < 80: continue` threshold exceeded maximum achievable clip size. Validation printed plausible losses making it completely invisible. **Lesson: always log skipped_batches.**

### 🔥 Second Most Painful — #23 (weeks of bad training)

Discriminator frozen at exactly 4.44 (maximum entropy = gave up completely). `optimizer.zero_grad()` called between disc and gen backward on every batch, wiping disc gradients. **Lesson: in GAN training, disc and gen must accumulate together before any step.**

### 💡 Key Architectural Lesson

Standalone HiFiGAN vocoder trained on raw audio (mel→wav pairs, no text) is far more stable than fixing StyleTTS2's internal GAN on 12GB VRAM. Same quality ceiling, far fewer failure modes. The vocoder reaching 100k steps and working perfectly on real audio proves this approach is correct — the remaining problem is entirely in the StyleTTS2 decoder GAN.

---

## 🗂️ File Locations

```
/media/storage/
├── vani_dataset/wav/                    ← 15,000 Hindi WAVs at 24kHz
├── vani-training/                       ← StyleTTS2 repo (MAIN)
│   ├── Configs/config_ft_r3.yml         ← Round 3 config (next training)
│   ├── infer_working.py                 ← Current working inference script
│   ├── test_vocoder.py                  ← Proven vocoder test (real audio → perfect output)
│   └── .venv/                           ← Python venv
├── vani_checkpoints/
│   └── epoch_2nd_00049.pth              ← Round 1 best (stable predictor)
├── vani_checkpoints_r2/
│   └── epoch_2nd_00029.pth              ← Round 2 best (use for Round 3 base)
└── hifi-gan/
    ├── config_hindi.json                ← Vocoder config (sr=24000, hop=300)
    └── cp_hifigan/g_00100000            ← Vocoder at 100k steps (confirmed working)
```

---

## 🚀 Quick Start (after model release)

> ⚠️ Model weights not yet released — Round 3 StyleTTS2 training pending. Star the repo to get notified.

```bash
pip install vani-tts
```

```python
from vani import VaniTTS
tts = VaniTTS()
tts.synthesize("नमस्ते, मेरा नाम वाणी है।", output="output.wav")
```

---

## 📅 Roadmap

- [x] Phase 0 — Environment setup (Ubuntu 24.04, CUDA 13.0)
- [x] Phase 1 — Dataset pipeline (IndicVoices-R, 15,000 samples, 24kHz)
- [x] Phase 2 — Phonemization (espeak-ng IPA, 178 tokens)
- [x] Phase 3 — Pretrained weights + StyleTTS2 config
- [x] Phase 4 — Training loop stabilized (40 bugs fixed)
- [x] Phase 5 — Round 1: 50 epochs acoustic model ✅
- [x] Phase 5b — Round 2: 30 epochs GAN fine-tuning ✅
- [x] Phase 5c — Root cause diagnosed: acoustic structure ✅, decoder GAN ❌
- [x] Phase 5d — Standalone HiFiGAN setup and training started ✅
- [x] Phase 5e — HiFiGAN 100k steps complete, confirmed working on real audio ✅
- [x] Phase 5f — End-to-end pipeline working: vowels clear, pauses correct ✅
- [ ] **Phase 5g — Round 3 StyleTTS2 training → sharp consonants** ← 🔄 NEXT
- [ ] Phase 6 — Evaluation (MOS, WER via Whisper)
- [ ] Phase 7 — ONNX export + INT8 quantization (target <200MB)
- [ ] Phase 8 — Android integration (ONNX Runtime)
- [ ] Phase 9 — iOS integration (CoreML)
- [ ] Phase 10 — pip package + HuggingFace upload
- [ ] Phase 11 — Multiple Hindi voices
- [ ] Phase 12 — Hinglish support

---

## 📈 Evaluation

| Metric | Target | Current |
|---|---|---|
| MOS Score | > 3.8 / 5.0 | ~2.2 (vowels clear, consonants blurry) |
| Word Error Rate (WER) | < 8% | Not yet measurable |
| Real-Time Factor (CPU) | < 0.3x | Not yet measured |
| Model Size (quantized) | < 200 MB | 2.1 GB (unquantized) |

---

## 🙏 Acknowledgements

- [AI4Bharat](https://ai4bharat.iitm.ac.in/) for IndicVoices-R
- [yl4579](https://github.com/yl4579/StyleTTS2) for StyleTTS2
- [jik876](https://github.com/jik876/hifi-gan) for HiFiGAN
- [hexgrad](https://huggingface.co/hexgrad/Kokoro-82M) for Kokoro-82M inference reference

---

## 📄 License

Apache 2.0 — free to use, modify, and deploy commercially.

---

*Built in Hyderabad 🇮🇳 — making Hindi voice AI accessible to everyone, everywhere, offline.*
