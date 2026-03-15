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

**Vani fills the gap: quality Hindi TTS that runs on your phone, offline, for free.**

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
espeak-ng (IPA phonemes)
      ↓
StyleTTS2 Acoustic Model
  ├── PLBERT (prosody-aware BERT)
  ├── Style Diffusion (voice identity)
  ├── Prosody Predictor (F0 + energy)
  └── Mel Spectrogram output
      ↓
HiFiGAN Vocoder (trained separately on Hindi)
      ↓
24kHz Waveform
```

**Two-stage design:** The StyleTTS2 acoustic model learns phoneme-to-mel mapping with correct Hindi rhythm, timing, and prosody. The HiFiGAN vocoder learns mel-to-waveform conversion with sharp, perceptually realistic audio. Training them separately is more stable than joint adversarial training on limited VRAM.

**Key choices:**
- **StyleTTS2** over Piper — style diffusion gives significantly better prosody ceiling
- **HiFiGAN V1** vocoder — trained from scratch on Hindi audio for native consonant sharpness
- **espeak-ng phonemizer** — correct Hindi IPA via `phonemizer` library, 178-symbol vocabulary
- **Single speaker** — curated single-voice subset for maximum voice consistency
- **Not larger models** — Parler-TTS/Veena (0.9B–3B params) require GPU; cannot run on mobile

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | [ai4bharat/indicvoices_r](https://huggingface.co/datasets/ai4bharat/indicvoices_r) |
| Language | Hindi (hi) |
| Sample Rate | 24,000 Hz |
| Raw samples downloaded | 15,000 |
| After duration filter (≤10s) | 11,658 training / 750 validation |
| Vocoder training data | All 15,000 WAVs (no transcripts needed) |
| Phoneme Tokens | 178 IPA symbols (full StyleTTS2 vocabulary) |

---

## ⚙️ Training

### Acoustic Model — StyleTTS2 Fine-tuning

#### Round 1 (Complete ✅ — Mar 3–11, 2026)

| Parameter | Value |
|---|---|
| Base checkpoint | LibriTTS `epochs_2nd_00020.pth` (736MB) |
| Effective epochs | ~87 (50 scheduled + ~37 before power cut) |
| Batch size | 2 |
| Learning rate | 5e-5 / 5e-6 (BERT) |
| Training time | ~8 days on RTX 3060 12GB |
| Final mel loss | ~0.23–0.31 |
| Final Gen loss | 5–7 |
| Result | Correct Hindi rhythm and word boundaries. Consonants not intelligible. |

**Root cause diagnosed:** The HiFiGAN decoder inside StyleTTS2 never converged because batch_size=2 is too small for the GAN discriminator to build a reliable decision boundary. The acoustic structure (mel spectrogram) was correct — confirmed via Griffin-Lim reconstruction, which produced speech-shaped audio with correct Hindi phoneme patterns ("ghost talking but trying to say something"). Only the vocoder component was broken.

#### Round 2 (Complete ✅ — Mar 11–15, 2026)

| Parameter | Value |
|---|---|
| Base checkpoint | `epoch_2nd_00049.pth` from Round 1 |
| Epochs completed | ~30+ |
| Batch size | 2 with accum_steps=4 (effective batch=8) |
| Learning rate | 2e-5 / 2e-6 (BERT) |
| Result | GAN partially converged (Gen ~2.5), rhythm improved, still not fully intelligible |

**Decision:** Rather than continuing to fix StyleTTS2's internal GAN (which requires 12GB+ for effective adversarial training), switch to a standalone HiFiGAN vocoder trained directly on Hindi audio. The acoustic model is confirmed good; only the waveform synthesis stage needs fixing.

### Vocoder — Standalone HiFiGAN (🔄 In Progress — Mar 15, 2026)

| Parameter | Value |
|---|---|
| Architecture | HiFiGAN V1 |
| Training data | 15,000 Hindi WAVs at 24kHz (IndicVoices-R) |
| Sample rate | 24,000 Hz |
| Hop size | 300 (matches StyleTTS2 mel extraction exactly) |
| Win size | 1,200 |
| n_fft | 2,048 |
| Segment size | 9,600 samples |
| upsample_rates | [10, 5, 3, 2] |
| upsample_initial_channel | 512 |
| Batch size | 16 |
| Target steps | 100,000 |
| Estimated time | ~10 days on RTX 3060 |
| Current status | Step ~6,000 / Mel Error ~0.40 / actively training |

**Why standalone HiFiGAN works where internal GAN failed:** HiFiGAN trains on (mel, waveform) pairs directly from real audio — no phoneme alignment, no style diffusion, no multi-loss balancing. The task is much simpler: learn to invert a mel spectrogram. At 50k steps it produces clean waveforms; at 100k steps it should match or exceed the quality of the internal StyleTTS2 decoder.

**Integration:** The vocoder replaces `model.decoder` in `infer.py`. The StyleTTS2 acoustic model generates a mel spectrogram via its internal decoder (used only as a mel source), which is then passed to the standalone HiFiGAN generator for final audio synthesis.

---

## 📈 Key Metrics

| Metric | Round 1 | Round 2 | Vocoder Target |
|---|---|---|---|
| Mel loss (acoustic) | 0.23–0.31 | 0.20–0.28 | — |
| HiFiGAN Mel Error | — | — | < 0.10 at 50k steps |
| Gen (GAN) loss | 5–7 | 2.5–3.5 | — |
| Alignment (Dur/CE) | 0.00 ✅ | 0.00 ✅ | — |
| Word boundaries visible | ✅ Yes | ✅ Yes | ✅ |
| Intelligible words | ❌ No | ❌ Barely | ✅ Expected at 50k steps |
| MOS estimate | ~1.5 | ~1.8 | > 3.8 |

---

## 🛠️ Engineering — Bug Chronicle

Getting this system to work required solving **33 bugs** across environment, data, training loop, numerical stability, GAN convergence, inference, and vocoder compatibility.

### Environment & Dependencies (5 bugs)

| # | Problem | Fix |
|---|---|---|
| 1 | `torchcodec` import error | `Audio(decode=False)` + soundfile |
| 2 | `misaki` has no Hindi support | `phonemizer` with `backend='espeak', language='hi'` |
| 3 | `monotonic_align` Cython fails | Pure Python fallback |
| 4 | `torch.load` weights_only error (PyTorch 2.6) | `weights_only=False` everywhere |
| 5 | Decoder mismatch (iSTFT vs HiFiGAN) | `decoder.type: hifigan` in config |

### Training Loop (7 bugs)

| # | Problem | Fix |
|---|---|---|
| 6 | `mask_from_lens` 3-arg signature | Updated pure-Python implementation |
| 7 | `maximum_path` IndexError | Guard `[:, 0] if dim > 1` |
| 8 | **Every batch silently skipped** — BOS slice deleted, double-transpose is identity | Restored 3-line attention transform |
| 9 | `skipped_batches` NameError | Initialize to 0 before batch loop |
| 10 | **`gt.size(-1) < 80` skipped 100% of batches** — threshold > max possible clip size | Raised `max_len`, lowered guard to `< 40` |
| 11 | `import copy` missing | Added import |
| 12 | `pin_memory=True` + `num_workers=0` deadlock | `pin_memory=False` |

### Numerical Stability (3 bugs)

| # | Problem | Fix |
|---|---|---|
| 13 | NaN cascade mid-training | `GradScaler` + `clip_grad_norm_(5.0)` |
| 14 | Discriminator fp16 overflow | Run `dl()` in fp32 via `.float()` cast |
| 15 | Generator grad contaminating discriminator | `requires_grad_(False)` on MPD/MSD during gen backward |

### System & Infrastructure (2 bugs)

| # | Problem | Fix |
|---|---|---|
| 16 | Cascade OOM | `expandable_segments:True` + `empty_cache()` + explicit `del` of activations |
| 17 | No stdout when piped through `tee` | `PYTHONUNBUFFERED=1` + `flush=True` |

### Inference (3 bugs)

| # | Problem | Fix |
|---|---|---|
| 18 | Only vowels, no consonants | Style vector split `[:128]`/`[128:]` was reversed vs training |
| 19 | Duration explosion (22-second output) | `clamp(min=1, max=8) * 1.0` + hard frame cap at 100 |
| 20 | Buzzy audio | Use `s_acoustic` from diffusion output, not `ref_s` from reference wav |

### Round 2 GAN Stabilization (10 bugs)

| # | Problem | Root Cause | Fix |
|---|---|---|---|
| 21 | OOM cascade at batch ~4000 | Cache clear every 50 batches not frequent enough | `del` activation tensors every batch + `empty_cache()` every 20 steps |
| 22 | Validation OOM — all batches failed | Training tensors still allocated entering val | `empty_cache()` before each val batch + reduced val to 10 batches |
| 23 | **Broken gradient accumulation** — disc never updated | `optimizer.zero_grad()` called between disc and gen backward; `scaler.update()` called twice | Zero_grad only after full accumulation window; `scaler.update()` called exactly once |
| 24 | `d_loss` NaN cascade | `dl()` inside `autocast` — spectral-normed convolutions overflow fp16 | Remove autocast from `dl()`, run in fp32 |
| 25 | `g_loss` NaN cascade after fix #24 | `gl()` still inside autocast — same fp16 overflow | Remove autocast from `gl()`, run in fp32 |
| 26 | `RuntimeError: Input type (Half) and bias type (float)` | `y_rec` exits autocast as fp16 but `gl()` expects fp32 | `gl(wav.float(), y_rec.float())` |
| 27 | Discriminator frozen at 4.44 (max entropy) | `g_loss.backward()` updated disc weights opposite to `d_loss.backward()` | Freeze disc with `requires_grad_(False)` before gen backward, unfreeze after |
| 28 | `RuntimeError: element 0 of tensors does not require grad` | NaN `continue` bypassed unfreeze, leaving disc frozen into next batch | Safety unfreeze at start of every batch |
| 29 | `accum_count` not reset on skip | Partial gradients from failed batches accumulated into next window | Reset `accum_count = 0` on every `continue` path |
| 30 | GAN lazy equilibrium — stuck for 6+ epochs | LR too low for generator to recover after discriminator started working | `lambda_gen: 0.5` + `lr: 1e-5` for stable convergence |

### HiFiGAN Vocoder Compatibility (3 bugs)

| # | Problem | Root Cause | Fix |
|---|---|---|---|
| 31 | `librosa.filters.mel()` positional args error | New librosa API requires keyword arguments | `sr=`, `n_fft=`, `n_mels=`, `fmin=`, `fmax=` |
| 32 | `torch.stft()` requires `return_complex` | PyTorch 2.x API change | `return_complex=True`; magnitude: `spec.real.pow(2) + spec.imag.pow(2)` |
| 33 | Feature matching size mismatch (1600 vs 1602) | Conv padding causes slight length difference between real/generated | Clip to `min_len` before L1 loss |

---

### 🏆 Most Painful Bug — #10 (8 hours of wasted compute)

Training appeared to work — epochs completed, losses printed, checkpoints saved. But weights never changed. Every batch was silently discarded by `if gt.size(-1) < 80: continue` because the threshold exceeded the maximum achievable clip size for `max_len=70`. Validation ran fine and printed plausible losses, making it invisible.

**Lesson:** Always log `skipped_batches`. If it equals total batches, nothing trained.

---

### 🔥 Second Most Painful — #23 (broken accumulation, weeks of bad training)

Every Round 2 training run showed the discriminator frozen at exactly 4.44 — maximum entropy — meaning it had completely given up. The GAN looked active from the logs but was doing nothing.

Root cause: `optimizer.zero_grad()` was called between the discriminator backward and the generator backward on every single batch. This wiped disc gradients before gen ran. Combined with `scaler.update()` being called twice per step, the disc never actually updated its weights despite appearing to train.

**Lesson:** In GAN training, disc and gen must accumulate gradients together before any optimizer step. Never zero_grad between them.

---

### 💡 Key Architectural Lesson — separate vocoder training

The internal StyleTTS2 GAN (discriminator inside the same training loop as alignment, duration, style, diffusion) creates conflicting gradient signals that are extremely hard to balance on 12GB VRAM with small batch sizes. Training a standalone HiFiGAN vocoder on the same audio data — with no text, no alignment, just (mel, wav) pairs — is dramatically more stable because it is a single-objective task. Same audio quality result, far fewer failure modes.

---

## 🗂️ Project Structure

```
vani-tts/
├── training/
│   └── StyleTTS2/
│       ├── train_finetune.py       # Training script (patched, 30 bugs fixed)
│       ├── meldataset.py           # Dataloader (pin_memory fix, TextCleaner)
│       ├── infer.py                # Inference — StyleTTS2 acoustic + HiFiGAN vocoder
│       └── Configs/
│           ├── config_ft.yml       # Round 1 config (batch=2, 50 epochs)
│           ├── config_ft_r2.yml    # Round 2 config (accum_steps=4, effective batch=8)
│           └── config_ft_r3.yml    # Round 3 config (lambda_gen=0.5, lr=1e-5)
├── vocoder/
│   └── hifi-gan/                   # Standalone HiFiGAN (patched for PyTorch 2.6+)
│       ├── config_hindi.json       # 24kHz, hop=300, seg=9600, upsample=[10,5,3,2]
│       ├── meldataset.py           # Patched (return_complex, keyword args, min_len)
│       ├── models.py               # Patched (feature matching min_len fix)
│       └── cp_hifigan/             # Vocoder checkpoints (every 5000 steps)
├── export/
│   ├── export_onnx.py
│   └── export_coreml.py
├── android/
├── ios/
└── README.md
```

---

## 🚀 Quick Start (after model release)

> ⚠️ Model weights not yet released — vocoder training in progress (~50k steps needed). Star the repo to get notified.

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
- [x] Phase 4 — Training loop stabilized (33 bugs fixed)
- [x] Phase 5 — Round 1: 50 epochs acoustic model (Mar 3–11, 2026)
- [x] Phase 5b — Round 2: 30 epochs GAN fine-tuning (Mar 11–15, 2026)
- [x] Phase 5c — Root cause diagnosed: acoustic model ✅, vocoder ❌
- [x] Phase 5d — Standalone HiFiGAN vocoder setup and training started (Mar 15, 2026)
- [ ] **Phase 5e — Vocoder reaches 50k steps → first intelligibility test** ← 🔄 IN PROGRESS
- [ ] Phase 5f — Vocoder integration finalized (infer.py, duration calibration)
- [ ] Phase 6 — Evaluation (MOS, WER via Whisper, RTF on CPU)
- [ ] Phase 7 — ONNX export (opset 17) + INT8 quantization (target <200MB)
- [ ] Phase 8 — Android integration (ONNX Runtime)
- [ ] Phase 9 — iOS integration (CoreML)
- [ ] Phase 10 — pip package + HuggingFace model upload
- [ ] Phase 11 — Multiple Hindi voices
- [ ] Phase 12 — Hinglish support

---

## 📈 Evaluation Targets

| Metric | Target | Current |
|---|---|---|
| MOS Score | > 3.8 / 5.0 | ~1.8 (pre-vocoder fix) |
| Word Error Rate (WER) | < 8% | ~85% (unintelligible) |
| Real-Time Factor (CPU) | < 0.3x | Not yet measured |
| Model Size (quantized) | < 200 MB | 2.1 GB (unquantized) |
| Android latency | < 300ms/sec audio | Not yet measured |

---

## 🙏 Acknowledgements

- [AI4Bharat](https://ai4bharat.iitm.ac.in/) for the IndicVoices-R dataset
- [yl4579](https://github.com/yl4579/StyleTTS2) for the StyleTTS2 framework
- [jik876](https://github.com/jik876/hifi-gan) for the HiFiGAN vocoder
- [hexgrad](https://huggingface.co/hexgrad/Kokoro-82M) for Kokoro-82M inference reference

---

## 📄 License

Apache 2.0 — free to use, modify, and deploy commercially.

---

## 🤝 Contributing

Contributions welcome. If you speak Hindi natively and want to help evaluate naturalness (MOS scoring), please open an issue.

---

*Built in Hyderabad 🇮🇳 — making Hindi voice AI accessible to everyone, everywhere, offline.*
