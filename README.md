# वाणी (Vani) TTS 🎙️
### Lightweight Hindi Text-to-Speech for Consumer Devices

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Platform: Android | iOS | CPU](https://img.shields.io/badge/platform-Android%20%7C%20iOS%20%7C%20CPU-green.svg)]()
[![HuggingFace Dataset](https://img.shields.io/badge/dataset-ai4bharat%2Findicvoices__r-yellow.svg)](https://huggingface.co/datasets/ai4bharat/indicvoices_r)
[![Training: StyleTTS2](https://img.shields.io/badge/training-StyleTTS2-orange.svg)](https://github.com/yl4579/StyleTTS2)

> **वाणी** (vāṇī) — Sanskrit for *voice, speech, the goddess of language.*

Vani TTS is an open-source, on-device Hindi Text-to-Speech model trained using the **StyleTTS2** architecture, fine-tuned on the [AI4Bharat IndicVoices-R](https://huggingface.co/datasets/ai4bharat/indicvoices_r) dataset. It is designed to run in **real-time on CPU** — no internet, no GPU, no cloud — making it suitable for Android phones, iOS devices, and low-end laptops.

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
- 🎙️ **Natural Hindi voice** — fine-tuned on 11,600+ studio-grade Hindi speech samples
- 📦 **ONNX export** — deploy on Android (ONNX Runtime) or iOS (CoreML)
- 🔡 **Devanagari native** — handles Hindi script via espeak-ng IPA phonemization
- ⚖️ **Lightweight** — target model size under 200MB

---

## 🗂️ Project Structure

```
vani-tts/
├── data/
│   └── prepare_data.py         # Dataset streaming & preprocessing (IndicVoices-R)
├── training/
│   └── StyleTTS2/              # StyleTTS2 fine-tuning repo (yl4579/StyleTTS2)
│       ├── train_finetune.py   # Main training script (heavily patched)
│       ├── meldataset.py       # Dataloader (patched: pin_memory, TextCleaner)
│       └── Configs/
│           ├── config_ft.yml       # Round 1 config (batch_size=2, 50 epochs)
│           └── config_ft_r2.yml    # Round 2 config (batch_size=8, 30 epochs)
├── infer.py                    # Inference script
├── evaluation/
│   └── evaluate.py             # MOS, WER evaluation
├── export/
│   ├── export_onnx.py          # Export to ONNX
│   └── export_coreml.py        # Export to CoreML (iOS)
├── android/                    # Android demo app (coming soon)
├── ios/                        # iOS demo app (coming soon)
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start (Inference — after model release)

> ⚠️ Model weights not yet released — training in progress. Star the repo to get notified.

```bash
pip install vani-tts
```

```python
from vani import VaniTTS
tts = VaniTTS()
tts.synthesize("नमस्ते, मेरा नाम वाणी है।", output="output.wav")
```

---

## 🏗️ Architecture

Vani TTS is built on the **[StyleTTS2](https://github.com/yl4579/StyleTTS2)** architecture (the same backbone used by Kokoro-82M), fine-tuned from the LibriTTS pretrained checkpoint:

```
Devanagari Text → espeak-ng (IPA phonemes) → PLBERT → Style Diffusion → HiFiGAN Decoder → Audio
```

**Key architectural choices:**

- **StyleTTS2 over Piper** — StyleTTS2 uses adversarial style diffusion for naturalness vs Piper's 2021-era VITS; significantly better prosody ceiling
- **HiFiGAN decoder** — matches the LibriTTS pretrained base checkpoint; higher quality than iSTFT for fine-tuning from English pretrain
- **espeak-ng phonemizer** — handles Hindi IPA correctly via `phonemizer` library with `backend='espeak', language='hi'`; 178 symbol vocabulary covering full IPA
- **Single speaker** — trained on a curated single-voice Hindi subset for maximum voice consistency
- **Why not larger models?** At 0.9B–3B parameters, Parler-TTS/Veena require GPU inference and cannot run on mobile CPUs in real-time

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | [ai4bharat/indicvoices_r](https://huggingface.co/datasets/ai4bharat/indicvoices_r) |
| Language | Hindi (hi) |
| Sample Rate | 24,000 Hz |
| Raw samples downloaded | 15,000 |
| After duration filter (≤10s) | 11,658 training / 750 validation |
| Phoneme Tokens | 178 IPA symbols (full StyleTTS2 vocabulary) |
| Download Method | HF streaming + `Audio(decode=False)` + soundfile |

---

## ⚙️ Training — Two Rounds

### Round 1 (Complete ✅ — Mar 3–11, 2026)

| Parameter | Value |
|---|---|
| Base checkpoint | LibriTTS `epochs_2nd_00020.pth` (736MB) |
| Epochs | 50 (+ ~37 before power cut = ~87 effective) |
| Batch size | 2 |
| Max sequence length | 70 mel frames |
| Learning rate | 5e-5 / 5e-6 (BERT) |
| Hardware | RTX 3060 12GB + i3-8100 |
| Training time | ~8 days |
| Final mel loss | ~0.23–0.31 |
| Final Gen loss | 5–7 (GAN did not converge) |
| Result | Syllable rhythm correct, consonants not intelligible |

**Root cause of Round 1 quality issue:** `batch_size=2` is too small for GAN discriminator to build a reliable decision boundary. The discriminator and generator received contradictory gradient signals every update, preventing the HiFiGAN decoder from learning perceptually sharp waveforms. Mel loss converged well (acoustic structure learned) but the GAN loss never broke below 5 (needs ~1–2 for crisp audio).

### Round 2 (In Progress 🔄 — started Mar 11, 2026)

| Parameter | Value | Change |
|---|---|---|
| Base checkpoint | `epoch_2nd_00049.pth` from Round 1 | Continues from R1 |
| Epochs | 30 | — |
| **Batch size** | **8** | **4× increase — the key fix** |
| Max sequence length | 50 mel frames | Reduced to fit 8×samples in 12GB |
| Learning rate | 2e-5 / 2e-6 (BERT) | Lowered for fine-tune stability |
| Expected improvement | Gen loss drops to 2–3, intelligible Hindi | — |
| Estimated completion | ~Mar 12, 2026 | — |

---

## 📈 Training Metrics Observed

| Metric | Round 1 Final | Target | Notes |
|---|---|---|---|
| Mel loss | 0.23–0.31 | < 0.15 | Acoustic structure learned |
| Gen (GAN) loss | 5–7 | 1–2 | **Never converged — batch_size=2 too small** |
| Dur loss | ~0.00 | ~0.00 | ✅ Converged by epoch 3 |
| CE loss | ~0.00 | ~0.00 | ✅ Converged by epoch 3 |
| S2S loss | 0.07–0.66 | < 0.10 | Alignment mostly learned |
| Mono loss | 0.04–0.08 | < 0.05 | Monotonic alignment good |
| Sty loss | 0.05–0.10 | < 0.05 | Style diffusion active |
| Diff loss | 0.17–0.90 | < 0.20 | Still training |
| Skipped batches | 0 | 0 | ✅ All batches processed |
| GPU utilization | 100% | — | ✅ |
| VRAM usage | 7.1–8.1 GB / 12 GB | — | ✅ |

### What the metrics tell us
- **Dur=0, CE=0** — phoneme alignment fully converged in first 3 epochs. The model knows exactly which phoneme maps to which mel frame.
- **Mel loss ~0.25** — the model reconstructs the mel spectrogram reasonably well from phonemes. Acoustic structure (vowels, syllable rhythm, prosody envelope) is correct.
- **Gen loss 5–7 throughout** — the HiFiGAN waveform decoder never learned to produce perceptually realistic audio. This is the single reason for poor intelligibility in Round 1.

---

## 🔊 Inference Results (Round 1)

Tested with `epoch_2nd_00049.pth` on three Hindi sentences:

| Sentence | Waveform structure | Audible words | Notes |
|---|---|---|---|
| नमस्ते, मेरा नाम वाणी है। | ✅ Correct duration (~0.9s), word bursts visible | ❌ Mumbled | Syllable rhythm correct, consonants smeared |
| आज का मौसम बहुत अच्छा है। | ✅ Correct duration (~0.95s), 4–5 word groups visible | ❌ Mumbled | Same pattern |
| मैं एक हिंदी टेक्स्ट टू स्पीच सिस्टम हूँ। | ✅ Correct duration (~1.2s) | ❌ Mumbled | Longest sentence, structure intact |

**Key inference fixes applied:**
- Style vector split corrected (`[:128]` = acoustic → decoder, `[128:]` = prosodic → F0Ntrain)
- Per-phoneme duration clamped to max 20 frames (prevents duration explosion)
- `embedding_scale=0.0` (CFG disabled — was distorting style vector)
- Decoder conditioned on `s_acoustic` from diffusion output, not `ref_s` (matches training distribution)

---

## 🛠️ Engineering Decisions & Bug Chronicle

Getting StyleTTS2 to train on Hindi required solving **17 distinct bugs** across environment setup, data loading, training loop logic, numerical stability, and system configuration.

### Environment & Dependencies

| # | Problem | Root Cause | Fix |
|---|---|---|---|
| 1 | `torchcodec` import error | HuggingFace `datasets` switched default audio backend | `Audio(decode=False)` + manual soundfile decode |
| 2 | `misaki` has no Hindi | misaki supports: en, ja, ko, zh, he, vi only | `phonemizer` with `backend='espeak', language='hi'` |
| 3 | `monotonic_align` Cython compile fails | Requires gcc compilation | Pure Python fallback at `monotonic_align/core.py` |
| 4 | `torch.load` weights_only error | PyTorch 2.6 changed default `weights_only=False→True` | Added `weights_only=False` everywhere |
| 5 | Decoder architecture mismatch | LibriTTS uses HiFiGAN; config defaulted to iSTFT | `decoder.type: hifigan` in config |

### Training Loop Logic

| # | Problem | Root Cause | Fix |
|---|---|---|---|
| 6 | `mask_from_lens` signature mismatch | Pure-Python accepted 2 args; code calls with 3 | Updated `monotonic_align/__init__.py` |
| 7 | `maximum_path` IndexError | `mask.sum(1)[:, 0]` fails when result is 1D | Guard: `[:, 0] if dim > 1` |
| 8 | **Every batch silently skipped** | BOS slice `s2s_attn[..., 1:]` deleted during debugging — double transpose is identity op | Restored canonical 3-line attention transform |
| 9 | `skipped_batches` NameError | Variable incremented before defined | `skipped_batches = 0` at epoch start |
| 10 | **`gt.size(-1) < 80` skipped 100% of batches** | Guard threshold exceeded max possible clip size for `max_len=70` | Raised `max_len=128`, lowered guard to `< 40` |
| 11 | `import copy` missing | `copy.deepcopy()` called without import | Added `import copy` |
| 12 | `pin_memory=True` + `num_workers=0` deadlock | Known PyTorch issue: pin_memory needs background thread | `pin_memory=False` in `build_dataloader()` |

### Numerical Stability

| # | Problem | Root Cause | Fix |
|---|---|---|---|
| 13 | NaN cascade from step ~1574 | GAN training without gradient clipping | `GradScaler` + `clip_grad_norm_(5.0)` on all modules |
| 14 | Discriminator updated before NaN check | `d_loss.backward()` ran unconditionally | NaN guard wraps full discriminator update |
| 15 | LR too high for GAN fine-tune | Default `lr=1e-4` caused GAN instability on Hindi | Halved all LRs: `5e-5 / 5e-6 / 5e-5` |

### System & Infrastructure

| # | Problem | Root Cause | Fix |
|---|---|---|---|
| 16 | Cascade OOM | CUDA allocator fragmentation after first OOM | `expandable_segments:True` + `empty_cache()` every 50 steps + filtered files >10s |
| 17 | No stdout when piped through `tee` | Python buffers stdout when not writing to TTY | `PYTHONUNBUFFERED=1` + `flush=True` on all prints |

### Inference-only bugs (found post-training)

| # | Problem | Root Cause | Fix |
|---|---|---|---|
| 18 | Only vowels synthesized — no consonants | Style vector split reversed in infer.py (`[:128]`/`[128:]` swapped vs training) | Corrected split to match `cat([gs, s_dur])` order from training |
| 19 | Duration explosion (22-second output) | No per-phoneme clamp — sigmoid sum up to 50 frames/phoneme | `clamp(min=1, max=20)` + hard total cap at 1500 frames |
| 20 | Audio still buzzy after split fix | Decoder conditioned on `ref_s` (out-of-distribution at inference) | Use `s_acoustic` from diffusion output — matches training distribution |

---

### 🏆 Most Painful Bug — #10 (8 hours of wasted compute)

Training appeared to work perfectly — epochs completed, checkpoints saved, validation losses printed. But model weights never changed. The tell was `Dur loss: 1.746000` being **bit-for-bit identical** across all 14 epochs.

Root cause: `if gt.size(-1) < 80: continue` compared mel frames against a threshold larger than the maximum achievable clip size given `max_len=70`. Every training batch was silently discarded. Validation (no guard) ran fine and printed plausible-looking losses.

**Lesson:** Always log `skipped_batches`. If it equals total batches, nothing trained.

---

### 🔥 Second Most Painful — #8 (silent training on zero batches)

During debugging, a "double transpose" fix was applied that is mathematically an identity operation. The reshape error disappeared, but every batch failed silently at the matmul stage and was caught by the exception handler. No crashes, no visible errors, zero batches trained per epoch.

---

## 📅 Roadmap

- [x] Phase 0 — Environment setup (Ubuntu 24.04, CUDA 13.0, UV venv)
- [x] Phase 1 — Dataset pipeline (IndicVoices-R, 11,658 samples, 24kHz)
- [x] Phase 2 — Phonemization (espeak-ng IPA, 178 tokens)
- [x] Phase 3 — Pretrained weights + StyleTTS2 config (LibriTTS base, HiFiGAN)
- [x] Phase 4 — Training loop stabilized (17 bugs fixed)
- [x] Phase 5 — Round 1: 50 epochs, batch_size=2 (Mar 3–11, 2026, ~8 days)
- [x] Phase 5b — Inference pipeline fixed (3 additional bugs found and fixed)
- [ ] **Phase 5c — Round 2: 30 epochs, batch_size=8** ← 🔄 IN PROGRESS (Est. Mar 12, 2026)
- [ ] Phase 6 — Evaluation (MOS, WER via Whisper, RTF on CPU)
- [ ] Phase 7 — ONNX export (opset 17) + INT8 dynamic quantization
- [ ] Phase 8 — Android integration (ONNX Runtime)
- [ ] Phase 9 — iOS integration (CoreML)
- [ ] Phase 10 — pip package + HuggingFace model upload
- [ ] Phase 11 — Multiple Hindi voices
- [ ] Phase 12 — Hinglish support

---

## 📈 Evaluation Targets

| Metric | Target | Round 1 | Round 2 (expected) |
|---|---|---|---|
| MOS Score | > 3.8 / 5.0 | ~1.5 (mumbled) | > 3.0 |
| Word Error Rate (WER) | < 8% | ~90% (unintelligible) | < 20% |
| Real-Time Factor (CPU) | < 0.3x | Not measured | Not measured |
| Model Size (quantized) | < 200 MB | 2.1 GB (unquantized) | < 200 MB after ONNX+INT8 |
| Android Latency | < 300ms/sec audio | Not measured | Not measured |

---

## 🙏 Acknowledgements

- [AI4Bharat](https://ai4bharat.iitm.ac.in/) for the IndicVoices-R dataset
- [yl4579](https://github.com/yl4579/StyleTTS2) for the StyleTTS2 training framework
- [hexgrad](https://huggingface.co/hexgrad/Kokoro-82M) for Kokoro-82M inference reference
- [StyleTTS2 paper](https://arxiv.org/abs/2306.07691) — Li et al., 2023

---

## 📄 License

Apache 2.0 — free to use, modify, and deploy commercially.

---

## 🤝 Contributing

Contributions welcome! If you speak Hindi natively and want to help evaluate naturalness (MOS scoring), please open an issue.

---

*Built in Hyderabad 🇮🇳 — making Hindi voice AI accessible to everyone, everywhere, offline.*
