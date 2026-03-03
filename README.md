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
| **Vani TTS** | ✅ **Good→Great** | ✅ **Yes** | ✅ **Yes** | ✅ **Yes** |

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
│           └── config_ft.yml   # Vani training configuration
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
Devanagari Text → espeak-ng (IPA phonemes) → PLBERT → Style Encoder → HiFiGAN Decoder → Audio
```

**Key architectural choices:**

- **StyleTTS2 over Piper** — StyleTTS2 uses adversarial style diffusion for naturalness vs Piper's 2021-era VITS; significantly better prosody ceiling
- **HiFiGAN decoder** — matches the LibriTTS pretrained base checkpoint; higher quality than iSTFT for fine-tuning from English pretrain
- **espeak-ng phonemizer** — handles Hindi IPA correctly via `phonemizer` library with `backend='espeak', language='hi'`; 49 unique Hindi phoneme tokens
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
| After duration filter (≤10s) | ~11,658 training samples |
| Validation Samples | 750 |
| Duration Filter | 1.0s – 10.0s (files >10s removed — caused cascade OOM) |
| Normalization | −20 dB RMS |
| Phoneme Tokens | 49 unique IPA tokens |
| Download Method | HF streaming + `Audio(decode=False)` + soundfile (avoids torchcodec) |

---

## ⚙️ Training Configuration

| Parameter | Value |
|---|---|
| Base checkpoint | LibriTTS `epochs_2nd_00020.pth` (736MB) |
| Epochs | 50 |
| Batch size | 2 |
| Max sequence length | 128 frames |
| Sample rate | 24,000 Hz |
| Decoder | HiFiGAN |
| Mixed precision | AMP fp16 + GradScaler |
| Gradient clipping | 5.0 (all modules) |
| Learning rate | 5e-5 (halved from default to prevent GAN blowup) |
| BERT learning rate | 5e-6 |
| Fine-tune LR | 5e-5 |
| Joint/SLM stage | Disabled (`joint_epoch: 999`) — requires >12GB VRAM |
| Diffusion stage | Starts at epoch 10 |
| Steps per epoch | ~5,829 |
| Step time | ~1.2s/step |
| Hardware | NVIDIA RTX 3060 12GB + Intel i3-8100 (4-core) |
| Estimated total training time | ~97 hours |
| CUDA alloc config | `expandable_segments:True` (prevents OOM cascade) |

---

## 📅 Roadmap

- [x] Phase 0 — Environment setup (Ubuntu 24.04, CUDA 13.0, UV venv)
- [x] Phase 1 — Dataset pipeline (15k IndicVoices-R samples, 24kHz, −20dB RMS)
- [x] Phase 2 — Phonemization (espeak-ng IPA, 49 tokens, 14,250 train / 750 val)
- [x] Phase 3 — Pretrained weights + StyleTTS2 config (LibriTTS base, HiFiGAN decoder)
- [x] Phase 4 — Training loop stabilized (17 bugs fixed — see Engineering Decisions below)
- [x] Phase 4b — Dataset filtered (files >10s removed), full 11,658-sample run started
- [ ] **Phase 5 — 50 epochs training** ← 🔄 IN PROGRESS (Est. completion: ~Mar 8 2026)
- [ ] Phase 6 — Evaluation (MOS score, WER via Whisper, RTF on CPU)
- [ ] Phase 7 — ONNX export (opset 17) + INT8 dynamic quantization
- [ ] Phase 8 — Android integration (ONNX Runtime)
- [ ] Phase 9 — iOS integration (CoreML)
- [ ] Phase 10 — pip package release + HuggingFace model upload
- [ ] Phase 11 — Multiple Hindi voices (male/female)
- [ ] Phase 12 — Hinglish support (code-switching)

---

## 📈 Evaluation Targets

| Metric | Target | Current |
|---|---|---|
| MOS Score | > 3.8 / 5.0 | 🔄 Training (epoch 1/50) |
| Word Error Rate (WER) | < 8% | 🔄 Training |
| Real-Time Factor (CPU) | < 0.3x | 🔄 Training |
| Model Size (quantized) | < 200 MB | 🔄 Training |
| Android Latency | < 300ms/sec audio | 🔄 Training |

---

## 🛠️ Engineering Decisions & Bug Chronicle

Getting StyleTTS2 to train on Hindi required solving **17 distinct bugs** across environment setup, data loading, training loop logic, numerical stability, and system configuration. Documented here for anyone attempting cross-lingual StyleTTS2 fine-tuning.

### Environment & Dependencies

| # | Problem | Root Cause | Fix |
|---|---|---|---|
| 1 | `torchcodec` import error on dataset load | HuggingFace `datasets` switched default audio backend to torchcodec in newer versions | `Audio(decode=False)` + manual soundfile decode |
| 2 | `misaki` has no Hindi module | `misaki` phonemizer supports: en, ja, ko, zh, he, vi — not Hindi | Used `phonemizer` library with `backend='espeak', language='hi'` directly |
| 3 | `monotonic_align` Cython compile fails | StyleTTS2 ships it as a Cython extension requiring `gcc` compilation | Pure Python fallback at `monotonic_align/core.py` |
| 4 | `torch.load` raises `weights_only` error | PyTorch 2.6 changed `weights_only` default from `False` → `True` | Added `weights_only=False` to all `torch.load` calls in `models.py` |
| 5 | Decoder architecture mismatch at checkpoint load | LibriTTS pretrained uses HiFiGAN; config defaulted to iSTFT | `decoder.type: hifigan` in config; removed iSTFT-specific params |

### Training Loop Logic Bugs

| # | Problem | Root Cause | Fix |
|---|---|---|---|
| 6 | `mask_from_lens` signature mismatch | Pure-Python implementation accepted 2 args; StyleTTS2 calls it with 3 | Updated `monotonic_align/__init__.py` to handle 3-arg call |
| 7 | `maximum_path` IndexError on `mask.sum(1)[:, 0]` | Fails when result is 1D | Guarded: `_mask_sum[:, 0] if _mask_sum.dim() > 1 else _mask_sum` in `utils.py` |
| 8 | **Every batch silently skipped — model never trained** | BOS token removal slice `s2s_attn[..., 1:]` was deleted during debugging and replaced with a no-op double transpose. Without it, `s2s_attn` text-dim is N but `t_en` is N-1 → matmul fails every batch → silently caught → skipped | Restored canonical 3-line attention transform: `transpose → [..., 1:] → transpose` |
| 9 | **`skipped_batches` NameError** | Variable incremented before being defined | Added `skipped_batches = 0` at top of each epoch loop |
| 10 | **`gt.size(-1) < 80` skipped every single batch silently** | `max_len=70` → `mel_len=35` → `gt=70 frames` → `70 < 80` → skip. Guard threshold exceeded the maximum possible clip size, so 100% of training batches were discarded while validation (no guard) ran fine | Raised `max_len` to `128`; lowered guard to `< 40` |
| 11 | `import copy` missing | `copy.deepcopy()` called but module never imported | Added `import copy` at top of `train_finetune.py` |
| 12 | **`pin_memory=True` + `num_workers=0` deadlock** | Known PyTorch issue: `pin_memory` requires a background thread that doesn't exist when `num_workers=0`, causing an infinite hang after model load | Set `pin_memory=False` in `build_dataloader()` in `meldataset.py` |

### Numerical Stability

| # | Problem | Root Cause | Fix |
|---|---|---|---|
| 13 | **NaN/Inf cascade from step ~1574 onward** | GAN training without gradient clipping: one bad batch corrupts discriminator weights → all subsequent batches produce NaN → `optimizer.zero_grad()` skip doesn't reset corrupted weights | Added `GradScaler` for fp16 AMP; gradient clipping (`max_norm=5.0`) on all modules via `scaler.unscale_()` before clip |
| 14 | **Discriminator updated before NaN check** | `d_loss.backward()` ran unconditionally even if loss was NaN | Moved NaN guard to wrap discriminator update; skip entire batch if `d_loss` is NaN |
| 15 | **Learning rate too high for GAN fine-tune** | Default `lr=1e-4` caused GAN instability on out-of-domain (Hindi) data within first epoch | Halved all learning rates: `lr=5e-5`, `bert_lr=5e-6`, `ft_lr=5e-5` |

### System & Infrastructure

| # | Problem | Root Cause | Fix |
|---|---|---|---|
| 16 | **Cascade OOM** — hundreds of consecutive batches fail after first OOM | CUDA allocator fragments on first OOM; `empty_cache()` in except block doesn't defragment; every subsequent batch fails immediately | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` + proactive `empty_cache()` every 50 steps + filtered files >10s from dataset |
| 17 | **No stdout output when piped through `tee`** | Python buffers stdout when not writing to a TTY; all `print()` output held in buffer for minutes | `PYTHONUNBUFFERED=1` env var + `flush=True` on all `print()` calls |

---

### 🏆 The Most Painful Bug (Bug #10)

Bug #10 wasted **~8 hours of compute** and is the most insidious failure mode in ML training loops.

The training appeared to work perfectly — epochs completed, checkpoints saved every 5 epochs, validation losses printed with realistic-looking numbers. But the model weights never changed. The tell was `Dur loss: 1.746000` being **bit-for-bit identical** across all 14 epochs. A frozen loss means frozen weights.

Root cause: the guard `if gt.size(-1) < 80: continue` compared mel frame count against a threshold that was larger than the maximum achievable clip size given `max_len=70` in the config. Every training batch was silently discarded. Validation doesn't have this guard, so it ran fine and printed plausible-looking losses.

**Lesson for future cross-lingual fine-tuners:** Always log `skipped_batches` per epoch. If it equals total batches, nothing trained. Add a sanity check that at least one batch updates weights before trusting validation metrics.

---

### 🔥 The Second Most Painful (Bug #8)

During debugging of the `RuntimeError: Expected size for first two dimensions of batch2 tensor` error (which was actually caused by the BOS slice removal), the fix attempted was a "double transpose" that is mathematically an identity operation. This meant the fix was applied, the error disappeared (because the reshape now matched), but every single batch then failed silently at the matmul stage and was caught by the outer exception handler.

The code looked reasonable, ran without crashing, and produced no visible errors — it just trained on zero batches per epoch.

---

## 🙏 Acknowledgements

- [AI4Bharat](https://ai4bharat.iitm.ac.in/) for the IndicVoices-R dataset
- [yl4579](https://github.com/yl4579/StyleTTS2) for the StyleTTS2 training framework
- [hexgrad](https://huggingface.co/hexgrad/Kokoro-82M) for Kokoro-82M (inference pipeline reference)
- [StyleTTS2 paper](https://arxiv.org/abs/2306.07691) — Li et al., 2023
- [mychen76](https://huggingface.co/mychen76/styletts2) for ASR + JDC utility weights (not on official HF repos)

---

## 📄 License

Apache 2.0 — free to use, modify, and deploy commercially.

---

## 🤝 Contributing

Contributions welcome! If you speak Hindi natively and want to help evaluate naturalness (MOS scoring), please open an issue. Voice sample donations for future multi-speaker training also welcome.

---

*Built in Hyderabad 🇮🇳 — making Hindi voice AI accessible to everyone, everywhere, offline.*
