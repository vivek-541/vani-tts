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
- 🎙️ **Natural Hindi voice** — fine-tuned on 15,000 studio-grade Hindi speech samples
- 📦 **ONNX export** — deploy on Android (ONNX Runtime) or iOS (CoreML)
- 🔡 **Devanagari native** — handles Hindi script directly via espeak-ng IPA phonemization
- ⚖️ **Lightweight** — target model size under 200MB

---

## 🗂️ Project Structure

```
vani-tts/
├── data/
│   └── prepare_data.py         # Dataset streaming & preprocessing (IndicVoices-R)
├── training/
│   └── StyleTTS2/              # StyleTTS2 fine-tuning repo (yl4579/StyleTTS2)
│       ├── train_finetune.py   # Main training script
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

- **StyleTTS2 over Piper** — StyleTTS2 uses adversarial style diffusion for naturalness vs Piper's 2021-era VITS; significantly better prosody and expressiveness ceiling
- **HiFiGAN decoder** — matches the LibriTTS pretrained base checkpoint; higher quality than iSTFT for fine-tuning from English pretrain
- **espeak-ng phonemizer** — handles Hindi IPA correctly via `phonemizer` library with `backend='espeak', language='hi'`; 49 unique Hindi phoneme tokens
- **Single speaker** — trained on a curated single-voice Hindi subset for maximum voice consistency
- **Why not larger models (Parler-TTS, Veena)?** At 0.9B–3B parameters, they require GPU inference and cannot run on mobile CPUs in real-time

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | [ai4bharat/indicvoices_r](https://huggingface.co/datasets/ai4bharat/indicvoices_r) |
| Language | Hindi (hi) |
| Sample Rate | 24,000 Hz |
| Training Samples | 5,000 (curated from 14,250) |
| Validation Samples | 250 |
| Duration Filter | 1.0s – 12.0s |
| Normalization | −20 dB RMS |
| Phoneme Tokens | 49 unique IPA tokens |
| Download Method | HF streaming + soundfile decode (no torchcodec required) |

---

## ⚙️ Training Configuration

| Parameter | Value |
|---|---|
| Base checkpoint | LibriTTS `epochs_2nd_00020.pth` (736MB) |
| Epochs | 50 |
| Batch size | 2 |
| Max sequence length | 160 frames |
| Sample rate | 24,000 Hz |
| Decoder | HiFiGAN |
| Mixed precision | AMP fp16 |
| Hardware | NVIDIA RTX 3060 12GB + Intel i3-8100 |
| Estimated training time | ~46 hours |
| Step time | ~1.3s/step |

---

## 📅 Roadmap

- [x] Phase 0 — Environment setup (Ubuntu 24.04, CUDA 13.0, UV venv)
- [x] Phase 1 — Dataset pipeline (15k IndicVoices-R samples, 24kHz, −20dB RMS)
- [x] Phase 2 — Phonemization (espeak-ng IPA, 49 tokens, 14,250 train / 750 val)
- [x] Phase 3 — Pretrained weights + StyleTTS2 config (LibriTTS base, HiFiGAN decoder)
- [x] Phase 4 — Training loop stabilized (10 bugs fixed — see Engineering Decisions below)
- [ ] **Phase 5 — 50 epochs training** ← 🔄 IN PROGRESS (Est. completion: ~46h from Mar 1)
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
| MOS Score | > 3.8 / 5.0 | 🔄 Training |
| Word Error Rate (WER) | < 8% | 🔄 Training |
| Real-Time Factor (CPU) | < 0.3x | 🔄 Training |
| Model Size (quantized) | < 200 MB | 🔄 Training |
| Android Latency | < 300ms/sec audio | 🔄 Training |

---

## 🛠️ Engineering Decisions & Bug Chronicle

Getting StyleTTS2 to train on Hindi required solving 12 distinct bugs. Documented here for anyone attempting similar cross-lingual fine-tuning.

### Environment & Dependencies

| # | Problem | Root Cause | Fix |
|---|---|---|---|
| 1 | `torchcodec` import error on dataset load | HuggingFace `datasets` switched default audio backend to torchcodec in new versions | `Audio(decode=False)` + manual soundfile decode |
| 2 | `misaki` has no Hindi module | `misaki` phonemizer only supports: en, ja, ko, zh, he, vi, espeak-wrappers — not Hindi directly | Used `phonemizer` library with `backend='espeak', language='hi'` |
| 3 | `monotonic_align` Cython compile fails | StyleTTS2 ships monotonic_align as a Cython extension requiring compilation | Pure Python fallback implementation at `monotonic_align/core.py` |
| 4 | `torch.load` raises `weights_only` error | PyTorch 2.6 changed `weights_only` default from `False` to `True` | Added `weights_only=False` to all `torch.load` calls in `models.py` |
| 5 | Decoder architecture mismatch at checkpoint load | LibriTTS pretrained checkpoint uses HiFiGAN decoder; config defaulted to iSTFT | Set `decoder.type: hifigan` in `config_ft.yml`; removed iSTFT-specific params |

### Training Loop Bugs

| # | Problem | Root Cause | Fix |
|---|---|---|---|
| 6 | `mask_from_lens` signature mismatch | Our pure-Python implementation only accepted 2 args; StyleTTS2 calls it with 3 | Updated `monotonic_align/__init__.py` to handle 3-arg call with tensor step |
| 7 | `maximum_path` IndexError on `mask.sum(1)[:, 0]` | Fails when `mask.sum(1)` is 1D with a single element | Guarded with `_mask_sum[:, 0] if _mask_sum.dim() > 1 else _mask_sum` in `utils.py` |
| 8 | **Every batch silently skipped — model never trained** | Attention matrix slice `s2s_attn[..., 1:]` (BOS token removal) was accidentally deleted during debugging. Double no-op transpose replaced it. Without the slice, `s2s_attn` has text-dim N but `t_en` has dim N-1 → matmul shape mismatch on every batch → caught by except → skipped | Restored the 3-line canonical attention transform: `transpose → [..1:] → transpose` |
| 9 | **`skipped_batches` NameError crash** | Variable used before assignment — `skipped_batches += 1` inside the alignment try-block but `skipped_batches = 0` was never initialized before the batch loop | Added `skipped_batches = 0` at the top of each epoch loop |
| 10 | **`gt.size(-1) < 80` skipped every batch silently** | `max_len=70` → `mel_len = max_len//2 = 35` → `gt frames = 35×2 = 70` → `70 < 80` → skip. The guard threshold was larger than the maximum possible clip size, so 100% of batches were discarded after passing alignment | Raised `max_len` to `160` (giving `gt=160 frames`) and lowered guard to `< 40` |
| 11 | **Cascade OOM** — hundreds of consecutive batches fail after first OOM | PyTorch CUDA allocator fragments on OOM; `empty_cache()` in the except block doesn't defragment | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` + proactive `empty_cache()` every 50 batches |
| 12 | **stdout buffering** — no output when piped through `tee` | Python buffers stdout when not writing to a TTY | `PYTHONUNBUFFERED=1` env var + `flush=True` on all `print()` calls |

### The Most Painful Bug (Bug #10)

Bug #10 wasted ~8 hours of compute. The training appeared to run — epochs completed, checkpoints saved, validation losses printed — but the model weights never changed. The tell was `Dur loss: 1.746` being **identical** across all 14 epochs. A frozen loss means frozen weights. The batch loop was executing in seconds (only validation was running). The root cause: a minimum frame-length guard `if gt.size(-1) < 80: continue` was set larger than the maximum achievable clip size given the `max_len` config parameter, silently discarding every training batch while validation (which doesn't have this guard) ran fine.

**Lesson:** Always log `skipped_batches` per epoch. If it equals total batches, nothing trained.

---

## 🙏 Acknowledgements

- [AI4Bharat](https://ai4bharat.iitm.ac.in/) for the IndicVoices-R dataset
- [yl4579](https://github.com/yl4579/StyleTTS2) for the StyleTTS2 training framework
- [hexgrad](https://huggingface.co/hexgrad/Kokoro-82M) for Kokoro-82M (inference pipeline)
- [StyleTTS2 paper](https://arxiv.org/abs/2306.07691) — Li et al., 2023
- [mychen76](https://huggingface.co/mychen76/styletts2) for ASR + JDC utility weights

---

## 📄 License

Apache 2.0 — free to use, modify, and deploy commercially.

---

## 🤝 Contributing

Contributions welcome! If you speak Hindi natively and want to help evaluate naturalness (MOS scoring), please open an issue. Voice sample donations for future multi-speaker training also welcome.

---

*Built in Hyderabad 🇮🇳 — making Hindi voice AI accessible to everyone, everywhere, offline.*
