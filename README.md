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
| Training Samples | 14,250 |
| Validation Samples | 750 |
| Total Samples | 15,000 |
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
| Batch size | 1–4 (GPU memory dependent) |
| Sample rate | 24,000 Hz |
| Decoder | HiFiGAN |
| Mixed precision | AMP fp16 |
| Hardware used | NVIDIA RTX 3060 12GB |
| Estimated training time | 48–72 hours |

---

## 📅 Roadmap

- [x] Phase 0 — Environment setup (Ubuntu 24.04, CUDA 13.0, UV venv)
- [x] Phase 1 — Dataset pipeline (15k IndicVoices-R samples, 24kHz, −20dB RMS)
- [x] Phase 2 — Phonemization (espeak-ng IPA, 49 tokens, 14,250 train / 750 val)
- [x] Phase 3 — Pretrained weights + StyleTTS2 config (LibriTTS base, HiFiGAN decoder)
- [x] Phase 4 — Training loop stabilized (bug fixes: monotonic_align, mask_from_lens, AMP)
- [ ] **Phase 5 — 50 epochs training** ← 🔄 IN PROGRESS
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

## 🛠️ Notable Engineering Decisions

| Problem | Solution |
|---|---|
| `torchcodec` missing in new datasets | `Audio(decode=False)` + manual soundfile decode |
| `misaki` has no Hindi module | `phonemizer` with `backend='espeak', language='hi'` |
| `monotonic_align` needs Cython compile | Pure Python fallback implementation |
| PyTorch 2.6 `weights_only` default changed | Added `weights_only=False` to all `torch.load` calls |
| LibriTTS checkpoint uses HiFiGAN not iSTFT | Set `decoder.type: hifigan` in config |
| RTX 3060 12GB VRAM with full StyleTTS2 | AMP fp16 + `batch_size=1` + `max_len=150` |

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
