# वाणी (Vani) TTS 🎙️
### Lightweight Hindi Text-to-Speech for Consumer Devices

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Platform: Android | iOS | CPU](https://img.shields.io/badge/platform-Android%20%7C%20iOS%20%7C%20CPU-green.svg)]()
[![HuggingFace Dataset](https://img.shields.io/badge/dataset-ai4bharat%2Findicvoices__r-yellow.svg)](https://huggingface.co/datasets/ai4bharat/indicvoices_r)

> **वाणी** (vāṇī) — Sanskrit for *voice, speech, the goddess of language.*

Vani TTS is an open-source, on-device Hindi Text-to-Speech model fine-tuned on the [AI4Bharat IndicVoices-R](https://huggingface.co/datasets/ai4bharat/indicvoices_r) dataset. It is designed to run in **real-time on CPU** — no internet, no GPU, no cloud — making it suitable for Android phones, iOS devices, and low-end laptops.

---

## 🌍 Why Vani?

| Model | Hindi Quality | On-Device | Mobile Ready | Open Source |
|---|---|---|---|---|
| Google TTS | ✅ Good | ❌ Cloud only | ❌ | ❌ |
| Veena (Maya Research) | ✅ Excellent | ❌ No CPU support | ❌ | ❌ |
| AI4Bharat Indic-TTS | ✅ Good | ⚠️ Heavy | ❌ | ✅ |
| Piper TTS (hi) | ⚠️ Poor | ✅ | ✅ | ✅ |
| **Vani TTS** | ✅ **Good** | ✅ **Yes** | ✅ **Yes** | ✅ **Yes** |

**Vani fills the gap: quality Hindi TTS that runs on your phone, offline, for free.**

---

## ✨ Features

- 🏃 **Real-time on CPU** — runs on any Android/iOS device
- 📴 **Fully offline** — no internet connection required
- 🎙️ **Natural Hindi voice** — fine-tuned on studio-grade speech data
- 📦 **ONNX export** — deploy on Android (ONNX Runtime) or iOS (CoreML)
- 🔡 **Devanagari native** — handles Hindi script directly, no transliteration needed
- ⚖️ **Lightweight** — target model size under 200MB

---

## 🗂️ Project Structure

```
vani-tts/
├── data/
│   └── phase1_dataset.py       # Dataset exploration & preprocessing
├── training/
│   └── finetune_piper.py       # Piper TTS fine-tuning
├── evaluation/
│   └── evaluate.py             # MOS, WER evaluation
├── export/
│   ├── export_onnx.py          # Export to ONNX
│   └── export_coreml.py        # Export to CoreML (iOS)
├── android/                    # Android demo app (coming soon)
├── ios/                        # iOS demo app (coming soon)
├── configs/
│   └── vani_config.json        # Training configuration
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/vani-tts.git
cd vani-tts

# Install dependencies
pip install -r requirements.txt

# Step 1: Prepare dataset
python data/phase1_dataset.py

# Step 2: Fine-tune
python training/finetune_piper.py

# Step 3: Export to ONNX
python export/export_onnx.py
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | [ai4bharat/indicvoices_r](https://huggingface.co/datasets/ai4bharat/indicvoices_r) |
| Language | Hindi (hi) |
| Sample Rate | 22050 Hz |
| Training Samples | 5,000–20,000 |
| Duration Filter | 0.5s – 10.0s |
| Normalization | -20 dB RMS |

---

## 🏗️ Architecture

Vani TTS is based on **[Piper TTS](https://github.com/rhasspy/piper)** which uses the **VITS** (Variational Inference with adversarial learning for end-to-end Text-to-Speech) architecture:

```
Text → Phonemizer → VITS Encoder → Flow → HiFi-GAN Vocoder → Audio
```

- **Why VITS?** Single-stage end-to-end training, faster than two-stage systems
- **Why Piper?** Designed for on-device CPU inference, ONNX export built-in
- **Why not Whisper/LLM-based TTS?** Too large for mobile (1B+ parameters)

---

## 📅 Roadmap

- [x] Phase 1 — Dataset exploration & preprocessing
- [ ] Phase 2 — Piper TTS fine-tuning on Hindi
- [ ] Phase 3 — Evaluation (MOS score, naturalness)
- [ ] Phase 4 — ONNX export & optimization
- [ ] Phase 5 — Android integration (ONNX Runtime)
- [ ] Phase 6 — iOS integration (CoreML)
- [ ] Phase 7 — Multiple Hindi voices (male/female)
- [ ] Phase 8 — Hinglish support (code-switching)

---

## 📈 Evaluation (Target)

| Metric | Target | Current |
|---|---|---|
| MOS Score | > 3.5 / 5.0 | WIP |
| Real-Time Factor (CPU) | < 1.0 | WIP |
| Model Size | < 200 MB | WIP |
| Android Latency | < 500ms | WIP |

---

## 🙏 Acknowledgements

- [AI4Bharat](https://ai4bharat.iitm.ac.in/) for the IndicVoices-R dataset
- [Piper TTS](https://github.com/rhasspy/piper) by rhasspy for the base architecture
- [VITS Paper](https://arxiv.org/abs/2106.06103) — Kim et al., 2021

---

## 📄 License

Apache 2.0 — free to use, modify, and deploy commercially.

---

## 🤝 Contributing

Contributions are welcome! If you speak Hindi natively and want to donate voice samples or help evaluate naturalness, please open an issue.

---

*Built in Hyderabad 🇮🇳 with the goal of making Hindi voice AI accessible to everyone, everywhere, offline.*
