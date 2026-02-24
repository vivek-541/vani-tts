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
| Veena (Maya Research) | ✅ Excellent | ❌ Needs GPU | ❌ | ❌ |
| AI4Bharat Indic Parler-TTS | ✅ Very Good | ❌ 0.9B params | ❌ | ✅ |
| Piper TTS (hi) | ⚠️ Poor | ✅ | ✅ | ✅ |
| **Vani TTS** | ✅ **Good→Great** | ✅ **Yes** | ✅ **Yes** | ✅ **Yes** |

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
│   └── prepare_data.py         # Dataset streaming & preprocessing
├── training/
│   └── finetune_kokoro.py      # Kokoro TTS fine-tuning
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
git clone https://github.com/vivek-541/vani-tts.git
cd vani-tts

# Install dependencies
pip install -r requirements.txt

# Step 1: Prepare dataset
python data/prepare_data.py

# Step 2: Verify baseline Kokoro Hindi voice
python -c "
from kokoro import KPipeline
import soundfile as sf
pipe = KPipeline(lang_code='h')
audio, sr = pipe('नमस्ते, मेरा नाम वाणी है।')
sf.write('baseline.wav', audio, sr)
print('Baseline saved to baseline.wav')
"

# Step 3: Fine-tune
python training/finetune_kokoro.py

# Step 4: Export to ONNX
python export/export_onnx.py
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | [ai4bharat/indicvoices_r](https://huggingface.co/datasets/ai4bharat/indicvoices_r) |
| Language | Hindi (hi) |
| Sample Rate | 24000 Hz |
| Training Samples | 15,000 |
| Duration Filter | 1.0s – 12.0s |
| Normalization | -20 dB RMS |
| Download Method | Streaming (no full download needed) |

---

## 🏗️ Architecture

Vani TTS is fine-tuned on **[Kokoro TTS](https://huggingface.co/hexgrad/Kokoro-82M)** — an 82M parameter model based on the **StyleTTS2** architecture with an iSTFT decoder:

```
Text → Misaki G2P (Hindi phonemes) → Style Encoder → Decoder → iSTFT → Audio
```

- **Why Kokoro over Piper?** Kokoro uses a modern StyleTTS2 architecture vs Piper's 2021-era VITS — significantly better quality ceiling, especially for prosody and naturalness
- **Why not Piper?** Piper uses espeak-ng for Hindi phonemization, which produces incorrect stress patterns and mechanical pauses on Devanagari text. The Hindi model was added late (August 2025) and the repo was archived shortly after — no active development
- **Why not larger models (Parler-TTS, Veena)?** At 0.9B–3B parameters, they require GPU inference and cannot run on mobile CPUs in real-time
- **Misaki G2P** — Kokoro's phonemizer has native Hindi support, handling Devanagari script correctly without transliteration

---

## 📅 Roadmap

- [x] Phase 0 — Architecture research & decision (Kokoro > Piper)
- [x] Phase 1 — Dataset pipeline (IndicVoices-R, streaming, 24kHz)
- [ ] Phase 2 — Kokoro baseline evaluation on Hindi
- [ ] Phase 3 — Fine-tuning on 15k IndicVoices-R samples
- [ ] Phase 4 — Evaluation (MOS score, WER, RTF)
- [ ] Phase 5 — ONNX export & INT8 quantization
- [ ] Phase 6 — Android integration (ONNX Runtime)
- [ ] Phase 7 — iOS integration (CoreML)
- [ ] Phase 8 — Multiple Hindi voices (male/female)
- [ ] Phase 9 — Hinglish support (code-switching)

---

## 📈 Evaluation (Target)

| Metric | Target | Current |
|---|---|---|
| MOS Score | > 3.8 / 5.0 | WIP |
| Word Error Rate (WER) | < 8% | WIP |
| Real-Time Factor (CPU) | < 0.3x | WIP |
| Model Size | < 200 MB | WIP |
| Android Latency | < 300ms/sec audio | WIP |

---

## 🙏 Acknowledgements

- [AI4Bharat](https://ai4bharat.iitm.ac.in/) for the IndicVoices-R dataset
- [Kokoro TTS](https://huggingface.co/hexgrad/Kokoro-82M) by hexgrad for the base model
- [StyleTTS2](https://arxiv.org/abs/2306.07691) — Li et al., 2023
- [Misaki G2P](https://github.com/hexgrad/misaki) for Hindi phonemization

---

## 📄 License

Apache 2.0 — free to use, modify, and deploy commercially.

---

## 🤝 Contributing

Contributions are welcome! If you speak Hindi natively and want to donate voice samples or help evaluate naturalness, please open an issue.

---

*Built in Hyderabad 🇮🇳 with the goal of making Hindi voice AI accessible to everyone, everywhere, offline.*
