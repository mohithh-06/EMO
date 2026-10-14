# 👔 emo — Multimodal Deep Learning Outfit Critic & AI Wardrobe Stylist

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![Hugging Face Datasets](https://img.shields.io/badge/🤗%20Hugging%20Face-Datasets-blue.svg)](https://huggingface.co/datasets/mvasil/polyvore-outfits)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**emo** is a state-of-the-art multimodal deep learning outfit critic and automated AI wardrobe stylist that synthesizes vision-language foundation representations with classical sartorial colorimetry, silhouette proportion laws, and multi-task neural classification.

---

## 👥 Project Team & Academic Guidance

- **G. Harisha** (23EG110D24)
- **J. Uday Kumar** (23EG110D27)
- **Mohith Ch** (23EG110D44)

**Under the Guidance of:**  
**B. Jyothi**, Assistant Professor  
*Department of Data Science, Anurag University, Hyderabad*  
**Domain:** Deep Learning, Computer Vision & Multimodal Representation Learning

---

## 🚀 Key Technical Highlights & Innovations

### 1. 👁️ Two-Stage Vision Decoupling (Zero Ethnic / Leather Hallucination)
- **Problem:** Naive full-frame classifiers frequently confuse South Asian skin tones and domestic walls with ethnic kurta sets or leather motorcycle jackets on casual wear.
- **Solution:** An automated dual-stream inference pipeline where full-frame images provide scene context while a calibrated **lower-torso crop (lower 55% ROI)** strictly isolates garment textiles for pure attribute extraction.

### 2. 🧠 Custom PyTorch Multi-Task Head (`EmoFashionNeuralHead`)
- Deep residual MLP backbone (`512 -> 384 -> 192`) with Batch Normalization, LeakyReLU activations, and Dropout (`p=0.25`).
- **Concurrent Joint Prediction:** Simultaneously classifies Gender Presentation (3 classes), 10-class Garment Taxonomy, Occasion Formality (4 tiers), and Fabric Textures (5 tactile types).
- **Benchmark Performance:**
  - **Top-1 Multi-Task Accuracy:** **98.8%**
  - **Top-3 Multi-Task Accuracy:** **100.0%**
  - **Neural Inference Latency:** **1.38 ms** (Total end-to-end pipeline: **205.7 ms** on standard CPU).

### 3. 🤗 Official Hugging Face Hub Dataset Grounding
- **`mvasil/polyvore-outfits`**: Trained on 25,132 high-resolution fashion items with relational outfit graphs and fine-grained categories.
- **`formido/outfit_recomendation`**: Ingests 3,193 expert instruction-tuning styling profiles and curated editorial capsules directly into the recommendation engine.

### 4. 🎨 CIE-LAB K-Means Engine & W3C Relative Luminance
- Quantizes dominant colors in perceptually uniform **CIE-LAB space** with adaptive background wall subtraction.
- Computes mathematical optical contrast ratios via ITU-R BT.709 relative luminance formulas ($Y = 0.2126R + 0.7152G + 0.0722B$) to prevent duplicate color tags (e.g. Heather Grey vs Slate Grey).

### 5. 🛠️ 3 Strategic emo Styling Pivots & 100-Point Scorecard
- Evaluates looks across 5 weighted dimensions: Silhouette (30%), Color Harmony (25%), Texture Depth (15%), Occasion Fit (15%), and Polish (15%).
- Generates 3 evidence-grounded tactical upgrades complete with target hex color chips and styling laws (e.g., The 60-30-10 Rule, Sandwich Layering, Footwear Visual Weight).

---

## 📊 Experimental Evaluation & Benchmarks

| Model / Architecture | Garment Top-1 Acc | Formality Acc | Fabric Texture Acc | Overall Multi-Task | Inference Latency |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Naive Majority Class** | 35.0% | 45.0% | 31.7% | 44.6% | 0.05 ms |
| **Zero-Shot CLIP (ViT-B/32)** | 78.3% | 81.7% | 75.0% | 78.4% | 185.2 ms |
| **emo Multi-Task Neural Head (Ours)** | **98.3%** | **98.3%** | **100.0%** | **98.8%** | **1.38 ms** |

*Evaluated on an independent held-out test set ($N=60$ samples) with AdamW optimizer and Cosine Annealing learning rate scheduling.*

---

## 🗂️ Project Architecture & File Hierarchy

```
.
├── .streamlit/
│   └── config.toml                   # Production theme, security tokens & CORS settings
├── fitcheck/
│   ├── Fashion AI Advisor.py         # Main Haute Editorial Dashboard & Camera Studio
│   ├── design_system.py              # Design tokens, semantic colors & WCAG AA components
│   ├── vision_engine.py              # Two-stage torso crop & CLIP ViT-B/32 embedder
│   ├── model_architecture.py         # PyTorch EmoFashionNeuralHead multi-task network
│   ├── train_model.py                # Multi-task training pipeline with AdamW & Cosine Annealing
│   ├── evaluate_models.py            # Comprehensive test evaluation & baseline benchmark suite
│   ├── hf_datasets_loader.py         # Hugging Face Hub dataset loader (Polyvore & Formido)
│   ├── color_engine.py               # CIE-LAB K-Means, wall subtraction & W3C contrast ratios
│   ├── critique_generator.py         # 100-pt scorecard & 3 evidence-grounded styling pivots
│   ├── tagging.py                    # Deep-vision closet tagging engine
│   ├── models/
│   │   ├── emo_multitask_net.pt      # Trained PyTorch model weights (98.8% accuracy)
│   │   └── evaluation_metrics.json   # Full benchmark evaluation metrics
│   └── pages/
│       ├── 1_Add_to_Inventory.py     # Wardrobe Digitizer with local-first privacy
│       ├── 2_Get_Outfit_Suggestion.py# AI Stylist Deck & Hugging Face Editorial Capsules
│       └── 3_Model_Training_&_Evaluation.py # Interactive DL Training Lab & Confusion Matrix
├── Closet/                           # Digitized personal wardrobe database
├── Images/                           # Curated fashion test assets
├── Dockerfile                        # Multi-stage container definition for Cloud deployment
├── requirements.txt                  # Production dependencies
├── run_app.bat                       # 1-Click offline Windows launcher
└── README.md
```

---

## 💻 Local Installation & Setup

### Prerequisites
- Python 3.10+ (Tested on Python 3.12)
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/mohithh-06/EMO.git
cd EMO
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Hugging Face Token (Optional for Private Hub)
```bash
# Windows PowerShell
$env:HF_TOKEN="your_huggingface_token"

# Linux / macOS
export HF_TOKEN="your_huggingface_token"
```

### 5. Launch the Application
```bash
streamlit run "fitcheck/Fashion AI Advisor.py"
```
The application will launch on `http://localhost:8501`.

---

## ☁️ Deployment Guide

### Option A: Streamlit Community Cloud (Recommended)
1. Go to **[share.streamlit.io](https://share.streamlit.io/)** and sign in with GitHub.
2. Click **"New app"**.
3. Select your repository: **`mohithh-06/EMO`**, Branch: `main`, Main file path: `fitcheck/Fashion AI Advisor.py`.
4. In **Advanced Settings -> Secrets**, add:
   ```toml
   HF_TOKEN = "your_huggingface_token"
   ```
5. Click **"Deploy!"**.

### Option B: Docker Container
```bash
# Build Docker image
docker build -t emo-fashion-ai .

# Run container locally on port 8501
docker run -p 8501:8501 -e HF_TOKEN="your_token" emo-fashion-ai
```

---

## 🔒 Privacy & Data Ethics Policy

- **Truthful Privacy Architecture:** All deep learning vision passes and neural classifications run **locally in-memory**.
- **Ephemeral Camera Buffer:** Camera captures and uploaded outfit selfies are held strictly in an ephemeral RAM buffer during inference and are automatically purged upon session reset or browser close.
- **No Third-Party Transmission:** Personal photos are never sent to external commercial vision APIs or third-party cloud trackers.

---

## 📄 References

1. A. Radford et al., *"Learning Transferable Visual Models From Natural Language Supervision (CLIP)"*, ICML 2021.
2. Z. Liu et al., *"DeepFashion: Powering Robust Clothes Recognition and Retrieval with Rich Annotations"*, CVPR 2016.
3. M. I. Vasileva et al., *"Learning Type-Aware Embeddings for Fashion Collection Retrieval (Polyvore Outfits)"*, ECCV 2018.
4. R. Caruana, *"Multitask Learning"*, Machine Learning, 1997.
5. World Wide Web Consortium (W3C), *"Web Content Accessibility Guidelines (WCAG) 2.1: Relative Luminance & Contrast"*, 2018.

---

## 📜 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
