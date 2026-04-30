# XenonBrain: Advanced Logic Processing & Pattern Recognition System

<div align="center">
  <img src="https://img.shields.io/badge/Developer-Mohamed%20Ashraf-blue?style=for-the-badge&logo=github" alt="Developer: Mohamed Ashraf">
  <img src="https://img.shields.io/badge/AI-Transformer--based-orange?style=for-the-badge" alt="AI: Transformer-based">
  <img src="https://img.shields.io/badge/Status-Production--Ready-green?style=for-the-badge" alt="Status: Production-Ready">
  <img src="https://img.shields.io/badge/Automation-GitHub%20Actions-blueviolet?style=for-the-badge&logo=github-actions" alt="Automation: GitHub Actions">
</div>

---

## 🚀 Overview

**XenonBrain** is a cutting-edge AI engine developed by **[Mohamed Ashraf](https://github.com/Mohamed-Ashraf-Ai-Dev)**. It is specifically engineered for **Advanced Logic Processing** and **Automated Pattern Recognition**. Unlike conventional models, XenonBrain bridges the gap between raw data and logical inference by integrating real-world multi-modal data streams, including Global Technology News and Financial Market Indicators.

The system is **100% autonomous**, utilizing a sophisticated CI/CD pipeline via GitHub Actions for continuous data ingestion, automated training, and seamless model deployment.

---

## 🧠 Core Architecture

Developed with a focus on high-performance logic extraction, XenonBrain utilizes a deep **Transformer-based Encoder** architecture:

*   **Contextual Awareness:** Leverages `Sentence-Transformers` to translate global tech news into high-dimensional semantic embeddings, providing the "brain" with a sense of current global trends.
*   **Pattern Recognition:** A multi-layered Transformer architecture (6 layers) processes sequences of combined textual and numerical data to identify non-linear correlations that traditional algorithms miss.
*   **Logic Gate:** A proprietary Multi-Layer Perceptron (MLP) head that executes final logical classifications and predictive inferences.

---

## 📊 Real-World Data Integration

XenonBrain doesn't play with "dummy" data. It is hardwired to live, high-impact data streams:

| Source Type | Provider | Description |
| :--- | :--- | :--- |
| **Tech Intelligence** | MIT AI, TechCrunch, The Verge | Real-time RSS feeds for global AI and tech breakthroughs. |
| **Market Indicators** | Yahoo Finance (S&P 500) | Live financial indices used as a baseline for global economic patterns. |
| **Temporal Context** | Sequential Windowing | Processes data in 5-step time windows to understand evolution and momentum. |

---

## ⚙️ Fully Automated Training (The "Perpetual Brain")

The entire lifecycle of XenonBrain is governed by an automated workflow (`.github/workflows/main_training.yml`):

1.  **Scheduled Pulse:** Runs automatically every day at 00:00 UTC.
2.  **Environment Sync:** Auto-provisions a high-performance Python 3.11 environment.
3.  **Data Harvesting:** Fetches the most recent 30 days of market data and the latest news headlines.
4.  **Continuous Learning:** Retrains the core engine using the `AdamW` optimizer with `StepLR` scheduling for maximum stability.
5.  **Auto-Deployment:** Commits and pushes the evolved `.pth` model back to the repository, ensuring the brain is always at its peak performance.

---

## 🛠️ Quick Start

### Installation
```bash
git clone https://github.com/Mohamed-Ashraf-Ai-Dev/XenonBrain.git
cd XenonBrain
pip install -r requirements.txt
```

### Manual Training
If you wish to force a training cycle manually:
```bash
python src/training/train.py
```

### Inference (Logical Prediction)
To use the trained brain for real-time logical pattern recognition:
```bash
python src/inference/predict.py
```

---

## 📁 Project Structure

```
XenonBrain/
├── .github/workflows/main_training.yml  # The Automation Heart
├── src/
│   ├── models/xenon_model.py            # Transformer Architecture
│   ├── data/data_processor.py           # Real-time Data Collector
│   ├── training/train.py                # Production Training Loop
│   └── inference/predict.py             # Inference Engine
├── config/config.yaml                   # System Hyperparameters
├── requirements.txt                     # Production Dependencies
└── README.md                            # Documentation
```

---

## 👨‍💻 Developer

**Mohamed Ashraf**
*   **GitHub:** [Mohamed-Ashraf-Ai-Dev](https://github.com/Mohamed-Ashraf-Ai-Dev)
*   **Role:** Lead AI Developer & Architect

---

## 📜 License
This project is open-source and designed for advanced AI research and automated pattern recognition. Built with passion for the future of autonomous logic.
