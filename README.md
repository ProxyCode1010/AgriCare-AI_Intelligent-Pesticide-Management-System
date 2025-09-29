# 🌾 AgriCare AI - Intelligent Pesticide Management System

**AI-Powered Plant Disease Detection & Precision Pesticide Recommendation for Sustainable Farming**

[🎥 **Watch Demo**](https://youtu.be/7oitnCcwMUE?feature=shared)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Key Features](#-key-features)
- [Technology Stack](#-technology-stack)
- [System Architecture](#-system-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Performance](#-model-performance)
- [Analytics Dashboard](#-analytics-dashboard)
- [Impact & Benefits](#-impact--benefits)
- [Future Scope](#-future-scope)
- [License](#-license)
- [Developed During SIH 2025](#-developed-during-sih-2025)
- [Conclusion](#-conclusion)

---

## 🌟 Overview

**AgriCare AI** is an intelligent pesticide management system that uses deep learning and IoT-inspired integration to help farmers detect plant diseases early and apply pesticides only where necessary.

Our solution **reduces pesticide overuse**, **cuts costs**, **improves crop yield**, and **promotes sustainable agriculture** by combining AI-driven plant disease detection with precision recommendations.

---

## 🚨 Problem Statement

Traditional pesticide spraying techniques apply chemicals uniformly, regardless of actual infection. This leads to:

- 🌱 **Soil and water contamination**
- 🐝 **Harm to beneficial organisms**
- 👩‍🌾 **Health risks to farmers and consumers**
- 💰 **Economic losses from wastage**
- 🌍 **Long-term environmental damage**

---

## ✨ Key Features

### 🔍 AI Disease Detection
**38+ plant diseases detected with 97% accuracy**

### 🎯 Precision Pesticide Recommendation
**Smart dosage suggestions reduce usage by 60%**

### 📊 Interactive Dashboard
**Real-time KPIs, health tracking, and cost savings**

### 🤖 IoT Control Simulation
**Smart sprayer integration for targeted spraying**

### 🌍 Sustainable Farming Impact
**Lower chemical footprint, healthier crops**

---

## 🛠 Technology Stack

### AI/ML
- TensorFlow
- Keras
- OpenCV
- NumPy
- Pandas

### Dashboard
- Streamlit
- Plotly
- Pillow

### IoT (Simulated)
- Smart sprayer recommendations
- Telemetry monitoring

### Deployment
- Python 3.8+
- Streamlit

---

## 🏗 System Architecture

```
┌─────────────────────┐
│  📷 Plant Image     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  🧠 AI Model - CNN  │
└─────┬──────────┬────┘
      │          │
      ▼          ▼
┌──────────┐  ┌──────────────┐
│ 📊 Dash  │  │ 🤖 IoT Ctrl │
└──────────┘  └──────┬───────┘
                     │
                     ▼
              ┌──────────────┐
              │ 💧 Precision │
              │    Sprayer   │
              └──────────────┘
```

---

## 💻 Installation

### 1. Clone repository
```bash
git clone https://github.com/ProxyCode1010/AgriCare-AI_Intelligent-Pesticide-Management-System.git
cd AgriCare-AI_Intelligent-Pesticide-Management-System
```

### 2. Setup virtual environment
```bash
# Linux/Mac
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run application
```bash
streamlit run main.py
```

---

## ▶️ Usage

**Step 1:** Launch application
```bash
streamlit run main.py
```

**Step 2:** Upload plant leaf image

**Step 3:** View disease prediction + confidence

**Step 4:** Get recommended pesticide & dosage

**Step 5:** Explore analytics dashboard for insights

---

## 📈 Model Performance

### Dataset Information
- **Source:** Enhanced PlantVillage (~87,000 images)
- **Diseases:** 38 classes across 14 crops (Apple, Corn, Tomato, Potato, etc.)

### Performance Metrics

| Metric | Value |
|--------|-------|
| **Training Accuracy** | 97.02% |
| **Validation Accuracy** | 94.6% |
| **Precision** | 94.8% |
| **Recall** | 93.9% |
| **F1-Score (macro)** | 0.96 |
| **Model Size** | 25 MB (optimized) |

---

## 📊 Analytics Dashboard

### Features Include:

- 🌿 **Crop Health Monitoring** (Healthy vs Infected)
- 💧 **Pesticide Savings Calculator**
- 📈 **Disease Trend Analysis** (last 30 days)
- 🌍 **Environmental & Economic Impact Metrics**

---

## 🌍 Impact & Benefits

### Environmental Impact
✅ **60% less pesticide usage**  
✅ **Reduced soil/water contamination**

### Economic Benefits
✅ **40% cost savings for farmers**  
✅ **Improved long-term crop yield**

### Social Impact
✅ **Safer food & farmer health**  
✅ **Sustainable agriculture practices**

---

## 🚀 Future Scope

### Planned Enhancements:

- 🚁 **Drone-based live field monitoring**
- 📡 **IoT sensor + satellite data fusion**
- 🤖 **Reinforcement learning for adaptive control**
- 📱 **Dedicated farmer mobile app**

---

## 📄 License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

## 🎓 Developed During SIH 2025

This project was created as part of **Smart India Hackathon 2025**  
**(Problem ID: 25015, Government of Punjab)**

[📺 **Watch Demo Video**](https://youtu.be/7oitnCcwMUE?feature=shared)

---

## ✅ Conclusion

**AgriCare AI** demonstrates how **AI + IoT** can transform agriculture by:

- **Reducing chemical usage**
- **Lowering costs**
- **Protecting the environment**

With further scaling, it can empower millions of farmers to practice **sustainable precision farming** while ensuring **food security** and **environmental safety**.

---

**Made with 🌱 for a sustainable future**
```

**Instructions to use:**

1. Copy everything between the triple backticks (```markdown to ```)
2. Create a file named `README.md` in your repository root
3. Paste the content
4. Commit and push to GitHub

GitHub will automatically render it with proper formatting, headings, tables, and emojis.
