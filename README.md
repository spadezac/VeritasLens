# VeritasLens: A Hybrid-Context Ensemble Framework for Robust Fake News Detection

**VeritasLens** is a research-driven initiative focused on addressing the systemic challenges of digital misinformation. By leveraging a multi-track ensemble architecture, this project distinguishes between authentic reporting and deceptive content using deep learning models specialized in both semantic context and stylistic patterns.

---

## Research Team
* **Simran Gupta**
* **Bhuvanyu Geel**
* **Anipra Pandya**
* **Parth Sinha**
* **Aviral Yadav**

**Under the Guidance of:** **Dr. Adarsh Patel**

---

## Table of Contents
1. [Abstract](#-abstract)
2. [Key Features](#-key-features)
3. [Performance Benchmarks](#-performance-benchmarks)
4. [Technical Stack](#-technical-stack)
5. [Installation & Usage](#-installation--usage)
6. [Core Methodology](#-core-methodology)
7. [How it Works: The X-Ray Logic](#-how-it-works-the-x-ray-logic)
8. [Acknowledgments](#-acknowledgments)
9. [License](#-license)

---

## Abstract
The rapid spread of "Fake News" poses a significant threat to information integrity. Our research demonstrates that no single model is universally effective; instead, a **Hybrid-Context Ensemble** provides a superior "safety net." By ensembling a **High-Context Specialist (BERT)** for deep semantic analysis with a **Low-Context Specialist (CNN/RoBERTa)** for stylistic cues, we mitigate the biases inherent in single-model systems. 

Key innovations include the use of **Abstractive Summarization (T5/PEGASUS)** to optimize inference and **Explainable AI (SHAP/LIME)** to demystify "black-box" neural decisions.

---

## Key Features
* **Dual-Track Ensemble:** Simultaneously analyzes full-text semantics and headline/summary style to resolve specialist disagreements.
* **Linguistic X-Ray:** Real-time highlighting of deceptive markers (e.g., hedging terms like "allegedly") vs. factual markers (e.g., datelines like "Reuters").
* **Cross-Dataset Validation:** Rigorously tested on ISOT, LIAR, TI-CNN, and Fake News Corpus (FNC).
* **Resource Efficiency:** Summarization-optimized logic allows for **99% accuracy retention** with a **1.8x reduction** in computational overhead.

---

## Performance Benchmarks

| Dataset | Model Architecture | Accuracy | Key Insight |
| :--- | :--- | :--- | :--- |
| **ISOT** | Hybrid Ensemble | **100.0%** | Perfect "Safety Net" against Article Laundering. |
| **TI-CNN** | CNN / BERT | **97.0%** | High performance on structured datasets. |
| **FNC** | CNN (Baseline) | **82.0%** | CNNs outperformed Transformers on noisy data. |
| **LIAR** | Hybrid Ensemble | **70.0%** | Best-in-class performance on short-form claims. |

---

## Core Methodology

### 1. Mathematical Evaluation
We utilized a robust set of metrics to ensure our models were not just guessing based on class imbalance:

* **Accuracy:** $$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$
* **Precision:** $$Precision = \frac{TP}{TP + FP}$$
* **Recall:** $$Recall = \frac{TP}{TP + FN}$$
* **F1-Score:** $$F1\text{-}Score = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$$
* **Matthews Correlation Coefficient (MCC):** $$MCC = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}$$

### 2. Architecture Selection
* **LSTM:** Captured sequential dependencies but struggled with long-range context.
* **CNN:** Highly effective at identifying local, position-invariant textual patterns (97% on TI-CNN).
* **BERT:** Provided deep semantic comprehension through bidirectional transformer encoders.

### 3. The Overfitting Trap
Our research identified that Transformer models peak extremely early (often by Step 500). We implemented **Early Stopping** and **Regularization** to ensure our 100% accuracy on ISOT was generalizable and not mere memorization.

---

## How it Works: The X-Ray Logic

* **Red Highlights:** Indicators of sensationalism or hedging (e.g., *"shocking"*, *"purportedly"*).
* **Green Highlights:** Indicators of factual reporting (e.g., *"confirmed"*, *"official"*).

---

## Technical Stack
* **Language:** Python 3.10
* **Frameworks:** PyTorch 2.0, TensorFlow 2.10, Streamlit
* **Transformers:** Hugging Face (BERT, RoBERTa, T5, PEGASUS)
* **Visualization:** Plotly, Matplotlib
* **Compute:** NVIDIA RTX 4090, Tesla P100

---

## Installation & Usage

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/VeritasLens.git](https://github.com/your-username/VeritasLens.git)
cd VeritasLens
```

### 2. Install Dependencies
```bash
pip install streamlit pandas numpy torch transformers plotly
```

### 3. Run the VeritasLens App
```bash
streamlit run app.py
```

## Acknowledgments
We express our deepest gratitude to Dr. Adarsh Patel for his invaluable mentorship and technical guidance throughout the duration of this research. His insights into deep learning regularization and interpretability were pivotal to the project's success.

## License
MIT License

Copyright (c) 2026 Simran Gupta, Bhuvanyu Geel, Anipra Pandya, Parth Sinha, Aviral Yadav

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
