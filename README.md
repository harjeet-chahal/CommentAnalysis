# 🛡️ CommentAnalysis: AI Content Moderation System
## Demo
YouTube Demo: https://www.youtube.com/watch?v=SrXcQuA-9kw
## **CommentAnalysis** is a production-grade AI system designed to detect and classify toxic content in real-time. It features a **3-Tier Model Architecture** to balance the trade-off between latency and accuracy, and includes a comprehensive interactive dashboard for live scanning, benchmarking, and robustness testing.

---

## 🚀 Key Features

*   **3-Tier Architecture**:
    *   **Tier 1 (Baseline)**: TF-IDF + Logistic Regression (Low Latency, Low Accuracy).
    *   **Tier 2 (Balanced)**: Bi-LSTM with Global Pooling (balanced performance).
    *   **Tier 3 (State-of-the-art)**: Fine-tuned **RoBERTa** Transformer (High Accuracy, Higher Latency).
*   **Interactive Dashboard**: Built with **Streamlit** to visualize model performance and test comments in real-time.
*   **Reliability Lab**: A dedicated environment to test model robustness against "noise" (e.g., typos, adversarial attacks).
*   **Performance Benchmarks**: Interactive visualizations showing the exact trade-offs between model size, inference speed, and safety recall.

---

## 🛠️ Tech Stack

*   **Languages**: Python 3.9+
*   **Deep Learning**: PyTorch, Hugging Face Transformers
*   **Machine Learning**: Scikit-Learn, Pandas, NumPy
*   **Frontend**: Streamlit, Altair
*   **Backend**: FastAPI (for serving the models)
*   **Deployment**: Docker

---

## 📊 Performance & Trade-offs

The system is designed to allow stakeholders to choose the right model for their specific constraints.

| Model | F1 Score | Safety Recall | Latency (ms) | Size (MB) |
| :--- | :---: | :---: | :---: | :---: |
| **TF-IDF (Baseline)** | 0.54 | 83% | **5 ms** | **50 MB** |
| **Bi-LSTM** | 0.53 | 84% | 15 ms | 200 MB |
| **RoBERTa (SOTA)** | **0.47** | **94%** | 80 ms | 500 MB |

> **Insight**: RoBERTa offers the highest safety recall (94%) but comes with a 16x latency cost compared to the baseline. For real-time chat applications, Tier 2 might be preferred; for offline forum moderation, Tier 3 is optimal.

---

## 💻 App Interface

The web application features three main tabs:

### 1. Live Scanner
A real-time interface to type or paste text and get an instant toxicity analysis.
*   **Inputs**: User text.
*   **Outputs**: "Safe" vs "Toxic" banner, specific flags (e.g., `obscene`, `threat`), and a probability distribution chart.

### 2. Model Benchmarks
An interactive Bubble Chart visualization proving the "accuracy vs. latency" and "accuracy vs. model size" trade-offs.

### 3. Reliability Lab
Demonstrates the system's robustness. Users can inject random noise (typos) into text to see how different models react.
*   *RoBERTa acts robustly to noise (maintains prediction).*
*   *TF-IDF often flips prediction with simple typos (e.g., "idiot" -> "idoit").*

---

## ⚙️ Setup & Usage

### 1. Dependencies
```bash
pip install -r requirements.txt
```

### 2. Data Preparation
Ensure the Jigsaw Toxic Comment Classification Challenge data is placed in `data/raw/`:
*   `train.csv`
*   `test.csv`
*   `test_labels.csv`

### 3. Training
Train the specific tier you need:

**Tier 1 (Baseline)**
```bash
python train.py --model_type tier1 --config configs/config.yaml
```

**Tier 2 (Bi-LSTM)**
```bash
python train.py --model_type tier2 --config configs/config.yaml
```

**Tier 3 (RoBERTa)**
```bash
python train.py --model_type tier3 --config configs/model/tier3_roberta.yaml
```

### 4. Running the Dashboard
Start the Streamlit frontend (ensure your Backend API is running if checking live inference):

```bash
streamlit run app/frontend.py
```

---

## 📂 Project Structure

```
├── app/                  # Frontend & Backend Code
│   ├── frontend.py       # Streamlit Dashboard
│   └── main.py           # FastAPI Backend
├── configs/              # YAML Configuration Files
├── data/                 # Raw and Processed Data
├── models/               # Saved Model Checkpoints
├── src/                  # Source Code
│   ├── modeling/         # Model Architectures (RNN, Transformer, etc.)
│   ├── training/         # Training Loops & Loss Functions
│   └── evaluation/       # Metrics & Robustness Tests
├── notebooks/            # Jupyter Notebooks for Analysis
├── train.py              # Master Training Script
└── evaluate.py           # Evaluation Script
```
