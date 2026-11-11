# 🏛️ Consumer Complaint Classification & Automated Routing Pipeline

An end-to-end Deep Learning and Natural Language Processing (NLP) system designed to automatically classify customer complaints into five product categories to streamline department routing, minimize handling costs, and resolve customer issues faster.

---

## 📌 Project Overview & Objectives
- **End-to-End NLP Classification Pipeline:** From raw textual complaint narratives to automated department routing.
- **Architectural Benchmark:** Comparative evaluation of:
  - **SimpleRNN** (Recurrent Neural Network baseline)
  - **LSTM** (Long Short-Term Memory Network)
  - **GRU** (Gated Recurrent Unit)
  - **Fine-Tuned HuggingFace Transformer** (`distilbert-base-uncased`)
- **Evaluation Metrics:** Accuracy, Precision (Weighted), Recall (Weighted), F1-Score (Weighted & Macro), and Confusion Matrices.
- **Production-Ready Gradio Web App:** Interactive user interface for real-time narrative inference, confidence scoring, and automated routing recommendations.

---

## 📂 Project Structure
```
Consumer Complaint Classification/
│
├── Dataset/
│   └── complaints_processed.csv       # Preprocessed dataset (~162,421 records)
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py               # Text cleaning, Tokenizer, Sequence Padding, LabelEncoder
│   ├── models.py                      # SimpleRNN, LSTM, GRU, and DistilBERT model architectures
│   └── evaluate.py                    # Metric calculations, confusion matrices & comparison charts
│
├── artifacts/                         # Generated models, weights, tokenizers & plots
│   ├── rnn_model.pt                   # Trained SimpleRNN weights
│   ├── lstm_model.pt                  # Trained LSTM weights
│   ├── gru_model.pt                   # Trained GRU weights
│   ├── transformer_model/             # Fine-tuned DistilBERT weights & tokenizer
│   ├── tokenizer.pickle               # Fitted TextTokenizer with vocabulary
│   ├── label_encoder.pickle           # Fitted LabelEncoder
│   ├── metadata.pickle                # Pipeline hyperparameter & training metadata
│   ├── metrics_comparison.csv         # Benchmark evaluation comparison table
│   └── plots/
│       ├── original_class_distribution.png
│       ├── metrics_comparison.png
│       └── all_confusion_matrices.png
│
├── app.py                             # Interactive Gradio web application
├── run_pipeline.py                    # Complete end-to-end training and evaluation script
├── build_notebook.py                  # Notebook generator script
├── Consumer_Complaint_Classification.ipynb  # Step-by-step Jupyter Notebook
└── README.md
```

---

## 🔬 Benchmark Results

All four models were trained and evaluated on identical stratified test sets across the 5 target complaint classes:
- `credit_card`
- `credit_reporting`
- `debt_collection`
- `mortgages_and_loans`
- `retail_banking`

| Architecture | Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) | F1-Score (Macro) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **SimpleRNN** | 34.82% | 0.3454 | 0.3482 | 0.3105 | 0.3105 |
| **LSTM** | 61.33% | 0.6095 | 0.6133 | 0.5973 | 0.5973 |
| **Fine-Tuned Transformer (DistilBERT)** | 72.00% | 0.7231 | 0.7200 | 0.7154 | 0.7147 |
| **GRU (Best Model 🏆)** | **80.84%** | **0.8077** | **0.8084** | **0.8076** | **0.8076** |

### Key Findings & Insights:
1. **Vanishing Gradients in SimpleRNN:** SimpleRNN struggled on long sequences ($\approx 120$ tokens) with 34.8% accuracy, illustrating the vanishing gradient limitation of standard recurrent cells.
2. **LSTM Cell State Benefit:** Introducing input, forget, and output gating allowed LSTM to jump by **+26.5% accuracy** to 61.33%.
3. **GRU Superiority:** GRU achieved the highest overall score (**80.84% accuracy, 80.76% F1**). Its streamlined gating (reset and update gates) converged efficiently without overfitting on sequence length 120.
4. **Contextual Transformer Attention:** DistilBERT reached 72.0% accuracy in just 1 fine-tuning epoch, demonstrating strong zero-shot pre-trained representations.

---

## 🚀 How to Run

### 1. Run the End-to-End Pipeline
To retrain models or re-generate all evaluation plots:
```bash
python run_pipeline.py
```

### 2. Launch the Gradio Web Application
Launch the interactive web app locally:
```bash
python app.py
```
Open your browser at `http://127.0.0.1:7860`. You can:
- Type or paste any raw complaint narrative.
- Select between **GRU (Best Model)**, **LSTM**, **SimpleRNN**, or **Transformer**.
- View predicted complaint product category, confidence score %, probability distributions, and automated department routing SLAs.
- Click on pre-filled real-world complaint examples for instant demonstration.

### 3. Open the Jupyter Notebook
Open `Consumer_Complaint_Classification.ipynb` in VS Code or Jupyter Lab to inspect the complete step-by-step workflow with visualizations.

