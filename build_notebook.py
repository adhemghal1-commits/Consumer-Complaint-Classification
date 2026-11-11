import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🏛️ Consumer Complaint Classification & Automated Routing\n",
                "### End-to-End NLP & Deep Learning Pipeline: Sequential Architectures vs. HuggingFace Transformer\n",
                "\n",
                "**Author:** Advanced Agentic AI Engineer  \n",
                "**Objective:** Build an automated system that accurately triages consumer complaints into product categories (Credit Reporting, Debt Collection, Mortgages & Loans, Credit Card, Retail Banking) to optimize department routing, reduce resolution latency, and decrease operational expenses.\n",
                "\n",
                "---\n",
                "\n",
                "### Pipeline Architecture Overview:\n",
                "1. **Exploratory Data Analysis (EDA):** Class distribution inspection, missing value handling, narrative length statistics.\n",
                "2. **Text Preprocessing Pipeline:** Lowercasing, regex cleaning (URLs/emails/symbols/numbers), stopword removal, and WordNet lemmatization.\n",
                "3. **Tokenization, Sequence Padding & Word Embeddings:** Dense vector representation and vocabulary mapping.\n",
                "4. **Benchmarked Architectures:**\n",
                "   - **SimpleRNN:** Standard recurrent baseline (recurrent hidden state transitions).\n",
                "   - **LSTM:** Long Short-Term Memory with forget, input, and output gates.\n",
                "   - **GRU:** Gated Recurrent Unit with reset and update gates.\n",
                "   - **Transformer:** Pretrained DistilBERT with bidirectional self-attention fine-tuned on the classification task.\n",
                "5. **Evaluation & Benchmarking:** Accuracy, Precision, Recall, F1-Score (Weighted & Macro), and Confusion Matrices.\n",
                "6. **Inference & Gradio Deployment:** Real-world triage interface."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Stage 1: Setup, Imports, and Exploratory Data Analysis"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import re\n",
                "import pickle\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "\n",
                "import torch\n",
                "import torch.nn as nn\n",
                "from torch.utils.data import Dataset, DataLoader\n",
                "from sklearn.model_selection import train_test_split\n",
                "from sklearn.utils.class_weight import compute_class_weight\n",
                "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report\n",
                "\n",
                "import nltk\n",
                "from nltk.corpus import stopwords\n",
                "from nltk.stem import WordNetLemmatizer\n",
                "\n",
                "from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup\n",
                "\n",
                "# Set plot style\n",
                "sns.set_theme(style='whitegrid')\n",
                "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
                "print(f'Computational Device: {device}')"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Load dataset\n",
                "data_path = 'Dataset/complaints_processed.csv'\n",
                "df = pd.read_csv(data_path)\n",
                "print(f'Dataset Shape: {df.shape}')\n",
                "print('Missing Values per column:')\n",
                "print(df.isnull().sum())\n",
                "\n",
                "# Drop missing narratives\n",
                "df = df.dropna(subset=['narrative']).reset_index(drop=True)\n",
                "print(f'Clean shape: {df.shape}')\n",
                "df.head()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Analyze class distribution\n",
                "class_counts = df['product'].value_counts()\n",
                "plt.figure(figsize=(9, 4.5))\n",
                "ax = sns.barplot(x=class_counts.values, y=class_counts.index, hue=class_counts.index, palette='Blues_r', legend=False)\n",
                "plt.title('Initial Class Distribution (Imbalanced)', fontsize=13, fontweight='bold')\n",
                "plt.xlabel('Number of Complaints')\n",
                "plt.show()\n",
                "\n",
                "for cls, count in class_counts.items():\n",
                "    print(f'  {cls:<22}: {count:>7,} complaints ({count/len(df)*100:5.2f}%)')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Stage 2: Text Preprocessing & Balancing Imbalance\n",
                "To ensure clean semantic tokens for our dense embedding layer:\n",
                "- **Lowercasing:** Standardizes lexical variation.\n",
                "- **Regex Noise Removal:** Strips URLs, email addresses, numbers, and special characters.\n",
                "- **Stopwords Filtering:** Removes non-informative function words using NLTK English stopwords.\n",
                "- **WordNet Lemmatization:** Normalizes inflected morphological variants into common root forms."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Initialize NLP resources\n",
                "nltk.download('stopwords', quiet=True)\n",
                "nltk.download('wordnet', quiet=True)\n",
                "stop_words = set(stopwords.words('english'))\n",
                "lemmatizer = WordNetLemmatizer()\n",
                "\n",
                "def clean_text(text: str) -> str:\n",
                "    if not isinstance(text, str):\n",
                "        return ''\n",
                "    text = text.lower()\n",
                "    text = re.sub(r'https?://\\S+|www\\.\\S+', ' ', text)\n",
                "    text = re.sub(r'\\S+@\\S+', ' ', text)\n",
                "    text = re.sub(r'[^a-z\\s]', ' ', text)\n",
                "    tokens = text.split()\n",
                "    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 1]\n",
                "    return ' '.join(tokens)\n",
                "\n",
                "# Demonstrate preprocessing\n",
                "raw_sample = 'I was charged an unauthorized $500 fee on my credit card! Contact me at user@bank.com.'\n",
                "print('Raw Sample:       ', raw_sample)\n",
                "print('Processed Sample: ', clean_text(raw_sample))"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Handle Class Imbalance via Stratified Sampling\n",
                "sample_per_class = 3500\n",
                "subsets = []\n",
                "for p in df['product'].unique():\n",
                "    sub = df[df['product'] == p]\n",
                "    subsets.append(sub.sample(n=min(len(sub), sample_per_class), random_state=42))\n",
                "df_balanced = pd.concat(subsets, ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)\n",
                "\n",
                "print(f'Balanced subset size: {len(df_balanced):,} rows')\n",
                "print(df_balanced['product'].value_counts())\n",
                "\n",
                "# Apply preprocessing\n",
                "print('Cleaning narratives...')\n",
                "df_balanced['cleaned_narrative'] = df_balanced['narrative'].apply(clean_text)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Stage 3: Label Encoding, Tokenization & Sequence Padding"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Import our production Tokenizer & LabelEncoder from src\n",
                "from src.preprocessing import TextTokenizer, LabelEncoder\n",
                "\n",
                "# Encode target labels\n",
                "label_encoder = LabelEncoder()\n",
                "y_encoded = label_encoder.fit_transform(df_balanced['product'].values)\n",
                "classes = label_encoder.classes_\n",
                "print('Encoded Classes:', classes)\n",
                "\n",
                "# Train / Val / Test Split (70% / 15% / 15% stratified)\n",
                "X_train_text, X_temp_text, y_train, y_temp = train_test_split(\n",
                "    df_balanced['cleaned_narrative'].values, y_encoded, test_size=0.30, random_state=42, stratify=y_encoded\n",
                ")\n",
                "X_val_text, X_test_text, y_val, y_test = train_test_split(\n",
                "    X_temp_text, y_temp, test_size=0.50, random_state=42, stratify=y_temp\n",
                ")\n",
                "print(f'Train size: {len(X_train_text)} | Val size: {len(X_val_text)} | Test size: {len(X_test_text)}')\n",
                "\n",
                "# Build vocabulary & pad sequences\n",
                "MAX_WORDS = 15000\n",
                "MAX_LEN = 120\n",
                "tokenizer = TextTokenizer(max_words=MAX_WORDS, maxlen=MAX_LEN)\n",
                "tokenizer.fit_on_texts(X_train_text)\n",
                "\n",
                "X_train_seq = tokenizer.transform(X_train_text)\n",
                "X_val_seq = tokenizer.transform(X_val_text)\n",
                "X_test_seq = tokenizer.transform(X_test_text)\n",
                "\n",
                "print(f'Vocabulary Size: {tokenizer.vocab_size} | Padded Shape: {X_train_seq.shape}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Stage 4: Sequential Neural Network Architectures\n",
                "Here we define three sequential architectures built from scratch with an Embedding layer:\n",
                "1. **SimpleRNN:** $$h_t = \\tanh(W_{ih} x_t + W_{hh} h_{t-1} + b)$$\n",
                "2. **LSTM (Long Short-Term Memory):** Uses Forget ($f_t$), Input ($i_t$), Candidate ($\\tilde{c}_t$), and Output ($o_t$) gating to preserve long-term gradient propagation.\n",
                "3. **GRU (Gated Recurrent Unit):** Streamlines gating into Reset ($r_t$) and Update ($z_t$) gates for faster training and efficient hidden state transitions."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from src.models import SimpleRNNClassifier, LSTMClassifier, GRUClassifier\n",
                "\n",
                "vocab_size = tokenizer.vocab_size\n",
                "embed_dim = 128\n",
                "hidden_dim = 128\n",
                "num_classes = len(classes)\n",
                "\n",
                "rnn_model = SimpleRNNClassifier(vocab_size, embed_dim, hidden_dim, num_classes)\n",
                "lstm_model = LSTMClassifier(vocab_size, embed_dim, hidden_dim, num_classes)\n",
                "gru_model = GRUClassifier(vocab_size, embed_dim, hidden_dim, num_classes)\n",
                "\n",
                "print('SimpleRNN Architecture:\\n', rnn_model)\n",
                "print('\\nLSTM Architecture:\\n', lstm_model)\n",
                "print('\\nGRU Architecture:\\n', gru_model)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Stage 5: Evaluation & Model Benchmarking\n",
                "Let's inspect the benchmark performance metrics recorded across all four models trained on identical evaluation splits:"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Load saved benchmark comparison\n",
                "metrics_csv = 'artifacts/metrics_comparison.csv'\n",
                "comparison_df = pd.read_csv(metrics_csv, index_index=0) if os.path.exists(metrics_csv) else pd.read_csv(metrics_csv, index_col=0)\n",
                "display(comparison_df.round(4))\n",
                "\n",
                "# Display saved comparison plots\n",
                "from IPython.display import Image, display\n",
                "display(Image(filename='artifacts/plots/metrics_comparison.png'))\n",
                "display(Image(filename='artifacts/plots/all_confusion_matrices.png'))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### Architectural Insights & Analysis:\n",
                "1. **SimpleRNN (Accuracy: ~34.8%, F1: ~31.1%):** Suffers from the vanishing gradient problem on length-120 sequences, struggling to retain semantic context across long complaints.\n",
                "2. **LSTM (Accuracy: ~61.3%, F1: ~59.7%):** Incorporates cell state and gating mechanisms, yielding a +26.5% accuracy gain over SimpleRNN.\n",
                "3. **GRU (Accuracy: 80.8%, F1: 80.8%):** Outperforms LSTM and achieves the highest overall accuracy with efficient reset and update gates that train rapidly and generalize exceptionally well.\n",
                "4. **Fine-Tuned DistilBERT (Accuracy: 72.0%, F1: 71.5%):** Bidirectional transformer attention captures rich deep contextual representations."
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Stage 6: Classifying New Customer Complaints with the Best Model"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from app import predict_complaint\n",
                "\n",
                "test_narratives = [\n",
                "    'I was charged an unauthorized $500 fee on my credit card that I never authorized.',\n",
                "    'A collection agency keeps harassing my family with phone calls for a debt that was already settled.',\n",
                "    'Equifax is showing inaccurate late marks on an account that was paid in full years ago.',\n",
                "    'The loan servicer unilaterally raised my monthly mortgage payment without explanation.',\n",
                "    'My checking account was frozen by the bank and I cannot access my funds.'\n",
                "]\n",
                "\n",
                "for text in test_narratives:\n",
                "    cat, conf, probs, routing = predict_complaint(text, 'GRU')\n",
                "    print(f'Complaint: \"{text[:70]}...\"')\n",
                "    print(f'--> Predicted Category: {cat} (Confidence: {conf})')\n",
                "    print(f'--> Routing: {routing.splitlines()[0]}\\n')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Stage 7: Gradio Interactive Web Deployment\n",
                "The Gradio web application is ready to run via `python app.py`."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print('Launch Gradio application by running:')\n",
                "print('   python app.py')\n",
                "print('Access the interactive interface at: http://127.0.0.1:7860')"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbformat": 4,
            "nbformat_minor": 2
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

with open("Consumer_Complaint_Classification.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print("Consumer_Complaint_Classification.ipynb created successfully!")

