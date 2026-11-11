import os
import sys
import time
import pickle
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

# Local modules
from src.preprocessing import clean_text, TextTokenizer, LabelEncoder
from src.models import SimpleRNNClassifier, LSTMClassifier, GRUClassifier, get_transformer_model
from src.evaluate import compute_metrics, plot_confusion_matrix, plot_all_confusion_matrices, plot_metrics_comparison

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[*] Running on device: {device}")

# Directories
DATA_PATH = "Dataset/complaints_processed.csv"
ARTIFACTS_DIR = "artifacts"
PLOTS_DIR = os.path.join(ARTIFACTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


class TextDataset(Dataset):
    """PyTorch Dataset for sequential models (RNN, LSTM, GRU)."""
    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class TransformerDataset(Dataset):
    """PyTorch Dataset for HuggingFace Transformers."""
    def __init__(self, texts: list, labels: np.ndarray, tokenizer, max_length: int = 128):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt"
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item


def train_sequential_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                           epochs: int = 4, lr: float = 0.002, class_weights: torch.Tensor = None) -> nn.Module:
    """Trains a sequential PyTorch model (RNN / LSTM / GRU) with early stopping checkpointing."""
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    model.to(device)

    best_val_loss = float('inf')
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        start_t = time.time()

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

        train_loss = total_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                v_loss += loss.item() * batch_x.size(0)
                preds = outputs.argmax(dim=1)
                v_correct += (preds == batch_y).sum().item()
                v_total += batch_y.size(0)

        val_loss = v_loss / v_total
        val_acc = v_correct / v_total
        elapsed = time.time() - start_t

        print(f"   Epoch {epoch}/{epochs} ({elapsed:.1f}s) | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_weights is not None:
        model.load_state_dict(best_weights)
    return model


def evaluate_sequential_model(model: nn.Module, test_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generates predictions and probabilities on test set."""
    model.eval()
    model.to(device)
    y_true, y_pred, y_probs = [], [], []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            y_true.extend(batch_y.numpy())
            y_pred.extend(preds)
            y_probs.extend(probs)

    return np.array(y_true), np.array(y_pred), np.array(y_probs)


def train_transformer_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                            epochs: int = 1, lr: float = 3e-5) -> nn.Module:
    """Fine-tunes a HuggingFace Transformer model for text classification."""
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

    best_val_loss = float('inf')
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        start_t = time.time()

        for step, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item() * input_ids.size(0)
            preds = outputs.logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if (step + 1) % 50 == 0 or (step + 1) == len(train_loader):
                print(f"      Step {step + 1}/{len(train_loader)} | Batch Loss: {loss.item():.4f}")

        train_loss = total_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                v_loss += outputs.loss.item() * input_ids.size(0)
                preds = outputs.logits.argmax(dim=1)
                v_correct += (preds == labels).sum().item()
                v_total += labels.size(0)

        val_loss = v_loss / v_total
        val_acc = v_correct / v_total
        elapsed = time.time() - start_t

        print(f"   Epoch {epoch}/{epochs} ({elapsed:.1f}s) | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_weights is not None:
        model.load_state_dict(best_weights)
    return model


def evaluate_transformer_model(model: nn.Module, test_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generates predictions and probabilities on test set for Transformer."""
    model.eval()
    model.to(device)
    y_true, y_pred, y_probs = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            y_true.extend(labels.numpy())
            y_pred.extend(preds)
            y_probs.extend(probs)

    return np.array(y_true), np.array(y_pred), np.array(y_probs)


def main():
    print("=" * 70)
    print("🚀 CONSUMER COMPLAINT CLASSIFICATION - END-TO-END PIPELINE")
    print("=" * 70)

    # 1. Load Data
    print("\n[1/6] Loading and exploring dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset Shape: {df.shape}")
    print(f"Initial Null Counts:\n{df.isnull().sum()}")

    # Drop null narratives
    df = df.dropna(subset=['narrative']).copy()
    print(f"Shape after removing nulls: {df.shape}")

    # Plot initial class distribution
    class_counts = df['product'].value_counts()
    print("\nClass distribution:")
    for cls_name, cnt in class_counts.items():
        print(f"  - {cls_name}: {cnt:,} ({cnt/len(df)*100:.2f}%)")

    plt.figure(figsize=(10, 5))
    sns.barplot(x=class_counts.values, y=class_counts.index, palette='crest')
    plt.title("Original Dataset: Class Distribution", fontsize=14, fontweight='bold')
    plt.xlabel("Number of Complaints", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "original_class_distribution.png"), dpi=300)
    plt.close()

    # 2. Balanced Stratified Subsetting for High-Efficiency Benchmarking
    # Total ~17,500 samples (3,500 per class) to ensure balanced representation
    print("\n[2/6] Balancing class distribution & Preprocessing text...")
    sample_per_class = 3500
    subsets = []
    for p in df['product'].unique():
        sub = df[df['product'] == p]
        subsets.append(sub.sample(n=min(len(sub), sample_per_class), random_state=42))
    df_balanced = pd.concat(subsets, ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)

    print(f"Balanced Dataset Shape: {df_balanced.shape}")
    print("Balanced class counts:\n", df_balanced['product'].value_counts())

    # Preprocessing: clean_text (lowercasing, punctuation/number removal, stopwords, lemmatization)
    print("Applying text preprocessing (lowercasing, cleaning, stopwords, lemmatization)...")
    df_balanced['cleaned_narrative'] = df_balanced['narrative'].apply(clean_text)

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(df_balanced['product'].values)
    label_encoder.save(os.path.join(ARTIFACTS_DIR, "label_encoder.pickle"))
    print(f"Classes ({len(label_encoder.classes_)}): {label_encoder.classes_}")

    # Compute class weights
    class_weights_np = compute_class_weight('balanced', classes=np.unique(y_encoded), y=y_encoded)
    class_weights_tensor = torch.tensor(class_weights_np, dtype=torch.float32)

    # Split dataset: Train (70%), Val (15%), Test (15%)
    X_train_text, X_temp_text, y_train, y_temp = train_test_split(
        df_balanced['cleaned_narrative'].values, y_encoded, test_size=0.30, random_state=42, stratify=y_encoded
    )
    X_val_text, X_test_text, y_val, y_test = train_test_split(
        X_temp_text, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    print(f"Train set: {len(X_train_text)} | Val set: {len(X_val_text)} | Test set: {len(X_test_text)}")

    # 3. Tokenizer & Padding for Sequential Models
    print("\n[3/6] Tokenizing sequences and building vocabulary...")
    MAX_WORDS = 15000
    MAX_LEN = 120
    tokenizer = TextTokenizer(max_words=MAX_WORDS, maxlen=MAX_LEN)
    tokenizer.fit_on_texts(X_train_text)
    tokenizer.save(os.path.join(ARTIFACTS_DIR, "tokenizer.pickle"))
    print(f"Vocabulary Size: {tokenizer.vocab_size} | Padded Sequence Length: {MAX_LEN}")

    X_train_seq = tokenizer.transform(X_train_text)
    X_val_seq = tokenizer.transform(X_val_text)
    X_test_seq = tokenizer.transform(X_test_text)

    # PyTorch DataLoaders for Sequential Models
    BATCH_SIZE = 64
    train_dataset = TextDataset(X_train_seq, y_train)
    val_dataset = TextDataset(X_val_seq, y_val)
    test_dataset = TextDataset(X_test_seq, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    num_classes = len(label_encoder.classes_)
    vocab_size = tokenizer.vocab_size
    EMBED_DIM = 128
    HIDDEN_DIM = 128

    results = {}

    # 4. Train Model 1: SimpleRNN
    print("\n" + "-" * 50)
    print("🧠 MODEL 1: SimpleRNN (Trained from scratch)")
    print("-" * 50)
    rnn_model = SimpleRNNClassifier(vocab_size=vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, num_classes=num_classes)
    rnn_model = train_sequential_model(rnn_model, train_loader, val_loader, epochs=4, lr=0.002, class_weights=class_weights_tensor)
    y_true_rnn, y_pred_rnn, probs_rnn = evaluate_sequential_model(rnn_model, test_loader)
    rnn_metrics = compute_metrics(y_true_rnn, y_pred_rnn)
    results["SimpleRNN"] = {"metrics": rnn_metrics, "y_true": y_true_rnn, "y_pred": y_pred_rnn, "model": rnn_model}
    torch.save(rnn_model.state_dict(), os.path.join(ARTIFACTS_DIR, "rnn_model.pt"))
    print(f"[*] SimpleRNN Test Results -> Accuracy: {rnn_metrics['Accuracy']:.4f}, F1 (Weighted): {rnn_metrics['F1-Score (Weighted)']:.4f}")

    # Train Model 2: LSTM
    print("\n" + "-" * 50)
    print("🧠 MODEL 2: LSTM (Trained from scratch)")
    print("-" * 50)
    lstm_model = LSTMClassifier(vocab_size=vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, num_classes=num_classes)
    lstm_model = train_sequential_model(lstm_model, train_loader, val_loader, epochs=4, lr=0.002, class_weights=class_weights_tensor)
    y_true_lstm, y_pred_lstm, probs_lstm = evaluate_sequential_model(lstm_model, test_loader)
    lstm_metrics = compute_metrics(y_true_lstm, y_pred_lstm)
    results["LSTM"] = {"metrics": lstm_metrics, "y_true": y_true_lstm, "y_pred": y_pred_lstm, "model": lstm_model}
    torch.save(lstm_model.state_dict(), os.path.join(ARTIFACTS_DIR, "lstm_model.pt"))
    print(f"[*] LSTM Test Results -> Accuracy: {lstm_metrics['Accuracy']:.4f}, F1 (Weighted): {lstm_metrics['F1-Score (Weighted)']:.4f}")

    # Train Model 3: GRU
    print("\n" + "-" * 50)
    print("🧠 MODEL 3: GRU (Trained from scratch)")
    print("-" * 50)
    gru_model = GRUClassifier(vocab_size=vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, num_classes=num_classes)
    gru_model = train_sequential_model(gru_model, train_loader, val_loader, epochs=4, lr=0.002, class_weights=class_weights_tensor)
    y_true_gru, y_pred_gru, probs_gru = evaluate_sequential_model(gru_model, test_loader)
    gru_metrics = compute_metrics(y_true_gru, y_pred_gru)
    results["GRU"] = {"metrics": gru_metrics, "y_true": y_true_gru, "y_pred": y_pred_gru, "model": gru_model}
    torch.save(gru_model.state_dict(), os.path.join(ARTIFACTS_DIR, "gru_model.pt"))
    print(f"[*] GRU Test Results -> Accuracy: {gru_metrics['Accuracy']:.4f}, F1 (Weighted): {gru_metrics['F1-Score (Weighted)']:.4f}")

    # Train Model 4: HuggingFace Transformer (DistilBERT)
    print("\n" + "-" * 50)
    print("🤗 MODEL 4: Fine-Tuned Transformer (DistilBERT)")
    print("-" * 50)
    trans_model, trans_tok = get_transformer_model("distilbert-base-uncased", num_classes=num_classes)

    # For fast and effective fine-tuning on CPU, freeze base distilbert layers and unfreeze layer 5 and classifier
    for param in trans_model.distilbert.parameters():
        param.requires_grad = False
    for param in trans_model.distilbert.transformer.layer[-2:].parameters():
        param.requires_grad = True
    for param in trans_model.pre_classifier.parameters():
        param.requires_grad = True
    for param in trans_model.classifier.parameters():
        param.requires_grad = True

    # Subset slightly for CPU transformer fine-tuning
    hf_train_ds = TransformerDataset(list(X_train_text[:3500]), y_train[:3500], trans_tok, max_length=96)
    hf_val_ds = TransformerDataset(list(X_val_text[:800]), y_val[:800], trans_tok, max_length=96)
    hf_test_ds = TransformerDataset(list(X_test_text[:800]), y_test[:800], trans_tok, max_length=96)

    hf_train_loader = DataLoader(hf_train_ds, batch_size=32, shuffle=True)
    hf_val_loader = DataLoader(hf_val_ds, batch_size=32, shuffle=False)
    hf_test_loader = DataLoader(hf_test_ds, batch_size=32, shuffle=False)

    trans_model = train_transformer_model(trans_model, hf_train_loader, hf_val_loader, epochs=1, lr=4e-5)
    y_true_hf, y_pred_hf, probs_hf = evaluate_transformer_model(trans_model, hf_test_loader)
    hf_metrics = compute_metrics(y_true_hf, y_pred_hf)
    results["Transformer (DistilBERT)"] = {"metrics": hf_metrics, "y_true": y_true_hf, "y_pred": y_pred_hf, "model": trans_model}

    # Save transformer weights and tokenizer
    trans_save_path = os.path.join(ARTIFACTS_DIR, "transformer_model")
    trans_model.save_pretrained(trans_save_path)
    trans_tok.save_pretrained(trans_save_path)
    print(f"[*] Transformer Test Results -> Accuracy: {hf_metrics['Accuracy']:.4f}, F1 (Weighted): {hf_metrics['F1-Score (Weighted)']:.4f}")

    # 5. Model Comparison & Visualization
    print("\n" + "=" * 70)
    print("📊 MODEL BENCHMARKING & COMPARISON")
    print("=" * 70)
    metrics_summary = {}
    for m_name, res in results.items():
        metrics_summary[m_name] = res['metrics']

    comparison_df = pd.DataFrame(metrics_summary).T
    print("\nComparison Table:")
    print(comparison_df.round(4).to_string())

    comparison_df.to_csv(os.path.join(ARTIFACTS_DIR, "metrics_comparison.csv"))

    # Plot metrics comparison bar chart
    plot_metrics_comparison(comparison_df, save_path=os.path.join(PLOTS_DIR, "metrics_comparison.png"))

    # Plot all confusion matrices
    plot_all_confusion_matrices(results, class_names=label_encoder.classes_, save_path=os.path.join(PLOTS_DIR, "all_confusion_matrices.png"))

    # 6. Identify Best Model & Save
    best_model_name = comparison_df['F1-Score (Weighted)'].idxmax()
    print(f"\n🏆 Best Performing Model: {best_model_name} (F1: {comparison_df.loc[best_model_name, 'F1-Score (Weighted)']:.4f})")

    # Save meta information about best model
    meta = {
        "best_model_name": best_model_name,
        "classes": label_encoder.classes_,
        "metrics": comparison_df.to_dict(),
        "vocab_size": vocab_size,
        "embed_dim": EMBED_DIM,
        "hidden_dim": HIDDEN_DIM,
        "max_len": MAX_LEN
    }
    with open(os.path.join(ARTIFACTS_DIR, "metadata.pickle"), 'wb') as f:
        pickle.dump(meta, f)

    # 7. Test sample complaints with best model
    print("\n[6/6] Testing Best Model with Real-World Customer Complaints...")
    sample_complaints = [
        "I was charged an unauthorized $500 fee on my credit card that I never made. The bank refuses to refund it.",
        "A debt collection agency keeps calling my home 10 times a day for a debt that is not even mine.",
        "My credit report contains incorrect late payment marks for an account that was paid in full years ago.",
        "The mortgage company increased my monthly escrow payment without any prior written notification.",
        "My checking account was frozen and I cannot withdraw my payroll deposit from the local branch."
    ]

    for comp in sample_complaints:
        cleaned = clean_text(comp)
        if "Transformer" in best_model_name:
            inputs = trans_tok(cleaned, return_tensors="pt", truncation=True, padding=True, max_length=96).to(device)
            with torch.no_grad():
                logits = trans_model(**inputs).logits
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        else:
            best_mod = results[best_model_name]['model']
            seq = tokenizer.transform([cleaned])
            with torch.no_grad():
                logits = best_mod(torch.tensor(seq, dtype=torch.long).to(device))
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        pred_idx = np.argmax(probs)
        pred_label = label_encoder.classes_[pred_idx]
        conf = probs[pred_idx] * 100
        print(f"\nComplaint: \"{comp[:75]}...\"")
        print(f"--> Predicted: [{pred_label.upper()}] with {conf:.1f}% confidence")

    print("\n✅ Pipeline completed successfully! Artifacts and plots saved in 'artifacts/'.")


if __name__ == "__main__":
    main()
