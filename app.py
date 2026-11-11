import os
import pickle
import numpy as np
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Local imports
from src.preprocessing import clean_text, TextTokenizer, LabelEncoder
from src.models import SimpleRNNClassifier, LSTMClassifier, GRUClassifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ARTIFACTS_DIR = "artifacts"

# Load metadata
meta_path = os.path.join(ARTIFACTS_DIR, "metadata.pickle")
tok_path = os.path.join(ARTIFACTS_DIR, "tokenizer.pickle")
le_path = os.path.join(ARTIFACTS_DIR, "label_encoder.pickle")

meta = None
tokenizer = None
label_encoder = None
models_dict = {}

if os.path.exists(meta_path) and os.path.exists(tok_path) and os.path.exists(le_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    tokenizer = TextTokenizer.load(tok_path)
    label_encoder = LabelEncoder.load(le_path)

    classes = meta["classes"]
    vocab_size = meta["vocab_size"]
    embed_dim = meta["embed_dim"]
    hidden_dim = meta["hidden_dim"]
    num_classes = len(classes)

    # Load SimpleRNN
    rnn_pt = os.path.join(ARTIFACTS_DIR, "rnn_model.pt")
    if os.path.exists(rnn_pt):
        rnn = SimpleRNNClassifier(vocab_size, embed_dim, hidden_dim, num_classes)
        rnn.load_state_dict(torch.load(rnn_pt, map_location=device, weights_only=True))
        rnn.to(device).eval()
        models_dict["SimpleRNN"] = rnn

    # Load LSTM
    lstm_pt = os.path.join(ARTIFACTS_DIR, "lstm_model.pt")
    if os.path.exists(lstm_pt):
        lstm = LSTMClassifier(vocab_size, embed_dim, hidden_dim, num_classes)
        lstm.load_state_dict(torch.load(lstm_pt, map_location=device, weights_only=True))
        lstm.to(device).eval()
        models_dict["LSTM"] = lstm

    # Load GRU
    gru_pt = os.path.join(ARTIFACTS_DIR, "gru_model.pt")
    if os.path.exists(gru_pt):
        gru = GRUClassifier(vocab_size, embed_dim, hidden_dim, num_classes)
        gru.load_state_dict(torch.load(gru_pt, map_location=device, weights_only=True))
        gru.to(device).eval()
        models_dict["GRU"] = gru

    # Load Transformer
    hf_path = os.path.join(ARTIFACTS_DIR, "transformer_model")
    if os.path.exists(hf_path):
        try:
            hf_mod = AutoModelForSequenceClassification.from_pretrained(hf_path)
            hf_tok = AutoTokenizer.from_pretrained(hf_path)
            hf_mod.to(device).eval()
            models_dict["Transformer (DistilBERT)"] = (hf_mod, hf_tok)
        except Exception as e:
            print(f"Notice: Transformer model loading error: {e}")

else:
    classes = ['credit_card', 'credit_reporting', 'debt_collection', 'mortgages_and_loans', 'retail_banking']


CATEGORY_METADATA = {
    'credit_card': {
        'display': 'Credit Card Services',
        'dept': 'Cardholder Services & Fraud Prevention',
        'sla': 'Within 24 Hours (High Priority)'
    },
    'debt_collection': {
        'display': 'Debt Collection',
        'dept': 'Collections Compliance & Legal Operations',
        'sla': 'Within 48 Hours'
    },
    'credit_reporting': {
        'display': 'Credit Reporting',
        'dept': 'Credit Bureau Disputes & Compliance',
        'sla': 'Within 3 business days'
    },
    'mortgages_and_loans': {
        'display': 'Mortgages & Loans',
        'dept': 'Loan Servicing & Escrow Management',
        'sla': 'Within 48 Hours'
    },
    'retail_banking': {
        'display': 'Retail Banking',
        'dept': 'Branch Operations & Account Resolution',
        'sla': 'Within 24 Hours'
    }
}


def predict_complaint(narrative: str, model_choice: str):
    """Predicts complaint category and returns scores and routing advice."""
    if not narrative or not narrative.strip():
        return (
            "⚠️ Please enter a complaint narrative.",
            "0.0%",
            {},
            "Enter narrative text above to receive automated department routing."
        )

    cleaned = clean_text(narrative)
    if not cleaned:
        cleaned = narrative.lower()

    if not models_dict:
        return (
            "Model artifacts not found. Please run 'python run_pipeline.py' first.",
            "N/A",
            {},
            "Artifacts not yet generated."
        )

    # Select model
    if model_choice not in models_dict:
        model_choice = list(models_dict.keys())[0]

    if model_choice == "Transformer (DistilBERT)":
        hf_mod, hf_tok = models_dict[model_choice]
        inputs = hf_tok(cleaned, return_tensors="pt", truncation=True, padding=True, max_length=96).to(device)
        with torch.no_grad():
            outputs = hf_mod(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    else:
        model = models_dict[model_choice]
        seq = tokenizer.transform([cleaned])
        with torch.no_grad():
            tensor_seq = torch.tensor(seq, dtype=torch.long).to(device)
            logits = model(tensor_seq)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    pred_class = classes[pred_idx]
    confidence = float(probs[pred_idx])

    # Class probabilities dict for Gradio Label component
    prob_dict = {classes[i]: float(probs[i]) for i in range(len(classes))}

    meta_info = CATEGORY_METADATA.get(pred_class, {
        'display': pred_class.replace('_', ' ').title(),
        'dept': 'General Customer Support',
        'sla': 'Standard Response'
    })

    summary_text = f"**{meta_info['display']}**"
    conf_text = f"{confidence * 100:.2f}%"
    routing_info = (
        f"**Target Department:** {meta_info['dept']}\n\n"
        f"**Target SLA Resolution:** {meta_info['sla']}\n\n"
        f"**Active Classifier:** {model_choice}\n\n"
        f"**Cleaned Tokens Sample:** `{cleaned[:80]}...`"
    )

    return summary_text, conf_text, prob_dict, routing_info


# Gradio Interface
available_models = list(models_dict.keys()) if models_dict else ["Default"]
best_name = meta.get("best_model_name", available_models[0]) if meta else available_models[0]
if best_name not in available_models and available_models:
    best_name = available_models[0]

sample_examples = [
    ["I noticed an unauthorized charge of $840 on my credit card statement from an overseas merchant. I called immediately to block the card and dispute the fraudulent charge, but the bank has yet to credit my account.", best_name],
    ["A debt collector named Portfolio Recovery has been harassing my elderly parents with daily phone calls demanding payment for an ancient medical bill that was settled 6 years ago.", best_name],
    ["Equifax and Experian are reporting a delinquent mortgage account that does not belong to me. I filed three separate dispute letters with documentation, but the false derogatory mark remains on my credit file.", best_name],
    ["My mortgage servicer improperly increased my monthly payment by $350 claiming an escrow shortage, despite my escrow account maintaining a positive surplus balance all year.", best_name],
    ["I deposited a certified cashier check into my checking account at the local branch, but the bank put an indefinite hold on my funds and locked my debit card, leaving me unable to pay my rent.", best_name]
]

custom_css = """
.gradio-container { max-width: 950px !important; margin: auto !important; }
.hero-box { background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); color: white; padding: 25px; border-radius: 12px; margin-bottom: 20px; }
.hero-box h1 { color: #ffffff !important; margin-bottom: 8px; }
.hero-box p { color: #e0e6ed !important; font-size: 15px; margin: 0; }
"""

with gr.Blocks(title="Consumer Complaint Automated Classifier") as demo:
    gr.HTML("""
    <div class="hero-box">
        <h1>🏛️ Consumer Complaint Classification & Routing AI</h1>
        <p>End-to-End Deep Learning & Transformer NLP Pipeline for Automated Customer Complaint Triage.</p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=3):
            complaint_input = gr.Textbox(
                lines=5,
                placeholder="Paste or type customer complaint narrative here...",
                label="Customer Complaint Narrative",
                info="Raw text will be preprocessed through our NLP pipeline (cleaning, tokenization, lemmatization)."
            )

            model_dropdown = gr.Dropdown(
                choices=available_models,
                value=best_name,
                label="Classification Model",
                info="Compare predictions across SimpleRNN, LSTM, GRU, and Fine-Tuned Transformer."
            )

            submit_btn = gr.Button("🚀 Classify Complaint", variant="primary", size="lg")

        with gr.Column(scale=2):
            pred_category_output = gr.Markdown(label="Predicted Complaint Category", value="*Awaiting submission...*")
            conf_output = gr.Textbox(label="Model Confidence Score", interactive=False)
            label_output = gr.Label(num_top_classes=5, label="Category Probability Distribution")
            routing_output = gr.Markdown(label="Automated Routing Recommendations", value="*Routing details will appear here.*")

    submit_btn.click(
        fn=predict_complaint,
        inputs=[complaint_input, model_dropdown],
        outputs=[pred_category_output, conf_output, label_output, routing_output]
    )

    gr.Examples(
        examples=sample_examples,
        inputs=[complaint_input, model_dropdown],
        outputs=[pred_category_output, conf_output, label_output, routing_output],
        fn=predict_complaint,
        cache_examples=False,
        label="📋 Try Sample Real-World Complaints"
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, css=custom_css)
