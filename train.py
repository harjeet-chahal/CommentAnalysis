
import argparse
import pandas as pd
import torch
import torch.optim as optim
import joblib
import os
from transformers import AutoTokenizer, BertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from src.utils import load_config, set_seed
from src.data_loader import get_dataloaders
from src.training.loss import WeightedBCELoss
from src.training.trainer import Trainer

# Models
from src.modeling.baselines import Tier1Baseline
from src.modeling.rnn import BiLSTMClassifier
from src.modeling.transformer import ToxicTransformer
from src.preprocessing import clean_text

def train_tier1(config):
    print("--- Training Tier 1 Baseline (TF-IDF + LR) ---")
    df = pd.read_csv(config['paths']['train'])
    print("Preprocessing...")
    df['comment_text'] = df['comment_text'].apply(clean_text)
    
    X = df['comment_text'].values
    y = df[config['labels']].values
    
    print("Splitting...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = Tier1Baseline()
    print("Fitting model...")
    model.fit(X_train, y_train)
    
    print("Evaluating...")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=config['labels'], zero_division=0))
    
    save_path = os.path.join(os.path.dirname(config['paths']['train']), '../../models/baseline_tfidf_lr.joblib')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(model, save_path)
    print(f"Model saved to {save_path}")

def train_deep_learning_model(config, model_type):
    print(f"--- Training {model_type} ---")
    set_seed(config['training']['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 1. Tokenizer & Model
    if model_type == 'tier2':
        # RNN uses BERT tokenizer but trains own embeddings
        print("Loading BERT Tokenizer for RNN...")
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        print("Initializing BiLSTM...")
        model = BiLSTMClassifier(
            vocab_size=len(tokenizer),
            embedding_dim=300,
            hidden_dim=128,
            num_labels=config['model']['num_labels']
        )
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        checkpoint_name = 'bilstm_best.pt'
        
    elif model_type == 'tier3':
        # Transformer uses its own tokenizer and model
        model_name = config['model']['name'] # e.g. roberta-base
        print(f"Loading {model_name} Tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print(f"Initializing {model_name} Model...")
        model = ToxicTransformer(model_name=model_name, num_labels=config['model']['num_labels'])
        optimizer = optim.AdamW(model.parameters(), lr=float(config['training']['learning_rate']))
        checkpoint_name = 'roberta_best.pt'
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = model.to(device)

    # 2. Data Loaders
    print("Preparing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(config, tokenizer)

    # 3. Loss (Weighted)
    print("Calculating Class Weights...")
    df = pd.read_csv(config['paths']['train'])
    label_cols = config['labels']
    class_counts = [df[col].sum() for col in label_cols]
    total_samples = len(df)
    
    criterion = WeightedBCELoss(class_counts, total_samples, device)

    # 4. Trainer
    save_dir = f"models/{model_type}"
    trainer = Trainer(model, optimizer, criterion, device, checkpoint_dir=save_dir)
    
    # 5. Loop
    num_epochs = config['training']['epochs']
    print(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(1, num_epochs + 1):
        train_loss = trainer.train_one_epoch(train_loader, epoch)
        val_loss = trainer.evaluate(val_loader)
        
        print(f"Epoch {epoch}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        trainer.save_checkpoint(val_loss, model_name=checkpoint_name)
        
    print(f"Training Complete. Best model saved to {save_dir}/{checkpoint_name}")

def main():
    parser = argparse.ArgumentParser(description="CommentAnalysis Training CLI")
    parser.add_argument('--model_type', type=str, required=True, choices=['tier1', 'tier2', 'tier3'], 
                        help="tier1 (Baseline), tier2 (LSTM), tier3 (Transformer)")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help="Path to config file")
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    if args.model_type == 'tier1':
        train_tier1(config)
    else:
        train_deep_learning_model(config, args.model_type)

if __name__ == "__main__":
    main()
