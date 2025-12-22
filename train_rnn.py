
import torch
import pandas as pd
import numpy as np
from transformers import BertTokenizer
import os
import torch.optim as optim

from configs.config import Config
from src.utils import load_config, set_seed
from src.data_loader import get_dataloaders
from src.modeling.rnn import BiLSTMClassifier
from src.training.loss import WeightedBCELoss
from src.training.trainer import Trainer

def main():
    config = load_config("configs/config.yaml")
    set_seed(config['training']['seed'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Prepare Data
    print("Loading data and tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(config['model']['name'])
    train_loader, val_loader, test_loader = get_dataloaders(config, tokenizer)
    
    # 2. Calculate Class Weights for Loss
    # We need to reload the train_df or pass it out to get counts. 
    # For now, let's just load the raw csv again to count quickly.
    print("Calculating class weights...")
    df = pd.read_csv(config['paths']['train'])
    # We used indices in data_loader, but roughly the distribution is the same.
    # To be precise we should use the subset, but for 'pos_weight' global stats are usually fine.
    
    label_cols = config['labels']
    class_counts = [df[col].sum() for col in label_cols]
    total_samples = len(df)
    
    print(f"Class Counts: {dict(zip(label_cols, class_counts))}")
    
    criterion = WeightedBCELoss(class_counts, total_samples, device)
    
    # 3. Model
    print("Initializing BiLSTM...")
    model = BiLSTMClassifier(
        vocab_size=len(tokenizer),
        embedding_dim=300,
        hidden_dim=128,
        num_labels=config['model']['num_labels']
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3) # Standard LR for LSTM
    
    # 4. Trainer
    trainer = Trainer(model, optimizer, criterion, device, checkpoint_dir='models')
    
    num_epochs = config['training']['epochs']
    print(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(1, num_epochs + 1):
        train_loss = trainer.train_one_epoch(train_loader, epoch)
        val_loss = trainer.evaluate(val_loader)
        
        print(f"Epoch {epoch}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        trainer.save_checkpoint(val_loss, model_name='bilstm_best.pt')
        
    print("Training Complete!")

if __name__ == "__main__":
    main()
