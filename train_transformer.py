
import torch
import pandas as pd
import torch.optim as optim
from transformers import AutoTokenizer

from src.utils import load_config, set_seed
from src.data_loader import get_dataloaders
from src.modeling.transformer import ToxicTransformer
from src.training.loss import WeightedBCELoss
from src.training.trainer import Trainer

def main():
    config = load_config("configs/model/tier3_roberta.yaml")
    set_seed(config['training']['seed'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Prepare Data
    print(f"Loading tokenizer: {config['model']['name']}...")
    tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
    
    print("Creating DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(config, tokenizer)
    
    # 2. Calculate Class Weights
    print("Calculating class weights...")
    df = pd.read_csv(config['paths']['train'])
    label_cols = config['labels']
    class_counts = [df[col].sum() for col in label_cols]
    total_samples = len(df)
    
    criterion = WeightedBCELoss(class_counts, total_samples, device)
    
    # 3. Model
    print(f"Initializing {config['model']['name']}...")
    model = ToxicTransformer(
        model_name=config['model']['name'],
        num_labels=config['model']['num_labels']
    ).to(device)
    
    # Optimization parameters (simple global LR for now as per plan, though differential is better)
    optimizer = optim.AdamW(model.parameters(), lr=float(config['training']['learning_rate']))
    
    # 4. Trainer
    trainer = Trainer(model, optimizer, criterion, device, checkpoint_dir='models/roberta')
    
    num_epochs = config['training']['epochs']
    print(f"Starting training for {num_epochs} epochs...")
    
    for epoch in range(1, num_epochs + 1):
        train_loss = trainer.train_one_epoch(train_loader, epoch)
        val_loss = trainer.evaluate(val_loader)
        
        print(f"Epoch {epoch}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        trainer.save_checkpoint(val_loss, model_name='roberta_best.pt')
        
    print("Training Complete!")

if __name__ == "__main__":
    main()
