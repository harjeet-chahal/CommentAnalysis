
import sys
import os
import torch
import transformers
from transformers import BertTokenizer

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import get_dataloaders
from src.utils import load_config, set_seed

def verify_pipeline():
    print("Loading config...")
    config = load_config("configs/config.yaml")
    
    print("Setting seed...")
    set_seed(config['training']['seed'])
    
    print("Initializing tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(config['model']['name'])
    
    print("Creating DataLoaders (this might take a moment)...")
    train_loader, val_loader, test_loader = get_dataloaders(config, tokenizer)
    
    print("\n--- Verification Results ---")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    batch = next(iter(train_loader))
    print("\nBatch Keys:", batch.keys())
    print("Input IDs Shape:", batch['ids'].shape)
    print("Targets Shape:", batch['targets'].shape)
    
    assert batch['ids'].shape == (config['training']['batch_size'], config['training']['max_len'])
    assert batch['targets'].shape == (config['training']['batch_size'], config['model']['num_labels'])
    
    print("\nSUCCESS: Data pipeline implementation verified!")

if __name__ == "__main__":
    verify_pipeline()
