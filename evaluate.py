
import torch
import pandas as pd
import numpy as np
import json
import os
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, BertTokenizer

from src.utils import load_config, set_seed
from src.data_loader import get_dataloaders, JigsawDataset
from src.modeling.rnn import BiLSTMClassifier
from src.modeling.transformer import ToxicTransformer
from src.evaluation.metrics import calculate_metrics
from src.evaluation.thresholding import optimize_thresholds
from src.evaluation.robustness import get_length_buckets, get_noisy_dataset

def predict(model, dataloader, device):
    model.eval()
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference"):
            ids = batch['ids'].to(device)
            # Handle model-specific args
            mask = batch['mask'].to(device) if 'mask' in batch else None
            # token_type_ids = batch['token_type_ids'].to(device) if 'token_type_ids' in batch else None
            
            # Simple check for our two models
            if isinstance(model, BiLSTMClassifier):
                 # RNN expects ids (and maybe mask/tokentype if we updated signature, but primarily ids)
                 # We updated signature to accept mask/token_type_ids in Phase 5
                 logits = model(ids) 
            else:
                 # Transformer
                 token_type_ids = batch['token_type_ids'].to(device)
                 logits = model(ids, mask, token_type_ids)

            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
            
            if 'targets' in batch:
                all_targets.append(batch['targets'].numpy())
                
    return np.vstack(all_probs), np.vstack(all_targets) if all_targets else None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='rnn', choices=['rnn', 'transformer'], help='Model type to evaluate')
    parser.add_argument('--checkpoint', type=str, default='models/bilstm_best.pt', help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config file')
    args = parser.parse_args()

    # Load Config
    config = load_config(args.config)
    set_seed(config['training']['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load Tokenizer & Model

    # Load Tokenizer & Model
    model_name = config['model']['name']
    
    # Check for potential mismatch
    if 'roberta' in args.checkpoint and 'bert-base' in model_name:
        print(f"WARNING: Checkpoint '{args.checkpoint}' appears to be RoBERTa, but config specifies '{model_name}'.")
        print("Did you forget to pass '--config configs/model/tier3_roberta.yaml'?")
        
    print(f"Loading {args.model_type} model: {model_name}...")
    
    if args.model_type == 'rnn':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') # Assuming fixed for RNN as per plan
        model = BiLSTMClassifier(len(tokenizer), 300, 128, 6)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = ToxicTransformer(model_name, 6)

    # Load Checkpoint
    print(f"Loading weights from {args.checkpoint}...")
    try:
        state_dict = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state_dict)
    except FileNotFoundError:
        print("Checkpoint not found! Using random weights for demonstration.")
    except Exception as e:
        print(f"Error loading weights: {e}")
        if "Missing key(s)" in str(e) or "Unexpected key(s)" in str(e):
             print("\nCRITICAL: State Dictionary Mismatch!")
             print(f"Model expects: {model_name}")
             print("Checkpoint seems to have different keys. Please check your --config.")
        raise e

        
    model.to(device)
    
    # Data Loaders
    print("Preparing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(config, tokenizer)
    
    # 1. Threshold Tuning (on Val Set)
    print("\n--- Tuning Thresholds (Validation) ---")
    val_probs, val_targets = predict(model, val_loader, device)
    best_thresholds, best_f1s = optimize_thresholds(val_targets, val_probs)
    
    print("Optimal Thresholds per Label:")
    for i, label in enumerate(config['labels']):
        print(f"{label}: {best_thresholds[i]:.2f} (F1: {best_f1s[i]:.4f})")
        
    # 2. Main Evaluation (on Test Set)
    print("\n--- Main Evaluation (Test) ---")
    test_probs, test_targets = predict(model, test_loader, device)
    
    # Apply thresholds
    test_preds = np.zeros_like(test_probs)
    for i in range(6):
        test_preds[:, i] = (test_probs[:, i] >= best_thresholds[i]).astype(int)
        
    metrics = calculate_metrics(test_targets, test_preds, test_probs)
    print("Test Metrics:", metrics)
    
    # 3. Robustness: Length
    print("\n--- Robustness: Length Analysis ---")
    # We need texts to check length. Re-reading test df implies sync issues if shuffled.
    # Ideally our dataset/loader could return texts, or indices.
    # For simplicity, we assume we can match by index if we re-split similarly.
    # Or, simpler: Iterate the test_loader again and decode input_ids -> text.
    
    # Decoding is slow but safer for exact logic
    # Or we modify Data Loader to return 'text'. 
    # Let's decode a subset or use the 'get_dataloaders' logic to get the dataframe.
    # `get_dataloaders` doesn't return DF. We will reload and resplit (deterministic seed).
    
    df = pd.read_csv(config['paths']['train'])
    from sklearn.model_selection import StratifiedShuffleSplit
    split_col = df['toxic'].values # Same logic as data_loader
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=config['training']['seed'])
    _, temp_index = next(sss.split(df, split_col))
    temp_df = df.iloc[temp_index]
    temp_split_col = split_col[temp_index]
    sss_val_test = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=config['training']['seed'])
    _, test_index = next(sss_val_test.split(temp_df, temp_split_col))
    test_df = temp_df.iloc[test_index]
    
    test_texts = test_df['comment_text'].values
    short_idx, long_idx = get_length_buckets(test_texts)
    
    print(f"Short samples: {len(short_idx)}, Long samples: {len(long_idx)}")
    
    if len(short_idx) > 0:
        short_met = calculate_metrics(test_targets[short_idx], test_preds[short_idx])
        print("Short Metrics (Macro F1):", short_met['f1'])
    
    if len(long_idx) > 0:
        long_met = calculate_metrics(test_targets[long_idx], test_preds[long_idx])
        print("Long Metrics (Macro F1):", long_met['f1'])

    # 4. Robustness: Noise
    print("\n--- Robustness: Noise Injection ---")
    noisy_texts = get_noisy_dataset(test_texts, noise_level=0.1)
    
    # Create new loader for noisy data
    noisy_dataset = JigsawDataset(
        pd.DataFrame({'comment_text': noisy_texts, **{c: test_df[c] for c in config['labels']}}),
        tokenizer,
        config['training']['max_len'],
        config['labels']
    )
    noisy_loader = torch.utils.data.DataLoader(noisy_dataset, batch_size=config['training']['batch_size'])
    
    noisy_probs, _ = predict(model, noisy_loader, device)
    noisy_preds = np.zeros_like(noisy_probs)
    for i in range(6):
        noisy_preds[:, i] = (noisy_probs[:, i] >= best_thresholds[i]).astype(int)
        
    noisy_metrics = calculate_metrics(test_targets, noisy_preds)
    print("Noisy Metrics (Macro F1):", noisy_metrics['f1'])
    print(f"F1 Drop: {(metrics['f1'] - noisy_metrics['f1']):.4f}")

    # Save Results
    results = {
        'thresholds': {k: float(v) for k, v in zip(config['labels'], best_thresholds)},
        'metrics': metrics,
        'robustness': {
            'short_f1': short_met['f1'] if len(short_idx) > 0 else None,
            'long_f1': long_met['f1'] if len(long_idx) > 0 else None,
            'noisy_f1': noisy_metrics['f1'],
            'noise_drop': metrics['f1'] - noisy_metrics['f1']
        }
    }
    
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=4)
        print("Saved results.json")

if __name__ == "__main__":
    main()
