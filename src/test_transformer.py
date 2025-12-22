
import torch
import torch.optim as optim
from src.modeling.transformer import ToxicTransformer
from src.training.loss import WeightedBCELoss
from src.training.trainer import Trainer

def test_transformer_training_step():
    print("Testing ToxicTransformer Training Step...")
    
    # Mock Data
    batch_size = 2
    seq_len = 16
    num_labels = 6
    vocab_size = 100 # Not used by RoBERTa directly but good for mock logic
    
    # RoBERTa Ids (random integers within vocab range of roberta which is ~50k)
    ids = torch.randint(0, 50000, (batch_size, seq_len))
    mask = torch.ones((batch_size, seq_len), dtype=torch.long)
    token_type_ids = torch.zeros((batch_size, seq_len), dtype=torch.long)
    targets = torch.rand((batch_size, num_labels))
    
    batch = {
        'ids': ids,
        'mask': mask,
        'token_type_ids': token_type_ids,
        'targets': targets
    }
    
    dataloader = [batch] # Mock dataloader
    
    # Model
    print("Initializing Model (downloading config)...")
    # Trying BERT to see if RoBERTa specifically is the issue
    model = ToxicTransformer(model_name="bert-base-uncased", num_labels=num_labels)
    
    # Loss
    class_counts = [1, 1, 1, 1, 1, 1]
    total_samples = 2
    criterion = WeightedBCELoss(class_counts, total_samples, torch.device("cpu"))
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    
    # Trainer
    # Force CPU to debug segfault
    device = torch.device("cpu")
    print(f"Using device: {device}")
    model = model.to(device)
    
    trainer = Trainer(model, optimizer, criterion, device, checkpoint_dir='models/test_trans')
    
    # Run 1 epoch
    try:
        print("Running forward/backward pass...")
        loss = trainer.train_one_epoch(dataloader, epoch=1)
        print(f"Training Step Success. Loss: {loss:.4f}")
    except Exception as e:
        print(f"Training Step FAILED: {e}")
        raise e
        
    print("SUCCESS: Transformer Architecture and Trainer verified.")

if __name__ == "__main__":
    test_transformer_training_step()
