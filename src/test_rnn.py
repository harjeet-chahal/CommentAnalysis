
import torch
import torch.optim as optim
from src.modeling.rnn import BiLSTMClassifier
from src.training.loss import WeightedBCELoss
from src.training.trainer import Trainer

def test_rnn_training_step():
    print("Testing BiLSTM Training Step...")
    
    # Mock Data
    batch_size = 4
    seq_len = 10
    vocab_size = 100
    num_labels = 6
    embedding_dim = 32 # small for test
    hidden_dim = 16
    
    # Random inputs
    ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    targets = torch.rand((batch_size, num_labels)) # BCE targets are floats
    
    batch = {
        'ids': ids,
        'targets': targets
    }
    
    dataloader = [batch] # Mock dataloader
    
    # Model
    model = BiLSTMClassifier(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_labels=num_labels
    )
    
    # Loss
    class_counts = [2, 2, 2, 2, 2, 2] # Dummy counts
    total_samples = 10
    criterion = WeightedBCELoss(class_counts, total_samples, torch.device("cpu"))
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Trainer
    trainer = Trainer(model, optimizer, criterion, torch.device("cpu"), checkpoint_dir='models')
    
    # Run 1 epoch
    try:
        loss = trainer.train_one_epoch(dataloader, epoch=1)
        print(f"Training Step Success. Loss: {loss:.4f}")
    except Exception as e:
        print(f"Training Step FAILED: {e}")
        raise e
        
    print("Testing BiLSTM Evaluation Step...")
    try:
        val_loss = trainer.evaluate(dataloader)
        print(f"Eval Step Success. Loss: {val_loss:.4f}")
    except Exception as e:
        print(f"Eval Step FAILED: {e}")
        raise e
        
    print("SUCCESS: LSTM Architecture and Trainer verified.")

if __name__ == "__main__":
    test_rnn_training_step()
