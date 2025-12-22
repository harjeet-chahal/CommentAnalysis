
import torch
import os

class Config:
    # Paths
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(ROOT_DIR, 'data', 'raw')
    TRAIN_PATH = os.path.join(DATA_DIR, 'train.csv')
    TEST_PATH = os.path.join(DATA_DIR, 'test.csv')
    TEST_LABELS_PATH = os.path.join(DATA_DIR, 'test_labels.csv')
    
    # Model
    MODEL_NAME = "bert-base-uncased"
    NUM_LABELS = 6
    
    # Training
    MAX_LEN = 128
    TRAIN_BATCH_SIZE = 32
    VALID_BATCH_SIZE = 16
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    
    # Hardware
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Labels
    LABEL_COLUMNS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    # Random Seed
    SEED = 42
