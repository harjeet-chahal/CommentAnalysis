
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from .preprocessing import clean_text

class JigsawDataset(Dataset):
    def __init__(self, df, tokenizer, max_len, label_columns=None):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_columns = label_columns
        self.text = df['comment_text'].apply(clean_text).values
        
        if self.label_columns:
            self.labels = df[self.label_columns].values
        else:
            self.labels = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        text = self.text[index]
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']

        item = {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long)
        }

        if self.labels is not None:
            item['targets'] = torch.tensor(self.labels[index], dtype=torch.float)
            
        return item

def get_dataloaders(config, tokenizer):
    """
    Creates DataLoaders for train, validation, and test sets using Stratified Shuffle Split.
    
    Args:
        config (dict): Configuration dictionary loaded from yaml.
        tokenizer: Transformers tokenizer.
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    df = pd.read_csv(config['paths']['train'])
    
    # 80% Train, 10% Val, 10% Test
    # Since StratifiedShuffleSplit does not support multi-label natively, 
    # we create a 'stratify_group' by checking if any label is present (binary 'is_toxic')
    # or use the 'toxic' column as the main stratification target as it's the most common.
    # A more robust way for simple projects is to stratify on the 'toxic' label 
    # or the combination of labels if mapped to strings, but keeping it simple as requested.
    
    # Strategy: Stratify on the 'toxic' column (most representative)
    # Alternatively, we could create a 'hash' of all labels, but rare combinations breaks stratification (singletons).
    # We will stratify on the 'toxic' column for now.
    
    split_col = df['toxic'].values
    
    # First split: 80% Train, 20% Temp (Val + Test)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=config['training']['seed'])
    
    for train_index, temp_index in sss.split(df, split_col):
        train_df = df.iloc[train_index]
        temp_df = df.iloc[temp_index]
        temp_split_col = split_col[temp_index]
        
    # Second split: Split the 20% Temp into 50% Val and 50% Test (resulting in 10% total each)
    sss_val_test = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=config['training']['seed'])
    
    for val_index, test_index in sss_val_test.split(temp_df, temp_split_col):
        val_df = temp_df.iloc[val_index]
        test_df = temp_df.iloc[test_index]
        
    print(f"Train Size: {len(train_df)}")
    print(f"Val Size: {len(val_df)}")
    print(f"Test Size: {len(test_df)}")

    label_cols = config['labels']
    max_len = config['training']['max_len']
    batch_size = config['training']['batch_size']
    
    train_dataset = JigsawDataset(train_df, tokenizer, max_len, label_cols)
    val_dataset = JigsawDataset(val_df, tokenizer, max_len, label_cols)
    # Note: If we had a separate test set with labels, we'd use it. 
    # But here we are splitting the training data to create a local test set for evaluation.
    test_dataset = JigsawDataset(test_df, tokenizer, max_len, label_cols)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, test_loader
