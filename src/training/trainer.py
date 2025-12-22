
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import os

class Trainer:
    def __init__(self, model, optimizer, criterion, device, checkpoint_dir='models'):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            
        self.best_val_loss = float('inf')

    def train_one_epoch(self, dataloader, epoch):
        self.model.train()
        train_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Training Epoch {epoch}")
        
        for batch in pbar:
            ids = batch['ids'].to(self.device)
            mask = batch['mask'].to(self.device)
            token_type_ids = batch['token_type_ids'].to(self.device)
            targets = batch['targets'].to(self.device)
            
            self.optimizer.zero_grad()
            
            logits = self.model(ids, mask, token_type_ids)
            loss = self.criterion(logits, targets)
            
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        return train_loss / len(dataloader)

    def evaluate(self, dataloader):
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                ids = batch['ids'].to(self.device)
                mask = batch['mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                targets = batch['targets'].to(self.device)
                
                logits = self.model(ids, mask, token_type_ids)
                loss = self.criterion(logits, targets)
                val_loss += loss.item()
                
        return val_loss / len(dataloader)

    def save_checkpoint(self, val_loss, model_name='best_model.pt'):
        if val_loss < self.best_val_loss:
            print(f"Validation loss improved from {self.best_val_loss:.4f} to {val_loss:.4f}. Saving model...")
            self.best_val_loss = val_loss
            path = os.path.join(self.checkpoint_dir, model_name)
            torch.save(self.model.state_dict(), path)
