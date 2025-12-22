
import torch
import torch.nn as nn

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=300, hidden_dim=128, num_labels=6, dropout=0.3):
        super(BiLSTMClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=2, # Usually 2 layers is good for this size
            bidirectional=True,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0
        )
        
        # Global Pooling (Max + Avg) -> 2 * hidden_dim * 2 (bidirectional)
        # But we pool across time, so the output dim is hidden_dim * 2 (bidirectional)
        # We concatenate Max and Avg, so it becomes hidden_dim * 2 * 2 = hidden_dim * 4
        
        self.output_dim = hidden_dim * 2 * 2 # 128 * 4 = 512
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.output_dim, num_labels)
        
    def forward(self, ids, mask=None, token_type_ids=None):
        # ids: [batch_size, seq_len]
        
        x = self.embedding(ids) # [batch_size, seq_len, emb_dim]
        
        lstm_out, _ = self.lstm(x) # [batch_size, seq_len, hidden_dim * 2]
        
        # Global Max Pooling
        # mp = torch.max(lstm_out, dim=1)[0] # [batch_size, hidden_dim * 2]
        # Valid length handling could be added, but standard max pool works for padded too usually (if padded with 0 and relu used... here raw LSTM output)
        # Masking is safer but for baseline we stick to simple max/avg
        
        avg_pool = torch.mean(lstm_out, 1)
        max_pool, _ = torch.max(lstm_out, 1)
        
        # Concatenate
        out = torch.cat((avg_pool, max_pool), 1) # [batch_size, hidden_dim * 4]
        
        out = self.dropout(out)
        logits = self.fc(out) # [batch_size, num_labels]
        
        return logits
