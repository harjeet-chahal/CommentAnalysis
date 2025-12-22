
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoConfig

class ToxicTransformer(nn.Module):

    def __init__(self, model_name, num_labels=6):
        super(ToxicTransformer, self).__init__()
        print(f"ToxicTransformer initializing with: {model_name}")
        self.config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, config=self.config)

    def forward(self, ids, mask, token_type_ids=None):
        # RoBERTa does not use token_type_ids. BERT does.
        # We check model type from config to decide whether to pass it.
        # However, passing it to RoBERTa is usually ignored or handled, but let's be safe.
        
        if 'roberta' in self.config.model_type:
            output = self.model(input_ids=ids, attention_mask=mask)
        else:
            output = self.model(input_ids=ids, attention_mask=mask, token_type_ids=token_type_ids)
            
        return output.logits
