
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import json
import os
import sys

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config, set_seed
from src.modeling.transformer import ToxicTransformer

# Global State
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load Config
    config_path = "configs/model/tier3_roberta.yaml"
    print(f"Loading config from {config_path}...")
    try:
        config = load_config(config_path)
    except FileNotFoundError:
        # Fallback if specific config not found (e.g. running from wrong dir)
        print("Config not found. Using defaults.")
        config = {'model': {'name': 'roberta-base', 'num_labels': 6}, 'labels': ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']}
        
    ml_models['config'] = config
    
    # Load Thresholds
    thresholds_path = "results.json"
    print(f"Loading thresholds from {thresholds_path}...")
    try:
        with open(thresholds_path, 'r') as f:
            results = json.load(f)
            ml_models['thresholds'] = results.get('thresholds', {})
            # Ensure all labels have a threshold
            for label in config['labels']:
                if label not in ml_models['thresholds']:
                     ml_models['thresholds'][label] = 0.5
    except FileNotFoundError:
        print("results.json not found! Using default 0.5 thresholds.")
        ml_models['thresholds'] = {l: 0.5 for l in config['labels']}
        
    # Load Tokenizer & Model
    model_name = config['model']['name']
    print(f"Loading Tokenizer & Model: {model_name}...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ml_models['device'] = device
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ml_models['tokenizer'] = tokenizer
    
    model = ToxicTransformer(model_name, num_labels=6)
    
    checkpoint_path = "models/tier3/roberta_best.pt"
    print(f"Loading weights from {checkpoint_path}...")
    try:
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    except FileNotFoundError:
        print("Checkpoint not found! Using initialized weights (UNTRAINED).")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Using initialized weights (UNTRAINED).")
        
    model.to(device)
    model.eval()
    ml_models['model'] = model
    
    yield
    
    # Clean up
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    scores: dict[str, float]
    flags: list[str]

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    if not request.text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
        
    model = ml_models['model']
    tokenizer = ml_models['tokenizer']
    config = ml_models['config']
    device = ml_models['device']
    thresholds = ml_models['thresholds']
    
    # Tokenize
    inputs = tokenizer(
        request.text,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    ids = inputs['input_ids'].to(device)
    mask = inputs['attention_mask'].to(device)
    
    # Inference
    # RoBERTa doesn't use token_type_ids, but model handles it safely now
    with torch.no_grad():
        logits = model(ids, mask)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()
        
    # Format Response
    scores = {}
    flags = []
    
    # Handle single batch dimension squeeze if batch size is 1
    if probs.ndim == 0:
        # Should not happen with squeeze on (1, 6) -> (6,)
        pass
        
    for i, label in enumerate(config['labels']):
        score = float(probs[i])
        scores[label] = score
        if score >= thresholds.get(label, 0.5):
            flags.append(label)
            
    return PredictResponse(scores=scores, flags=flags)


from lime.lime_text import LimeTextExplainer
import numpy as np

class ExplainRequest(BaseModel):
    text: str

class ExplainResponse(BaseModel):
    html: str

def get_prediction_probs(texts):
    """
    Helper for LIME. Takes list of strings, returns (N, num_labels) numpy array of probs.
    """
    model = ml_models['model']
    tokenizer = ml_models['tokenizer']
    device = ml_models['device']
    
    # Tokenize
    inputs = tokenizer(
        texts, 
        padding=True, 
        truncation=True, 
        max_length=128, 
        return_tensors="pt"
    )
    
    ids = inputs['input_ids'].to(device)
    mask = inputs['attention_mask'].to(device)
    
    model.eval()
    with torch.no_grad():
        logits = model(ids, mask)
        probs = torch.sigmoid(logits).cpu().numpy()
        
    return probs

@app.post("/explain", response_model=ExplainResponse)
async def explain(request: ExplainRequest):
    if not request.text:
         raise HTTPException(status_code=400, detail="Text cannot be empty.")
         
    config = ml_models['config']
    class_names = config['labels']
    
    # Create Explainer
    explainer = LimeTextExplainer(class_names=class_names)
    
    # Explain
    # num_samples determines how many perturbations LIME creates. 
    # Lower = faster but less accurate. 100 is a decent balance for interactive.
    exp = explainer.explain_instance(
        request.text, 
        get_prediction_probs, 
        num_features=10, 
        num_samples=100,
        labels=range(len(class_names))
    )
    
    # Get HTML
    # We generate HTML for the top predicted class, or we could do all. 
    # By default explain_instance computes for 'labels'.
    # as_html() usually shows the top class or specific labels.
    html_content = exp.as_html()
    
    return ExplainResponse(html=html_content)

@app.get("/health")
def health():
    return {"status": "ok"}

import csv
from datetime import datetime

class FeedbackRequest(BaseModel):
    text: str
    suggested_labels: list[str]
    user_comment: str

@app.post("/feedback")
async def feedback(request: FeedbackRequest):
    # Ensure directory exists
    feedback_dir = "data/feedback"
    os.makedirs(feedback_dir, exist_ok=True)
    
    file_path = os.path.join(feedback_dir, "corrections.csv")
    file_exists = os.path.isfile(file_path)
    
    # Write to CSV
    try:
        with open(file_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Header
            if not file_exists:
                writer.writerow(["timestamp", "text", "suggested_labels", "user_comment"])
                
            # Content
            writer.writerow([
                datetime.now().isoformat(),
                request.text,
                "|".join(request.suggested_labels),
                request.user_comment
            ])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save feedback: {str(e)}")
        
    return {"status": "success", "message": "Feedback received"}
