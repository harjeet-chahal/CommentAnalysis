
# CommentAnalysis: Multi-Label Toxic Comment Classification

## Overview
A strict, modular machine learning project to classify toxic comments using the Jigsaw Dataset.
Includes three tiers of modeling:
1.  **Tier 1**: TF-IDF + Logistic Regression (Baseline).
2.  **Tier 2**: BiLSTM with Global Pooling.
3.  **Tier 3**: Fine-tuned RoBERTa Transformer.

## Setup

1.  **Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Data**:
    Ensure `data/raw/` contains:
    - `train.csv`
    - `test.csv`
    - `test_labels.csv`

## Usage

Use the master CLI script `train.py` to train models.

### Tier 1: Baseline
```bash
python train.py --model_type tier1 --config configs/config.yaml
```
*Saves to `models/baseline_tfidf_lr.joblib`*

### Tier 2: LSTM
```bash
python train.py --model_type tier2 --config configs/config.yaml
```
*Saves to `models/tier2/bilstm_best.pt`*

### Tier 3: Transformer (RoBERTa)
**Note**: Use the specific config for RoBERTa parameters (e.g. Learning Rate).
```bash
python train.py --model_type tier3 --config configs/model/tier3_roberta.yaml
```
*Saves to `models/tier3/roberta_best.pt`*

## Evaluation

Run the evaluation script to generate metrics and robustness analysis (`results.json`).

```bash
python evaluate.py --model_type rnn --checkpoint models/tier2/bilstm_best.pt
# or
python evaluate.py --model_type transformer --checkpoint models/tier3/roberta_best.pt
```

## Structure
- `src/`: Source code.
  - `modeling/`: Model definitions.
  - `training/`: Loss and Trainer.
  - `evaluation/`: Metrics and Robustness.
- `configs/`: YAML configuration files.
- `models/`: Saved model checkpoints.
