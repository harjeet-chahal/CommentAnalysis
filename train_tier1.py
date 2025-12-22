
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from src.modeling.baselines import Tier1Baseline
from configs.config import Config
from src.preprocessing import clean_text

def main():
    print("Loading data...")
    df = pd.read_csv(Config.TRAIN_PATH)
    
    # Basic Preprocessing
    print("Preprocessing text...")
    df['comment_text'] = df['comment_text'].apply(clean_text)
    
    X = df['comment_text'].values
    y = df[Config.LABEL_COLUMNS].values
    
    # Split Data (80/20 for baseline evaluation)
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and Train Model
    print("Training Tier 1 Baseline (TF-IDF + LR)...")
    baseline = Tier1Baseline()
    baseline.fit(X_train, y_train)
    
    # Evaluate
    print("Evaluating...")
    y_pred = baseline.predict(X_test)
    
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=Config.LABEL_COLUMNS, zero_division=0))
    
    # Save Model
    save_path = os.path.join(Config.ROOT_DIR, 'models', 'baseline_tfidf_lr.joblib')
    print(f"Saving model to {save_path}...")
    joblib.dump(baseline, save_path)
    print("Done!")

if __name__ == "__main__":
    main()
