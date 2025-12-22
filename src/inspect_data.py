
import pandas as pd
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import Config

def inspect_data():
    print(f"Loading data from {Config.TRAIN_PATH}...")
    try:
        df = pd.read_csv(Config.TRAIN_PATH)
    except FileNotFoundError:
        print(f"Error: File not found at {Config.TRAIN_PATH}")
        return

    print(f"Data Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    print("\n--- Label Distribution ---")
    for label in Config.LABEL_COLUMNS:
        if label in df.columns:
            count = df[label].sum()
            ratio = count / len(df)
            print(f"{label}: {count} ({ratio:.4%})")
        else:
            print(f"WARNING: Label '{label}' not found in DataFrame!")
            
    print("\n--- First 5 rows ---")
    print(df.head())

if __name__ == "__main__":
    inspect_data()
