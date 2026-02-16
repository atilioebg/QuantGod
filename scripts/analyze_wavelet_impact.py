import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path("data/processed/l2_features_1min.parquet")

def analyze():
    if not DATA_PATH.exists():
        print("Arquivo nao encontrado")
        return

    df = pd.read_parquet(DATA_PATH)
    
    features = ['close', 'max_spread', 'mean_obi', 'mean_deep_obi']
    
    print(f"{'Feature':<20} | {'Corr Raw':<10} | {'Corr Wave':<10} | {'Delta':<10}")
    print("-" * 60)
    
    for col in features:
        if f'{col}_wave' not in df.columns:
            print(f"Wavelet version for {col} not found.")
            continue
            
        corr_raw = df[col].corr(df['future_ret_5m'])
        corr_wave = df[f'{col}_wave'].corr(df['future_ret_5m'])
        delta = abs(corr_wave) - abs(corr_raw)
        
        sign = "+" if delta > 0 else ""
        print(f"{col:<20} | {corr_raw:10.4f} | {corr_wave:10.4f} | {sign}{delta:.4f}")

if __name__ == "__main__":
    analyze()
