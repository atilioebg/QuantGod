import pytest
import polars as pl
import pandas as pd
import numpy as np
from src.processing.processor import extract_meta_features

def test_extract_meta_features():
    # Mock Data (Polars DataFrame as processor expects)
    # Need enough rows for indicators (RSI=14, EMA=200) => 250 rows
    n = 300
    
    # helper to generate dates
    dates = pd.date_range("2024-01-01", periods=n, freq="15min")
    
    df = pl.DataFrame({
        "datetime": dates, # processor uses 'datetime' for seasonality
        "open": np.random.randn(n) + 100,
        "high": np.random.randn(n) + 101,
        "low": np.random.randn(n) + 99,
        "close": np.random.randn(n) + 100,
        "volume": np.random.rand(n) * 1000,
        "ofi": np.random.randn(n) # Optional but good to test
    })
    
    print("\n⚙️ Extracting Meta-Features...")
    meta_df = extract_meta_features(df)
    
    print(f"   Shape: {meta_df.shape}")
    print(f"   Columns: {meta_df.columns.tolist()}")
    
    # Note: Pandas TA naming can vary. ATR might be ATRr_14 or similar.
    # Just check if 'ATR' is in the name.
    expected_cols = ["RSI_14", "EMA_9", "EMA_200"]
    # Check basic cols
    for col in expected_cols:
        assert any(col in c for c in meta_df.columns), f"Missing {col}"
    
    # Check ATR specifically
    assert any("ATR" in c for c in meta_df.columns), "Missing ATR"
    
    # Check no NaNs in the *end* (beginning will have NaNs due to lookback)
    last_row = meta_df.iloc[-1]
    assert not last_row.isna().any(), "Last row contains NaNs"

if __name__ == "__main__":
    test_extract_meta_features()
