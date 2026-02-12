import pytest
import polars as pl
import numpy as np
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path.cwd()))

from src.processing.features import clean_trade_data, calculate_ofi, calculate_volatility

def test_clean_trade_data():
    raw_df = pl.LazyFrame({
        "p": ["100.0", "101.0", "102.5"],
        "q": ["1.0", "0.5", "2.0"],
        "T": [1609459200000, 1609459260000, 1609459320000],
        "m": [True, False, True]
    })
    
    cleaned_df = clean_trade_data(raw_df).collect()
    
    expected_cols = ["timestamp", "price", "quantity", "is_buyer_maker"]
    assert set(cleaned_df.columns) == set(expected_cols)
    assert cleaned_df["price"].dtype == pl.Float32
    assert cleaned_df["quantity"].dtype == pl.Float32
    assert cleaned_df["is_buyer_maker"].dtype == pl.Boolean

def test_calculate_ofi():
    # Buy (m=False): 10, Sell (m=True): 5 -> Net +5
    # Buy (m=False): 2, Sell (m=True): 8 -> Net -6
    trades_df = pl.DataFrame({
        "timestamp": [1000, 2000, 15000, 16000], 
        "price": [100.0, 100.0, 101.0, 101.0],
        "quantity": [10.0, 5.0, 2.0, 8.0],
        "is_buyer_maker": [False, True, False, True]
    })
    
    # Use small window "10s" to group
    # 0-10s: +5
    # 10-20s: -6
    
    # We pass DataFrame but function converts to LazyFrame internally if needed
    ofi_df = calculate_ofi(trades_df.lazy(), window="10s")
    
    print(ofi_df)
    
    assert ofi_df.height == 2
    
    # Sort by datetime to ensure order
    ofi_df = ofi_df.sort("datetime")
    
    assert ofi_df["ofi"][0] == 5.0
    assert ofi_df["ofi"][1] == -6.0

def test_calculate_volatility():
    # Constant return scenario
    # P1=100 -> P2=110 (10%) -> P3=121 (10%)
    trades_df = pl.DataFrame({
        "timestamp": [1000, 2000, 3000],
        "price": [100.0, 110.0, 121.0],
    })
    
    # Window covers all
    vol_df = calculate_volatility(trades_df.lazy(), window="1m")
    
    # Std dev of [0.0953, 0.0953] should be ~0
    assert vol_df["volatility"][0] < 1e-4
