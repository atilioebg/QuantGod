import pytest
import polars as pl
import numpy as np
from src.processing.tensor_builder import build_tensor_4d
from src.processing.simulation import build_simulated_book
from datetime import datetime

def test_build_tensor_4d_simulation():
    # Mock Simulated Data (Long Format)
    # Snapshot 1: Price 100, 101, 99
    # Snapshot 2: Price 100
    
    # Create DataFrame manually
    df = pl.DataFrame({
        "snapshot_time": [datetime(2024,1,1,10,0), datetime(2024,1,1,10,0), datetime(2024,1,1,10,15)],
        "price": [100.0, 101.0, 100.0],
        "bid_vol": [10.0, 0.0, 5.0],
        "ask_vol": [0.0, 20.0, 5.0],
        "ofi_level": [-10.0, 20.0, 0.0],
        "trade_count": [1, 1, 1]
    })
    
    # Run
    tensor = build_tensor_4d(df, n_levels=10, is_simulation=True)
    
    # Expect: 2 Snapshots -> (2, 4, 10)
    assert tensor.shape == (2, 4, 10)
    
    # Check Snapshot 1 (Index 0)
    # Price 100: Bid 10 -> Ch 0
    # Price 101: Ask 20 -> Ch 1
    # Tensor fills by row. 
    # Row 0 (sorted by price? Logic sorts by price).
    # 100 < 101.
    # Index 0 -> Price 100. Bid=log1p(10) ~ 2.39
    
    # Check values
    assert tensor[0, 0, 0] > 0 # Bid Vol at price 100
    assert tensor[0, 1, 1] > 0 # Ask Vol at price 101

def test_build_tensor_4d_empty():
    df = pl.DataFrame()
    tensor = build_tensor_4d(df, n_levels=10, is_simulation=True)
    assert tensor.shape == (0, 4, 10)
