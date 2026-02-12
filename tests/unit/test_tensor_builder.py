import pytest
import polars as pl
import numpy as np
import sys
from pathlib import Path
from datetime import datetime

# Add project root to sys.path
sys.path.append(str(Path.cwd()))

from src.processing.tensor_builder import build_tensor_4d

def test_build_tensor_shape():
    # Create synthetic simulation data
    # 2 snapshots
    # Snapshot 1: 2024-01-01 10:00:00
    # Snapshot 2: 2024-01-01 10:15:00
    
    # We need enough rows to fill 128 levels or test padding
    n_levels = 128
    
    data_df = pl.DataFrame({
        "snapshot_time": [datetime(2024, 1, 1, 10, 0), datetime(2024, 1, 1, 10, 15)],
        "price": [100.0, 101.0],
        "bid_vol": [10.0, 20.0],
        "ask_vol": [5.0, 8.0],
        "trade_count": [100, 200],
        "ofi_level": [5.0, -10.0]
    })
    
    # Run builder
    tensor = build_tensor_4d(data_df, n_levels=n_levels, is_simulation=True)
    
    # Check shape
    # 2 snapshots -> (2, 4, 128)
    assert tensor.shape == (2, 4, 128)
    
    # Check dtype
    assert tensor.dtype == np.float32

def test_tensor_normalization():
    # Test specific values to verify normalization
    # Channel 0 (Bid): log1p(10) / 10 = log(11)/10 approx 0.2398
    # Channel 2 (OFI): tanh(5 / 10) = tanh(0.5) approx 0.4621
    
    data_df = pl.DataFrame({
        "snapshot_time": [datetime(2024, 1, 1, 10, 0)],
        "price": [100.0],
        "bid_vol": [10.0],
        "ask_vol": [0.0],
        "trade_count": [0],
        "ofi_level": [5.0]
    })
    
    tensor = build_tensor_4d(data_df, n_levels=1, is_simulation=True)
    
    # Extract values
    val_bid = tensor[0, 0, 0]
    val_ofi = tensor[0, 2, 0]
    
    expected_bid = np.log1p(10.0) / 10.0
    expected_ofi = np.tanh(5.0 / 10.0)
    
    assert np.isclose(val_bid, expected_bid, atol=1e-5)
    assert np.isclose(val_ofi, expected_ofi, atol=1e-5)
    
    # Test clipping logic (if values are huge)
    data_huge = pl.DataFrame({
        "snapshot_time": [datetime(2024, 1, 1, 10, 0)],
        "price": [100.0],
        "bid_vol": [1e9], # Huge volume
        "ask_vol": [0.0],
        "trade_count": [0],
        "ofi_level": [1e9] # Huge OFI
    })
    
    tensor_huge = build_tensor_4d(data_huge, n_levels=1, is_simulation=True)
    
    # Should be clipped to 1.0 or -1.0
    # log1p(1e9) is approx 20.7. Divided by 10 is 2.07. Clipped to 1.0.
    assert tensor_huge[0, 0, 0] == 1.0
    
    # tanh(1e8) is 1.0. Clipped to 1.0.
    assert tensor_huge[0, 2, 0] == 1.0
