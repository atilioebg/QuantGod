import pytest
import polars as pl
import numpy as np
from src.processing.tensor_builder import build_tensor_6d
from datetime import datetime

def test_build_tensor_6d_simulation():
    # Mock Simulated Data (Long Format)
    # Snapshot 1: Price 100, 101, 99
    # Snapshot 2: Price 100
    
    # Create DataFrame manually
    df = pl.DataFrame({
        "snapshot_time": [datetime(2024,1,1,10,0), datetime(2024,1,1,10,0), datetime(2024,1,1,10,15)],
        "price": [100.0, 101.0, 100.0],
        "bid_vol": [10.0, 0.0, 5.0],
        "ask_vol": [0.0, 20.0, 5.0],
        # OFI Raw
        "ofi_level": [-10.0, 20.0, 0.0],
        "trade_count": [1, 1, 1]
    })
    
    # Run
    # 6 Channels: Bids, Asks, OFI Raw, Price Raw, OFI Wavelet, Price Wavelet
    tensor = build_tensor_6d(df, n_levels=10, is_simulation=True)
    
    # Expect: 2 Snapshots -> (2, 6, 10)
    assert tensor.shape == (2, 6, 10)
    
    # Check Snapshot 1 (Index 0)
    # Price 100: Bid 10 -> Ch 0
    # Price 101: Ask 20 -> Ch 1
    
    # Check channels exist
    # Channel 4 (OFI Wavelet) and 5 (Price Wavelet) should be populated 
    # (though with few points/levels, wavelets might be trivial or zero, but shape must be correct)
    
    assert tensor.shape[1] == 6

def test_build_tensor_6d_empty():
    df = pl.DataFrame()
    tensor = build_tensor_6d(df, n_levels=10, is_simulation=True)
    assert tensor.shape == (0, 6, 10)
