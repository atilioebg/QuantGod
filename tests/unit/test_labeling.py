import pytest
import polars as pl
import numpy as np
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to sys.path
sys.path.append(str(Path.cwd()))

from src.processing.labeling import generate_labels

def test_triple_barrier_hit_target():
    # Setup: 
    # Current Close = 100
    # Target +10% = 110
    # Next High = 111 (Hit Target)
    
    start = datetime(2024, 1, 1, 10, 0)
    
    klines_df = pl.DataFrame({
        "timestamp": [start, start + timedelta(hours=1)],
        "close": [100.0, 105.0],
        "high": [100.0, 111.0], # Next candle hits 111
        "low": [100.0, 104.0]
    })
    
    # Target 10%, Stop 10%
    labels_df = generate_labels(
        klines_df, window_hours=2, target_pct=0.10, stop_pct=0.10
    )
    
    # First candle (index 0): 
    # Future window checks index 1.
    # Index 1 High is 111 >= 100 * 1.10 (110). Target hit.
    # Index 1 Low is 104 >= 100 * 0.90 (90). Stop NOT hit.
    # Label should be 2 (Buy/Target)
    
    print(labels_df)
    assert labels_df["label"][0] == 2

def test_triple_barrier_hit_stop():
    # Setup:
    # Current Close = 100
    # Stop -10% = 90
    # Next Low = 89 (Hit Stop)
    
    start = datetime(2024, 1, 1, 10, 0)
    
    klines_df = pl.DataFrame({
        "timestamp": [start, start + timedelta(hours=1)],
        "close": [100.0, 95.0],
        "high": [100.0, 96.0], 
        "low": [100.0, 89.0] # Next candle hits 89
    })
     
    labels_df = generate_labels(
        klines_df, window_hours=2, target_pct=0.10, stop_pct=0.10
    )
    
    # Label should be 1 (Sell/Stop)
    assert labels_df["label"][0] == 1

def test_triple_barrier_timeout():
    # Neutral market
    start = datetime(2024, 1, 1, 10, 0)
    
    # 3 candles, price stays flat
    klines_df = pl.DataFrame({
        "timestamp": [start, start + timedelta(hours=1), start + timedelta(hours=2)],
        "close": [100.0, 100.0, 100.0],
        "high": [101.0, 101.0, 101.0], # Never hits 110
        "low": [99.0, 99.0, 99.0] # Never hits 90
    })
    
    labels_df = generate_labels(
        klines_df, window_hours=2, target_pct=0.10, stop_pct=0.10
    )
    
    # Label should be 0 (Neutral)
    assert labels_df["label"][0] == 0

def test_triple_barrier_conflict():
    # Conservative logic: if both hit in same candle, prioritize Stop
    start = datetime(2024, 1, 1, 10, 0)
    
    klines_df = pl.DataFrame({
        "timestamp": [start, start + timedelta(hours=1)],
        "close": [100.0, 100.0],
        "high": [100.0, 115.0], # Hits Target (110)
        "low": [100.0, 85.0] # Hits Stop (90)
    })
    
    # Assuming code prioritizes Stop (Class 1) if indices are same
    labels_df = generate_labels(
        klines_df, window_hours=2, target_pct=0.10, stop_pct=0.10
    )
    
    assert labels_df["label"][0] == 1
