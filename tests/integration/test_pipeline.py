import pytest
import polars as pl
import numpy as np
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to sys.path
sys.path.append(str(Path.cwd()))

from src.processing.simulation import build_simulated_book
from src.processing.tensor_builder import build_tensor_4d
from src.processing.labeling import generate_labels

def test_full_pipeline_flow():
    # 1. Setup Data (2 hours of data)
    start_time = datetime(2025, 1, 1, 10, 0, 0)
    
    # Trades: continuous activity
    # 2 hours * 60 min * 60 sec = 7200 seconds
    timestamps = [start_time + timedelta(seconds=i*30) for i in range(240)] # Every 30s
    prices = [100.0 + (i*0.01) for i in range(240)] # Upward trend
    quantities = [1.0] * 240
    is_buyer_maker = [False] * 240 # All buys
    
    trades_df = pl.DataFrame({
        "timestamp": timestamps,
        "price": prices,
        "quantity": quantities,
        "is_buyer_maker": is_buyer_maker
    })
    
    # Klines: 1 minute data
    kline_timestamps = [start_time + timedelta(minutes=i) for i in range(120)]
    highs = [100.0 + (i*0.6) + 0.1 for i in range(120)] # Reaches target
    lows = [100.0 + (i*0.6) - 0.1 for i in range(120)]
    closes = [100.0 + (i*0.6) for i in range(120)]
    
    klines_df = pl.DataFrame({
        "timestamp": kline_timestamps,
        "high": highs,
        "low": lows,
        "close": closes
    })
    
    # 2. Simulation (15m window)
    sim_book = build_simulated_book(trades_df, window="15m")
    
    # Check if simulation returned data
    assert sim_book.height > 0
    # 2 hours / 15m = 8 snapshots approx
    assert sim_book["snapshot_time"].n_unique() >= 8
    
    # 3. Features (Tensor)
    tensor = build_tensor_4d(sim_book, n_levels=128, is_simulation=True)
    assert tensor.shape[0] == sim_book["snapshot_time"].n_unique()
    assert tensor.shape[1] == 4
    assert tensor.shape[2] == 128
    
    # 4. Labeling
    # Window 1 hours
    labels_df = generate_labels(klines_df, window_hours=1, target_pct=0.01, stop_pct=0.01)
    
    # 5. Join (Alignment)
    # Simulation (15m) <-> Labels (1m)
    # We need to filter labels to 15m snapshots
    
    # Normalize label timestamps to 15m to match simulation
    labels_aligned = labels_df.with_columns(
        pl.col("timestamp").dt.truncate("15m").alias("snapshot_time")
    ).filter(
        pl.col("timestamp") == pl.col("snapshot_time")
    )
    
    dataset = sim_book.join(labels_aligned, on="snapshot_time", how="inner")
    
    # Check if join preserved data
    # Timestamps in trades: 10:00 ... 11:59
    # Timestamps in klines: 10:00 ... 11:59
    # Should match perfectly
    assert dataset.height > 0
    
    # Verify we have X and Y candidates
    # X = tensor mapped to dataset indices
    # Y = dataset["label"]
    
    # Tensor has unique snapshots. Dataset might have multiple rows per snapshot if book is long format?
    # Ah, Tensor is built from sim_book. 
    # sim_book is Long Format (Stack of Levels).
    # tensor_builder aggregates by snapshot_time internally.
    # So tensor[i] corresponds to unique_snapshots[i].
    
    unique_snaps = sim_book.select("snapshot_time").unique().sort("snapshot_time")
    
    # Check alignment
    assert len(unique_snaps) == len(tensor)
    
    # Final Dataset Construction simulation
    # Join Y to unique_snaps
    y_final = unique_snaps.join(labels_aligned, on="snapshot_time", how="left")
    
    assert len(y_final) == len(tensor)
    # Check for nulls (maybe last windows don't have labels because of future lookahead)
    # generate_labels fills 0 if not enough data, so should be fine?
    # actually generate_labels might return 0 if end of data.
    
    pass
