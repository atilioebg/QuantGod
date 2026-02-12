import pytest
import polars as pl
import numpy as np
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to sys.path
sys.path.append(str(Path.cwd()))

from src.processing.simulation import build_simulated_book

def test_build_simulated_book_small():
    # Create trades for a single 15m window
    # Timestamp base: 2024-01-01 10:00:00
    base_ts = 1704103200000 
    
    # Trades:
    # 1. Price 100, Qty 10, Sell (m=True) -> Bid Vol
    # 2. Price 100, Qty 5, Buy (m=False) -> Ask Vol
    # 3. Price 101, Qty 2, Buy (m=False) -> Ask Vol
    
    trades_df = pl.DataFrame({
        "timestamp": [base_ts, base_ts + 1000, base_ts + 2000],
        "price": [100.0, 100.0, 101.0],
        "quantity": [10.0, 5.0, 2.0],
        "is_buyer_maker": [True, False, False]
    })
    
    # Clean/cast types as expected by simulation
    trades_df = trades_df.with_columns([
        pl.col("price").cast(pl.Float32),
        pl.col("quantity").cast(pl.Float32),
        pl.col("is_buyer_maker").cast(pl.Boolean)
    ])
    
    sim_book = build_simulated_book(trades_df, window="15m")
    
    # Check output
    # Should have 2 rows (one for price 100, one for price 101) for the same snapshot
    assert sim_book.height == 2
    assert "bid_vol" in sim_book.columns
    assert "ask_vol" in sim_book.columns
    assert "ofi_level" in sim_book.columns
    
    # Filter for price 100
    row_100 = sim_book.filter(pl.col("price") == 100.0)
    assert row_100["bid_vol"][0] == 10.0 # m=True (Sell Aggression hits Bid)
    assert row_100["ask_vol"][0] == 5.0  # m=False (Buy Aggression hits Ask)
    assert row_100["ofi_level"][0] == 5.0 - 10.0 # Ask Vol - Bid Vol = -5? 
    # Logic in simulation.py: 
    # ofi = (Buy_Maker==False sum) - (Buy_Maker==True sum)
    #     = Ask Vol - Bid Vol
    #     = 5 - 10 = -5. Correct.
    
    # Filter for price 101
    row_101 = sim_book.filter(pl.col("price") == 101.0)
    assert row_101["bid_vol"][0] == 0.0
    assert row_101["ask_vol"][0] == 2.0
    assert row_101["ofi_level"][0] == 2.0

def test_simulation_empty():
    empty_df = pl.DataFrame({
        "timestamp": [], "price": [], "quantity": [], "is_buyer_maker": []
    }, schema={
        "timestamp": pl.Int64, "price": pl.Float32, "quantity": pl.Float32, "is_buyer_maker": pl.Boolean
    })
    
    sim_book = build_simulated_book(empty_df, window="15m")
    assert sim_book.height == 0
