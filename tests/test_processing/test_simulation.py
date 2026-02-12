import pytest
import polars as pl
from src.processing.simulation import build_simulated_book
from datetime import datetime

def test_build_simulated_book():
    # Mock Trades
    # 1 Buy (Ask aggr) at 100, Qty 10
    # 1 Sell (Bid aggr) at 99, Qty 5
    # Timestamp: Both same window
    data = pl.DataFrame({
        "timestamp": [1000, 2000],
        "price": [100.0, 99.0],
        "quantity": [10.0, 5.0],
        "is_buyer_maker": [False, True] 
    })
    
    # Run
    book = build_simulated_book(data, window="1m")
    
    # Should have 2 rows (one per price level in the snapshot)
    # Snapshot time same for both.
    assert book.height == 2
    
    # Row 1: Price 99 (Bid hit)
    row_99 = book.filter(pl.col("price") == 99.0)
    assert row_99["bid_vol"][0] == 5.0
    assert row_99["ask_vol"][0] == 0.0
    assert row_99["ofi_level"][0] == -5.0
    
    # Row 2: Price 100 (Ask hit)
    row_100 = book.filter(pl.col("price") == 100.0)
    assert row_100["bid_vol"][0] == 0.0
    assert row_100["ask_vol"][0] == 10.0
    assert row_100["ofi_level"][0] == 10.0
