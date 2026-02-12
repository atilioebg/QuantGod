import pytest
import polars as pl
from src.processing.features import calculate_ofi, calculate_volatility, calculate_liquidation_proxy
from datetime import datetime

def test_calculate_ofi():
    # Mock Data
    # 2 Trades: 1 Buy (agg=False), 1 Sell (agg=True) at same price
    # Buy Volume = 10, Sell Volume = 5 -> OFI = 5
    data = pl.DataFrame({
        "timestamp": [1000, 2000],  # ms
        "price": [100.0, 100.0],
        "quantity": [10.0, 5.0],
        "is_buyer_maker": [False, True]  # False=Buy Aggression, True=Sell Aggression
    })
    
    # Run
    # Window "1s" -> Both in same bucket if 1000, 2000 ms = 1s, 2s. 
    # Let's align timestamps to be in sum interval. 1000ms = 1s. 2000ms = 2s.
    # Group by dynamic "1m". Both should be in minute 0.
    
    # Fix timestamps to be robust for dynamic GroupBy
    # 1000 ms could be truncated.
    
    result = calculate_ofi(data, window="1m")
    
    assert result.height == 1
    assert result["vol_buy"][0] == 10.0
    assert result["vol_sell"][0] == 5.0
    assert result["ofi"][0] == 5.0
    assert result["tib"][0] == pytest.approx(5.0 / 15.0, rel=1e-5)

def test_calculate_volatility():
    # Mock Data: Price going 100 -> 110 -> 100 -> 120
    # Log returns: ln(1.1), ln(0.909), ln(1.2)
    data = pl.DataFrame({
        "timestamp": [1000, 2000, 3000, 4000], 
        "price": [100.0, 110.0, 100.0, 120.0],
        "quantity": [1, 1, 1, 1],
        "is_buyer_maker": [False, False, False, False]
    })
    
    result = calculate_volatility(data, window="1m")
    
    assert "volatility" in result.columns
    assert result["volatility"][0] > 0
    assert result["vwap"][0] == 107.5  # (100+110+100+120)/4

def test_calculate_liquidation_proxy():
    # Mock High Volume Burst
    data = pl.DataFrame({
        "timestamp": [1000, 1001, 1002],
        "price": [100.0, 100.0, 100.0],
        "quantity": [1000.0, 1000.0, 10.0], # Burst
        "is_buyer_maker": [True, True, True]
    })
    
    result = calculate_liquidation_proxy(data)
    # Aggregated by 1s.
    # 1000, 1001, 1002 ms -> All in same second (00:00:01)? 
    # Timestamp is ms from epoch. 1000ms = 1s. 2000ms = 2s.
    # 1000, 1001, 1002 are in second 1.
    
    assert result["vol_1s"][0] == 2010.0
    assert result["count"][0] == 3
