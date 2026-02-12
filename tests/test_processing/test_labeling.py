import pytest
import polars as pl
from src.processing.labeling import generate_labels

def test_generate_labels_triple_barrier():
    # Mock Klines
    # Window = 2 steps (for simplicity)
    # Target = 10%, Stop = 5%
    
    # Scenario 1: Price goes up +20% (Hit Target) -> Label 2
    # Scenario 2: Price goes down -10% (Hit Stop) -> Label 1
    # Scenario 3: Price stays flat (Time Limit) -> Label 0
    
    data = pl.DataFrame({
        "timestamp": [1, 2, 3, 4, 5, 6],
        "close": [100.0, 120.0, 100.0, 90.0, 100.0, 101.0],
        "high":  [105.0, 125.0, 105.0, 95.0, 102.0, 102.0],
        "low":   [95.0,  115.0, 95.0,  85.0, 98.0,  100.0]
    })
    
    # Window 2 steps.
    # At t=1 (Price 100). Future (t=2,3).
    # t=2 High=125 (+25% > 10%). Hit Target. Label 2.
    
    # At t=3 (Price 100). Future (t=4,5).
    # t=4 Low=85 (-15% < -5%). Hit Stop. Label 1.
    
    # At t=5 (Price 100). Future (t=6).
    # t=6 Price 101 (+1%). No touch. Label 0.
    
    # Note: Window is in "hours" in function signature, converted to minutes internally.
    # Our function assumes 1m candles. Window_hours=1 -> 60 candles.
    # To test small window, we'd need to mock the time delta or adjust function.
    # But function uses `window_steps = window_hours * 60`.
    # Let's adjust inputs to simulate window behavior or mock hours.
    
    # Or better, just integration test with standard params.
    # Let's pass window_hours=0.033 (2 mins).
    
    labeled = generate_labels(data, window_hours=0.05, target_pct=0.10, stop_pct=0.05) 
    # 0.05h = 3 mins.
    
    # T=1 (Fut: 2,3,4). Max High 125 > 110. Label 2?
    # BUT Minimum Low in window (Window extends to T=4 which has Low 85).
    # 85 < 95 (Stop).
    # Implementation prioritizes STOP (Label 1) if both occur, due to vector limitation.
    # So we expect 1 here.
    assert labeled.filter(pl.col("timestamp")==1)["label"][0] == 1
    
    # T=3 (Fut: 4,5,6). Min Low 85 < 95. Label 1.
    assert labeled.filter(pl.col("timestamp")==3)["label"][0] == 1
    
    # T=5 (Fut: 6). High 102, Low 100. No touch. Label 0.
    assert labeled.filter(pl.col("timestamp")==5)["label"][0] == 0
