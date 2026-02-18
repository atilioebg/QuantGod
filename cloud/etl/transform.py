import json
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
from scipy.stats import zscore
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)

class L2Transformer:
    def __init__(self, levels: int = 200, sampling_ms: int = 1000):
        self.levels = levels
        self.sampling_ms = sampling_ms
        self.bids_book: Dict[float, float] = {}
        self.asks_book: Dict[float, float] = {}
        self.last_sample_ts: int = -1

    def reset_book(self):
        self.bids_book = {}
        self.asks_book = {}
        self.last_sample_ts = -1

    def process_message(self, msg: Dict) -> Optional[Dict]:
        """
        Processes a single L2 message (snapshot or delta) and returns a sampled row if interval reached.
        """
        msg_type = msg.get("type")
        ts = msg.get("ts")
        data = msg.get("data", {})

        if not ts: return None

        # 1. Update Orderbook
        if msg_type == "snapshot":
            self.bids_book = {float(p): float(s) for p, s in data.get("b", [])}
            self.asks_book = {float(p): float(s) for p, s in data.get("a", [])}
        else:
            # Deltas
            for p, s in data.get("b", []):
                price, size = float(p), float(s)
                if size == 0: self.bids_book.pop(price, None)
                else: self.bids_book[price] = size
            for p, s in data.get("a", []):
                price, size = float(p), float(s)
                if size == 0: self.asks_book.pop(price, None)
                else: self.asks_book[price] = size

        # 2. Temporal Sampling
        if self.last_sample_ts == -1 or ts - self.last_sample_ts >= self.sampling_ms:
            self.last_sample_ts = (ts // self.sampling_ms) * self.sampling_ms
            return self._capture_state(self.last_sample_ts)
        
        return None

    def _capture_state(self, ts: int) -> Dict:
        """Captures the top N levels (Hard Cut) and basic features."""
        # Top 200 Bids (Desc)
        sorted_bids = sorted(self.bids_book.keys(), reverse=True)[:self.levels]
        # Top 200 Asks (Asc)
        sorted_asks = sorted(self.asks_book.keys())[:self.levels]

        row = {"ts": ts}
        
        # Micro-price and basic features using Top 1
        bid0_p = sorted_bids[0] if sorted_bids else np.nan
        bid0_s = self.bids_book[bid0_p] if sorted_bids else 0.0
        ask0_p = sorted_asks[0] if sorted_asks else np.nan
        ask0_s = self.asks_book[ask0_p] if sorted_asks else 0.0

        if sorted_bids and sorted_asks:
            row['micro_price'] = (bid0_p * ask0_s + ask0_p * bid0_s) / (bid0_s + ask0_s) if (bid0_s + ask0_s) > 0 else np.nan
            row['spread'] = ask0_p - bid0_p
            row['obi_l0'] = (bid0_s - ask0_s) / (bid0_s + ask0_s) if (bid0_s + ask0_s) > 0 else 0.0
        else:
            row['micro_price'] = np.nan
            row['spread'] = np.nan
            row['obi_l0'] = 0.0

        # Hard Cut 200 levels
        for i in range(self.levels):
            if i < len(sorted_bids):
                p = sorted_bids[i]
                row[f"bid_{i}_p"] = p
                row[f"bid_{i}_s"] = self.bids_book[p]
            else:
                row[f"bid_{i}_p"] = np.nan
                row[f"bid_{i}_s"] = 0.0
                
            if i < len(sorted_asks):
                p = sorted_asks[i]
                row[f"ask_{i}_p"] = p
                row[f"ask_{i}_s"] = self.asks_book[p]
            else:
                row[f"ask_{i}_p"] = np.nan
                row[f"ask_{i}_s"] = 0.0
        
        return row

    def apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies 1min resampling, log-returns and log-volume."""
        if df.empty: return df

        df['datetime'] = pd.to_datetime(df['ts'], unit='ms')
        df.set_index('datetime', inplace=True)
        
        # Resampling 1min
        # Note: We keep only micro_price for OHLC and other aggregates as per legacy
        resampled_ohlc = df['micro_price'].resample('1min').ohlc()
        resampled_others = df.resample('1min').agg({
            'micro_price': 'std',
            'spread': 'max',
            'obi_l0': 'mean'
        })
        
        # For Log Volume, we use tick count in the interval
        df['tick_count'] = 1
        resampled_vol = df['tick_count'].resample('1min').sum()

        final_df = pd.concat([resampled_ohlc, resampled_others, resampled_vol], axis=1)
        final_df.columns = ['open', 'high', 'low', 'close', 'volatility', 'max_spread', 'mean_obi', 'tick_count']
        
        # Cleanup
        final_df.dropna(inplace=True)

        # Stationarity
        prev_close = final_df['close'].shift(1)
        final_df['log_ret_open'] = np.log(final_df['open'] / prev_close)
        final_df['log_ret_high'] = np.log(final_df['high'] / prev_close)
        final_df['log_ret_low'] = np.log(final_df['low'] / prev_close)
        final_df['log_ret_close'] = np.log(final_df['close'] / prev_close)
        final_df['log_volume'] = np.log1p(final_df['tick_count'])

        return final_df.dropna()

    def apply_zscore(self, df: pd.DataFrame, scaler_path: Optional[str] = None) -> pd.DataFrame:
        """Applies Z-Score normalization and saves/loads scaler."""
        cols_to_norm = ['volatility', 'max_spread', 'mean_obi', 'log_volume', 
                        'log_ret_open', 'log_ret_high', 'log_ret_low', 'log_ret_close']
        
        if df.empty: return df

        # Simple Z-Score for now, could be enhanced with sklearn StandardScaler for persistence
        if scaler_path and Path(scaler_path).exists():
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            df[cols_to_norm] = scaler.transform(df[cols_to_norm])
        else:
            # Calculate and save if requested
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            df[cols_to_norm] = scaler.fit_transform(df[cols_to_norm])
            if scaler_path:
                Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
                with open(scaler_path, 'wb') as f:
                    pickle.dump(scaler, f)
        
        return df
