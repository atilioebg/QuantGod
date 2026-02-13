import polars as pl
import pandas as pd
import pandas_ta as ta
import numpy as np
from pathlib import Path
from src.config import settings

def extract_meta_features(df: pl.DataFrame) -> pd.DataFrame:
    """
    Calcula um vetor de meta-features tabular rico para o XGBoost.
    Usa pandas_ta para cálculos otimizados.
    """
    # Converte para Pandas para usar pandas_ta
    pdf = df.to_pandas()
    
    # Garante que as colunas necessárias existam (baseado em klines)
    # df deve ter [datetime, open, high, low, close, volume]
    
    # 1. Tendência
    pdf.ta.ema(length=9, append=True)
    pdf.ta.ema(length=21, append=True)
    pdf.ta.ema(length=50, append=True)
    pdf.ta.ema(length=200, append=True)
    
    # Distância Crítica para EMA 200
    if "EMA_200" in pdf.columns:
        pdf["dist_ema200"] = (pdf["close"] - pdf["EMA_200"]) / pdf["EMA_200"]
    else:
        pdf["dist_ema200"] = 0.0

    # 2. Momentum & Osciladores
    pdf.ta.rsi(length=14, append=True)
    pdf.ta.adx(length=14, append=True) # Cria ADX_14, DMP_14, DMN_14

    # 3. Volatilidade
    pdf.ta.atr(length=14, append=True)
    pdf.ta.bbands(length=20, std=2, append=True)
    # Bandwidth das Bollinger Bands
    if "BBU_20_2.0" in pdf.columns and "BBL_20_2.0" in pdf.columns:
        pdf["bb_width"] = (pdf["BBU_20_2.0"] - pdf["BBL_20_2.0"]) / pdf["BBM_20_2.0"]
    else:
        pdf["bb_width"] = 0.0

    # 4. Microestrutura (Se OFI estiver disponível no df)
    if "ofi" in pdf.columns:
        pdf["ofi_sma_10"] = pdf["ofi"].rolling(10).mean()
    else:
        pdf["ofi_sma_10"] = 0.0
        
    if "spread" in pdf.columns:
        pdf["spread_avg"] = pdf["spread"].rolling(10).mean()
    else:
        pdf["spread_avg"] = 0.0

    # 5. Sazonalidade
    if "datetime" in pdf.columns:
        pdf["hour"] = pdf["datetime"].dt.hour
        pdf["day_of_week"] = pdf["datetime"].dt.dayofweek
    
    # Limpeza de NaNs gerados por indicadores de janela
    pdf = pdf.fillna(0)
    
    return pdf

def save_meta_features(df: pd.DataFrame, timestamp: str):
    """
    Salva as meta-features em formato Parquet para máxima performance.
    """
    output_dir = settings.PROCESSED_DIR / "meta"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = output_dir / f"meta_{timestamp}.parquet"
    df.to_parquet(file_path, index=False)
