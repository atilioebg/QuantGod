import polars as pl
import numpy as np

def clean_trade_data(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Padroniza os nomes das colunas da Binance para nomes legíveis.
    Binance Raw: 'p' (price), 'q' (qty), 'T' (timestamp), 'm' (is_buyer_maker)
    """
    return df.select([
        pl.col("T").alias("timestamp"),
        pl.col("p").cast(pl.Float32).alias("price"),
        pl.col("q").cast(pl.Float32).alias("quantity"),
        pl.col("m").alias("is_buyer_maker")  # True = Venda a Mercado (Agressão de Venda)
    ])

def calculate_ofi(df: pl.DataFrame, window: str = "15m") -> pl.DataFrame:
    """
    Calcula o Order Flow Imbalance (OFI) em janelas de tempo.
    """
    if isinstance(df, pl.DataFrame):
        df = df.lazy()        

    return (
        df
        .with_columns([
            pl.when(pl.col("is_buyer_maker") == False)
            .then(pl.col("quantity"))
            .otherwise(0)
            .alias("vol_buy"),
            
            pl.when(pl.col("is_buyer_maker") == True)
            .then(pl.col("quantity"))
            .otherwise(0)
            .alias("vol_sell")
        ])
        .with_columns(
            pl.from_epoch("timestamp", time_unit="ms").alias("datetime")
        )
        .group_by_dynamic("datetime", every=window)
        .agg([
            pl.sum("vol_buy"),
            pl.sum("vol_sell"),
            pl.col("price").last().alias("close_price")
        ])
        .with_columns([
            (pl.col("vol_buy") - pl.col("vol_sell")).alias("ofi"),
            ((pl.col("vol_buy") - pl.col("vol_sell")) / (pl.col("vol_buy") + pl.col("vol_sell") + 1e-9)).fill_nan(0).alias("tib") 
        ])
        .collect()
    )

def calculate_volatility(df: pl.DataFrame, window: str = "15m") -> pl.DataFrame:
    """
    Calcula volatilidade (Desvio Padrão dos Log Returns) na janela.
    """
    if isinstance(df, pl.DataFrame):
        df = df.lazy()

    return (
        df
        .with_columns(
            pl.from_epoch("timestamp", time_unit="ms").alias("datetime")
        )
        .sort("datetime")
        .with_columns(
            (pl.col("price").log() - pl.col("price").log().shift(1)).alias("log_return")
        )
        .group_by_dynamic("datetime", every=window)
        .agg([
            pl.col("log_return").std().alias("volatility"),
            pl.col("price").mean().alias("vwap")
        ])
        .collect()
        .fill_null(0)
    )

def calculate_liquidation_proxy(df: pl.DataFrame) -> pl.DataFrame:
    """
    Estima liquidações detectando spikes de volume em curtíssimo prazo (1s).
    """
    if isinstance(df, pl.DataFrame):
        df = df.lazy()

    # Simplificação vetorial: Agrupa por segundo e vê se volume > threshold (ex: top 95% ou valor fixo)
    # Para ser mais preciso precisaria de rolling window sobre linhas, mas group_by_dynamic "1s" serve como proxy.
    
    return (
        df
        .with_columns(
            pl.from_epoch("timestamp", time_unit="ms").alias("datetime")
        )
        .group_by_dynamic("datetime", every="1s")
        .agg([
            pl.col("quantity").sum().alias("vol_1s"),
            pl.col("price").mean().alias("price_avg"),
            pl.count().alias("count")
        ])
        # Filtra onde volume é anormal (ex: > 10 BTC num segundo?)
        # Idealmente threshold dinâmico, mas vamos marcar a feature 'burst_intensity'
        .with_columns(
            (pl.col("vol_1s") / pl.col("count")).alias("avg_trade_size")
        )
        .collect()
    )
