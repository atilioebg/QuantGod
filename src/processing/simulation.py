import polars as pl
import numpy as np
import sys

def build_simulated_book(trades_df: pl.DataFrame, window: str = "15m", price_levels: int = 128) -> pl.DataFrame:
    """
    Reconstrói um Order Book simulado a partir de trades - VERSÃO OTIMIZADA PARA GRANDES VOLUMES.
    Processa em chunks temporais para evitar OOM.
    Retorna um DataFrame compatível com o Tensor Builder (formato longo/profile).
    """
    print(f"         [DEBUG] Iniciando simulação OTIMIZADA...", flush=True)
    sys.stdout.flush()
    print(f"         [DEBUG] Input: {trades_df.height:,} trades", flush=True)
    sys.stdout.flush()
    
    # 1. Preparar timestamps
    print(f"         [DEBUG] Passo 1: Truncando timestamps...", flush=True)
    sys.stdout.flush()
    
    # Verificação de tipo
    if trades_df.schema["timestamp"] in (pl.UInt64, pl.Int64):
        df = trades_df.with_columns(
            pl.from_epoch("timestamp", time_unit="ms").alias("datetime")
        ).with_columns(
             pl.col("datetime").dt.truncate(window).alias("snapshot_time")
        )
    else:
        # Já é datetime
        df = trades_df.with_columns(
            pl.col("timestamp").dt.truncate(window).alias("snapshot_time")
        )
    
    # 2. OTIMIZAÇÃO: Processar em chunks de 1 dia para evitar OOM
    print(f"         [DEBUG] Passo 2: Processando em chunks diários...", flush=True)
    sys.stdout.flush()
    
    # Obter range de datas
    df = df.with_columns(
        pl.col("snapshot_time").dt.date().alias("date")
    )
    
    unique_dates = df.select("date").unique().sort("date")
    total_dates = unique_dates.height
    
    print(f"         [DEBUG] Total de dias: {total_dates}", flush=True)
    sys.stdout.flush()
    
    profiles = []
    
    for i, date_row in enumerate(unique_dates.iter_rows(named=True)):
        date = date_row["date"]
        print(f"         [DEBUG] Processando dia {i+1}/{total_dates}: {date}", flush=True, end="\r")
        sys.stdout.flush()
        
        # Filtrar trades deste dia
        day_df = df.filter(pl.col("date") == date)
        
        # Group by para este dia
        day_profile = (
            day_df.group_by(["snapshot_time", "price"])
            .agg([
                pl.col("quantity")
                .filter(pl.col("is_buyer_maker") == True)
                .sum()
                .alias("bid_vol"),
                
                pl.col("quantity")
                .filter(pl.col("is_buyer_maker") == False)
                .sum()
                .alias("ask_vol"),
                
                pl.count("quantity").alias("trade_count"),
                
                (
                    pl.col("quantity").filter(pl.col("is_buyer_maker") == False).sum().fill_null(0) -
                    pl.col("quantity").filter(pl.col("is_buyer_maker") == True).sum().fill_null(0)
                ).alias("ofi_level")
            ])
        )
        
        profiles.append(day_profile)
    
    print(f"\n         [DEBUG] Passo 3: Concatenando {len(profiles)} dias...", flush=True)
    sys.stdout.flush()
    
    if not profiles:
        # Retorna DataFrame vazio com schema correto
        schema = {
            "snapshot_time": pl.Datetime, 
            "price": pl.Float32, 
            "bid_vol": pl.Float32, 
            "ask_vol": pl.Float32, 
            "trade_count": pl.UInt32, 
            "ofi_level": pl.Float32
        }
        return pl.DataFrame([], schema=schema)

    # Concatenar todos os dias
    profile = pl.concat(profiles).sort("snapshot_time")
    
    print(f"         [DEBUG] Passo 4: Concluído!", flush=True)
    sys.stdout.flush()
    print(f"         [DEBUG] Output: {profile.height:,} price-levels", flush=True)
    sys.stdout.flush()

    return profile
