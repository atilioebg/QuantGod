
import polars as pl
import numpy as np
from pathlib import Path

# Configurações
INPUT_FILE = Path("data/L2/interim/l2_merged_raw.parquet")
OUTPUT_FILE = Path("data/processed/l2_features_1min_final.parquet")

def process_bybit_features():
    print("============================================================")
    print("BYBIT L2 FEATURE ENGINEERING (Polars Optimized) 💎")
    print("============================================================")
    
    if not INPUT_FILE.exists():
        print(f"[ERRO] Arquivo de entrada não encontrado: {INPUT_FILE}")
        return

    print(f"[INFO] Carregando {INPUT_FILE}...")
    df = pl.read_parquet(INPUT_FILE)
    
    # 1. Feature Engineering em nível de Snapshot (1s)
    print("[INFO] Calculando Micro-Price, Imbalance e Spread (1s level)...")
    
    # Pre-calcular volumes acumulados para Deep OBI
    df = df.with_columns([
        (pl.col("bid_0_size") + pl.col("bid_1_size") + pl.col("bid_2_size") + pl.col("bid_3_size") + pl.col("bid_4_size")).alias("bid_vol_5"),
        (pl.col("ask_0_size") + pl.col("ask_1_size") + pl.col("ask_2_size") + pl.col("ask_3_size") + pl.col("ask_4_size")).alias("ask_vol_5")
    ])
    
    df = df.with_columns([
        # Micro Price: (BidP * AskV + AskP * BidV) / (BidV + AskV)
        ((pl.col("bid_0_price") * pl.col("ask_0_size") + pl.col("ask_0_price") * pl.col("bid_0_size")) / 
         (pl.col("bid_0_size") + pl.col("ask_0_size"))).alias("micro_price"),
        
        # Spread
        (pl.col("ask_0_price") - pl.col("bid_0_price")).alias("spread"),
        
        # OBI L0
        ((pl.col("bid_0_size") - pl.col("ask_0_size")) / (pl.col("bid_0_size") + pl.col("ask_0_size"))).alias("obi_l0"),
        
        # Deep OBI 5
        ((pl.col("bid_vol_5") - pl.col("ask_vol_5")) / (pl.col("bid_vol_5") + pl.col("ask_vol_5"))).alias("deep_obi_5")
    ])
    
    # 2. Agregação Temporal (1 Minuto)
    print("[INFO] Agrupando em janelas de 1 minuto...")
    
    # Polars group_by_dynamic requer que o datetime esteja ordenado (já está)
    resampled = (
        df.group_by_dynamic("datetime", every="1m")
        .agg([
            # OHLC do Micro-Price
            pl.col("micro_price").first().alias("open"),
            pl.col("micro_price").max().alias("high"),
            pl.col("micro_price").min().alias("low"),
            pl.col("micro_price").last().alias("close"),
            
            # Volatilidade (Std do micro_price dentro do minuto)
            pl.col("micro_price").std().alias("volatility"),
            
            # Spread Max
            pl.col("spread").max().alias("max_spread"),
            
            # Mean OBI
            pl.col("obi_l0").mean().alias("mean_obi"),
            
            # Mean Deep OBI
            pl.col("deep_obi_5").mean().alias("mean_deep_obi"),
            
            # Volume (Log da contagem de updates de 1s)
            pl.count().cast(pl.Float64).alias("tick_count")
        ])
    )
    
    # 3. Stationarity Fix (Log-Returns)
    print("[INFO] Aplicando Stationarity Fix (Log-Returns)...")
    
    # Precisamos do close anterior
    resampled = resampled.with_columns([
        pl.col("close").shift(1).alias("prev_close")
    ])
    
    # Remover a primeira linha pois não tem prev_close
    resampled = resampled.drop_nulls(subset=["prev_close"])
    
    resampled = resampled.with_columns([
        (pl.col("open") / pl.col("prev_close")).log().alias("log_ret_open"),
        (pl.col("high") / pl.col("prev_close")).log().alias("log_ret_high"),
        (pl.col("low") / pl.col("prev_close")).log().alias("log_ret_low"),
        (pl.col("close") / pl.col("prev_close")).log().alias("log_ret_close"),
        (pl.col("tick_count")).log1p().alias("log_volume")
    ])
    
    # 4. Seleção Final das 9 Features
    final_cols = [
        "log_ret_open", "log_ret_high", "log_ret_low", "log_ret_close", 
        "max_spread", "mean_obi", "mean_deep_obi", "log_volume", "volatility"
    ]
    
    # Verificar se há NaNs residuais em qualquer coluna (importante para o modelo)
    resampled = resampled.select(final_cols).drop_nulls()
    
    print(f"[INFO] Dataset Final Gerado: {len(resampled)} candles.")
    print(f"[INFO] Salvando em {OUTPUT_FILE}...")
    
    # Garantir diretório
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    resampled.write_parquet(OUTPUT_FILE)
    
    print("\n[SUCESSO] Processamento concluído!")
    print(resampled.head(5))

if __name__ == "__main__":
    process_bybit_features()
