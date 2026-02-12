"""
Versão otimizada da função load_dataset para evitar OOM
Processar cada mês separadamente em vez de concatenar tudo primeiro
"""

def load_dataset_optimized(months: list[str], dataset_name: str = "train") -> pl.DataFrame:
    """
    Carrega e processa dados mês por mês para evitar OOM.
    
    Args:
        months: Lista de meses no formato 'YYYY-MM'
        dataset_name: Nome do dataset (para logging)
    
    Returns:
        pl.DataFrame com tensores e labels sincronizados
    """
    import polars as pl
    import gc
    from pathlib import Path
    from src.config import settings
    from src.processing.simulation import build_simulated_book
    from src.processing.labeling import generate_labels
    
    print(f"\n📚 [{dataset_name}] Carregando dados: {months}...")
    
    # OTIMIZAÇÃO: Processar cada mês separadamente
    monthly_datasets = []
    
    for month in months:
        print(f"\n   📅 Processando {month}...")
        
        # Load files
        t_file = settings.RAW_HISTORICAL_DIR / f"aggTrades_{month}.parquet"
        k_file = settings.RAW_HISTORICAL_DIR / f"klines_{month}.parquet"
        
        if not t_file.exists() or not k_file.exists():
            print(f"      ⚠️ Arquivos não encontrados, pulando...")
            continue
        
        print(f"      -> Carregando...")
        df_trades = pl.read_parquet(t_file)
        df_klines = pl.read_parquet(k_file)
        print(f"         Trades: {df_trades.height:,}, Klines: {df_klines.height:,}")
        
        # Normalize timestamps
        if "transact_time" in df_trades.columns:
            df_trades = df_trades.with_columns(pl.col("transact_time").alias("timestamp"))
        
        if "open_time" in df_klines.columns:
            df_klines = df_klines.with_columns(
                pl.from_epoch(pl.col("open_time"), time_unit="ms").alias("timestamp")
            )
        
        # Simulate Order Book
        print(f"      -> Simulando Order Book...")
        try:
            sim_book = build_simulated_book(df_trades, window="15m")
            print(f"         ✅ {sim_book.height:,} snapshots")
        except Exception as e:
            print(f"         ❌ ERRO: {e}")
            continue
        
        # Generate Labels
        print(f"      -> Gerando Labels...")
        try:
            df_labels = generate_labels(df_klines, window_hours=6, target_pct=0.008, stop_pct=0.004)
            print(f"         ✅ {df_labels.height:,} labels")
        except Exception as e:
            print(f"         ❌ ERRO: {e}")
            continue
        
        # Synchronize
        print(f"      -> Sincronizando...")
        df_labels = df_labels.with_columns(
            pl.col("timestamp").dt.truncate("15m").alias("snapshot_time")
        )
        month_dataset = sim_book.join(df_labels, on="snapshot_time", how="inner")
        month_dataset = month_dataset.drop_nulls(subset=["label"])
        
        print(f"         ✅ {month_dataset.height:,} amostras")
        monthly_datasets.append(month_dataset)
        
        # Cleanup
        del df_trades, df_klines, sim_book, df_labels, month_dataset
        gc.collect()
    
    if not monthly_datasets:
        raise FileNotFoundError(f"Nenhum dado válido para {dataset_name}: {months}")
    
    # Concatenate
    print(f"\n   -> Concatenando {len(monthly_datasets)} meses...")
    dataset_df = pl.concat(monthly_datasets)
    
    del monthly_datasets
    gc.collect()
    
    print(f"   -> [{dataset_name}] Dataset Final: {dataset_df.height:,} amostras")
    return dataset_df
