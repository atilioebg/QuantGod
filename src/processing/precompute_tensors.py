import torch
import polars as pl
import numpy as np
from pathlib import Path
from tqdm import tqdm
import gc
from src.config import settings
from src.processing.simulation import build_simulated_book
from src.processing.tensor_builder import build_tensor_6d
from src.processing.labeling import generate_hierarchical_labels
from src.training.train import generate_month_list
from src.processing.processor import extract_meta_features

def precompute_all_data():
    """
    Transforma trades brutos e klines em Tensores 6D e Labels prontos para o ViViT.
    Salva em data/processed/tensors/ para treino ultra-rápido.
    
    [NEW] Também gera Artefatos para o Risk Manager (XGBoost):
    - data/processed/meta/meta_{month}.parquet (Tabular Features)
    - data/processed/meta/target_{month}.csv  (Aligned Labels)
    """
    print("=" * 60)
    print("QUANT GOD - PRECOMPUTE PIPELINE")
    print("=" * 60)
    
    # 1. Setup de Diretórios
    output_dir_tensors = settings.PROCESSED_DIR / "tensors"
    output_dir_meta = settings.PROCESSED_DIR / "meta"
    
    output_dir_tensors.mkdir(parents=True, exist_ok=True)
    output_dir_meta.mkdir(parents=True, exist_ok=True)
    
    # 2. Lista Total de Meses (2020-01 até 2026-01)
    all_months = generate_month_list("2020-01", "2026-01")
    
    print(f"[INFO] Processando {len(all_months)} meses...")
    
    for month in tqdm(all_months, desc="Meses"):
        out_file_tensor = output_dir_tensors / f"processed_{month}.npz"
        out_file_meta = output_dir_meta / f"meta_{month}.parquet"
        out_file_target = output_dir_meta / f"target_{month}.csv"
        
        # Pular se TODOS já existirem
        if out_file_tensor.exists() and out_file_meta.exists() and out_file_target.exists():
            continue
            
        t_file = settings.RAW_HISTORICAL_DIR / f"aggTrades_{month}.parquet"
        k_file = settings.RAW_HISTORICAL_DIR / f"klines_{month}.parquet"
        
        if not t_file.exists() or not k_file.exists():
            print(f"\n   [WARN] Arquivos faltando para {month}. Pulando...")
            continue
            
        try:
            # 1. Carregar Dados
            df_trades = pl.read_parquet(t_file)
            df_klines = pl.read_parquet(k_file)
            
            # Ajustar colunas de tempo
            if "transact_time" in df_trades.columns:
                df_trades = df_trades.with_columns(pl.col("transact_time").alias("timestamp"))
            
            if "open_time" in df_klines.columns:
                df_klines = df_klines.with_columns(
                    pl.from_epoch(pl.col("open_time"), time_unit="ms").alias("timestamp")
                )

            # ====================================================================
            # ARTEFATO 1: TABULAR META-FEATURES (XGBoost)
            # ====================================================================
            # Calcula features técnicas (RSI, EMA, Volatilidade)
            # Retorna Pandas DataFrame
            meta_df_pandas = extract_meta_features(df_klines)
            
            # Adicionar coluna timestamp string para join seguro
            # O índice do meta_df é numérico (reset_index), mas precisamos do tempo
            # extract_meta_features retorna DataFrame com colunas, mas o 'datetime' pode ou não ser index/coluna.
            # Olhando processor.py: df.to_pandas()... se datetime era coluna, virou coluna.
            # Feature feature: convert 'datetime' to timestamp string for key
            if "datetime" in meta_df_pandas.columns:
                 meta_df_pandas["timestamp_str"] = meta_df_pandas["datetime"].astype(str)
            else:
                 # Fallback: tentar recuperar do df_klines original se alinhamento for garantido (são 1:1)
                 # Mas extract_meta_features pode ter dropado linhas (e.g. janelas)?
                 # Ele usa fillna(0), então deve manter shape.
                 ts_list = df_klines["timestamp"].dt.to_string().to_list()
                 meta_df_pandas["timestamp_str"] = ts_list[:len(meta_df_pandas)]
            
            # Salvar Meta Features
            meta_df_pandas.to_parquet(out_file_meta, index=False)


            # ====================================================================
            # ARTEFATO 2: SIMULATION (ViViT) & LABELS (Common)
            # ====================================================================
            # 2. Simulação de Order Book
            sim_book = build_simulated_book(df_trades, window=settings.SIM_WINDOW)
            
            # 3. Geração de Labels Hierárquicos
            df_labels = generate_hierarchical_labels(
                df_klines,
                window_hours=settings.LABEL_WINDOW_HOURS,
                target_pct=settings.LABEL_TARGET_PCT,
                stop_pct=settings.LABEL_STOP_PCT
            )
            
            # 4. Sincronização (Join)
            # Sincronizar pelo snapshot_time (15min)
            df_labels = df_labels.with_columns(
                pl.col("timestamp").dt.truncate("15m").alias("snapshot_time")
            ).filter(pl.col("timestamp") == pl.col("snapshot_time"))
            
            # Inner Join para garantir alinhamento perfeito
            dataset_chunk = sim_book.join(df_labels, on="snapshot_time", how="inner")
            dataset_chunk = dataset_chunk.drop_nulls(subset=["label"])
            
            if dataset_chunk.height == 0:
                print(f"\n   [WARN] {month} resultou em chunk vazio.")
                continue
                
            # ====================================================================
            # ARTEFATO 3: TARGETS (XGBoost)
            # ====================================================================
            # Salvar labels alinhados com timestamp (chave de join)
            # Selecionar apenas colunas essenciais
            target_df_pandas = dataset_chunk.select([
                pl.col("snapshot_time").dt.to_string().alias("timestamp"), # Chave
                "label",
                "close" # Útil para debug/visualização
            ]).to_pandas()
            
            target_df_pandas.to_csv(out_file_target, index=False)
            
            # ====================================================================
            # ARTEFATO 4: TENSOR 6D (ViViT)
            # ====================================================================
            # 5. Build Tensor 6D (Canais: Bids, Asks, OFI, Price, OFI_W, Price_W)
            X_np = build_tensor_6d(dataset_chunk, n_levels=128, is_simulation=True)
            
            # Extrair Labels (Y) para Tensor (garantir ordenação)
            dataset_chunk = dataset_chunk.sort("snapshot_time")
            Y_np = dataset_chunk["label"].to_numpy()
            
            # 6. Salvar (npz para compressão eficiente)
            np.savez_compressed(
                out_file_tensor, 
                x=X_np.astype(np.float32), 
                y=Y_np.astype(np.int32),
                timestamps=dataset_chunk["snapshot_time"].dt.to_string().to_list()
            )
            
            # Limpeza de Memória
            del df_trades, df_klines, sim_book, df_labels, dataset_chunk, X_np, Y_np, meta_df_pandas, target_df_pandas
            gc.collect()
            
        except Exception as e:
            print(f"\n   [ERR] Falha ao processar {month}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "=" * 60)
    print("PRÉ-PROCESSAMENTO CONCLUÍDO COM SUCESSO!")
    print(f"Dados salvos em:\n - {output_dir_tensors}\n - {output_dir_meta}")
    print("=" * 60)

if __name__ == "__main__":
    precompute_all_data()
