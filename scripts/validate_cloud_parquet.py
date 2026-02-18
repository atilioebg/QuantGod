
import pandas as pd
import numpy as np
import time
import os

def validate_parquet():
    file_path = 'data/L2/pre_processed/test_gdrive/2026-01-01_BTCUSDT_ob200.data.parquet'
    print(f"--- Relatório de Sanidade: Opção B ---")
    print(f"Arquivo: {file_path}")
    
    if not os.path.exists(file_path):
        print("ERRO: Arquivo não encontrado.")
        return

    # 1. ESTRUTURA E SHAPE
    start_time = time.time()
    df = pd.read_parquet(file_path)
    load_time = time.time() - start_time
    
    print(f"\n1. ESTRUTURA E SHAPE:")
    print(f"   - Shape Total: {df.shape}")
    print(f"   - Total Colunas: {len(df.columns)}")
    
    # 2. QUALIDADE DOS 200 NÍVEIS
    print(f"\n2. QUALIDADE DOS 200 NÍVEIS (Amostra Meio do Arquivo):")
    sample_idx = len(df) // 2
    row = df.iloc[sample_idx]
    
    # Check Bids (Decrescente)
    bids_p = [row[f'bid_{i}_p'] for i in range(200) if f'bid_{i}_p' in df.columns]
    bids_sorted = all(bids_p[i] >= bids_p[i+1] for i in range(len(bids_p)-1) if not np.isnan(bids_p[i+1]))
    
    # Check Asks (Crescente)
    asks_p = [row[f'ask_{i}_p'] for i in range(200) if f'ask_{i}_p' in df.columns]
    asks_sorted = all(asks_p[i] <= asks_p[i+1] for i in range(len(asks_p)-1) if not np.isnan(asks_p[i+1]))
    
    print(f"   - Bids Ordenados (Decrescente): {'✅' if bids_sorted else '❌'}")
    print(f"   - Asks Ordenados (Crescente): {'✅' if asks_sorted else '❌'}")
    
    # Check Deep Layers (150-200)
    deep_nans = df[[f'bid_{i}_p' for i in range(150, 200)]].isna().sum().sum()
    print(f"   - NaNs nas camadas 150-200: {deep_nans} (Esperado se o book for raso no momento)")

    # 3. CONSISTÊNCIA DAS FEATURES AGREGADAS
    print(f"\n3. CONSISTÊNCIA DAS FEATURES AGREGADAS:")
    # Recalcular OBI L5 manual para a amostra
    b5_sum = sum(row[f'bid_{i}_s'] for i in range(5))
    a5_sum = sum(row[f'ask_{i}_s'] for i in range(5))
    manual_obi = (b5_sum - a5_sum) / (b5_sum + a5_sum) if (b5_sum + a5_sum) > 0 else 0
    saved_obi = row['mean_deep_obi']
    
    diff = abs(manual_obi - saved_obi)
    print(f"   - OBI L5 Manual: {manual_obi:.6f}")
    print(f"   - OBI L5 Salvo:  {saved_obi:.6f}")
    print(f"   - Diferença:     {diff:.8f} {'✅' if diff < 1e-5 else '❌'}")

    # 4. TESTE DE CARREGAMENTO (TRAIN.PY SIMULATION)
    print(f"\n4. CARREGAMENTO (train.py simulation):")
    feature_cols = [
        'log_ret_open', 'log_ret_high', 'log_ret_low', 'log_ret_close',
        'volatility', 'max_spread', 'mean_obi', 'mean_deep_obi', 'log_volume'
    ]
    
    start_iso = time.time()
    X = df[feature_cols].values
    y = df['close'].values # close serves for target gen
    iso_time = time.time() - start_iso
    
    mem_usage = df.memory_usage(deep=True).sum() / 1024 / 1024
    
    print(f"   - Tempo total load Parquet: {load_time:.4f}s")
    print(f"   - Tempo isolamento 9 colunas: {iso_time:.4f}s")
    print(f"   - Consumo RAM (Doc Inteiro): {mem_usage:.2f} MB")
    print(f"   - Features Isuradas Shape: {X.shape}")

if __name__ == "__main__":
    validate_parquet()
