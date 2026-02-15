import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Caminho para os dados
DATA_DIR = Path("data/raw/historical")

def audit_data():
    print("============================================================")
    print("AUDITORIA DE DADOS L1 (OHLCV) - QUANT GOD")
    print("============================================================")
    
    # 1. Identificar arquivos (Todos disponiveis, foco 2019+)
    files = sorted(list(DATA_DIR.glob("klines_20*.parquet")))
    
    if not files:
        print("[ERRO] Nenhum arquivo encontrado em", DATA_DIR)
        return

    print(f"[INFO] Encontrados {len(files)} arquivos para auditoria (Historico Completo).")
    
    # 2. Carregar Dados
    dfs = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            dfs.append(df)
        except Exception as e:
            print(f"[ERRO] Falha ao ler {f}: {e}")
            
    if not dfs:
        print("[ERRO] Falha ao carregar dataframes.")
        return

    full_df = pd.concat(dfs)
    
    # Garantir datetime e ordenar
    if 'open_time' in full_df.columns:
        full_df['datetime'] = pd.to_datetime(full_df['open_time'], unit='ms')
    elif 'datetime' not in full_df.columns:
         # Tenta inferir se o index é datetime ou se tem outra coluna
         pass 

    # Se o indice não for datetime, setar
    if 'datetime' in full_df.columns:
        full_df = full_df.set_index('datetime').sort_index()
    
    print(f"[INFO] Dataset Carregado: {len(full_df)} linhas.")
    print(f"[INFO] Range: {full_df.index.min()} até {full_df.index.max()}")
    
    # 3. Health Checks
    print("\n------------------------------------------------------------")
    print("CHECAGEM DE SAÚDE (HEALTH CHECKS)")
    print("------------------------------------------------------------")
    
    # Check de Nulos
    nulls = full_df[['open', 'high', 'low', 'close', 'volume']].isnull().sum()
    if nulls.sum() > 0:
        print("[FALHA] Nulos Encontrados:\n", nulls[nulls > 0])
    else:
        print("[OK] Zero Nulos.")
        
    # Check de Zeros
    zeros_price = (full_df[['open', 'high', 'low', 'close']] == 0).sum().sum()
    zeros_vol = (full_df['volume'] == 0).sum()
    
    if zeros_price > 0:
        print(f"[FALHA] Preços Zerados encontrados: {zeros_price}")
    else:
        print("[OK] Preços Válidos (Sempre > 0).")
        
    if zeros_vol > 0:
        print(f"[AVISO] Volume Zerado encontrados: {zeros_vol} (Pode ser normal em momentos parados).")
    else:
        print("[OK] Volume sempre > 0.")

    # Check de Continuidade (GAPS) - Assumindo 1min (ajuste se for 15m)
    # Detectar frequência inferida
    diffs = full_df.index.to_series().diff().dropna()
    inferred_freq = diffs.mode()[0]
    
    print(f"[INFO] Frequência Inferida (Moda): {inferred_freq}")
    
    gaps = diffs[diffs > inferred_freq]
    
    if len(gaps) > 0:
        print(f"[FALHA] GAPS TEMPORAIS DETECTADOS: {len(gaps)}")
        print("Maiores Gaps:")
        print(gaps.sort_values(ascending=False).head(5))
    else:
        print("[OK] Continuidade Temporal Perfeita.")

    # 4. Estatísticas Descritivas
    print("\n------------------------------------------------------------")
    print("ESTATÍSTICAS")
    print("------------------------------------------------------------")
    print(full_df[['close', 'volume']].describe().loc[['min', 'max', 'mean']])
    
    # Relatório Final
    print("\n============================================================")
    if nulls.sum() == 0 and zeros_price == 0 and len(gaps) == 0:
        print("RESULTADO: [DADOS APROVADOS]")
    else:
        print("RESULTADO: [DADOS COM PROBLEMAS] - Verifique acima.")
    print("============================================================")

if __name__ == "__main__":
    audit_data()
