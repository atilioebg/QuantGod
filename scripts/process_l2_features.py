import pandas as pd
import numpy as np
import pywt
from pathlib import Path

# Configuração
INPUT_FILE = Path("data/L2/raw/l2_samples/binance_btc_l2_sample.csv")
OUTPUT_FILE = Path("data/processed/l2_features_1min.parquet")

def process_l2_features():
    print("============================================================")
    print("L2 FEATURE ENGINEERING - QUANT GOD")
    print("============================================================")
    
    if not INPUT_FILE.exists():
        print(f"[ERRO] Arquivo de entrada não encontrado: {INPUT_FILE}")
        return

    # 1. Carregamento Eficiente
    print(f"[INFO] Carregando {INPUT_FILE}...")
    try:
        # Carregando colunas essenciais para o cálculo (bids/asks 0 a 4)
        # Se precisar de todas, remova 'usecols'
        # columns to load: received_time, bid_0_price, bid_0_size, ..., bid_4_size, ask_0_price...
        df = pd.read_csv(INPUT_FILE)
        
        # Datetime conversion
        if 'received_time' in df.columns:
            df['datetime'] = pd.to_datetime(df['received_time'])
            df.set_index('datetime', inplace=True)
            df.sort_index(inplace=True)
        else:
            print("[ERRO] Coluna 'received_time' não encontrada.")
            return

    except Exception as e:
        print(f"[ERRO] Falha ao carregar CSV: {e}")
        return

    print(f"[INFO] Dados Carregados: {len(df)} ticks. Range: {df.index.min()} - {df.index.max()}")

    # 2. Cálculo de Features (Vectorized for speed)
    print("[INFO] Calculando Micro-Price, Imbalance e Spread...")
    
    # Aliases para facilitar
    bid0_p = df['bid_0_price']
    bid0_s = df['bid_0_size']
    ask0_p = df['ask_0_price']
    ask0_s = df['ask_0_size']
    
    # --- Micro-Price ---
    # (Bid_P * Ask_V + Ask_P * Bid_V) / (Bid_V + Ask_V)
    df['micro_price'] = (bid0_p * ask0_s + ask0_p * bid0_s) / (bid0_s + ask0_s)
    
    # --- Spread ---
    df['spread'] = ask0_p - bid0_p
    
    # --- Imbalance (OBI) Level 0 ---
    # (Bid_V - Ask_V) / (Bid_V + Ask_V)
    df['obi_l0'] = (bid0_s - ask0_s) / (bid0_s + ask0_s)
    
    # --- Deep Imbalance (Levels 0-4) ---
    # Sum of volumes for levels 0 to 4
    bid_vol_5 = df[[c for c in df.columns if 'bid_' in c and int(c.split('_')[1]) < 5 and 'size' in c]].sum(axis=1)
    ask_vol_5 = df[[c for c in df.columns if 'ask_' in c and int(c.split('_')[1]) < 5 and 'size' in c]].sum(axis=1)
    
    df['deep_obi_5'] = (bid_vol_5 - ask_vol_5) / (bid_vol_5 + ask_vol_5)
    
    # --- Weighted Mid Price (Approximation using L0-L4 volumes and prices) ---
    # Simple WMP using top 5 levels: sum(P_i * V_i) / sum(V_i)
    # bid_pv_sum = sum(df[f'bid_{i}_price'] * df[f'bid_{i}_size'] for i in range(5))
    # ask_pv_sum = sum(df[f'ask_{i}_price'] * df[f'ask_{i}_size'] for i in range(5))
    # df['wmp_5'] = (bid_pv_sum + ask_pv_sum) / (bid_vol_5 + ask_vol_5)
    
    # Vamos usar apenas Micro-Price como proxy principal de preço para o OHLC
    
    # 3. Resampling (1 Minuto)
    print("[INFO] Agregando para 1 Minuto (OHLCV)...")
    
    # Adicionando contagem de ticks como proxy de volume
    df['tick_count'] = 1
    
    resampled_ohlc = df['micro_price'].resample('1min').ohlc()
    resampled_others = df.resample('1min').agg({
        'micro_price': 'std',             # Volatilidade (desvio padrão)
        'spread': 'max',                  # Max Spread (Iliquidez)
        'obi_l0': 'mean',                 # Mean Imbalance
        'deep_obi_5': 'mean',             # Mean Deep Imbalance
        'tick_count': 'sum'               # Volume (contagem de updates)
    })
    
    # Compor o DataFrame final
    resampled = pd.concat([resampled_ohlc, resampled_others], axis=1)
    
    # Renomear colunas para garantir consistência
    resampled.columns = [
        'open', 'high', 'low', 'close', 'volatility', 
        'max_spread', 'mean_obi', 'mean_deep_obi', 'tick_count'
    ]
    
    # Remover minutos vazios
    resampled.dropna(inplace=True)
    
    # --- STATIONARITY FIX (Log-Returns) ---
    print("[INFO] Aplicando Stationarity Fix (Log-Returns)...")
    
    # Guardar o preço bruto para calcular o target ANTES de transformar em retorno
    raw_close = resampled['close'].copy()
    
    # Pre-calculo do log-return baseado no close anterior (Lag=1)
    prev_close = resampled['close'].shift(1)
    
    resampled['log_ret_open'] = np.log(resampled['open'] / prev_close)
    resampled['log_ret_high'] = np.log(resampled['high'] / prev_close)
    resampled['log_ret_low'] = np.log(resampled['low'] / prev_close)
    resampled['log_ret_close'] = np.log(resampled['close'] / prev_close)
    
    # Log Volume
    resampled['log_volume'] = np.log1p(resampled['tick_count'])
    
    # New Final Feature List (9):
    # log_ret_open, log_ret_high, log_ret_low, log_ret_close, volatility, max_spread, mean_obi, mean_deep_obi, log_volume
    final_features = [
        'log_ret_open', 'log_ret_high', 'log_ret_low', 'log_ret_close',
        'volatility', 'max_spread', 'mean_obi', 'mean_deep_obi', 'log_volume'
    ]
    
    # 4. Labeling (Baseado no preço original)
    print("[INFO] Gerando Targets (5 min forward)...")
    
    # Retorno Logarítmico em 5 candles (5 min) - Usando raw_close para evitar erro de escala
    resampled['future_ret_5m'] = np.log(raw_close.shift(-5) / raw_close)
    
    # Classes: 0 (Cai), 1 (Neutro), 2 (Sobe)
    THRESHOLD = 0.001
    resampled['target_class'] = 1 # Neutro
    resampled.loc[resampled['future_ret_5m'] > THRESHOLD, 'target_class'] = 2 # Sobe
    resampled.loc[resampled['future_ret_5m'] < -THRESHOLD, 'target_class'] = 0 # Cai
    
    # Drop NaNs (Primeira linha por causa do shift(1) e últimas 5 pelo shift(-5))
    resampled.dropna(subset=final_features + ['target_class'], inplace=True)
    
    # Filtrar apenas o necessário (Keep 'close' for audit/plots)
    resampled = resampled[final_features + ['close', 'future_ret_5m', 'target_class']]
    
    print(f"[INFO] Dataset Final Gerado: {len(resampled)} linhas, {len(final_features)} features.")
    
    # 5. Saída e Análise
    print("\n------------------------------------------------------------")
    print("AMOSTRA DE DADOS PROCESSADOS (STATIONARY)")
    print("------------------------------------------------------------")
    print(resampled[['log_ret_close', 'volatility', 'log_volume', 'target_class']].head())
    
    # Correlação
    corr_vol = resampled['volatility'].corr(resampled['future_ret_5m'])
    corr_deep = resampled['mean_deep_obi'].corr(resampled['future_ret_5m'])
    
    print("\n------------------------------------------------------------")
    print("ANÁLISE DE CORRELAÇÃO (Alpha Check)")
    print("------------------------------------------------------------")
    print(f"Correlação Volatility vs Retorno 5m: {corr_vol:.4f}")
    print(f"Correlação Deep OBI vs Retorno 5m: {corr_deep:.4f}")

    # Convertendo para float32
    cols_float = final_features + ['future_ret_5m']
    resampled[cols_float] = resampled[cols_float].astype('float32')
    
    # Criar diretório
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    resampled.to_parquet(OUTPUT_FILE)
    print(f"\n[SUCESSO] Dados salvos em: {OUTPUT_FILE}")

if __name__ == "__main__":
    process_l2_features()
