import lakeapi
import pandas as pd
from pathlib import Path
import os

# Configuração
SYMBOL = "BTC-USDT"
EXCHANGE = "BINANCE"
TABLE = "book" 
OUTPUT_DIR = Path("data/L2/raw/l2_samples")
OUTPUT_FILE = OUTPUT_DIR / "binance_btc_l2_sample.csv"

def download_and_inspect():
    print("============================================================")
    print(f"L2 DATA DOWNLOADER - {SYMBOL} ({EXCHANGE})")
    print("============================================================")
    
    # Enable anonymous access for sample data
    lakeapi.use_sample_data(anonymous_access=True)
    
    # 1. Configurar Diretório
    if not OUTPUT_DIR.exists():
        print(f"[INFO] Criando diretório: {OUTPUT_DIR}")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
    # 2. Carregar Dados (Amostra Gratuita)
    print(f"[INFO] Baixando amostra de '{TABLE}' via LakeAPI...")
    try:
        # LakeAPI gerencia o cache e download. 
        # Sem start/end, ele pega a amostra disponível (geralmente um dia ou range específico)
        df = lakeapi.load_data(
            table=TABLE, 
            symbols=[SYMBOL], 
            exchanges=[EXCHANGE], 
            start=None, 
            end=None
        )
    except Exception as e:
        print(f"[ERRO] Falha no download: {e}")
        return

    if df is None or df.empty:
        print("[ERRO] DataFrame vazio retornado.")
        return

    # 3. Inspeção
    print("\n------------------------------------------------------------")
    print("INSPEÇÃO DE DADOS")
    print("------------------------------------------------------------")
    
    # Datas
    if 'received_time' in df.columns:
        df['datetime'] = pd.to_datetime(df['received_time'])
        df = df.sort_values('datetime')
        print(f"Intervalo Temporal: {df['datetime'].min()} <--> {df['datetime'].max()}")
        print(f"Total de Registros: {len(df)}")
    else:
        print("[AVISO] Coluna 'received_time' não encontrada para ordenação.")
        
    # Colunas (Depth Check)
    print(f"\nColunas Disponíveis ({len(df.columns)}):")
    print(list(df.columns)[:10], "... [truncado]")
    
    # Verificar se temos Depth (bid_0, bid_1...)
    bids = [c for c in df.columns if 'bid_' in c]
    asks = [c for c in df.columns if 'ask_' in c]
    print(f"\nProfundidade Detectada: {len(bids)} Bids / {len(asks)} Asks")
    
    # Amostra
    print("\nPrimeiras 5 linhas:")
    print(df.head())
    
    # 4. Salvamento
    print("\n------------------------------------------------------------")
    print("SALVAMENTO")
    print("------------------------------------------------------------")
    try:
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"[SUCESSO] Arquivo salvo em: {OUTPUT_FILE}")
        print(f"Tamanho estimado: {os.path.getsize(OUTPUT_FILE) / (1024*1024):.2f} MB")
    except Exception as e:
        print(f"[ERRO] Falha ao salvar CSV: {e}")

if __name__ == "__main__":
    download_and_inspect()
