import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import sys
import datetime
from pathlib import Path

# Adiciona raiz ao path
sys.path.append(str(Path.cwd()))

from src.config import settings
from src.processing.simulation import build_simulated_book
from src.processing.tensor_builder import build_tensor_4d
from src.processing.labeling import generate_labels

def verify_data_labels():
    print("🕵️ INICIANDO AUDITORIA VISUAL DE RÓTULOS...")
    
    # 1. Carregar uma amostra de dados
    # Tenta achar arquivos disponíveis automaticamente se o mês fixo falhar
    month = "2025-11"
    t_file = settings.RAW_HISTORICAL_DIR / f"aggTrades_{month}.parquet"
    k_file = settings.RAW_HISTORICAL_DIR / f"klines_{month}.parquet"
    
    # Fallback: Se não achar Nov/25, pega o primeiro que encontrar
    if not t_file.exists():
        files = list(settings.RAW_HISTORICAL_DIR.glob("aggTrades_*.parquet"))
        if files:
            t_file = files[0]
            # Tenta deduzir o arquivo de klines correspondente
            month_found = t_file.name.replace("aggTrades_", "").replace(".parquet", "")
            k_file = settings.RAW_HISTORICAL_DIR / f"klines_{month_found}.parquet"
            print(f"⚠️ {month} não encontrado. Usando {month_found}.")
        else:
            print("❌ Nenhum arquivo encontrado.")
            return

    print(f"📂 Carregando Trades: {t_file.name}")
    print(f"📂 Carregando Klines: {k_file.name}")
    
    df_trades = pl.read_parquet(t_file).head(500_000) 
    df_klines = pl.read_parquet(k_file)

    # --- CORREÇÃO AQUI (Padronização de Colunas) ---
    
    # 1. Ajuste Trades (transact_time -> timestamp)
    if "transact_time" in df_trades.columns:
        df_trades = df_trades.with_columns(
            pl.from_epoch(pl.col("transact_time"), time_unit="ms").alias("timestamp")
        )
        
    # 2. Ajuste Klines (open_time -> timestamp)
    # O Polars precisa de Datetime para fazer rolling window, não Int
    if "open_time" in df_klines.columns:
        df_klines = df_klines.with_columns(
            pl.from_epoch(pl.col("open_time"), time_unit="ms").alias("timestamp")
        )
    # -----------------------------------------------

    # 2. Gerar Input (Simulação) e Output (Labels)
    print("⚙️ Processando Labels...")
    
    # Parâmetros Scalping (Alinhados com train.py)
    TARGET = 0.008  # 0.8% (Alvo realista para Bitcoin intraday)
    STOP = 0.004    # 0.4% (Stop conservador, ratio 2:1)
    WINDOW = 6      # 6 horas (Janela reduzida para capturar movimentos intraday)
    
    # Agora df_klines tem a coluna 'timestamp' correta
    df_labels = generate_labels(df_klines, window_hours=WINDOW, target_pct=TARGET, stop_pct=STOP)
    
    print("⚙️ Simulando Order Book...")
    sim_book = build_simulated_book(df_trades, window="15m")
    
    # Join
    print("🔗 Cruzando dados...")
    # Arredonda timestamp das labels para 15m para bater com o simulador
    df_labels = df_labels.with_columns(pl.col("timestamp").dt.truncate("15m").alias("snapshot_time"))
    
    dataset = sim_book.join(df_labels, on="snapshot_time", how="inner")
    dataset = dataset.drop_nulls(subset=["label"]).sort("snapshot_time")
    
    print(f"✅ {dataset.height} amostras prontas para inspeção.")
    
    if dataset.height == 0:
        print("❌ Dataset vazio após o join. Verifique se as datas dos Trades e Klines batem.")
        return

    print("👀 Janela gráfica abrindo... (Pressione ENTER no terminal para avançar)")

    # 3. Loop de Visualização (Com passo de 100 para não ser repetitivo)
    step = 100
    for i in range(0, dataset.height, step):
        row = dataset.row(i, named=True)
        
        snap_time = row['snapshot_time']
        label = row['label']
        close_price = row['close_price'] 
        
        # Busca futuro para plotar
        future_klines = df_klines.filter(
            (pl.col("timestamp") >= snap_time) & 
            (pl.col("timestamp") <= snap_time + datetime.timedelta(hours=WINDOW))
        ).sort("timestamp")
        
        if future_klines.height == 0:
            continue
            
        times = future_klines["timestamp"].to_list()
        prices = future_klines["close"].to_list()
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(times, prices, color='white', linewidth=2, label="Preço Futuro")
        
        upper_barrier = close_price * (1 + TARGET)
        lower_barrier = close_price * (1 - STOP)
        
        ax.axhline(upper_barrier, color='green', linestyle='--', label="Alvo")
        ax.axhline(lower_barrier, color='red', linestyle='--', label="Stop")
        ax.axhline(close_price, color='yellow', linestyle=':', label="Entrada")
        
        label_map = {0: ("NEUTRO", "gray"), 1: ("VENDA", "red"), 2: ("COMPRA", "green")}
        lbl_name, bg_color = label_map.get(label, ("UNK", "black"))
        
        ax.set_facecolor('#1e1e1e')
        fig.patch.set_facecolor(bg_color)
        fig.patch.set_alpha(0.3)
        
        plt.title(f"Rótulo: {lbl_name} | {snap_time}", color='black', fontweight='bold')
        plt.legend()
        plt.tight_layout()
        plt.show(block=False)
        
        user_input = input(f"[{i}/{dataset.height}] Enter p/ Próximo, 'q' p/ Sair: ")
        plt.close()
        
        if user_input.lower() == 'q':
            break

if __name__ == "__main__":
    verify_data_labels()