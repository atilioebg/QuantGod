import torch
import polars as pl
import numpy as np
from pathlib import Path
import sys
import warnings

# Adiciona raiz ao path para importar módulos do projeto
sys.path.append(str(Path.cwd()))

from src.config import settings
from src.processing.simulation import build_simulated_book
from src.processing.tensor_builder import build_tensor_4d
from src.models.vivit import SAIMPViViT

warnings.filterwarnings("ignore", category=FutureWarning)

# --- CONFIGURAÇÃO DO SNIPER MODE ---
MODEL_PATH = settings.DATA_DIR / "saimp_best.pth"
STREAM_MONTH = "2026-02"  # Mês de Teste (Futuro)
SEQ_LEN = 32              # 4 Horas de Contexto (32 * 15m)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Alvos do Sniper (Devem bater com a lógica do treino)
TARGET_GAIN = 0.015  # 1.5%
STOP_LOSS = 0.0075   # 0.75%
CONFIDENCE_THRESHOLD = 0.40  # Filtro de Confiança (40%)

def run_backtest():
    print(f"⚔️ INICIANDO BACKTEST SNIPER EM {STREAM_MONTH}...")
    print(f"🎯 Alvo: {TARGET_GAIN*100}% | Stop: {STOP_LOSS*100}%")
    
    if not Path(MODEL_PATH).exists():
        print(f"❌ Erro: Modelo não encontrado em {MODEL_PATH}")
        return

    # 1. Carregar o Modelo
    print("🧠 Carregando Cérebro (SAIMPViViT)...")
    model = SAIMPViViT(seq_len=SEQ_LEN, input_channels=4, price_levels=128, num_classes=3)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()

    # 2. Localizar Arquivos de Dados
    hist_dir = settings.RAW_HISTORICAL_DIR
    files = sorted(list(hist_dir.glob(f"aggTrades_{STREAM_MONTH}.parquet")))
    
    if not files:
        stream_dir = settings.RAW_STREAM_DIR / "trades"
        search_pattern = f"**/*trades_{STREAM_MONTH.replace('-', '')}*.parquet"
        files = sorted(list(stream_dir.glob(search_pattern)))
    
    if not files:
        print(f"❌ Nenhum arquivo encontrado para {STREAM_MONTH}")
        return

    print(f"📂 Arquivos encontrados: {len(files)}")
    
    # 3. Carregar e Combinar Todos os Dados
    print("📥 Carregando e unindo dados...")
    all_trades = []
    
    for file_path in files:
        try:
            df = pl.read_parquet(file_path)
            mapping = {"p": "price", "q": "quantity", "T": "timestamp", "m": "is_buyer_maker"}
            for old, new in mapping.items():
                if old in df.columns and new not in df.columns:
                    df = df.rename({old: new})
            if "transact_time" in df.columns and "timestamp" not in df.columns:
                df = df.rename({"transact_time": "timestamp"})
            
            df = df.with_columns([
                pl.col("price").cast(pl.Float32),
                pl.col("quantity").cast(pl.Float32),
                pl.col("is_buyer_maker").cast(pl.Boolean)
            ])
            if df.schema["timestamp"] == pl.String:
                df = df.with_columns(pl.col("timestamp").cast(pl.Int64))
                
            all_trades.append(df.select(["timestamp", "price", "quantity", "is_buyer_maker"]))
        except Exception as e:
            print(f"   ⚠️ Erro ao ler {file_path.name}: {e}")

    if not all_trades:
        print("❌ Nenhum dado válido carregado.")
        return

    df_trades = pl.concat(all_trades).unique(subset=["timestamp", "price"]).sort("timestamp")
    print(f"✅ Total de trades combinados: {len(df_trades):,}")

    # 4. Iniciar Simulação Global
    print("\n🎬 Iniciando Simulação...")
    sim_book = build_simulated_book(df_trades, window="15m")
    
    df_prices = df_trades.with_columns(
        pl.from_epoch("timestamp", time_unit="ms").dt.truncate("15m").alias("snapshot_time")
    ).group_by("snapshot_time").agg([
        pl.col("price").last().alias("close_price")
    ]).sort("snapshot_time")

    if len(df_prices) < SEQ_LEN + 16:
        print(f"❌ Dados insuficientes ({len(df_prices)} snapshots).")
        return

    full_tensor_np = build_tensor_4d(sim_book, n_levels=128, is_simulation=True)
    total_snaps = len(full_tensor_np)
    
    unique_times = sim_book.select("snapshot_time").unique().sort("snapshot_time")
    price_map = unique_times.join(df_prices, on="snapshot_time", how="left").fill_null(strategy="forward")
    
    # 5. Loop de Inferência
    stats = {"total_opportunities": 0, "trades_executed": 0, "wins": 0, "losses": 0, "neutrals": 0}
    print(f"🧠 Rodando inferência sobre {total_snaps} snapshots...")
    
    for i in range(0, total_snaps - SEQ_LEN - 16, 4):
        stats["total_opportunities"] += 1
        input_seq = full_tensor_np[i : i + SEQ_LEN]
        input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1)
            confidence, prediction = torch.max(probs, 1)
            pred_idx = prediction.item()
            conf_val = confidence.item()
        
        if pred_idx != 0 and conf_val >= CONFIDENCE_THRESHOLD:
            stats["trades_executed"] += 1
            current_time = price_map['snapshot_time'][i + SEQ_LEN - 1]
            entry_price = price_map['close_price'][i + SEQ_LEN - 1]
            future_prices = price_map['close_price'][i + SEQ_LEN : i + SEQ_LEN + 16].to_list()
            max_p, min_p = max(future_prices), min(future_prices)
            
            outcome = "NEUTRO"
            if pred_idx == 2: # COMPRA
                if max_p >= entry_price * (1 + TARGET_GAIN): outcome = "WIN"
                elif min_p <= entry_price * (1 - STOP_LOSS): outcome = "LOSS"
            elif pred_idx == 1: # VENDA
                if min_p <= entry_price * (1 - TARGET_GAIN): outcome = "WIN"
                elif max_p >= entry_price * (1 + STOP_LOSS): outcome = "LOSS"
            
            if outcome == "WIN": stats["wins"] += 1
            elif outcome == "LOSS": stats["losses"] += 1
            else: stats["neutrals"] += 1
            
            icon = "🟢" if outcome == "WIN" else "🔴" if outcome == "LOSS" else "⚪"
            print(f"      {icon} {outcome} | {current_time} | Pred: {pred_idx} ({conf_val:.2f})")

    # Relatório Final
    print(f"\n========================================")
    print(f"📊 RELATÓRIO FINAL DE BACKTEST ({STREAM_MONTH})")
    print(f"========================================")
    total = stats["trades_executed"]
    if total > 0:
        win_rate = (stats["wins"] / total) * 100
        ev_r = (stats["wins"] * 2) - (stats["losses"] * 1)
        print(f"Total Trades: {total} (de {stats['total_opportunities']} oportunidades)")
        print(f"✅ Wins:       {stats['wins']}")
        print(f"❌ Losses:     {stats['losses']}")
        print(f"⚪ Neutrals:   {stats['neutrals']} (Expirou tempo)")
        print(f"🎯 Win Rate:   {win_rate:.2f}%")
        print(f"💰 Saldo Líquido (R): {ev_r:.2f}R")
    else:
        print("🤷‍♂️ Nenhum trade realizado.")

if __name__ == "__main__":
    run_backtest()