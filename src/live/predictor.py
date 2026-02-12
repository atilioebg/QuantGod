import torch
import numpy as np
import polars as pl
from pathlib import Path
import sys
import logging
sys.path.append(str(Path.cwd()))

from src.config import settings
from src.processing.simulation import build_simulated_book
from src.processing.tensor_builder import build_tensor_4d
from src.models.vivit import SAIMPViViT

logger = logging.getLogger("SAIMP.SniperBrain")

class SniperBrain:
    def __init__(self, model_path="data/saimp_best.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SAIMPViViT(seq_len=32, input_channels=4, price_levels=128, num_classes=3)
        
        if Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(checkpoint)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"🧠 Modelo carregado com sucesso: {model_path}")
        else:
            logger.error(f"❌ Modelo não encontrado em {model_path}")
            raise FileNotFoundError(f"Modelo não encontrado: {model_path}")

    def analyze(self, df_trades):
        if df_trades is None or df_trades.height < 1000: 
            return None
        
        try:
            # 1. Simulação do Order Book (Dado Integral para Níveis)
            sim_book = build_simulated_book(df_trades, window="15m")
            
            # --- CÁLCULO DE NÍVEIS (Zero Lag) ---
            last_snap_time = sim_book["snapshot_time"].max()
            last_rows = sim_book.filter(pl.col("snapshot_time") == last_snap_time)
            
            def get_scored_levels(rows, col_vol):
                if rows.height == 0: return []
                sorted_rows = rows.sort(col_vol, descending=True).head(3)
                levels = []
                for r in sorted_rows.iter_rows(named=True):
                    vol = r[col_vol]
                    trades = r['trade_count']
                    realness = 0.95 if trades > 0 else 0.45
                    levels.append({"price": r["price"], "realness": realness, "volume": vol})
                return levels

            # Suportes e Resistências 15m (Snapshot Atual)
            supports = get_scored_levels(last_rows, "bid_vol")
            resistances = get_scored_levels(last_rows, "ask_vol")

            # Níveis Multi-Timeframe
            now_ms = df_trades["timestamp"][-1]
            
            # 1h
            h1_book = sim_book.filter(pl.col("snapshot_time") >= (now_ms - 3600000)).group_by("price").agg([pl.col("bid_vol").sum(), pl.col("ask_vol").sum(), pl.col("trade_count").sum()])
            supports_1h = get_scored_levels(h1_book, "bid_vol")
            resistances_1h = get_scored_levels(h1_book, "ask_vol")

            # Pred Context (ex: 8h)
            context_ms = settings.SEQ_LEN * 15 * 60 * 1000
            pred_book = sim_book.filter(pl.col("snapshot_time") >= (now_ms - context_ms)).group_by("price").agg([pl.col("bid_vol").sum(), pl.col("ask_vol").sum(), pl.col("trade_count").sum()])
            supports_pred = get_scored_levels(pred_book, "bid_vol")
            resistances_pred = get_scored_levels(pred_book, "ask_vol")

            # 1d
            d1_book = sim_book.group_by("price").agg([pl.col("bid_vol").sum(), pl.col("ask_vol").sum(), pl.col("trade_count").sum()])
            supports_1d = get_scored_levels(d1_book, "bid_vol")
            resistances_1d = get_scored_levels(d1_book, "ask_vol")

            # --- CONSTRUÇÃO DO TENSOR (Com Truncamento Geométrico) ---
            rows = len(sim_book)
            remainder = rows % 128
            sim_book_tensor = sim_book[:rows - remainder] if remainder != 0 else sim_book

            if len(sim_book_tensor) < (32 * 128): return None

            logger.info(f"🏗️ Construindo Tensor 4D de {len(sim_book_tensor)} linhas...")
            tensor = build_tensor_4d(sim_book_tensor, n_levels=128, is_simulation=True)
            
            if tensor.shape[0] < 32: return None
            input_seq = tensor[-32:] 
            
            # 2. Inferência
            input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(self.device)
            logger.info("🧠 Executando Inferência...")
            
            with torch.no_grad():
                logits = self.model(input_tensor)
                probs = torch.softmax(logits, dim=1)
                conf, pred = torch.max(probs, 1)
            
            current_price = df_trades["price"][-1]
            last_ofi_total = last_rows["ofi_level"].sum()
            
            # Lógica de Tendência Granular
            if last_ofi_total > 0.1: trend = "Alta"
            elif last_ofi_total < -0.1: trend = "Baixa"
            else: trend = "Neutro"
            
            return {
                "signal": pred.item(),
                "confidence": conf.item(),
                "price": current_price,
                "ofi": last_ofi_total,
                "trend_intent": trend,
                "supports_15m": supports,
                "resistances_15m": resistances,
                "supports_1h": supports_1h,
                "resistances_1h": resistances_1h,
                "supports_pred": supports_pred,
                "resistances_pred": resistances_pred,
                "supports_1d": supports_1d,
                "resistances_1d": resistances_1d
            }
            
        except Exception as e:
            logger.error(f"❌ Erro na Inferência: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None