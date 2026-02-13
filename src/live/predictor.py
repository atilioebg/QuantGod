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

from datetime import datetime
import pandas as pd

logger = logging.getLogger("SAIMP.SniperBrain")

class PredictionValidator:
    def __init__(self, log_path="data/prediction_log.csv"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(exist_ok=True, parents=True)
        if not self.log_path.exists():
            df = pd.DataFrame(columns=[
                "timestamp", "price", "signal", "confidence", 
                "ofi", "verdict", "strategy", "result", "pl_est"
            ])
            df.to_csv(self.log_path, index=False)
        self.last_heartbeat = datetime.now()

    def register_prediction(self, price, signal, confidence, ofi, verdict, strategy):
        try:
            now = datetime.now()
            elapsed_heartbeat = (now - self.last_heartbeat).total_seconds() / 3600 # horas
            
            # 1. Filtro de Gatilho (Trigger Logic)
            strat_upper = str(strategy).upper()
            is_neutral = "AGUARDAR" in strat_upper or "NEUTRO" in strat_upper
            
            # Se for neutro MAS passou 4h, forçamos o log como Heartbeat
            is_heartbeat = False
            if is_neutral and elapsed_heartbeat >= 4.0:
                is_heartbeat = True
                strategy = "💓 HEARTBEAT"
                self.last_heartbeat = now
            elif is_neutral:
                # Neutro comum: ignora
                return False

            # Lógica de Resultado Visual
            res_val = "⏳ (PENDING)" if not is_heartbeat else "⚪ (STATUS)"

            new_row = {
                "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
                "price": price,
                "signal": signal,
                "confidence": confidence,
                "ofi": ofi,
                "verdict": verdict,
                "strategy": strategy,
                "result": res_val,
                "pl_est": 0.0
            }
            
            # Append atomicamente
            df = pd.DataFrame([new_row])
            df.to_csv(self.log_path, mode='a', header=False, index=False)
            
            if not is_heartbeat:
                # Se foi um sinal real (não heartbeat), resetamos o timer do heartbeat também
                self.last_heartbeat = now
                
            return True
        except Exception as e:
            logger.error(f"❌ Falha ao registrar predição: {e}")
            return False

    def process_pending_outcomes(self, current_price, current_time=None):
        """Atualiza o resultado (P&L) de ordens pendentes após 15 minutos."""
        if not self.log_path.exists(): return
        
        try:
            df = pd.read_csv(self.log_path)
            if df.empty: return
            
            if "pl_est" not in df.columns: df["pl_est"] = 0.0 # Migração
            
            updated = False
            now = current_time if current_time else datetime.now()
            
            for idx, row in df.iterrows():
                if "PENDING" in str(row["result"]):
                    try:
                        entry_time = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
                        elapsed = (now - entry_time).total_seconds() / 60 # minutos
                        
                        if elapsed >= 15: # 1 Candle Completo
                            entry_price = float(row["price"])
                            signal = str(row["signal"]).upper()
                            
                            # Cálculo P&L
                            if "COMPRA" in signal:
                                pl_pct = ((current_price - entry_price) / entry_price) * 100
                            elif "VENDA" in signal:
                                pl_pct = ((entry_price - current_price) / entry_price) * 100
                            else: pl_pct = 0.0
                            
                            # Veredito
                            outcome = "✅ WIN" if pl_pct > 0 else "❌ LOSS"
                            
                            df.at[idx, "result"] = outcome
                            df.at[idx, "pl_est"] = round(pl_pct, 2)
                            updated = True
                    except Exception: pass
            
            if updated:
                df.to_csv(self.log_path, index=False)
        except Exception as e:
            logger.error(f"Erro ao processar P&L: {e}")

    def get_log(self):
        if self.log_path.exists():
            return pd.read_csv(self.log_path).sort_values("timestamp", ascending=False)
        return pd.DataFrame()

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
            
            # DEBUG CRÍTICO OFI ZERO
            logger.info(f"📊 DEBUG OFI: SnapshotTime={last_snap_time} | Rows={last_rows.height} | OFI_Total={last_ofi_total:.4f}")
            if last_ofi_total == 0.0:
                logger.warning("⚠️ ALERTA: OFI Zerado identificado! Verifique ingestão de trades.")
            
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