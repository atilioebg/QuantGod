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
from src.models.vivit import QuantGodViViT

from datetime import datetime
import pandas as pd

logger = logging.getLogger("QuantGod.SniperBrain")

class PredictionValidator:
    def __init__(self, log_path="data/prediction_log.csv"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(exist_ok=True, parents=True)
        if not self.log_path.exists():
            df = pd.DataFrame(columns=[
                "timestamp", "price", "signal", "confidence", 
                "ofi", "verdict", "strategy", "result", "pl_est"
            ])
            df.to_csv(self.log_path, index=False, encoding='utf-8-sig')
        self.last_heartbeat = datetime.now()
        
        # --- ESTADOS DE VALIDAÇÃO (Debounce & Cooldown) ---
        self.last_signal = None
        self.consecutive_count = 0
        self.last_logged_signal = None
        self.last_log_timestamp = datetime(2000, 1, 1) # Passado distante

    def register_prediction(self, price, signal, confidence, ofi, verdict, strategy):
        try:
            now = datetime.now()
            
            # 1. Filtro de Tipo de Sinal
            strat_upper = str(strategy).upper()
            is_neutral = "AGUARDAR" in strat_upper or "NEUTRO" in strat_upper
            
            # --- LÓGICA DE HEARTBEAT (Prova de Vida) ---
            # Se já passou 4h, forçamos um log independente de qualquer trava
            elapsed_heartbeat = (now - self.last_heartbeat).total_seconds() / 3600
            if elapsed_heartbeat >= 4.0:
                self._write_log(now, price, signal, confidence, ofi, verdict, "💓 HEARTBEAT", "⚪ (STATUS)")
                self.last_heartbeat = now
                return True

            # --- LÓGICA DE CONFIRMAÇÃO (Debounce - 3 Ciclos) ---
            # Aplicamos a confirmação de 3 ciclos para QUALQUER sinal (incluindo Neutro)
            # para garantir que a mudança de estado é estável.
            if signal == self.last_signal:
                self.consecutive_count += 1
            else:
                self.last_signal = signal
                self.consecutive_count = 1
                
            if self.consecutive_count < 3:
                return False # Aguardando o sinal estabilizar

            # --- LÓGICA DE MUDANÇA DE ESTADO / COOLDOWN ---
            # Estados: 0 (Neutro), 1 (Venda), 2 (Compra)
            elapsed_log = (now - self.last_log_timestamp).total_seconds()
            
            # Mudança de Estado: O sinal atual estabilizado é diferente do último que logamos?
            is_state_change = (signal != self.last_logged_signal)
            
            # Regras para Gravar:
            # A) Mudou o estado (ex: de Compra para Neutro, ou Neutro para Venda) - GRAVA NA HORA
            # B) O estado é o mesmo, mas passou o Cooldown de 14m30s (Manutenção de Perspectiva) - GRAVA
            # C) Se for NEUTRO, só gravamos a mudança inicial (não repetimos a cada 14m30s para não poluir)
            
            should_log = False
            log_strategy = strategy
            log_result = "⏳ (PENDING)"

            if is_state_change:
                should_log = True
                if is_neutral:
                    log_result = "⚪ (NEUTRAL)"
            elif not is_neutral and elapsed_log >= 870:
                # Se ainda está em sinal ativo (Buy/Sell), renova o log a cada 14:30
                should_log = True
                log_strategy = f"🔄 PERSIST: {strategy}"

            if should_log:
                self._write_log(now, price, signal, confidence, ofi, verdict, log_strategy, log_result)
                self.last_log_timestamp = now
                self.last_logged_signal = signal
                self.last_heartbeat = now # Reset heartbeat ao ter atividade de log
                return True
                
            return False
        except Exception as e:
            logger.error(f"❌ Falha ao registrar predição: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _write_log(self, dt, price, signal, confidence, ofi, verdict, strategy, result):
        """Auxiliar de escrita atômica no CSV."""
        new_row = {
            "timestamp": dt.strftime("%Y-%m-%d %H:%M:%S"),
            "price": price,
            "signal": signal,
            "confidence": confidence,
            "ofi": ofi,
            "verdict": verdict,
            "strategy": strategy,
            "result": result,
            "pl_est": 0.0
        }
        df = pd.DataFrame([new_row])
        df.to_csv(self.log_path, mode='a', header=False, index=False, encoding='utf-8-sig')

    def process_pending_outcomes(self, current_price, current_time=None):
        """Atualiza o resultado (P&L) de ordens pendentes após 15 minutos."""
        if not self.log_path.exists(): return
        
        try:
            df = pd.read_csv(self.log_path, encoding='utf-8-sig')
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
                df.to_csv(self.log_path, index=False, encoding='utf-8-sig')
        except Exception as e:
            logger.error(f"Erro ao processar P&L: {e}")

    def get_log(self):
        if self.log_path.exists():
            try:
                return pd.read_csv(self.log_path, encoding='utf-8-sig').sort_values("timestamp", ascending=False)
            except:
                return pd.read_csv(self.log_path).sort_values("timestamp", ascending=False)
        return pd.DataFrame()

class SniperBrain:
    def __init__(self, model_path="data/quantgod_best_model.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = QuantGodViViT(seq_len=32, input_channels=4, price_levels=128, num_classes=3)
        
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