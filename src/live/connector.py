import time
import logging
import threading
from collections import deque
import polars as pl
from binance.websocket.spot.websocket_stream import SpotWebsocketStreamClient
from binance.spot import Spot

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BinanceConnector:
    def __init__(self, symbol="BTCUSDT", buffer_size=5000000): # Buffer aumentado para 5M (7 dias)
        self.symbol = symbol.upper()
        self.trade_buffer = deque(maxlen=buffer_size)
        self.rest_client = Spot()
        self.ws_client = None
        self.lock = threading.Lock()

    def _on_message(self, _, message):
        try:
            if 'e' in message and message['e'] == 'aggTrade':
                trade = {
                    'price': float(message['p']),
                    'quantity': float(message['q']),
                    'timestamp': int(message['T']),
                    'is_buyer_maker': bool(message['m'])
                }
                with self.lock: self.trade_buffer.append(trade)
        except Exception: pass

    def start(self):
        self.ws_client = SpotWebsocketStreamClient(on_message=self._on_message)
        self.ws_client.agg_trade(symbol=self.symbol.lower())

    def stop(self):
        """Para a conexão websocket e libera recursos."""
        if self.ws_client:
            try:
                self.ws_client.stop()
                logger.info("🔌 Conexão Websocket encerrada com sucesso.")
            except Exception as e:
                logger.error(f"Erro ao encerrar Websocket: {e}")
            self.ws_client = None

    def get_data(self, minutes=500):
        now = int(time.time() * 1000)
        cutoff = now - (minutes * 60 * 1000)
        with self.lock: data = list(self.trade_buffer)
        
        if not data: return None
        
        # Cria DataFrame Polars
        df = pl.DataFrame(data)
        
        # Filtra pelo tempo e ordena
        df = df.filter(pl.col("timestamp") >= cutoff).sort("timestamp")
        
        return df if df.height > 0 else None

    def warm_up(self, lookback_minutes=10080): # 7 dias
        logger.info(f"🔥 Iniciando Hybrid Warm-up (7d Estrutura + 12h Fluxo)...")
        end_time = int(time.time() * 1000)
        
        # 1. PARTE MACRO: Buscar 7 dias em velas de 1h (Deep Scan)
        # Isso popula os pivots históricos sem baixar milhões de trades
        logger.info("📍 Sincronizando Estrutura Macro (7 dias)...")
        try:
            klines = self.rest_client.klines(
                symbol=self.symbol,
                interval="1h",
                limit=168 # 7 dias * 24h
            )
            if klines:
                with self.lock:
                    for k in klines:
                        # Injetamos 4 "trades" por vela para marcar as zonas de preço
                        ts = int(k[0])
                        vols = float(k[5]) / 4
                        self.trade_buffer.append({'price': float(k[1]), 'quantity': vols, 'timestamp': ts, 'is_buyer_maker': True}) # Open
                        self.trade_buffer.append({'price': float(k[2]), 'quantity': vols, 'timestamp': ts, 'is_buyer_maker': False}) # High
                        self.trade_buffer.append({'price': float(k[3]), 'quantity': vols, 'timestamp': ts, 'is_buyer_maker': True}) # Low
                        self.trade_buffer.append({'price': float(k[4]), 'quantity': vols, 'timestamp': ts, 'is_buyer_maker': False}) # Close
            logger.info(f"✅ Estrutura Macro carregada: {len(klines)} pivots injetados.")
        except Exception as e:
            logger.error(f"Erro no Warm-up Macro: {e}")

        # 2. PARTE MICRO: Buscar últimas 12h em trades reais (Alta Fidelidade)
        logger.info("⚡ Sincronizando Fluxo Intraday (12 horas)...")
        lookback_micro = 720 # 12 horas
        start_time_micro = end_time - (lookback_micro * 60 * 1000)
        curr = start_time_micro
        step_ms = 5 * 60 * 1000 # 5 min
        
        count = 0
        total_steps = lookback_micro // 5
        step_idx = 0
        
        while curr < end_time:
            next_end = min(curr + step_ms, end_time)
            try:
                trades = self.rest_client.agg_trades(symbol=self.symbol, startTime=curr, endTime=next_end, limit=1000)
                if trades:
                    with self.lock:
                        for t in trades:
                            self.trade_buffer.append({
                                'price': float(t['p']),
                                'quantity': float(t['q']),
                                'timestamp': int(t['T']),
                                'is_buyer_maker': bool(t['m'])
                            })
                    count += len(trades)
                
                step_idx += 1
                if step_idx % 20 == 0:
                    logger.info(f"📈 Progresso Fluxo: {int((step_idx/total_steps)*100)}% ({count} trades)")
                
                curr = next_end
                time.sleep(0.05) 
            except Exception as e:
                logger.error(f"Erro no Warm-up Micro: {e}")
                break
                
        logger.info(f"🔥 Warm-up Concluído! Buffer: {len(self.trade_buffer)} eventos em memória.")