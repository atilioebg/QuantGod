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
    def __init__(self, symbol="BTCUSDT", buffer_size=500000): # Buffer aumentado
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

    def warm_up(self, lookback_minutes=480):
        logger.info(f"🔥 Iniciando Warm-up denso ({lookback_minutes} min)...")
        end_time = int(time.time() * 1000)
        start_time = end_time - (lookback_minutes * 60 * 1000)
        
        curr = start_time
        count = 0
        
        # Loop mais agressivo: Pula de 5 em 5 minutos para garantir preenchimento
        step_ms = 5 * 60 * 1000 
        
        while curr < end_time:
            next_end = min(curr + step_ms, end_time)
            
            try:
                trades = self.rest_client.agg_trades(
                    symbol=self.symbol, 
                    startTime=curr, 
                    endTime=next_end, 
                    limit=1000
                )
                
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
                
                curr = next_end
                # Pequeno delay para não estourar limite da API (1200 reqs/min)
                time.sleep(0.1) 
                
            except Exception as e:
                logger.error(f"Erro no Warm-up: {e}")
                break
                
        logger.info(f"🔥 Warm-up concluído. Buffer: {len(self.trade_buffer)} trades. Sistema pronto!")