
import asyncio
import json
import websockets
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
import sys
import logging

# Adicionar raiz do projeto ao path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import settings

# Configuração de Logs
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class L2LiveCollector:
    def __init__(self, symbol="BTCUSDT"):
        self.symbol = symbol
        self.ws_url = settings.BYBIT_WS_URL
        self.bids = {} # Price -> Size
        self.asks = {} # Price -> Size
        
        # Agregação
        self.current_minute = None
        self.ticks_in_minute = []
        self.prev_close = None 
        
        # Path
        settings.LIVE_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    def calculate_instant_features(self, ts):
        if not self.bids or not self.asks:
            return None
            
        # Ordenar e pegar top 5
        s_bids = sorted(self.bids.keys(), reverse=True)[:5]
        s_asks = sorted(self.asks.keys())[:5]
        
        if len(s_bids) < 1 or len(s_asks) < 1:
            return None
            
        # Micro Price
        b0_p, b0_s = s_bids[0], self.bids[s_bids[0]]
        a0_p, a0_s = s_asks[0], self.asks[s_asks[0]]
        micro_price = (b0_p * a0_s + a0_p * b0_s) / (b0_s + a0_s)
        
        # OBI L0
        obi_l0 = (b0_s - a0_s) / (b0_s + a0_s)
        
        # Deep OBI (Top 5)
        bid_vol_5 = sum(self.bids[p] for p in s_bids)
        ask_vol_5 = sum(self.asks[p] for p in s_asks)
        deep_obi_5 = (bid_vol_5 - ask_vol_5) / (bid_vol_5 + ask_vol_5)
        
        return {
            "ts": ts,
            "micro_price": micro_price,
            "spread": a0_p - b0_p,
            "obi_l0": obi_l0,
            "deep_obi_5": deep_obi_5,
            "bid_vol_5": bid_vol_5,
            "ask_vol_5": ask_vol_5
        }

    async def save_candle(self, candle_data):
        """Salva ou faz append no parquet de live inference"""
        df = pd.DataFrame([candle_data])
        
        # Se arquivo existe, ler e concatenar (usando Parquet append é complexo em filesystems simples, 
        # mais fácil manter um buffer ou sobrescrever um 'last_candle.parquet')
        path = settings.LIVE_DATA_PATH
        if path.exists():
            existing = pd.read_parquet(path)
            # Manter apenas as últimas 100 velas para evitar overhead de arquivo gigante em 'live'
            df = pd.concat([existing, df]).tail(100)
            
        df.to_parquet(path)
        logger.info(f"📊 Candle processado e salvo em {path}")

    def finalize_minute(self, minute_ts):
        if not self.ticks_in_minute:
            return
            
        df_m = pd.DataFrame(self.ticks_in_minute)
        
        # 9 Features Estacionárias
        # Micro-Price OHLC
        ohlc = {
            "open": df_m['micro_price'].iloc[0],
            "high": df_m['micro_price'].max(),
            "low": df_m['micro_price'].min(),
            "close": df_m['micro_price'].iloc[-1]
        }
        
        # Se não temos prev_close, o primeiro candle terá log_ret_close = 0 ou similar
        # Em produção, o ideal é carregar o último close do banco
        if self.prev_close is None:
            self.prev_close = ohlc['open']
            
        # Log Returns
        candle = {
            "datetime": minute_ts,
            "log_ret_open": np.log(ohlc['open'] / self.prev_close),
            "log_ret_high": np.log(ohlc['high'] / self.prev_close),
            "log_ret_low": np.log(ohlc['low'] / self.prev_close),
            "log_ret_close": np.log(ohlc['close'] / self.prev_close),
            "volatility": df_m['micro_price'].std(),
            "max_spread": df_m['spread'].max(),
            "mean_obi": df_m['obi_l0'].mean(),
            "mean_deep_obi": df_m['deep_obi_5'].mean(),
            "log_volume": np.log1p(len(df_m)),
            "close": ohlc['close'] # Próximo prev_close
        }
        
        self.prev_close = ohlc['close']
        self.ticks_in_minute = [] # Reset
        
        # Async save
        asyncio.create_task(self.save_candle(candle))

    async def handle_message(self, msg):
        obj = json.loads(msg)
        topic = obj.get("topic")
        msg_type = obj.get("type")
        data = obj.get("data", {})
        ts = obj.get("ts")
        
        if not topic or "orderbook" not in topic:
            return

        # 1. Atualizar Estado do Book
        if msg_type == "snapshot":
            self.bids = {float(p): float(s) for p, s in data.get("b", [])}
            self.asks = {float(p): float(s) for p, s in data.get("a", [])}
            logger.info(f"🔄 Snapshot do Book recebido para {self.symbol}")
        elif msg_type == "delta":
            for p, s in data.get("b", []):
                price, size = float(p), float(s)
                if size == 0: self.bids.pop(price, None)
                else: self.bids[price] = size
            for p, s in data.get("a", []):
                price, size = float(p), float(s)
                if size == 0: self.asks.pop(price, None)
                else: self.asks[price] = size

        # 2. Extrair Features Instantâneas
        features = self.calculate_instant_features(ts)
        if not features:
            return

        # 3. Agregação por Minuto
        dt = datetime.fromtimestamp(ts/1000, tz=timezone.utc)
        current_minute_ts = dt.replace(second=0, microsecond=0)
        
        if self.current_minute is None:
            self.current_minute = current_minute_ts
            
        if current_minute_ts > self.current_minute:
            # Fechar minuto anterior
            self.finalize_minute(self.current_minute)
            self.current_minute = current_minute_ts
            
        self.ticks_in_minute.append(features)

    async def start(self):
        logger.info(f"📡 Conectando ao WebSocket Bybit: {self.ws_url}")
        
        while True:
            try:
                async with websockets.connect(self.ws_url) as ws:
                    # Subscrever
                    sub_msg = {
                        "op": "subscribe",
                        "args": [f"orderbook.50.{self.symbol}"]
                    }
                    await ws.send(json.dumps(sub_msg))
                    logger.info(f"✅ Inscrito no canal Orderbook 50 para {self.symbol}")
                    
                    async for message in ws:
                        await self.handle_message(message)
                        
            except Exception as e:
                logger.error(f"❌ Erro de conexão ou processamento: {e}. Reconectando em 5s...")
                await asyncio.sleep(5)

if __name__ == "__main__":
    collector = L2LiveCollector(symbol="BTCUSDT")
    try:
        asyncio.run(collector.start())
    except KeyboardInterrupt:
        logger.info("🛑 Coleta encerrada pelo usuário.")
