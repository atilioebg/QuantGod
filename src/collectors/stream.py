import asyncio
import websockets
import json
import polars as pl
import signal
from datetime import datetime
from src.config import settings
from src.utils.logger import logger
from typing import List, Dict

class DataStreamer:
    def __init__(self, symbol: str = settings.SYMBOL):
        self.symbol = symbol.lower()
        self.ws_url = f"{settings.WS_URL}/{self.symbol}@depth20@100ms/{self.symbol}@aggTrade"
        self.buffer_depth: List[Dict] = []
        self.buffer_trades: List[Dict] = []
        self.running = False
        self.last_flush = datetime.now()
        
        # Stats
        self.total_msgs = 0

    async def connect(self):
        """Main connection loop with exponential backoff."""
        backoff = 1
        while True:
            try:
                logger.info(f"Connecting to {self.ws_url}...")
                async with websockets.connect(self.ws_url) as ws:
                    logger.info("Connected to WebSocket API.")
                    self.running = True
                    backoff = 1  # Reset backoff on successful connection
                    
                    await self.listen(ws)
            except (websockets.ConnectionClosed, Exception) as e:
                logger.error(f"Connection lost: {e}. Retrying in {backoff}s...")
                self.running = False
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60) # Max 60s wait

    async def listen(self, ws):
        """Listens for messages and handles buffering/flushing."""
        try:
            while self.running:
                msg = await ws.recv()
                data = json.loads(msg)
                self.process_message(data)
                self.total_msgs += 1
                
                # Check flush conditions
                await self.check_flush()
                
        except asyncio.CancelledError:
            logger.info("Listen task cancelled.")
            raise

    def process_message(self, data: dict):
        """Routes message to appropriate buffer based on event type."""
        # Handle Multiplexed Stream format ({"stream":"...", "data":{...}})
        if "stream" in data and "data" in data:
            data = data["data"]

        event_type = data.get("e")
        
        # Inject receipt timestamp
        data["_received_at"] = datetime.now().timestamp()
        
        if event_type == "depthUpdate":
            # For @depth20 stream, payload is direct, event type might be missing or different in raw stream?
            # Actually @depth20@100ms payload structure:
            # { "lastUpdateId": ..., "bids": [...], "asks": [...] }
            # It DOES NOT usually have "e": "depthUpdate" in the partial book stream.
            # Let's inspect payload structure. 
            pass 
        
        # Check for specific stream types based on payload keys
        if event_type == "aggTrade":
            self.buffer_trades.append(data)
        elif event_type == "depthUpdate" or ("b" in data and "a" in data):
            # Depth Snapshot/Update (Futures uses 'b' and 'a' keys)
            # Normalize to bids/asks if needed or keep as is. 
            # The simulator might expect bids/asks though.
            if "b" in data and "bids" not in data:
                data["bids"] = data.pop("b")
            if "a" in data and "asks" not in data:
                data["asks"] = data.pop("a")
            self.buffer_depth.append(data)

    async def check_flush(self):
        """Checks if buffer needs flushing based on time or size."""
        now = datetime.now()
        time_diff = (now - self.last_flush).total_seconds()
        
        # Simple size check (approximate)
        depth_len = len(self.buffer_depth)
        trades_len = len(self.buffer_trades)
        
        # Flush if interval passed OR buffer has significant items (> 10k items approx 5-10MB)
        if time_diff >= settings.STREAM_FLUSH_INTERVAL_SECONDS or (depth_len + trades_len) >= 20000:
            await self.flush_buffer()

    async def flush_buffer(self):
        """Writes in-memory buffers to Parquet."""
        if not self.buffer_depth and not self.buffer_trades:
            return

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(f"Flushing buffers... Depth: {len(self.buffer_depth)}, Trades: {len(self.buffer_trades)}")

        # Save Depth
        if self.buffer_depth:
            try:
                # Convert list of dicts to Polars
                # flatten bids/asks? For raw storage, keeping them as structured lists/structs is okay
                # But Polars handles list<list> well.
                df_depth = pl.DataFrame(self.buffer_depth)
                
                # Path: data/raw/stream/depth/YYYY-MM-DD/
                subdir = settings.RAW_STREAM_DIR / "depth" / datetime.now().strftime("%Y-%m-%d")
                subdir.mkdir(parents=True, exist_ok=True)
                
                file_path = subdir / f"depth_{timestamp_str}.parquet"
                df_depth.write_parquet(file_path, compression="zstd")
                self.buffer_depth.clear()
            except Exception as e:
                logger.error(f"Error flushing depth buffer: {e}")

        # Save Trades
        if self.buffer_trades:
            try:
                df_trades = pl.DataFrame(self.buffer_trades)
                
                subdir = settings.RAW_STREAM_DIR / "trades" / datetime.now().strftime("%Y-%m-%d")
                subdir.mkdir(parents=True, exist_ok=True)
                
                file_path = subdir / f"trades_{timestamp_str}.parquet"
                df_trades.write_parquet(file_path, compression="zstd")
                self.buffer_trades.clear()
            except Exception as e:
                logger.error(f"Error flushing trades buffer: {e}")

        self.last_flush = datetime.now()

    def handle_shutdown(self, signum, frame):
        """Graceful shutdown handler."""
        logger.info("Shutdown signal received. Flushing buffers...")
        # Since this is a signal handler, we can't await directly.
        # We set running to False and let the loop exit, or trigger a final blocking flush if needed.
        # Ideally, we rely on the main loop catching the exit.
        self.running = False
        # Running sync flush here might be risky if async loop is active, but we need to ensure data is saved.
        # We will dispatch a task to flush.
        asyncio.create_task(self.flush_buffer())

async def main():
    streamer = DataStreamer()
    
    # Register signals logic is tricky in asyncio.run, better done in loop.
    # However, running python script usually handles SIGINT via KeyboardInterrupt
    
    try:
        await streamer.connect()
    except asyncio.CancelledError:
        await streamer.flush_buffer()
    except KeyboardInterrupt:
        logger.info("Keyboard Interrupt. Stopping...")
    finally:
        await streamer.flush_buffer()
        logger.info("Exiting.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # Catch upper level interrupt
        pass