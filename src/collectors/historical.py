import asyncio
import io
import zipfile
import datetime
import aiohttp
import polars as pl
from src.config import settings
from src.utils.logger import setup_logger

logger = setup_logger("historical")

BINANCE_DATA_URL = "https://data.binance.vision/data/futures/um/monthly"

KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume", "count",
    "taker_buy_volume", "taker_buy_quote_volume", "ignore"
]

AGG_COLUMNS = [
    "agg_trade_id", "price", "quantity", "first_trade_id", 
    "last_trade_id", "transact_time", "is_buyer_maker"
]

async def download_file(session: aiohttp.ClientSession, url: str) -> bytes:
    try:
        async with session.get(url) as response:
            if response.status == 200:
                return await response.read()
            elif response.status == 404:
                logger.warning(f"Arquivo não encontrado (404): {url}")
                return None
            else:
                logger.error(f"Erro {response.status} ao baixar {url}")
                return None
    except Exception as e:
        logger.error(f"Erro de conexão ao baixar {url}: {e}")
        return None

def process_zip_data(zip_bytes: bytes, data_type: str) -> pl.DataFrame:
    """
    Versão Corrigida: Resolve o erro 'casting from Utf8View to Boolean'.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
            csv_filename = z.namelist()[0]
            with z.open(csv_filename) as f:
                # LÊ TUDO COMO TEXTO PRIMEIRO (Para robustez)
                if data_type == "klines":
                    df = pl.read_csv(
                        f.read(),
                        has_header=False,
                        new_columns=KLINE_COLUMNS,
                        infer_schema_length=0 
                    )
                    df = df.filter(pl.col("open_time").str.contains(r"^\d+$"))
                    
                    df = df.with_columns([
                        pl.col("open_time").cast(pl.UInt64),
                        pl.col("open").cast(pl.Float32),
                        pl.col("high").cast(pl.Float32),
                        pl.col("low").cast(pl.Float32),
                        pl.col("close").cast(pl.Float32),
                        pl.col("volume").cast(pl.Float32),
                        pl.col("close_time").cast(pl.UInt64),
                        pl.col("quote_asset_volume").cast(pl.Float32),
                        pl.col("count").cast(pl.UInt32),
                        pl.col("taker_buy_volume").cast(pl.Float32),
                        pl.col("taker_buy_quote_volume").cast(pl.Float32)
                    ]).drop("ignore")

                elif data_type == "aggTrades":
                    df = pl.read_csv(
                        f.read(),
                        has_header=False,
                        new_columns=AGG_COLUMNS,
                        infer_schema_length=0
                    )
                    df = df.filter(pl.col("agg_trade_id").str.contains(r"^\d+$"))
                    
                    # --- CORREÇÃO AQUI ---
                    # Em vez de .cast(pl.Boolean), usamos comparação direta
                    # A Binance usa "true" (minúsculo) ou "True". 
                    # .str.to_lowercase() garante que pegamos ambos.
                    is_buyer_maker_expr = pl.col("is_buyer_maker").str.to_lowercase() == "true"
                    
                    df = df.with_columns([
                        pl.col("agg_trade_id").cast(pl.UInt64),
                        pl.col("price").cast(pl.Float32),
                        pl.col("quantity").cast(pl.Float32),
                        pl.col("first_trade_id").cast(pl.UInt64),
                        pl.col("last_trade_id").cast(pl.UInt64),
                        pl.col("transact_time").cast(pl.UInt64),
                        is_buyer_maker_expr.alias("is_buyer_maker") # Substitui a coluna texto pela booleana
                    ])

                return df

    except Exception as e:
        logger.error(f"Erro ao processar CSV em memória: {e}")
        return None

async def download_and_process(
    sem: asyncio.Semaphore, 
    session: aiohttp.ClientSession, 
    date_str: str, 
    data_type: str
):
    async with sem:
        output_path = settings.RAW_HISTORICAL_DIR / f"{data_type}_{date_str}.parquet"
        
        if output_path.exists():
            logger.info(f"Arquivo já existe, pulando: {output_path}")
            return

        if data_type == "klines":
            filename = f"{settings.SYMBOL}-1m-{date_str}.zip"
            url = f"{BINANCE_DATA_URL}/klines/{settings.SYMBOL}/1m/{filename}"
        else:
            filename = f"{settings.SYMBOL}-aggTrades-{date_str}.zip"
            url = f"{BINANCE_DATA_URL}/aggTrades/{settings.SYMBOL}/{filename}"

        logger.info(f"Downloading {url}...")
        zip_bytes = await download_file(session, url)
        
        if not zip_bytes:
            return

        df = process_zip_data(zip_bytes, data_type)
        
        if df is not None:
            output_path = settings.RAW_HISTORICAL_DIR / f"{data_type}_{date_str}.parquet"
            df.write_parquet(output_path, compression="zstd")
            logger.info(f"Salvo: {output_path} ({df.shape[0]} linhas)")

async def main():
    logger.info(f"Iniciando download histórico para {settings.SYMBOL}...")
    
    today = datetime.date.today()
    start_date = datetime.datetime.strptime(settings.HISTORICAL_START_DATE, "%Y-%m-%d").date()
    
    dates = []
    current = start_date
    while current < datetime.date(today.year, today.month, 1):
        dates.append(current.strftime("%Y-%m"))
        if current.month == 12:
            current = datetime.date(current.year + 1, 1, 1)
        else:
            current = datetime.date(current.year, current.month + 1, 1)

    logger.info(f"Meses a processar: {dates}")

    sem = asyncio.Semaphore(3) 

    async with aiohttp.ClientSession() as session:
        tasks = []
        for date_str in dates:
            tasks.append(download_and_process(sem, session, date_str, "klines"))
            tasks.append(download_and_process(sem, session, date_str, "aggTrades"))
        
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    try:
        if hasattr(asyncio, 'WindowsSelectorEventLoopPolicy'):
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Download interrompido pelo usuário.")