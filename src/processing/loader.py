import polars as pl
from pathlib import Path
from src.config import settings

def load_stream_data(date_str: str = None) -> pl.DataFrame:
    """Carrega dados de stream (depth) de um dia específico."""
    path = settings.RAW_STREAM_DIR / "depth"
    # Se particionamos por data em 'depth/YYYY-MM-DD', precisamos buscar lá.
    # O coletor stream.py salva em: RAW_STREAM_DIR / "depth" / "YYYY-MM-DD"
    
    try:
        if date_str:
            target_path = path / date_str
            files = list(target_path.rglob("*.parquet"))
        else:
            files = list(path.rglob("*.parquet"))
            
        if not files:
            return pl.DataFrame()
            
        return pl.read_parquet(files) # Para MVP lê tudo, produção usar scan_parquet
    except Exception as e:
        print(f"Erro ao carregar stream depth: {e}")
        return pl.DataFrame()

def load_trade_data() -> pl.LazyFrame:
    """Carrega dados de trades (aggTrade) históricos e de stream unificados."""
    # O stream.py salva em RAW_STREAM_DIR / "trades" / ...
    path = settings.RAW_STREAM_DIR / "trades"
    
    # Verifica se path existe
    if not path.exists():
         return pl.DataFrame().lazy()

    # Scan parquet recursivo
    try:
        return pl.scan_parquet(path.joinpath("**/*.parquet"))
    except Exception as e:
        print(f"Erro ao carregar stream trades: {e}")
        return pl.DataFrame().lazy()
