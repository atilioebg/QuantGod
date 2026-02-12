"""
Debug script para testar build_simulated_book isoladamente
"""
import polars as pl
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import settings
from src.processing.simulation import build_simulated_book

def test_simulation():
    print("=" * 80)
    print("🔬 DEBUG: TESTANDO build_simulated_book()")
    print("=" * 80)
    
    # Carregar apenas Nov/2025 (menor dataset)
    month = "2025-11"
    t_file = settings.RAW_HISTORICAL_DIR / f"aggTrades_{month}.parquet"
    
    print(f"\n📂 Carregando: {t_file.name}")
    df_trades = pl.read_parquet(t_file)
    
    print(f"   Total de trades: {df_trades.height:,}")
    print(f"   Colunas: {df_trades.columns}")
    print(f"   Memória: {df_trades.estimated_size() / 1e9:.2f} GB")
    
    # Normalizar timestamp
    if "transact_time" in df_trades.columns:
        print(f"\n⚙️ Normalizando timestamp...")
        df_trades = df_trades.with_columns(pl.col("transact_time").alias("timestamp"))
    
    # Testar simulação
    print(f"\n🔬 Iniciando build_simulated_book(window='15m')...")
    print(f"   (Se travar aqui, o problema está dentro da função)")
    
    try:
        # Adicionar timeout manual se possível
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Simulação travou por mais de 60 segundos!")
        
        # Set timeout (apenas Unix/Linux, não funciona no Windows)
        # signal.signal(signal.SIGALRM, timeout_handler)
        # signal.alarm(60)
        
        sim_book = build_simulated_book(df_trades, window="15m")
        
        # signal.alarm(0)  # Cancel timeout
        
        print(f"\n✅ SUCESSO!")
        print(f"   Snapshots criados: {sim_book.height:,}")
        print(f"   Colunas: {sim_book.columns}")
        print(f"   Primeiros 5 snapshots:")
        print(sim_book.head(5))
        
        return 0
        
    except TimeoutError as e:
        print(f"\n❌ TIMEOUT: {e}")
        print("   A função build_simulated_book() está travando!")
        return 1
        
    except Exception as e:
        print(f"\n❌ ERRO: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(test_simulation())
