"""
Script para Validar Frequência dos Candles (Klines)
"""
import polars as pl
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config import settings

def check_candle_frequency():
    print("=" * 80)
    print("🔍 VALIDAÇÃO DE FREQUÊNCIA DOS CANDLES")
    print("=" * 80)
    
    months = ["2025-11", "2025-12", "2026-01"]
    
    for month in months:
        print(f"\n📅 Analisando {month}...")
        
        k_file = settings.RAW_HISTORICAL_DIR / f"klines_{month}.parquet"
        
        if not k_file.exists():
            print(f"   ❌ Arquivo não encontrado: {k_file}")
            continue
        
        # Carregar klines
        df = pl.read_parquet(k_file)
        
        print(f"   Total de candles: {df.height:,}")
        
        # Normalizar timestamp se necessário
        if "open_time" in df.columns:
            df = df.with_columns(
                pl.from_epoch(pl.col("open_time"), time_unit="ms").alias("timestamp")
            )
        
        # Ordenar por timestamp
        df = df.sort("timestamp")
        
        # Calcular diferença entre timestamps consecutivos
        df = df.with_columns([
            pl.col("timestamp").diff().alias("time_diff")
        ])
        
        # Estatísticas da diferença
        time_diffs = df.filter(pl.col("time_diff").is_not_null())["time_diff"]
        
        if time_diffs.len() == 0:
            print("   ⚠️ Não foi possível calcular diferenças de tempo")
            continue
        
        # Converter para segundos
        time_diffs_seconds = time_diffs.dt.total_seconds()
        
        # Calcular estatísticas
        min_diff = time_diffs_seconds.min()
        max_diff = time_diffs_seconds.max()
        median_diff = time_diffs_seconds.median()
        mode_diff = time_diffs_seconds.mode().to_list()[0] if time_diffs_seconds.mode().len() > 0 else None
        
        print(f"\n   📊 Estatísticas de Intervalo:")
        print(f"      Mínimo: {min_diff:.0f}s ({min_diff/60:.1f}min)")
        print(f"      Mediana: {median_diff:.0f}s ({median_diff/60:.1f}min)")
        print(f"      Máximo: {max_diff:.0f}s ({max_diff/60:.1f}min)")
        if mode_diff:
            print(f"      Moda (mais comum): {mode_diff:.0f}s ({mode_diff/60:.1f}min)")
        
        # Determinar frequência predominante
        if mode_diff:
            if mode_diff == 60:
                freq = "1 minuto"
            elif mode_diff == 300:
                freq = "5 minutos"
            elif mode_diff == 900:
                freq = "15 minutos"
            elif mode_diff == 3600:
                freq = "1 hora"
            else:
                freq = f"{mode_diff/60:.1f} minutos"
            
            print(f"\n   ✅ Frequência Detectada: {freq}")
            
            # Calcular quantos candles em 6 horas
            candles_in_6h = int((6 * 3600) / mode_diff)
            print(f"   📏 Candles em 6 horas: {candles_in_6h}")
        
        # Mostrar primeiros timestamps
        print(f"\n   📋 Primeiros 5 timestamps:")
        print(df.select(["timestamp"]).head(5))

if __name__ == "__main__":
    check_candle_frequency()
