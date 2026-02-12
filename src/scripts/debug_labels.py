"""
Script de Debug para Investigar Distribuição de Labels
"""
import polars as pl
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.config import settings
from src.processing.labeling import generate_labels

def debug_labels():
    print("=" * 80)
    print("🔬 DEBUG: ANÁLISE DE LABELS BRUTOS")
    print("=" * 80)
    
    months = ["2025-11", "2025-12", "2026-01"]
    
    for month in months:
        print(f"\n📅 Analisando {month}...")
        
        k_file = settings.RAW_HISTORICAL_DIR / f"klines_{month}.parquet"
        
        if not k_file.exists():
            print(f"   ❌ Arquivo não encontrado: {k_file}")
            continue
        
        # Carregar klines
        df_klines = pl.read_parquet(k_file)
        
        # Normalizar timestamp
        if "open_time" in df_klines.columns:
            df_klines = df_klines.with_columns(
                pl.from_epoch(pl.col("open_time"), time_unit="ms").alias("timestamp")
            )
        
        print(f"   Total de candles: {df_klines.height:,}")
        
        # Gerar labels
        df_labels = generate_labels(df_klines, window_hours=6, target_pct=0.008, stop_pct=0.004)
        
        # Contar distribuição
        label_counts = df_labels.group_by("label").agg(pl.len()).sort("label")
        
        total = df_labels.height
        print(f"   Total de labels: {total:,}")
        
        class_names = {0: "Neutro", 1: "Venda/Stop", 2: "Compra/Alvo"}
        
        for row in label_counts.iter_rows(named=True):
            label = row["label"]
            count = row["len"]
            pct = 100 * count / total
            class_name = class_names.get(label, f"Classe {label}")
            emoji = "⚪" if label == 0 else ("🔴" if label == 1 else "🟢")
            print(f"   {emoji} {class_name}: {count:,} ({pct:.2f}%)")
        
        # Verificar se há NaNs ou nulls
        null_count = df_labels.filter(pl.col("label").is_null()).height
        if null_count > 0:
            print(f"   ⚠️ Labels nulos: {null_count}")
        
        # Amostra de labels
        print(f"\n   📋 Primeiros 10 labels:")
        print(df_labels.head(10).select(["timestamp", "label", "close_price"]))

if __name__ == "__main__":
    debug_labels()
