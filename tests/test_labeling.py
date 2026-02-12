"""
Script de teste isolado para validar o labeling
"""
import polars as pl
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import settings
from src.processing.labeling import generate_labels

def test_labeling():
    print("=" * 80)
    print("🧪 TESTE ISOLADO DO LABELING")
    print("=" * 80)
    
    # Carregar klines de Nov/2025
    k_file = settings.RAW_HISTORICAL_DIR / "klines_2025-11.parquet"
    
    print(f"\n📂 Carregando: {k_file.name}")
    df_klines = pl.read_parquet(k_file)
    
    print(f"   Total de candles: {df_klines.height:,}")
    
    # Normalizar timestamp
    if "open_time" in df_klines.columns:
        df_klines = df_klines.with_columns(
            pl.from_epoch(pl.col("open_time"), time_unit="ms").alias("timestamp")
        )
    
    print(f"\n⚙️ Gerando labels (window=6h, target=0.8%, stop=0.4%)...")
    
    try:
        df_labels = generate_labels(df_klines, window_hours=6, target_pct=0.008, stop_pct=0.004)
        print(f"   ✅ Labels gerados com sucesso!")
        print(f"   Total: {df_labels.height:,}")
        
        # Contar distribuição
        label_counts = df_labels.group_by("label").agg(pl.count()).sort("label")
        
        total = df_labels.height
        print(f"\n📊 Distribuição:")
        
        class_names = {0: "Neutro", 1: "Venda/Stop", 2: "Compra/Alvo"}
        
        for row in label_counts.iter_rows(named=True):
            label = row["label"]
            count = row["count"]
            pct = 100 * count / total
            class_name = class_names.get(label, f"Classe {label}")
            emoji = "⚪" if label == 0 else ("🔴" if label == 1 else "🟢")
            print(f"   {emoji} {class_name}: {count:,} ({pct:.2f}%)")
        
        print(f"\n✅ TESTE PASSOU!")
        
    except Exception as e:
        print(f"\n❌ ERRO: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(test_labeling())
