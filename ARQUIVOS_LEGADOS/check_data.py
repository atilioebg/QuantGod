import polars as pl
from pathlib import Path
import sys

def inspect_parquet(path_str):
    path = Path(path_str)
    
    if not path.exists():
        print(f"⚠️ Diretório não encontrado: {path_str}")
        return

    # Recursive search for parquet files
    files = list(path.rglob("*.parquet"))
    
    if not files:
        print(f"⚠️ Nenhum arquivo encontrado em {path_str} ainda.")
        return

    # Pega o arquivo mais recente
    latest_file = max(files, key=lambda f: f.stat().st_mtime)
    print(f"🔬 Inspecionando: {latest_file.name}")
    print(f"📂 Caminho: {latest_file}")
    
    try:
        # Lê o arquivo com Polars
        df = pl.read_parquet(latest_file)
        print(f"📏 Dimensões: {df.shape}")
        print(f"📋 Colunas: {df.columns}")
        print("\n🔎 Amostra (Head):")
        print(df.head())
        print("-" * 50)
    except Exception as e:
        print(f"❌ Erro ao ler arquivo: {e}")

if __name__ == "__main__":
    print("========================================")
    print("      DEEPSWING DATA CHECKUP 🩺      ")
    print("========================================")
    
    print("\n--- 🚜 DADOS HISTÓRICOS ---")
    inspect_parquet("data/raw/historical")
    
    print("\n--- 🔴 DADOS DE STREAM ---")
    inspect_parquet("data/raw/stream")
