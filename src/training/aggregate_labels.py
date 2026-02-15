import numpy as np
from pathlib import Path
from src.config import settings
from tqdm import tqdm

def aggregate_final_balance():
    tensor_dir = settings.PROCESSED_DIR / "tensors"
    files = list(tensor_dir.glob("processed_*.npz"))
    
    total_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    total_samples = 0
    
    print(f"Lendo {len(files)} arquivos para calcular balanceamento final...")
    
    for f in tqdm(files):
        try:
            data = np.load(f)
            y = data['y']
            unique, counts = np.unique(y, return_counts=True)
            for u, c in zip(unique, counts):
                total_counts[u] += c
            total_samples += len(y)
        except Exception as e:
            print(f"Erro ao ler {f.name}: {e}")
            
    class_names = {0: "NEUTRO", 1: "STOP", 2: "LONG", 3: "SUPER LONG"}
    
    print("\n" + "=" * 60)
    print("BALANCEAMENTO FINAL ACUMULADO (2020-2026)")
    print("=" * 60)
    print(f"{'CLASSE':<15} | {'QUANTIDADE':<15} | {'PERCENTUAL'}")
    print("-" * 60)
    
    for i in range(4):
        count = total_counts[i]
        pct = (count / total_samples * 100) if total_samples > 0 else 0
        print(f"{class_names[i]:<15} | {count:>14,} | {pct:>10.2f}%")
        
    print("-" * 60)
    print(f"{'TOTAL':<15} | {total_samples:>14,} | 100.00%")
    print("=" * 60)

if __name__ == "__main__":
    aggregate_final_balance()
