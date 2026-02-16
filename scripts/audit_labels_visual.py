import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import random

# Config
DATA_PATH = Path("data/processed/l2_features_1min.parquet")
PLOT_DIR = Path("data/plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

def analyze_and_plot():
    print("============================================================")
    print("ANÁLISE DE LABELS E VISUALIZAÇÃO - QUANT GOD")
    print("============================================================")
    
    if not DATA_PATH.exists():
        print(f"[ERRO] Arquivo {DATA_PATH} não encontrado.")
        return

    df = pd.read_parquet(DATA_PATH)
    
    # 1. Estatísticas
    target_col = 'target_class'
    counts = df[target_col].value_counts().sort_index()
    percentages = df[target_col].value_counts(normalize=True).sort_index() * 100
    
    print("\n------------------------------------------------------------")
    print("DISTRIBUIÇÃO DE CLASSES")
    print("------------------------------------------------------------")
    for cls in counts.index:
        print(f"Classe {cls}: {counts[cls]:5d} ({percentages[cls]:6.2f}%)")
        
    print(f"\nTotal: {len(df)}")
    
    # 2. Plotting (2 samples per class)
    print("\n------------------------------------------------------------")
    print("GERANDO PLOTS DE CONFERÊNCIA...")
    print("------------------------------------------------------------")
    
    window_size = 15 # 15 min antes e depois
    
    for cls in [0, 1, 2]:
        # Filtrar índices da classe
        indices = df[df[target_col] == cls].index
        
        # Selecionar 2 aleatórios (com margem para janela)
        possible_indices = [idx for idx in indices if idx > df.index[window_size] and idx < df.index[-window_size-5]] # -5 for future lookahead
        
        if len(possible_indices) < 2:
            print(f"[AVISO] Classe {cls} tem poucos exemplos para plotar.")
            samples = possible_indices
        else:
            samples = random.sample(possible_indices, 2)
            
        for i, idx in enumerate(samples):
            # Recortar janela
            start_loc = df.index.get_loc(idx) - window_size
            end_loc = df.index.get_loc(idx) + window_size + 5 # +5 para ver o futuro
            
            subset = df.iloc[start_loc:end_loc]
            
            # Plot
            plt.figure(figsize=(10, 6))
            plt.plot(subset.index, subset['close'], label='Micro-Price', color='blue', alpha=0.7)
            
            # Marcar o ponto analisado
            point = df.loc[idx]
            plt.scatter(idx, point['close'], color='red', s=100, zorder=5, label=f'Target={cls}')
            
            # Annotate Return
            ret = point['future_ret_5m']
            plt.title(f"Class {cls} | Return 5m: {ret:.6f} | Time: {idx}")
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Save
            filename = PLOT_DIR / f"label_audit_class_{cls}_sample_{i+1}.png"
            plt.savefig(filename)
            plt.close()
            print(f"[PLOT] Salvo: {filename}")

    print("\n[SUCESSO] Análise Concluída.")

if __name__ == "__main__":
    analyze_and_plot()
