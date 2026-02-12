import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from torch.utils.data import DataLoader
import polars as pl
from datetime import datetime

# Adiciona a raiz do projeto ao path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

# Importa Dataset
try:
    from src.training.streaming_dataset import StreamingDataset
    print("✅ StreamingDataset importado com sucesso!")
except ImportError as e:
    print(f"❌ Erro ao importar StreamingDataset: {e}")
    sys.path.append(str(Path.cwd()))
    from src.training.streaming_dataset import StreamingDataset

def inspect_first_batch():
    print("🕵️ INICIANDO INSPEÇÃO DO PRIMEIRO BATCH...")
    
    # 1. Configuração
    MONTHS = ["2026-01"] 
    SEQ_LEN = 32
    BATCH_SIZE = 4
    
    # 2. Carrega Dataset e DataLoader
    print(f"   📅 Meses: {MONTHS}")
    dataset = StreamingDataset(MONTHS, seq_len=SEQ_LEN)
    # Define um range curto para ser rápido
    start_date = "2026-01-01"
    end_date = "2026-01-02"
    print(f"   ⏳ Range: {start_date} a {end_date}")
    dataset.set_date_range(start_date, end_date)
    
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)
    
    # 3. Pega o 1º Batch
    try:
        print("   🔄 Iterando DataLoader...")
        # Itera manual para pegar só o primeiro
        data_iter = iter(loader)
        inputs, targets = next(data_iter)
        
        print("\n✅ BATCH CAPTURADO COM SUCESSO!")
        print(f"📦 Input Shape: {inputs.shape}  (Batch, Seq, Channels, Height)")
        print(f"🎯 Target Shape: {targets.shape} (Batch,)")
        print(f"🏷️ Labels no Batch: {targets.tolist()}")
        
        # 4. Análise Estatística (Detectando Normalização)
        print("\n📊 ESTATÍSTICAS DOS CANAIS:")
        channels = ["Bids (Log Vol)", "Asks (Log Vol)", "OFI", "Activity (Log Count)"]
        for c in range(4):
            # Pega todos os dados deste canal no batch
            chan_data = inputs[:, :, c, :]
            print(f"   🔹 Canal {c} ({channels[c]}):")
            print(f"      Min: {chan_data.min():.4f} | Max: {chan_data.max():.4f} | Mean: {chan_data.mean():.4f}")
            
            if chan_data.max() > 20: 
                print("      ⚠️ ALERTA: Valores altos (>20)! Verifique se está usando Log.")
            
        
        # 5. Visualização (Plota o 1º sample do batch)
        print("\n🖼️ GERANDO IMAGEM DE DIAGNÓSTICO...")
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # Pega o primeiro exemplo do batch (index 0)
        sample_idx = 0
        
        for c in range(4):
            # Heatmap: Eixo Y=Preço (128), Eixo X=Tempo (SEQ_LEN)
            # Transpose -> (Height, Seq)
            heatmap = inputs[sample_idx, :, c, :].T 
            
            sns.heatmap(heatmap, ax=axes[c], cmap="viridis", cbar=True)
            axes[c].set_title(f"Canal {c}: {channels[c]}")
            axes[c].invert_yaxis() 
            axes[c].set_xlabel("Tempo (Frames)")
            axes[c].set_ylabel("Níveis de Preço")

        plt.suptitle(f"Raio-X do Input (Sample {sample_idx}, Label={targets[sample_idx]})", fontsize=16)
        plt.tight_layout()
        output_file = project_root / "debug_batch_inspection.png"
        plt.savefig(output_file)
        print(f"📸 Imagem salva como '{output_file}'. Abra para ver!")
        
    except StopIteration:
        print("❌ Erro: O DataLoader retornou vazio. Verifique se há dados no mês selecionado.")
    except Exception as e:
        print(f"❌ Erro Crítico: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    inspect_first_batch()
