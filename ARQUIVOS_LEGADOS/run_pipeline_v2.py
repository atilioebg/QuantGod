import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from src.config import settings
from src.processing.features import clean_trade_data
from src.processing.simulation import build_simulated_book
from src.processing.tensor_builder import build_tensor_4d
import sys

def main():
    print("🚀 Iniciando Teste de Simulação 4D...")
    
    # 1. Buscar um arquivo histórico de trades
    try:
        hist_files = list(settings.RAW_HISTORICAL_DIR.glob("aggTrades*.parquet"))
        if not hist_files:
            print("❌ Nenhum arquivo histórico encontrado. O download funcionou?")
            return
        
        target_file = hist_files[0]
        print(f"📂 Carregando: {target_file.name}")
        df = pl.read_parquet(target_file)
        
        # Pega apenas uma amostra (ex: 1 dia) para não travar o teste
        df = df.head(1_000_000) 
        
    except Exception as e:
        print(f"❌ Erro ao ler arquivo: {e}")
        return

    # 2. Limpeza
    print("🧹 Limpando dados...")
    # Precisamos renomear colunas do histórico para o padrão (p, q, m -> price, quantity...)
    # Se usou meu script corrigido, os nomes são: price, quantity, is_buyer_maker, transact_time
    
    # Ajuste de nomes para o simulador
    if "transact_time" in df.columns:
        df = df.with_columns(pl.col("transact_time").alias("timestamp"))
        
    # Se não tiver 'timestamp' mas tiver 'T', renomeia (caso seja do stream salvo errado)
    if "T" in df.columns and "timestamp" not in df.columns:
        df = df.rename({"T": "timestamp", "p": "price", "q": "quantity", "m": "is_buyer_maker"})

    # Garante tipos
    df = df.select([
        pl.col("timestamp"), pl.col("price"), pl.col("quantity"), pl.col("is_buyer_maker")
    ])

    # 3. Simulação (A Mágica)
    print("🔮 Simulando Order Book (Volume Profile)...")
    try:
        simulated_book = build_simulated_book(df, window="1h") # Janelas de 1h para visualização
        print(f"   -> Snapshots gerados: {simulated_book.select('snapshot_time').n_unique()}")
    except Exception as e:
        print(f"❌ Erro na simulação: {e}")
        return

    # 4. Construção do Tensor 4D
    print("🖼️ Construindo Tensor 4 Canais...")
    try:
        tensor = build_tensor_4d(simulated_book, n_levels=128, is_simulation=True)
    except Exception as e:
        print(f"❌ Erro na construção do tensor: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"✅ Tensor Final: {tensor.shape}")
    print("   Expectativa: (Tempo, 4, 128)")

    if tensor.shape[0] > 0:
        # 5. Visualização
        print("📊 Plotando Canais...")
        fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
        channels = ["Bids (Liquidez)", "Asks (Liquidez)", "OFI (Fluxo)", "Activity (Calor)"]
        
        for i in range(4):
            # Transpõe para (Height, Time)
            im = axes[i].imshow(tensor[:, i, :].T, aspect='auto', origin='lower', cmap='inferno')
            axes[i].set_title(channels[i])
            plt.colorbar(im, ax=axes[i])
            
        plt.tight_layout()
        print("Salvando teste_visualizacao_v2.png...")
        plt.savefig("teste_visualizacao_v2.png")
        # plt.show()
        print("Gráfico salvo.")
    else:
        print("⚠️ Tensor vazio. Verifique se os dados estão corretos.")

if __name__ == "__main__":
    main()
