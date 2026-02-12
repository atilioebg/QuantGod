import polars as pl
import numpy as np
from src.config import settings
from src.processing.features import clean_trade_data, calculate_ofi
from src.processing.tensor_builder import build_book_image
from src.processing.loader import load_stream_data, load_trade_data
import matplotlib.pyplot as plt

def main():
    print("🚀 Iniciando DeepSwing Pipeline Test...")
    
    print("-" * 50)
    
    # 1. Carregar Dados de Trades (Stream)
    print("📂 Carregando Stream Trades...")
    try:
        trades_lf = load_trade_data()
        # Preview
        # Verifica se tem dados
        # O LazyFrame só "sabe" se tem dados ao coletar ou fetch.
        # Vamos tentar um fetch(1)
        if trades_lf.select(pl.len()).collect().item() == 0:
             print("⚠️ Nenhum arquivo de trade encontrado.")
             trades_loaded = False
        else:
             print("   -> Dados de trade detectados.")
             trades_loaded = True
             
    except Exception as e:
        print(f"❌ Erro ao detectar trades: {e}")
        trades_loaded = False

    # 2. Carregar Dados de Depth (Stream)
    print("📂 Carregando Stream Depth...")
    depth_df = load_stream_data()
    print(f"   -> Snapshots de Livro carregados: {depth_df.height}")
    
    if not trades_loaded and depth_df.height == 0:
        print("❌ Nenhum dado encontrado. O stream.py rodou tempo suficiente?")
        return

    # 3. Calcular OFI
    if trades_loaded:
        print("\n🧮 Calculando OFI (1 minuto)...")
        # Limpeza
        trades_clean = clean_trade_data(trades_lf)
        
        # OFI
        try:
            ofi_df = calculate_ofi(trades_clean, window="1m")
            print(ofi_df.head())
        except Exception as e:
            print(f"❌ Erro ao calcular OFI: {e}")
            print("Verifique se as colunas 'p', 'q', 'm' existem no parquet.")

    # 4. Rasterizar Imagem
    if depth_df.height > 0:
        print("\n🖼️ Gerando Tensores (Imagens do Livro)...")
        # Pega apenas os últimos 100 snapshots para teste visual
        sample_depth = depth_df.tail(100)
        
        try:
            tensor = build_book_image(sample_depth, n_levels=128)
            print(f"✅ Tensor Gerado com Sucesso! Shape: {tensor.shape}")
            print("   (Time, Channels, Height) -> (T, 2, 128)")
            
            # 5. Visualização Rápida
            if tensor.shape[0] > 0:
                print("📊 Plotando Heatmap do Canal de Bids...")
                plt.figure(figsize=(10, 6))
                # Plotamos o canal 0 (Bids) transposto
                plt.imshow(tensor[:, 0, :].T, aspect='auto', cmap='viridis', origin='lower')
                plt.colorbar(label='Log(Volume)')
                plt.title("Visualização do Tensor (Bids) - DeepSwing Eye")
                plt.xlabel("Tempo (Snapshots)")
                plt.ylabel("Níveis de Preço (0 = Topo do Livro)")
                
                print("Salvando teste_visualizacao.png...")
                plt.savefig("teste_visualizacao.png")
                # plt.show() # Pode travar em ambiente sem display
                print("Gráfico salvo.")
        except Exception as e:
            print(f"❌ Erro ao construir tensor: {e}")

if __name__ == "__main__":
    main()
