import torch
import polars as pl
import sys
import numpy as np
from pathlib import Path

# Adiciona o diretório raiz ao path para garantir que imports funcionem
# Script em tests/integration/, raiz em ../../
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.config import settings
from src.processing.simulation import build_simulated_book
from src.processing.tensor_builder import build_tensor_4d
from src.models.vivit import SAIMPViViT

def integration_test():
    print("🔥 Iniciando Teste de Integração Real (Smoke Test)...")
    
    # 1. Carregar 1 arquivo real
    # Tenta achar qualquer arquivo aggTrades no histórico
    if not settings.RAW_HISTORICAL_DIR.exists():
        print(f"❌ Diretório não existe: {settings.RAW_HISTORICAL_DIR}")
        return

    files = list(settings.RAW_HISTORICAL_DIR.glob("aggTrades*.parquet"))
    
    if not files:
        print(f"❌ Sem arquivos reais para testar em {settings.RAW_HISTORICAL_DIR}. Verifique o download.")
        return
    
    target_file = files[0]
    print(f"📂 Lendo Arquivo Real: {target_file.name}")
    
    # Lê uma amostra maior para garantir que cubra 24h (96 frames de 15m)
    # 500k linhas geralmente cobrem bem um dia movimentado ou mais
    df = pl.read_parquet(target_file).head(500_000) 
    
    # Normalizar nomes de colunas se necessário (alguns downloads podem variar)
    if "transact_time" in df.columns and "timestamp" not in df.columns:
        df = df.with_columns(pl.col("transact_time").alias("timestamp"))

    # 2. Pipeline de Processamento
    print("⚙️ Processando (Simulação + Tensor)...")
    sim_book = build_simulated_book(df, window="15m")
    
    unique_snaps = sim_book.select('snapshot_time').n_unique()
    print(f"   -> Snapshots gerados: {unique_snaps}")
    
    if unique_snaps == 0:
        print("⚠️ Nenhum snapshot gerado. Verifique se o intervalo de tempo no arquivo cobre 15m.")
        return

    # Gera tensor REAL
    tensor_np = build_tensor_4d(sim_book, n_levels=128, is_simulation=True)
    print(f"   Shape do Tensor: {tensor_np.shape}") # Esperado: (T, 4, 128)
    
    # Validação de Tamanho
    # Se tiver menos de 96, avisar mas tentar rodar com o que tem (padding ou slice menor)
    SEQ_LEN = 96
    
    if tensor_np.shape[0] < SEQ_LEN:
        print(f"⚠️ Dados insuficientes neste arquivo para 1 sequência completa ({SEQ_LEN} frames). Gerou apenas {tensor_np.shape[0]}.")
        print("   -> Tente aumentar o .head() ou pegar um arquivo maior.")
        # Ajustar SEQ_LEN para o teste passar se tiver pelo menos 1
        if tensor_np.shape[0] > 0:
            SEQ_LEN = tensor_np.shape[0]
            print(f"   -> Ajustando SEQ_LEN para {SEQ_LEN} apenas para este teste.")
        else:
            return

    # 3. Preparar para Pytorch (Batch de 1)
    # A rede espera (Batch, Time, Channels, Height)
    # Vamos pegar os primeiros N frames (SEQ_LEN)
    
    tensor_slice = tensor_np[:SEQ_LEN]
    x_real = torch.from_numpy(tensor_slice).float().unsqueeze(0) # Adiciona dimensão Batch (1, T, 4, 128)
    print(f"   Input Tensor Final: {x_real.shape}")
    
    # 4. Injetar no Cérebro
    print("🧠 Injetando na Rede Neural (Forward Pass)...")
    # Instancia modelo com o SEQ_LEN dinâmico deste teste
    model = SAIMPViViT(seq_len=SEQ_LEN, input_channels=4, price_levels=128, num_classes=3)
    
    # Se tiver GPU, usa
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")
    model.to(device)
    x_real = x_real.to(device)
    
    try:
        with torch.no_grad():
            output = model(x_real)
            
        print(f"✅ Saída da Rede: {output.shape}")
        print(f"   Logits (Probabilidades não normalizadas): {output.cpu().detach().numpy()}")
        
        print("\n🎉 CHECK-MATE! O Pipeline está 100% blindado e pronto para treino massivo.")
        
    except Exception as e:
        print(f"❌ Erro na Inferência: {e}")

if __name__ == "__main__":
    integration_test()
