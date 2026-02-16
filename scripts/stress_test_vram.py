
import torch
import torch.nn as nn
import sys
from pathlib import Path
import gc
import time

# Adicionar raiz do projeto ao path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.models.model import QuantGodModel

def get_vram_usage():
    """Retorna VRAM alocada em GB"""
    return torch.cuda.memory_allocated() / (1024**3)

def run_stress_test():
    print("============================================================")
    print("🚀 QUANTGOD VRAM STRESS TEST - LONG CONTEXT (720 steps)")
    print("============================================================")
    
    if not torch.cuda.is_available():
        print("❌ Erro: CUDA não disponível. O teste exige uma GPU.")
        return

    device = torch.device("cuda")
    seq_len = 720
    num_features = 9
    batch_sizes = [2, 4, 8, 16, 32, 64, 128, 256]
    
    # Configuração do Modelo (Fase Realista)
    model_params = {
        "num_features": num_features,
        "seq_len": seq_len,
        "d_model": 128,
        "nhead": 4,
        "num_layers": 2,
        "num_classes": 3
    }

    max_batch = 0
    
    # Cores ANSI
    GREEN = "\033[92m"
    RED = "\033[91m"
    RESET = "\033[0m"
    CYAN = "\033[96m"

    for b in batch_sizes:
        print(f"Testing Batch Size: {b}...", end="\r")
        
        # Limpar memória antes de cada teste
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        try:
            # 1. Instanciar Modelo e Otimizador (simular overhead real)
            model = QuantGodModel(**model_params).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.CrossEntropyLoss()
            
            # 2. Criar Dados Dummy
            x = torch.randn(b, seq_len, num_features).to(device)
            y = torch.randint(0, 3, (b,)).to(device)
            
            # 3. Forward Pass
            logits = model(x)
            
            # 4. Backward Pass (Onde a memória explode por causa dos gradientes)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
            vram = get_vram_usage()
            print(f"{GREEN}✅ Batch {b:3d}: OK - VRAM: {vram:5.2f} GB{RESET}")
            max_batch = b
            
            # Deletar objetos para liberar memória
            del model, optimizer, x, y, logits, loss
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"{RED}❌ Batch {b:3d}: OOM (Out of Memory){RESET}")
                # Na primeira falha, interrompemos
                break
            else:
                print(f"{RED}❌ Erro inesperado no Batch {b}: {e}{RESET}")
                break
        except Exception as e:
            print(f"{RED}❌ Erro no Batch {b}: {e}{RESET}")
            break

    print("\n------------------------------------------------------------")
    print(f"{CYAN}📊 RELATÓRIO FINAL DE CAPACIDADE{RESET}")
    print("------------------------------------------------------------")
    if max_batch > 0:
        safe_batch = int(max_batch * 0.8) # Margem de 20%
        # Garantir que é pelo menos potência de 2 se possível, ou apenas rounded
        print(f"Max Batch Suportado: {max_batch}")
        print(f"Recomendação Safe  : {GREEN}{safe_batch}{RESET}")
        print("\nUtilize este valor nas configurações de treino para evitar")
        print("crashes durante a produção.")
    else:
        print(f"{RED}Nenhum batch size suportado. Reduza 'seq_len' ou 'd_model'.{RESET}")

if __name__ == "__main__":
    run_stress_test()
