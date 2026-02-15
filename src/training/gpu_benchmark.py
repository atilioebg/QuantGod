import torch
from src.models.vivit import QuantGodViViT
import gc

def benchmark():
    device = torch.device('cuda')
    model = QuantGodViViT().to(device)
    model.train()
    
    configs = [3, 8, 16, 32, 64, 128, 256, 512, 1024]
    
    print(f"{'Batch Size':<12} | {'VRAM (MB)':<12} | {'Status':<10}")
    print("-" * 40)
    
    for b in configs:
        try:
            torch.cuda.empty_cache()
            gc.collect()
            
            x = torch.randn(b, 96, 6, 128).to(device)
            y = torch.randint(0, 4, (b,)).to(device)
            
            # Forward + Backward
            output = model(x)
            loss = torch.nn.functional.cross_entropy(output, y)
            loss.backward()
            
            vram = torch.cuda.memory_allocated() / 1024**2
            print(f"{b:<12} | {vram:<12.2f} | SUCCESS")
            
        except torch.OutOfMemoryError:
            print(f"{b:<12} | {'-':<12} | OOM ERROR")
            break
        except Exception as e:
            print(f"{b:<12} | Error      | {str(e)}")
            break

if __name__ == "__main__":
    benchmark()
