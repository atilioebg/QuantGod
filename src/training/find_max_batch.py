import torch
import gc
import time
from src.config import settings
from src.models.vivit import QuantGodViViT as QuantGodModel
from src.processing.tensor_builder import build_tensor_6d
import numpy as np

def find_max_batch():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu':
        print("CUDA not available.")
        return

    # Model Config
    model = QuantGodModel().to(device)
    model.train()
    
    # Mock Input Shape based on 6D Tensor (B, 96, 6, 128)
    seq_len = 96
    channels = 6
    features = 128
    
    results = []
    batch_size = 1 # Start from 1 as requested or current settings.BATCH_SIZE
    
    print(f"{'Batch Size':<12} | {'VRAM Allocated (MB)':<20} | {'Status':<10}")
    print("-" * 50)

    try:
        while True:
            try:
                # Clear cache
                torch.cuda.empty_cache()
                gc.collect()
                
                # Create mock data
                x = torch.randn(batch_size, seq_len, channels, features).to(device)
                y = torch.randint(0, 4, (batch_size,)).to(device)
                
                # Forward pass
                output = model(x)
                loss = torch.nn.functional.cross_entropy(output, y)
                
                # Backward pass (this is where most memory is consumed)
                loss.backward()
                
                mem_alloc = torch.cuda.memory_allocated(0) / 1024 / 1024
                print(f"{batch_size:<12} | {mem_alloc:<20.2f} | SUCCESS")
                results.append({"batch_size": batch_size, "vram_mb": mem_alloc, "status": "SUCCESS"})
                
                batch_size += 8 # Jump faster
                
            except Exception as e:
                if 'out of memory' in str(e).lower():
                    print(f"{batch_size:<12} | {'-':<20} | OOM ERROR")
                    break
                else:
                    raise e
    except KeyboardInterrupt:
        print("Test interrupted.")

    # Final Summary Table
    print("\n--- Comparative Table ---")
    print(f"{'Batch Size':<12} | {'VRAM (MB)':<12}")
    for res in results:
         print(f"{res['batch_size']:<12} | {res['vram_mb']:<12.2f}")
    
    if results:
        print(f"\nMax Batch Size found: {results[-1]['batch_size']}")

if __name__ == "__main__":
    find_max_batch()
