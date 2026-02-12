"""
SAIMP Training Script - Professional Edition
Optimized for 32GB RAM + 2GB VRAM

Techniques Implemented:
1. Gradient Accumulation (Virtual Batch Size)
2. Mixed Precision Training (AMP)
3. Chronological Split (No Data Leakage)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import polars as pl
import numpy as np
from pathlib import Path
import sys
from datetime import datetime
import gc  # Garbage Collector
import matplotlib.pyplot as plt
# Add project root to path
sys.path.append(str(Path.cwd()))

from src.utils.logger import log_memory_usage

from src.config import settings
from src.processing.simulation import build_simulated_book
from src.processing.tensor_builder import build_tensor_4d
from src.processing.labeling import generate_labels
from src.models.vivit import SAIMPViViT

# ============================================================================
# LOGGING INFRASTRUCTURE
# ============================================================================
class TeeLogger:
    """
    Redirects print() and stdout/stderr to both console and a file.
    """
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def close(self):
        self.log.close()

def setup_training_logger():
    """
    Sets up the log file and redirects stdout/stderr.
    Returns the log file path.
    """
    log_dir = settings.LOGS_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"train_run_{timestamp}.txt"
    
    # Redirect stdout and stderr
    sys.stdout = TeeLogger(log_path)
    # sys.stderr = TeeLogger(log_path) # Optional: Redirect stderr too
    
    return log_path

def plot_training_history(history, save_path):
    """
    Plots training and validation metrics.
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    plt.plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    plt.title('Training & Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\n[GRAPH] Gráfico de evolução salvo em: {save_path}")
    plt.close()

def generate_month_list(start_month: str, end_month: str) -> list[str]:
    """
    Gera uma lista de meses no formato 'YYYY-MM' entre start_month e end_month (inclusive).
    """
    start_dt = datetime.strptime(start_month, "%Y-%m")
    end_dt = datetime.strptime(end_month, "%Y-%m")
    
    months = []
    curr_dt = start_dt
    while curr_dt <= end_dt:
        months.append(curr_dt.strftime("%Y-%m"))
        # Move para o primeiro dia do próximo mês
        if curr_dt.month == 12:
            curr_dt = curr_dt.replace(year=curr_dt.year + 1, month=1)
        else:
            curr_dt = curr_dt.replace(month=curr_dt.month + 1)
    return months

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss, val_acc, model, save_path, current_epoch, total_epochs):
        score = -val_loss

        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, val_acc, model, save_path)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f'   [EARLY STOP] Counter: {self.counter} of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, val_acc, model, save_path)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_acc, model, save_path):
        '''Saves model when validation loss decrease.'''
        if self.val_loss_min != np.inf:  # Don't print on first save
            print(f'   [SAVE] Val Loss improved ({self.val_loss_min:.4f} --> {val_loss:.4f}). Saving model...')
        else:
            print(f'   [SAVE] Model Saved (Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%)')
            
        torch.save(model.state_dict(), save_path)
        self.val_loss_min = val_loss

# ============================================================================
# CONFIGURATION - Professional ML Engineering
# ============================================================================

# Chronological Split (80/20) - Full Historical Mode
TRAIN_MONTHS = generate_month_list("2023-01", "2025-10")
VAL_MONTHS = generate_month_list("2025-11", "2026-01")

print(f"\n[DATA] Full History Setup:")
print(f"   Treino (Total):    {len(TRAIN_MONTHS)} meses")
print(f"   Validacao (Total): {len(VAL_MONTHS)} meses")

# Memory Optimization
BATCH_SIZE = settings.BATCH_SIZE
ACCUMULATION_STEPS = settings.ACCUMULATION_STEPS
EFFECTIVE_BATCH_SIZE = BATCH_SIZE * ACCUMULATION_STEPS

# Training Hyperparameters
SEQ_LEN = settings.SEQ_LEN
EPOCHS = settings.EPOCHS
LEARNING_RATE = settings.LEARNING_RATE
WEIGHT_DECAY = settings.WEIGHT_DECAY

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

# ============================================================================
# DATA LOADING - STREAMING MODE
# ============================================================================

from src.training.streaming_dataset import StreamingDataset

# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train():
    """
    Main training loop with professional ML engineering techniques.
    """
    # Setup Logger
    log_path = setup_training_logger()
    
    print("=" * 80)
    print("SAIMP TRAINING - PROFESSIONAL EDITION (STREAMING)")
    print("=" * 80)
    print(f"[LOG] Output saved to: {log_path}")
    print(f"[TIME] Start: {datetime.now()}")
    print(f"\n[INFO] Configuracao:")
    print(f"   Treino: {TRAIN_MONTHS}")
    print(f"   Validacao: {VAL_MONTHS}")
    print(f"   Batch Fisico: {BATCH_SIZE}")
    print(f"   Gradient Accumulation: {ACCUMULATION_STEPS} steps")
    print(f"   Batch Efetivo: {EFFECTIVE_BATCH_SIZE}")
    print(f"   Sequencia: {SEQ_LEN} frames (24h @ 15min)")
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Hardware: {device}")
    
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        try:
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"   VRAM: {total_mem:.2f} GB")
        except:
            pass
        print(f"   Mixed Precision (AMP): [ENABLED]")
    else:
        print(f"   [WARN] Running on CPU - Training will be slow")

    # ========================================================================
    # STEP 1: Initialize Streaming Datasets
    # ========================================================================
    print("\n[INFO] Inicializando Streaming Datasets...")
    log_memory_usage()
    
    # Streaming does not support shuffle=True in DataLoader traditionally
    # We rely on the stream yielding data in a way that is acceptable.
    # For Time Series, sequential is actually often fine or necessary for stateful models, 
    # but here we use stateless sequences.
    
    train_dataset = StreamingDataset(TRAIN_MONTHS, seq_len=SEQ_LEN)
    # Ignoramos datas exatas do config para usar o range completo dos meses gerados
    train_dataset.set_date_range(None, None) 
    
    val_dataset = StreamingDataset(VAL_MONTHS, seq_len=SEQ_LEN)
    # Ignoramos datas exatas do config para usar o range completo dos meses gerados
    val_dataset.set_date_range(None, None) 
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,  # IterableDataset does not support shuffle
        pin_memory=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
        num_workers=0
    )
    
    print(f"   [OK] Datasets Prontos (Lazy Loading)")

    # ========================================================================
    # STEP 2: Initialize Model
    # ========================================================================
    print("\n[INFO] Inicializando Modelo...")
    model = SAIMPViViT(
        seq_len=settings.SEQ_LEN,
        input_channels=settings.INPUT_CHANNELS,
        price_levels=settings.PRICE_LEVELS,
        num_classes=settings.NUM_CLASSES
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Parametros: {trainable_params:,} (Total: {total_params:,})")

    # ========================================================================
    # STEP 3: Loss, Optimizer, and AMP Scaler
    # ========================================================================
    # Class weights (penalize neutral predictions)
    class_weights = torch.tensor(settings.CLASS_WEIGHTS).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    # Mixed Precision Scaler
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    print(f"\n[INFO] Otimizador: AdamW (lr={LEARNING_RATE}, wd={WEIGHT_DECAY})")
    print(f"   Loss: CrossEntropyLoss (weights={class_weights.tolist()})")

    # ========================================================================
    # STEP 4: Training Loop
    # ========================================================================
    best_val_acc = 0.0
    best_val_loss = float('inf')
    save_path = settings.DATA_DIR / "saimp_best.pth"
    
    # Initialize Early Stopping
    early_stopping = EarlyStopping(patience=3, min_delta=0.001)
    
    # Store metrics for plotting
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    print("\n" + "=" * 80)
    print("[START] INICIANDO TREINAMENTO")
    print("=" * 80)

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        print(f"\n[EPOCH] Epoch {epoch + 1}/{EPOCHS}")
        print("-" * 80)
        
        optimizer.zero_grad()
        
        # NOTE: enumerate(train_loader) with IterableDataset might not return length
        # So we cannot easily show progress bar with total batches.
        
        batch_count = 0
        log_memory_usage()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Ensures target is 1D (Batch,)
            if target.dim() > 1:
                target = target.squeeze()
            
            # Mixed Precision Forward Pass
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    output = model(data)
                    loss = criterion(output, target)
                    loss = loss / ACCUMULATION_STEPS
                
                scaler.scale(loss).backward()
            else:
                output = model(data)
                loss = criterion(output, target)
                loss = loss / ACCUMULATION_STEPS
                loss.backward()
            
            # Gradient Accumulation
            if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                optimizer.zero_grad()
            
            # Metrics
            train_loss += loss.item() * ACCUMULATION_STEPS
            _, predicted = torch.max(output.data, 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
            
            batch_count += 1
            
            # Progress logging
            if batch_idx % 1 == 0:
                current_acc = 100.0 * train_correct / train_total if train_total > 0 else 0
                print(f"   Batch {batch_idx:4d} | "
                      f"Loss: {loss.item() * ACCUMULATION_STEPS:.4f} | "
                      f"Acc: {current_acc:.2f}%", flush=True)

            if batch_idx % 10 == 0:
                log_memory_usage()
        
        # End of Epoch Optimizer Step (if residue)
        if batch_count % ACCUMULATION_STEPS != 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        
        avg_train_loss = train_loss / batch_count if batch_count > 0 else 0
        train_acc = 100.0 * train_correct / train_total if train_total > 0 else 0
        
        # VALIDATION PHASE
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_batches = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                
                if target.dim() > 1:
                    target = target.squeeze()
                
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        output = model(data)
                        loss = criterion(output, target)
                else:
                    output = model(data)
                    loss = criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
                val_batches += 1
                
                # Memória: Limpeza a cada batch de validação se necessário, 
                # mas o pedido foi no final de cada iteração (loop) de validação.
                # Como o loop é o que gera os batches, vamos colocar após cada batch 
                # ou ao final do loop total. O pedido diz "No final de cada iteração de validação".
                # Interpretando como final do loop de batches da validação.
        
        gc.collect() # Final da iteração de validação
        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
        val_acc = 100.0 * val_correct / val_total if val_total > 0 else 0
        
        print(f"\n[INFO] Epoch {epoch + 1} Summary:")
        print(f"   Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        # Early Stopping & Model Checkpoint Logic
        early_stopping(avg_val_loss, val_acc, model, save_path, epoch, EPOCHS)
        
        if early_stopping.early_stop:
            print("\n[STOP] Early Stopping triggered! Parando treino para evitar Overfitting.")
            break
        
        # Keep track of best accuracy separately just for reporting
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        gc.collect() # Final da época de treino (pós validação)
        log_memory_usage()
        # Track History
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Plot Metrics
    plot_path = log_path.with_suffix('.png')
    plot_training_history(history, plot_path)

    print("\n" + "=" * 80)
    print("[DONE] TREINAMENTO CONCLUIDO")
    print("=" * 80)
    print(f"   Melhor Val Acc: {best_val_acc:.2f}%")
    print(f"   Modelo Salvo: {save_path}")
    print("\n[INFO] Proximos Passos:")
    print("   1. Avaliar modelo com test set (2026-02+)")
    print("   2. Implementar dashboard de inferencia ao vivo")
    print("   3. Escalar para nuvem (GPU com mais VRAM)")


if __name__ == "__main__":
    try:
        train()
    except KeyboardInterrupt:
        print("\n[STOP] Treino interrompido pelo usuario.")
    except Exception as e:
        print(f"\n[ERROR] Erro critico: {e}")
        import traceback
        traceback.print_exc()
        
        if "CUDA out of memory" in str(e):
            print("\n[TIP] DICA: Reduza BATCH_SIZE para 2 ou aumente ACCUMULATION_STEPS.")
