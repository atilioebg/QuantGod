import optuna
from optuna.pruners import MedianPruner
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import json
from pathlib import Path
import sys
import gc

# Project imports
sys.path.append(str(Path.cwd()))
from src.config import settings
from src.models.vivit import QuantGodViViT
from src.training.streaming_dataset import StreamingDataset
from src.training.train import generate_month_list
from src.models.loss import FocalLoss

# ============================================================================
# CONFIGURAÇÃO DA OTIMIZAÇÃO
# ============================================================================
N_TRIALS = 50           # Número total de tentativas
STARTUP_TRIALS = 5      # Primeiras tentativas sem pruning
WARMUP_EPOCHS = 3       # Épocas antes de começar a podar
EPOCHS_PER_TRIAL = 5    # Treino curto para teste rápido
SUBSET_YEARS = True     # Usar apenas 2023-2024 para agilizar
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def objective(trial):
    # 1. Sugestão de Hiperparâmetros (Search Space - SAFE MODE)
    # ------------------------------------------------------------------------
    # Pesos das Classes (Conservador para evitar explosão de gradiente)
    w_stop = trial.suggest_float('w_stop', 1.0, 2.5)
    w_long = trial.suggest_float('w_long', 1.0, 2.5)
    # Reduzido de [2.0, 10.0] para evitar penalidade excessiva
    w_super = trial.suggest_float('w_super', 1.5, 4.0) 
    
    class_weights = torch.tensor([1.0, w_stop, w_long, w_super]).to(DEVICE)
    
    # Otimizador (Limited LR)
    lr = trial.suggest_float('lr', 1e-5, 5e-4, log=True)
    weight_decay = trial.suggest_float('wd', 1e-6, 1e-3, log=True)
    
    # Arquitetura / Regularização (Light Dropout)
    dropout = trial.suggest_float('dropout', 0.1, 0.3)
    label_smoothing = trial.suggest_float('lbl_smooth', 0.01, 0.1)
    
    # 2. Setup de Dados (Subset para velocidade)
    # ------------------------------------------------------------------------
    if SUBSET_YEARS:
        train_months = generate_month_list("2023-01", "2024-10") # Recente
    else:
        train_months = settings.TRAIN_MONTHS
        
    val_months = settings.VAL_MONTHS # Validação padrão (2024-11 em diante)
    
    # Datasets (Streaming com Cache NPZ)
    train_dataset = StreamingDataset(train_months, seq_len=settings.SEQ_LEN)
    train_dataset.set_date_range(None, None)
    
    val_dataset = StreamingDataset(val_months, seq_len=settings.SEQ_LEN)
    val_dataset.set_date_range(None, None)
    
    # Batch Size ajustado para não estourar (usando margem segura)
    batch_size = 512 
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0, pin_memory=True)
    
    # 3. Modelo
    # ------------------------------------------------------------------------
    model = QuantGodViViT(
        seq_len=settings.SEQ_LEN,
        input_channels=settings.INPUT_CHANNELS,
        price_levels=settings.PRICE_LEVELS,
        num_classes=settings.NUM_CLASSES,
        dropout=dropout # Injetando dropout dinâmico
    ).to(DEVICE)
    
    # criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    criterion = FocalLoss(weight=class_weights, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Scaler para AMP
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
    
    # 4. Loop de Treino Reduzido
    # ------------------------------------------------------------------------
    print(f"\n[TRIAL #{trial.number}] Params: {trial.params}")
    
    for epoch in range(EPOCHS_PER_TRIAL):
        model.train()
        
        # Limite de batches por época para o tuning não demorar dias
        # Ex: Treinar apenas com 10% dos dados ou 1000 batches
        max_batches = 500 
        batch_count = 0
        
        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            if target.dim() > 1: target = target.squeeze()
            
            optimizer.zero_grad()
            
            if scaler:
                with torch.amp.autocast('cuda'):
                    output = model(data)
                    loss = criterion(output, target)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
            batch_count += 1
            if batch_count >= max_batches:
                break
        
        # Validação
        model.eval()
        val_loss = 0.0
        # F1-Score Macro simplificado (precisamos das contagens TP, FP, FN por classe)
        # Para velocidade no Optuna, vamos focar na Acurácia Ponderada ou Loss de Validação
        # O User pediu F1-Score Macro. Vamos calcular usando sklearn para precisão ou manual.
        # Manual para evitar overhead de CPU conversion grande.
        
        # Matriz de confusão [4x4]
        confusion = torch.zeros(settings.NUM_CLASSES, settings.NUM_CLASSES).to(DEVICE)
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                if target.dim() > 1: target = target.squeeze()
                
                if scaler:
                    with torch.amp.autocast('cuda'):
                        output = model(data)
                        val_loss += criterion(output, target).item()
                else:
                    output = model(data)
                    val_loss += criterion(output, target).item()
                
                _, preds = torch.max(output, 1)
                
                for t, p in zip(target.view(-1), preds.view(-1)):
                    confusion[t.long(), p.long()] += 1
        
        # Calcular F1 Macro a partir da Matriz de Confusão
        f1_scores = []
        for i in range(settings.NUM_CLASSES):
            tp = confusion[i, i]
            fp = confusion[:, i].sum() - tp
            fn = confusion[i, :].sum() - tp
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
            f1_scores.append(f1)
            
        macro_f1 = torch.stack(f1_scores).mean().item()
        
        print(f"   Epoch {epoch+1}/{EPOCHS_PER_TRIAL} | Val Loss: {val_loss:.4f} | Macro F1: {macro_f1:.4f}")
        
        # 5. Pruning (O Ceifador)
        # --------------------------------------------------------------------
        trial.report(macro_f1, epoch)
        
        if trial.should_prune():
            print(f"   [PRUNED] Trial cortado na época {epoch+1}")
            raise optuna.exceptions.TrialPruned()
            
        gc.collect()

    return macro_f1

def run_optimization():
    print("=" * 60)
    print("QUANT GOD - BAYESIAN HYPERPARAMETER OPTIMIZATION")
    print("=" * 60)
    
    # Criar estudo com direção MAXIMIZE (F1-Score)
    # Pruner: MedianPruner (Corta se for pior que a mediana dos trials anteriores)
    pruner = MedianPruner(n_startup_trials=STARTUP_TRIALS, n_warmup_steps=WARMUP_EPOCHS)
    
    study = optuna.create_study(direction="maximize", pruner=pruner)
    
    print(f"[INFO] Iniciando otimização com {N_TRIALS} tentativas...")
    print(f"[INFO] Objetivo: Maximizar Macro F1-Score")
    
    try:
        study.optimize(objective, n_trials=N_TRIALS, gc_after_trial=True)
    except KeyboardInterrupt:
        print("\n[STOP] Otimização interrompida pelo usuário.")
    
    # Resultado
    print("\n" + "=" * 60)
    print("OTIMIZAÇÃO CONCLUÍDA")
    print("=" * 60)
    print(f"Melhor Trial: #{study.best_trial.number}")
    print(f"Melhor F1-Score: {study.best_value:.4f}")
    print("\nMelhores Parâmetros:")
    for key, value in study.best_params.items():
        print(f"   {key}: {value}")
        
    # Salvar em JSON
    best_params_path = settings.BASE_DIR / "best_params.json"
    with open(best_params_path, "w") as f:
        json.dump(study.best_params, f, indent=4)
        
    print(f"\n[SAVED] Parâmetros salvos em: {best_params_path}")

if __name__ == "__main__":
    run_optimization()
