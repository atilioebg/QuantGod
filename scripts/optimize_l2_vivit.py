import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import optuna
from sklearn.preprocessing import StandardScaler
import sys

# Project imports
sys.path.append(str(Path.cwd()))
from src.models.model import QuantGodModel

# Config
DATA_PATH = Path("data/processed/l2_features_1min.parquet")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_TRIALS = 20

def create_sequences(X, y, seq_len):
    """
    Cria janelas deslizantes [t : t+seq_len] -> Target[t+seq_len-1]
    """
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i : i+seq_len])
        ys.append(y[i + seq_len - 1]) # Target alinhado ao último passo
    return np.array(Xs), np.array(ys)

def optimize_l2():
    print("============================================================")
    print("AUDITORIA DE DADOS CONCLUÍDA") # Step 5 Requirement
    print("L2 MICRO-FINE-TUNING - QUANT GOD")
    print("============================================================")
    
    # 1. Carregamento e Inspeção (Validator & Human Check)
    if not DATA_PATH.exists():
        print(f"[ERRO] Arquivo {DATA_PATH} não encontrado.")
        return

    df = pd.read_parquet(DATA_PATH)
    
    # Step 1.1: Validação de Classes
    target_col = 'target_class'
    class_counts = df[target_col].value_counts().sort_index()
    total_samples = len(df)
    
    print("\n[INFO] Distribuição de Classes:")
    print(class_counts)
    
    # Calcular pesos (inverso da frequência)
    # weight[c] = Total / (Num_Classes * Count[c])
    weights = total_samples / (len(class_counts) * class_counts)
    class_weights = torch.tensor(weights.values, dtype=torch.float32).to(DEVICE)
    print(f"[INFO] Class Weights Calculados: {class_weights}")

    # Step 1.2: Conferência Humana
    print("\n[INFO] Amostra para Validação Humana:")
    # Garantir colunas existem
    cols_check = ['close', target_col, 'future_ret_5m'] # Micro-Price virou 'close' no resampling
    # Adicionar datetime se estiver no index ou coluna
    if not isinstance(df.index, pd.DatetimeIndex):
         if 'datetime' in df.columns:
             cols_check.insert(0, 'datetime')
    
    print(df[cols_check].head(5))
    print("------------------------------------------------------------")
    
    # 2. Preparação dos Tensores (Dynamic Dimensions)
    # X = Tudo exceto Target, Return e Datas
    drop_cols = [target_col, 'future_ret_5m', 'datetime', 'open_time', 'close_time', 'close'] # Lista segura
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    X_raw = df[feature_cols].values.astype(np.float32)
    y_raw = df[target_col].values.astype(np.int64)
    
    NUM_FEATURES = X_raw.shape[1]
    print(f"[INFO] Features Auto-Detectadas: {NUM_FEATURES} ({feature_cols})")
    
    # Split Chronological (80/20)
    split_idx = int(len(X_raw) * 0.8)
    
    X_train_raw = X_raw[:split_idx]
    y_train_raw = y_raw[:split_idx]
    
    X_val_raw = X_raw[split_idx:]
    y_val_raw = y_raw[split_idx:]
    
    # Normalização (StandardScaler fit no treino)
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train_raw)
    X_val_norm = scaler.transform(X_val_raw)
    
    def objective(trial):
        # 3. Hiperparâmetros
        seq_len = trial.suggest_int('seq_len', 30, 60)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        d_model = trial.suggest_categorical('d_model', [64, 128])
        lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
        dropout = trial.suggest_float('dropout', 0.2, 0.5)
        
        # Sliding Window Dataset (Precisa refazer pois seq_len muda)
        X_train_seq, y_train_seq = create_sequences(X_train_norm, y_train_raw, seq_len)
        X_val_seq, y_val_seq = create_sequences(X_val_norm, y_val_raw, seq_len)
        
        # Tensor & DataLoader
        train_ds = TensorDataset(torch.tensor(X_train_seq).to(DEVICE), torch.tensor(y_train_seq).to(DEVICE))
        val_ds = TensorDataset(torch.tensor(X_val_seq).to(DEVICE), torch.tensor(y_val_seq).to(DEVICE))
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        
        # Modelo
        model = QuantGodModel(
            num_features=NUM_FEATURES,
            seq_len=seq_len,
            d_model=d_model,
            num_classes=3,
            dropout=dropout
        ).to(DEVICE)
        
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=class_weights) # Usando Pesos calculados
        
        # Loop de Treino
        best_f1 = 0.0
        patience = 5
        no_improve = 0
        
        for epoch in range(50): # Max Epochs
            model.train()
            for X_b, y_b in train_loader:
                optimizer.zero_grad()
                out = model(X_b)
                loss = criterion(out, y_b)
                loss.backward()
                optimizer.step()
            
            # Validação
            model.eval()
            val_loss = 0.0
            
            # Confusion Matrix para Macro F1
            confusion = torch.zeros(3, 3).to(DEVICE)
            
            with torch.no_grad():
                for X_b, y_b in val_loader:
                    out = model(X_b)
                    loss = criterion(out, y_b)
                    val_loss += loss.item()
                    _, preds = torch.max(out, 1)
                    
                    for t, p in zip(y_b.view(-1), preds.view(-1)):
                        confusion[t.long(), p.long()] += 1
            
            # Calcular F1 Macro (Manual para evitar Sklearn overhead no loop/GPU)
            tp = torch.diag(confusion)
            fp = confusion.sum(0) - tp
            fn = confusion.sum(1) - tp
            
            f1_classes = 2 * tp / (2 * tp + fp + fn + 1e-8)
            macro_f1 = f1_classes.mean().item()
            
            # Early Stopping Check
            trial.report(macro_f1, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            
            if macro_f1 > best_f1:
                best_f1 = macro_f1
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break
                    
        return best_f1

    # 4. Otimização
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=N_TRIALS)
    
    print("\n------------------------------------------------------------")
    print(f"MELHOR F1 SCORE L2: {study.best_value:.4f}")
    print(f"Melhores Params: {study.best_params}")
    print("------------------------------------------------------------")

    # Exportar para JSON para automação (MLOps)
    out_path = Path("data/artifacts/best_l2_vivit_params.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(out_path, 'w') as f:
        json.dump(study.best_params, f, indent=4)
    print(f"[INFO] Melhores parâmetros salvos em {out_path}")

if __name__ == "__main__":
    optimize_l2()
