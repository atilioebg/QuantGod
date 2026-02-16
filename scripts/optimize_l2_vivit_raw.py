
import polars as pl
import numpy as np
import torch
import torch.nn as nn
import optuna
from pathlib import Path
import matplotlib.pyplot as plt
import os
import json
import sys
import random
import time

# Adicionar raiz do projeto
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.models.model import QuantGodModel
from src.config import settings

# Configurações
DATA_PATH = Path("data/processed/l2_features_1min_final.parquet")
LOG_DIR = Path("logs/debug_plots")
PARAM_FILE = Path("best_l2_params.json")
SEQ_LEN = 720
LOOKAHEAD = 60
THRESHOLD_LONG = 0.008
THRESHOLD_SHORT = -0.004

def prepare_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset não encontrado em {DATA_PATH}")
    
    df = pl.read_parquet(DATA_PATH)
    
    # 1. Calcular Target (Future 60m Return)
    # log_ret_close é log(c_t / c_t-1). A soma de 60 períodos é log(c_t+60 / c_t)
    df = df.with_columns([
        pl.col("log_ret_close").shift(-LOOKAHEAD).rolling_sum(window_size=LOOKAHEAD).alias("future_return_60m")
    ])
    
    # 2. Labelling Assimétrico
    df = df.with_columns([
        pl.when(pl.col("future_return_60m") > THRESHOLD_LONG).then(2)   # LONG
        .when(pl.col("future_return_60m") < THRESHOLD_SHORT).then(0)  # SHORT
        .otherwise(1)                                                 # NEUTRAL
        .alias("target")
    ])
    
    # 3. Limpeza
    # Remover warmup (720) e NaNs do shift (60)
    df = df.slice(SEQ_LEN, len(df) - SEQ_LEN - LOOKAHEAD)
    
    # Extrair Close Acumulado para plots (audit)
    df = df.with_columns([
        pl.col("log_ret_close").cum_sum().alias("cum_price")
    ])
    
    return df

def verify_labels(df):
    """
    Gera 14 gráficos de auditoria visual para validar os labels (Swing Profile).
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"SEARCH: Iniciando Auditoria Visual em {LOG_DIR}...", flush=True)
    
    # Precisamos do Price Acumulado real para o gráfico
    # log_ret_close é o retorno entre t-1 e t.
    # O preço relativo em t é exp(cumsum(log_ret_close))
    df_plot = df.with_columns([
        (pl.col("log_ret_close").cum_sum()).exp().alias("rel_price")
    ]).to_pandas()
    
    count = 0
    # Categorias: (target_value, num_samples, name, color)
    categories = [
        (2, 6, "LONG", "green"),
        (0, 6, "SHORT", "red"),
        (1, 2, "NEUTRAL", "gray")
    ]
    
    for target_val, num, label_name, color in categories:
        subset_indices = df_plot[df_plot["target"] == target_val].index
        if len(subset_indices) == 0:
            print(f"⚠️ Aviso: Nenhuma amostra de {label_name} encontrada.")
            continue
            
        sampled_indices = random.sample(list(subset_indices), min(num, len(subset_indices)))
        
        for idx in sampled_indices:
            count += 1
            # Janela de visualização: lookback (720) + lookahead (60)
            # idx é o ponto do 'presente'. 
            # Contexto: [idx - 720, idx]
            # Target: [idx, idx + 60]
            
            if idx < SEQ_LEN or idx > len(df_plot) - LOOKAHEAD:
                continue
                
            window = df_plot.iloc[idx - SEQ_LEN : idx + LOOKAHEAD]
            time_axis = np.arange(-SEQ_LEN, LOOKAHEAD)
            prices = window["rel_price"].values
            # Normalizar para o preço no 'instante zero' (idx) ser 1.0 para comparação visual
            prices = prices / prices[SEQ_LEN] 
            
            plt.figure(figsize=(12, 6))
            # Plotar Preço
            plt.plot(time_axis[:SEQ_LEN+1], prices[:SEQ_LEN+1], color="blue", label="History (12h)")
            plt.plot(time_axis[SEQ_LEN:], prices[SEQ_LEN:], color="orange", linestyle="--", label="Target (1h)")
            
            # Highlight Background
            plt.axvspan(-SEQ_LEN, 0, color="lightgray", alpha=0.2)
            plt.axvspan(0, LOOKAHEAD, color=color, alpha=0.15)
            
            # Linhas de Threshold
            plt.axhline(1.0, color="black", linewidth=0.8, linestyle=":")
            plt.axhline(1.0 + THRESHOLD_LONG, color="green", alpha=0.3, linestyle="--")
            plt.axhline(1.0 + THRESHOLD_SHORT, color="red", alpha=0.3, linestyle="--")
            
            plt.title(f"Swing Audit #{count} | Target: {label_name} | Retorno: {window['future_return_60m'].iloc[SEQ_LEN]:.4%}")
            plt.xlabel("Minutes from Decision Point (t=0)")
            plt.ylabel("Relative Price")
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            save_path = LOG_DIR / f"label_check_{count}_{label_name}.png"
            plt.savefig(save_path)
            plt.close()

    print(f"DONE: {count} graficos de verificacao salvos em {LOG_DIR}", flush=True)

from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

def objective(trial):
    # Search Space Focado na "Gold Zone" (Architectures with F1 > 0.30)
    d_model = trial.suggest_categorical("d_model", [128, 256])
    nhead = trial.suggest_categorical("nhead", [4, 8])
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64]) 
    lr = trial.suggest_float("lr", 1e-5, 1e-4, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.4)
    
    # 1. Preparar Dados para Treino
    df = prepare_data()
    
    drop_cols = ['target', 'future_return_60m', 'datetime', 'cum_price', 'close']
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    X_raw = df.select(feature_cols).to_numpy().astype(np.float32)
    y_raw = df.select('target').to_numpy().flatten().astype(np.int64)
    
    # Split
    split_idx = int(len(X_raw) * 0.8)
    X_train_raw, y_train_raw = X_raw[:split_idx], y_raw[:split_idx]
    X_val_raw, y_val_raw = X_raw[split_idx:], y_raw[split_idx:]
    
    # Normalização
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train_raw)
    X_val_norm = scaler.transform(X_val_raw)
    
    # Sequenciamento (Simples para otimização)
    def fast_sequence(X, y, seq_len):
        # Para otimização, podemos usar passos (stride) para agilizar se necessário
        stride = 5 
        Xs, ys = [], []
        for i in range(0, len(X) - seq_len, stride):
            Xs.append(X[i : i+seq_len])
            ys.append(y[i + seq_len - 1])
        return np.array(Xs), np.array(ys)

    X_train_seq, y_train_seq = fast_sequence(X_train_norm, y_train_raw, SEQ_LEN)
    X_val_seq, y_val_seq = fast_sequence(X_val_norm, y_val_raw, SEQ_LEN)
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = TensorDataset(torch.tensor(X_train_seq).to(DEVICE), torch.tensor(y_train_seq).to(DEVICE))
    val_ds = TensorDataset(torch.tensor(X_val_seq).to(DEVICE), torch.tensor(y_val_seq).to(DEVICE))
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    # Modelo
    model = QuantGodModel(
        num_features=len(feature_cols),
        seq_len=SEQ_LEN,
        d_model=d_model,
        nhead=nhead,
        num_layers=2,
        num_classes=3,
        dropout=dropout
    ).to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Loop Curto para Otimizacao (3 Epocas)
    print(f"  > Trial {trial.number} iniciada: d_model={d_model}, lr={lr:.6f}", flush=True)
    best_f1 = 0
    for epoch in range(3):
        start_time = time.time()
        model.train()
        for X_b, y_b in train_loader:
            optimizer.zero_grad()
            out = model(X_b)
            loss = criterion(out, y_b)
            loss.backward()
            optimizer.step()
            
        # Eval
        model.eval()
        confusion = torch.zeros(3, 3).to(DEVICE)
        with torch.no_grad():
            for X_b, y_b in val_loader:
                out = model(X_b)
                _, preds = torch.max(out, 1)
                for t, p in zip(y_b.view(-1), preds.view(-1)):
                    confusion[t.long(), p.long()] += 1
        
        tp = torch.diag(confusion)
        fp = confusion.sum(0) - tp
        fn = confusion.sum(1) - tp
        f1 = (2 * tp / (2 * tp + fp + fn + 1e-8)).mean().item()
        
        if f1 > best_f1:
            best_f1 = f1
            
        duration = time.time() - start_time
        print(f"    - Epoch {epoch+1}/3: F1={f1:.4f} ({duration:.1f}s)", flush=True)
        
        # Report prunning
        trial.report(f1, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
    return best_f1

def run_optimization():
    df = prepare_data()
    verify_labels(df)
    
    print("OPTUNA: Iniciando Optuna (20 Trials)...")
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=20)
    
    print(f"Best Trial: {study.best_trial.value}")
    with open(PARAM_FILE, "w") as f:
        json.dump(study.best_params, f)
    print(f"DONE: Melhores parâmetros salvos em {PARAM_FILE}")

if __name__ == "__main__":
    run_optimization()
