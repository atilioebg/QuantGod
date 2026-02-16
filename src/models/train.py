import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import polars as pl
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from src.config import settings
from src.models.model import QuantGodModel
import time

def create_sequences(X, y, seq_len):
    """Cria sequencias 3D (Batch, Time, Features)"""
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i : i+seq_len])
        ys.append(y[i + seq_len - 1])
    return np.array(Xs), np.array(ys)

def train():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"INFO: Iniciando Treino Final no dispositivo: {DEVICE}")
    
    # 1. Carregamento e Labelling (Swing Trade Profile)
    if not Path(settings.DATA_PATH).exists():
        print(f"ERROR: Dataset nao encontrado em {settings.DATA_PATH}")
        return

    print(f"INFO: Carregando dados de {settings.DATA_PATH}...")
    df = pl.read_parquet(settings.DATA_PATH)
    
    # Parametros da Estrategia
    lookahead = 60 # 1h
    threshold_long = 0.008 # +0.8%
    threshold_short = -0.004 # -0.4%
    
    # Calcular Target
    print("INFO: Aplicando rotulagem assimetrica (Swing)...")
    df = df.with_columns([
        pl.col("log_ret_close").shift(-lookahead).rolling_sum(window_size=lookahead).alias("future_return_60m")
    ])
    
    df = df.with_columns([
        pl.when(pl.col("future_return_60m") > threshold_long).then(2)
        .when(pl.col("future_return_60m") < threshold_short).then(0)
        .otherwise(1)
        .alias("target")
    ])
    
    # Limpeza de NaNs gerados pelo shift/rolling
    df = df.slice(settings.SEQ_LEN, len(df) - settings.SEQ_LEN - lookahead)
    
    # Separar Features e Target
    drop_cols = ['target', 'future_return_60m', 'datetime', 'close']
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    print(f"INFO: Usando {len(feature_cols)} features.")
    
    X_raw = df.select(feature_cols).to_numpy().astype(np.float32)
    y_raw = df.select('target').to_numpy().flatten().astype(np.int64)
    
    # Split Chronological (80/20)
    split_idx = int(len(X_raw) * 0.8)
    X_train_raw, y_train_raw = X_raw[:split_idx], y_raw[:split_idx]
    X_val_raw, y_val_raw = X_raw[split_idx:], y_raw[split_idx:]
    
    # Normalização
    print("INFO: Normalizando dados...")
    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train_raw)
    X_val_norm = scaler.transform(X_val_raw)
    
    # Sequenciamento
    print(f"INFO: Criando sequencias (Contexto: {settings.SEQ_LEN} min)...")
    X_train_seq, y_train_seq = create_sequences(X_train_norm, y_train_raw, settings.SEQ_LEN)
    X_val_seq, y_val_seq = create_sequences(X_val_norm, y_val_raw, settings.SEQ_LEN)
    
    # Datasets e Loaders
    train_ds = TensorDataset(torch.tensor(X_train_seq).to(DEVICE), torch.tensor(y_train_seq).to(DEVICE))
    val_ds = TensorDataset(torch.tensor(X_val_seq).to(DEVICE), torch.tensor(y_val_seq).to(DEVICE))
    
    train_loader = DataLoader(train_ds, batch_size=settings.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=settings.BATCH_SIZE, shuffle=False)
    
    # Modelo
    print(f"INFO: Arquitetura d_model={settings.D_MODEL}, heads={settings.NHEAD}")
    model = QuantGodModel(
        num_features=len(feature_cols),
        seq_len=settings.SEQ_LEN,
        d_model=settings.D_MODEL,
        nhead=settings.NHEAD,
        num_classes=3,
        dropout=settings.DROPOUT
    ).to(DEVICE)
    
    # Loss ponderada para lidar com desbalanceamento (Neutral costuma dominar)
    class_counts = np.bincount(y_train_raw)
    weights = 1.0 / (class_counts + 1e-8)
    weights = weights / weights.sum()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float).to(DEVICE))
    
    optimizer = optim.AdamW(model.parameters(), lr=settings.LEARNING_RATE)
    
    # Loop de Treino
    print("\n" + "="*40)
    print("ETAPA: TREINAMENTO")
    print("="*40)
    
    best_f1 = 0
    epochs = 10
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        start_time = time.time()
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # Validação
        model.eval()
        correct = 0
        total = 0
        confusion = torch.zeros(3, 3).to(DEVICE)
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
                
                for t, p in zip(y_batch.view(-1), predicted.view(-1)):
                    confusion[t.long(), p.long()] += 1
        
        # F1 Score Macro
        tp = torch.diag(confusion)
        fp = confusion.sum(0) - tp
        fn = confusion.sum(1) - tp
        f1 = (2 * tp / (2 * tp + fp + fn + 1e-8)).mean().item()
        
        epoch_time = time.time() - start_time
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {100*correct/total:.2f}% | F1-Macro: {f1:.4f} | Time: {epoch_time:.1f}s")
        
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), "logs/best_l2_model_swing.pt")
            print("  -> Modelo salvo (Melhor F1)")

    print("\nDONE: Treinamento Finalizado.")

if __name__ == "__main__":
    train()
