import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import polars as pl
import numpy as np
import yaml
import logging
from pathlib import Path
import sys
import os

# Add project root to sys.path to allow importing from src/
project_root = str(Path(__file__).parents[2])
if project_root not in sys.path:
    sys.path.append(project_root)

from src.models.model import QuantGodModel

# Logger setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_sequences(X, y, seq_len):
    """Cria sequencias 3D (Batch, Time, Features)"""
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i : i+seq_len])
        ys.append(y[i + seq_len - 1])
    return np.array(Xs), np.array(ys)

def run_training():
    # 1. Load Config
    if len(sys.argv) > 1:
        config_path = Path(sys.argv[1])
    else:
        config_path = Path("cloud/treino/training_config.yaml")

    if not config_path.exists():
        logger.error(f"Config file not found at {config_path}")
        return

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 2. Load All Labelled Data
    labelled_dir = Path(config['paths']['labelled_dir'])
    parquet_files = sorted(list(labelled_dir.glob("*.parquet")))
    
    if not parquet_files:
        logger.error(f"No labelled data found in {labelled_dir}")
        return

    logger.info(f"Loading {len(parquet_files)} labelled files...")
    
    # Selective loading of features only
    feature_cols = [
        'log_ret_open', 'log_ret_high', 'log_ret_low', 'log_ret_close',
        'volatility', 'max_spread', 'mean_obi', 'mean_deep_obi', 'log_volume'
    ]
    
    dfs = []
    for pf in parquet_files:
        dfs.append(pl.read_parquet(pf, columns=feature_cols + ['target']))
    
    df = pl.concat(dfs)
    logger.info(f"Combined dataset shape: {df.shape}")

    # 3. Prepare Tensors
    X_raw = df.select(feature_cols).to_numpy().astype(np.float32)
    y_raw = df.select('target').to_numpy().flatten().astype(np.int64)
    
    seq_len = config['hyperparameters']['seq_len']
    logger.info(f"Creating sequences with seq_len={seq_len}...")
    X_seq, y_seq = create_sequences(X_raw, y_raw, seq_len)
    
    # 4. Model Setup
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Unified device: {DEVICE}")
    
    model = QuantGodModel(
        num_features=len(feature_cols),
        seq_len=seq_len,
        d_model=config['hyperparameters']['d_model'],
        nhead=config['hyperparameters']['nhead'],
        num_layers=3, 
        num_classes=3,
        dropout=config['hyperparameters']['dropout']
    ).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['hyperparameters']['lr'])
    
    # DataLoader
    dataset = TensorDataset(torch.from_numpy(X_seq), torch.from_numpy(y_seq))
    train_loader = DataLoader(dataset, batch_size=config['hyperparameters']['batch_size'], shuffle=True)
    
    # 5. Training Loop (Single Phase for Cloud POC)
    epochs = config['hyperparameters']['epochs']
    logger.info(f"Starting training for {epochs} epochs...")
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")

    # 6. Save Model
    output_path = Path(config['paths']['model_output'])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    logger.info(f"Model saved to {output_path}")

if __name__ == "__main__":
    run_training()
