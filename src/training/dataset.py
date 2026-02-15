import torch
from torch.utils.data import Dataset
import polars as pl
import numpy as np
import os
from pathlib import Path
from src.processing.tensor_builder import build_tensor_4d

class QuantGodDataset(Dataset):
    """
    Dataset PyTorch que carrega dados de Simulação (Parquet) e gera Tensores 4D on-the-fly via TensorBuilder.
    Sincroniza com Labels pré-calculados via Timestamp.
    """
    def __init__(self, tensor_files, labels_df: pl.DataFrame):
        """
        Args:
            tensor_files (list): Lista de caminhos para arquivos .parquet (Output do Simulation).
            labels_df (pl.DataFrame): DataFrame com colunas ['timestamp', 'label'].
        """
        self.tensor_files = [str(f) for f in tensor_files]
        self.labels = labels_df
        
        # Carregar metadados dos arquivos de simulação para indexação
        # Precisamos saber quais snapshots tem em cada arquivo
        # Para performance, vamos assumir que cabe na RAM carregar os metadados (timestamp) de todos
        # Se for muito grande, usaríamos indexação em disco.
        
        # Unificar todos os "profile/simulated books" em um LazyFrame e Join com Labels
        # Isso cria o mapa final de (Snapshot -> Label).
        
        print(f"QuantGodDataset: Indexando {len(tensor_files)} arquivos...")
        
        try:
             # Ler todos os arquivos de simulação (profiles)
            lf_sim = pl.scan_parquet(self.tensor_files)
            
            # Precisamos apenas das colunas necessárias para o TensorBuilder + Timestamp para join
            # Simulation cols: snapshot_time, price, bid_vol, ask_vol, trade_count, ofi_level...
            
            # Join com Labels
            # Labels tem timestamp, Label.
            # Simulation tem snapshot_time.
            # Join key: snapshot_time == timestamp
            
            self.data = (
                lf_sim
                .join(self.labels.lazy(), left_on="snapshot_time", right_on="timestamp", how="inner")
                .collect() # Traz para memória (dataset deve caber na RAM nesta fase)
            )
            
            # Agora 'self.data' é um DataFrame gigante com todo o histórico flat.
            # Mas o TensorBuilder espera um DataFrame representando O snapshot (várias linhas por snapshot).
            # Se 'self.data' for flat (linhas de preço), __getitem__ precisa filtrar o snapshot.
            
            # Problema: Filtrar DF gigante a cada getitem é lento se não particionado.
            # Solução Otimizada:
            # partition_by("snapshot_time") cria uma lista de DataFrames pequenos.
            # Isso é perfeito para Dataset.
            
            print("QuantGodDataset: Particionando Snapshots...")
            self.snapshots = self.data.partition_by("snapshot_time", maintain_order=True)
            
            # Verificar integridade
            valid_snapshots = []
            for snap in self.snapshots:
                # Checa se tem label (já fizemos inner join, então deve ter)
                if snap.height > 0:
                     valid_snapshots.append(snap)
            self.snapshots = valid_snapshots
            
            print(f"QuantGodDataset: Pronta com {len(self.snapshots)} amostras.")
            
        except Exception as e:
            print(f"Erro ao inicializar Dataset: {e}")
            self.snapshots = []

    def __len__(self):
        return len(self.snapshots)

    def __getitem__(self, idx):
        # 1. Pegar o DataFrame do Snapshot
        snapshot_df = self.snapshots[idx]
        
        # 2. Extrair Label (assumindo que é constante para o snapshot)
        # Pega da primeira linha
        label = snapshot_df["label"][0]
        
        # 3. Construir Tensor
        # build_tensor_4d espera um DF com estrutura canônica da simulação.
        # Ele retorna (1, 4, 128) pois processa lista.
        # Nós queremos (4, 128).
        
        tensor_batch = build_tensor_4d(snapshot_df, n_levels=128, is_simulation=True)
        
        # Remove dimensão batch (T=1)
        tensor = tensor_batch[0] 
        
        # Converter para PyTorch
        tensor_torch = torch.from_numpy(tensor).float() # (4, 128)
        label_torch = torch.tensor(label, dtype=torch.long)
        
        return tensor_torch, label_torch
