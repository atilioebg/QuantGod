import torch
import torch.nn as nn
import math

class SpatialFeatureExtractor(nn.Module):
    """
    O 'Olho' da IA.
    Analisa um único snapshot (6 canais x 128 níveis) e extrai um vetor de características.
    Usa Convoluções 1D ao longo do eixo do preço.
    """
    def __init__(self, input_channels=6, d_model=128):
        super().__init__()
        self.conv_layers = nn.Sequential(
            # Input: (Batch, 6, 128)
            nn.Conv1d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2), # (Batch, 32, 64)
            
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2), # (Batch, 64, 32)
            
            nn.Conv1d(64, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1) # (Batch, d_model, 1) -> Esmaga a dimensão espacial restante
        )

    def forward(self, x):
        # x shape: (Batch * Time, 6, 128) -> Processamos todos os frames juntos
        x = self.conv_layers(x)
        return x.squeeze(-1) # (Batch * Time, d_model)

class QuantGodViViT(nn.Module):
    """
    Video Vision Transformer para Swing Trade.
    """
    def __init__(self, seq_len=96, input_channels=6, price_levels=128, d_model=128, nhead=4, num_layers=2, num_classes=4, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        
        # 1. Extrator Espacial (CNN)
        self.spatial_encoder = SpatialFeatureExtractor(input_channels, d_model)
        
        # 2. Positional Encoding (Para o Transformer saber a ordem do tempo)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model))
        
        # 3. Encoder Temporal (Transformer)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=512, batch_first=True, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Cabeça de Classificação
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes) # 4 Classes: 0(Neutral), 1(Stop), 2(Long), 3(Super Long)
        )

    def forward(self, x):
        # Input x: (Batch, Time, Channels, Height) ex: (32, 96, 4, 128)
        b, t, c, h = x.shape
        
        # Achatar Batch e Time para passar na CNN
        # (B*T, C, H)
        x_flat = x.view(b * t, c, h)
        
        # Passar pela CNN
        spatial_feats = self.spatial_encoder(x_flat) # (B*T, d_model)
        
        # Desachatar de volta para sequência temporal
        # (B, T, d_model)
        sequence = spatial_feats.view(b, t, self.d_model)
        
        # Adicionar Posição
        # Se t < seq_len (batch menor), corta pos_emb. Se t > seq_len, erro (ou interpolar).
        # Assumimos t <= seq_len
        if t > self.pos_embedding.shape[1]:
             raise ValueError(f"Input sequence length {t} exceeds max sequence length {self.pos_embedding.shape[1]}")
             
        sequence = sequence + self.pos_embedding[:, :t, :]
        
        # Passar pelo Transformer
        transformer_out = self.transformer_encoder(sequence)
        
        # Pegar apenas o último estado (último frame da sequência) para decisão
        last_state = transformer_out[:, -1, :]
        
        # Classificar
        logits = self.classifier(last_state)
        
        return logits