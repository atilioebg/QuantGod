import torch
import torch.nn as nn

class SAIMPViViT(nn.Module):
    def __init__(self, seq_len=32, input_channels=4, price_levels=128, d_model=128, nhead=4, num_layers=2, num_classes=3):
        super().__init__()
        
        self.seq_len = seq_len
        self.price_levels = price_levels
        
        # 1. Feature Extraction (CNN Espacial)
        # Entra: [Batch, Channels, Height, Width] -> [Batch, 4, 128, 1] (considerando cada time step como width=1 na entrada bruta, mas aqui processamos diferente)
        # Na verdade, o input do ViViT costuma ser [Batch, Seq, Channels, Height]
        
        # Camada de Embedding para projetar a dimensão de preço (128) para d_model
        # Vamos usar uma CNN 1D para processar a coluna de preço de cada frame
        self.price_encoder = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2), # 128 -> 64
            nn.Conv1d(in_channels=64, out_channels=d_model, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)  # 64 -> 32 (Dimensão espacial reduzida)
        )
        
        # Agora temos features espaciais. Precisamos achatar para entrar no Transformer
        # Feature size atual: 32 * d_model. Vamos projetar para d_model.
        self.feature_projection = nn.Linear(32 * d_model, d_model)
        
        # 2. Positional Encoding (Temporal)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model))
        
        # 3. Transformer Encoder (Temporal)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Head de Classificação
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: [Batch, Seq_Len, Channels, Price_Levels]
        b, s, c, h = x.shape
        
        # Precisamos processar cada passo de tempo (s) independentemente na CNN
        # Merge Batch e Seq: [B*S, C, H]
        x = x.view(b * s, c, h)
        
        # Passa pela CNN
        x = self.price_encoder(x) # [B*S, d_model, H/4] -> [B*S, 128, 32]
        
        # Flatten espacial
        x = x.view(b, s, -1) # [B, S, 128*32]
        
        # Projeta para d_model
        x = self.feature_projection(x) # [B, S, 128]
        
        # Adiciona Posição
        x = x + self.pos_embedding[:, :s, :]
        
        # Transformer
        x = self.transformer(x) # [B, S, 128]
        
        # Pega apenas o último estado (último time step) para prever o futuro
        last_state = x[:, -1, :] # [B, 128]
        
        # Classifica
        logits = self.classifier(last_state)
        
        return logits
