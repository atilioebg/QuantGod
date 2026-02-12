import torch
import numpy as np
from pathlib import Path
import sys
import polars as pl

# Add project root to sys.path
sys.path.append(str(Path.cwd()))

from src.processing.simulation import build_simulated_book
from src.processing.tensor_builder import build_tensor_4d
from src.models.vivit import SAIMPViViT
from src.config import settings

class SniperBrain:
    def __init__(self, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SAIMPViViT(seq_len=32, input_channels=4, price_levels=128, num_classes=3)
        
        # Default path from settings if none provided
        if model_path is None:
            model_path = settings.DATA_DIR / "saimp_best.pth"
            
        path_obj = Path(model_path)
        if path_obj.exists():
            print(f"🧠 Carregando modelo de: {path_obj}")
            self.model.load_state_dict(torch.load(path_obj, map_location=self.device, weights_only=True))
            self.model.to(self.device)
            self.model.eval()
        else:
            raise FileNotFoundError(f"Modelo não encontrado: {path_obj}")

    def analyze(self, df_trades):
        if df_trades is None or df_trades.height < 100: 
            return None
        
        # Simulação e Tensor
        # Requer pelo menos 32 snapshots de 15m (8 horas) para contexto
        # Se df for curto (ex: warm-up incompleto), build_simulated_book pode gerar menos rows
        
        try:
            sim_book = build_simulated_book(df_trades, window="15m")
        except Exception as e:
            print(f"Erro na simulação: {e}")
            return None

        if sim_book.height == 0:
            return None

        # Build Tensor
        # Nota: build_tensor_4d JÁ APLICA NORMALIZAÇÃO (div 10, tanh, clip)
        tensor = build_tensor_4d(sim_book, n_levels=128, is_simulation=True)
        
        if len(tensor) < 32: 
            return None
            
        # Pega última sequência válida
        input_seq = tensor[-32:]
        
        # NÃO RE-APLICAR NORMALIZAÇÃO AQUI (redundância removida)
        
        # Inferência
        input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = torch.softmax(logits, dim=1)
            conf, pred = torch.max(probs, 1)
        
        # Lógica de Barreiras (OFI analysis)
        # Pega apenas o último snapshot para calcular OFI total
        last_snap_time = sim_book["snapshot_time"].max()
        last_rows = sim_book.filter(pl.col("snapshot_time") == last_snap_time)
        
        # Soma OFI_level de todos os price levels ativos
        ofi = last_rows["ofi_level"].sum()
        
        # Preço atual (último trade)
        last_price = df_trades["price"][-1]
        
        trend = "Alta" if ofi > 0 else "Baixa"
        
        return {
            "signal": pred.item(), # 0=Neutro, 1=Venda, 2=Compra
            "confidence": conf.item(),
            "price": last_price,
            "ofi": ofi,
            "trend_intent": trend
        }
