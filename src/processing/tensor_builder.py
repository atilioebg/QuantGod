import polars as pl
import numpy as np

def build_tensor_4d(data_df: pl.DataFrame, n_levels: int = 128, is_simulation: bool = False) -> np.ndarray:
    """
    Gera Tensor (Time, 4, Height)
    Canais:
    0: Bid Liquidity (Log Volume)
    1: Ask Liquidity (Log Volume)
    2: OFI (Order Flow Imbalance)
    3: Activity (Trade Count / Volatility)
    """
    if data_df.height == 0:
        return np.zeros((0, 4, n_levels), dtype=np.float32)

    tensor_list = []
    
    # Lógica para Simulação (Dados Históricos)
    if is_simulation:
        # Agrupa por snapshot_time
        # partition_by requer Polars mais recente.
        snapshots = data_df.partition_by("snapshot_time", maintain_order=True)
        
        for snap in snapshots:
            # Pega o preço de referência (fechamento ou médio da janela)
            # Para simplificar, pegamos o preço onde houve mais volume/atividade
            if snap.height == 0:
                continue
                
            # Identifica preço de referência (ex: preço mais negociado)
            ref_row = snap.sort("trade_count", descending=True).head(1)
            if ref_row.height == 0:
                continue
                
            ref_price = ref_row.select("price").item()
            
            # Filtra range de preço (+/- 10% ou fixo em levels)
            # Aqui simplificamos: pegamos os 128 níveis mais ativos ao redor do preço?
            # O código original dizia "sort by price" e pegava head(n_levels).
            # Isso pega os MENORES preços se não invertermos?
            # Vamos ordenar por preço.
            snap = snap.sort("price")
            
            # Inicializa canais
            ch_bid = np.zeros(n_levels, dtype=np.float32)
            ch_ask = np.zeros(n_levels, dtype=np.float32)
            ch_ofi = np.zeros(n_levels, dtype=np.float32)
            ch_act = np.zeros(n_levels, dtype=np.float32)
            
            # Preenchimento simples
            # Vamos pegar os n_levels disponíveis no snapshot. 
            # (Em produção idealmente centralizaria no ref_price)
            
            rows = snap.head(n_levels)
            
            # Extrai arrays com to_numpy()
            vals_bid = np.log1p(rows["bid_vol"].to_numpy())
            vals_ask = np.log1p(rows["ask_vol"].to_numpy())
            vals_ofi = rows["ofi_level"].to_numpy() # Não usa log no OFI pois pode ser negativo
            vals_act = np.log1p(rows["trade_count"].to_numpy())
            
            limit = len(rows)
            ch_bid[:limit] = vals_bid
            ch_ask[:limit] = vals_ask
            ch_ofi[:limit] = vals_ofi
            ch_act[:limit] = vals_act
            
            tensor_list.append(np.stack([ch_bid, ch_ask, ch_ofi, ch_act]))

    # Lógica para Stream Real (Depth Snapshots)
    else:
        # Mantém a lógica anterior, mas adiciona canais vazios ou derivados
        for row in data_df.iter_rows(named=True):
            # Bids/Asks vêm como listas de [price, qty]
            try:
                bids = np.array(row['bids'], dtype=np.float32) if row['bids'] else np.empty((0,2), dtype=np.float32)
                asks = np.array(row['asks'], dtype=np.float32) if row['asks'] else np.empty((0,2), dtype=np.float32)
            except:
                bids = np.empty((0,2), dtype=np.float32)
                asks = np.empty((0,2), dtype=np.float32)
            
            ch_bid = np.zeros(n_levels, dtype=np.float32)
            ch_ask = np.zeros(n_levels, dtype=np.float32)
            ch_ofi = np.zeros(n_levels, dtype=np.float32) # Stream simples não tem OFI por nível ainda
            ch_act = np.zeros(n_levels, dtype=np.float32) # Stream simples não tem activity por nível ainda

            if len(bids) > 0:
                limit = min(len(bids), n_levels)
                # Bids usually sorted desc in order book logic, but tensor implies spatial logic?
                # Let's assume input comes sorted appropriately or we sort if needed.
                # Assuming raw lists: usually sorted.
                ch_bid[:limit] = np.log1p(bids[:limit, 1])
            
            if len(asks) > 0:
                limit = min(len(asks), n_levels)
                ch_ask[:limit] = np.log1p(asks[:limit, 1])
                
            tensor_list.append(np.stack([ch_bid, ch_ask, ch_ofi, ch_act]))

    # Final Normalization & Clipping
    tensor = np.array(tensor_list, dtype=np.float32)
    
    if tensor.shape[0] > 0:
        # Channels 0 (Bid), 1 (Ask), 3 (Activity) -> Log values, scale by 0.1
        tensor[:, 0, :] /= 10.0
        tensor[:, 1, :] /= 10.0
        tensor[:, 3, :] /= 10.0
        
        # Channel 2 (OFI) -> Tanh compression
        tensor[:, 2, :] = np.tanh(tensor[:, 2, :] / 10.0)
        
        # Global Clip Security
        tensor = np.clip(tensor, -1.0, 1.0)

    return tensor
