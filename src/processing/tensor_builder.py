import polars as pl
import numpy as np
import pywt

def apply_wavelet_denoising(signal: np.ndarray, wavelet: str = 'db4', level: int = 1) -> np.ndarray:
    """
    Aplica denoising via Wavelet (Soft Thresholding).
    Retorna a aproximação (sinal limpo).
    """
    if len(signal) < 2:
        return signal
    
    # Ensure signal is writable (Polars to_numpy can be read-only)
    if not signal.flags.writeable:
        signal = signal.copy()
    
    # Decomposição
    coeffs = pywt.wavedec(signal, wavelet, mode='per')
    
    # Thresholding nos detalhes (coeffs[1:])
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))
    
    new_coeffs = [coeffs[0]]
    for i in range(1, len(coeffs)):
        new_coeffs.append(pywt.threshold(coeffs[i], threshold, mode='soft'))
        
    # Reconstrução
    denoised = pywt.waverec(new_coeffs, wavelet, mode='per')
    
    # Ajuste de tamanho caso waverec retorne 1 elemento a mais
    return denoised[:len(signal)]

def build_tensor_6d(data_df: pl.DataFrame, n_levels: int = 128, is_simulation: bool = False) -> np.ndarray:
    """
    Gera Tensor (Time, 6, Height)
    Canais:
    0: Bids (Liquidez Compra)
    1: Asks (Liquidez Venda)
    2: OFI Raw (Fluxo Bruto)
    3: Price Raw (Preço Bruto)
    4: OFI Wavelet (Fluxo Estrutural/Limpo)
    5: Price Wavelet (Preço Estrutural/Limpo)
    """
    if data_df.height == 0:
        return np.zeros((0, 6, n_levels), dtype=np.float32)

    tensor_list = []
    
    if is_simulation:
        snapshots = data_df.partition_by("snapshot_time", maintain_order=True)
        
        for snap in snapshots:
            if snap.height == 0:
                continue
                
            snap = snap.sort("price")
            rows = snap.head(n_levels)
            limit = len(rows)
            
            # Inicializa canais
            chs = np.zeros((6, n_levels), dtype=np.float32)
            
            # 1. Bids/Asks (Log Liquidity)
            chs[0, :limit] = np.log1p(rows["bid_vol"].to_numpy())
            chs[1, :limit] = np.log1p(rows["ask_vol"].to_numpy())
            
            # 2. OFI Raw
            ofi_raw = rows["ofi_level"].to_numpy()
            chs[2, :limit] = ofi_raw
            
            # 3. Price Raw (Normalizado localmente para o tensor)
            # Como o tensor foca na vizinhança, usamos o desvio relativo ao preço médio do snap
            prices = rows["price"].to_numpy()
            avg_price = prices.mean()
            price_rel = (prices - avg_price) / (avg_price + 1e-9)
            chs[3, :limit] = price_rel
            
            # 4. OFI Wavelet
            chs[4, :limit] = apply_wavelet_denoising(ofi_raw)
            
            # 5. Price Wavelet
            chs[5, :limit] = apply_wavelet_denoising(price_rel)
            
            tensor_list.append(chs)

    else:
        # Stream Real
        for row in data_df.iter_rows(named=True):
            try:
                bids = np.array(row['bids'], dtype=np.float32) if row['bids'] else np.empty((0,2), dtype=np.float32)
                asks = np.array(row['asks'], dtype=np.float32) if row['asks'] else np.empty((0,2), dtype=np.float32)
            except:
                bids = np.empty((0,2), dtype=np.float32)
                asks = np.empty((0,2), dtype=np.float32)
            
            chs = np.zeros((6, n_levels), dtype=np.float32)
            
            if len(bids) > 0:
                limit = min(len(bids), n_levels)
                chs[0, :limit] = np.log1p(bids[:limit, 1])
            
            if len(asks) > 0:
                limit = min(len(asks), n_levels)
                chs[1, :limit] = np.log1p(asks[:limit, 1])
                
            # Nota: OFI e Price em Stream requerem lógica de agregação prévia
            # Por enquanto, preenchemos com 0 se não houver dados, aguardando processor.py
            
            tensor_list.append(chs)

    tensor = np.array(tensor_list, dtype=np.float32)
    
    if tensor.shape[0] > 0:
        # Normalização específica por canal
        tensor[:, 0, :] /= 10.0 # Bids
        tensor[:, 1, :] /= 10.0 # Asks
        
        # OFI Raw & Wavelet (2 e 4) -> Tanh compression
        tensor[:, 2, :] = np.tanh(tensor[:, 2, :] / 10.0)
        tensor[:, 4, :] = np.tanh(tensor[:, 4, :] / 10.0)
        
        # Price Raw & Wavelet (3 e 5) -> Scale (já estão em retorno relativo)
        tensor[:, 3, :] *= 100.0
        tensor[:, 5, :] *= 100.0
        
        tensor = np.clip(tensor, -1.0, 1.0)

    return tensor
