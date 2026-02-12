from datetime import timedelta
import numpy as np
import polars as pl

def generate_labels(klines_df: pl.DataFrame, window_hours: int = 6, target_pct: float = 0.008, stop_pct: float = 0.004) -> pl.DataFrame:
    """
    Gera rótulos baseados no Triple Barrier Method - VERSÃO OTIMIZADA (NumPy).
    
    Otimizações:
    - Uso de np.searchsorted para evitar máscaras booleanas no loop (O(N) vs O(N^2)).
    - Lógica de "First Touch" correta.
    
    Retorna DataFrame com: [timestamp, label, close_price]
    
    Labels:
    - 0: Neutro
    - 1: Venda/Stop
    - 2: Compra/Alvo
    """
    print(f"   🏷️ Gerando labels com janela temporal de {window_hours}h (Otimizado)...")
    
    # Garante ordenação
    df = klines_df.sort("timestamp")
    
    # Arrays numpy
    timestamps = df["timestamp"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    
    n = len(df)
    labels = np.zeros(n, dtype=np.int32)
    
    # 1. Pré-calcular os índices de fim de janela para todos os pontos
    # window_duration_ns se timestamps forem datetime64[ns]
    # Se timestamps forem int64 (ms), converter hrs para ms
    
    # Verificação de tipo
    is_datetime = np.issubdtype(timestamps.dtype, np.datetime64)
    if is_datetime:
        window_duration = np.timedelta64(int(window_hours * 3600 * 1e9), 'ns')
        future_times = timestamps + window_duration
    else:
        # Assumindo ms (epoch)
        window_duration_ms = int(window_hours * 3600 * 1000)
        future_times = timestamps + window_duration_ms

    # searchsorted retorna o índice de inserção.
    # timestamps está ordenado. Encontramos onde (t + window) se encaixa.
    # side='right' -> indice logo após o fim da janela
    end_indices = np.searchsorted(timestamps, future_times, side='right')
    
    print(f"      Processando {n:,} candles...")
    
    # Loop otimizado
    for i in range(n):
        current_close = closes[i]
        end_idx = end_indices[i]
        
        # Se não há dados futuros suficientes na janela
        if end_idx <= i + 1:
            labels[i] = 0
            continue
            
        # Slicing é muito rápido
        window_highs = highs[i+1 : end_idx]
        window_lows = lows[i+1 : end_idx]
        
        if len(window_highs) == 0:
            labels[i] = 0
            continue
            
        # Calcula barreiras
        target_price = current_close * (1 + target_pct)
        stop_price = current_close * (1 - stop_pct)
        
        # Verifica se tocou (vetorizado na janela)
        # Queremos o PRIMEIRO toque.
        # np.argmax retorna o primeiro índice do máximo (True).
        
        hit_target_mask = window_highs >= target_price
        hit_stop_mask = window_lows <= stop_price
        
        has_hit_target = hit_target_mask.any()
        has_hit_stop = hit_stop_mask.any()
        
        if has_hit_target and not has_hit_stop:
            labels[i] = 2
        elif has_hit_stop and not has_hit_target:
            labels[i] = 1
        elif has_hit_target and has_hit_stop:
            # Ambos atingidos, verificar qual foi primeiro
            first_target_idx = np.argmax(hit_target_mask)
            first_stop_idx = np.argmax(hit_stop_mask)
            
            if first_stop_idx < first_target_idx:
                labels[i] = 1 # Stop veio primeiro
            else:
                labels[i] = 2 # Alvo veio primeiro (ou empate, prioriza target?)
                # Empate no mesmo candle (idx igual):
                # Se low <= stop e high >= target no mesmo candle.
                # Assumimos STOP para ser conservador no pior caso (wicks grandes)
                if first_stop_idx == first_target_idx:
                     labels[i] = 1
        else:
            labels[i] = 0
            
        if (i + 1) % 50000 == 0:
             print(f"         Progresso: {i+1:,}/{n:,} ({100*(i+1)/n:.1f}%)", end="\r")

    print(f"         Progresso: {n:,}/{n:,} (100.0%) ✅")
    
    # Análise de distribuição
    unique, counts = np.unique(labels, return_counts=True)
    print(f"      📊 Distribuição de Labels:")
    class_names = {0: "Neutro", 1: "Venda/Stop", 2: "Compra/Alvo"}
    for u, c in zip(unique, counts):
        pct = 100 * c / n
        class_name = class_names.get(u, f"Classe {u}")
        emoji = "⚪" if u == 0 else ("🔴" if u == 1 else "🟢")
        print(f"         {emoji} {class_name}: {c:,} ({pct:.2f}%)")
    
    return df.select([
        pl.col("timestamp"),
        pl.col("close").alias("close_price"),
    ]).with_columns([
        pl.Series(name="label", values=labels)
    ])