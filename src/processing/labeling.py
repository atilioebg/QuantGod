from datetime import timedelta
import numpy as np
import polars as pl

def generate_hierarchical_labels(klines_df: pl.DataFrame, window_hours: int = 2, target_pct: float = 0.008, stop_pct: float = 0.0075) -> pl.DataFrame:
    """
    Gera rótulos hierárquicos (Meta-Labeling):
    1. STOP (Label 1): -0.75% atingido.
    2. SUPER LONG (Label 3): +0.8% atingido E depois +1.6% atingido.
    3. LONG (Label 2): +0.8% atingido.
    4. NEUTRO (Label 0).
    """
    print(f"   🏷️ Gerando labels HIERÁRQUICOS (Janela: {window_hours}h)...")
    
    df = klines_df.sort("timestamp")
    timestamps = df["timestamp"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    
    n = len(df)
    labels = np.zeros(n, dtype=np.int32)
    
    # Pré-cálculo de índices de fim de janela
    is_datetime = np.issubdtype(timestamps.dtype, np.datetime64)
    if is_datetime:
        window_duration = np.timedelta64(int(window_hours * 3600 * 1e9), 'ns')
        future_times = timestamps + window_duration
    else:
        window_duration_ms = int(window_hours * 3600 * 1000)
        future_times = timestamps + window_duration_ms

    end_indices = np.searchsorted(timestamps, future_times, side='right')
    
    for i in range(n):
        current_close = closes[i]
        end_idx = end_indices[i]
        
        if end_idx <= i + 1:
            labels[i] = 0
            continue
            
        w_highs = highs[i+1 : end_idx]
        w_lows = lows[i+1 : end_idx]
        
        # Barreiras
        stop_price = current_close * (1 - stop_pct)
        target_1 = current_close * (1 + target_pct)
        target_2 = current_close * (1 + 2 * target_pct) # +1.6%
        
        # 1. Prioridade: STOP (Tocou em qualquer momento antes ou junto com targets?)
        hit_stop_mask = w_lows <= stop_price
        hit_target1_mask = w_highs >= target_1
        
        if hit_stop_mask.any():
            if not hit_target1_mask.any():
                labels[i] = 1 # Apenas stop
                continue
            else:
                # Verificamos quem veio primeiro
                first_stop = np.argmax(hit_stop_mask)
                first_t1 = np.argmax(hit_target1_mask)
                if first_stop <= first_t1:
                    labels[i] = 1 # Stop primeiro ou no mesmo candle
                    continue
        
        # Se chegou aqui, não houve stop antes do Target 1
        if hit_target1_mask.any():
            # Houve alvo 1. Verificamos Alvo 2 (Super Long)
            first_t1_idx = np.argmax(hit_target1_mask)
            
            # Sub-janela após T1
            if first_t1_idx + 1 < len(w_highs):
                w_highs_after_t1 = w_highs[first_t1_idx + 1:]
                if (w_highs_after_t1 >= target_2).any():
                    labels[i] = 3 # Super Long
                else:
                    labels[i] = 2 # Long comum
            else:
                labels[i] = 2 # Acabou a janela no T1
        else:
            labels[i] = 0 # Neutro
            
        if (i + 1) % 50000 == 0:
             print(f"         Progresso: {i+1:,}/{n:,}", end="\r")

    print(f"         Progresso: {n:,}/{n:,} ✅")
    
    # Distribuição
    unique, counts = np.unique(labels, return_counts=True)
    class_names = {0: "Neutro", 1: "STOP", 2: "LONG", 3: "SUPER LONG"}
    for u, c in zip(unique, counts):
        pct = 100 * c / n
        print(f"         {class_names.get(u)}: {c:,} ({pct:.2f}%)")
    
    return df.select([
        pl.col("timestamp"),
        pl.col("close").alias("close_price"),
    ]).with_columns([
        pl.Series(name="label", values=labels)
    ])
