import torch
from torch.utils.data import IterableDataset
import polars as pl
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import gc
import sys

# Project imports
from src.config import settings
from src.processing.simulation import build_simulated_book
from src.processing.tensor_builder import build_tensor_6d
from src.processing.labeling import generate_hierarchical_labels

class StreamingDataset(IterableDataset):
    """
    Dataset que carrega dados em chunks (ex: dias) para evitar OOM.
    """
    def __init__(self, months: list[str], seq_len: int = 96):
        """
        Args:
            months: Lista de meses no formato "YYYY-MM"
            seq_len: Comprimento da sequência (ex: 96 para 24h)
        """
        self.months = months
        self.seq_len = seq_len
        self.start_date = None
        self.end_date = None
        self.files_map = self._map_files(months)

    def set_date_range(self, start_date: str = None, end_date: str = None):
        """
        Define range de datas para filtrar o dataset.
        Args:
            start_date: "YYYY-MM-DD"
            end_date: "YYYY-MM-DD"
        """
        if start_date:
            self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        if end_date:
            self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
        
    def _map_files(self, months):
        """Mapeia arquivos disponíveis para os meses solicitados."""
        files = []
        for m in months:
            t_file = settings.RAW_HISTORICAL_DIR / f"aggTrades_{m}.parquet"
            k_file = settings.RAW_HISTORICAL_DIR / f"klines_{m}.parquet"
            if t_file.exists() and k_file.exists():
                files.append({"month": m, "trades": t_file, "klines": k_file})
            else:
                print(f"[WARN] Arquivos não encontrados para {m}")
        return files

    def _process_daily_chunk(self, date, df_trades_month, df_klines_month):
        """
        Processa um dia específico com buffer.
        """
        # Calcular range do dia + buffer
        # Buffer: precisamos de dados antes (para book building) e depois (para labels)
        # Book building precisa de histórico recente? Sim, para estado acumulado.
        # Labels precisam de futuro (Triple Barrier).
        
        # Simplificação V1: Processar o MÊS inteiro de uma vez se couber, 
        # mas iterar gerando tensores.
        # SE o mês não couber, teríamos que filtrar o Polars DataFrame por datas.
        
        # Vamos tentar filtrar o LazyFrame ou DataFrame por dia.
        
        start_ts = date
        end_ts = date + timedelta(days=1)
        
        # Buffers
        # Label precisa de ~6h futuro
        buffer_future = timedelta(hours=8) 
        # Simulação precisa de histórico? O estado do book é reconstruído do zero a cada chunk?
        # Se reconstruirmos do zero, perdemos o estado anterior.
        # Idealmente, o estado deveria persistir.
        # Para simplificar a V1 e evitar complexidade de estado entre chunks:
        # Vamos carregar um buffer de passado razoável (ex: 30h) para garantir sequencia.
        # Seq_Len = 96 (24h). Precisamos de pelo menos 24h de histórico.
        buffer_past = timedelta(hours=30)
        
        chunk_start = start_ts - buffer_past
        chunk_end = end_ts + buffer_future
        
        # Filtrar dados (Assumindo que df_trades_month já tem colunas de timestamp compatíveis)
        # Se df valer apenas para o mês, filtrar por data
        
        # ... A lógica de carregar o mês todo e fltrar dia a dia pode ser lenta se for eager.
        # Mas carregar parquet por filtro de linha é complexo sem particionamento hive.
        
        # Vamos assumir a estratégia:
        # Carregar Trades do Mês (Lazy/Scan) -> Filter por Dia -> Collect -> Build Tensor -> Yield -> Free
        pass

    def __iter__(self):
        """
        Iterador principal.
        """
        worker_info = torch.utils.data.get_worker_info()
        
        # Se tiver multiplos workers, teríamos que dividir os arquivos entre eles.
        # Por enquanto, assumimos num_workers=0 ou 1.
        
        for file_info in self.files_map:
            month = file_info['month']
            processed_path = settings.PROCESSED_DIR / "tensors" / f"processed_{month}.npz"
            
            # --- CENÁRIO A: USAR DADOS PRÉ-PROCESSADOS (ULTRA FAST) ---
            if processed_path.exists():
                print(f"\n[STREAM] Carregando Tensores Pré-Processados: {month}...", flush=True)
                try:
                    data = np.load(processed_path)
                    X_np = data['x']
                    Y_np = data['y']
                    # Timestamps vêm como string ou datetime opcional
                    timestamps_raw = data['timestamps']
                    
                    # Converter timestamps para datetime para filtrar range
                    # (Assume-se formato ISO ou similar salvo no precompute)
                    times = [datetime.fromisoformat(t) if isinstance(t, str) else t for t in timestamps_raw]
                    
                    yield_count = 0
                    if len(X_np) > self.seq_len:
                        for i in range(len(X_np) - self.seq_len):
                            target_idx = i + self.seq_len - 1
                            target_time = times[target_idx]
                            
                            # Filtro de Data opcional (set_date_range)
                            if (not self.start_date or target_time >= self.start_date) and \
                               (not self.end_date or target_time < self.end_date):
                                
                                x_seq = X_np[i : i+self.seq_len]
                                y_seq = Y_np[target_idx]
                                
                                yield (
                                    torch.tensor(x_seq, dtype=torch.float32),
                                    torch.tensor(y_seq, dtype=torch.long)
                                )
                                yield_count += 1
                    
                    print(f"      -> Yielded {yield_count} sequences (Cached) for {month}", flush=True)
                    del X_np, Y_np, data
                    gc.collect()
                    continue # Próximo mês
                except Exception as e:
                    print(f"   [WARN] Falha ao carregar cache {month}: {e}. Tentando Processamento on-the-fly...")

            # --- CENÁRIO B: PROCESSAMENTO ON-THE-FLY (FALLBACK) ---
            print(f"\n[STREAM] Carregando mês {month} (On-the-fly)...", flush=True)
            
            try:
                # 1. Identificar dias únicos no mês usando Scan (leve)
                lf_klines = pl.scan_parquet(file_info['klines'])
                
                # Converter open_time se necessário
                # Assumindo estrutura padrão
                # Pegar min/max timestamps para iterar dias
                stats = lf_klines.select([
                    pl.from_epoch(pl.col("open_time"), time_unit="ms").min().alias("min_time"),
                    pl.from_epoch(pl.col("open_time"), time_unit="ms").max().alias("max_time")
                ]).collect()
                
                min_date = stats["min_time"][0].replace(hour=0, minute=0, second=0, microsecond=0)
                max_date = stats["max_time"][0].replace(hour=0, minute=0, second=0, microsecond=0)
                
                # Apply optional date filtering
                if self.start_date and min_date < self.start_date:
                    min_date = self.start_date
                if self.end_date and max_date > self.end_date:
                    max_date = self.end_date
                
                if min_date > max_date:
                    print(f"[WARN] Intervalo de datas invalido ou fora do arquivo: {min_date} > {max_date}")
                    continue

                # Iterar dia a dia
                
                # Iterar dia a dia
                current_date = min_date
                while current_date <= max_date:
                    next_date = current_date + timedelta(days=1)
                    
                    print(f"[STREAM] Processando dia {current_date.date()}...", flush=True)
                    
                    # Definir janelas com buffer DINÂMICO
                    # Buffer Past: SEQ_LEN * 15min + Margem (ex: 2h)
                    # Buffer Future: LABEL_WINDOW + Margem (ex: 2h)
                    
                    past_hours = (self.seq_len * 15 / 60) + 2
                    future_hours = settings.LABEL_WINDOW_HOURS + 2
                    
                    t_start = current_date - timedelta(hours=past_hours)
                    t_end = next_date + timedelta(hours=future_hours)
                    
                    # Carregar dados DO CHUNK (Eager load apenas do pedaço)
                    # Trades
                    q_trades = (
                        pl.scan_parquet(file_info['trades'])
                        .filter(
                            (pl.col("transact_time") >= int(t_start.timestamp()*1000)) &
                            (pl.col("transact_time") < int(t_end.timestamp()*1000))
                        )
                    )
                    
                    # Klines
                    q_klines = (
                        pl.scan_parquet(file_info['klines'])
                        .filter(
                            (pl.col("open_time") >= int(t_start.timestamp()*1000)) &
                            (pl.col("open_time") < int(t_end.timestamp()*1000))
                        )
                    )
                    
                    try:
                        df_trades = q_trades.collect()
                        df_klines = q_klines.collect()
                    except Exception as e:
                        print(f"   [WARN] Falha ao ler chunk: {e}")
                        current_date = next_date
                        continue
                        
                    if df_trades.height == 0 or df_klines.height == 0:
                        current_date = next_date
                        continue
                    
                    # ---- PROCESSAMENTO (SIMULAÇÃO + LABELS) ----
                    
                    # Ajustar colunas de tempo
                    if "transact_time" in df_trades.columns:
                        df_trades = df_trades.with_columns(pl.col("transact_time").alias("timestamp"))
                    
                    if "open_time" in df_klines.columns:
                        df_klines = df_klines.with_columns(
                            pl.from_epoch(pl.col("open_time"), time_unit="ms").alias("timestamp")
                        )

                    # 1. Build Book (Simulação)
                    # window="15m" fixo por enquanto
                    try:
                        sim_book = build_simulated_book(df_trades, window=settings.SIM_WINDOW)
                    except Exception as e:
                        print(f"   [ERR] Erro no build_simulated_book: {e}")
                        current_date = next_date
                        continue

                    # 2. Labels Hierárquicos
                    try:
                        df_labels = generate_hierarchical_labels(
                            df_klines, 
                            window_hours=settings.LABEL_WINDOW_HOURS, 
                            target_pct=settings.LABEL_TARGET_PCT, 
                            stop_pct=settings.LABEL_STOP_PCT
                        )
                    except Exception as e:
                        print(f"   [ERR] Erro no generate_hierarchical_labels: {e}")
                        current_date = next_date
                        continue
                    
                    # 3. Sincronizar (Join)
                    # Filter labels to 15m intervals
                    df_labels = df_labels.with_columns(
                        pl.col("timestamp").dt.truncate("15m").alias("snapshot_time")
                    ).filter(pl.col("timestamp") == pl.col("snapshot_time"))

                    # Inner Join
                    print(f"      -> Join Sim ({sim_book.height}) + Labels ({df_labels.height})...", flush=True)
                    dataset_chunk = sim_book.join(df_labels, on="snapshot_time", how="inner")
                    dataset_chunk = dataset_chunk.drop_nulls(subset=["label"])
                    
                    print(f"      -> Chunk Size: {dataset_chunk.height}", flush=True)

                    if dataset_chunk.height == 0:
                        print(f"      [WARN] Chunk vazio após join.", flush=True)
                        current_date = next_date
                        continue

                    # 4. Filter Valid Time Range (remove buffer dates from output)
                    
                    # 5. Build Tensor (6D)
                    print(f"      -> Building 6-channel Tensor...", flush=True)
                    X_np = build_tensor_6d(dataset_chunk, n_levels=128, is_simulation=True)
                    print(f"      -> Tensor Shape: {X_np.shape}", flush=True)
                    
                    # Validar Labels
                    unique_snaps = dataset_chunk.select("snapshot_time").unique().sort("snapshot_time")
                    
                    # Verificar alinhamento
                    if len(unique_snaps) != len(X_np):
                        # Mismatch crítico
                        print(f"   [ERR] Mismatch X vs Time: {len(X_np)} vs {len(unique_snaps)}", flush=True)
                        current_date = next_date
                        continue
                        
                    # Extrair Y
                    y_df = unique_snaps.join(
                        dataset_chunk.select(["snapshot_time", "label"]).unique(),
                        on="snapshot_time", 
                        how="left"
                    )
                    Y_np = y_df["label"].to_numpy()
                    
                    # Timestamps para controle de duplicação
                    times = y_df["snapshot_time"].to_list()
                    
                    # 6. Generate Sequences (Sliding Window)
                    # seq_len = 96
                    
                    print(f"      -> DEBUG: Start Yield Loop. X_np.shape={X_np.shape}", flush=True)
                    yield_count = 0
                    if len(X_np) > self.seq_len:
                        for i in range(len(X_np) - self.seq_len):
                            # O target (label) é do último passo da sequência
                            target_idx = i + self.seq_len - 1
                            target_time = times[target_idx]
                            
                            if target_time >= current_date and target_time < next_date:
                                x_seq = X_np[i : i+self.seq_len]
                                y_seq = Y_np[target_idx]
                                
                                # Safety Check
                                # print(f"DEBUG YIELD: x={x_seq.shape}")
                                if x_seq.shape[0] != self.seq_len:
                                    print(f"      [ERR] Shape incorreto no yield: {x_seq.shape} != {self.seq_len}", flush=True)
                                    continue
                                    
                                yield (
                                    torch.tensor(x_seq, dtype=torch.float32),
                                    torch.tensor(y_seq, dtype=torch.long)
                                )
                                yield_count += 1
                    
                    print(f"      -> Yielded {yield_count} sequences for {current_date.date()}", flush=True)

                    gc.collect()
                    
                    # Próximo dia
                    current_date = next_date
                    
            except Exception as e:
                print(f"[ERR] Erro ao processar mês {month}: {e}")
