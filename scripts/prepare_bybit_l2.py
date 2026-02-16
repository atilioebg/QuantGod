
import polars as pl
import json
import zipfile
from pathlib import Path
import io
import time
from concurrent.futures import ProcessPoolExecutor
import os

# Configurações
RAW_DIR = Path("data/L2/raw/l2_samples")
INTERIM_DIR = Path("data/L2/interim")
OUTPUT_FILE = INTERIM_DIR / "l2_merged_raw.parquet"
SAMPLING_INTERVAL_MS = 1000 # 1 Segundo

def reconstruct_and_sample(zip_path):
    """
    Processa um único arquivo ZIP: reconstrói o book e amostra a cada 1s.
    """
    bids_book = {}
    asks_book = {}
    results = []
    last_sample_ts = -1
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            # Pega o primeiro arquivo do ZIP
            name = z.namelist()[0]
            with z.open(name) as f:
                for line in f:
                    if not line: continue
                    try:
                        obj = json.loads(line)
                    except:
                        continue
                        
                    msg_type = obj.get("type")
                    ts = obj.get("ts")
                    data = obj.get("data", {})
                    
                    # Atualizar Book
                    if msg_type == "snapshot":
                        bids_book = {float(p): float(s) for p, s in data.get("b", [])}
                        asks_book = {float(p): float(s) for p, s in data.get("a", [])}
                    else:
                        for p, s in data.get("b", []):
                            price, size = float(p), float(s)
                            if size == 0: bids_book.pop(price, None)
                            else: bids_book[price] = size
                        for p, s in data.get("a", []):
                            price, size = float(p), float(s)
                            if size == 0: asks_book.pop(price, None)
                            else: asks_book[price] = size
                    
                    # Amostragem Temporal (1s)
                    if ts - last_sample_ts >= SAMPLING_INTERVAL_MS:
                        last_sample_ts = (ts // SAMPLING_INTERVAL_MS) * SAMPLING_INTERVAL_MS
                        
                        sorted_bids = sorted(bids_book.keys(), reverse=True)[:5]
                        sorted_asks = sorted(asks_book.keys())[:5]
                        
                        row = {"received_time": ts}
                        for i in range(5):
                            if i < len(sorted_bids):
                                p = sorted_bids[i]
                                row[f"bid_{i}_price"] = p
                                row[f"bid_{i}_size"] = bids_book[p]
                            else:
                                row[f"bid_{i}_price"] = None
                                row[f"bid_{i}_size"] = None
                                
                            if i < len(sorted_asks):
                                p = sorted_asks[i]
                                row[f"ask_{i}_price"] = p
                                row[f"ask_{i}_size"] = asks_book[p]
                            else:
                                row[f"ask_{i}_price"] = None
                                row[f"ask_{i}_size"] = None
                        
                        results.append(row)
                        
        return results
    except Exception as e:
        print(f"⚠️ Erro em {zip_path.name}: {e}")
        return []

def process_bybit_l2():
    print("============================================================")
    print("BYBIT L2 PARALLEL INGESTION (1s Sampled) 🚀")
    print("============================================================")
    
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    zip_files = sorted(list(RAW_DIR.glob("*.zip")))
    
    if not zip_files:
        print("[ERRO] Nenhum ZIP encontrado.")
        return

    print(f"[INFO] Processando {len(zip_files)} arquivos em paralelo...")
    
    # Usar CPU_COUNT - 1 para não travar a máquina
    max_workers = max(1, os.cpu_count() - 1)
    
    all_results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(reconstruct_and_sample, z): z for z in zip_files}
        
        for i, future in enumerate(futures):
            zip_name = futures[future].name
            res = future.result()
            if res:
                all_results.append(pl.DataFrame(res))
            print(f"[{i+1}/{len(zip_files)}] ✅ {zip_name} concluído. ({len(res)} amostras)")

    if not all_results:
        print("[ERRO] Nenhum dado extraído.")
        return

    print("[INFO] Concatenando e salvando Parquet final...")
    full_df = pl.concat(all_results)
    
    # Conversão e ordenação
    full_df = full_df.with_columns(
        pl.from_epoch(pl.col("received_time"), time_unit="ms").alias("datetime")
    ).sort("datetime")
    
    full_df.write_parquet(OUTPUT_FILE)
    print(f"\n[SUCESSO] Dataset salvo em {OUTPUT_FILE}")
    print(f"Total Rows: {len(full_df)}")
    print(full_df.head(5))

if __name__ == "__main__":
    process_bybit_l2()
