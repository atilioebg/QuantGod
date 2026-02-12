
import polars as pl
from pathlib import Path
import datetime
import sys

# Ajuste do path para importar configurações
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.config import settings

def audit_full_stream():
    stream_dir = settings.RAW_STREAM_DIR
    print(f"📡 Iniciando Auditoria Completa do Stream em: {stream_dir}")
    
    # Busca recursivamente todos os parquets
    all_files = list(stream_dir.rglob("*.parquet"))
    
    if not all_files:
        print("❌ Nenhum arquivo de stream encontrado.")
        return

    print(f"📦 Total de arquivos encontrados: {len(all_files)}")
    
    # Extrai metadados dos arquivos
    file_stats = []
    
    for f in all_files:
        try:
            # Tenta extrair timestamp do nome do arquivo
            # Formato esperado: trades_YYYYMMDD_HHMMSS.parquet
            # Ex: trades_20260210_123133.parquet
            name_parts = f.stem.split("_")
            if len(name_parts) >= 3:
                date_part = name_parts[1]
                time_part = name_parts[2]
                dt_str = f"{date_part}{time_part}"
                timestamp = datetime.datetime.strptime(dt_str, "%Y%m%d%H%M%S")
            else:
                # Fallback para data de modificação se o nome não bater
                timestamp = datetime.datetime.fromtimestamp(f.stat().st_mtime)
            
            size_kb = f.stat().st_size / 1024
            file_stats.append({"file": f, "timestamp": timestamp, "size_kb": size_kb})
            
        except Exception as e:
            print(f"⚠️ Erro ao processar arquivo {f.name}: {e}")

    # Ordena por Timestamp
    file_stats.sort(key=lambda x: x["timestamp"])
    
    if not file_stats:
        print("❌ Falha ao extrair timestamps dos arquivos.")
        return

    # Análise de Gaps
    start_time = file_stats[0]["timestamp"]
    end_time = file_stats[-1]["timestamp"]
    duration = end_time - start_time
    
    print(f"\n⏳ Período Analisado: {start_time} até {end_time}")
    print(f"⏱️ Duração Total: {duration}")
    
    print("\n🔎 Análise de Continuidade:")
    
    gaps_found = 0
    max_gap = datetime.timedelta(0)
    
    # Tolerância de gap (ex: 20 min se o flush é a cada 15)
    GAP_TOLERANCE_MINUTES = 25 
    
    for i in range(1, len(file_stats)):
        prev = file_stats[i-1]
        curr = file_stats[i]
        
        diff = curr["timestamp"] - prev["timestamp"]
        
        if diff > max_gap:
            max_gap = diff
            
        if diff > datetime.timedelta(minutes=GAP_TOLERANCE_MINUTES):
            gaps_found += 1
            print(f"   ⚠️ GAP DETECTADO: Entre {prev['timestamp'].strftime('%H:%M:%S')} e {curr['timestamp'].strftime('%H:%M:%S')}")
            print(f"      Duração da interrupção: {diff}")
            
    if gaps_found == 0:
        print(f"✅ NENHUM GAP SIGNIFICATIVO (> {GAP_TOLERANCE_MINUTES}min) detectado.")
    else:
        print(f"❌ Foram encontrados {gaps_found} interrupções no fluxo.")

    print(f"\n📊 Estatísticas:")
    print(f"   Maior Intervalo entre arquivos: {max_gap}")
    print(f"   Tamanho Médio: {sum(f['size_kb'] for f in file_stats)/len(file_stats):.2f} KB")
    
    # Checar arquivos muito pequenos (corrompidos/vazios)
    small_files = [f for f in file_stats if f["size_kb"] < 1.0] # Menor que 1KB
    if small_files:
        print(f"\n⚠️ Arquivos Suspeitos (Muito Pequenos): {len(small_files)}")
        for f in small_files[:5]:
            print(f"   -> {f['file'].name} ({f['size_kb']:.2f} KB)")
        if len(small_files) > 5:
            print(f"   ... e mais {len(small_files)-5} arquivos.")
    else:
        print("✅ Todos os arquivos parecem ter tamanho saudável (>1KB).")

if __name__ == "__main__":
    audit_full_stream()
