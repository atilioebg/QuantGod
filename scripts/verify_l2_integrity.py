import polars as pl
import sys
from pathlib import Path
from datetime import datetime
import argparse

def verify_l2_integrity(file_path: str, gap_threshold_sec: int = 60):
    print(f"\n{'='*60}")
    print(f"📊 QGOD L2 DATA AUDITOR - Market Data Integrity")
    print(f"{'='*60}")
    
    path = Path(file_path)
    if not path.exists():
        print(f"❌ ERRO: Arquivo não encontrado em {file_path}")
        return

    print(f"🔍 Analisando: {path.name} ({path.stat().st_size / 1024 / 1024:.2f} MB)")

    try:
        # 1. Carregamento Rápido via Polars
        if path.suffix == '.parquet':
            df = pl.read_parquet(path)
        elif path.suffix in ['.csv', '.gz']:
            df = pl.read_csv(path)
        else:
            print(f"⚠️ Formato {path.suffix} não suportado nativamente. Tentando ler como CSV...")
            df = pl.read_csv(path)
        
        print("✅ Arquivo carregado com sucesso.")
    except Exception as e:
        print(f"❌ ARQUIVO CORROMPIDO ou formato inválido: {e}")
        return

    # 2. Identificação de Colunas
    cols = df.columns
    print(f"📝 Colunas detectadas: {cols}")
    
    # Tenta encontrar coluna de tempo
    time_col = None
    # Adicionando 'datetime' (comum em arquivos processados) e outros aliases
    for c in ['datetime', 'timestamp', 'time', 'ts', 'local_timestamp']:
        if c in cols: # Removendo lower() para evitar mismatch se houver camelCase
            time_col = c
            break
            
    if not time_col:
        # Busca insensível a maiúsculas/minúsculas se a busca exata falhar
        for c in cols:
            if c.lower() in ['datetime', 'timestamp', 'time', 'ts', 'local_timestamp']:
                time_col = c
                break
            
    if not time_col:
        print("ℹ️ Info: Coluna de timestamp nominal não localizada. Prosseguindo com auditoria estrutural.")
        # return  <-- Removido para permitir fallback

    # 3. Auditoria de Tempo
    if time_col:
        try:
            if df[time_col].dtype in [pl.Int64, pl.Float64]:
                first_val = df[time_col][0]
                if first_val > 1e15: # us
                    df = df.with_columns(pl.from_epoch(pl.col(time_col), time_unit="us").alias("_dt"))
                elif first_val > 1e12: # ms
                    df = df.with_columns(pl.from_epoch(pl.col(time_col), time_unit="ms").alias("_dt"))
                else: # s
                    df = df.with_columns(pl.from_epoch(pl.col(time_col), time_unit="s").alias("_dt"))
            else:
                df = df.with_columns(pl.col(time_col).str.to_datetime().alias("_dt"))
                
            df = df.sort("_dt")
            start_ts = df["_dt"].min()
            end_ts = df["_dt"].max()
            duration = end_ts - start_ts
            print(f"\n📅 Janela Temporal:")
            print(f"   Início: {start_ts}")
            print(f"   Fim:    {end_ts}")
            print(f"   Duração Total: {duration}")

            # 4. Detecção de Gaps (Tempo)
            print(f"\n🕵️ Verificando Gaps Gaps > {gap_threshold_sec}s...")
            df = df.with_columns([pl.col("_dt").diff().alias("gap")])
            gaps = df.filter(pl.col("gap") > pl.duration(seconds=gap_threshold_sec))
            if len(gaps) > 0:
                print(f"⚠️ ALERTA: Encontrados {len(gaps)} gaps maiores que {gap_threshold_sec} segundos!")
                for row in gaps.sort("gap", descending=True).head(5).to_dicts():
                    gap_dur = row['gap']
                    gap_end = row['_dt']
                    print(f"   - GAP: {gap_dur} (Termina em {gap_end})")
            else:
                print("✅ Nenhum gap detectado. Fluxo de dados contínuo.")
        except Exception as e:
            print(f"⚠️ Falha na auditoria temporal: {e}")
    else:
        print("\n⚠️ AVISO: Coluna de tempo não encontrada. Realizando auditoria baseada em sequência/estatística.")
        print(f"📅 Registros Totais: {len(df)}")
        # Em arquivos de 1min, 64800 rows = 45 dias exatos.
        expected_rows_45d = 45 * 24 * 60
        if len(df) == expected_rows_45d:
            print(f"✅ Densidade Perfeita: {len(df)} linhas para 45 dias (1min/linha).")
        else:
            print(f"ℹ️ Tamanho da Amostra: {len(df)} linhas (~{len(df)/1440:.1f} dias).")

    # 5. Estatísticas de Densidade
    total_rows = len(df)
    if 'duration' in locals() and duration:
        minutes = max(1, duration.total_seconds() / 60)
        density = total_rows / minutes
        print(f"\n📊 Estatísticas de Densidade:")
        print(f"   Total de Linhas: {total_rows}")
        print(f"   Média: {density:.2f} updates/minuto")
    else:
        print(f"\n📊 Estatísticas de Densidade:")
        print(f"   Total de Linhas: {total_rows}")

    # 6. Verificação de Nulos em colunas críticas
    print("\n🚫 Verificação de Nulos:")
    critical_cols = [c for c in cols if any(k in c.lower() for k in ['price', 'size', 'side', 'qty'])]
    null_counts = df.select([pl.col(c).null_count().alias(c) for c in critical_cols])
    
    has_nulls = False
    for col_name in critical_cols:
        count = null_counts[col_name][0]
        if count > 0:
            print(f"   - ⚠️ COLUNA [{col_name}]: {count} valores nulos encontrados!")
            has_nulls = True
    
    if not has_nulls:
        print("   ✅ Nenhuma coluna crítica possui valores nulos.")

    print(f"\n{'='*60}")
    print("✅ AUDITORIA FINALIZADA")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auditoria de Integridade de Dados L2")
    parser.add_argument("file", type=str, help="Caminho para o arquivo (Parquet ou CSV)")
    parser.add_argument("--gap", type=int, default=60, help="Limiar de Gap em segundos (Default: 60)")
    
    args = parser.parse_args()
    verify_l2_integrity(args.file, args.gap)
