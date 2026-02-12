
import polars as pl
import datetime
import os
import sys
from pathlib import Path

# Adiciona a raiz do projeto ao sys.path
# Script em src/audit/, raiz em ../../
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.config import settings

def check_stream_health():
    print(f"📡 Auditando Stream em: {settings.RAW_STREAM_DIR}")
    
    # 1. Verificar Existência de Arquivos
    # O stream pode salvar em subpastas ou na raiz, vamos procurar recursivamente
    if not settings.RAW_STREAM_DIR.exists():
        print(f"❌ Diretório de stream não existe: {settings.RAW_STREAM_DIR}")
        return

    all_files = list(settings.RAW_STREAM_DIR.rglob("*.parquet"))
    
    if not all_files:
        print("\n⚠️ NENHUM ARQUIVO ENCONTRADO AINDA.")
        print("   Motivo provável: O 'Stream' guarda dados na Memória RAM por 15 minutos.")
        print("   -> Se você iniciou o stream há menos de 15 min, isso é NORMAL.")
        print("   -> Apenas espere o primeiro 'Flush'.")
        return

    # 2. Ordenar por Data de Modificação (Mais recente primeiro)
    all_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    latest_file = all_files[0]
    
    # 3. Verificar "Batimentos Cardíacos" (Recência)
    last_modified = datetime.datetime.fromtimestamp(latest_file.stat().st_mtime)
    now = datetime.datetime.now()
    time_diff = now - last_modified
    
    print(f"\n📂 Arquivo mais recente: {latest_file.name}")
    print(f"⏱️ Última atualização: {last_modified.strftime('%H:%M:%S')} (Há {time_diff.seconds // 60} minutos)")
    
    if time_diff.seconds > 1200: # 20 minutos
        print("❌ ALERTA: O stream parece estar parado! O último arquivo é muito antigo (>20min).")
        print("   -> Verifique se o terminal do 'python -m src.collectors.stream' está rodando.")
    else:
        print("✅ STATUS: O Stream está VIVO e gravando.")

    # 4. Biópsia do Arquivo (Conteúdo)
    print("\n🔬 Inspecionando Conteúdo do Último Arquivo...")
    try:
        df = pl.read_parquet(latest_file)
        print(f"   📏 Dimensões: {df.shape} (Linhas, Colunas)")
        print(f"   📋 Colunas: {df.columns}")
        
        # Normalização de nomes de colunas (Stream pode usar nomes diferentes)
        # Depth geralmente tem bids/asks. Trades tem p/q ou price/quantity.
        
        has_depth = "bids" in df.columns or "asks" in df.columns
        # Pode estar nested em 'data' ou algo assim, mas o script de stream deve ter achatado.
        
        has_trades = "p" in df.columns or "price" in df.columns or "e" in df.columns or "E" in df.columns
        # Stream trade columns often: 'e', 'E', 's', 't', 'p', 'q', 'b', 'a', 'T', 'm', 'M' (Binance format)

        if has_depth:
            print("   ✅ DADOS DE DEPTH (Order Book) DETECTADOS.")
            # Mostra uma amostra para ver se não está tudo nulo
            try:
                print(df.select(['bids', 'asks']).head(2))
            except:
                pass
        
        if has_trades:
            print("   ✅ DADOS DE TRADES (Execuções) DETECTADOS.")
        
        if not has_depth and not has_trades:
            print("   ⚠️ O arquivo existe mas não identifiquei colunas padrão (bids/asks ou price/qty).")
            print("      Verifique o schema abaixo:")
            print(df.head(2))

    except Exception as e:
        print(f"❌ Erro ao ler o arquivo parquet: {e}")

if __name__ == "__main__":
    check_stream_health()
