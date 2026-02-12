
import polars as pl
import sys
from pathlib import Path
from datetime import date, timedelta

# Adiciona a raiz do projeto ao sys.path para importar 'src' corretamente
# Como este script está em src/audit/, a raiz é ../../
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.config import settings

def generate_expected_months(start_date_str):
    """Gera lista de strings 'YYYY-MM' do início até o mês atual."""
    start = date.fromisoformat(start_date_str)
    today = date.today()
    # Ajuste para o contexto do seu sistema (que está em 2026)
    # Se quiser forçar até hoje real, use date.today()
    
    months = []
    current = start.replace(day=1)
    while current <= today.replace(day=1):
        months.append(current.strftime("%Y-%m"))
        # Avança mês
        next_month = current.replace(day=28) + timedelta(days=4)
        current = next_month.replace(day=1)
    return months

def check_files():
    print(f"🕵️  Auditando diretório: {settings.RAW_HISTORICAL_DIR}")
    
    if not settings.RAW_HISTORICAL_DIR.exists():
        print("❌ Erro CRÍTICO: Pasta de dados históricos não existe!")
        return

    # 1. Definição do Escopo
    expected_months = generate_expected_months(settings.HISTORICAL_START_DATE)
    print(f"📅 Período Esperado: {expected_months[0]} até {expected_months[-1]} ({len(expected_months)} meses)")
    
    missing_klines = []
    missing_trades = []
    corrupted_files = []

    print("\n--- 🔍 Verificando Integridade ---")
    
    for month in expected_months:
        # Check Klines
        kline_file = settings.RAW_HISTORICAL_DIR / f"klines_{month}.parquet"
        if not kline_file.exists():
            missing_klines.append(month)
        elif kline_file.stat().st_size < 1000: # Menor que 1KB é suspeito
            corrupted_files.append(f"Klines {month} (Muito pequeno)")

        # Check AggTrades
        trade_file = settings.RAW_HISTORICAL_DIR / f"aggTrades_{month}.parquet"
        if not trade_file.exists():
            missing_trades.append(month)
        elif trade_file.stat().st_size < 1000:
            corrupted_files.append(f"Trades {month} (Muito pequeno)")

    # 2. Relatório de Erros
    has_errors = False
    
    if missing_klines:
        print(f"❌ [KLINES] Meses Faltando ({len(missing_klines)}): {missing_klines}")
        has_errors = True
    else:
        print("✅ [KLINES] Todos os meses presentes.")

    if missing_trades:
        print(f"❌ [TRADES] Meses Faltando ({len(missing_trades)}): {missing_trades}")
        has_errors = True
    else:
        print("✅ [TRADES] Todos os meses presentes.")

    if corrupted_files:
        print(f"⚠️ [CORROMPIDOS] Arquivos suspeitos (0kb): {corrupted_files}")
        has_errors = True
    
    # 3. Teste de Leitura (Sampling)
    if not has_errors:
        print("\n--- 🧪 Teste de Leitura (Amostragem) ---")
        try:
            # Pega o último mês para testar
            if len(expected_months) >= 2:
                last_month = expected_months[-2] # Penúltimo para garantir que está completo
            else:
                last_month = expected_months[-1]

            test_file = settings.RAW_HISTORICAL_DIR / f"aggTrades_{last_month}.parquet"
            
            if not test_file.exists():
                 # Tenta outro se este falhou na verificação anterior mas passou no check? Improvável.
                 print(f"⚠️ Arquivo de teste {test_file.name} não encontrado apesar da verificação.")
                 return

            print(f"📖 Lendo amostra: {test_file.name}...")
            df = pl.read_parquet(test_file)
            print(f"   -> Sucesso! Shape: {df.shape}")
            print(f"   -> Colunas: {df.columns}")
            print("✅ Auditoria Concluída: DADOS ÍNTEGROS.")
            
        except Exception as e:
            print(f"❌ Erro ao ler arquivo de amostra: {e}")
            print("   -> Seus arquivos existem, mas podem estar com formato inválido.")

if __name__ == "__main__":
    check_files()
