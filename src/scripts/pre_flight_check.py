"""
Pre-Flight Check Script
Validates all dependencies and data before training
"""

import sys
from pathlib import Path

# Add project root
# Add project root (calculated relative to this script)
sys.path.append(str(Path(__file__).resolve().parents[2]))

def check_imports():
    """Verify all required imports work"""
    print("=" * 80)
    print("🔍 VERIFICAÇÃO DE IMPORTS")
    print("=" * 80)
    
    errors = []
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"   CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            print(f"   AMP Support: {hasattr(torch.cuda, 'amp')}")
    except Exception as e:
        errors.append(f"PyTorch: {e}")
    
    try:
        import polars as pl
        print(f"✅ Polars: {pl.__version__}")
    except Exception as e:
        errors.append(f"Polars: {e}")
    
    try:
        import numpy as np
        print(f"✅ NumPy: {np.__version__}")
    except Exception as e:
        errors.append(f"NumPy: {e}")
    
    # Check custom modules
    try:
        from src.config import settings
        print(f"✅ Config: Loaded")
    except Exception as e:
        errors.append(f"Config: {e}")
    
    try:
        from src.processing.simulation import build_simulated_book
        print(f"✅ Simulation: Loaded")
    except Exception as e:
        errors.append(f"Simulation: {e}")
    
    try:
        from src.processing.tensor_builder import build_tensor_4d
        print(f"✅ Tensor Builder: Loaded")
    except Exception as e:
        errors.append(f"Tensor Builder: {e}")
    
    try:
        from src.processing.labeling import generate_labels
        print(f"✅ Labeling: Loaded")
    except Exception as e:
        errors.append(f"Labeling: {e}")
    
    try:
        from src.models.vivit import SAIMPViViT
        print(f"✅ Model (ViViT): Loaded")
    except Exception as e:
        errors.append(f"Model: {e}")
    
    if errors:
        print("\n❌ ERROS DE IMPORT:")
        for err in errors:
            print(f"   - {err}")
        return False
    
    print("\n✅ Todos os imports OK!")
    return True


def check_data_files():
    """Verify required data files exist"""
    print("\n" + "=" * 80)
    print("📁 VERIFICAÇÃO DE ARQUIVOS")
    print("=" * 80)
    
    from src.config import settings
    
    required_months = {
        "TRAIN": ["2025-11", "2025-12"],
        "VAL": ["2026-01"]
    }
    
    missing = []
    
    for dataset, months in required_months.items():
        print(f"\n[{dataset}]")
        for month in months:
            t_file = settings.RAW_HISTORICAL_DIR / f"aggTrades_{month}.parquet"
            k_file = settings.RAW_HISTORICAL_DIR / f"klines_{month}.parquet"
            
            t_ok = t_file.exists()
            k_ok = k_file.exists()
            
            status_t = "✅" if t_ok else "❌"
            status_k = "✅" if k_ok else "❌"
            
            print(f"   {month}: Trades {status_t} | Klines {status_k}")
            
            if not t_ok:
                missing.append(f"aggTrades_{month}.parquet")
            if not k_ok:
                missing.append(f"klines_{month}.parquet")
    
    if missing:
        print("\n❌ ARQUIVOS FALTANDO:")
        for f in missing:
            print(f"   - {f}")
        return False
    
    print("\n✅ Todos os arquivos presentes!")
    return True


def check_data_integrity():
    """Quick sanity check on data loading"""
    print("\n" + "=" * 80)
    print("🔬 VERIFICAÇÃO DE INTEGRIDADE DOS DADOS")
    print("=" * 80)
    
    try:
        import polars as pl
        from src.config import settings
        
        # Test load one file
        test_file = settings.RAW_HISTORICAL_DIR / "aggTrades_2025-11.parquet"
        print(f"\n📖 Testando leitura: {test_file.name}")
        
        df = pl.read_parquet(test_file).head(1000)
        print(f"   Shape: {df.shape}")
        print(f"   Colunas: {df.columns[:5]}... ({len(df.columns)} total)")
        
        # Check for timestamp column
        if "transact_time" in df.columns or "timestamp" in df.columns:
            print(f"   ✅ Coluna de timestamp encontrada")
        else:
            print(f"   ⚠️ Coluna de timestamp não encontrada")
            return False
        
        # Test klines
        k_file = settings.RAW_HISTORICAL_DIR / "klines_2025-11.parquet"
        print(f"\n📖 Testando leitura: {k_file.name}")
        
        df_k = pl.read_parquet(k_file).head(100)
        print(f"   Shape: {df_k.shape}")
        print(f"   Colunas: {df_k.columns[:5]}... ({len(df_k.columns)} total)")
        
        required_cols = ["open", "high", "low", "close"]
        missing_cols = [c for c in required_cols if c not in df_k.columns]
        
        if missing_cols:
            print(f"   ❌ Colunas faltando: {missing_cols}")
            return False
        else:
            print(f"   ✅ Colunas OHLC presentes")
        
        print("\n✅ Integridade dos dados OK!")
        return True
        
    except Exception as e:
        print(f"\n❌ Erro ao verificar dados: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_model_instantiation():
    """Test model can be created"""
    print("\n" + "=" * 80)
    print("🧠 VERIFICAÇÃO DO MODELO")
    print("=" * 80)
    
    try:
        import torch
        from src.models.vivit import SAIMPViViT
        
        print("\n🏗️ Instanciando modelo...")
        model = SAIMPViViT(
            seq_len=96,
            input_channels=4,
            price_levels=128,
            num_classes=3
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   Parâmetros: {total_params:,}")
        
        # Test forward pass
        print("\n🔄 Testando forward pass...")
        dummy_input = torch.randn(2, 96, 4, 128)  # Batch=2
        
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"   Input shape: {dummy_input.shape}")
        print(f"   Output shape: {output.shape}")
        
        if output.shape == (2, 3):
            print(f"   ✅ Output shape correto!")
        else:
            print(f"   ❌ Output shape incorreto! Esperado (2, 3), obtido {output.shape}")
            return False
        
        print("\n✅ Modelo OK!")
        return True
        
    except Exception as e:
        print(f"\n❌ Erro no modelo: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_pipeline_mini():
    """Run mini pipeline test"""
    print("\n" + "=" * 80)
    print("⚙️ TESTE DE PIPELINE MINI")
    print("=" * 80)
    
    try:
        import polars as pl
        from src.config import settings
        from src.processing.simulation import build_simulated_book
        from src.processing.tensor_builder import build_tensor_4d
        from src.processing.labeling import generate_labels
        
        print("\n📚 Carregando amostra pequena...")
        t_file = settings.RAW_HISTORICAL_DIR / "aggTrades_2025-11.parquet"
        k_file = settings.RAW_HISTORICAL_DIR / "klines_2025-11.parquet"
        
        df_trades = pl.read_parquet(t_file).head(50000)
        df_klines = pl.read_parquet(k_file).head(5000)
        
        # Normalize
        if "transact_time" in df_trades.columns:
            df_trades = df_trades.with_columns(pl.col("transact_time").alias("timestamp"))
        
        if "open_time" in df_klines.columns:
            df_klines = df_klines.with_columns(
                pl.from_epoch(pl.col("open_time"), time_unit="ms").alias("timestamp")
            )
        
        print("   ✅ Dados carregados")
        
        print("\n🔄 Simulando Order Book...")
        sim_book = build_simulated_book(df_trades, window="15m")
        print(f"   Snapshots: {sim_book.select('snapshot_time').n_unique()}")
        
        print("\n🏷️ Gerando Labels...")
        labels = generate_labels(df_klines, window_hours=24, target_pct=0.035, stop_pct=0.015)
        print(f"   Labels: {labels.height}")
        
        print("\n🖼️ Construindo Tensor...")
        tensor = build_tensor_4d(sim_book, n_levels=128, is_simulation=True)
        print(f"   Tensor shape: {tensor.shape}")
        
        if len(tensor.shape) == 3 and tensor.shape[1] == 4 and tensor.shape[2] == 128:
            print(f"   ✅ Tensor shape correto!")
        else:
            print(f"   ❌ Tensor shape incorreto! Esperado (T, 4, 128)")
            return False
        
        print("\n✅ Pipeline OK!")
        return True
        
    except Exception as e:
        print(f"\n❌ Erro no pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all checks"""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "SAIMP PRE-FLIGHT CHECK" + " " * 36 + "║")
    print("╚" + "=" * 78 + "╝")
    
    checks = [
        ("Imports", check_imports),
        ("Arquivos de Dados", check_data_files),
        ("Integridade dos Dados", check_data_integrity),
        ("Modelo Neural", check_model_instantiation),
        ("Pipeline Completo", check_pipeline_mini),
    ]
    
    results = []
    
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Erro fatal em '{name}': {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 80)
    print("📋 RESUMO")
    print("=" * 80)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} - {name}")
    
    all_passed = all(r[1] for r in results)
    
    print("\n" + "=" * 80)
    if all_passed:
        print("🎉 TODOS OS TESTES PASSARAM!")
        print("=" * 80)
        print("\n✅ Sistema pronto para treinamento.")
        print("\n🚀 Para iniciar o treino, execute:")
        print("   python -m src.training.train")
        return 0
    else:
        print("❌ ALGUNS TESTES FALHARAM")
        print("=" * 80)
        print("\n⚠️ Corrija os erros acima antes de treinar.")
        return 1


if __name__ == "__main__":
    exit(main())
