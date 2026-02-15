import polars as pl
import numpy as np
from pathlib import Path
from src.config import settings
from src.processing.labeling import generate_hierarchical_labels
from src.training.train import generate_month_list

def diagnose_full_balance():
    print("=" * 60)
    print("DIAGNÓSTICO DE BALANCEAMENTO - QUANT GOD PROTOCOL")
    print("=" * 60)
    
    # 1. Definir Periodos
    train_months = generate_month_list("2020-01", "2024-10")
    val_months = generate_month_list("2024-11", "2026-01")
    
    def get_distribution(months, label_name):
        print(f"\n[ANALISANDO] {label_name} ({len(months)} meses)...")
        all_klines = []
        for m in months:
            k_file = settings.RAW_HISTORICAL_DIR / f"klines_{m}.parquet"
            if k_file.exists():
                all_klines.append(pl.read_parquet(k_file))
        
        if not all_klines:
            print(f"   [ERR] Nenhum dado encontrado para {label_name}")
            return None
            
        df_full = pl.concat(all_klines).sort("timestamp" if "timestamp" in all_klines[0].columns else "open_time")
        
        # Ajustar timestamp se necessário para o labeling
        if "open_time" in df_full.columns:
            df_full = df_full.with_columns(pl.from_epoch(pl.col("open_time"), time_unit="ms").alias("timestamp"))

        # Gerar Labels (Silenciosamente para o print customizado)
        df_labels = generate_hierarchical_labels(
            df_full, 
            window_hours=settings.LABEL_WINDOW_HOURS,
            target_pct=settings.LABEL_TARGET_PCT,
            stop_pct=settings.LABEL_STOP_PCT
        )
        
        unique, counts = np.unique(df_labels["label"].to_numpy(), return_counts=True)
        dist = {u: c for u, c in zip(unique, counts)}
        total = sum(counts)
        return dist, total

    # Executar
    train_dist, train_total = get_distribution(train_months, "TREINO (2020-2024)")
    val_dist, val_total = get_distribution(val_months, "VALIDAÇÃO (2024-2026)")

    # Tabela Final
    class_names = {0: "NEUTRO", 1: "STOP", 2: "LONG", 3: "SUPER LONG"}
    
    print("\n" + "=" * 80)
    print(f"{'CLASSE':<15} | {'TREINO %':<15} | {'VALIDAÇÃO %':<15} | {'STATUS'}")
    print("-" * 80)
    
    for i in range(4):
        t_count = train_dist.get(i, 0)
        v_count = val_dist.get(i, 0)
        t_pct = (t_count / train_total * 100) if train_total > 0 else 0
        v_pct = (v_count / val_total * 100) if val_total > 0 else 0
        
        # Check drift
        drift = abs(t_pct - v_pct)
        status = "✅ OK" if drift < 5 else "⚠️ DRIFT"
        
        name = class_names[i]
        print(f"{name:<15} | {t_pct:>13.2f}% | {v_pct:>14.2f}% | {status}")
    
    print("-" * 80)
    print(f"TOTAL AMOSTRAS  | {train_total:>14,} | {val_total:>15,} |")
    print("=" * 80)

if __name__ == "__main__":
    diagnose_full_balance()
