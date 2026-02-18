import os
from datetime import datetime, timedelta
from pathlib import Path

BASE_DIR = Path(r"C:\Users\Atilio\Downloads\btcustd_L2_2025")
START_DATE = datetime(2025, 1, 1)
END_DATE = datetime(2025, 12, 31)

def fast_check():
    curr = START_DATE
    missing = []
    found_patterns = {}
    
    while curr <= END_DATE:
        date_str = curr.strftime("%Y-%m-%d")
        patterns = [
            f"{date_str}_BTCUSDT_ob200.data.zip",
            f"{date_str}_BTCUSDT_0400.data.zip",
            f"{date_str}_BTCUSDT_ob400.data.zip",
            f"{date_str}_BTCUSDT_ob500.data.zip",
            f"{date_str}_BTCUSDT_ob100.data.zip" # Just in case
        ]
        
        found = False
        for p in patterns:
            if (BASE_DIR / p).exists():
                suffix = p.split("_")[-1]
                found_patterns[suffix] = found_patterns.get(suffix, 0) + 1
                found = True
                break
        
        if not found:
            missing.append(date_str)
        
        curr += timedelta(days=1)
    
    print("\n--- FAST PATTERN CHECK ---")
    print(f"Total dias checkados: {(END_DATE - START_DATE).days + 1}")
    print(f"Arquivos encontrados: {365 - len(missing)}")
    print(f"Arquivos ausentes: {len(missing)}")
    print("\nPadroes encontrados:")
    for suff, count in found_patterns.items():
        print(f"  - {suff}: {count} arquivos")
    
    if missing:
        print("\nExemplos de datas ausentes:")
        print(missing[:10])

if __name__ == "__main__":
    fast_check()
