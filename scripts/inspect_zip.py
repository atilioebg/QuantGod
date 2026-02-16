
import zipfile
from pathlib import Path

zip_path = Path("data/L2/raw/l2_samples/2026-02-14_BTCUSDT_ob200.data.zip")
with zipfile.ZipFile(zip_path, 'r') as z:
    name = z.namelist()[0]
    print(f"File in ZIP: {name}")
    with z.open(name) as f:
        content = f.read(2000).decode('utf-8')
        print(f"First 2000 chars of {name}:")
        print(content)
