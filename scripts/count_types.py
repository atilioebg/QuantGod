
import zipfile
import json
from pathlib import Path

zip_path = Path("data/L2/raw/l2_samples/2026-02-14_BTCUSDT_ob200.data.zip")
with zipfile.ZipFile(zip_path, 'r') as z:
    name = z.namelist()[0]
    snapshots = 0
    deltas = 0
    with z.open(name) as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line)
            if obj.get("type") == "snapshot":
                snapshots += 1
            else:
                deltas += 1
    print(f"File: {name} | Snapshots: {snapshots} | Deltas: {deltas}")
