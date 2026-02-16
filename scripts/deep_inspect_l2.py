
import zipfile
import json
from pathlib import Path

zip_path = Path("data/L2/raw/l2_samples/2026-02-14_BTCUSDT_ob200.data.zip")
with zipfile.ZipFile(zip_path, 'r') as z:
    name = z.namelist()[0]
    print(f"File in ZIP: {name}")
    with z.open(name) as f:
        # Lendo as primeiras 10 linhas para ver a variedade
        for i in range(20):
            line = f.readline()
            if not line:
                break
            obj = json.loads(line)
            msg_type = obj.get("type")
            ts = obj.get("ts")
            data = obj.get("data", {})
            bids = data.get("b", [])
            asks = data.get("a", [])
            print(f"Line {i}: Type={msg_type}, TS={ts}, BidsLen={len(bids)}, AsksLen={len(asks)}")
            if i == 0:
                print(f"Sample First Line Data Keys: {data.keys()}")
                if len(bids) > 0: print(f"Sample Bid 0: {bids[0]}")
