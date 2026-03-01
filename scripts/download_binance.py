import os
import requests

BASE_URL = "https://data.binance.vision/data/spot/monthly/klines/ETHUSDT/1h/"
YEARS = range(2018, 2026)  # 2018 → 2025

os.makedirs("data/binance/ethusdt/1h", exist_ok=True)

for year in YEARS:
    for month in range(1, 13):
        filename = f"ETHUSDT-1h-{year}-{month:02d}.zip"
        url = BASE_URL + filename
        out_path = f"data/binance/ethusdt/1h/{filename}"

        if os.path.exists(out_path):
            print(f"Skipping (exists): {filename}")
            continue

        print(f"Downloading: {filename}")
        r = requests.get(url)

        if r.status_code == 200:
            with open(out_path, "wb") as f:
                f.write(r.content)
            print("✔ downloaded")
        else:
            print(f"✘ not found (status {r.status_code})")

print("Done.")