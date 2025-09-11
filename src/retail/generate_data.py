import numpy as np, pandas as pd
from pathlib import Path

RAW = Path("data/raw")
RAW.mkdir(parents=True, exist_ok=True)

np.random.seed(7)

dates = pd.date_range("2023-01-01", "2024-12-31", freq="D")
products = [f"P{i:03d}" for i in range(1, 21)]
stores = [f"S{i:02d}" for i in range(1, 11)]

rows = []
for d in dates:
    dow = d.dayofweek
    season = 1.0 + 0.2*np.sin(2*np.pi*(d.dayofyear/365))
    for p in products:
        base_price = 10 + (hash(p) % 10)
        for s in stores:
            base_demand = 20 + (hash(p+s) % 15)
            promo = np.random.binomial(1, 0.08)
            price = base_price * (0.9 if promo else 1.0) * (1 + np.random.normal(0, 0.03))
            noise = np.random.normal(0, 2)
            demand = (base_demand * season * (1.2 if dow in [4,5] else 1.0)
                      * (0.8 if price>base_price else 1.1) * (1.2 if promo else 1.0)) + noise
            demand = max(0, round(demand))
            rows.append([d, p, s, price, promo, demand])

df = pd.DataFrame(rows, columns=["date","product_id","store_id","price","promo","units"])
df.to_csv(RAW / "sales.csv", index=False)
print(f"Wrote {len(df):,} rows to {RAW/'sales.csv'}")
