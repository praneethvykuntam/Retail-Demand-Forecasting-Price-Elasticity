import pandas as pd, numpy as np
from pathlib import Path

PROC = Path("data/processed"); PROC.mkdir(parents=True, exist_ok=True)

def lag_feats(g, lags=[1,7,14]):
    for L in lags:
        g[f"lag_{L}"] = g["units"].shift(L)
    g["rolling_7"] = g["units"].rolling(7).mean()
    g["price_change"] = g["price"].pct_change().fillna(0)
    return g

def main():
    df = pd.read_parquet(PROC/"sales_clean.parquet")
    df = df.sort_values(["product_id","store_id","date"])
    df = df.groupby(["product_id","store_id"], group_keys=False).apply(lag_feats)
    df = df.dropna().reset_index(drop=True)

    # train/valid split by date
    split_date = df["date"].quantile(0.8)
    df["is_train"] = (df["date"] <= split_date)
    df.to_parquet(PROC/"features.parquet", index=False)
    print("Features ->", PROC/"features.parquet", "| Split date:", split_date.date())

if __name__ == "__main__":
    main()
