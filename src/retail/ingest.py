import pandas as pd
from pathlib import Path

RAW = Path("data/raw")
PROC = Path("data/processed"); PROC.mkdir(parents=True, exist_ok=True)

def main():
    df = pd.read_csv(RAW/"sales.csv", parse_dates=["date"])
    # basic cleaning
    df = df.dropna().query("price>0 and units>=0")
    df["dow"] = df["date"].dt.dayofweek
    df["week"] = df["date"].dt.isocalendar().week.astype(int)
    df["year"] = df["date"].dt.year
    df.to_parquet(PROC/"sales_clean.parquet", index=False)
    print("Ingested ->", PROC/"sales_clean.parquet")

if __name__ == "__main__":
    main()
