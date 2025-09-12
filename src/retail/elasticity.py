from pathlib import Path
import numpy as np, pandas as pd

PROC = Path("data/processed"); REPORTS = Path("data/reports"); REPORTS.mkdir(parents=True, exist_ok=True)

def elasticity_by_sku(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (p, s), g in df.groupby(["product_id","store_id"]):
        g = g.dropna(subset=["units","price"]).copy()
        g = g[(g["units"]>0) & (g["price"]>0)]
        if len(g) < 30:  # need enough points
            continue
        g["lu"] = np.log(g["units"])
        g["lp"] = np.log(g["price"])
        # OLS slope for ln(units) ~ ln(price)
        x = g["lp"].to_numpy(); y = g["lu"].to_numpy()
        x = np.c_[np.ones(len(x)), x]
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
        elasticity = beta[1]
        out.append({"product_id": p, "store_id": s, "elasticity": elasticity, "n": len(g)})
    return pd.DataFrame(out).sort_values("elasticity")

def main():
    df = pd.read_parquet(PROC/"features.parquet")
    cols_needed = {"product_id","store_id","date","units","price"}
    if not cols_needed.issubset(df.columns):
        # re-load sales to get 'price' if features were saved without it
        sales = pd.read_parquet(PROC/"sales_clean.parquet")[["product_id","store_id","date","price","units"]]
        df = df.merge(sales, on=["product_id","store_id","date"], how="left")
    out = elasticity_by_sku(df)
    out.to_csv(REPORTS/"elasticity_by_sku.csv", index=False)
    print("Wrote ->", REPORTS/"elasticity_by_sku.csv", "| rows:", len(out))

if __name__ == "__main__":
    main()
