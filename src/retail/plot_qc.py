
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA = Path("data")
PROC = DATA / "processed"
REPORTS = DATA / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

# Sales trend
features_fp = PROC / "features.parquet"
if features_fp.exists():
    df = pd.read_parquet(features_fp)
    if {"product_id","store_id","date","units"}.issubset(df.columns):
        top_pair = (df.groupby(["product_id","store_id"])["units"]
                      .sum().sort_values(ascending=False).head(1).index[0])
        sample = df[(df["product_id"]==top_pair[0]) & (df["store_id"]==top_pair[1])].copy()
        sample = sample.sort_values("date")
        plt.figure(figsize=(10,4))
        plt.plot(pd.to_datetime(sample["date"]), sample["units"], label="Units Sold")
        plt.title(f"Sales Trend – product {top_pair[0]}, store {top_pair[1]}")
        plt.xlabel("Date"); plt.ylabel("Units"); plt.legend()
        plt.savefig(REPORTS / "sales_trend.png", bbox_inches="tight", dpi=150)
        plt.close()

# Actual vs predicted scatter + total series + error hist
pred_fp = REPORTS / "predictions.csv"
if pred_fp.exists():
    pred = pd.read_csv(pred_fp, parse_dates=["date"])
    if {"units","pred_units"}.issubset(pred.columns):
        # scatter
        plt.figure(figsize=(6,6))
        plt.scatter(pred["units"], pred["pred_units"], alpha=0.3)
        lo = 0
        hi = max(pred["units"].max(), pred["pred_units"].max())
        plt.plot([lo, hi], [lo, hi], linestyle="--")
        plt.title("Actual vs Predicted Demand")
        plt.xlabel("Actual Units"); plt.ylabel("Predicted Units")
        plt.savefig(REPORTS / "actual_vs_pred.png", bbox_inches="tight", dpi=150)
        plt.close()

        # total series
        ts = pred.groupby("date")[["units","pred_units"]].sum().sort_index()
        ax = ts.plot(figsize=(10,4))
        ax.set_title("Total Demand: Actual vs Predicted (daily)")
        ax.set_xlabel("Date"); ax.set_ylabel("Units")
        plt.tight_layout(); plt.savefig(REPORTS / "total_series.png", bbox_inches="tight", dpi=150)
        plt.close()

        # error hist
        pred = pred.copy()
        pred["error"] = pred["pred_units"] - pred["units"]
        ax = pred["error"].hist(bins=40, figsize=(8,4))
        ax.set_title("Prediction Error Distribution (pred - actual)")
        ax.set_xlabel("Error"); ax.set_ylabel("Count")
        plt.tight_layout(); plt.savefig(REPORTS / "error_hist.png", bbox_inches="tight", dpi=150)
        plt.close()

# Elasticity scatter (log-log) for a representative pair
sales_fp = PROC / "sales_clean.parquet"
if features_fp.exists():
    feat = pd.read_parquet(features_fp)
    if "price" not in feat.columns and sales_fp.exists():
        sales = pd.read_parquet(sales_fp)[["product_id","store_id","date","price","units"]]
        feat = feat.merge(sales, on=["product_id","store_id","date"], how="left", suffixes=("","_sales"))
        if "price" not in feat.columns and "price_sales" in feat.columns:
            feat["price"] = feat["price_sales"]
    if {"product_id","store_id","date","units","price"}.issubset(feat.columns):
        pair = (feat.groupby(["product_id","store_id"])["units"]
                  .sum().sort_values(ascending=False).head(1).index[0])
        g = feat[(feat["product_id"]==pair[0]) & (feat["store_id"]==pair[1])].dropna(subset=["units","price"]).copy()
        g = g[(g["units"]>0) & (g["price"]>0)]
        if len(g) >= 20:
            g["lu"] = np.log(g["units"]); g["lp"] = np.log(g["price"])
            X = np.c_[np.ones(len(g)), g["lp"].to_numpy()]
            y = g["lu"].to_numpy()
            b = np.linalg.lstsq(X, y, rcond=None)[0]
            slope = b[1]
            import numpy as np
            import matplotlib.pyplot as plt
            plt.figure(figsize=(6,5))
            plt.scatter(g["lp"], g["lu"], alpha=0.5)
            xline = np.linspace(g["lp"].min(), g["lp"].max(), 100)
            yline = b[0] + slope * xline
            plt.plot(xline, yline, linestyle="--")
            plt.title(f"Log-Log Price vs Units (elasticity ≈ {slope:.2f})\nproduct {pair[0]}, store {pair[1]}")
            plt.xlabel("log(price)"); plt.ylabel("log(units)")
            plt.savefig(REPORTS / "elasticity_scatter.png", bbox_inches="tight", dpi=150)
            plt.close()
