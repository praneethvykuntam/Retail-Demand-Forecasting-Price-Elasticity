import pandas as pd
from pathlib import Path
from lightgbm import Booster

PROC = Path("data/processed"); REPORTS = Path("data/reports"); REPORTS.mkdir(parents=True, exist_ok=True)
MODELS = Path("models")

FEATS = ["price","promo","dow","week","lag_1","lag_7","lag_14","rolling_7","price_change"]
TARGET = "units"

def load_model():
    bst = Booster(model_file=str(MODELS/"retail_lgbm.txt"))
    return bst

def main():
    df = pd.read_parquet(PROC/"features.parquet")
    model = load_model()
    # score last 28 days of valid set
    eval_df = df[~df["is_train"]].copy()
    eval_df["pred_units"] = model.predict(eval_df[FEATS])
    out = eval_df[["date","product_id","store_id","units","pred_units"]]
    out.to_csv(REPORTS/"predictions.csv", index=False)
    print("Wrote scores ->", REPORTS/"predictions.csv")

if __name__ == "__main__":
    main()
