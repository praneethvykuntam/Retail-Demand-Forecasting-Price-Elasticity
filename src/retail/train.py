import pandas as pd, json
from pathlib import Path
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error

PROC = Path("data/processed"); MODELS = Path("models"); MODELS.mkdir(exist_ok=True)

FEATS = ["price","promo","dow","week","lag_1","lag_7","lag_14","rolling_7","price_change"]
TARGET = "units"

def main():
    df = pd.read_parquet(PROC/"features.parquet")
    train = df[df["is_train"]]
    valid = df[~df["is_train"]]

    model = LGBMRegressor(n_estimators=600, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8)
    model.fit(train[FEATS], train[TARGET])

    pred = model.predict(valid[FEATS])
    mae = mean_absolute_error(valid[TARGET], pred)
    (MODELS/"retail_lgbm.txt").write_bytes(model.booster_.model_to_string().encode())
    (MODELS/"metrics.json").write_text(json.dumps({"valid_mae": float(mae)}, indent=2))
    print("Trained. MAE:", mae)

if __name__ == "__main__":
    main()
